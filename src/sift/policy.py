"""Sift — researcher consent policy for schema exposure.

Schema depth—how much structural information the selected model sees about a
dataset — was historically a code default (``names_types``). This
module makes it an explicit researcher policy, stored in
``<cwd>/.sift/policy.json``:

    {
      "version": 1,
      "default_max_depth": "names_types",
      "datasets": {
        "survey.csv": {
          "max_depth": "names_types_labels",
          "set_at": "2026-04-21T14:20:00+00:00"
        }
      }
    }

The policy sets a **ceiling** on what the model can request for a given
dataset. ``get_schema(dataset, depth=...)`` consults the policy and
denies requests that exceed the ceiling. If a dataset has no entry,
``default_max_depth`` applies (conservative by default).

Depths, from most-private to most-permissive (each tier includes
everything above it):

- ``names_only`` — just the list of variable names.
- ``names_types`` — + a coarse type per variable (default).
- ``names_types_labels`` — + variable labels and value labels
  (the categorical-level dictionaries in `.dta` files).
- ``names_types_labels_summary`` — + per-variable NA count and
  distinct-value count for categoricals.

Never at any depth: raw observation values, min, max, median,
quantiles, or any frequency distributions. Those belong to
``request_data`` (with its own SDC rules) or ``submit_script``
(sanitized through the result pipeline).

Design notes:

- The policy is a **ceiling**, not a fixed value. The model may
  request a lower depth than the ceiling — e.g., a dataset with a
  ``names_types_labels`` ceiling can still be queried at
  ``names_only`` when labels are unnecessary for the task at
  hand. Sift enforces "at most", not "exactly".

- Interactive policy editing is wired up in both frontends through
  the same backend method (see ``ui.py:set_dataset_policy``); the
  web UI exposes a compact "Policy" chip next to the composer (see
  ``web/app.js:updatePolicyChip``) that unfurls a per-dataset
  dropdown. The JSON file is still the single source of truth —
  both UIs just read and write it — so a researcher comfortable
  editing it directly can still do that.

- A malformed policy file (truncated JSON, wrong shape, future
  schema version) fails closed: the in-memory policy returned has
  ``default_max_depth = "names_only"`` so schema access is denied
  until the file is repaired. A broken consent file is not a
  fresh-session signal — it's most often a partial write or editor
  mishap, and silently reverting to the rich default would expose
  metadata the researcher had previously restricted. Per-entry
  malformations clamp to the strictest tier on the same reasoning.
  Loading never raises.

Per-dataset privacy profile
----------------------------

Each dataset entry can also carry a ``privacy_profile`` — a
human-scale classification (``public`` / ``internal`` / ``confidential``
/ ``regulated``, in ``PRIVACY_PROFILES``, least to most restrictive)
that maps to its OWN schema-depth ceiling
(``PROFILE_MAX_DEPTH_CEILING``). ``effective_max_depth()`` is the
authoritative ceiling query: it returns the MORE restrictive of the
dataset's explicit ``max_depth`` (or the file-wide default, if unset)
and its privacy profile's ceiling. Every enforcement call site
(``get_schema``, ``search_schema``) uses ``effective_max_depth()``,
never ``get_max_depth()`` alone, so a profile can never be
circumvented by a looser ``max_depth`` value in the same entry (or
vice versa — whichever is stricter always wins).

Same "researcher consent, never silently changed" posture as the
rest of this module: a dataset with no explicit ``privacy_profile``
defaults to ``"internal"`` (the same non-alarmist default a fresh
``max_depth`` gets) — Sift never auto-classifies a dataset as
confidential/regulated on its own, even when the local-only PII/PHI
detector (``dataset_profile.py``) flags it. That detector's output is
surfaced to the researcher as information to act on, not as a
trigger this module reacts to; changing the effective policy always
requires the researcher's own edit, exactly like ``max_depth`` today.
An unrecognised ``privacy_profile`` string in the file clamps to
``"regulated"`` (the strictest tier), matching the fail-closed
handling every other malformed field in this module already gets.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from sift.text_safety import banned_key


# Depth identifiers. Must match the ``depth`` values in ``schema.py``
# and the documented enumeration in the ``get_schema`` tool help.
VALID_DEPTHS: tuple[str, ...] = (
    "names_only",
    "names_types",
    "names_types_labels",
    "names_types_labels_summary",
)

# Default schema depth for new datasets. NA count + distinct count is
# the richest non-leaky tier (below that, Claude is reasoning about
# variables with almost no metadata). Researchers can still dial it
# down per dataset via the Permission chip.
DEFAULT_MAX_DEPTH = "names_types_labels_summary"

# Fail-closed depth used when the policy file exists but is broken
# (truncated JSON, wrong shape, future schema version, malformed
# entry). The premise: a researcher who wrote a policy.json had
# opinions about their data, and the most likely reason their file
# is unreadable is a partial write or an editor mishap — NOT a fresh
# session. Defaulting to the richest tier in that state would silently
# expose metadata they had previously restricted. Falling back to the
# strictest tier instead means schema queries get denied until the
# file is repaired, which is the right loudness for "your consent
# policy is unreadable."
FAIL_CLOSED_MAX_DEPTH = "names_only"

# Map depth → rank so we can compare "is requested at most the ceiling".
_DEPTH_RANK: dict[str, int] = {d: i for i, d in enumerate(VALID_DEPTHS)}

POLICY_FILE = Path(".sift") / "policy.json"
_POLICY_KEYS = frozenset({"version", "default_max_depth", "datasets"})
_DATASET_POLICY_KEYS = frozenset({
    "max_depth",
    "set_at",
    "non_disclosive_variables",
    "privacy_profile",
    "banned_variables",
    "exportable",
    "dp_epsilon",
    "excel_sheet",
})

# Least to most restrictive. Deliberately a small, fixed vocabulary
# (not free text) so every consumer — this module, the UI dropdown,
# a future export-rules engine (see the docstring) — can rely on an
# exhaustive, orderable set rather than parsing researcher-typed
# strings.
PRIVACY_PROFILES: tuple[str, ...] = (
    "public", "internal", "confidential", "regulated",
)

# Default profile for a dataset with no explicit entry. "internal" —
# not "public" — because the vast majority of research data is not
# meant for open publication even when it isn't classically sensitive;
# but not "confidential" either, since defaulting every fresh dataset
# to a restrictive ceiling would contradict this module's existing
# "fresh session = permissive default, researcher dials down" posture
# for ``max_depth``.
DEFAULT_PRIVACY_PROFILE = "internal"

# Fail-closed profile for an unrecognised value in the policy file —
# same reasoning as ``FAIL_CLOSED_MAX_DEPTH``.
FAIL_CLOSED_PRIVACY_PROFILE = "regulated"

# Each profile's OWN schema-depth ceiling. "public" and "internal"
# don't add any restriction beyond the file/dataset's own
# ``max_depth`` (they cap at the richest tier, i.e. no-op as a
# ceiling); "confidential" blocks the summary-statistics tier
# (NA/distinct counts); "regulated" — HIPAA-style data, national ID
# registries — allows only variable names, nothing else, regardless
# of what ``max_depth`` says.
PROFILE_MAX_DEPTH_CEILING: dict[str, str] = {
    "public": "names_types_labels_summary",
    "internal": "names_types_labels_summary",
    "confidential": "names_types_labels",
    "regulated": "names_only",
}
_PROFILE_RANK: dict[str, int] = {p: i for i, p in enumerate(PRIVACY_PROFILES)}

# Each profile's allowance of GRANTED releases (request_data answers
# plus submit_script results) touching one dataset within a session
# before ``privacy_budget.py`` starts adaptively tightening SDC
# suppression thresholds for that dataset. This is deliberately NOT
# an access control (nothing is blocked once a budget is exceeded --
# every release still passes the sanitizer's own per-call rules) --
# it is a session-level nudge toward more conservative thresholds as
# a dataset accumulates disclosures, mitigating the combination-of-
# releases risk ``query_fingerprint.py`` can only advise about.
# ``None`` means no budget pressure applies (suppression never
# tightens). "public" data has no meaningful re-identification risk
# from repeated queries by definition of the profile, so it stays
# unbounded; every other profile gets a finite allowance, tightening
# with the profile's own restrictiveness.
PRIVACY_BUDGET_BY_PROFILE: dict[str, int | None] = {
    "public": None,
    "internal": 150,
    "confidential": 40,
    "regulated": 15,
}


@dataclass(frozen=True)
class DatasetPolicy:
    """Policy for a single dataset.

    ``set_at`` is an ISO-8601 UTC string marking when the researcher
    wrote this entry. Empty means the entry is inherited (default)
    rather than explicitly set.

    ``non_disclosive_variables`` is the per-variable opt-in list:
    variables the researcher has explicitly judged safe to expose
    raw min / max / median for in descriptive results. Default empty
    — the conservative posture is "every variable's min/max could
    identify outlier individuals". Researchers add a variable here
    only after checking that its extremes don't single anyone out
    (e.g., ``age`` in years, ``year_of_birth``, ``education_years``;
    NOT ``salary`` or ``rare_diagnosis_code``).
    """
    max_depth: str = DEFAULT_MAX_DEPTH
    set_at: str = ""
    non_disclosive_variables: tuple[str, ...] = ()
    privacy_profile: str = DEFAULT_PRIVACY_PROFILE
    # Variables the researcher has explicitly excluded from Claude's
    # view entirely — the opposite direction from
    # ``non_disclosive_variables``. Stored as SANITIZED (safe_key'd)
    # names, matching how every disclosure-facing lookup in
    # ``data_request._resolve_variable`` already keys its
    # raw-column-name map, so a raw column containing control
    # characters or excess length still matches the same banned entry
    # the researcher typed. Enforced at TWO points, both Sift-owned
    # (never inside the submit_script sanitizer — see the long
    # comment on ``SDCConfig.banned_variables`` in sanitizer.py for
    # why that boundary can't safely enforce a name-based block):
    # schema exposure (get_schema/search_schema drop the variable
    # from the response entirely) and request_data (denies any
    # request naming a banned variable, checked against the RESOLVED
    # real column, not the model's requested string).
    banned_variables: tuple[str, ...] = ()
    # When False, this dataset is excluded from codebook/report
    # exports built from the session (research_export.py) — for data
    # under a use restriction that permits analysis but not any
    # artifact leaving the researcher's own session state.
    exportable: bool = True
    # Per-query epsilon for the ``noisy_count`` request_data type
    # (differential_privacy.py) — a SEPARATE, explicit opt-in from
    # every other field here. ``None`` (default) disables DP entirely
    # for this dataset; ``noisy_count`` is denied outright regardless
    # of any other policy setting. Only the field's TYPE is validated
    # on load (a non-numeric or corrupted value also degrades to
    # "disabled", the maximally conservative outcome for an opt-in
    # mechanism); the semantic range check
    # (``differential_privacy.MIN_EPSILON`` /``MAX_EPSILON``) is
    # deliberately left to ``data_request._noisy_count`` at actual
    # use time, not duplicated here — that keeps this module's job
    # limited to "is this a usable number" and the mechanism's own
    # module authoritative for "is this number a sound privacy
    # parameter".
    dp_epsilon: float | None = None
    # Researcher's saved worksheet choice for a multi-sheet ``.xlsx``
    # file — ``None`` (default) means "first worksheet", the same
    # behaviour every reader in this codebase used before this field
    # existed. Only the field's TYPE is validated on load (a
    # non-string or corrupted value degrades to None, i.e. first
    # worksheet — never an error, and never a silently WRONG sheet);
    # whether the named sheet actually exists in a given file is
    # checked at read time by the reader itself (pandas/openpyxl
    # raise a clear error for an unknown sheet name), not duplicated
    # here. Irrelevant for every non-``.xlsx`` dataset — carried on
    # every ``DatasetPolicy`` regardless of format for the same
    # reason ``dp_epsilon`` is (uniform shape, ignored where it
    # doesn't apply, same posture documented on that field above).
    excel_sheet: str | None = None


@dataclass(frozen=True)
class SiftPolicy:
    """Top-level policy document.

    ``datasets`` keys are dataset filenames (not full paths) — the
    policy lives inside ``<cwd>/.sift/policy.json`` so paths are
    already relative to cwd.
    """
    version: int = 1
    default_max_depth: str = DEFAULT_MAX_DEPTH
    datasets: dict[str, DatasetPolicy] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def policy_path(cwd: Path) -> Path:
    """Return the canonical path to the policy file for ``cwd``."""
    return cwd / POLICY_FILE


def _fail_closed_policy() -> SiftPolicy:
    """Return the policy used when ``policy.json`` exists but is unreadable.

    Set ``default_max_depth`` to the strictest tier so a researcher
    who tightened their consent policy doesn't see it silently revert
    to the richest tier just because the JSON file got a stray
    character. Schema requests for unrestricted datasets will be
    denied until the file is repaired — the correct loudness for "we
    can't read your policy file."
    """
    return SiftPolicy(default_max_depth=FAIL_CLOSED_MAX_DEPTH)


def load_policy(cwd: Path) -> SiftPolicy:
    """Load the policy for ``cwd``.

    Never raises. Behavior in three regimes:

    - File absent: return the permissive in-memory default
      (``DEFAULT_MAX_DEPTH``). A fresh session has no expressed
      researcher opinion to honor, so the rich-by-default tier is
      correct — they can dial it down per-dataset later.

    - File present but unparseable / wrong shape / future version:
      fail closed. Return ``_fail_closed_policy()`` so schema access
      defaults to ``names_only`` until the file is repaired. The
      previous behavior fell back to the rich default here, which
      meant a partial write could silently expose metadata that the
      researcher had explicitly restricted.

    - File present and parseable: honor the document. Per-entry
      malformations (unknown ``max_depth``, missing fields) clamp the
      offending entry to the strictest tier rather than to the rich
      default, on the same fail-closed reasoning.
    """
    p = policy_path(cwd)
    if not p.is_file():
        return SiftPolicy()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _fail_closed_policy()
    if not isinstance(data, dict):
        return _fail_closed_policy()
    if data.get("version") != 1:
        # Future versions should have a migration path, but until one
        # exists, fail closed rather than misinterpret. A version-skewed
        # file wasn't written for this code path; treating its absent
        # entries as "no opinion" would silently re-open access the
        # newer version may have tightened.
        return _fail_closed_policy()
    if not set(data).issubset(_POLICY_KEYS):
        # Unknown keys are usually misspelled controls. Ignoring a typo such
        # as ``default_max_dept`` would silently restore the rich default.
        return _fail_closed_policy()

    default_max = data.get("default_max_depth", DEFAULT_MAX_DEPTH)
    if default_max not in VALID_DEPTHS:
        default_max = FAIL_CLOSED_MAX_DEPTH

    datasets: dict[str, DatasetPolicy] = {}
    raw = data.get("datasets")
    if "datasets" in data and not isinstance(raw, dict):
        return _fail_closed_policy()
    if isinstance(raw, dict):
        for name, entry in raw.items():
            if not isinstance(name, str):
                # JSON keys are always strings, but be defensive.
                continue
            if not isinstance(entry, dict):
                # Malformed entry shape (e.g. ``"survey.csv": "names_only"``
                # written as shorthand without the wrapping dict).
                # Skipping the entry would silently fall back to
                # ``default_max_depth``, contradicting the "per-entry
                # malformations clamp to the strictest tier" rule in the
                # module docstring. Record a fail-closed entry so the
                # researcher's apparent intent — they wrote a key with
                # this dataset name — is honoured at the strictest tier.
                datasets[name] = DatasetPolicy(max_depth=FAIL_CLOSED_MAX_DEPTH)
                continue
            if not set(entry).issubset(_DATASET_POLICY_KEYS):
                # A misspelled per-dataset restriction must not disappear.
                datasets[name] = DatasetPolicy(
                    max_depth=FAIL_CLOSED_MAX_DEPTH,
                    privacy_profile=FAIL_CLOSED_PRIVACY_PROFILE,
                    exportable=False,
                )
                continue
            max_depth = entry.get("max_depth", FAIL_CLOSED_MAX_DEPTH)
            if max_depth not in VALID_DEPTHS:
                # The entry exists — researcher had an opinion — but
                # the depth string is unrecognised. Clamp to the
                # strictest tier rather than letting the file-wide
                # default take over (which could be more permissive
                # than what the researcher intended).
                max_depth = FAIL_CLOSED_MAX_DEPTH
            set_at = entry.get("set_at", "")
            if not isinstance(set_at, str):
                set_at = ""
            ndv_raw = entry.get("non_disclosive_variables", [])
            if isinstance(ndv_raw, list):
                # ``banned_key`` (safe_key + casefold), not bare
                # ``str`` -- same reasoning as ``banned_variables``
                # just below: request_data resolves a requested name
                # to a REAL DataFrame column (see
                # ``data_request._numeric_bounds``), and a policy
                # file listing "Age" must still match a column
                # literally named "age". Bare string comparison here
                # would let a case mismatch alone silently defeat the
                # opt-in -- the researcher believes the variable's
                # real min/max are exposed, and numeric_bounds keeps
                # quietly returning percentiles only.
                non_disclosive = tuple(
                    banned_key(str(v)) for v in ndv_raw
                    if isinstance(v, str) and v
                )
            else:
                non_disclosive = ()
            # Unlike ``max_depth`` (the field every entry has always
            # had), ``privacy_profile`` is new — a policy.json written
            # before this field existed, or by a researcher who only
            # ever touched the depth chip, will have entries with NO
            # ``privacy_profile`` key at all. That's the ordinary,
            # expected case, not corruption, so ABSENCE defaults to
            # ``DEFAULT_PRIVACY_PROFILE`` ("no opinion expressed on
            # this axis yet" — same reasoning ``get_privacy_profile``
            # uses for datasets with no entry at all). Only a PRESENT
            # but unrecognised value fails closed — that genuinely is
            # a corruption signal (a future-version profile name, a
            # hand-edit typo), and silently defaulting a corrupted
            # value to "internal" could re-open access a researcher
            # had deliberately tightened.
            profile = entry.get("privacy_profile", DEFAULT_PRIVACY_PROFILE)
            if profile not in PRIVACY_PROFILES:
                profile = FAIL_CLOSED_PRIVACY_PROFILE
            banned_raw = entry.get("banned_variables", [])
            if isinstance(banned_raw, list):
                # ``banned_key`` (safe_key + casefold), not bare
                # ``safe_key`` — a policy file listing "SSN" must
                # also ban a dataset column literally named "ssn";
                # see banned_key's docstring for why a case mismatch
                # here would otherwise silently defeat the ban.
                banned = tuple(
                    banned_key(str(v)) for v in banned_raw
                    if isinstance(v, str) and v
                )
            else:
                # The user tried to set a deny-list but it is unreadable.
                # There is no safe way to infer which fields they intended,
                # so make the whole entry non-exportable and metadata-minimal.
                datasets[name] = DatasetPolicy(
                    max_depth=FAIL_CLOSED_MAX_DEPTH,
                    privacy_profile=FAIL_CLOSED_PRIVACY_PROFILE,
                    exportable=False,
                )
                continue
            # Absence means the researcher has expressed no export
            # restriction, preserving the historical default. A PRESENT but
            # malformed value is different: treating ``"false"`` or a
            # damaged value as True silently re-opens export. Fail closed on
            # that axis until the policy entry is repaired.
            if "exportable" not in entry:
                exportable = True
            else:
                exportable_raw = entry.get("exportable")
                exportable = (
                    exportable_raw if isinstance(exportable_raw, bool) else False
                )
            # Absence -> None (disabled), same as every other opt-in
            # default in this dataclass. A present-but-unusable value
            # (wrong type, NaN/inf, or the JSON boolean literals —
            # ``isinstance(x, bool)`` is checked first since ``bool``
            # is a ``int`` subclass in Python and would otherwise
            # silently pass the numeric check) also degrades to None
            # rather than raising or propagating a garbage epsilon
            # into the sanitizer config — "disabled" is already the
            # strictest possible outcome for this field, so there is
            # no separate fail-closed value to reach for the way
            # ``privacy_profile`` needed one.
            dp_epsilon_raw = entry.get("dp_epsilon")
            if (isinstance(dp_epsilon_raw, (int, float))
                    and not isinstance(dp_epsilon_raw, bool)
                    and math.isfinite(dp_epsilon_raw)):
                dp_epsilon = float(dp_epsilon_raw)
            else:
                dp_epsilon = None
            excel_sheet_raw = entry.get("excel_sheet")
            excel_sheet = (
                excel_sheet_raw
                if isinstance(excel_sheet_raw, str) and excel_sheet_raw
                else None
            )
            datasets[name] = DatasetPolicy(
                max_depth=max_depth,
                set_at=set_at,
                non_disclosive_variables=non_disclosive,
                privacy_profile=profile,
                banned_variables=banned,
                exportable=exportable,
                dp_epsilon=dp_epsilon,
                excel_sheet=excel_sheet,
            )

    return SiftPolicy(
        version=1, default_max_depth=default_max, datasets=datasets
    )


def save_policy(cwd: Path, policy: SiftPolicy) -> None:
    """Persist ``policy`` to ``<cwd>/.sift/policy.json``.

    Creates the ``.sift`` directory if it doesn't already exist.
    """
    p = policy_path(cwd)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "version": policy.version,
        "default_max_depth": policy.default_max_depth,
        "datasets": {
            name: {
                "max_depth": dp.max_depth,
                "set_at": dp.set_at,
                # Only emit the field when populated — keeps the
                # default policy file tidy for datasets that don't
                # use the opt-in.
                **(
                    {"non_disclosive_variables": list(dp.non_disclosive_variables)}
                    if dp.non_disclosive_variables else {}
                ),
                # Only emit when non-default, same tidiness rationale
                # as non_disclosive_variables above.
                **(
                    {"privacy_profile": dp.privacy_profile}
                    if dp.privacy_profile != DEFAULT_PRIVACY_PROFILE else {}
                ),
                **(
                    {"banned_variables": list(dp.banned_variables)}
                    if dp.banned_variables else {}
                ),
                **(
                    {"exportable": dp.exportable}
                    if dp.exportable is not True else {}
                ),
                **(
                    {"dp_epsilon": dp.dp_epsilon}
                    if dp.dp_epsilon is not None else {}
                ),
                **(
                    {"excel_sheet": dp.excel_sheet}
                    if dp.excel_sheet is not None else {}
                ),
            }
            for name, dp in policy.datasets.items()
        },
    }
    # Write atomically via a sibling tmp file + ``os.replace``. Direct
    # ``write_text`` truncates ``policy.json`` and then streams bytes;
    # a crash mid-write (or a second writer that wins the race) leaves
    # a half-written file on disk, which the next ``load_policy`` reads
    # as malformed JSON and silently falls back to defaults — silently
    # widening every dataset's max_depth ceiling. Two known concurrent-
    # writer paths exist today: the web UI's policy editor and the
    # researcher TUI both call ``save_policy``, and the researcher can
    # have both open. ``NamedTemporaryFile(dir=p.parent)`` puts the tmp
    # file on the same filesystem so ``os.replace`` is a true atomic
    # rename (cross-fs ``os.replace`` falls back to copy-then-unlink,
    # which loses atomicity).
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".policy.json.", suffix=".tmp", dir=p.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, p)
    except Exception:
        # Best-effort cleanup of the orphan tmp file. ``os.replace``
        # consumes the source on success, so this only matters when
        # the write or rename failed.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_max_depth(policy: SiftPolicy, dataset_name: str) -> str:
    """Return the ceiling depth for ``dataset_name`` under ``policy``.

    Falls back to ``policy.default_max_depth`` if the dataset has no
    explicit entry. This is the RAW ``max_depth`` field only — it
    does NOT factor in the dataset's privacy profile. Enforcement
    call sites should use ``effective_max_depth()`` instead; this
    function is kept for callers (and tests) that specifically need
    the un-combined ``max_depth`` value.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return policy.default_max_depth
    return entry.max_depth


def get_privacy_profile(policy: SiftPolicy, dataset_name: str) -> str:
    """Return the privacy profile for ``dataset_name`` under ``policy``.

    Falls back to ``DEFAULT_PRIVACY_PROFILE`` if the dataset has no
    explicit entry — same "no expressed opinion yet" reasoning as
    ``get_max_depth``'s fallback to ``default_max_depth``.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return DEFAULT_PRIVACY_PROFILE
    return entry.privacy_profile


def effective_max_depth(policy: SiftPolicy, dataset_name: str) -> str:
    """Return the ACTUAL enforced ceiling for ``dataset_name``: the
    more restrictive of its explicit/default ``max_depth`` and its
    privacy profile's own ceiling.

    This is the function every enforcement call site (``get_schema``,
    ``search_schema``) must consult — never ``get_max_depth()`` alone,
    which only reflects the ``max_depth`` field and would let a
    ``regulated``-profile dataset with a stale, looser ``max_depth``
    value leak past its profile's intended ceiling.
    """
    raw_ceiling = get_max_depth(policy, dataset_name)
    profile = get_privacy_profile(policy, dataset_name)
    profile_ceiling = PROFILE_MAX_DEPTH_CEILING.get(
        profile, FAIL_CLOSED_MAX_DEPTH,
    )
    if raw_ceiling not in _DEPTH_RANK:
        raw_ceiling = FAIL_CLOSED_MAX_DEPTH
    # Lower rank == stricter (VALID_DEPTHS is ordered least to most
    # permissive) — the effective ceiling is whichever ranks lower.
    return (
        raw_ceiling
        if _DEPTH_RANK[raw_ceiling] <= _DEPTH_RANK[profile_ceiling]
        else profile_ceiling
    )


def depth_allowed(requested: str, ceiling: str) -> bool:
    """Return ``True`` iff ``requested`` depth is at or below ``ceiling``.

    Unknown depths are rejected (not the caller's bug to silently
    accept — callers should have validated against ``VALID_DEPTHS``
    before consulting the policy).
    """
    if requested not in _DEPTH_RANK or ceiling not in _DEPTH_RANK:
        return False
    return _DEPTH_RANK[requested] <= _DEPTH_RANK[ceiling]


def has_explicit_policy(policy: SiftPolicy, dataset_name: str) -> bool:
    """Return ``True`` iff ``policy`` has an explicit entry for the
    dataset (not inherited from ``default_max_depth``).
    """
    return dataset_name in policy.datasets


def non_disclosive_for(
    policy: SiftPolicy, dataset_name: str
) -> frozenset[str]:
    """Return the set of variable names the researcher has explicitly
    marked as non-disclosive for ``dataset_name``.

    Empty set when the dataset has no explicit entry — the default
    posture is "no variable is opted-in to min/max disclosure".
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return frozenset()
    return frozenset(entry.non_disclosive_variables)


def banned_for(policy: SiftPolicy, dataset_name: str) -> frozenset[str]:
    """Return the set of SANITIZED variable names banned from Claude's
    view entirely for ``dataset_name``.

    Empty set when the dataset has no explicit entry — the default
    posture is "no variable is banned". Callers must compare against
    a ``banned_key()``-normalized candidate name (matching how the
    values were normalized on load: ``safe_key`` collapsing control
    characters / excess length / embedded whitespace, PLUS case-
    folding) — never the raw column name and never a bare
    ``safe_key()`` result, either of which could differ in case from
    what the researcher typed and silently fail to match.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return frozenset()
    return frozenset(entry.banned_variables)


def is_exportable(policy: SiftPolicy, dataset_name: str) -> bool:
    """Return whether ``dataset_name`` may appear in session exports
    (codebook, analysis reports) built from ``research_export.py``.

    Defaults to ``True`` — the default posture is "analysis is fine,
    exporting artifacts about it is fine" unless a researcher
    explicitly restricts a dataset. This mirrors the "fresh session,
    no expressed opinion, permissive default" reasoning ``max_depth``
    and ``privacy_profile`` both use for their own no-entry case.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return True
    return entry.exportable


def get_dp_epsilon(policy: SiftPolicy, dataset_name: str) -> float | None:
    """Return the opt-in differential-privacy epsilon for
    ``dataset_name``, or ``None`` when DP is disabled (no explicit
    entry, or an entry that never set ``dp_epsilon``). ``None`` is
    the correct default for "no expressed opinion" here — unlike
    ``privacy_profile`` / ``max_depth``, this mechanism doesn't have
    a rich-by-default posture to fall back to; DP is off until a
    researcher turns it on.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return None
    return entry.dp_epsilon


def get_excel_sheet(policy: SiftPolicy, dataset_name: str) -> str | None:
    """Return the researcher's saved worksheet choice for
    ``dataset_name``, or ``None`` when none is set (no explicit
    entry, or an entry that never set ``excel_sheet``) — meaning
    "read the first worksheet," the pre-existing default every
    ``.xlsx`` reader in this codebase used before this field
    existed. Irrelevant for non-``.xlsx`` datasets; callers only
    consult this for files where a sheet selection makes sense.
    """
    entry = policy.datasets.get(dataset_name)
    if entry is None:
        return None
    return entry.excel_sheet
