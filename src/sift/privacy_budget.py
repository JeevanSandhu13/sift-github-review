"""Sift — privacy budgets and adaptive suppression.

Every individual release (a ``request_data`` answer, a
``submit_script`` result) is disclosure-controlled on its own by the
sanitizer/SDC layer, using FIXED thresholds — that layer is
deliberately stateless (see ``sanitizer.SDCConfig`` and
``query_fingerprint.py``'s docstring for why). This module adds a
session-level, dataset-scoped notion of accumulated exposure: as a
dataset accumulates more granted releases, the SDC thresholds applied
to FUTURE releases against that same dataset are adaptively
tightened. This is a mitigation for the combination-of-releases risk
``query_fingerprint.py`` can only advise about — instead of only
flagging the pattern after the fact, later releases against a
heavily-queried dataset are held to a stricter bar.

**What this module is not.** It is not access control — no release is
ever blocked by a budget on its own (a request that would otherwise
be granted stays granted; only the suppression MATH applied to it
gets stricter, which for a small enough result can turn a former
"granted" into a "denied" through the normal min-N gate, exactly as
if the researcher had configured stricter thresholds by hand). It is
not differential privacy — there is no formal epsilon accounting
here; see ``differential_privacy.py`` for the module that
scopes an honest, opt-in DP mechanism. This module's guarantee is
much weaker and more mechanical: thresholds only ever move in the
conservative direction as consumption rises, never the reverse, and
every adjustment is a deterministic function of a single visible
number (releases granted so far against this dataset this session).

**Budget accounting.** The budget is read from the dataset's privacy
profile (``policy.PRIVACY_BUDGET_BY_PROFILE``) — a count of granted
releases allowed before suppression starts tightening. Consumption is
counted from the existing release ledger (no new logging): a
``request_data`` record counts if its response's ``status`` was
``"granted"``; a ``submit_script`` record counts once per matching
``source_dataset`` entry in its recorded facts (top-level or, for a
batch call, each ``facts.results`` entry) whose status is ``"ok"``.
Rejected payloads are deliberately retained in the ledger for audit,
but they did not disclose an analysis result and therefore consume no
release budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from sift.policy import PRIVACY_BUDGET_BY_PROFILE
from sift.release_ledger import same_dataset, verified_ledger_snapshot
from sift.sanitizer import SDCConfig

# Consumption-ratio (consumed / budget) breakpoints. Ratio strictly
# below the first breakpoint is TIER_NORMAL (no adjustment); at or
# above it but below the second is TIER_ELEVATED; at or above the
# second is TIER_STRICT. Chosen so a dataset that's merely AT its
# budget isn't punished yet (crossing 1x is exactly the interesting
# moment — before that, "budget" would just be a synonym for a hard
# cutoff, which isn't the intent), but a dataset running well past
# its allowance (2x+) gets the strongest available tightening.
TIER_NORMAL = 0
TIER_ELEVATED = 1
TIER_STRICT = 2

_ELEVATED_RATIO = 1.0
_STRICT_RATIO = 2.0

# Multipliers applied to count-style thresholds (higher = stricter:
# more observations required before a result is releasable at all).
# Applied with ceil() so a multiplier can never round DOWN and
# accidentally loosen an integer threshold.
_COUNT_MULTIPLIER = {TIER_NORMAL: 1.0, TIER_ELEVATED: 1.5, TIER_STRICT: 2.0}

# Multipliers applied to the dominance threshold (lower = stricter:
# a smaller excess share is enough to trigger suppression). Floored
# so tightening can't degrade into "every cell is suppressed
# regardless of content", which would stop being a meaningful signal.
_DOMINANCE_MULTIPLIER = {TIER_NORMAL: 1.0, TIER_ELEVATED: 0.9, TIER_STRICT: 0.75}
_DOMINANCE_FLOOR = 0.5

# The SDCConfig fields this module adjusts. Every one of these is a
# minimum-N-style gate (higher = stricter) EXCEPT dominance_threshold,
# handled separately below. Kept as an explicit tuple (not
# introspected via dataclasses.fields) so a future SDCConfig field
# addition doesn't silently start being adjusted without a deliberate
# decision about its direction.
_COUNT_FIELDS: tuple[str, ...] = (
    "min_n_regression", "min_n_descriptive", "min_n_ttest_group",
    "cell_suppression_threshold", "min_n_did_cohort",
)


def tier_name(tier: int) -> str:
    return {TIER_NORMAL: "normal", TIER_ELEVATED: "elevated",
            TIER_STRICT: "strict"}.get(tier, "normal")


@dataclass(frozen=True)
class BudgetStatus:
    dataset: str
    privacy_profile: str
    budget: int | None
    consumed: int
    tier: int
    accounting_ok: bool = True
    accounting_detail: str = "ok"

    @property
    def tier_label(self) -> str:
        return tier_name(self.tier)

    @property
    def unbounded(self) -> bool:
        return self.budget is None


def budget_for_profile(profile: str) -> int | None:
    """Granted-release allowance for ``profile`` before adaptive
    suppression begins tightening. ``None`` means unbounded (no
    tightening ever applies) — currently only "public". An
    unrecognised profile string is treated the same as the strictest
    known profile's budget, matching this codebase's established
    fail-closed posture for unrecognised policy values (see
    ``policy.FAIL_CLOSED_PRIVACY_PROFILE``)."""
    if profile in PRIVACY_BUDGET_BY_PROFILE:
        return PRIVACY_BUDGET_BY_PROFILE[profile]
    return min(
        (b for b in PRIVACY_BUDGET_BY_PROFILE.values() if b is not None),
        default=None,
    )


def _request_data_granted_count(
    records: list[dict[str, Any]], dataset: str, *, cwd: Path | None = None,
) -> int:
    n = 0
    for rec in records:
        if rec.get("tool") != "request_data":
            continue
        args = rec.get("args") or {}
        if not same_dataset(
            args.get("dataset"), dataset, cwd=cwd,
            record_identity=args.get("dataset_identity"),
        ):
            continue
        facts = rec.get("facts") or {}
        if facts.get("status") == "granted":
            n += 1
    return n


def _submit_script_granted_count(
    records: list[dict[str, Any]], dataset: str, *, cwd: Path | None = None,
) -> int:
    n = 0
    for rec in records:
        if rec.get("tool") != "submit_script":
            continue
        facts = rec.get("facts") or {}
        candidates = [facts] + list(facts.get("results") or [])
        for c in candidates:
            if not isinstance(c, dict) or c.get("status") != "ok":
                continue
            raw_sources = c.get("source_datasets")
            sources = list(raw_sources) if isinstance(raw_sources, list) else []
            if c.get("source_dataset"):
                sources.append(c["source_dataset"])
            identities = c.get("source_dataset_identities")
            identity_by_index = (
                identities if isinstance(identities, list) else []
            )
            singular_identity = c.get("source_dataset_identity")
            matched = False
            for index, source in enumerate(sources):
                record_identity = (
                    singular_identity
                    if index == len(sources) - 1 and c.get("source_dataset")
                    else identity_by_index[index]
                    if index < len(identity_by_index)
                    else None
                )
                if same_dataset(
                    source, dataset, cwd=cwd,
                    record_identity=record_identity,
                ):
                    matched = True
                    break
            if matched:
                n += 1
    return n


def consumed_for_dataset(
    records: list[dict[str, Any]], dataset: str, *, cwd: Path | None = None,
) -> int:
    """Count of granted releases touching ``dataset`` in ``records``.

    With ``cwd``, new records are matched by their canonical session-relative
    identity so symlink aliases collapse while distinct same-named files do
    not. Legacy/synthetic records fall back to conservative basename matching.
    """
    return (_request_data_granted_count(records, dataset, cwd=cwd)
            + _submit_script_granted_count(records, dataset, cwd=cwd))


def tier_for_consumption(consumed: int, budget: int | None) -> int:
    """Pure function from (consumed, budget) to a suppression tier.
    ``budget=None`` (unbounded) always yields TIER_NORMAL."""
    if budget is None or budget <= 0:
        return TIER_NORMAL
    ratio = consumed / budget
    if ratio >= _STRICT_RATIO:
        return TIER_STRICT
    if ratio >= _ELEVATED_RATIO:
        return TIER_ELEVATED
    return TIER_NORMAL


def status_for_dataset(
    cwd: Path, dataset: str, privacy_profile: str,
) -> BudgetStatus:
    """Full budget status for ``dataset`` under ``privacy_profile``.

    This heuristic budget never blocks on its own, but an unreadable,
    malformed, or gap-bearing ledger must not reset suppression to the
    baseline.  Bounded profiles therefore fail closed to the strictest
    adaptive tier when history cannot be trusted.  Public data remains
    unbounded.  The status reports integrity separately so callers can
    explain why strict suppression is active.
    """
    accounting_ok = False
    accounting_detail = "privacy accounting history is unavailable"
    records, accounting_ok, accounting_detail = verified_ledger_snapshot(
        Path(cwd),
    )
    budget = budget_for_profile(privacy_profile)
    consumed = consumed_for_dataset(records, dataset, cwd=Path(cwd))
    tier = (
        tier_for_consumption(consumed, budget)
        if accounting_ok or budget is None
        else TIER_STRICT
    )
    return BudgetStatus(dataset=dataset, privacy_profile=privacy_profile,
                        budget=budget, consumed=consumed, tier=tier,
                        accounting_ok=accounting_ok,
                        accounting_detail=accounting_detail)


def adjusted_sdc_config(base: SDCConfig, tier: int) -> SDCConfig:
    """Return a NEW ``SDCConfig`` with count-style thresholds scaled
    up and the dominance threshold scaled down for ``tier`` — always
    at least as strict as ``base``, never looser, regardless of
    ``tier``'s value (an unrecognised/negative tier falls back to
    TIER_NORMAL's 1.0 multipliers via ``.get(..., 1.0)`` rather than
    raising, so a caller passing a bad tier fails safe to "no
    adjustment" rather than an unhandled exception surfacing mid
    dataset access).

    Uses ``dataclasses.replace`` — ``base`` itself, and the
    ``SDCConfig``/``sanitizer.py`` definitions, are never mutated or
    imported for anything other than constructing the adjusted copy.
    """
    count_mult = _COUNT_MULTIPLIER.get(tier, 1.0)
    dom_mult = _DOMINANCE_MULTIPLIER.get(tier, 1.0)

    updates: dict[str, Any] = {}
    for field_name in _COUNT_FIELDS:
        base_value = getattr(base, field_name)
        updates[field_name] = max(
            base_value, math.ceil(base_value * count_mult),
        )
    updates["dominance_threshold"] = max(
        _DOMINANCE_FLOOR,
        min(base.dominance_threshold, base.dominance_threshold * dom_mult),
    )
    return replace(base, **updates)


def advisory_note(status: BudgetStatus) -> str | None:
    """Short researcher-facing advisory string for a non-normal tier,
    ``None`` at TIER_NORMAL or when the profile is unbounded. Mirrors
    ``query_fingerprint.py``'s advisory style — informational, not a
    warning that anything went wrong; every release this describes
    already passed its (now stricter) SDC check on its own merits."""
    if status.unbounded or status.tier == TIER_NORMAL:
        return None
    if not status.accounting_ok:
        return (
            f"privacy note: disclosure-accounting history for "
            f"{status.dataset!r} could not be verified "
            f"({status.accounting_detail}); the strictest adaptive "
            f"suppression tier is active until accounting is repaired."
        )
    return (
        f"privacy note: {status.dataset!r} has accumulated "
        f"{status.consumed} granted releases this session against a "
        f"{status.privacy_profile!r}-profile budget of {status.budget} "
        f"before suppression tightens ({status.tier_label} tier now "
        f"active) — disclosure-control thresholds for further "
        f"releases against this dataset are stricter than the "
        f"profile's baseline."
    )


def resolve_adaptive_sdc_config(
    cwd: Path, dataset: str, privacy_profile: str, base: SDCConfig,
) -> tuple[SDCConfig, BudgetStatus]:
    """Convenience: compute the budget status for ``dataset`` and
    return the correspondingly adjusted config alongside it, so a
    caller (``tools._resolve_sdc_and_source_n``) can both enforce the
    stricter config AND surface the status for advisory display in
    one call. Never raises."""
    status = status_for_dataset(cwd, dataset, privacy_profile)
    try:
        cfg = adjusted_sdc_config(base, status.tier)
    except Exception:  # noqa: BLE001 — must fail safe to the base config
        cfg = base
    return cfg, status
