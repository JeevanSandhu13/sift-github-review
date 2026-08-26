"""Sift — local dataset profile for the researcher.

A researcher opening a dataset should immediately see what they are
working with: how many rows, which variables, what is missing, what
looks like an identifier, what is constant, whether there are
duplicate rows. Sift surfaced only names and types; the questions
above were answerable only by asking the model to run a script, which
is slow, spends tokens, and — for the "is this file what I think it
is" question — is the wrong tool entirely.

**Privacy framing (the important part).** This profile is computed
locally and rendered locally. It is for the researcher looking at
their own data on their own machine, and it is *never* sent to the
model. That is why it can safely include things the disclosure
boundary deliberately withholds from the model, such as per-variable
minima and maxima or exact distinct counts.

The separation is structural, not a convention to remember:

- This module is reachable only from the UI bridge
  (``SiftBridge.get_dataset_profile``). It is not registered as a
  tool, so the model has no way to invoke it.
- It returns a plain dict to the frontend. Nothing in the tool layer
  imports it, so a profile cannot become a tool response by accident.
- What the *model* may learn about a dataset remains governed by
  ``policy.py`` (the schema-depth ceiling) and ``data_request.py``
  (bounded, SDC-checked facts). Neither is affected by this module.

Memory safety: profiling reads the file, so it honours the same
full-load ceiling as ``schema.load_data``. Above the ceiling it
profiles a bounded head sample instead of refusing outright — a
partial profile clearly labelled as such is more useful to a
researcher than nothing — and the result says exactly what it was
computed from.
"""

from __future__ import annotations

import copy
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from sift.schema import (
    DATA_EXTENSIONS, full_load_max_bytes, list_excel_sheets, row_count,
)
from sift.text_safety import safe_key

# Rows read when a file is too large to load whole. Enough to
# characterise types, missingness patterns and obvious identifier
# columns; explicitly reported as a sample so nothing here is mistaken
# for a full-file statistic.
_SAMPLE_ROWS = 50_000

# A column whose distinct-value count equals its non-null count is a
# candidate identifier. Requiring a minimum height avoids labelling
# every column in a 3-row file an "identifier".
_ID_MIN_ROWS = 20

# The profile panel is opened repeatedly while a researcher explores a
# session. Profiling a 50k-row sample can take seconds, yet the answer is
# immutable until the file changes. Cache a small number of completed local
# profiles by file identity/signature and return defensive copies so UI code
# cannot mutate the cached canonical value.
_PROFILE_CACHE_MAX = 16
_PROFILE_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_PROFILE_CACHE_LOCK = threading.RLock()


def _clear_profile_cache() -> None:
    """Test/session-lifecycle hook; normal invalidation is signature-based."""
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE.clear()


def _cached_profile(key: tuple[Any, ...]) -> dict[str, Any] | None:
    with _PROFILE_CACHE_LOCK:
        value = _PROFILE_CACHE.get(key)
        if value is None:
            return None
        _PROFILE_CACHE.move_to_end(key)
        return copy.deepcopy(value)


def _remember_profile(
    key: tuple[Any, ...], profile: dict[str, Any],
) -> dict[str, Any]:
    with _PROFILE_CACHE_LOCK:
        _PROFILE_CACHE[key] = copy.deepcopy(profile)
        _PROFILE_CACHE.move_to_end(key)
        while len(_PROFILE_CACHE) > _PROFILE_CACHE_MAX:
            _PROFILE_CACHE.popitem(last=False)
    return profile


def _fmt_bytes(n: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


class _NoSampledPathError(Exception):
    """A format with no partial reader was over the full-load ceiling."""

    def __init__(self, suffix: str):
        self.suffix = suffix
        super().__init__(suffix)


def profile_dataset(
    path: Path, *, sheet: str | int | None = None,
    session_root: Path | None = None,
) -> dict[str, Any]:
    """Return a local profile of the dataset at ``path``.

    ``sheet`` selects a worksheet for ``.xlsx``/``.xls``/``.ods`` (name or 0-based
    index; ``None`` reads the first, same default as everywhere else
    that reads Excel files in this codebase). Ignored for every other
    format.

    Never raises for ordinary problems (unreadable file, unsupported
    format, malformed contents): those come back as ``{"ok": False,
    "reason": ...}`` so the panel can render a message instead of the
    UI losing a frame to an exception.
    """
    path = Path(path)
    if not path.is_file():
        return {"ok": False, "reason": "file not found"}
    if path.suffix.lower() not in DATA_EXTENSIONS:
        return {"ok": False, "reason": f"unsupported format: {path.suffix}"}

    try:
        stat = path.stat()
        size_bytes = stat.st_size
    except OSError as e:
        return {"ok": False, "reason": f"could not stat file: {e}"}

    load_ceiling = full_load_max_bytes()
    sampled = size_bytes > load_ceiling
    selection = {"worksheet": sheet if sheet is not None else 0} \
        if path.suffix.casefold() in {".xlsx", ".xls", ".ods"} else {}
    canonical = None
    try:
        from sift.canonical_dataset import current_manifest, ensure_manifest
        canonical = current_manifest(
            session_root or path.parent, path, selection=selection,
        )
        if canonical is None and sampled:
            canonical = ensure_manifest(
                session_root or path.parent, path, selection=selection,
            )
    except Exception as e:  # noqa: BLE001 — parser diagnostics stay local
        return {
            "ok": False,
            "reason": f"could not establish dataset identity: {type(e).__name__}",
        }

    cache_key = None
    if canonical is not None:
        cache_key = (
            str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, repr(sheet), load_ceiling, canonical["fingerprint"],
        )
        cached = _cached_profile(cache_key)
        if cached is not None:
            return cached
    try:
        df, truncated = _read_frame(
            path, sampled, sheet=sheet, session_root=session_root,
        )
    except _NoSampledPathError as e:
        return {"ok": False,
                "reason": (
                    f"this {e.suffix} file is larger than the in-memory "
                    "profiling ceiling and the format has no partial "
                    "reader, so it cannot be profiled without loading "
                    "it whole. Analysis through scripts still works. "
                    "Raise SIFT_MAX_LOAD_BYTES if this machine has "
                    "the RAM."
                )}
    except MemoryError:
        return {"ok": False,
                "reason": "not enough memory to profile this file"}
    except Exception as e:  # noqa: BLE001 — malformed files are expected
        return {"ok": False, "reason": f"could not read file: {type(e).__name__}"}

    if df is None:
        return {"ok": False, "reason": "no tabular data found in file"}
    if canonical is None:
        try:
            canonical = ensure_manifest(
                session_root or path.parent, path, selection=selection,
            )
        except Exception as e:  # noqa: BLE001
            return {
                "ok": False,
                "reason": f"could not establish dataset identity: {type(e).__name__}",
            }
        cache_key = (
            str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
            stat.st_mtime_ns, repr(sheet), load_ceiling, canonical["fingerprint"],
        )

    profile: dict[str, Any] = {
        "ok": True,
        "name": path.name,
        "size_bytes": size_bytes,
        "size_display": _fmt_bytes(size_bytes),
        "sampled": bool(truncated),
        "rows_profiled": int(len(df)),
        "columns": int(df.shape[1]),
        "canonical_fingerprint": canonical["fingerprint"],
    }

    if path.suffix.lower() in {".xlsx", ".xls", ".ods"}:
        # Worksheet scope is a real decision for a multi-sheet
        # workbook — say which one this profile is of, and what else
        # is available, the same way schema.extract does for the
        # model-facing view.
        profile["sheet_read"] = 0 if sheet is None else sheet
        try:
            profile["available_sheets"] = list_excel_sheets(path)
        except Exception:  # noqa: BLE001 — advisory only
            pass

    # True row count for the whole file where a cheap path exists, so a
    # sampled profile still reports the real N.
    total_rows = row_count(path) if truncated else int(len(df))
    profile["rows"] = total_rows if total_rows is not None else int(len(df))
    profile["rows_exact"] = total_rows is not None or not truncated

    variables, flags = _profile_columns(df, truncated)
    profile["variables"] = variables

    # Whole-file structural flags. Duplicate detection is only honest
    # on a complete read; on a sample it would report the sample's
    # duplicates as the file's, so it is omitted rather than guessed.
    if not truncated:
        try:
            profile["duplicate_rows"] = int(df.duplicated().sum())
        except Exception:  # noqa: BLE001 — unhashable dtypes
            profile["duplicate_rows"] = None
    else:
        profile["duplicate_rows"] = None

    total_cells = max(1, int(df.shape[0]) * int(df.shape[1]))
    missing_cells = sum(v["missing"] for v in variables)
    profile["missing_pct"] = round(100.0 * missing_cells / total_cells, 2)
    profile.update(flags)
    profile["health"] = _compute_health(profile, variables, flags)
    # Expanded deterministic checks remain local-only, like this profile.
    from sift.data_quality import assess_frame
    profile["quality"] = assess_frame(df, sampled=bool(truncated))
    return _remember_profile(cache_key, profile) if cache_key is not None else profile


def _read_frame(
    path: Path, sampled: bool, *, sheet: str | int | None = None,
    session_root: Path | None = None,
):
    """Read the dataset, or a bounded head sample of it.

    Returns ``(dataframe_or_None, truncated)``. ``sheet`` selects a
    worksheet for spreadsheet formats; ignored for every other format.
    """
    import pandas as pd

    suffix = path.suffix.lower()

    if not sampled:
        from sift.canonical_dataset import load_canonical_data
        selection = {"worksheet": sheet if sheet is not None else 0} \
            if path.suffix.casefold() in {".xlsx", ".xls", ".ods"} else {}
        return load_canonical_data(
            session_root or path.parent, path, selection=selection,
        ), False

    # Sampled paths, per format.
    if suffix in (".csv", ".tsv"):
        from sift.schema import _csv_has_header, text_table_params
        enc, sep, dec = text_table_params(path, suffix)
        header = 0 if _csv_has_header(path, sep) else None
        df = pd.read_csv(path, sep=sep, encoding=enc, decimal=dec,
                         nrows=_SAMPLE_ROWS, header=header,
                         low_memory=False)
        return df, True
    if suffix == ".parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=_SAMPLE_ROWS):
            return batch.to_pandas(), True
        return pf.schema_arrow.empty_table().to_pandas(), True
    if suffix in (".feather", ".arrow", ".ipc"):
        from sift.schema import _arrow_batches, _open_arrow_ipc

        with _open_arrow_ipc(path) as (reader, streaming):
            first = next(_arrow_batches(reader, streaming), None)
            if first is not None:
                return first.slice(0, _SAMPLE_ROWS).to_pandas(), True
            return reader.schema.empty_table().to_pandas(), True
    if suffix == ".orc":
        import pyarrow.orc as orc

        reader = orc.ORCFile(str(path))
        if reader.nstripes:
            return reader.read_stripe(0).slice(0, _SAMPLE_ROWS).to_pandas(), True
        return reader.schema.empty_table().to_pandas(), True
    if suffix == ".dta":
        import pyreadstat
        df, _meta = pyreadstat.read_dta(str(path), row_limit=_SAMPLE_ROWS)
        return df, True
    if suffix in (".jsonl", ".ndjson"):
        df = pd.read_json(path, lines=True, nrows=_SAMPLE_ROWS)
        return df, True
    if suffix == ".json":
        # Standard JSON has no general bounded-row reader. The caller only
        # reaches this branch when the full-load ceiling requested sampling.
        raise _NoSampledPathError(path.suffix)
    if suffix in (".sav", ".zsav", ".sas7bdat", ".xpt"):
        import pyreadstat
        readstat_reader: Any = {
            ".sav": pyreadstat.read_sav,
            ".zsav": pyreadstat.read_sav,
            ".sas7bdat": pyreadstat.read_sas7bdat,
            ".xpt": pyreadstat.read_xport,
        }[suffix]
        df, _meta = readstat_reader(str(path), row_limit=_SAMPLE_ROWS)
        return df, True
    if suffix in (".xlsx", ".xls", ".ods"):
        df = pd.read_excel(
            path, sheet_name=(0 if sheet is None else sheet),
            nrows=_SAMPLE_ROWS,
        )
        return df, True

    # ``.rds`` has no partial reader (pyreadr loads whole objects), so
    # an over-ceiling .rds cannot be sampled. Signal that specifically
    # — the caller turns it into an honest "too large to profile"
    # message rather than the misleading "no tabular data found".
    raise _NoSampledPathError(path.suffix)


# --- PII / PHI detection: local-only, name + value-pattern heuristics ---
#
# Same posture as every other detector in this module (design role,
# sentinel-missing, outliers): computed locally, reported to the
# researcher as a flag on the profile panel, and NEVER forwarded to
# the model. This is deliberately the same file that already carries
# the "safe to be disclosive because it's local-only" framing in the
# module docstring — a column flagged here is exactly the kind of
# thing that framing exists to protect.
#
# Two independent signals, either of which is enough to flag a
# column:
#   - NAME-based: the column's own name (e.g. "ssn", "patient_mrn")
#     is a strong, fast, zero-cost signal — and importantly, column
#     NAMES already reach the model via ``get_schema`` regardless of
#     this detector, so a name-based hit is flagging something
#     already partially exposed, not something this detector itself
#     exposes for the first time.
#   - VALUE-based: a sample of the column's own values matches a
#     structural pattern (SSN format, email shape, a Luhn-valid
#     digit string in card-number-length range). This is the signal
#     that catches a column named something innocuous
#     ("contact_info", "field_7") that nonetheless holds real PII.
#
# This module only DETECTS and surfaces. Downstream enforcement
# (blocking a flagged column from deeper schema depths, banning it
# from being named in scripts, etc.) is a separate, deliberately
# distinct concern — see the per-dataset privacy profile and policy
# engine work this detector feeds.

# Column-NAME token → category. Matched against underscore/hyphen/
# camelCase-split tokens from the column name, not substring search,
# so a column called ``phoneme_rate`` (linguistics) doesn't match
# "phone" and a column called ``id`` alone doesn't match every
# *_id foreign-key column in a normal relational extract (see the
# whole-token / suffix-token matching in ``_detect_pii_phi_by_name``).
_PII_NAME_CATEGORIES: tuple[tuple[str, frozenset[str]], ...] = (
    ("Social Security / national ID number", frozenset({
        "ssn", "ssns", "sin", "nino", "nationalid", "national",
        "socialsecurity", "socialsecuritynumber", "socialsecurityno",
    })),
    ("passport number", frozenset({"passport", "passportno", "passportnumber"})),
    ("driver's license number", frozenset({
        "driverslicense", "driverlicense", "dlnumber", "dlno",
    })),
    ("email address", frozenset({"email", "emailaddress", "e-mail"})),
    ("phone number", frozenset({
        "phone", "phonenumber", "telephone", "tel", "mobile", "cell",
        "fax",
    })),
    ("street address", frozenset({
        "address", "streetaddress", "homeaddress", "mailingaddress",
    })),
    ("credit/debit card number", frozenset({
        "creditcard", "cardnumber", "cardno", "ccnumber", "cvv", "cvc",
        "cardholder", "creditcardnumber", "creditcardno", "debitcard",
        "debitcardnumber",
    })),
    ("bank account / routing number", frozenset({
        "bankaccount", "accountnumber", "routingnumber", "iban", "swift",
    })),
    ("date of birth", frozenset({
        "dob", "dateofbirth", "birthdate", "birthday",
    })),
    ("medical record number", frozenset({
        "mrn", "medicalrecord", "medicalrecordnumber", "patientid",
        "chartnumber",
    })),
    ("health plan / insurance ID", frozenset({
        "healthplan", "insuranceid", "policynumber", "memberid",
    })),
    ("diagnosis / clinical code", frozenset({
        "diagnosis", "icd9", "icd10", "icd", "diagnosiscode",
    })),
    ("full legal name", frozenset({
        "fullname", "patientname", "legalname",
    })),
    ("biometric identifier", frozenset({
        "fingerprint", "biometric", "faceid", "retina", "iris",
    })),
)


def _name_tokens(name: str) -> frozenset[str]:
    """Normalize a column name to a token set for the name-based
    detectors: lowercase, split on ``_``/``-``/whitespace, AND also
    offer the fully-glued form (``date_of_birth`` -> also
    ``dateofbirth``) since real-world column names mix conventions
    (``dob``, ``DOB``, ``date_of_birth``, ``DateOfBirth`` all appear
    in the wild for the same concept)."""
    low = str(name).strip().lower()
    glued = re.sub(r"[^a-z0-9]", "", low)
    parts = frozenset(p for p in re.split(r"[^a-z0-9]+", low) if p)
    return parts | {glued}


def _detect_pii_phi_by_name(name: str) -> str | None:
    """Return a PII/PHI category if the column NAME matches a known
    pattern, or None. Whole-token or whole-glued-name match only —
    substring matching would flag ``phoneme_rate`` for "phone" and
    ``embed`` for nothing (fine) but ``card_id`` for "card" when it's
    an unrelated foreign key. Precision over recall: a missed PII
    column is a gap to close later; a wrongly-flagged ordinary column
    trains researchers to ignore the flag."""
    tokens = _name_tokens(name)
    for category, patterns in _PII_NAME_CATEGORIES:
        if tokens & patterns:
            return category
    return None


# Value-pattern detectors: (category, compiled regex, optional extra
# validator). The regex runs against each sampled value's string form
# (already stripped); the validator, if present, gets the matched
# string and must also return True for the value to count as a hit.
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[a-zA-Z]{2,}$")
_SSN_RE = re.compile(r"^\d{3}-\d{2}-\d{4}$")
_PHONE_RE = re.compile(
    r"^\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}$"
)
_IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
# Digits only, after stripping common card separators — length range
# covers every major issuer (Visa/Mastercard 16, Amex 15, some debit
# 13-19). The Luhn check below is what actually distinguishes a real
# card number from an arbitrary same-length integer.
_CARD_DIGITS_RE = re.compile(r"^\d{13,19}$")


def _luhn_valid(digits: str) -> bool:
    """Standard Luhn checksum. The discriminating signal for the
    credit-card value detector: an arbitrary 13-19 digit integer
    (a large ID, a phone number with country code, a barcode) passes
    this by chance only about 1 time in 10, so requiring it on top of
    the length match is what keeps the detector from flagging every
    long numeric ID column in a dataset as a card number."""
    total = 0
    reverse_digits = digits[::-1]
    for i, ch in enumerate(reverse_digits):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_value_hit(s: str) -> bool:
    stripped = re.sub(r"[\s-]", "", s)
    if not _CARD_DIGITS_RE.match(stripped):
        return False
    return _luhn_valid(stripped)


_PII_VALUE_PATTERNS: tuple[tuple[str, "re.Pattern[str] | None", Any], ...] = (
    ("Social Security number", _SSN_RE, None),
    ("email address", _EMAIL_RE, None),
    ("phone number", _PHONE_RE, None),
    ("credit/debit card number", None, _card_value_hit),
    ("IP address", _IPV4_RE, None),
)

# How many non-null values to sample per column for the value-pattern
# check (cheap — a plain string compare/regex per value, not a
# dataframe-wide operation) and how high the match rate must be
# before flagging. High threshold (not "any match") because a
# free-text notes column can easily contain ONE email-shaped
# substring without being an email column.
_PII_VALUE_SAMPLE_SIZE = 200
_PII_VALUE_MATCH_THRESHOLD = 0.8


def _detect_pii_phi_by_values(col: Any, non_null: int) -> str | None:
    """Sample the column's own values and check them against known
    PII/PHI value shapes. Returns the first category whose match rate
    over the sample clears the threshold, or None."""
    if non_null < 5:
        return None
    try:
        sample = col.dropna().astype(str).head(_PII_VALUE_SAMPLE_SIZE)
    except Exception:  # noqa: BLE001 — exotic dtypes, bail out quietly
        return None
    if len(sample) == 0:
        return None

    for category, pattern, validator in _PII_VALUE_PATTERNS:
        hits = 0
        for raw in sample:
            s = raw.strip()
            if not s:
                continue
            if pattern is not None:
                if pattern.match(s):
                    hits += 1
            elif validator is not None and validator(s):
                hits += 1
        if hits / len(sample) >= _PII_VALUE_MATCH_THRESHOLD:
            return category
    return None


def _detect_pii_phi(name: str, col: Any, non_null: int) -> dict[str, Any] | None:
    """Combine the name- and value-based signals for one column.

    Returns ``{"category": ..., "basis": [...]}`` where ``basis`` is
    one or both of ``"name"`` / ``"value_pattern"`` — surfacing WHY
    something was flagged matters for a researcher deciding whether
    to trust the flag (a value-pattern hit is stronger evidence than
    a name hit alone)."""
    name_hit = _detect_pii_phi_by_name(name)
    value_hit = _detect_pii_phi_by_values(col, non_null)
    if name_hit is None and value_hit is None:
        return None
    basis = []
    if name_hit is not None:
        basis.append("name")
    if value_hit is not None:
        basis.append("value_pattern")
    # Prefer the value-pattern category when both fire and disagree
    # (rare, but a column named "notes" that happens to be full of
    # SSNs should report "SSN", not whatever the name detector — which
    # didn't fire here — would have said); otherwise whichever fired.
    category = value_hit if value_hit is not None else name_hit
    return {"category": category, "basis": basis}


def _profile_columns(df: Any, sampled: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-variable profile plus whole-dataset structural flags."""
    import pandas as pd

    n_rows = int(len(df))
    n_cols = int(df.shape[1])
    variables: list[dict[str, Any]] = []
    likely_ids: list[str] = []
    constants: list[str] = []
    all_missing: list[str] = []
    design_columns: list[dict[str, str]] = []
    pii_phi_columns: list[dict[str, Any]] = []
    target_candidates: list[dict[str, Any]] = []

    for position, raw_name in enumerate(df.columns):
        # Column names are data-origin text and land in the DOM, so
        # they pass the same text-safety treatment as everywhere else.
        name = safe_key(str(raw_name))
        col = df[raw_name]
        try:
            missing = int(col.isna().sum())
        except Exception:  # noqa: BLE001
            missing = 0
        non_null = n_rows - missing

        entry: dict[str, Any] = {
            "name": name,
            "dtype": _friendly_dtype(col),
            "missing": missing,
            "missing_pct": round(100.0 * missing / n_rows, 1) if n_rows else 0.0,
        }

        try:
            distinct = int(col.nunique(dropna=True))
            entry["distinct"] = distinct
        except Exception:  # noqa: BLE001 — unhashable / exotic dtypes
            distinct = None

        if pd.api.types.is_numeric_dtype(col) and non_null:
            try:
                entry["min"] = _clean_number(col.min())
                entry["max"] = _clean_number(col.max())
                entry["mean"] = _clean_number(col.mean())
            except Exception:  # noqa: BLE001
                pass
            sentinel = _detect_sentinel_missing(col, non_null)
            if sentinel is not None:
                entry["possible_missing_code"] = sentinel
            outliers = _detect_outliers(col, non_null)
            if outliers is not None:
                entry["outliers"] = {
                    "count": outliers[0], "share": outliers[1],
                }
        design_role = _detect_design_role(name, col, non_null)
        if design_role is not None:
            entry["survey_design_role"] = design_role
            design_columns.append({"name": name, "role": design_role})
        pii_phi = _detect_pii_phi(name, col, non_null)
        if pii_phi is not None:
            entry["pii_phi"] = pii_phi
            pii_phi_columns.append({
                "name": name, "category": pii_phi["category"],
                "basis": pii_phi["basis"],
            })
        bad_dates = _detect_impossible_dates(name, col, non_null)
        if bad_dates is not None:
            entry["impossible_dates"] = bad_dates
        imbalance = _detect_imbalance(col, non_null)
        if imbalance is not None:
            entry["imbalance"] = imbalance

        if non_null == 0 and n_rows:
            all_missing.append(name)
            entry["flag"] = "all missing"
        elif distinct == 1:
            constants.append(name)
            entry["flag"] = "constant"
        elif (distinct is not None and non_null >= _ID_MIN_ROWS
              and distinct == non_null and not sampled):
            # On a sample, "unique within the sample" is weak evidence,
            # so the identifier flag is withheld rather than guessed.
            likely_ids.append(name)
            entry["flag"] = "likely identifier"

        semantic_type = _infer_semantic_type(name, col, entry, non_null, distinct)
        entry["semantic_type"] = semantic_type

        target_score = _score_target_candidate(
            name, entry, semantic_type, position, n_cols,
        )
        if target_score is not None:
            score, reasons = target_score
            target_candidates.append({
                "name": name, "score": round(score, 2), "reasons": reasons,
            })

        variables.append(entry)

    target_candidates.sort(key=lambda c: c["score"], reverse=True)

    return variables, {
        "likely_identifiers": likely_ids,
        "constant_columns": constants,
        "all_missing_columns": all_missing,
        "survey_design_columns": design_columns,
        "pii_phi_columns": pii_phi_columns,
        # Capped and pre-sorted: the panel wants "what should I look at
        # first", not an exhaustive per-column dump — and a low-scoring
        # tail would dilute the signal for the columns that actually
        # matter here.
        "likely_target_candidates": target_candidates[:_MAX_TARGET_CANDIDATES],
    }


# Values survey and statistical packages conventionally use to encode
# "missing" as a number. Reported as a *question*, never auto-recoded:
# -99 can be a real temperature and 999 a real millisecond. The flag
# fires only when the value sits at the column's own extreme AND
# accounts for a non-trivial share of observations — the signature of
# a code, not a measurement.
_SENTINEL_CANDIDATES = (-9999, -999, -99, -9, -1, 99, 999, 9999)
_SENTINEL_MIN_SHARE = 0.01


# Complex-survey design columns. An entire methodology — national
# health and social surveys (NHANES, BRFSS, ESS, PISA, LFS, HRS and
# their equivalents worldwide) — is drawn with unequal probabilities,
# clustering and stratification. Analysed as if it were a simple
# random sample, the point estimates are biased and the standard
# errors are badly wrong, and nothing in the output looks unusual.
#
# Detection is name-based and reported as a question, never acted on:
# a column called ``weight`` in a clinical dataset is body weight, not
# a sampling weight, which is why the numeric-plausibility test below
# matters and why the label says "looks like".
# Tokens that mark a column as a measured quantity rather than a
# survey-design variable, whatever else its name contains.
_MEASUREMENT_TOKENS = frozenset({
    "body", "birth", "kg", "kgs", "lb", "lbs", "gram", "grams", "g",
    "pound", "pounds", "net", "gross", "curb", "gain", "loss",
    "baseline", "height", "bmi", "mass",
})

_DESIGN_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sampling weight", (
        "wt", "wgt", "weight", "pweight", "pwgt", "finalwt", "finalweight",
        "sampwt", "sampweight", "svywt", "wtint", "wtmec", "perwt",
        "hhwt", "raking", "postwt", "analysiswt", "aweight", "fweight",
    )),
    ("stratum", ("strata", "stratum", "sdmvstra", "vstrat", "str")),
    ("primary sampling unit", (
        "psu", "cluster", "sdmvpsu", "vpsu", "primary_sampling_unit",
    )),
    ("finite population correction", ("fpc",)),
    ("replicate weights", ("repwt", "brrwt", "jkwt", "replicate")),
)


def _detect_design_role(name: str, col: Any, non_null: int) -> str | None:
    """Return a survey-design role for a column name, or None."""
    import pandas as pd

    low = name.strip().lower()
    token_list = [p for p in low.replace("-", "_").split("_") if p]
    tokens = set(token_list)
    # A measurement token anywhere means this is a measured quantity,
    # not a design variable: ``body_weight_kg`` and ``birth_weight``
    # are the classic false positives, and telling a clinical
    # researcher their outcome variable is a sampling weight would be
    # exactly the kind of confident-but-wrong output this layer exists
    # to avoid.
    if tokens & _MEASUREMENT_TOKENS:
        return None
    for role, patterns in _DESIGN_PATTERNS:
        pattern_set = set(patterns)
        # Whole-name match, or the design token in first/last position.
        # A design token buried mid-name (``body_weight_kg``) is
        # describing something else.
        hit = low in pattern_set
        if not hit and token_list:
            hit = (token_list[0] in pattern_set
                   or token_list[-1] in pattern_set)
        if not hit:
            # Glued forms: ``wtmec2yr``, ``repwt17``, ``finalwt``.
            # Patterns of length <= 3 (``wt``, ``str``, ``psu``,
            # ``fpc``) are EXCLUDED from this loose substring check --
            # a 2-3 character fragment is too likely to appear inside
            # an unrelated real word for startswith/endswith alone to
            # be trustworthy (``stress``, ``strength``, ``street``,
            # ``stroke``, ``streak``, ``strike``, ``structure`` all
            # start with ``str``; found by fuzzing, not review). Those
            # short patterns are still fully eligible via the
            # whole-name and first/last-TOKEN checks above, which
            # respect ``_``/``-`` word boundaries and don't have this
            # false-positive mode -- only the boundary-blind substring
            # match is restricted to longer, more distinctive patterns
            # (``wtmec``, ``sdmvstra``, ``repwt``, ...) where an
            # accidental collision with an unrelated English word is
            # implausible.
            hit = any(
                (low.startswith(pat) or low.endswith(pat))
                and len(low) > len(pat)
                for pat in patterns if len(pat) > 3
            )
        if not hit:
            continue
        # Plausibility gate: design variables are numeric. A string
        # column called "cluster" is a label, not a PSU.
        try:
            if not pd.api.types.is_numeric_dtype(col) or non_null == 0:
                return None
            # Sampling weights are positive by construction.
            if role == "sampling weight" and float(col.min()) <= 0:
                return None
        except Exception:  # noqa: BLE001
            return None
        return role
    return None


def _detect_sentinel_missing(col: Any, non_null: int) -> float | int | None:
    """Return a suspected coded-missing value in ``col``, or None."""
    try:
        lo = col.min()
        hi = col.max()
        for cand in _SENTINEL_CANDIDATES:
            if cand != lo and cand != hi:
                continue
            share = float((col == cand).sum()) / max(1, non_null)
            if share >= _SENTINEL_MIN_SHARE:
                return int(cand)
    except Exception:  # noqa: BLE001 — advisory only
        return None
    return None


# --- Semantic type inference + likely-target detection -------------------
#
# dtype alone (int/float/text/bool/date) says almost nothing about
# what a column MEANS. "numeric, 4 distinct values" is consistent with
# a Likert-scale rating, a count of children, or a region code stored
# as an integer — very different things to feed into a regression.
# This layer adds a best-effort semantic label on top of dtype, and —
# since researchers overwhelmingly open a dataset already knowing
# roughly what they want to explain — a heuristic guess at which
# column(s) look like the outcome they are most likely to model.
#
# Same epistemic posture as every other detector in this module: name-
# and value-based pattern matching, reported as a labelled guess with
# reasons attached (so a researcher can see WHY and override it in a
# glance), never silently acted on, and never fed into the SDC
# boundary — this stays local-only per the module docstring, same as
# the PII/PHI and survey-design detectors above.

_PERCENTAGE_NAME_TOKENS = frozenset({
    "pct", "percent", "percentage", "rate", "share", "proportion",
    "ratio", "fraction",
})
_CURRENCY_NAME_TOKENS = frozenset({
    "price", "cost", "amount", "salary", "income", "revenue", "fee",
    "payment", "wage", "wages", "earnings", "expense", "expenses",
    "spend", "spending", "budget", "fare", "premium", "balance",
    "profit", "loss", "worth",
})
_COUNT_NAME_TOKENS = frozenset({
    "count", "num", "number", "total", "freq", "frequency", "tally",
    "occurrences", "visits", "sessions", "clicks", "orders",
    "quantity", "qty",
})
_GEO_NAME_TOKENS = frozenset({
    "lat", "latitude", "lon", "lng", "longitude", "zip", "zipcode",
    "postalcode", "postal", "country", "countrycode", "state",
    "province", "city", "town", "region", "county", "geo",
    "geolocation", "coordinates",
})
_FREE_TEXT_NAME_TOKENS = frozenset({
    "notes", "note", "comment", "comments", "description", "desc",
    "feedback", "review", "text", "remarks", "summary", "narrative",
    "freetext", "message", "body",
})

# Average sampled string length above which a text column is treated
# as free-form prose rather than a short categorical label, even
# without a name match — "yes"/"no"/"north"/"south" average well
# under this; open-ended survey responses and notes fields do not.
_FREE_TEXT_AVG_LEN = 40
_FREE_TEXT_SAMPLE_SIZE = 200


def _avg_string_length(col: Any, non_null: int) -> float | None:
    if non_null == 0:
        return None
    try:
        sample = col.dropna().astype(str).head(_FREE_TEXT_SAMPLE_SIZE)
        if len(sample) == 0:
            return None
        return float(sum(len(s) for s in sample) / len(sample))
    except Exception:  # noqa: BLE001 — advisory only
        return None


# Bounded sample + parse-success threshold for confirming a text
# column whose name merely *suggests* a date actually contains
# date-shaped values. Same threshold family as
# ``_PII_VALUE_MATCH_THRESHOLD`` — a high bar, because the false-
# positive cost (mislabelling an ordinary text column "date") is
# larger than the false-negative cost (leaving a genuine date column
# unlabelled, where it still gets a reasonable "categorical"/
# "free_text" fallback).
_DATE_TEXT_PARSE_SAMPLE = 200
_DATE_TEXT_PARSE_MIN_RATE = 0.8


def _looks_like_date_text(col: Any, non_null: int) -> bool:
    import warnings

    import pandas as pd

    if non_null < 5:
        return False
    try:
        sample = col.dropna().astype(str).head(_DATE_TEXT_PARSE_SAMPLE)
        if len(sample) == 0:
            return False
        # This is a plausibility probe on a name-matched column, not a
        # promise the column IS dates — a low parse rate (handled via
        # the threshold below) is an expected, not exceptional, outcome
        # for a false-positive name match, so pandas's "guessed a mixed
        # format" advisory is noise here rather than signal.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce", utc=True)
        rate = float(parsed.notna().sum()) / len(sample)
    except Exception:  # noqa: BLE001 — advisory only
        return False
    return rate >= _DATE_TEXT_PARSE_MIN_RATE


def _infer_semantic_type(
    name: str,
    col: Any,
    entry: dict[str, Any],
    non_null: int,
    distinct: int | None,
) -> str:
    """Best-effort semantic label for one column, on top of its dtype.

    Structural facts already computed for this column (constant,
    all-missing, likely-identifier — set as ``entry["flag"]`` just
    before this runs) take precedence over any name/value guess.
    """
    import pandas as pd

    flag = entry.get("flag")
    dtype = entry.get("dtype")
    if flag == "constant":
        return "constant"
    if flag == "all missing":
        return "unknown"

    # dtype-driven signals outrank the "likely identifier" flag: a
    # date/time or boolean column is checked FIRST, because a
    # timestamp column that happens to be all-distinct (an event log
    # with one row per event) is still fundamentally a date, not a
    # meaningless key.
    if dtype == "date/time":
        return "date"
    if dtype == "boolean":
        return "binary"

    tokens = _name_tokens(name)

    # A text/categorical column whose NAME suggests a date (CSV/TSV
    # never auto-parse dates, so this is the common on-disk shape for
    # one) is checked next, ahead of the identifier-flag deferral —
    # otherwise a daily-granularity date column such as
    # ``signup_date`` (every value distinct) would be mislabelled
    # "identifier" instead of "date". Gated on an actual parse of a
    # sample, not the name alone, so generic tokens shared with
    # non-date columns (``start_balance``, ``end_state``) don't
    # false-positive.
    if (dtype in ("text", "categorical") and (tokens & _DATE_NAME_TOKENS)
            and _looks_like_date_text(col, non_null)):
        return "date"

    if flag == "likely identifier":
        # The identifier flag fires whenever every non-null value in
        # the column is distinct. For an integer or text/categorical
        # column that is a strong identifier signal (sequential IDs,
        # UUIDs, order numbers). For a FLOAT column it usually is not:
        # continuous measurements (income, price, a sensor reading)
        # are routinely all-distinct in any real sample simply because
        # floating-point values rarely collide — that is a statistical
        # artifact of continuous measurement, not a designed key, so
        # "identifier" would be the wrong semantic label for what is
        # actually a legitimate continuous variable. Only defer to the
        # identifier flag for dtypes where uniqueness is a real key
        # signal; float columns fall through to the numeric branch
        # below and land on "continuous".
        if dtype in ("integer", "text", "categorical"):
            return "identifier"

    if pd.api.types.is_numeric_dtype(col) and non_null:
        if distinct == 2:
            return "binary"
        if tokens & _PERCENTAGE_NAME_TOKENS:
            try:
                lo, hi = float(col.min()), float(col.max())
                if lo >= -0.01 and hi <= 100.01:
                    return "percentage"
            except Exception:  # noqa: BLE001
                pass
        if tokens & _CURRENCY_NAME_TOKENS:
            return "currency"
        if tokens & _GEO_NAME_TOKENS:
            return "geographic"
        if tokens & _COUNT_NAME_TOKENS and pd.api.types.is_integer_dtype(col):
            try:
                if float(col.min()) >= 0:
                    return "count"
            except Exception:  # noqa: BLE001
                pass
        if dtype == "integer" and distinct is not None and non_null:
            if 2 < distinct <= 15 and distinct / non_null < 0.2:
                return "ordinal"
        return "continuous"

    if dtype == "categorical":
        return "categorical"

    # Text dtype. Date-name-but-text-dtype is left to the dedicated
    # impossible-dates detector rather than guessed here.
    if tokens & _GEO_NAME_TOKENS:
        return "geographic"
    if tokens & _FREE_TEXT_NAME_TOKENS:
        return "free_text"
    if distinct is not None and non_null:
        avg_len = _avg_string_length(col, non_null)
        if avg_len is not None and avg_len >= _FREE_TEXT_AVG_LEN:
            return "free_text"
        if distinct == 2:
            return "binary"
        if distinct / non_null < 0.5:
            return "categorical"
    return "free_text"


# Column-NAME tokens that strongly suggest "this is the thing being
# predicted" — the vocabulary researchers and ML practitioners
# actually use for a dependent/response/label variable.
_TARGET_NAME_TOKENS = frozenset({
    "target", "label", "labels", "outcome", "response", "y",
    "dependent", "dv", "churn", "churned", "default", "defaulted",
    "conversion", "converted", "survived", "class", "diagnosis",
    "readmitted", "readmission", "purchased", "clicked", "fraud",
    "approved", "success", "failed", "failure", "result",
})

# Semantic types that are structurally implausible as a modelling
# target, regardless of name match — an identifier or a constant can
# be NAMED "target" in the wild and still be useless as one.
_TARGET_IMPLAUSIBLE_TYPES = frozenset({
    "identifier", "constant", "unknown", "free_text", "geographic",
})

# How many candidates the panel surfaces — "what should I look at
# first", not an exhaustive per-column dump.
_MAX_TARGET_CANDIDATES = 3


def _score_target_candidate(
    name: str,
    entry: dict[str, Any],
    semantic_type: str,
    position: int,
    n_cols: int,
) -> tuple[float, list[str]] | None:
    """Heuristic plausibility score for "is this column the outcome a
    researcher is likely to model?", or None if structurally
    implausible. Scores are only ever compared to each other within
    one dataset — there is no fixed threshold meant to mean something
    on its own."""
    if semantic_type in _TARGET_IMPLAUSIBLE_TYPES:
        return None
    if entry.get("pii_phi") is not None:
        return None
    if entry.get("survey_design_role") is not None:
        return None

    score = 0.0
    reasons: list[str] = []

    tokens = _name_tokens(name)
    if tokens & _TARGET_NAME_TOKENS:
        score += 3.0
        reasons.append(
            "column name matches common outcome-variable vocabulary")

    if semantic_type == "binary":
        score += 1.5
        reasons.append(
            "binary — a common outcome shape (e.g. event / no event)")
    elif semantic_type in ("continuous", "count", "percentage", "currency"):
        score += 0.5
        reasons.append(f"{semantic_type} — a plausible outcome shape")
    elif semantic_type == "ordinal":
        score += 0.75
        reasons.append(
            "ordinal — a plausible outcome shape (e.g. a rating scale)")

    # Positional convention: many tabular ML datasets place the target
    # last. Weak on its own — worth a small nudge, not a claim.
    if n_cols > 1 and position == n_cols - 1:
        score += 0.5
        reasons.append(
            "last column in the file — a common convention for the target")

    imbalance = entry.get("imbalance")
    if imbalance is not None and semantic_type == "binary":
        # A heavily imbalanced binary column is *more* likely to be a
        # rare-event outcome (churn, fraud, default) than an ordinary
        # covariate — most covariates aren't this lopsided by chance.
        score += 0.5
        reasons.append(
            "heavily imbalanced binary column — typical of a "
            "rare-event outcome")

    if score <= 0:
        return None
    return score, reasons


# --- Dataset Health: deterministic, locally-computed issue detectors ---
#
# Same posture as the design-role and sentinel detectors above: every
# flag here is a computed fact about the sampled/loaded rows, reported
# as a count or share, never a claim about the whole file when only a
# sample was read, and never silently acted on. This stays local-only
# (see module docstring) — none of it is disclosive since it never
# reaches the model, so it can be as specific as it needs to be to be
# useful to the researcher looking at their own data.

# Minimum non-null observations before an outlier / imbalance /
# date-range check runs at all. Below this, "3 of 8 values look
# extreme" is noise, not a finding — same threshold family as
# ``_ID_MIN_ROWS``.
_HEALTH_MIN_ROWS = 20

# A numeric value farther than this many IQRs from the nearest quartile
# is "extreme" for health-panel purposes. 3x (Tukey's "far out" fence)
# rather than the more common 1.5x, because 1.5x flags a meaningful
# share of ordinary skewed data (income, response times, counts) as
# "extreme" and would make the health score noisy on perfectly normal
# datasets. The panel is meant to catch genuinely unusual values, not
# ordinary right-skew.
_OUTLIER_IQR_MULTIPLIER = 3.0
# A column is only flagged if the extreme share clears this floor —
# a single far-out point in a 10,000-row column is not worth a
# researcher's attention.
_OUTLIER_MIN_SHARE = 0.005

# Column-name tokens suggesting a date/time column worth range-
# checking even when pandas has left it as text (the common case for
# CSV/TSV, which never auto-parse dates). Deliberately narrower than a
# general date-string sniffer — false positives here would silently
# mis-parse an unrelated text column and report bogus "impossible
# dates".
_DATE_NAME_TOKENS = frozenset({
    "date", "dt", "time", "timestamp", "day", "month", "year",
    "created", "updated", "signup", "birth", "dob", "start", "end",
})
# A parsed date before this is treated as implausible for the kind of
# operational/survey data Sift targets (pre-1900 dates are almost
# always a parsing artifact or a sentinel value, not a real event).
_DATE_MIN_YEAR = 1900


def _detect_outliers(col: Any, non_null: int) -> tuple[int, float] | None:
    """IQR-fence outlier count for a numeric column.

    Returns ``(count, share)`` or ``None`` when the column is too
    small to judge, has no spread (IQR == 0, e.g. near-constant), or
    the quartiles can't be computed (exotic dtype).
    """
    if non_null < _HEALTH_MIN_ROWS:
        return None
    try:
        q1 = float(col.quantile(0.25))
        q3 = float(col.quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            return None
        lo = q1 - _OUTLIER_IQR_MULTIPLIER * iqr
        hi = q3 + _OUTLIER_IQR_MULTIPLIER * iqr
        count = int(((col < lo) | (col > hi)).sum())
    except Exception:  # noqa: BLE001 — advisory only
        return None
    if count == 0:
        return None
    share = count / non_null
    if share < _OUTLIER_MIN_SHARE:
        return None
    return count, round(share, 4)


def _detect_impossible_dates(
    name: str, col: Any, non_null: int,
) -> dict[str, Any] | None:
    """Count values that parse as dates but fall outside a plausible
    range (before ``_DATE_MIN_YEAR`` or after today).

    Only attempts a parse on columns whose name suggests a date AND
    that parse mostly-successfully as dates — this is a range check,
    not a date-format sniffer, so a column that doesn't clearly parse
    as dates is left alone rather than guessed at.
    """
    import pandas as pd

    if non_null < _HEALTH_MIN_ROWS:
        return None
    low = name.strip().lower()
    tokens = set(t for t in low.replace("-", "_").split("_") if t)
    if not (tokens & _DATE_NAME_TOKENS):
        return None
    if pd.api.types.is_numeric_dtype(col):
        # Bare numeric columns (an "id" containing "day", a "year"
        # integer column meant as a count) are not date strings —
        # parsing them as dates would fabricate a finding.
        return None
    try:
        # This detector intentionally accepts real-world mixed date
        # formats. State that contract explicitly so pandas neither emits
        # its per-value fallback warning nor repeats format inference.
        parsed = pd.to_datetime(
            col, errors="coerce", utc=True, format="mixed",
        )
        parsed_ok = parsed.notna()
        parsed_count = int(parsed_ok.sum())
        if parsed_count < max(_HEALTH_MIN_ROWS, int(0.5 * non_null)):
            # Fewer than half the non-null values parse as dates —
            # this isn't reliably a date column.
            return None
        now = pd.Timestamp.now(tz="UTC")
        min_bound = pd.Timestamp(year=_DATE_MIN_YEAR, month=1, day=1, tz="UTC")
        bad = parsed[parsed_ok & ((parsed < min_bound) | (parsed > now))]
        count = int(len(bad))
    except Exception:  # noqa: BLE001 — advisory only
        return None
    if count == 0:
        return None
    return {
        "count": count,
        "parsed": parsed_count,
        "earliest_bad": str(bad.min().date()) if count else None,
        "latest_bad": str(bad.max().date()) if count else None,
    }


# A column this lopsided is flagged as imbalanced. Not inherently a
# problem (rare-event outcomes are often exactly this shape) — framed
# to the researcher as a fact to be aware of, never as an error.
_IMBALANCE_SHARE = 0.95


def _detect_imbalance(col: Any, non_null: int) -> dict[str, Any] | None:
    """Top-category share for a low-cardinality column.

    Only evaluated on columns with 2–20 distinct values (the
    categorical/binary range) — continuous numeric columns are
    expected to have a "most common value" that means nothing.
    """
    if non_null < _HEALTH_MIN_ROWS:
        return None
    try:
        counts = col.value_counts(dropna=True)
        distinct = int(len(counts))
        if distinct < 2 or distinct > 20:
            return None
        top_value = counts.index[0]
        top_count = int(counts.iloc[0])
        share = top_count / non_null
    except Exception:  # noqa: BLE001 — advisory only
        return None
    if share < _IMBALANCE_SHARE:
        return None
    return {
        "top_value": str(top_value),
        "share": round(share, 4),
        "distinct": distinct,
    }


def _compute_health(
    profile: dict[str, Any],
    variables: list[dict[str, Any]],
    flags: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic health score + issue list from already-computed
    profile facts. Every deduction traces to a specific computed
    number; nothing here is a model judgment and nothing is deducted
    for a check that wasn't actually run (e.g. duplicate count is
    ``None`` on a sampled profile, so it contributes no penalty rather
    than being assumed clean).
    """
    issues: list[dict[str, Any]] = []
    score = 100.0

    missing_pct = profile.get("missing_pct")
    if isinstance(missing_pct, (int, float)) and missing_pct > 0:
        score -= min(20.0, missing_pct * 0.5)
        if missing_pct >= 5:
            issues.append({
                "severity": "warn" if missing_pct >= 20 else "info",
                "message": f"{missing_pct:.1f}% of cells are missing "
                           f"across the dataset.",
                "columns": [],
            })

    dup = profile.get("duplicate_rows")
    rows_profiled = profile.get("rows_profiled") or 0
    if isinstance(dup, int) and dup > 0 and rows_profiled:
        dup_pct = 100.0 * dup / rows_profiled
        score -= min(15.0, dup_pct)
        issues.append({
            "severity": "warn" if dup_pct >= 1 else "info",
            "message": f"{dup:,} duplicate rows "
                       f"({dup_pct:.1f}% of rows profiled).",
            "columns": [],
        })

    constants = flags.get("constant_columns") or []
    if constants:
        score -= min(10.0, 3.0 * len(constants))
        issues.append({
            "severity": "info",
            "message": f"{len(constants)} column(s) have a single "
                       f"constant value across every row.",
            "columns": constants,
        })

    all_missing = flags.get("all_missing_columns") or []
    if all_missing:
        score -= min(10.0, 5.0 * len(all_missing))
        issues.append({
            "severity": "warn",
            "message": f"{len(all_missing)} column(s) are entirely "
                       f"missing.",
            "columns": all_missing,
        })

    outlier_cols = [v["name"] for v in variables if v.get("outliers")]
    if outlier_cols:
        score -= min(15.0, 3.0 * len(outlier_cols))
        issues.append({
            "severity": "info",
            "message": f"{len(outlier_cols)} numeric column(s) have a "
                       f"notable share of extreme values (beyond "
                       f"{_OUTLIER_IQR_MULTIPLIER:.0f}x the "
                       f"interquartile range).",
            "columns": outlier_cols,
        })

    bad_date_cols = [v["name"] for v in variables if v.get("impossible_dates")]
    if bad_date_cols:
        score -= min(15.0, 5.0 * len(bad_date_cols))
        issues.append({
            "severity": "warn",
            "message": f"{len(bad_date_cols)} date-like column(s) "
                       f"contain values before {_DATE_MIN_YEAR} or in "
                       f"the future.",
            "columns": bad_date_cols,
        })

    imbalanced_cols = [v["name"] for v in variables if v.get("imbalance")]
    if imbalanced_cols:
        score -= min(10.0, 2.0 * len(imbalanced_cols))
        issues.append({
            "severity": "info",
            "message": f"{len(imbalanced_cols)} column(s) are heavily "
                       f"skewed toward one value "
                       f"(\u2265{_IMBALANCE_SHARE * 100:.0f}%).",
            "columns": imbalanced_cols,
        })

    # PII/PHI is a "warn" regardless of column count (unlike the
    # count-scaled deductions above) — a single Social Security
    # column is exactly as reportable a finding as five of them, and
    # scaling the score down further for "more of them" would imply
    # one flagged column is only a minor issue, which understates it.
    pii_phi_cols = flags.get("pii_phi_columns") or []
    if pii_phi_cols:
        score -= min(15.0, 5.0 * len(pii_phi_cols))
        categories = sorted({c["category"] for c in pii_phi_cols})
        issues.append({
            "severity": "warn",
            "message": (
                f"{len(pii_phi_cols)} column(s) look like they contain "
                f"personal or health information "
                f"({', '.join(categories)}). This is a local-only "
                f"flag (never sent to the model) — review before "
                f"sharing this dataset, any export built from it, or "
                f"a session recording."
            ),
            "columns": [c["name"] for c in pii_phi_cols],
        })

    severity_rank = {"warn": 0, "info": 1}
    issues.sort(key=lambda i: severity_rank.get(i["severity"], 2))

    return {
        "score": max(0, round(score)),
        "issues": issues,
    }


def _friendly_dtype(col: Any) -> str:
    import pandas as pd

    if pd.api.types.is_bool_dtype(col):
        return "boolean"
    if pd.api.types.is_integer_dtype(col):
        return "integer"
    if pd.api.types.is_float_dtype(col):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(col):
        return "date/time"
    if isinstance(col.dtype, pd.CategoricalDtype):
        return "categorical"
    return "text"


def _clean_number(x: Any) -> Any:
    """JSON-safe scalar: NaN / Inf become None, numpy types become Python."""
    try:
        val = float(x)
    except (TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):
        return None
    return round(val, 6) if val != int(val) else int(val)
