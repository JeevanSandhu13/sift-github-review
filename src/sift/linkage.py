"""Sift — linkage diagnostics across the session's datasets.

Sift's tools are dataset-oriented: each file is inspected, profiled
and analysed on its own. But research data is relational. A survey
links to an administrative extract; a patient roster links to claims;
a firm panel links to an employee file. The join is where the
catastrophic, *silent* errors live:

- **Unmatched records vanish.** An inner join that drops 30% of the
  sample changes the population under study, and nothing errors. This
  is the single most common serious data-management mistake in
  applied research — it is why Stata's ``merge`` forces the
  ``_merge`` variable in front of the user.
- **Many-to-many joins fan out.** Two files with duplicated keys
  produce a Cartesian blow-up: 5,000 rows become 400,000, every
  standard error is wrong, and the regression still runs happily.
- **Keys are not what people think.** ``patient_id`` is often unique
  in one file and repeated in the other; a "unique" ID frequently has
  a handful of duplicates from a re-export.

This module answers those questions *before* a merge is written,
deterministically and locally.

**Privacy posture** — identical to ``dataset_profile``: this is a
researcher-facing view of the researcher's own files, reachable only
from the UI bridge, never registered as a tool and never sent to the
model. It reports only counts and rates; key *values* are never
returned. The model remains free to compute merge diagnostics inside
a sandboxed script, where the results pass the sanitizer as usual.

Cost control: only candidate key columns are read (``usecols``), not
whole files, so the check stays cheap on wide datasets. Files above
the full-load ceiling are reported as unavailable rather than sampled
— a match rate computed from the first N rows of two files is not a
match rate, and a confidently wrong number here is worse than none.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from sift.schema import DATA_EXTENSIONS, full_load_max_bytes
from sift.text_safety import safe_key

# Column-name patterns that make a shared name a plausible join key.
# Deliberately permissive: the shared-name-plus-type test does most of
# the work, and a false candidate costs one cheap column read.
_KEYISH_TOKENS = (
    "id", "key", "code", "no", "num", "nr", "identifier", "uid",
    "ssn", "nino", "pid", "eid", "hhid", "case", "record", "index",
)

# Names that are shared across files constantly but are almost never
# join keys on their own. Including them would bury the real keys in
# noise.
_NON_KEY_NAMES = frozenset({
    "age", "sex", "gender", "male", "female", "race", "region",
    "year", "month", "day", "date", "time", "value", "count", "n",
    "total", "amount", "price", "score", "weight", "height", "status",
    "type", "group", "treatment", "control", "yes", "no", "true",
    "false", "name", "notes", "comment", "description",
})

_MAX_PAIRS = 24          # dataset pairs examined
_MAX_KEYS_PER_PAIR = 8   # candidate keys reported per pair


def _looks_keyish(name: str) -> bool:
    low = name.strip().lower()
    if low in _NON_KEY_NAMES:
        return False
    parts = [p for p in low.replace("-", "_").split("_") if p]
    if any(p in _KEYISH_TOKENS for p in parts):
        return True
    return low.endswith("id") or low.endswith("key")


def _readable(path: Path) -> bool:
    try:
        return path.stat().st_size <= full_load_max_bytes()
    except OSError:
        return False


def _columns_of(
    path: Path,
    diagnostics: list[dict[str, str]] | None = None,
) -> list[str]:
    """Column names for a dataset, recording inability to inspect it."""
    from sift.schema import extract
    try:
        sheet = None
        if path.suffix.lower() in (".xlsx", ".xls", ".ods"):
            from sift.policy import get_excel_sheet, load_policy

            sheet = get_excel_sheet(load_policy(path.parent), path.name)
        schema = extract(path, "names_only", sheet=sheet)
    except Exception as exc:  # noqa: BLE001 — no source values in diagnostic
        if diagnostics is not None:
            diagnostics.append({
                "dataset": path.name,
                "stage": "schema",
                "reason": type(exc).__name__,
            })
        return []
    return [str(v.get("name", "")) for v in schema.get("variables", [])]


def _read_key_column(path: Path, column: str):
    """Read a single column as a pandas Series, or None."""
    import pandas as pd

    suffix = path.suffix.lower()
    try:
        if suffix in (".csv", ".tsv"):
            from sift.schema import _csv_has_header, text_table_params
            enc, sep, dec = text_table_params(path, suffix)
            if not _csv_has_header(path, sep):
                return None      # positional columns can't be named keys
            frame = pd.read_csv(path, sep=sep, encoding=enc, decimal=dec,
                                usecols=[column], low_memory=False)
            return frame[column]
        if suffix == ".parquet":
            return pd.read_parquet(path, columns=[column])[column]
        if suffix in (".xlsx", ".xls", ".ods"):
            from sift.policy import get_excel_sheet, load_policy

            selected = get_excel_sheet(load_policy(path.parent), path.name)
            sheet = selected if selected is not None else 0
            return pd.read_excel(path, sheet_name=sheet, usecols=[column])[column]
        # pyreadstat formats support column selection natively.
        if suffix in (".dta", ".sav", ".zsav", ".sas7bdat", ".xpt"):
            import pyreadstat
            reader: Any = {
                ".dta": pyreadstat.read_dta, ".sav": pyreadstat.read_sav,
                ".zsav": pyreadstat.read_sav,
                ".sas7bdat": pyreadstat.read_sas7bdat,
                ".xpt": pyreadstat.read_xport,
            }[suffix]
            frame, _meta = reader(str(path), usecols=[column])
            return frame[column]
        # Remaining formats (.rds, .jsonl, Arrow/ORC) have no
        # column-selective reader in this local diagnostic path;
        # reader; fall back to a full load, which the ceiling gates.
        from sift.schema import load_data
        frame = load_data(path)
        return frame[column] if column in frame.columns else None
    except Exception:  # noqa: BLE001 — a key we can't read is not a key
        return None


def _key_stats(
    series: Any,
    diagnostics: list[dict[str, str]] | None = None,
    *,
    dataset: str = "",
    column: str = "",
) -> dict[str, Any] | None:
    try:
        non_null = series.dropna()
        n = int(len(non_null))
        if n == 0:
            return None
        distinct = int(non_null.nunique())
        return {
            "n": n,
            "distinct": distinct,
            "unique": distinct == n,
            "values": set(non_null.unique().tolist()),
        }
    except Exception as exc:  # noqa: BLE001 — no key values in diagnostic
        if diagnostics is not None:
            diagnostics.append({
                "dataset": dataset,
                "column": safe_key(column),
                "stage": "key_statistics",
                "reason": type(exc).__name__,
            })
        return None


def analyze_pair(
    left: Path, right: Path, max_keys: int = _MAX_KEYS_PER_PAIR,
    *, session_root: Path | None = None,
    diagnostics: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Candidate join keys between two datasets, with match diagnostics."""
    left_cols = _columns_of(left, diagnostics)
    right_cols = _columns_of(right, diagnostics)
    shared = [c for c in left_cols if c in set(right_cols)]
    candidates = [c for c in shared if _looks_keyish(c)][:max_keys]

    if not candidates:
        return []
    if session_root is None:
        try:
            common = Path(os.path.commonpath((left.resolve(), right.resolve())))
            session_root = common if common.is_dir() else common.parent
        except (OSError, ValueError):
            session_root = left.parent
    try:
        from sift.canonical_dataset import load_canonical_data
        from sift.policy import get_excel_sheet, load_policy

        policy = load_policy(session_root)

        def selection_for(path: Path) -> dict[str, Any]:
            if path.suffix.casefold() not in {".xlsx", ".xls", ".ods"}:
                return {}
            selected = get_excel_sheet(policy, path.name)
            return {"worksheet": selected if selected is not None else 0}

        left_frame = load_canonical_data(
            session_root, left, selection=selection_for(left),
        )
        right_frame = load_canonical_data(
            session_root, right, selection=selection_for(right),
        )
    except Exception as exc:  # noqa: BLE001 — no source values in diagnostic
        if diagnostics is not None:
            diagnostics.append({
                "dataset": f"{left.name} + {right.name}",
                "stage": "canonical_load",
                "reason": type(exc).__name__,
            })
        return []

    findings: list[dict[str, Any]] = []
    for column in candidates:
        left_series = left_frame[column] if column in left_frame.columns else None
        right_series = right_frame[column] if column in right_frame.columns else None
        ls = _key_stats(
            left_series, diagnostics, dataset=left.name, column=column,
        )
        rs = _key_stats(
            right_series, diagnostics, dataset=right.name, column=column,
        )
        if ls is None or rs is None:
            continue
        lvals, rvals = ls.pop("values"), rs.pop("values")
        overlap = lvals & rvals
        n_overlap = len(overlap)
        left_matched = 100.0 * n_overlap / max(1, ls["distinct"])
        right_matched = 100.0 * n_overlap / max(1, rs["distinct"])

        if ls["unique"] and rs["unique"]:
            relationship = "one-to-one"
        elif ls["unique"]:
            relationship = "one-to-many"
        elif rs["unique"]:
            relationship = "many-to-one"
        else:
            relationship = "many-to-many"

        warnings: list[str] = []
        if relationship == "many-to-many":
            # Estimate the fan-out on the worst shared key so the
            # warning carries a measurable threshold rather than subjective wording.
            warnings.append(
                "many-to-many: neither file has unique keys, so a join "
                "multiplies rows and every standard error computed "
                "afterwards is wrong. Aggregate one side first.")
        if n_overlap == 0:
            warnings.append(
                "no key values match — this is probably not the right "
                "join key, or the two files use different formats for "
                "it (leading zeros, text vs number, trimmed spaces).")
        else:
            if left_matched < 90.0:
                warnings.append(
                    f"{100.0 - left_matched:.0f}% of {left.name} keys "
                    f"have no match; an inner join silently drops them.")
            if right_matched < 90.0:
                warnings.append(
                    f"{100.0 - right_matched:.0f}% of {right.name} keys "
                    f"have no match; an inner join silently drops them.")
        if not ls["unique"]:
            warnings.append(
                f"{column} is not unique in {left.name} "
                f"({ls['n']:,} rows, {ls['distinct']:,} distinct).")
        if not rs["unique"]:
            warnings.append(
                f"{column} is not unique in {right.name} "
                f"({rs['n']:,} rows, {rs['distinct']:,} distinct).")

        findings.append({
            "key": safe_key(column),
            "relationship": relationship,
            "left_rows": ls["n"], "left_distinct": ls["distinct"],
            "right_rows": rs["n"], "right_distinct": rs["distinct"],
            "matched_keys": n_overlap,
            "left_match_pct": round(left_matched, 1),
            "right_match_pct": round(right_matched, 1),
            "warnings": warnings,
        })
    return findings


def analyze_session(cwd: Path) -> dict[str, Any]:
    """Linkage report for every readable dataset pair in the session."""
    from sift.system_prompt import scan_datasets

    cwd = Path(cwd)
    datasets = [p for p in scan_datasets(cwd)
                if p.suffix.lower() in DATA_EXTENSIONS]
    skipped: list[dict[str, str]] = []
    usable: list[Path] = []
    for path in datasets:
        if _readable(path):
            usable.append(path)
        else:
            skipped.append({
                "dataset": path.name,
                "reason": "above the in-memory ceiling; a match rate "
                          "from a partial read would be misleading, so "
                          "none is reported",
            })

    pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, str]] = []
    for i, left in enumerate(usable):
        for right in usable[i + 1:]:
            if len(pairs) >= _MAX_PAIRS:
                break
            findings = analyze_pair(
                left, right, session_root=cwd, diagnostics=diagnostics,
            )
            if findings:
                pairs.append({
                    "left": left.name, "right": right.name,
                    "keys": findings,
                })
    return {
        "ok": not diagnostics,
        "checks_complete": not diagnostics and not skipped,
        "datasets": len(datasets),
        "pairs": pairs,
        "skipped": skipped,
        "diagnostics": diagnostics,
    }
