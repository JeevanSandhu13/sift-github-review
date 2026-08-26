"""Deterministic, local-only data-quality assessment and corrections.

The quality engine intentionally reports *aggregates*, never observation
values.  It is context aware: checks such as foreign-key integrity, target
leakage, and panel balance are meaningful only after the researcher declares
the relevant columns.  Guesses are labelled with lower confidence and are
never used to mutate data.

Corrections are a separate, explicit operation.  They always create a new
Parquet dataset, re-run the requested findings against the current source,
and record the source fingerprint plus accepted finding IDs in canonical
lineage.  The source file is never opened for writing.
"""

from __future__ import annotations

import math
import hashlib
import json
import os
import tempfile
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_DATE = re.compile(r"(?:^|_)(?:date|time|timestamp|datetime|dt|year|month|day)(?:_|$)", re.I)
_ID = re.compile(r"(?:^|_)(?:id|key|uuid|guid|identifier)(?:_|$)", re.I)
_WEIGHT = re.compile(r"(?:^|_)(?:weight|wt|wgt|pweight|pwgt)(?:_|$)", re.I)
_LAT = re.compile(r"(?:^|_)(?:lat|latitude)(?:_|$)", re.I)
_LON = re.compile(r"(?:^|_)(?:lon|lng|long|longitude)(?:_|$)", re.I)
_SENTINELS = {-9999, -999, -99, -9, 99, 999, 9999}
_MOJIBAKE = re.compile(r"\ufffd|\u00c2|\u00c3|\u00e2\u20ac|\\x[0-9a-fA-F]{2}")
_SEVERITY = {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}


class DataQualityError(ValueError):
    pass


@dataclass(frozen=True)
class _Finding:
    code: str
    severity: str
    confidence: float
    columns: tuple[str, ...]
    summary: str
    evidence: Mapping[str, Any]
    recommendation: str
    correction: Mapping[str, Any] | None = None

    def row(self, ordinal: int) -> dict[str, Any]:
        result: dict[str, Any] = {
            "id": f"DQ-{ordinal:03d}-{self.code}",
            "code": self.code,
            "severity": self.severity,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 2),
            "columns": list(self.columns),
            "summary": self.summary,
            "evidence": dict(self.evidence),
            "recommendation": self.recommendation,
            "requires_approval": self.correction is not None,
        }
        if self.correction is not None:
            result["correction"] = dict(self.correction)
        return result


def _safe_columns(frame: Any, raw: object) -> list[str]:
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, Sequence) else []
    return [str(value) for value in values if str(value) in frame.columns]


def _context(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise DataQualityError("quality context must be an object")
    allowed = {
        "identifiers", "keys", "foreign_keys", "expected_categories", "units",
        "panel_id", "time", "treatment", "target", "features", "split",
        "weights", "latitude", "longitude", "expected_frequency",
    }
    return {str(key): value for key, value in raw.items() if key in allowed}


def _add(rows: list[_Finding], **kwargs: Any) -> None:
    rows.append(_Finding(**kwargs))


def _limited(
    rows: list[dict[str, Any]],
    check: str,
    exc: BaseException,
    *columns: str,
) -> None:
    """Record a non-sensitive check degradation instead of silently passing."""
    item = {
        "check": check,
        "columns": list(columns),
        "reason": type(exc).__name__,
    }
    if item not in rows:
        rows.append(item)


def _numeric_view(series: Any) -> tuple[Any, int, int]:
    import pandas as pd
    non_null = series.dropna()
    converted = pd.to_numeric(non_null, errors="coerce")
    return converted, int(converted.notna().sum()), int(len(non_null))


def _single_table_checks(
    frame: Any,
    context: dict[str, Any],
    sampled: bool,
    limitations: list[dict[str, Any]],
) -> list[_Finding]:
    import pandas as pd

    findings: list[_Finding] = []
    n = int(len(frame))
    if n == 0:
        _add(findings, code="empty_dataset", severity="critical", confidence=1.0,
             columns=(), summary="The dataset has no observations.", evidence={"rows": 0},
             recommendation="Verify the source selection or extraction query.")
        return findings

    # Whole-row duplication is exact and never lists the duplicated values.
    if not sampled:
        try:
            count = int(frame.duplicated().sum())
        except Exception as exc:
            _limited(limitations, "duplicate_rows", exc)
            count = 0
        if count:
            _add(findings, code="duplicate_rows", severity="high", confidence=1.0,
                 columns=(), summary="Exact duplicate observations are present.",
                 evidence={"count": count, "share": round(count / n, 4)},
                 recommendation="Confirm the unit of analysis before removing duplicates.",
                 correction={"operation": "drop_exact_duplicate_rows"})

    declared_keys = _safe_columns(frame, context.get("keys"))
    declared_ids = _safe_columns(frame, context.get("identifiers"))
    guessed_ids = [str(c) for c in frame.columns if _ID.search(str(c))]
    for column in dict.fromkeys(declared_keys + declared_ids + guessed_ids):
        series = frame[column]
        missing = int(series.isna().sum())
        duplicate = int(series.dropna().duplicated().sum())
        if missing or duplicate:
            declared = column in declared_keys or column in declared_ids
            code = "key_uniqueness" if column in declared_keys else "duplicate_identifier"
            _add(findings, code=code, severity="critical" if declared else "high",
                 confidence=1.0 if declared else 0.72, columns=(column,),
                 summary="A declared key is not unique and complete." if declared else
                         "An identifier-like column contains duplicate or missing entries.",
                 evidence={"duplicate_count": duplicate, "missing_count": missing,
                           "rows_checked": n, "sampled": sampled},
                 recommendation="Validate the intended key and the dataset's unit of analysis.")

    missing_masks: dict[str, Any] = {}
    for raw_column in frame.columns:
        column = str(raw_column)
        series = frame[raw_column]
        missing = int(series.isna().sum())
        if missing:
            missing_masks[column] = series.isna()
            _add(findings, code="variable_missingness", severity="medium" if missing / n >= .2 else "low",
                 confidence=1.0, columns=(column,), summary="This variable contains missing observations.",
                 evidence={"missing_count": missing, "missing_share": round(missing / n, 4),
                           "rows_checked": n, "sampled": sampled},
                 recommendation="State the missing-data assumption and inspect whether missingness is systematic.")

        converted, numeric_count, non_null = _numeric_view(series)
        if non_null and 0 < numeric_count < non_null:
            share = numeric_count / non_null
            if .1 <= share <= .9:
                _add(findings, code="mixed_numeric_text", severity="high", confidence=.94,
                     columns=(column,), summary="Numeric and non-numeric encodings are mixed in one variable.",
                     evidence={"numeric_parse_share": round(share, 4), "non_null": non_null},
                     recommendation="Confirm the intended type and recode invalid tokens in a derived copy.")

        try:
            distinct = int(series.nunique(dropna=True))
        except Exception as exc:
            _limited(limitations, "cardinality", exc, column)
            distinct = n
        if distinct <= 1:
            _add(findings, code="constant_variable", severity="medium", confidence=1.0,
                 columns=(column,), summary="The variable is constant or entirely missing.",
                 evidence={"distinct_non_missing": distinct},
                 recommendation="Exclude it from models unless the constant is substantively meaningful.")
        elif distinct <= max(1, math.ceil(.01 * max(1, series.notna().sum()))):
            try:
                top_share = float(series.value_counts(normalize=True, dropna=True).iloc[0])
            except Exception as exc:
                _limited(limitations, "near_constant", exc, column)
                top_share = 0
            if top_share >= .99:
                _add(findings, code="near_constant_variable", severity="medium", confidence=.96,
                     columns=(column,), summary="Almost every non-missing observation has the same value.",
                     evidence={"top_share": round(top_share, 4), "distinct_non_missing": distinct},
                     recommendation="Check coding and avoid unstable estimation with this variable.")
        if missing / n >= .95:
            _add(findings, code="extreme_sparsity", severity="high", confidence=1.0,
                 columns=(column,), summary="The variable is extremely sparse.",
                 evidence={"missing_share": round(missing / n, 4)},
                 recommendation="Confirm that the extraction or join did not lose most values.")

        # Missing-code and sentinel checks. The value is kept only in the local
        # correction instruction; model-facing summaries strip corrections.
        if numeric_count:
            clean = converted.dropna()
            for sentinel in sorted(_SENTINELS):
                count = int((clean == sentinel).sum())
                if count and count / max(1, numeric_count) >= .01:
                    at_edge = sentinel in {clean.min(), clean.max()}
                    if at_edge:
                        _add(findings, code="suspicious_sentinel", severity="high", confidence=.84,
                             columns=(column,), summary="A conventional missing-value code appears at a numeric extreme.",
                             evidence={"count": count, "share": round(count / numeric_count, 4)},
                             recommendation="Check the codebook before treating this value as missing.",
                             correction={"operation": "replace_value_with_missing", "column": column,
                                         "value": int(sentinel)})
                        break

            if numeric_count >= 20 and clean.nunique() >= 5:
                q1, q3 = clean.quantile(.25), clean.quantile(.75)
                iqr = float(q3 - q1)
                if iqr > 0:
                    extreme = int(((clean < q1 - 3 * iqr) | (clean > q3 + 3 * iqr)).sum())
                    if extreme / numeric_count >= .005:
                        _add(findings, code="aggregate_outliers", severity="medium", confidence=.88,
                             columns=(column,), summary="The variable has a notable share of extreme values.",
                             evidence={"count": extreme, "share": round(extreme / numeric_count, 4),
                                       "rule": "3x IQR"},
                             recommendation="Validate units and influential observations locally before choosing a model.")
                # Heaping at round multiples, reported without modal values.
                integer = clean[(clean % 1) == 0]
                if len(integer) >= 20:
                    heap_share = float(((integer % 5) == 0).mean())
                    if heap_share >= .8 and integer.nunique() >= 5:
                        _add(findings, code="heaping_rounding", severity="low", confidence=.78,
                             columns=(column,), summary="Values are unusually concentrated at round multiples.",
                             evidence={"multiple_of_5_share": round(heap_share, 4)},
                             recommendation="Check whether measurement or reporting was rounded.")

        if _DATE.search(column):
            non_null_series = series.dropna()
            try:
                parsed = pd.to_datetime(non_null_series, errors="coerce", format="mixed", utc=True)
                parse_share = float(parsed.notna().mean()) if len(non_null_series) else 0
                invalid = int(parsed.isna().sum())
                if parse_share >= .5 and invalid:
                    _add(findings, code="invalid_dates", severity="high", confidence=.93,
                         columns=(column,), summary="Some values in a date-like variable cannot be parsed.",
                         evidence={"invalid_count": invalid, "parse_share": round(parse_share, 4)},
                         recommendation="Choose an explicit date format and inspect invalid encodings locally.")
                if parse_share >= .5:
                    years = parsed.dropna().dt.year
                    bad = int(((years < 1900) | (parsed.dropna() > pd.Timestamp.now(tz="UTC"))).sum())
                    if bad:
                        _add(findings, code="impossible_dates", severity="high", confidence=.96,
                             columns=(column,), summary="A date-like variable contains implausible dates.",
                             evidence={"count": bad}, recommendation="Verify date parsing and source coding.")
            except Exception as exc:
                _limited(limitations, "date_validity", exc, column)
            # Mixed explicit timezone suffixes are detectable without exposing strings.
            if not pd.api.types.is_datetime64_any_dtype(series):
                try:
                    texts = series.dropna().astype(str)
                    tz = texts.str.extract(r"(Z|[+-]\d\d:?\d\d)$", expand=False).dropna()
                    if len(tz) and int(tz.nunique()) > 1:
                        _add(findings, code="inconsistent_timezones", severity="high", confidence=.9,
                             columns=(column,), summary="A timestamp variable mixes explicit UTC offsets.",
                             evidence={"distinct_offsets": int(tz.nunique()), "timestamp_count": int(len(tz))},
                             recommendation="Normalize timestamps to one declared timezone in a derived copy.")
                except Exception as exc:
                    _limited(limitations, "timezone_consistency", exc, column)

        if series.dtype == object or pd.api.types.is_string_dtype(series):
            try:
                # Pandas 3 defaults inferred text columns to the Arrow-backed
                # ``str`` dtype.  Its RE2 engine cannot execute the compiled
                # Unicode/mojibake expression below and raises ``ArrowInvalid``;
                # the broad safety boundary would then silently skip every
                # text-quality check for that column.  Select the Python string
                # engine explicitly so behavior is stable across pandas 2/3.
                texts = series.dropna().astype("string[python]")
                corrupt = int(texts.str.contains(_MOJIBAKE, regex=True).sum())
                if corrupt:
                    _add(findings, code="encoding_corruption", severity="high", confidence=.9,
                         columns=(column,), summary="Text contains replacement or mojibake markers.",
                         evidence={"affected_count": corrupt, "affected_share": round(corrupt / max(1, len(texts)), 4)},
                         recommendation="Re-import with the verified source encoding; do not guess character replacements.")
                lengths = texts.str.len()
                if len(lengths) >= 20 and lengths.max() >= 8:
                    max_share = float((lengths == lengths.max()).mean())
                    if max_share >= .15 and texts.nunique() > 2:
                        _add(findings, code="possible_truncation", severity="medium", confidence=.7,
                             columns=(column,), summary="Many text values end at the same maximum length.",
                             evidence={"maximum_length": int(lengths.max()), "at_maximum_share": round(max_share, 4)},
                             recommendation="Compare the field width with the source system schema.")
            except Exception as exc:
                _limited(limitations, "text_integrity", exc, column)

    # Missingness structure: another low-cardinality variable nearly determines
    # whether this one is missing. We report aggregate rates only.
    categorical = []
    for candidate in frame.columns:
        try:
            if 2 <= frame[candidate].nunique(dropna=True) <= 20:
                categorical.append(candidate)
        except Exception as exc:
            _limited(
                limitations, "structural_missingness_grouping", exc,
                str(candidate),
            )
    for missing_col, mask in missing_masks.items():
        if mask.mean() in (0, 1):
            continue
        for group_col in categorical[:40]:
            if str(group_col) == missing_col:
                continue
            try:
                rates = frame.assign(__missing=mask).groupby(group_col, dropna=False)["__missing"].mean()
                spread = float(rates.max() - rates.min())
                if spread >= .8 and len(rates) >= 2:
                    _add(findings, code="structural_missingness", severity="high", confidence=.9,
                         columns=(missing_col, str(group_col)),
                         summary="Missingness is strongly associated with another variable.",
                         evidence={"rate_spread": round(spread, 4), "groups_checked": int(len(rates))},
                         recommendation="Determine whether this is skip-logic/structural missingness before imputation.")
                    break
            except Exception as exc:
                _limited(
                    limitations, "structural_missingness", exc,
                    missing_col, str(group_col),
                )
                continue

    expected = context.get("expected_categories")
    if isinstance(expected, Mapping):
        for column, levels in expected.items():
            if str(column) not in frame.columns or not isinstance(levels, Sequence) or isinstance(levels, str):
                continue
            observed = set(frame[str(column)].dropna().astype(str).unique())
            allowed = {str(value) for value in levels}
            unexpected = observed - allowed
            if unexpected:
                count = int(frame[str(column)].dropna().astype(str).isin(unexpected).sum())
                _add(findings, code="unexpected_categories", severity="high", confidence=1.0,
                     columns=(str(column),), summary="Values outside the declared category set are present.",
                     evidence={"unexpected_level_count": len(unexpected), "affected_count": count},
                     recommendation="Confirm the codebook and explicitly map or reject unexpected levels.")

    units = context.get("units")
    if isinstance(units, Mapping):
        normalized: dict[str, list[str]] = {}
        for column, unit in units.items():
            if str(column) in frame.columns:
                stem = re.sub(r"_(kg|g|lb|lbs|cm|mm|m|km|c|f|usd|eur)$", "", str(column), flags=re.I)
                normalized.setdefault(stem, []).append(str(unit).casefold())
        for stem, declared_units in normalized.items():
            if len(set(declared_units)) > 1:
                cols = tuple(str(c) for c in units if str(c).startswith(stem) and str(c) in frame.columns)
                _add(findings, code="unit_mismatch", severity="critical", confidence=1.0,
                     columns=cols, summary="Related variables have conflicting declared units.",
                     evidence={"distinct_unit_count": len(set(declared_units))},
                     recommendation="Convert to one declared unit in a derived dataset before comparison or pooling.")

    _contextual_checks(frame, context, findings, limitations)
    return findings


def _contextual_checks(
    frame: Any,
    context: dict[str, Any],
    findings: list[_Finding],
    limitations: list[dict[str, Any]],
) -> None:
    import pandas as pd

    n = max(1, len(frame))
    panel = str(context.get("panel_id") or "")
    time = str(context.get("time") or "")
    if panel in frame.columns and time in frame.columns:
        work = frame[[panel, time]].dropna()
        if len(work):
            duplicate = int(work.duplicated().sum())
            if duplicate:
                _add(findings, code="duplicate_panel_time", severity="critical", confidence=1.0,
                     columns=(panel, time), summary="Panel and time do not uniquely identify observations.",
                     evidence={"duplicate_count": duplicate},
                     recommendation="Resolve the unit of analysis before longitudinal modelling.")
            sizes = work.groupby(panel).size()
            if len(sizes) > 1 and sizes.min() != sizes.max():
                _add(findings, code="panel_imbalance", severity="medium", confidence=1.0,
                     columns=(panel, time), summary="Panels contain different numbers of observed periods.",
                     evidence={"panels": int(len(sizes)), "minimum_periods": int(sizes.min()),
                               "maximum_periods": int(sizes.max())},
                     recommendation="Choose a method that supports unbalanced panels and inspect attrition.")
            try:
                parsed = pd.to_datetime(work[time], errors="coerce", format="mixed")
                valid = work.assign(__time=parsed).dropna(subset=["__time"]).sort_values([panel, "__time"])
                gaps = valid.groupby(panel)["__time"].diff().dropna()
                if len(gaps) >= 2:
                    seconds = gaps.dt.total_seconds()
                    # A low quantile is robust to missed waves; the median of
                    # one one-year and one two-year interval would otherwise
                    # invent an 18-month cadence and hide the actual gap.
                    positive = seconds[seconds > 0]
                    baseline = float(positive.quantile(.25))
                    count = int((seconds > baseline * 1.5).sum()) if baseline > 0 else 0
                    if count:
                        _add(findings, code="longitudinal_gaps", severity="medium", confidence=.9,
                             columns=(panel, time), summary="Some longitudinal intervals exceed the typical cadence.",
                             evidence={"gap_count": count, "intervals_checked": int(len(seconds))},
                             recommendation="Distinguish missed waves from the intended observation schedule.")
            except Exception as exc:
                _limited(limitations, "longitudinal_cadence", exc, panel, time)

    treatment = str(context.get("treatment") or "")
    features = _safe_columns(frame, context.get("features"))
    if treatment in frame.columns and frame[treatment].dropna().nunique() == 2:
        counts = frame[treatment].value_counts(dropna=True)
        minority = int(counts.min())
        share = minority / max(1, int(counts.sum()))
        if share < .1:
            _add(findings, code="treatment_imbalance", severity="high", confidence=1.0,
                 columns=(treatment,), summary="Treatment groups are severely imbalanced.",
                 evidence={"minority_count": minority, "minority_share": round(share, 4)},
                 recommendation="Assess positivity and use design-aware weighting or matching only if overlap supports it.")
        for feature in features:
            numeric = pd.to_numeric(frame[feature], errors="coerce")
            groups = [numeric[frame[treatment] == level].dropna() for level in counts.index]
            if all(len(group) for group in groups):
                overlap = min(float(group.max()) for group in groups) - max(float(group.min()) for group in groups)
                span = max(float(numeric.max() - numeric.min()), 1e-12)
                if overlap <= 0 or overlap / span < .05:
                    _add(findings, code="treatment_overlap", severity="critical", confidence=.95,
                         columns=(treatment, feature), summary="Treatment groups have little or no covariate overlap.",
                         evidence={"overlap_share_of_range": round(max(0, overlap) / span, 4)},
                         recommendation="Restrict the estimand/population or avoid unsupported causal extrapolation.")

    target = str(context.get("target") or "")
    if target in frame.columns:
        counts = frame[target].value_counts(dropna=True)
        if 2 <= len(counts) <= 20 and int(counts.sum()):
            minority_share = float(counts.min() / counts.sum())
            if minority_share < .05:
                _add(findings, code="class_imbalance", severity="high", confidence=1.0,
                     columns=(target,), summary="The target has a very rare class.",
                     evidence={"minority_count": int(counts.min()), "minority_share": round(minority_share, 4)},
                     recommendation="Use stratified evaluation and metrics appropriate for rare outcomes.")
        for feature in features:
            if feature == target:
                continue
            try:
                equal = frame[[target, feature]].dropna()
                identical = float((equal[target].astype(str) == equal[feature].astype(str)).mean())
            except Exception as exc:
                _limited(limitations, "target_leakage", exc, target, feature)
                identical = 0
            leak_name = any(token in feature.casefold() for token in ("outcome", "target", "label", "post", "after"))
            if identical >= .99 or (leak_name and identical >= .9):
                _add(findings, code="target_leakage", severity="critical", confidence=.99,
                     columns=(target, feature), summary="A feature nearly reproduces the target.",
                     evidence={"agreement_share": round(identical, 4)},
                     recommendation="Remove post-outcome or target-derived features before model selection.")

    split = str(context.get("split") or "")
    if split in frame.columns and frame[split].dropna().nunique() >= 2:
        feature_cols = [c for c in features if c != split] or [str(c) for c in frame.columns if str(c) != split]
        try:
            hashes = pd.util.hash_pandas_object(frame[feature_cols], index=False)
            split_counts = pd.DataFrame({"hash": hashes, "split": frame[split]}).dropna().groupby("hash")["split"].nunique()
            contaminated = int((split_counts > 1).sum())
            if contaminated:
                _add(findings, code="train_test_contamination", severity="critical", confidence=1.0,
                     columns=(split,), summary="Identical feature rows occur across data splits.",
                     evidence={"cross_split_duplicate_patterns": contaminated},
                     recommendation="Split by independent unit before preprocessing or augmentation.")
        except Exception as exc:
            _limited(limitations, "train_test_contamination", exc, split)

    weights = _safe_columns(frame, context.get("weights"))
    if not weights:
        weights = [str(c) for c in frame.columns if _WEIGHT.search(str(c))]
    for column in weights:
        numeric = pd.to_numeric(frame[column], errors="coerce").dropna()
        if len(numeric):
            nonpositive = int((numeric <= 0).sum())
            ratio = float(numeric.max() / numeric.median()) if float(numeric.median()) > 0 else math.inf
            if nonpositive or ratio > 100:
                _add(findings, code="survey_weight_anomaly", severity="high", confidence=1.0 if column in _safe_columns(frame, context.get("weights")) else .8,
                     columns=(column,), summary="A survey-weight variable has non-positive or extremely influential values.",
                     evidence={"nonpositive_count": nonpositive, "max_to_median_ratio": round(ratio, 2) if math.isfinite(ratio) else None},
                     recommendation="Verify weight construction, scaling, strata, and PSU definitions.")

    lat = str(context.get("latitude") or next((c for c in frame.columns if _LAT.search(str(c))), ""))
    lon = str(context.get("longitude") or next((c for c in frame.columns if _LON.search(str(c))), ""))
    for column, lower, upper in ((lat, -90, 90), (lon, -180, 180)):
        if column in frame.columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            bad = int(((numeric < lower) | (numeric > upper)).sum())
            if bad:
                _add(findings, code="impossible_coordinates", severity="high", confidence=.99,
                     columns=(column,), summary="Geographic coordinates fall outside valid bounds.",
                     evidence={"invalid_count": bad, "valid_range": [lower, upper]},
                     recommendation="Verify coordinate order, units, and coordinate reference system.")


def assess_frame(frame: Any, *, context: Mapping[str, Any] | None = None,
                 sampled: bool = False) -> dict[str, Any]:
    """Assess one in-memory table and return aggregate-only findings."""
    ctx = _context(context)
    limitations: list[dict[str, Any]] = []
    findings = _single_table_checks(frame, ctx, sampled, limitations)
    findings.sort(key=lambda row: (-_SEVERITY[row.severity], -row.confidence, row.code, row.columns))
    rows = [finding.row(index + 1) for index, finding in enumerate(findings)]
    return {
        "version": 1,
        "scope": "bounded_sample" if sampled else "complete_dataset",
        "rows_checked": int(len(frame)),
        "findings": rows,
        "summary": {
            "total": len(rows),
            "critical": sum(row["severity"] == "critical" for row in rows),
            "high": sum(row["severity"] == "high" for row in rows),
            "model_selection_blocked": any(
                row["severity"] == "critical" and row["confidence"] >= .9 for row in rows
            ),
            "checks_complete": not sampled and not limitations,
        },
        "limitations": limitations,
        "source_mutated": False,
    }


def assess_path(session_root: Path, source: Path, *, context: Mapping[str, Any] | None = None,
                selection: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load through the canonical layer and assess a local dataset."""
    from sift.canonical_dataset import ensure_manifest, load_canonical_dataset
    from sift.schema import full_load_max_bytes

    path = Path(source)
    sampled = path.stat().st_size > full_load_max_bytes()
    if sampled:
        manifest = ensure_manifest(session_root, path, selection=selection)
        try:
            from sift.dataset_profile import _read_frame
            frame, truncated = _read_frame(
                path, True, sheet=(selection or {}).get("worksheet"),
                session_root=session_root,
            )
        except Exception:
            # Some monolithic statistical formats cannot be sampled safely.
            # Keep that limitation explicit rather than loading past the memory
            # ceiling or falsely claiming the checks passed.
            return {
                "version": 1, "scope": "metadata_only", "rows_checked": 0,
                "findings": [{
                    "id": "DQ-001-quality_scan_deferred", "code": "quality_scan_deferred",
                    "severity": "info", "confidence": 1.0, "columns": [],
                    "summary": "Observation-level quality checks were deferred because this format has no bounded reader.",
                    "evidence": {"source_size_above_memory_ceiling": True},
                    "recommendation": "Materialize a partitioned Parquet copy or raise the local memory ceiling after review.",
                    "requires_approval": False,
                }],
                "summary": {"total": 1, "critical": 0, "high": 0,
                            "model_selection_blocked": False, "checks_complete": False},
                "source_mutated": False, "canonical_fingerprint": manifest["fingerprint"],
            }
        result = assess_frame(frame, context=context, sampled=bool(truncated))
        result["summary"]["checks_complete"] = bool(
            result["summary"].get("checks_complete") and not truncated
        )
    else:
        frame, manifest = load_canonical_dataset(session_root, path, selection=selection)
        result = assess_frame(frame, context=context)
        result["summary"]["checks_complete"] = bool(
            result["summary"].get("checks_complete")
        )
    result["canonical_fingerprint"] = manifest["fingerprint"]
    return result


def assess_relationships(
    frames: Mapping[str, Any], *, context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assess declared and inferred relationships between local tables."""
    ctx = _context(context)
    findings: list[_Finding] = []
    limitations: list[dict[str, Any]] = []
    names = list(frames)

    relations: list[tuple[str, str, str, str, bool]] = []
    raw_foreign = ctx.get("foreign_keys")
    if isinstance(raw_foreign, Sequence) and not isinstance(raw_foreign, str):
        for relation in raw_foreign:
            if not isinstance(relation, Mapping):
                continue
            child = str(relation.get("child_dataset") or "")
            parent = str(relation.get("parent_dataset") or "")
            child_column = str(relation.get("child_column") or "")
            parent_column = str(relation.get("parent_column") or "")
            if child in frames and parent in frames:
                relations.append((child, child_column, parent, parent_column, True))

    # Also inspect shared identifier-like names. These are advisory guesses,
    # never critical unless the researcher declared the relationship above.
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            shared = set(str(c) for c in frames[left].columns) & set(str(c) for c in frames[right].columns)
            for column in sorted(shared):
                if _ID.search(column):
                    candidate = (left, column, right, column, False)
                    reverse = (right, column, left, column, False)
                    if candidate not in relations and reverse not in relations:
                        relations.append(candidate)

    for child_name, child_column, parent_name, parent_column, declared in relations:
        child = frames[child_name]
        parent = frames[parent_name]
        if child_column not in child.columns or parent_column not in parent.columns:
            if declared:
                _add(findings, code="missing_relationship_column", severity="critical", confidence=1.0,
                     columns=(child_column, parent_column),
                     summary="A declared foreign-key relationship references a missing column.",
                     evidence={"child_dataset": child_name, "parent_dataset": parent_name},
                     recommendation="Correct the relationship declaration or extraction schema.")
            continue
        child_values = child[child_column].dropna()
        parent_values = parent[parent_column].dropna()
        try:
            child_unique = int(child_values.nunique()) == int(len(child_values))
            parent_unique = int(parent_values.nunique()) == int(len(parent_values))
            orphan = int((~child_values.isin(set(parent_values.unique()))).sum())
        except Exception as exc:
            _limited(
                limitations, "relationship_integrity", exc,
                child_name, child_column, parent_name, parent_column,
            )
            continue
        if orphan:
            _add(findings, code="orphan_foreign_keys", severity="critical" if declared else "high",
                 confidence=1.0 if declared else .75, columns=(child_column, parent_column),
                 summary="Child keys are absent from the referenced parent table.",
                 evidence={"orphan_count": orphan, "child_rows_checked": int(len(child_values)),
                           "child_dataset": child_name, "parent_dataset": parent_name},
                 recommendation="Review extraction filters and key normalization before joining.")
        if not child_unique and not parent_unique:
            _add(findings, code="many_to_many_merge", severity="critical" if declared else "high",
                 confidence=1.0 if declared else .82, columns=(child_column, parent_column),
                 summary="Both sides of a candidate join contain repeated keys.",
                 evidence={"child_dataset": child_name, "parent_dataset": parent_name,
                           "child_distinct": int(child_values.nunique()),
                           "parent_distinct": int(parent_values.nunique())},
                 recommendation="Aggregate or deduplicate the intended one-side before joining.")
        if declared and not parent_unique:
            _add(findings, code="parent_key_uniqueness", severity="critical", confidence=1.0,
                 columns=(parent_column,), summary="The declared parent key is not unique.",
                 evidence={"parent_dataset": parent_name, "rows_checked": int(len(parent_values)),
                           "distinct": int(parent_values.nunique())},
                 recommendation="Resolve parent-key duplication before using this foreign key.")

    findings.sort(key=lambda row: (-_SEVERITY[row.severity], -row.confidence, row.code, row.columns))
    rows = [finding.row(index + 1) for index, finding in enumerate(findings)]
    return {
        "version": 1, "scope": "dataset_relationships", "findings": rows,
        "summary": {
            "total": len(rows),
            "critical": sum(row["severity"] == "critical" for row in rows),
            "high": sum(row["severity"] == "high" for row in rows),
            "model_selection_blocked": any(
                row["severity"] == "critical" and row["confidence"] >= .9 for row in rows
            ),
            "checks_complete": not limitations,
        },
        "limitations": limitations,
        "source_mutated": False,
    }


def safe_preflight(report: Mapping[str, Any]) -> dict[str, Any]:
    """Strip correction values and exact sensitive counts for model exposure."""
    rows = []
    for finding in report.get("findings", []):
        rows.append({
            "code": finding.get("code"), "severity": finding.get("severity"),
            "confidence": finding.get("confidence"), "columns": finding.get("columns", []),
            "summary": finding.get("summary"), "recommendation": finding.get("recommendation"),
        })
    limitations = []
    for item in report.get("limitations", []):
        if isinstance(item, Mapping):
            limitations.append({
                "check": item.get("check"),
                "columns": list(item.get("columns", []))[:8],
                "reason": item.get("reason"),
            })
    return {
        "summary": dict(report.get("summary", {})),
        "findings": rows[:50],
        "limitations": limitations[:50],
    }


def apply_approved_corrections(
    session_root: Path, source: Path, *, approved_finding_ids: Sequence[str],
    output_name: str, context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply freshly verified, explicitly approved safe corrections to a copy."""
    from sift.canonical_dataset import create_manifest, load_canonical_dataset

    root = Path(session_root).resolve(strict=True)
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = root / source_path
    source_path = source_path.resolve(strict=True)
    if not source_path.is_relative_to(root):
        raise DataQualityError("source must be inside the session")
    if not approved_finding_ids:
        raise DataQualityError("at least one finding must be explicitly approved")
    if Path(output_name).name != output_name or not output_name.casefold().endswith(".parquet"):
        raise DataQualityError("output_name must be a simple .parquet filename")
    output = root / output_name
    if output.exists():
        raise DataQualityError("correction output already exists")

    frame, parent = load_canonical_dataset(root, source_path)
    report = assess_frame(frame, context=context)
    by_id = {row["id"]: row for row in report["findings"]}
    requested = list(dict.fromkeys(str(value) for value in approved_finding_ids))
    if any(value not in by_id for value in requested):
        raise DataQualityError("an approved finding is stale or does not belong to this source")
    operations = []
    result = frame.copy(deep=True)
    for finding_id in requested:
        row = by_id[finding_id]
        correction = row.get("correction")
        if not isinstance(correction, Mapping):
            raise DataQualityError(f"finding {finding_id} has no safe automatic correction")
        operation = correction.get("operation")
        if operation == "drop_exact_duplicate_rows":
            before = len(result)
            result = result.drop_duplicates().copy()
            operations.append({"operation": operation, "finding_id": finding_id,
                               "rows_removed": int(before - len(result))})
        elif operation == "replace_value_with_missing":
            column = str(correction.get("column"))
            if column not in result.columns:
                raise DataQualityError("correction column no longer exists")
            mask = result[column] == correction.get("value")
            changed = int(mask.sum())
            result.loc[mask, column] = None
            operations.append({"operation": operation, "finding_id": finding_id,
                               "column": column, "cells_changed": changed})
        else:
            raise DataQualityError(f"unsupported correction operation: {operation}")

    # Publish without an overwrite race: write a private temporary file then
    # hard-link it into the requested new name (which fails if another writer
    # created that name). The source descriptor is never opened for writing.
    fd, temporary_raw = tempfile.mkstemp(prefix=".quality-correction-", suffix=".parquet", dir=root)
    os.close(fd)
    temporary = Path(temporary_raw)
    try:
        result.to_parquet(temporary, index=False)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise DataQualityError("correction output already exists") from exc
        provenance_payload = json.dumps(operations, sort_keys=True, separators=(",", ":"))
        try:
            manifest = create_manifest(
                root, output, frame=result, dataset_kind="derived",
                parents=[parent["fingerprint"]],
                transformations=[{
                    "operation": "approved_data_quality_corrections",
                    "accepted_finding_ids": ",".join(requested),
                    "correction_count": len(operations),
                    "parameters_sha256": hashlib.sha256(provenance_payload.encode("utf-8")).hexdigest(),
                }],
            )
        except Exception:
            output.unlink(missing_ok=True)
            raise
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "ok": True, "source": source_path.relative_to(root).as_posix(),
        "output": output.name, "parent_fingerprint": parent["fingerprint"],
        "canonical_fingerprint": manifest["fingerprint"],
        "accepted_findings": requested, "corrections": operations,
        "source_mutated": False,
    }
