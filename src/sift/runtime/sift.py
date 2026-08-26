"""Sift runtime library for Python.

Imported at the top of every Python script Sift runs. Provides the
single sanctioned I/O surface for emitting structured results:

    sift.result(type="linear_regression", ...)
    sift.from_lm(model)              # statsmodels OLS / GLM result
    sift.from_t_test(res, n1=..., n2=...)   # scipy.stats t-test
    sift.from_summarize(variable, n, mean, sd, missing_count)
    sift.from_table(variable, counts, n=..., missing_count=...)
    sift.from_crosstab(table)
    sift.from_magnitude_table(df, group_var, value_var, aggregation="sum")

The script writes structured payloads to ``$SIFT_RESULT_PATH``. Raw
stdout / stderr are captured by the executor as the raw log the
researcher sees in the TUI; only the structured JSON reaches the
sanitizer (and from there, the model).

Hard requirements: ``pandas`` and ``numpy`` are needed to load the
module — the runtime ships a small numpy/pandas-aware JSON encoder
so floats / int64 / NaN serialise cleanly. ``statsmodels`` and
``scipy`` are needed only by the ``from_lm`` and ``from_t_test``
helpers; scripts that emit via the generic ``result(...)`` path or
the descriptive helpers don't need them.

NOTE on numeric precision: floats are emitted at full IEEE-754
precision. The Python sanitizer (``sift.sanitizer``) clamps
precision per-type using ``sigfigs_for_n`` after the payload comes
back. Mirrors the R / Stata libraries — language-of-origin doesn't
change what the model sees.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Per-run authenticity token
# ---------------------------------------------------------------------------
#
# Read the token once at import time, stash in module state, and
# clear the env var so user code that imports ``sift`` later can't
# read it via ``os.environ``. A determined script can still reach
# into ``sift._RUN_TOKEN`` directly — Python module state is
# inherently inspectable — but doing so requires code that obviously
# shows up in the script the researcher reviews. Same posture as the
# R library; see ``docs/architecture.md`` "runtime-library contract"
# for the deliberate limits of this measure.

_RUN_TOKEN: str = os.environ.pop("SIFT_RUN_TOKEN", "")
if not _RUN_TOKEN:
    raise RuntimeError(
        "SIFT_RUN_TOKEN not set. This script must be run through the "
        "Sift executor; direct ``python`` invocation of user code that "
        "emits result payloads isn't supported."
    )

_RESULT_PATH: str = os.environ.get("SIFT_RESULT_PATH", "")
if not _RESULT_PATH:
    raise RuntimeError(
        "SIFT_RESULT_PATH not set. The Sift executor sets this; if you "
        "see this error in normal usage, the executor wiring is broken."
    )


# ---------------------------------------------------------------------------
# JSON encoder that knows about pandas / numpy types
# ---------------------------------------------------------------------------


class _SiftJSONEncoder(json.JSONEncoder):
    """Handle the numeric / pandas / numpy types stats scripts emit.

    - numpy scalars (np.int64, np.float64, np.bool_) -> Python equivalents
    - numpy arrays / pandas Series / pandas Index -> lists
    - non-finite floats (NaN, Inf) -> JSON null (matches the R library)
    - dataclass instances -> dict (best-effort)
    """

    def default(self, obj: Any) -> Any:  # noqa: D401
        # Lazy imports so the runtime works without numpy/pandas if a
        # script never emits one of their types (rare in practice).
        try:
            import numpy as np
        except ImportError:
            np = None  # type: ignore[assignment]
        try:
            import pandas as pd
        except ImportError:
            pd = None  # type: ignore[assignment]

        if np is not None:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                f = float(obj)
                return f if math.isfinite(f) else None
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        if pd is not None:
            if isinstance(obj, (pd.Series, pd.Index)):
                return obj.tolist()
            if isinstance(obj, pd.DataFrame):
                # DataFrames serialise as a list of row-dicts so the
                # sanitizer (which scans by field name) can still read
                # them. Researchers rarely want this — they should
                # build an explicit payload via ``result(...)`` —
                # but the fallback prevents an inscrutable
                # TypeError in the JSON pass.
                return obj.to_dict(orient="records")
        # Standard floats with non-finite values.
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return super().default(obj)


def _scrub_non_finite(obj: Any) -> Any:
    """Replace non-finite *plain* Python floats with ``None``.

    The encoder's ``default()`` only fires for objects ``json.dumps``
    doesn't know how to serialise natively; ``float('nan')`` /
    ``float('inf')`` ARE natively serialisable, so the encoder's
    non-finite branch never sees them and ``allow_nan=True`` would
    emit RFC-8259-invalid ``NaN`` / ``Infinity`` tokens. Walk the
    payload first and substitute ``None`` so the wire stays
    strict-JSON, matching the R library's "non-finite -> null"
    contract.
    """
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _scrub_non_finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_scrub_non_finite(v) for v in obj]
    return obj


def _to_json(payload: dict[str, Any]) -> str:
    """Serialise ``payload`` with the numpy/pandas-aware encoder.

    ``allow_nan=False`` ensures any float Inf/NaN the pre-pass missed
    raises a ``ValueError`` rather than silently emitting RFC-8259-
    invalid tokens. Plain Python non-finite floats are scrubbed to
    ``None`` up front so a NaN coefficient (e.g. perfect collinearity)
    serialises as null rather than crashing the script post-fit;
    numpy non-finite floats still go through the encoder's
    ``default()``.
    """
    return json.dumps(
        _scrub_non_finite(payload), cls=_SiftJSONEncoder, allow_nan=False,
    )


# ---------------------------------------------------------------------------
# Core emit
# ---------------------------------------------------------------------------


def _write_result(payload: dict[str, Any]) -> None:
    """Embed the per-run token and append the JSON payload to disk.

    The result file is JSONL: one object per line. Each helper call
    appends its own line, so a script that calls multiple helpers
    surfaces every payload back to the executor. The executor
    validates the token on each line and strips it before the
    payload reaches the sanitizer. A hand-crafted line that
    bypasses this function will be rejected (no token).
    """
    if not isinstance(payload, dict):
        raise TypeError(
            f"sift payload must be a dict, got {type(payload).__name__}"
        )
    payload = dict(payload)  # don't mutate caller's dict
    payload["_token"] = _RUN_TOKEN
    with open(_RESULT_PATH, "a", encoding="utf-8") as f:
        f.write(_to_json(payload))
        f.write("\n")


def result(*, type: str, **fields: Any) -> None:  # noqa: A002 — match R API
    """Generic emit. Use one of the ``from_*`` helpers when there's
    a matching one — they pull standard fields out of common Python
    objects (statsmodels results, scipy ttest_result, pandas
    DataFrames) so the researcher doesn't have to assemble the dict
    by hand.

    ``_via_helper`` and ``_registry_method_id`` are reserved
    sanitizer-side markers. Typed helpers stamp them only after they
    have inspected the source object. Strip both here so a script
    cannot forge helper provenance or bind a generic coefficient
    table to an approved registry method.
    """
    fields = {
        k: v for k, v in fields.items()
        if k not in ("_via_helper", "_registry_method_id")
    }
    payload = {"type": type, **fields}
    _write_result(payload)


def from_method(
    method_id: str, *, n: int, diagnostics: dict[str, Any],
    estimates: dict[str, float] | None = None,
    standard_errors: dict[str, float] | None = None,
    p_values: dict[str, float] | None = None,
    ci_lower: dict[str, float] | None = None,
    ci_upper: dict[str, float] | None = None,
    metrics: dict[str, float] | None = None,
    **metadata: Any,
) -> None:
    """Emit a registry-backed, aggregate-only methodology result.

    Diagnostics are explicit rather than inferred from the fitted object: the
    registry contract varies by method and refusing an omitted diagnostic is
    safer than guessing that an absent check passed. Observation-level values,
    predictions, residuals, and influence rows have no accepted field.
    """
    payload: dict[str, Any] = {
        "method_id": method_id, "n": n, "diagnostics": diagnostics,
    }
    for key, value in {
        "estimates": estimates, "standard_errors": standard_errors,
        "p_values": p_values, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "metrics": metrics,
    }.items():
        if value is not None:
            payload[key] = value
    payload.update(metadata)
    result(type="method_result", **payload)


def _method_diagnostics(
    defaults: dict[str, Any], overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Merge a typed helper's conservative diagnostics with explicit checks.

    Defaults are deliberately ``warn``/``not_applicable`` when a fitted result
    does not retain enough source data to recompute an assumption check.  A
    helper must never turn "the library object lacks this information" into a
    fabricated pass.  Callers may supply results of checks they actually ran.
    """
    out = dict(defaults)
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError("diagnostics must be a dict")
        out.update(overrides)
    return out


_METHOD_QUANTITY_RE = re.compile(r"^[A-Za-z0-9_.(][A-Za-z0-9_.():^#]*$")


def _method_quantity_name(value: Any, *, field: str = "name") -> str:
    """Validate a name before emission so the sanitizer cannot drop it.

    This mirrors the method-result identifier boundary (40 characters and a
    formula/identifier alphabet).  Raising locally is preferable to writing a
    plausible-looking payload whose only estimate disappears later.
    """
    name = str(value)
    if len(name) > 40 or _METHOD_QUANTITY_RE.fullmatch(name) is None:
        raise ValueError(
            f"{field} must be a unique identifier/formula-shaped name of at most 40 characters"
        )
    return name


def _method_positive_int(value: Any, *, field: str) -> int:
    numeric = _safe_float(value)
    if numeric is None or numeric <= 0 or numeric != int(numeric):
        raise ValueError(f"{field} must be a positive integer")
    return int(numeric)


def _method_nonnegative_int(value: Any, *, field: str) -> int:
    numeric = _safe_float(value)
    if numeric is None or numeric < 0 or numeric != int(numeric):
        raise ValueError(f"{field} must be a non-negative integer")
    return int(numeric)


def _method_float(value: Any, *, field: str) -> float:
    """Return one required finite method scalar or refuse the helper call."""
    numeric = _safe_float(value)
    if numeric is None:
        raise ValueError(f"{field} must be a finite number")
    return numeric


def from_descriptive_confidence_interval(
    summary: Any, *, name: str = "mean", confidence: float = 0.95,
    missing_count: int | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit a mean CI from ``statsmodels.stats.weightstats.DescrStatsW``.

    ``DescrStatsW`` is the maintained reference implementation: the helper
    calls ``tconfint_mean`` and emits only N, mean, standard error and the two
    interval endpoints.  The observation vector and weights never cross the
    result boundary.
    """
    name = _method_quantity_name(name)
    conf = _safe_float(confidence)
    if conf is None or not 0.0 < conf < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    n = _method_positive_int(getattr(summary, "nobs", None), field="summary.nobs")
    estimate = _method_float(getattr(summary, "mean", None), field="summary.mean")
    standard_error = _method_float(
        getattr(summary, "std_mean", None), field="summary.std_mean",
    )
    try:
        lo, hi = summary.tconfint_mean(alpha=1.0 - conf)
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            "summary must provide statsmodels-compatible tconfint_mean()"
        ) from exc
    missing_status: Any = "not_applicable"
    if missing_count is not None:
        missing = _method_nonnegative_int(missing_count, field="missing_count")
        missing_status = missing
    diag = _method_diagnostics({
        "missingness": missing_status,
        "effective_sample_size": n,
        "confidence_level": conf,
    }, diagnostics)
    from_method(
        "descriptive_confidence_interval", n=n, diagnostics=diag,
        estimates={name: estimate}, standard_errors={name: standard_error},
        ci_lower={name: _method_float(lo, field="confidence interval lower")},
        ci_upper={name: _method_float(hi, field="confidence interval upper")},
        uncertainty_type="classical", **metadata,
    )


def from_nonparametric_test(
    test_result: Any, *, n: int, name: str = "rank_test",
    group_sizes: Any | None = None, ties_checked: Any = "not_applicable",
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit an aggregate SciPy Mann-Whitney/Wilcoxon/Kruskal result."""
    safe_n = _method_positive_int(n, field="n")
    name = _method_quantity_name(name)
    sizes_ok: Any = "not_applicable"
    if group_sizes is not None:
        try:
            sizes = [_safe_int(value) for value in group_sizes]
        except TypeError as exc:
            raise TypeError("group_sizes must be an iterable of counts") from exc
        sizes_ok = bool(
            sizes and all(value is not None and value >= 0 for value in sizes)
            and safe_n is not None and sum(value for value in sizes if value is not None) == safe_n
        )
    diag = _method_diagnostics({
        "group_sample_sizes": sizes_ok,
        "ties_and_zero_differences": ties_checked,
    }, diagnostics)
    from_method(
        "nonparametric_test", n=safe_n, diagnostics=diag,
        estimates={name: _method_float(
            getattr(test_result, "statistic", None), field="test statistic",
        )},
        p_values={name: _method_float(
            getattr(test_result, "pvalue", None), field="test p-value",
        )},
        uncertainty_type="classical", **metadata,
    )


def from_proportion_test(
    count: Any, nobs: Any, *, value: float | None = None,
    alternative: str = "two-sided", name: str = "proportion",
    confidence: float = 0.95, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Run and emit a one- or two-sample statsmodels proportion z-test.

    A one-sample result includes a Wilson interval for the proportion.  A
    two-sample result reports the proportion difference; no interval is
    fabricated because its appropriate construction depends on the requested
    estimand and method.
    """
    import numpy as np
    import pandas as pd
    from statsmodels.stats.proportion import proportion_confint, proportions_ztest

    name = _method_quantity_name(name)
    if alternative not in {"two-sided", "larger", "smaller"}:
        raise ValueError("alternative must be 'two-sided', 'larger', or 'smaller'")
    conf = _safe_float(confidence)
    if conf is None or not 0.0 < conf < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    counts = np.atleast_1d(np.asarray(count, dtype=float))
    totals = np.atleast_1d(np.asarray(nobs, dtype=float))
    if counts.size not in {1, 2} or totals.size != counts.size:
        raise ValueError("count and nobs must describe one or two groups")
    if (not np.all(np.isfinite(counts)) or not np.all(np.isfinite(totals))
            or np.any(totals <= 0) or np.any(counts < 0) or np.any(counts > totals)):
        raise ValueError("counts and denominators must be finite with 0 <= count <= nobs")
    if np.any(counts != np.floor(counts)) or np.any(totals != np.floor(totals)):
        raise ValueError("counts and denominators must be integers")
    statistic, pvalue = proportions_ztest(
        counts, totals, value=value, alternative=alternative,
    )
    # statsmodels returns length-one ndarrays for a one-sample call when the
    # input was array-shaped. Normalize both scalar and vector return forms.
    statistic = float(np.asarray(statistic).reshape(-1)[0])
    pvalue = float(np.asarray(pvalue).reshape(-1)[0])
    proportions = counts / totals
    estimate = float(proportions[0]) if counts.size == 1 else float(proportions[0] - proportions[1])
    expected_ok = bool(np.all(counts >= 5) and np.all(totals - counts >= 5))
    diag = _method_diagnostics({
        "group_sample_sizes": "pass",
        "expected_cell_counts": "pass" if expected_ok else "warn",
    }, diagnostics)
    kwargs: dict[str, Any] = {}
    if counts.size == 1:
        alpha = 1.0 - conf
        lo, hi = proportion_confint(counts[0], totals[0], alpha=alpha, method="wilson")
        kwargs.update(ci_lower={name: float(lo)}, ci_upper={name: float(hi)})
    from_method(
        "proportion_test", n=int(totals.sum()), diagnostics=diag,
        estimates={name: estimate}, p_values={name: pvalue},
        metrics={"z_statistic": statistic},
        uncertainty_type="classical", **kwargs, **metadata,
    )


def from_anova(
    model: Any, *, method_id: str = "anova", table: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit an ANOVA/ANCOVA table from a fitted statsmodels formula model."""
    if method_id not in {"anova", "ancova"}:
        raise ValueError("method_id must be 'anova' or 'ancova'")
    if table is None:
        from statsmodels.stats.anova import anova_lm
        table = anova_lm(model, typ=2)
    n = _method_positive_int(getattr(model, "nobs", None), field="model.nobs")
    metrics: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for raw_name, row in table.iterrows():
        if str(raw_name).lower() in {"residual", "residuals"}:
            continue
        key = _method_quantity_name(raw_name, field="ANOVA effect name")
        f_value = _safe_float(row.get("F"))
        p_value = _safe_float(row.get("PR(>F)"))
        if f_value is not None:
            metrics[key] = f_value
        if p_value is not None:
            p_values[key] = p_value
    defaults = {
        "group_sample_sizes": "not_applicable",
        "residual_distribution": "warn",
        "homogeneity_of_variance": "warn",
    }
    if method_id == "ancova":
        defaults["parallel_slopes"] = "warn"
    diag = _method_diagnostics(defaults, diagnostics)
    from_method(
        method_id, n=n, diagnostics=diag, metrics=metrics,
        p_values=p_values, uncertainty_type="classical", **metadata,
    )


def from_repeated_measures(
    fit: Any, *, n: int, subjects: int, records: int | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit a statsmodels ``AnovaRM.fit()`` aggregate result."""
    table = getattr(fit, "anova_table", None)
    if table is None:
        raise TypeError("fit must be a statsmodels AnovaRM result")
    safe_n = _method_positive_int(n, field="n")
    safe_subjects = _method_positive_int(subjects, field="subjects")
    safe_records = safe_n if records is None else _method_positive_int(records, field="records")
    if safe_subjects > safe_records or safe_records > safe_n:
        raise ValueError("require subjects <= records <= n")
    metrics: dict[str, float] = {}
    p_values: dict[str, float] = {}
    for raw_name, row in table.iterrows():
        key = _method_quantity_name(raw_name, field="repeated-measures effect name")
        f_value = _safe_float(row.get("F Value"))
        p_value = _safe_float(row.get("Pr > F"))
        if f_value is not None:
            metrics[key] = f_value
        if p_value is not None:
            p_values[key] = p_value
    cluster_size = None
    if safe_records is not None and safe_subjects:
        cluster_size = safe_records / safe_subjects
    diag = _method_diagnostics({
        "cluster_count": safe_subjects,
        "cluster_size": cluster_size,
        "complete_cases": "pass" if safe_records == safe_n else "warn",
        "sphericity_or_correction": "warn",
    }, diagnostics)
    from_method(
        "repeated_measures_test", n=safe_n, subjects=safe_subjects,
        records=safe_records, diagnostics=diag, metrics=metrics,
        p_values=p_values, uncertainty_type="classical", **metadata,
    )


def from_multiple_testing(
    p_values: Any, *, n: int, method: str = "holm", alpha: float = 0.05,
    labels: Any | None = None, **metadata: Any,
) -> None:
    """Apply statsmodels multiplicity correction and emit raw/adjusted p-values."""
    from statsmodels.stats.multitest import multipletests

    safe_n = _method_positive_int(n, field="n")
    safe_alpha = _safe_float(alpha)
    if safe_alpha is None or not 0.0 < safe_alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    values = [_safe_float(value) for value in p_values]
    if not values or len(values) > 100 or any(
        value is None or not 0.0 <= value <= 1.0 for value in values
    ):
        raise ValueError("p_values must contain 1..100 finite probabilities")
    allowed = {
        "holm": "holm", "bonferroni": "bonferroni",
        "benjamini_hochberg": "fdr_bh",
    }
    if method not in allowed:
        raise ValueError(f"method must be one of {sorted(allowed)}")
    if labels is None:
        names = [f"hypothesis_{index + 1}" for index in range(len(values))]
    else:
        names = [_method_quantity_name(value, field="label") for value in labels]
        if len(names) != len(values):
            raise ValueError("labels length must match p_values")
    if len(set(names)) != len(names):
        raise ValueError("labels must be unique")
    numeric_values = [value for value in values if value is not None]
    reject, corrected, _, _ = multipletests(
        numeric_values, alpha=safe_alpha, method=allowed[method],
    )
    from_method(
        "multiple_testing_correction", n=safe_n,
        diagnostics={"hypothesis_family": "pass", "correction_applied": "pass"},
        estimates=dict(zip(names, numeric_values, strict=True)),
        p_values={key: float(value) for key, value in zip(names, corrected, strict=True)},
        metrics={
            "hypothesis_count": float(len(values)),
            "rejection_count": float(sum(bool(value) for value in reject)),
            "alpha": safe_alpha,
        },
        multiple_testing=method, **metadata,
    )


# ---------------------------------------------------------------------------
# Missing-data helpers — bounded aggregates and explicit claim boundaries
# ---------------------------------------------------------------------------

def _missingness_aggregates(data: Any) -> tuple[int, int, float, float, int, float]:
    """Return aggregate missingness facts without emitting row-level patterns."""
    import numpy as np
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    rows, columns = data.shape
    if rows <= 0 or columns <= 0 or columns > 100:
        raise ValueError("data must contain rows and 1..100 columns")
    missing = data.isna().to_numpy(dtype=bool)
    missing_cells = int(missing.sum())
    complete = ~missing.any(axis=1)
    pattern_count = int(np.unique(missing, axis=0).shape[0])
    pattern_sizes = np.unique(missing, axis=0, return_counts=True)[1]
    return (
        rows, columns, missing_cells / float(rows * columns),
        float(complete.mean()), pattern_count,
        float(pattern_sizes.max() / rows),
    )


def from_missingness_pattern(
    data: Any, *, complete_case_warning_threshold: float = 0.10,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit joint missingness and complete-case *aggregates* from a DataFrame.

    Individual missingness patterns and variable values never cross the result
    boundary.  The helper intentionally makes no MCAR/MAR/MNAR classification:
    those mechanisms are not identified by an observed pattern table.
    """
    threshold = _safe_float(complete_case_warning_threshold)
    if threshold is None or not 0.0 < threshold < 1.0:
        raise ValueError("complete_case_warning_threshold must be between 0 and 1")
    n, columns, missing_fraction, complete_rate, patterns, largest = (
        _missingness_aggregates(data)
    )
    incomplete_rate = 1.0 - complete_rate
    diag = _method_diagnostics({
        "missingness_pattern": "pass",
        "complete_case_rate": complete_rate,
        "complete_case_warning": (
            "warn" if incomplete_rate >= threshold else "pass"
        ),
    }, diagnostics)
    from_method(
        "missingness_pattern", n=n, diagnostics=diag,
        metrics={
            "variable_count": float(columns),
            "missing_fraction": missing_fraction,
            "complete_case_rate": complete_rate,
            "complete_case_warning_threshold": threshold,
            "missingness_pattern_count": float(patterns),
            "largest_pattern_fraction": largest,
        },
        **metadata,
    )


def from_single_imputation(
    imputer: Any, original: Any, transformed: Any, *,
    scope: str, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Emit a fitted ``sklearn.SimpleImputer`` preprocessing audit.

    This helper refuses coefficients, tests, or intervals by construction.
    It is only an audit of deterministic preprocessing for prediction or a
    nuisance covariate, never a substitute for missing-data uncertainty.
    """
    import numpy as np
    import pandas as pd
    from sklearn.impute import SimpleImputer

    if not isinstance(imputer, SimpleImputer) or not hasattr(imputer, "statistics_"):
        raise TypeError("imputer must be a fitted sklearn SimpleImputer")
    if scope not in {"prediction_preprocessing", "deterministic_nuisance_covariate"}:
        raise ValueError(
            "scope must be prediction_preprocessing or deterministic_nuisance_covariate"
        )
    source = np.asarray(original)
    completed = np.asarray(transformed)
    if source.ndim != 2 or completed.ndim != 2 or source.shape[0] != completed.shape[0]:
        raise ValueError("original and transformed must be 2D with matching row counts")
    if source.shape[0] <= 0 or source.shape[1] <= 0 or source.shape[1] > 100:
        raise ValueError("original must contain rows and 1..100 columns")
    if completed.shape[1] != source.shape[1]:
        raise ValueError("SimpleImputer must retain every input feature")
    if hasattr(original, "columns"):
        original_columns = [str(value) for value in original.columns]
        fitted_columns = [str(value) for value in getattr(imputer, "feature_names_in_", ())]
        if fitted_columns != original_columns:
            raise ValueError("SimpleImputer feature names/order do not match original")
        if hasattr(transformed, "columns"):
            output_columns = [str(value) for value in transformed.columns]
            expected_columns = [str(value) for value in imputer.get_feature_names_out()]
            if output_columns != expected_columns:
                raise ValueError("transformed feature names/order do not match SimpleImputer")
    expected = np.asarray(imputer.transform(original))
    if expected.shape != completed.shape:
        raise ValueError("transformed shape does not match SimpleImputer.transform(original)")
    try:
        same_transform = np.allclose(
            completed.astype(float), expected.astype(float), rtol=1e-10, atol=1e-12,
            equal_nan=True,
        )
    except (TypeError, ValueError):
        same_transform = bool(np.array_equal(completed, expected))
    if not same_transform:
        raise ValueError("transformed values do not match SimpleImputer.transform(original)")
    missing = np.asarray(pd.isna(source), dtype=bool)
    if not missing.any():
        raise ValueError("single-imputation audit requires at least one missing value")
    try:
        completed_numeric = completed.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("transformed output must be finite numeric data") from exc
    if not np.all(np.isfinite(completed_numeric)):
        raise ValueError("transformed output still contains missing/non-finite values")
    n = int(source.shape[0])
    diag = _method_diagnostics({
        "missingness_pattern": "pass",
        "imputation_scope": "pass",
        "inferential_uncertainty_not_claimed": "pass",
    }, diagnostics)
    from_method(
        "single_imputation", n=n, diagnostics=diag,
        metrics={
            "feature_count": float(source.shape[1]),
            "output_feature_count": float(completed.shape[1]),
            "missing_fraction": float(missing.mean()),
            "affected_row_fraction": float(missing.any(axis=1).mean()),
            "imputed_cell_count": float(missing.sum()),
        },
        imputation_scope=scope, imputation_model="simple_deterministic",
        **metadata,
    )


def _rubin_pool(fits: Any) -> dict[str, Any]:
    """Pool homogeneous fitted-result objects using scalar Rubin rules."""
    import numpy as np
    from scipy import stats

    results = list(fits)
    if len(results) < 2 or len(results) > 50:
        raise ValueError("Rubin pooling requires 2..50 fitted analyses")
    first = results[0]
    raw_names = getattr(getattr(first, "model", None), "exog_names", None)
    if raw_names is None:
        raw_names = getattr(first, "exog_names", None)
    if raw_names is None:
        raw_names = getattr(getattr(first, "model", None), "data", None)
        raw_names = getattr(raw_names, "param_names", None)
    if raw_names is None:
        raise TypeError("fitted analyses must expose model.exog_names")
    names = [_method_quantity_name(value, field="parameter name") for value in raw_names]
    if (len(names) > 18 or len(set(names)) != len(names)
            or any(len(name) > 30 for name in names)):
        raise ValueError(
            "Rubin pooling supports 1..18 unique parameter names of at most 30 characters"
        )
    params: list[Any] = []
    covariances: list[Any] = []
    sample_sizes: list[int] = []
    complete_dfs: list[float] = []
    for fitted in results:
        current_names = getattr(getattr(fitted, "model", None), "exog_names", None)
        if current_names is None:
            current_names = getattr(fitted, "exog_names", None)
        if list(current_names or ()) != list(raw_names):
            raise ValueError("all imputation analyses must have identical parameters")
        vector = np.asarray(getattr(fitted, "params", None), dtype=float).reshape(-1)
        covariance = np.asarray(fitted.cov_params(), dtype=float)
        nobs = _method_positive_int(getattr(fitted, "nobs", None), field="fit.nobs")
        complete_df = _safe_float(getattr(fitted, "df_resid", None))
        if (vector.shape != (len(names),)
                or covariance.shape != (len(names), len(names))
                or not np.all(np.isfinite(vector))
                or not np.all(np.isfinite(covariance))
                or complete_df is None or complete_df <= 0):
            raise ValueError("each fitted analysis needs finite matching parameters/covariance")
        if not np.allclose(covariance, covariance.T, rtol=1e-8, atol=1e-10):
            raise ValueError("each fitted covariance matrix must be symmetric")
        scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10 * scale:
            raise ValueError("each fitted covariance matrix must be positive semidefinite")
        params.append(vector)
        covariances.append(covariance)
        sample_sizes.append(nobs)
        complete_dfs.append(complete_df)
    if len(set(sample_sizes)) != 1:
        raise ValueError("all imputation analyses must use the same sample size")
    if max(complete_dfs) - min(complete_dfs) > 1e-8:
        raise ValueError("all imputation analyses must have the same residual degrees of freedom")
    matrix = np.asarray(params)
    within = np.mean(np.asarray(covariances), axis=0)
    between = np.cov(matrix, rowvar=False, ddof=1)
    if np.ndim(between) == 0:
        between = np.asarray([[float(between)]])
    m = len(results)
    total = within + (1.0 + 1.0 / m) * between
    within_diag = np.diag(within)
    between_diag = np.diag(between)
    total_diag = np.diag(total)
    if (np.any(within_diag < -1e-12) or np.any(between_diag < -1e-12)
            or np.any(total_diag <= 0) or not np.all(np.isfinite(total_diag))):
        raise ValueError("Rubin variance components must be finite and non-negative")
    for label, covariance in (("within", within), ("between", between), ("total", total)):
        scale = max(1.0, float(np.linalg.norm(covariance, ord=2)))
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-10 * scale:
            raise ValueError(f"Rubin {label} covariance must be positive semidefinite")
    pooled = matrix.mean(axis=0)
    se = np.sqrt(total_diag)
    missing_variance = (1.0 + 1.0 / m) * np.maximum(between_diag, 0.0)
    lambda_missing = np.clip(missing_variance / total_diag, 0.0, 1.0)
    ratio = np.divide(
        missing_variance, np.maximum(within_diag, 1e-300),
        out=np.full_like(missing_variance, np.inf), where=within_diag > 0,
    )
    old_degrees = np.where(
        missing_variance <= 1e-15, np.inf,
        (m - 1.0) * (1.0 + 1.0 / np.maximum(ratio, 1e-300)) ** 2,
    )
    complete_df = complete_dfs[0]
    observed_degrees = (
        ((complete_df + 1.0) / (complete_df + 3.0))
        * complete_df * (1.0 - lambda_missing)
    )
    degrees = np.where(
        np.isinf(old_degrees), observed_degrees,
        1.0 / (1.0 / old_degrees + 1.0 / observed_degrees),
    )
    if np.any(degrees <= 0) or not np.all(np.isfinite(degrees)):
        raise ValueError("Barnard-Rubin degrees of freedom are not positive and finite")
    fmi = np.clip(
        (ratio + 2.0 / (degrees + 3.0)) / (ratio + 1.0), 0.0, 1.0,
    )
    critical = stats.t.ppf(0.975, degrees)
    statistic = pooled / se
    p_values = 2.0 * stats.t.sf(np.abs(statistic), degrees)
    half_width = critical * se
    return {
        "names": names, "n": sample_sizes[0], "m": m,
        "pooled": pooled, "se": se, "p": p_values,
        "lower": pooled - half_width, "upper": pooled + half_width,
        "within": np.maximum(within_diag, 0.0),
        "between": np.maximum(between_diag, 0.0),
        "lambda": lambda_missing, "fmi": fmi, "df": degrees,
        "complete_df": complete_df,
    }


def _rubin_maps(pooled: dict[str, Any]) -> dict[str, dict[str, float]]:
    names = pooled["names"]
    maps = {
        "estimates": dict(zip(names, pooled["pooled"], strict=True)),
        "standard_errors": dict(zip(names, pooled["se"], strict=True)),
        "p_values": dict(zip(names, pooled["p"], strict=True)),
        "ci_lower": dict(zip(names, pooled["lower"], strict=True)),
        "ci_upper": dict(zip(names, pooled["upper"], strict=True)),
        "metrics": {},
    }
    for index, name in enumerate(names):
        maps["metrics"][f"within#{name}"] = float(pooled["within"][index])
        maps["metrics"][f"between#{name}"] = float(pooled["between"][index])
        maps["metrics"][f"lambda#{name}"] = float(pooled["lambda"][index])
        maps["metrics"][f"fmi#{name}"] = float(pooled["fmi"][index])
        maps["metrics"][f"df#{name}"] = float(pooled["df"][index])
        maps["metrics"][f"complete_df#{name}"] = float(pooled["complete_df"])
    return maps


def _mice_trace_callback(mice_data: Any) -> list[float]:
    """Retain only per-variable means of currently imputed cells."""
    import numpy as np

    trace: list[float] = []
    for name in mice_data.data.columns:
        positions = np.asarray(mice_data.ix_miss[name], dtype=int)
        if positions.size:
            trace.append(float(np.mean(mice_data.data[name].iloc[positions])))
    return trace


def _trace_stability(history: Any, *, expected_imputations: int) -> float:
    """Split-trace standardized drift; a diagnostic, never convergence proof."""
    import numpy as np

    saved = list(history)
    # statsmodels invokes the callback once after update_all(burn_in), then
    # once after each retained MICE iteration.  Require that exact shape and
    # exclude the single post-burn snapshot, leaving retained draws only.
    if len(saved) != expected_imputations + 1:
        raise ValueError("MICE trace does not match the retained imputation count")
    trace = np.asarray(saved[1:], dtype=float)
    if trace.ndim != 2 or trace.shape[0] < 4 or trace.shape[1] < 1:
        raise ValueError("MICE trace requires at least four saved imputation iterations")
    split = trace.shape[0] // 2
    scale = np.std(trace, axis=0, ddof=1)
    difference = np.abs(trace[:split].mean(axis=0) - trace[split:].mean(axis=0))
    standardized = np.divide(
        difference, scale, out=np.zeros_like(difference), where=scale > 1e-12,
    )
    if not np.all(np.isfinite(standardized)):
        raise ValueError("MICE imputed-value trace is non-finite")
    return float(np.max(standardized))


def from_multiple_imputation(
    data: Any, *, formula: str, seed: int, burn_in: int,
    imputations: int = 20, matching_donors: int = 20,
    model_class: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Fit statsmodels MICE internally and emit verified Rubin-pooled inference.

    Fitting inside the helper binds the emitted seed, burn-in, donor count, and
    imputation count to the execution.  The caller cannot attach a plausible
    specification to an unrelated pre-fitted object.
    """
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.imputation.mice import MICE, MICEData

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty statsmodels formula")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    for value, field, minimum, maximum in (
        (burn_in, "burn_in", 1, 1000),
        (imputations, "imputations", 4, 50),
        (matching_donors, "matching_donors", 1, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    n, columns, missing_fraction, complete_rate, patterns, _ = _missingness_aggregates(data)
    if missing_fraction <= 0:
        raise ValueError("multiple imputation requires missing values")
    if model_class is None:
        model_class = sm.OLS
    if not str(getattr(model_class, "__module__", "")).startswith("statsmodels."):
        raise TypeError("model_class must be a statsmodels model class")
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        mice_data = MICEData(
            data.copy(), k_pmm=matching_donors,
            history_callback=_mice_trace_callback,
        )
        mice_model = MICE(formula, model_class, mice_data)
        fit = mice_model.fit(n_burnin=burn_in, n_imputations=imputations)
    finally:
        np.random.set_state(random_state)
    pooled = _rubin_pool(fit.model.results_list)
    public_cov = np.asarray(fit.cov_params(), dtype=float)
    total_diag = pooled["se"] ** 2
    if (not np.allclose(np.asarray(fit.params), pooled["pooled"], rtol=1e-8, atol=1e-10)
            or not np.allclose(np.diag(public_cov), total_diag, rtol=1e-8, atol=1e-10)):
        raise ValueError("MICE result does not match its retained imputation fits")
    trace_stability = _trace_stability(
        mice_data.history, expected_imputations=imputations,
    )
    maps = _rubin_maps(pooled)
    maps["metrics"].update({
        "parameter_count": float(len(pooled["names"])),
        "variable_count": float(columns), "missing_fraction": missing_fraction,
        "complete_case_rate": complete_rate,
        "missingness_pattern_count": float(patterns),
        "max_fraction_missing_information": float(np.max(pooled["fmi"])),
        "mean_fraction_missing_information": float(np.mean(pooled["fmi"])),
        "max_lambda_missing_information": float(np.max(pooled["lambda"])),
        "mean_between_imputation_variance": float(np.mean(pooled["between"])),
        "imputed_mean_trace_drift": trace_stability,
    })
    diag = _method_diagnostics({
        "missingness_pattern": "pass",
        "imputation_trace_stability": trace_stability,
        "between_imputation_variance": float(np.mean(pooled["between"])),
        "seed_recorded": "pass",
        "fraction_missing_information": float(np.max(pooled["fmi"])),
        "rubin_pooling": "pass",
    }, diagnostics)
    from_method(
        "multiple_imputation", n=n, diagnostics=diag,
        imputations=imputations, seed=seed, burn_in=burn_in,
        matching_donors=matching_donors,
        imputation_model="mice_predictive_mean_matching",
        uncertainty_type="multiple_imputation", **maps, **metadata,
    )


def from_mnar_sensitivity(
    data: Any, *, incomplete_outcome: str, formula: str, parameter: str,
    deltas: Any, seed: int, burn_in: int, imputations: int = 20,
    matching_donors: int = 20, model_class: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Internally fit and pool a delta-adjusted pattern-mixture grid."""
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from statsmodels.imputation.mice import MICEData

    if not isinstance(data, pd.DataFrame) or incomplete_outcome not in data.columns:
        raise ValueError("data must contain incomplete_outcome")
    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("formula must be a non-empty statsmodels formula")
    response = formula.split("~", 1)[0].strip() if formula.count("~") == 1 else ""
    if response != incomplete_outcome:
        raise ValueError("incomplete_outcome must be the formula response")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    for value, field, minimum, maximum in (
        (burn_in, "burn_in", 1, 1000),
        (imputations, "imputations", 4, 50),
        (matching_donors, "matching_donors", 1, 100),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ValueError(f"{field} must be an integer in [{minimum}, {maximum}]")
    if model_class is None:
        model_class = sm.OLS
    if not str(getattr(model_class, "__module__", "")).startswith("statsmodels."):
        raise TypeError("model_class must be a statsmodels model class")
    target = _method_quantity_name(parameter, field="parameter")
    deltas = sorted(float(value) for value in deltas)
    if (not all(math.isfinite(value) for value in deltas)
            or not 3 <= len(deltas) <= 21 or len(set(deltas)) != len(deltas)
            or not any(value < 0 for value in deltas)
            or 0.0 not in deltas or not any(value > 0 for value in deltas)):
        raise ValueError("delta grid must be unique, finite, and span negative, zero, positive")
    outcome_positions = np.flatnonzero(data[incomplete_outcome].isna().to_numpy())
    if not len(outcome_positions):
        raise ValueError("incomplete_outcome must contain missing values")
    if not pd.api.types.is_numeric_dtype(data[incomplete_outcome]):
        raise TypeError("delta adjustment requires a numeric incomplete outcome")
    scenario_fits: dict[float, list[Any]] = {delta: [] for delta in deltas}
    random_state = np.random.get_state()
    try:
        np.random.seed(seed)
        mice_data = MICEData(data.copy(), k_pmm=matching_donors)
        mice_data.update_all(burn_in)
        for _ in range(imputations):
            mice_data.update_all(1)
            baseline = mice_data.data.copy()
            for delta in deltas:
                completed = baseline.copy()
                column = completed.columns.get_loc(incomplete_outcome)
                completed.iloc[outcome_positions, column] += delta
                scenario_fits[delta].append(
                    model_class.from_formula(formula, completed).fit()
                )
    finally:
        np.random.set_state(random_state)
    pooled_scenarios = [_rubin_pool(scenario_fits[delta]) for delta in deltas]
    if len({item["n"] for item in pooled_scenarios}) != 1 or len(
        {item["m"] for item in pooled_scenarios}
    ) != 1:
        raise ValueError("all delta scenarios need identical n and imputation counts")
    values: list[float] = []
    ses: list[float] = []
    ps: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    fmis: list[float] = []
    for pooled in pooled_scenarios:
        if target not in pooled["names"]:
            raise ValueError(f"parameter {target!r} is absent from a scenario fit")
        index = pooled["names"].index(target)
        values.append(float(pooled["pooled"][index]))
        ses.append(float(pooled["se"][index]))
        ps.append(float(pooled["p"][index]))
        lowers.append(float(pooled["lower"][index]))
        uppers.append(float(pooled["upper"][index]))
        fmis.append(float(pooled["fmi"][index]))
    names = [f"scenario_{index + 1}" for index in range(len(deltas))]
    classifications = [
        -1 if upper < 0 else (1 if lower > 0 else 0)
        for lower, upper in zip(lowers, uppers, strict=True)
    ]
    stable = len(set(classifications)) == 1
    metrics: dict[str, float] = {
        "scenario_count": float(len(deltas)),
        "delta_min": min(deltas), "delta_max": max(deltas),
        "baseline_estimate": values[deltas.index(0.0)],
        "estimate_range": max(values) - min(values),
        "max_fraction_missing_information": max(fmis),
        "delta_applied_fraction": float(len(outcome_positions) / len(data)),
    }
    for index, name in enumerate(names):
        metrics[f"delta#{name}"] = deltas[index]
        metrics[f"fmi#{name}"] = fmis[index]
    diag = _method_diagnostics({
        "delta_grid": "pass", "baseline_included": "pass",
        "conclusion_stability": stable,
        # A statistical fit cannot establish a scientifically plausible delta.
        "sensitivity_parameter_justification": "warn",
    }, diagnostics)
    # Observed data cannot identify a scientifically plausible MNAR delta.
    # Until a researcher-bound structured justification exists, this cannot
    # be upgraded by a generic caller-supplied diagnostic override.
    diag["sensitivity_parameter_justification"] = "warn"
    from_method(
        "mnar_sensitivity", n=pooled_scenarios[0]["n"], diagnostics=diag,
        estimates=dict(zip(names, values, strict=True)),
        standard_errors=dict(zip(names, ses, strict=True)),
        p_values=dict(zip(names, ps, strict=True)),
        ci_lower=dict(zip(names, lowers, strict=True)),
        ci_upper=dict(zip(names, uppers, strict=True)), metrics=metrics,
        imputations=pooled_scenarios[0]["m"], seed=seed, burn_in=burn_in,
        matching_donors=matching_donors,
        imputation_model="mice_predictive_mean_matching",
        mnar_model="delta_adjusted_pattern_mixture",
        uncertainty_type="multiple_imputation", **metadata,
    )


# ---------------------------------------------------------------------------
# Predictive workflows — preprocessing is structurally nested in every split
# ---------------------------------------------------------------------------

def _predictive_bootstrap(
    observed: Any, predicted: Any, *, task: str, seed: int, replicates: int,
) -> tuple[float, float]:
    import numpy as np
    from sklearn.metrics import mean_squared_error, roc_auc_score

    y = np.asarray(observed)
    p = np.asarray(predicted)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        index = rng.integers(0, len(y), len(y))
        if task == "classification":
            if np.unique(y[index]).size != 2:
                continue
            values.append(float(roc_auc_score(y[index], p[index])))
        else:
            values.append(float(math.sqrt(mean_squared_error(y[index], p[index]))))
    if len(values) < max(100, int(replicates * 0.8)):
        raise ValueError("held-out bootstrap has insufficient valid replicates")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def from_predictive_workflow(
    X: Any, y: Any, *, task: str, evaluation: str = "train_validation_test",
    estimator: Any | None = None, preprocessor: Any | None = None,
    seed: int = 42, folds: int = 5, train_fraction: float = 0.6,
    validation_fraction: float = 0.2, bootstrap_replicates: int = 500,
    imbalance_strategy: str = "balanced_weight", calibrate: bool = True,
    **metadata: Any,
) -> None:
    """Fit a leakage-resistant sklearn predictive workflow and emit OOS metrics.

    Preprocessing is always a step inside a cloned ``Pipeline``.  Classification
    calibration is nested inside the training data (and inside every outer CV
    fold), so neither transformations nor calibration see evaluation targets.
    """
    import numpy as np
    import pandas as pd
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        average_precision_score, balanced_accuracy_score, brier_score_loss,
        mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
    )
    from sklearn.model_selection import (
        KFold, StratifiedKFold, cross_val_predict, train_test_split,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task not in {"regression", "classification"}:
        raise ValueError("task must be regression or classification")
    reserved = {
        "type", "method_id", "n", "diagnostics", "estimates", "metrics",
        "evaluation_split", "split_strategy", "seed", "_via_helper",
        "baseline_model", "calibration_method", "imbalance_strategy",
        "bootstrap_replicates", "training_observations",
        "validation_observations", "test_observations",
        "evaluated_observations", "folds", "uncertainty_type",
        "ci_lower", "ci_upper", "interval_method",
    }
    if reserved.intersection(metadata):
        raise ValueError("metadata cannot override predictive workflow fields")
    if evaluation not in {"train_validation_test", "cross_validation"}:
        raise ValueError("evaluation must be train_validation_test or cross_validation")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if isinstance(folds, bool) or not isinstance(folds, int) or not 3 <= folds <= 10:
        raise ValueError("folds must be an integer in [3, 10]")
    if (isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int)
            or not 200 <= bootstrap_replicates <= 2000):
        raise ValueError("bootstrap_replicates must be an integer in [200, 2000]")
    if isinstance(X, pd.DataFrame):
        if X.columns.has_duplicates:
            raise ValueError("X column names must be unique")
        matrix = X.copy()
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in matrix.dtypes):
            raise TypeError("predictive workflow currently requires numeric features")
    else:
        matrix = np.asarray(X, dtype=float)
    target = np.asarray(y).reshape(-1)
    if getattr(matrix, "ndim", None) != 2 or len(matrix) != len(target):
        raise ValueError("X and y must be 2D/1D with matching rows")
    n, feature_count = matrix.shape
    if n < 100 or not 1 <= feature_count <= 100:
        raise ValueError("predictive workflow requires >=100 rows and 1..100 features")
    if task == "regression" and not np.all(np.isfinite(target.astype(float))):
        raise ValueError("y must be finite")
    if preprocessor is None:
        preprocessor = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    if task == "classification":
        if not calibrate:
            raise ValueError("classification workflow requires nested probability calibration")
        try:
            numeric_target = target.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("binary classification requires 0/1 labels") from exc
        classes, counts = np.unique(numeric_target, return_counts=True)
        if (not np.all(np.isfinite(numeric_target)) or len(classes) != 2
                or set(classes) != {0.0, 1.0} or int(counts.min()) < 50):
            raise ValueError("binary classification requires 0/1 labels and at least 50 rows per class")
        target = numeric_target.astype(int)
        minority_fraction = float(counts.min() / n)
        if imbalance_strategy not in {"balanced_weight", "none"}:
            raise ValueError("imbalance_strategy must be balanced_weight or none")
        if minority_fraction < 0.4 and imbalance_strategy != "balanced_weight":
            raise ValueError("imbalanced classification requires balanced_weight")
        if estimator is None:
            estimator = LogisticRegression(
                max_iter=2000, random_state=seed, solver="liblinear",
            )
        if imbalance_strategy == "balanced_weight":
            parameters = estimator.get_params(deep=False)
            if "class_weight" not in parameters:
                raise TypeError("balanced_weight requires an estimator with class_weight")
            estimator = clone(estimator).set_params(class_weight="balanced")
    else:
        minority_fraction = 0.0
        if estimator is None:
            estimator = Ridge(alpha=1.0)
        imbalance_strategy = "not_applicable"
        calibrate = False
    pipeline = Pipeline([
        ("preprocess", clone(preprocessor)), ("model", clone(estimator)),
    ])
    workflow: Any = pipeline
    calibration_method = "not_applicable"
    if task == "classification" and calibrate:
        workflow = CalibratedClassifierCV(
            estimator=pipeline, method="sigmoid", cv=3,
        )
        calibration_method = "nested_sigmoid"

    indices = np.arange(n)
    validation_metrics: dict[str, float] = {}
    if evaluation == "train_validation_test":
        safe_train_fraction = _safe_float(train_fraction)
        safe_validation_fraction = _safe_float(validation_fraction)
        if (safe_train_fraction is None or safe_validation_fraction is None
                or not 0.4 <= safe_train_fraction <= 0.8
                or not 0.1 <= safe_validation_fraction <= 0.3
                or safe_train_fraction + safe_validation_fraction > 0.9):
            raise ValueError("train/validation fractions leave 10%+ for test")
        stratify = target if task == "classification" else None
        train_index, remainder = train_test_split(
            indices, train_size=safe_train_fraction,
            random_state=seed, stratify=stratify,
        )
        remainder_target = target[remainder] if task == "classification" else None
        relative_validation = safe_validation_fraction / (1.0 - safe_train_fraction)
        validation_index, test_index = train_test_split(
            remainder, train_size=relative_validation, random_state=seed + 1,
            stratify=remainder_target,
        )
        # Validation is a diagnostic holdout for this single declared model,
        # not evidence of hyperparameter tuning or model selection.
        training_X = (
            matrix.iloc[train_index]
            if isinstance(matrix, pd.DataFrame) else matrix[train_index]
        )
        diagnostic_fit = clone(workflow).fit(training_X, target[train_index])
        validation_X = matrix.iloc[validation_index] if isinstance(matrix, pd.DataFrame) else matrix[validation_index]
        if task == "classification":
            validation_probability = diagnostic_fit.predict_proba(validation_X)[:, 1]
            validation_metrics = {
                "validation_auc": float(roc_auc_score(target[validation_index], validation_probability)),
                "validation_brier": float(brier_score_loss(target[validation_index], validation_probability)),
            }
        else:
            validation_prediction = diagnostic_fit.predict(validation_X)
            validation_metrics = {
                "validation_rmse": float(math.sqrt(mean_squared_error(
                    target[validation_index], validation_prediction,
                ))),
            }
        development_index = np.concatenate([train_index, validation_index])
        test_X = matrix.iloc[test_index] if isinstance(matrix, pd.DataFrame) else matrix[test_index]
        final_fit = clone(workflow).fit(
            matrix.iloc[development_index] if isinstance(matrix, pd.DataFrame) else matrix[development_index],
            target[development_index],
        )
        if task == "classification":
            predictions = final_fit.predict_proba(test_X)[:, 1]
            baseline = DummyClassifier(strategy="prior").fit(
                np.zeros((len(development_index), 1)), target[development_index],
            ).predict_proba(np.zeros((len(test_index), 1)))[:, 1]
        else:
            predictions = final_fit.predict(test_X)
            baseline = DummyRegressor(strategy="mean").fit(
                np.zeros((len(development_index), 1)), target[development_index],
            ).predict(np.zeros((len(test_index), 1)))
        observed = target[test_index]
        split_metadata = {
            "training_observations": int(len(train_index)),
            "validation_observations": int(len(validation_index)),
            "test_observations": int(len(test_index)),
        }
        evaluation_split = "held_out"
    else:
        splitter: Any = (
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            if task == "classification"
            else KFold(n_splits=folds, shuffle=True, random_state=seed)
        )
        method = "predict_proba" if task == "classification" else "predict"
        predictions = cross_val_predict(
            clone(workflow), matrix, target, cv=splitter, method=method,
        )
        if task == "classification":
            predictions = predictions[:, 1]
            baseline_model = DummyClassifier(strategy="prior")
            baseline = cross_val_predict(
                baseline_model, matrix, target, cv=splitter, method="predict_proba",
            )[:, 1]
        else:
            baseline = cross_val_predict(
                DummyRegressor(strategy="mean"), matrix, target, cv=splitter,
            )
        observed = target
        split_metadata = {"evaluated_observations": n, "folds": folds}
        evaluation_split = "cross_validation"

    metrics: dict[str, float] = {
        "feature_count": float(feature_count), **validation_metrics,
    }
    estimates: dict[str, float]
    ci_lower: dict[str, float] | None = None
    ci_upper: dict[str, float] | None = None
    uncertainty_type: str | None = None
    if task == "classification":
        auc = float(roc_auc_score(observed, predictions))
        average_precision = float(average_precision_score(observed, predictions))
        predicted_class = (predictions >= 0.5).astype(int)
        brier = float(brier_score_loss(observed, predictions))
        baseline_brier = float(brier_score_loss(observed, baseline))
        baseline_auc = float(roc_auc_score(observed, baseline))
        clipped = np.clip(predictions, 1e-6, 1 - 1e-6)
        logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        calibration_fit = LogisticRegression(
            C=1e6, solver="liblinear",
        ).fit(logit, observed)
        calibration_slope = float(calibration_fit.coef_[0, 0])
        calibration_intercept = float(calibration_fit.intercept_[0])
        estimates = {"roc_auc": auc}
        metrics.update({
            "roc_auc": auc, "average_precision": average_precision,
            "balanced_accuracy": float(balanced_accuracy_score(observed, predicted_class)),
            "brier_score": brier, "baseline_brier_score": baseline_brier,
            "baseline_auc": baseline_auc, "calibration_slope": calibration_slope,
            "calibration_intercept": calibration_intercept,
            "minority_fraction": minority_fraction,
        })
        primary = auc
        baseline_primary = baseline_auc
    else:
        rmse = float(math.sqrt(mean_squared_error(observed, predictions)))
        baseline_rmse = float(math.sqrt(mean_squared_error(observed, baseline)))
        if float(np.std(predictions)) <= 1e-12:
            raise ValueError("regression calibration is undefined for constant predictions")
        slope, intercept = np.polyfit(predictions, observed, 1)
        estimates = {"rmse": rmse}
        metrics.update({
            "rmse": rmse, "mae": float(mean_absolute_error(observed, predictions)),
            "r2": float(r2_score(observed, predictions)),
            "baseline_rmse": baseline_rmse,
            "baseline_mae": float(mean_absolute_error(observed, baseline)),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
        })
        primary = -rmse
        baseline_primary = -baseline_rmse
    metrics["baseline_improvement"] = primary - baseline_primary
    if evaluation == "train_validation_test":
        low, high = _predictive_bootstrap(
            observed, predictions, task=task, seed=seed + 2,
            replicates=bootstrap_replicates,
        )
        key = "roc_auc" if task == "classification" else "rmse"
        ci_lower, ci_upper = {key: low}, {key: high}
        uncertainty_type = "bootstrap"
    calibration_ok = (
        0.8 <= metrics["calibration_slope"] <= 1.2
        and (task == "regression" or metrics["brier_score"] <= metrics["baseline_brier_score"])
    )
    diagnostics: dict[str, Any] = {
        "held_out_performance": "pass",
        "baseline_comparison": "pass" if metrics["baseline_improvement"] >= 0 else "warn",
        "calibration": "pass" if calibration_ok else "warn",
        "split_integrity": "pass", "preprocessing_inside_split": "pass",
        "uncertainty": "pass" if evaluation == "train_validation_test" else "not_applicable",
    }
    method_id = "predictive_regression"
    if task == "classification":
        method_id = "predictive_classification"
        diagnostics.update({
            "discrimination": auc, "class_balance": "pass",
        })
    payload: dict[str, Any] = {
        "type": "method_result", "method_id": method_id, "n": n,
        "diagnostics": diagnostics, "estimates": estimates, "metrics": metrics,
        "evaluation_split": evaluation_split, "split_strategy": evaluation,
        "seed": seed, "baseline_model": "simple_dummy",
        "calibration_method": calibration_method,
        "imbalance_strategy": imbalance_strategy,
        "bootstrap_replicates": (
            bootstrap_replicates if evaluation == "train_validation_test" else 0
        ),
        "_via_helper": "predictive_workflow_v1", **split_metadata, **metadata,
    }
    if ci_lower is not None:
        payload.update(ci_lower=ci_lower, ci_upper=ci_upper)
    if uncertainty_type is not None:
        payload["uncertainty_type"] = uncertainty_type
        # This resamples untouched evaluation cases while holding the fitted
        # workflow fixed. It is not model-refit uncertainty.
        payload["interval_method"] = "heldout_case_bootstrap"
    _write_result(payload)


# ---------------------------------------------------------------------------
# Exact panel / two-period DiD / calibration workflows
# ---------------------------------------------------------------------------

def _bounded_tabular_matrix(X: Any, *, minimum_rows: int = 30) -> tuple[Any, list[str]]:
    """Return a finite, modest-width numeric matrix and safe coefficient names."""
    import numpy as np
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        if X.columns.has_duplicates:
            raise ValueError("X column names must be unique")
        if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise TypeError("X must contain only numeric predictors")
        names = [_method_quantity_name(value, field="predictor name") for value in X.columns]
        matrix = X.to_numpy(dtype=float)
    else:
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        names = [f"x{index}" for index in range(1, matrix.shape[1] + 1)] if matrix.ndim == 2 else []
    if (matrix.ndim != 2 or len(matrix) < minimum_rows
            or not 1 <= matrix.shape[1] <= 50 or not np.all(np.isfinite(matrix))):
        raise ValueError(
            f"X must be a finite matrix with at least {minimum_rows} rows and 1..50 columns"
        )
    return matrix, names


def _panel_identifiers(entity: Any, time: Any, *, n: int) -> tuple[Any, Any, int, int]:
    """Validate and integer-code an exactly balanced, unique panel."""
    import numpy as np
    import pandas as pd

    entity_values = np.asarray(entity, dtype=object).reshape(-1)
    time_values = np.asarray(time, dtype=object).reshape(-1)
    if len(entity_values) != n or len(time_values) != n:
        raise ValueError("entity, time, outcome, and X must have matching rows")
    if pd.isna(entity_values).any() or pd.isna(time_values).any():
        raise ValueError("entity and time identifiers cannot be missing")
    entity_codes, entity_levels = pd.factorize(entity_values, sort=True)
    time_codes, time_levels = pd.factorize(time_values, sort=True)
    entities, periods = len(entity_levels), len(time_levels)
    if entities < 10 or periods < 2:
        raise ValueError("panel workflows require at least 10 entities and 2 periods")
    pairs = set(zip(entity_codes.tolist(), time_codes.tolist()))
    if len(pairs) != n:
        raise ValueError("each entity-time cell must contain exactly one observation")
    expected = {(unit, period) for unit in range(entities) for period in range(periods)}
    if pairs != expected:
        raise ValueError("the typed panel workflow requires a balanced panel")
    return entity_codes, time_codes, entities, periods


def from_panel_fixed_effects(
    outcome: Any, X: Any, entity: Any, time: Any, **metadata: Any,
) -> None:
    """Fit a balanced-panel entity fixed-effects model with clustered covariance.

    The fit uses the exact within-entity transformation and statsmodels OLS.  It
    deliberately supports entity effects only; generated code must not describe
    the result as two-way fixed effects.  Entity and time identifiers never leave
    this helper.
    """
    import numpy as np
    import statsmodels.api as sm
    from scipy.stats import f as f_distribution

    reserved = {
        "type", "method_id", "n", "diagnostics", "estimates",
        "standard_errors", "p_values", "ci_lower", "ci_upper", "metrics",
        "clusters", "records", "uncertainty_type", "design", "_via_helper",
    }
    if reserved.intersection(metadata):
        raise ValueError("metadata cannot override panel fixed-effects fields")
    matrix, names = _bounded_tabular_matrix(X)
    response = np.asarray(outcome, dtype=float).reshape(-1)
    if len(response) != len(matrix) or not np.all(np.isfinite(response)):
        raise ValueError("outcome must be finite and match X rows")
    n, predictors = matrix.shape
    groups, _times, entity_count, periods = _panel_identifiers(entity, time, n=n)
    if n - entity_count - predictors <= 0:
        raise ValueError("panel has insufficient residual degrees of freedom")

    within_y = response.copy()
    within_X = matrix.copy()
    for group in range(entity_count):
        mask = groups == group
        within_y[mask] -= float(np.mean(response[mask]))
        within_X[mask] -= np.mean(matrix[mask], axis=0)
    total_variance = np.sum((matrix - np.mean(matrix, axis=0)) ** 2, axis=0)
    within_variance = np.sum(within_X ** 2, axis=0)
    if np.any(total_variance <= 0):
        raise ValueError("every predictor must have non-zero total variation")
    ratios = within_variance / total_variance
    if np.any(ratios <= 1e-8) or np.linalg.matrix_rank(within_X) != predictors:
        raise ValueError("every predictor must have identified within-entity variation")

    fit = sm.OLS(within_y, within_X).fit(
        cov_type="cluster", cov_kwds={"groups": groups, "use_correction": True},
    )
    retained_groups = fit.cov_kwds.get("groups") if isinstance(fit.cov_kwds, dict) else None
    if fit.cov_type != "cluster" or retained_groups is None or not np.array_equal(retained_groups, groups):
        raise RuntimeError("statsmodels did not retain entity-clustered covariance")
    parameters = np.asarray(fit.params, dtype=float)
    errors = np.asarray(fit.bse, dtype=float)
    p_values = np.asarray(fit.pvalues, dtype=float)
    intervals = np.asarray(fit.conf_int(alpha=0.05), dtype=float)
    if not all(np.all(np.isfinite(value)) for value in (parameters, errors, p_values, intervals)):
        raise ValueError("fixed-effects fit produced non-finite inference")

    pooled = sm.OLS(response, sm.add_constant(matrix, has_constant="add")).fit()
    rss_pooled = float(np.sum(np.asarray(pooled.resid) ** 2))
    rss_fixed = float(np.sum(np.asarray(fit.resid) ** 2))
    numerator_df = entity_count - 1
    denominator_df = n - entity_count - predictors
    improvement = rss_pooled - rss_fixed
    tolerance = 1e-8 * max(1.0, rss_pooled)
    if improvement < -tolerance or rss_fixed <= 0:
        raise ValueError("fixed-effect comparison is numerically invalid")
    fixed_f = max(0.0, improvement) / numerator_df / (rss_fixed / denominator_df)
    fixed_p = float(f_distribution.sf(fixed_f, numerator_df, denominator_df))
    within_tss = float(np.sum(within_y ** 2))
    within_r2 = 1.0 - rss_fixed / within_tss if within_tss > 0 else 0.0

    estimates = dict(zip(names, parameters.tolist()))
    payload = {
        **metadata,
        "type": "method_result", "method_id": "panel_fixed_effects", "n": n,
        "diagnostics": {
            "cluster_count": entity_count, "cluster_size": periods,
            "convergence": True, "balanced_panel": "pass",
            "within_variation": float(np.min(ratios)),
            "fixed_effect_test": fixed_p, "clustered_uncertainty": "pass",
        },
        "estimates": estimates,
        "standard_errors": dict(zip(names, errors.tolist())),
        "p_values": dict(zip(names, p_values.tolist())),
        "ci_lower": dict(zip(names, intervals[:, 0].tolist())),
        "ci_upper": dict(zip(names, intervals[:, 1].tolist())),
        "metrics": {
            "within_r_squared": within_r2,
            "min_within_variation_ratio": float(np.min(ratios)),
            "fixed_effect_f_statistic": fixed_f,
            "fixed_effect_p_value": fixed_p,
            "entity_count": float(entity_count), "period_count": float(periods),
            "predictor_count": float(predictors),
        },
        "clusters": entity_count, "records": n,
        "uncertainty_type": "cluster_robust",
        "design": "panel_entity_fixed_effects",
        "_via_helper": "panel_fixed_effects_v1",
    }
    _write_result(payload)


def from_difference_in_differences(
    outcome: Any, treated_group: Any, post_period: Any,
    panel_id: Any, **metadata: Any,
) -> None:
    """Fit the deliberately narrow balanced two-period panel DiD estimand."""
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    reserved = {
        "type", "method_id", "n", "diagnostics", "estimates",
        "standard_errors", "p_values", "ci_lower", "ci_upper", "metrics",
        "clusters", "records", "treated", "controls", "uncertainty_type",
        "estimand", "design", "_via_helper",
    }
    if reserved.intersection(metadata):
        raise ValueError("metadata cannot override difference-in-differences fields")
    y = np.asarray(outcome, dtype=float).reshape(-1)
    group = np.asarray(treated_group).reshape(-1)
    post = np.asarray(post_period).reshape(-1)
    ids = np.asarray(panel_id, dtype=object).reshape(-1)
    n = len(y)
    if n < 40 or any(len(value) != n for value in (group, post, ids)):
        raise ValueError("two-period DiD requires at least 40 matching records")
    if not np.all(np.isfinite(y)) or pd.isna(ids).any():
        raise ValueError("outcomes and panel identifiers must be complete")
    try:
        group = group.astype(float)
        post = post.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("treated_group and post_period must be binary 0/1") from exc
    if (not np.all(np.isfinite(group)) or not np.all(np.isfinite(post))
            or set(np.unique(group)) != {0.0, 1.0}
            or set(np.unique(post)) != {0.0, 1.0}):
        raise ValueError("treated_group and post_period must be binary 0/1")
    group = group.astype(int); post = post.astype(int)
    entity_codes, levels = pd.factorize(ids, sort=True)
    entities = len(levels)
    if entities * 2 != n:
        raise ValueError("two-period DiD requires exactly one pre and one post row per entity")
    entity_treatment: list[int] = []
    for entity_index in range(entities):
        mask = entity_codes == entity_index
        if int(mask.sum()) != 2 or set(post[mask]) != {0, 1} or len(set(group[mask])) != 1:
            raise ValueError("treatment must be entity-invariant with one pre and one post row")
        entity_treatment.append(int(group[mask][0]))
    treated_entities = int(sum(entity_treatment))
    control_entities = entities - treated_entities
    if min(treated_entities, control_entities) < 10:
        raise ValueError("DiD requires at least 10 treated and 10 control entities")

    interaction = group * post
    design_matrix = np.column_stack([np.ones(n), group, post, interaction])
    fit = sm.OLS(y, design_matrix).fit(
        cov_type="cluster", cov_kwds={"groups": entity_codes, "use_correction": True},
    )
    retained_groups = fit.cov_kwds.get("groups") if isinstance(fit.cov_kwds, dict) else None
    if fit.cov_type != "cluster" or retained_groups is None or not np.array_equal(retained_groups, entity_codes):
        raise RuntimeError("statsmodels did not retain entity-clustered covariance")
    cell_means = {
        (arm, period): float(np.mean(y[(group == arm) & (post == period)]))
        for arm in (0, 1) for period in (0, 1)
    }
    raw_did = ((cell_means[(1, 1)] - cell_means[(1, 0)])
               - (cell_means[(0, 1)] - cell_means[(0, 0)]))
    att = float(fit.params[3])
    if not math.isclose(att, raw_did, rel_tol=1e-9, abs_tol=1e-9):
        raise RuntimeError("maintained DiD fit disagrees with the two-by-two contrast")
    standard_error = float(fit.bse[3]); p_value = float(fit.pvalues[3])
    interval = np.asarray(fit.conf_int(alpha=0.05), dtype=float)[3]
    if not all(math.isfinite(value) for value in (att, standard_error, p_value, *interval)):
        raise ValueError("DiD fit produced non-finite inference")

    payload = {
        **metadata,
        "type": "method_result", "method_id": "difference_in_differences", "n": n,
        "diagnostics": {
            "parallel_pretrends": "not_applicable",
            "treatment_timing": "pass", "balanced_two_period_panel": "pass",
            "clustered_uncertainty": "pass", "effect_uncertainty": "pass",
            "design_specific_falsification": "not_applicable",
        },
        "estimates": {"att": att}, "standard_errors": {"att": standard_error},
        "p_values": {"att": p_value}, "ci_lower": {"att": float(interval[0])},
        "ci_upper": {"att": float(interval[1])},
        "metrics": {
            "control_pre_mean": cell_means[(0, 0)],
            "control_post_mean": cell_means[(0, 1)],
            "treated_pre_mean": cell_means[(1, 0)],
            "treated_post_mean": cell_means[(1, 1)],
            "raw_did": raw_did, "entity_count": float(entities),
            "period_count": 2.0,
        },
        "clusters": entities, "records": n,
        "treated": treated_entities, "controls": control_entities,
        "uncertainty_type": "cluster_robust", "estimand": "att",
        "design": "two_by_two_panel_did",
        "_via_helper": "difference_in_differences_v1",
    }
    _write_result(payload)


def from_probability_calibration(
    X: Any, y: Any, *, estimator: Any | None = None,
    preprocessor: Any | None = None, seed: int = 42, test_fraction: float = 0.2,
    calibration_folds: int = 5, bootstrap_replicates: int = 500,
    imbalance_strategy: str = "balanced_weight", **metadata: Any,
) -> None:
    """Fit nested sigmoid calibration and release held-out aggregates only."""
    import numpy as np
    import statsmodels.api as sm
    from sklearn.base import clone
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    reserved = {
        "type", "method_id", "n", "diagnostics", "estimates", "metrics",
        "evaluation_split", "split_strategy", "seed", "_via_helper",
        "baseline_model", "calibration_method", "imbalance_strategy",
        "bootstrap_replicates", "training_observations", "test_observations",
        "uncertainty_type", "ci_lower", "ci_upper", "interval_method", "folds",
    }
    if reserved.intersection(metadata):
        raise ValueError("metadata cannot override probability-calibration fields")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2^32)")
    if (isinstance(calibration_folds, bool) or not isinstance(calibration_folds, int)
            or not 3 <= calibration_folds <= 10):
        raise ValueError("calibration_folds must be an integer in [3, 10]")
    if (isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int)
            or not 200 <= bootstrap_replicates <= 2000):
        raise ValueError("bootstrap_replicates must be an integer in [200, 2000]")
    safe_test_fraction = _safe_float(test_fraction)
    if safe_test_fraction is None or not 0.15 <= safe_test_fraction <= 0.3:
        raise ValueError("test_fraction must be in [0.15, 0.30]")
    matrix, _names = _bounded_tabular_matrix(X, minimum_rows=100)
    target = np.asarray(y).reshape(-1)
    if len(target) != len(matrix):
        raise ValueError("X and y must have matching rows")
    try:
        numeric_target = target.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probability calibration requires 0/1 labels") from exc
    classes, counts = np.unique(numeric_target, return_counts=True)
    if (not np.all(np.isfinite(numeric_target)) or set(classes) != {0.0, 1.0}
            or int(counts.min()) < 50):
        raise ValueError("probability calibration requires 0/1 labels and at least 50 rows per class")
    target = numeric_target.astype(int)
    minority_fraction = float(counts.min() / len(target))
    if imbalance_strategy not in {"balanced_weight", "none"}:
        raise ValueError("imbalance_strategy must be balanced_weight or none")
    if minority_fraction < 0.4 and imbalance_strategy != "balanced_weight":
        raise ValueError("imbalanced calibration requires balanced_weight")
    if preprocessor is None:
        preprocessor = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    if estimator is None:
        estimator = LogisticRegression(max_iter=2000, solver="liblinear", random_state=seed)
    if imbalance_strategy == "balanced_weight":
        parameters = estimator.get_params(deep=False)
        if "class_weight" not in parameters:
            raise TypeError("balanced_weight requires an estimator with class_weight")
        estimator = clone(estimator).set_params(class_weight="balanced")
    pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", clone(estimator))])

    indices = np.arange(len(target))
    development, test = train_test_split(
        indices, test_size=safe_test_fraction, random_state=seed, stratify=target,
    )
    development_X = matrix[development]
    test_X = matrix[test]
    uncalibrated = clone(pipeline).fit(development_X, target[development])
    if not hasattr(uncalibrated, "predict_proba"):
        raise TypeError("probability calibration requires an estimator with predict_proba")
    calibrated = CalibratedClassifierCV(
        estimator=clone(pipeline), method="sigmoid", cv=calibration_folds,
    ).fit(development_X, target[development])
    uncalibrated_probability = np.asarray(uncalibrated.predict_proba(test_X))[:, 1]
    probability = np.asarray(calibrated.predict_proba(test_X))[:, 1]
    observed = target[test]
    if (not np.all(np.isfinite(probability)) or not np.all(np.isfinite(uncalibrated_probability))
            or np.any((probability < 0) | (probability > 1))):
        raise ValueError("calibration produced invalid held-out probabilities")

    brier = float(brier_score_loss(observed, probability))
    uncalibrated_brier = float(brier_score_loss(observed, uncalibrated_probability))
    prevalence = float(np.mean(target[development]))
    prevalence_brier = float(brier_score_loss(observed, np.full(len(test), prevalence)))
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped))
    calibration_fit = sm.Logit(observed, sm.add_constant(logits)).fit(disp=False)
    calibration_intercept = float(calibration_fit.params[0])
    calibration_slope = float(calibration_fit.params[1])
    if not math.isfinite(calibration_intercept) or not math.isfinite(calibration_slope):
        raise ValueError("held-out calibration slope fit is not finite")

    bin_count = min(10, max(2, len(test) // 10))
    # Equal-frequency rank bins keep every released aggregate supported by
    # roughly ten held-out cases. Stable sorting makes tied probabilities
    # deterministic without releasing either ranks or bin rows.
    order = np.argsort(probability, kind="mergesort")
    assignments: Any = np.empty(len(test), dtype=int)
    assignments[order] = np.minimum(
        np.arange(len(test)) * bin_count // len(test), bin_count - 1,
    )
    gaps: list[float] = []
    supports: list[int] = []
    for index in range(bin_count):
        mask = assignments == index
        if not np.any(mask):
            continue
        supports.append(int(mask.sum()))
        gaps.append(abs(float(np.mean(observed[mask])) - float(np.mean(probability[mask]))))
    ece = float(sum(count * gap for count, gap in zip(supports, gaps)) / len(test))
    max_gap = float(max(gaps))
    nonempty_bins = len(gaps)
    min_bin_count = min(supports)

    rng = np.random.default_rng(seed)
    bootstrap_values: list[float] = []
    for _ in range(bootstrap_replicates):
        sample = rng.integers(0, len(test), len(test))
        bootstrap_values.append(float(brier_score_loss(observed[sample], probability[sample])))
    interval_quantiles: Any = np.quantile(bootstrap_values, [0.025, 0.975])
    lower = float(interval_quantiles[0])
    upper = float(interval_quantiles[1])
    calibration_ok = (
        abs(calibration_intercept) <= 0.5 and 0.8 <= calibration_slope <= 1.2
    )
    metrics = {
        "feature_count": float(matrix.shape[1]), "brier_score": brier,
        "uncalibrated_brier_score": uncalibrated_brier,
        "prevalence_brier_score": prevalence_brier,
        "uncalibrated_brier_improvement": uncalibrated_brier - brier,
        "baseline_improvement": prevalence_brier - brier,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
        "expected_calibration_error": ece,
        "max_calibration_gap": max_gap,
        "nonempty_calibration_bins": float(nonempty_bins),
        "minimum_calibration_bin_count": float(min_bin_count),
        "minority_fraction": minority_fraction,
        "roc_auc_context": float(roc_auc_score(observed, probability)),
    }
    payload = {
        **metadata,
        "type": "method_result", "method_id": "probability_calibration", "n": len(target),
        "diagnostics": {
            "held_out_performance": "pass",
            "baseline_comparison": "pass" if metrics["baseline_improvement"] >= 0 else "warn",
            "calibration": "pass" if calibration_ok else "warn",
            "calibration_curve": ece, "brier_score": brier,
            "split_integrity": "pass", "preprocessing_inside_split": "pass",
            "calibration_nested": "pass", "class_balance": "pass",
            "uncertainty": "pass",
        },
        "estimates": {"brier_score": brier}, "metrics": metrics,
        "ci_lower": {"brier_score": lower},
        "ci_upper": {"brier_score": upper},
        "evaluation_split": "held_out", "split_strategy": "train_test_calibration_cv",
        "seed": seed, "folds": calibration_folds,
        "baseline_model": "uncalibrated_classifier_and_prevalence",
        "calibration_method": "nested_sigmoid", "imbalance_strategy": imbalance_strategy,
        "training_observations": int(len(development)), "test_observations": int(len(test)),
        "bootstrap_replicates": bootstrap_replicates, "uncertainty_type": "bootstrap",
        "interval_method": "heldout_case_bootstrap",
        "_via_helper": "probability_calibration_v1",
    }
    _write_result(payload)


# ---------------------------------------------------------------------------
# Causal-design helpers — aggregate outputs only
# ---------------------------------------------------------------------------

def _causal_status(value: Any, *, field: str) -> Any:
    if value not in {"pass", "warn", "fail", "not_applicable", True, False}:
        raise ValueError(f"{field} must be a diagnostic status")
    return value


def _causal_arrays(X: Any, treatment: Any, outcome: Any) -> tuple[Any, Any, Any]:
    import numpy as np

    matrix = np.asarray(X, dtype=float)
    assigned = np.asarray(treatment, dtype=int).reshape(-1)
    response = np.asarray(outcome, dtype=float).reshape(-1)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or len(matrix) != len(assigned) or len(matrix) != len(response):
        raise ValueError("X, treatment, and outcome must have the same row count")
    if len(matrix) < 20 or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(response)):
        raise ValueError("causal design requires at least 20 complete finite rows")
    if set(np.unique(assigned)) != {0, 1}:
        raise ValueError("treatment must be binary with both 0 and 1 present")
    if min(int(assigned.sum()), int((1 - assigned).sum())) < 10:
        raise ValueError("each treatment arm requires at least 10 observations")
    return matrix, assigned, response


def _max_abs_smd(X: Any, treatment: Any, weights: Any | None = None) -> float:
    import numpy as np

    X = np.asarray(X, dtype=float)
    treatment = np.asarray(treatment, dtype=int)
    if weights is None:
        weights = np.ones(len(treatment), dtype=float)
    weights = np.asarray(weights, dtype=float)
    values: list[float] = []
    for column in range(X.shape[1]):
        x = X[:, column]
        group_stats: list[tuple[float, float]] = []
        for arm in (1, 0):
            mask = treatment == arm
            w = weights[mask]
            z = x[mask]
            mean = float(np.average(z, weights=w))
            var = float(np.average((z - mean) ** 2, weights=w))
            group_stats.append((mean, var))
        scale = math.sqrt(max((group_stats[0][1] + group_stats[1][1]) / 2.0, 0.0))
        difference = abs(group_stats[0][0] - group_stats[1][0])
        values.append(0.0 if scale == 0 and difference == 0 else difference / max(scale, 1e-12))
    return float(max(values, default=0.0))


def _propensity_scores(X: Any, treatment: Any, *, seed: int) -> Any:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=2000, random_state=seed)
    model.fit(X, treatment)
    return model.predict_proba(X)[:, 1]


def _overlap_fraction(scores: Any, treatment: Any) -> float:
    import numpy as np

    treated = scores[treatment == 1]
    control = scores[treatment == 0]
    lower = max(float(np.min(treated)), float(np.min(control)))
    upper = min(float(np.max(treated)), float(np.max(control)))
    if lower >= upper:
        return 0.0
    return float(np.mean((scores >= lower) & (scores <= upper)))


def from_propensity_matching(
    X: Any, treatment: Any, outcome: Any, *, estimand: str = "att",
    falsification_status: Any, seed: int = 42, **metadata: Any,
) -> None:
    """Fit propensity scores and emit 1:1 nearest-neighbour ATT matching.

    Matching is with replacement on the scalar fitted propensity score.  Only
    aggregate effects and design diagnostics leave the sandbox; match indices,
    individual scores, outcomes and covariate rows are never emitted.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    if estimand != "att":
        raise ValueError("propensity nearest-neighbour matching currently identifies ATT only")
    X, treatment, outcome = _causal_arrays(X, treatment, outcome)
    scores = _propensity_scores(X, treatment, seed=seed)
    treated_idx = np.flatnonzero(treatment == 1)
    control_idx = np.flatnonzero(treatment == 0)
    matcher = NearestNeighbors(n_neighbors=1).fit(scores[control_idx].reshape(-1, 1))
    nearest = matcher.kneighbors(scores[treated_idx].reshape(-1, 1), return_distance=False)[:, 0]
    matched_controls = control_idx[nearest]
    differences = outcome[treated_idx] - outcome[matched_controls]
    effect = float(np.mean(differences))
    matched_X = np.vstack([X[treated_idx], X[matched_controls]])
    matched_t = np.r_[np.ones(len(treated_idx), dtype=int), np.zeros(len(treated_idx), dtype=int)]
    before = _max_abs_smd(X, treatment)
    after = _max_abs_smd(matched_X, matched_t)
    overlap = _overlap_fraction(scores, treatment)
    unique_controls = len(np.unique(matched_controls))
    effective = float(len(treated_idx) + unique_controls)
    diagnostics = {
        "propensity_overlap": "pass" if overlap >= 0.8 else "warn",
        "standardized_mean_differences": "pass" if after <= 0.1 else "warn",
        "effective_matched_sample": effective,
        "effect_uncertainty": "not_applicable",
        "design_specific_falsification": _causal_status(
            falsification_status, field="falsification_status"
        ),
    }
    metrics = {
        "effect": effect, "max_abs_smd_before": before,
        "max_abs_smd_after": after, "overlap_fraction": overlap,
        "effective_sample_size": effective,
        "treated_score_p05": float(np.quantile(scores[treatment == 1], 0.05)),
        "treated_score_p95": float(np.quantile(scores[treatment == 1], 0.95)),
        "control_score_p05": float(np.quantile(scores[treatment == 0], 0.05)),
        "control_score_p95": float(np.quantile(scores[treatment == 0], 0.95)),
    }
    from_method(
        "matching", n=len(treatment), treated=int(treatment.sum()),
        controls=int((1 - treatment).sum()), seed=_method_nonnegative_int(seed, field="seed"),
        diagnostics=diagnostics, estimates={"att": effect}, metrics=metrics,
        estimand=estimand,
        design="propensity_nearest_neighbor", **metadata,
    )


def from_propensity_weighting(
    X: Any, treatment: Any, outcome: Any, *, estimand: str = "ate",
    falsification_status: Any, seed: int = 42, **metadata: Any,
) -> None:
    """Fit logistic propensities and emit ATE/ATT inverse-probability weights."""
    import numpy as np

    if estimand not in {"ate", "att"}:
        raise ValueError("propensity weighting supports estimand='ate' or 'att'")
    X, treatment, outcome = _causal_arrays(X, treatment, outcome)
    scores = _propensity_scores(X, treatment, seed=seed)
    if np.any(scores <= 1e-6) or np.any(scores >= 1 - 1e-6):
        raise ValueError("estimated propensity scores violate numerical positivity")
    if estimand == "ate":
        weights = treatment / scores + (1 - treatment) / (1 - scores)
    else:
        weights = treatment + (1 - treatment) * scores / (1 - scores)
    treated_mask = treatment == 1
    control_mask = ~treated_mask
    mean_t = float(np.average(outcome[treated_mask], weights=weights[treated_mask]))
    mean_c = float(np.average(outcome[control_mask], weights=weights[control_mask]))
    effect = mean_t - mean_c
    ess = float(weights.sum() ** 2 / np.sum(weights ** 2))
    before = _max_abs_smd(X, treatment)
    after = _max_abs_smd(X, treatment, weights)
    overlap = _overlap_fraction(scores, treatment)
    max_weight = float(np.max(weights))
    diagnostics = {
        "propensity_overlap": "pass" if overlap >= 0.8 else "warn",
        "weight_extremes": "pass" if max_weight <= 10 else "warn",
        "standardized_mean_differences": "pass" if after <= 0.1 else "warn",
        "effective_sample_size": ess,
        "effect_uncertainty": "not_applicable",
        "design_specific_falsification": _causal_status(
            falsification_status, field="falsification_status"
        ),
    }
    metrics = {
        "effect": effect, "max_abs_smd_before": before,
        "max_abs_smd_after": after, "overlap_fraction": overlap,
        "effective_sample_size": ess, "max_weight": max_weight,
        "treated_score_p05": float(np.quantile(scores[treatment == 1], 0.05)),
        "treated_score_p95": float(np.quantile(scores[treatment == 1], 0.95)),
        "control_score_p05": float(np.quantile(scores[treatment == 0], 0.05)),
        "control_score_p95": float(np.quantile(scores[treatment == 0], 0.95)),
    }
    from_method(
        "propensity_weighting", n=len(treatment), treated=int(treatment.sum()),
        controls=int((1 - treatment).sum()), seed=_method_nonnegative_int(seed, field="seed"),
        diagnostics=diagnostics, estimates={estimand: effect}, metrics=metrics,
        estimand=estimand,
        design="inverse_probability_weighting", **metadata,
    )


def _synthetic_weights(target_pre: Any, donors_pre: Any) -> Any:
    import numpy as np
    from scipy.optimize import minimize

    donor_count = donors_pre.shape[1]
    objective = lambda w: float(np.mean((target_pre - donors_pre @ w) ** 2))
    fit = minimize(
        objective, np.full(donor_count, 1.0 / donor_count), method="SLSQP",
        bounds=[(0.0, 1.0)] * donor_count,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    if not fit.success:
        raise RuntimeError(f"synthetic-control optimization failed: {fit.message}")
    return np.asarray(fit.x, dtype=float)


def from_synthetic_control(
    treated_series: Any, donor_series: Any, *, intervention_index: int,
    falsification_status: Any, **metadata: Any,
) -> None:
    """Fit constrained donor weights and donor-placebo inference."""
    import numpy as np

    treated = np.asarray(treated_series, dtype=float).reshape(-1)
    donors = np.asarray(donor_series, dtype=float)
    if donors.ndim != 2 or donors.shape[0] != len(treated):
        raise ValueError("donor_series must be time-by-donor and align with treated_series")
    if not np.all(np.isfinite(treated)) or not np.all(np.isfinite(donors)):
        raise ValueError("synthetic-control inputs must be finite")
    pre = _method_positive_int(intervention_index, field="intervention_index")
    post = len(treated) - pre
    if pre < 3 or post < 1 or donors.shape[1] < 3:
        raise ValueError("synthetic control requires >=3 pre periods, >=1 post period, and >=3 donors")
    weights = _synthetic_weights(treated[:pre], donors[:pre])
    synthetic = donors @ weights
    gaps = treated - synthetic
    pre_rmse = float(np.sqrt(np.mean(gaps[:pre] ** 2)))
    post_rmse = float(np.sqrt(np.mean(gaps[pre:] ** 2)))
    effect = float(np.mean(gaps[pre:]))
    ratio = post_rmse / max(pre_rmse, 1e-12)
    placebo_ratios: list[float] = []
    for index in range(donors.shape[1]):
        pool = np.delete(donors, index, axis=1)
        w = _synthetic_weights(donors[:pre, index], pool[:pre])
        placebo_gap = donors[:, index] - pool @ w
        pre_p = float(np.sqrt(np.mean(placebo_gap[:pre] ** 2)))
        post_p = float(np.sqrt(np.mean(placebo_gap[pre:] ** 2)))
        placebo_ratios.append(post_p / max(pre_p, 1e-12))
    placebo_p = float((1 + sum(value >= ratio for value in placebo_ratios)) / (1 + len(placebo_ratios)))
    scale = float(np.std(treated[:pre], ddof=1))
    fit_ratio = pre_rmse / max(scale, 1e-12)
    max_weight = float(np.max(weights))
    diagnostics = {
        "pre_treatment_fit": "pass" if fit_ratio <= 0.2 else "warn",
        "placebo_distribution": "pass" if len(placebo_ratios) >= 3 else "warn",
        "donor_weight_concentration": "pass" if max_weight <= 0.8 else "warn",
        "effect_uncertainty": "not_applicable",
        "design_specific_falsification": _causal_status(
            falsification_status, field="falsification_status"
        ),
    }
    metrics = {
        "effect": effect, "pre_rmse": pre_rmse, "post_rmse": post_rmse,
        "placebo_p_value": placebo_p, "max_donor_weight": max_weight,
    }
    n = int((donors.shape[1] + 1) * len(treated))
    from_method(
        "synthetic_control", n=n, donors=int(donors.shape[1]),
        pre_periods=pre, post_periods=post, diagnostics=diagnostics,
        estimates={"unit_time_att": effect}, metrics=metrics, estimand="unit_time_att",
        design="synthetic_control", **metadata,
    )


def from_treatment_heterogeneity(
    X: Any, treatment: Any, outcome: Any, *, falsification_status: Any,
    seed: int = 42, test_fraction: float = 0.4, **metadata: Any,
) -> None:
    """Fit an honest sample-split random-forest T-learner and emit CATE aggregates."""
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    X, treatment, outcome = _causal_arrays(X, treatment, outcome)
    if not 0.2 <= float(test_fraction) <= 0.5:
        raise ValueError("test_fraction must be between 0.2 and 0.5")
    indices = np.arange(len(treatment))
    train, test = train_test_split(
        indices, test_size=float(test_fraction), random_state=seed, stratify=treatment,
    )
    models = []
    for arm in (0, 1):
        arm_train = train[treatment[train] == arm]
        model = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=5, random_state=seed + arm, n_jobs=1,
        )
        model.fit(X[arm_train], outcome[arm_train])
        models.append(model)
    cate = models[1].predict(X[test]) - models[0].predict(X[test])
    average = float(np.mean(cate))
    cate_sd = float(np.std(cate, ddof=1))
    quartiles: Any = np.quantile(cate, [0.25, 0.75])
    q1, q4 = float(quartiles[0]), float(quartiles[1])
    low = float(np.mean(cate[cate <= q1]))
    high = float(np.mean(cate[cate >= q4]))
    propensity = _propensity_scores(X[train], treatment[train], seed=seed)
    from sklearn.linear_model import LogisticRegression
    propensity_model = LogisticRegression(max_iter=2000, random_state=seed).fit(X[train], treatment[train])
    p_test = np.clip(propensity_model.predict_proba(X[test])[:, 1], 1e-3, 1 - 1e-3)
    transformed = (treatment[test] - p_test) * outcome[test] / (p_test * (1 - p_test))
    calibration = float(np.corrcoef(cate, transformed)[0, 1])
    if not math.isfinite(calibration):
        raise ValueError("heterogeneity calibration is undefined on the honest test split")
    overlap = _overlap_fraction(propensity, treatment[train])
    balance = _max_abs_smd(X[train], treatment[train])
    diagnostics = {
        "propensity_overlap": "pass" if overlap >= 0.8 else "warn",
        "standardized_mean_differences": "pass" if balance <= 0.1 else "warn",
        "honest_sample_splitting": "pass",
        "subgroup_multiplicity": "not_applicable",
        "heterogeneity_calibration": calibration,
        "effect_uncertainty": "not_applicable",
        "design_specific_falsification": _causal_status(
            falsification_status, field="falsification_status"
        ),
    }
    metrics = {
        "average_cate": average, "cate_sd": cate_sd,
        "q4_q1_contrast": high - low, "calibration_correlation": calibration,
        "overlap_fraction": overlap, "max_abs_smd_before": balance,
    }
    from_method(
        "treatment_effect_heterogeneity", n=len(treatment), seed=_method_nonnegative_int(seed, field="seed"),
        diagnostics=diagnostics, estimates={"average_predicted_cate": average},
        metrics=metrics, estimand="average_predicted_cate",
        design="honest_t_learner", **metadata,
    )


def _sensemakr_robustness_value(
    t_statistic: float, dof: float, *, q: float = 1.0, alpha: float = 0.05,
) -> float:
    """Exact ``sensemakr::robustness_value.numeric`` algebra.

    Includes its df-1 critical value and the constraint-binding/extreme-RV
    branch. See the maintained Sensemakr ``R/sensitivity_stats.R`` source.
    """
    from scipy.stats import t as student_t

    if dof <= 1 or q <= 0 or not 0 < alpha <= 1:
        raise ValueError("dof>1, q>0, and 0<alpha<=1 are required")
    fq = q * abs(t_statistic / math.sqrt(dof))
    fcrit = abs(float(student_t.ppf(alpha / 2.0, dof - 1))) / math.sqrt(dof - 1)
    fqa = fq - fcrit
    if fqa <= 0:
        return 0.0
    binding = 2.0 / (1.0 + math.sqrt(1.0 + 4.0 / (fqa * fqa)))
    fq2, fcrit2 = fq * fq, fcrit * fcrit
    extreme = (fq2 - fcrit2) / (1.0 + fq2) if fq2 > fcrit2 else 0.0
    use_extreme = fcrit > 0 and fq > 1.0 / fcrit
    return float(extreme if use_extreme else binding)


def from_causal_sensitivity(
    model: Any, coefficient: str, *, falsification_status: Any,
    alpha: float = 0.05, q: float = 1.0, **metadata: Any,
) -> None:
    """Emit Cinelli-Hazlett partial-R2 robustness values from statsmodels."""
    name = _method_quantity_name(coefficient, field="coefficient")
    params = _to_dict(getattr(model, "params", {}))
    bse = _to_dict(getattr(model, "bse", {}))
    tvalues = _to_dict(getattr(model, "tvalues", {}))
    if name not in params or name not in bse or name not in tvalues:
        raise ValueError("coefficient is not present in model params/bse/tvalues")
    n = _method_positive_int(getattr(model, "nobs", None), field="model.nobs")
    df = _safe_float(getattr(model, "df_resid", None))
    safe_alpha = _safe_float(alpha)
    if df is None or df <= 0 or safe_alpha is None or not 0 < safe_alpha < 1:
        raise ValueError("model residual df and alpha must be valid")
    t_value = abs(float(tvalues[name]))
    safe_q = _safe_float(q)
    if safe_q is None or safe_q <= 0:
        raise ValueError("q must be positive")
    rv_zero = _sensemakr_robustness_value(t_value, df, q=safe_q, alpha=1.0)
    rv_alpha = _sensemakr_robustness_value(t_value, df, q=safe_q, alpha=safe_alpha)
    estimate = float(params[name])
    diagnostics = {
        "robustness_value": rv_zero,
        "assumption_grid": "pass",
        "design_specific_falsification": _causal_status(
            falsification_status, field="falsification_status"
        ),
    }
    metrics = {
        "robustness_value_zero": rv_zero,
        "robustness_value_alpha": rv_alpha,
        "t_statistic": t_value,
        "q": safe_q, "alpha": safe_alpha,
        "margin_equal_r2_01": rv_zero - 0.01,
        "margin_equal_r2_05": rv_zero - 0.05,
        "margin_equal_r2_10": rv_zero - 0.10,
    }
    from_method(
        "causal_sensitivity", n=n, diagnostics=diagnostics,
        estimates={name: estimate}, metrics=metrics,
        estimand="robustness_value", design="omitted_variable_sensitivity",
        **metadata,
    )


# ---------------------------------------------------------------------------
# Time-series helpers — ordered evaluation and aggregate diagnostics only
# ---------------------------------------------------------------------------

def _time_series_values(
    series: Any, *, time_index: Any | None, cadence: Any, frequency: int,
    ordered: bool | None = None, regular: bool | None = None, minimum: int = 20,
) -> tuple[Any, int, dict[str, float]]:
    import numpy as np
    import pandas as pd

    if ordered is not None or regular is not None:
        raise ValueError("ordered/regular flags are not evidence; supply a time_index and cadence")
    values = np.asarray(series, dtype=float).reshape(-1)
    freq = _method_positive_int(frequency, field="frequency")
    if len(values) < minimum or not np.all(np.isfinite(values)):
        raise ValueError(f"time series requires at least {minimum} finite observations")
    if time_index is None:
        index = getattr(series, "index", None)
        if index is None:
            raise ValueError("time_index is required unless series has an index")
        time_index = index
    pandas_index = pd.Index(time_index)
    raw_index = np.asarray(time_index).reshape(-1)
    if len(raw_index) != len(values):
        raise ValueError("time_index length must equal series length")
    is_datetime = isinstance(pandas_index, pd.DatetimeIndex)
    if is_datetime:
        if pandas_index.hasnans:
            raise ValueError("time_index contains missing timestamps")
        try:
            cadence_value = pd.Timedelta(cadence) if isinstance(cadence, np.timedelta64) else cadence
            offset = pd.tseries.frequencies.to_offset(cadence_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("datetime time_index requires a pandas-compatible cadence") from exc
        numeric_index = pandas_index.asi8
        if np.any(np.diff(numeric_index) <= 0):
            raise ValueError("time_index must be strictly increasing with no duplicates")
        expected_index = pd.DatetimeIndex([value + offset for value in pandas_index[:-1]])
        if not expected_index.equals(pandas_index[1:]):
            raise ValueError("time_index spacing is irregular or inconsistent with cadence")
        proof = {"cadence_min_ratio": 1.0, "cadence_max_ratio": 1.0,
                 "time_span_steps": float(len(values) - 1)}
        return values, freq, proof
    else:
        try:
            numeric_index = raw_index.astype(float)
            expected = float(cadence)
        except (TypeError, ValueError) as exc:
            raise ValueError("numeric time_index requires a numeric cadence") from exc
    if not np.all(np.isfinite(numeric_index)) or not math.isfinite(expected) or expected <= 0:
        raise ValueError("time_index and cadence must be finite and cadence must be positive")
    steps = np.diff(numeric_index)
    if np.any(steps <= 0):
        raise ValueError("time_index must be strictly increasing with no duplicates")
    tolerance = max(abs(expected) * 1e-9, np.finfo(float).eps * 16)
    if not np.allclose(steps, expected, rtol=1e-9, atol=tolerance):
        raise ValueError("time_index spacing is irregular or inconsistent with cadence")
    proof = {
        "cadence_min_ratio": float(np.min(steps) / expected),
        "cadence_max_ratio": float(np.max(steps) / expected),
        "time_span_steps": float((numeric_index[-1] - numeric_index[0]) / expected),
    }
    return values, freq, proof


def _time_diagnostics() -> dict[str, str]:
    """Statuses are emitted only after `_time_series_values` proves the index."""
    return {"temporal_order": "pass", "regular_frequency": "pass"}


def _stationarity_metrics(values: Any) -> tuple[dict[str, float], str]:
    from statsmodels.tsa.stattools import adfuller, kpss

    adf = adfuller(values, autolag="AIC")
    try:
        kpss_result = kpss(values, regression="c", nlags="auto")
        kpss_stat, kpss_p = float(kpss_result[0]), float(kpss_result[1])
    except Exception:  # noqa: BLE001 - constant/degenerate series
        kpss_stat, kpss_p = float("nan"), float("nan")
    metrics = {
        "adf_statistic": float(adf[0]), "adf_p_value": float(adf[1]),
        "adf_lags": float(adf[2]),
        "stationarity_statistic": float(adf[0]),
    }
    if math.isfinite(kpss_stat) and math.isfinite(kpss_p):
        metrics.update(kpss_statistic=kpss_stat, kpss_p_value=kpss_p)
    consensus = "pass" if adf[1] < 0.05 and kpss_p > 0.05 else "warn"
    return metrics, consensus


def _residual_autocorrelation(residuals: Any) -> tuple[float, str]:
    import numpy as np
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals = np.asarray(residuals, dtype=float)
    lag = max(1, min(10, len(residuals) // 5))
    result = acorr_ljungbox(residuals, lags=[lag], return_df=True)
    p_value = float(result["lb_pvalue"].iloc[-1])
    return p_value, "pass" if p_value > 0.05 else "warn"


def from_stationarity_diagnostic(
    series: Any, *, time_index: Any | None = None, cadence: Any = None,
    frequency: int, ordered: bool | None = None, regular: bool | None = None,
    **metadata: Any,
) -> None:
    """Run complementary ADF (unit-root null) and KPSS (stationary null)."""
    values, freq, proof = _time_series_values(
        series, time_index=time_index, cadence=cadence, frequency=frequency,
        ordered=ordered, regular=regular,
    )
    metrics, consensus = _stationarity_metrics(values)
    metrics.update(proof)
    diagnostics = {**_time_diagnostics(), "missingness": "pass",
                   "stationarity_consensus": consensus}
    from_method(
        "stationarity_diagnostic", n=len(values), diagnostics=diagnostics,
        metrics=metrics, frequency=freq, **metadata,
    )


def from_seasonal_decomposition(
    series: Any, *, time_index: Any | None = None, cadence: Any = None,
    frequency: int, ordered: bool | None = None, regular: bool | None = None,
    robust: bool = True, **metadata: Any,
) -> None:
    """Fit statsmodels STL and emit component-strength aggregates, never arrays."""
    import numpy as np
    from statsmodels.tsa.seasonal import STL

    values, freq, proof = _time_series_values(
        series, time_index=time_index, cadence=cadence, frequency=frequency,
        ordered=ordered, regular=regular,
        minimum=max(20, 2 * int(frequency)),
    )
    if freq < 2:
        raise ValueError("seasonal decomposition requires frequency >= 2")
    fit = STL(values, period=freq, robust=bool(robust)).fit()
    residual_var = float(np.var(fit.resid))
    trend_strength = max(0.0, 1.0 - residual_var / max(float(np.var(fit.trend + fit.resid)), 1e-12))
    seasonal_strength = max(0.0, 1.0 - residual_var / max(float(np.var(fit.seasonal + fit.resid)), 1e-12))
    metrics = {
        "trend_strength": min(1.0, trend_strength),
        "seasonal_strength": min(1.0, seasonal_strength),
        "residual_sd": float(np.std(fit.resid, ddof=1)),
        "residual_variance_share": residual_var / max(float(np.var(values)), 1e-12),
    }
    metrics.update(proof)
    diagnostics = {
        **_time_diagnostics(),
        "period_support": "pass" if len(values) >= 3 * freq else "warn",
        "residual_share": "pass" if metrics["residual_variance_share"] <= 0.5 else "warn",
    }
    from_method(
        "seasonal_decomposition", n=len(values), diagnostics=diagnostics,
        metrics=metrics, frequency=freq, **metadata,
    )


def _forecast_metrics(actual: Any, forecast: Any, lower: Any, upper: Any) -> dict[str, float]:
    import numpy as np

    actual = np.asarray(actual, dtype=float); forecast = np.asarray(forecast, dtype=float)
    lower = np.asarray(lower, dtype=float); upper = np.asarray(upper, dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean((actual - forecast) ** 2))),
        "mae": float(np.mean(np.abs(actual - forecast))),
        "prediction_interval_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
        "prediction_interval_mean_width": float(np.mean(upper - lower)),
        "nominal_coverage": 0.95,
        "mean_forecast": float(np.mean(forecast)),
        "mean_actual": float(np.mean(actual)),
    }


def _validate_arima_order(order: Any) -> tuple[int, int, int]:
    from numbers import Real

    try:
        raw = tuple(order)
    except (TypeError, ValueError) as exc:
        raise ValueError("order must be a (p,d,q) integer triple") from exc
    if len(raw) != 3 or any(
        isinstance(value, bool) or not isinstance(value, Real)
        or not math.isfinite(float(value)) or float(value) != int(float(value))
        for value in raw
    ):
        raise ValueError("order must be a (p,d,q) integer triple")
    parsed = tuple(int(value) for value in raw)
    if len(parsed) != 3 or any(value < 0 or value > 5 for value in parsed):
        raise ValueError("ARIMA p,d,q must each be between 0 and 5")
    return parsed  # type: ignore[return-value]


def from_arima(
    series: Any, *, order: Any, holdout: int, frequency: int,
    time_index: Any | None = None, cadence: Any = None,
    ordered: bool | None = None, regular: bool | None = None, **metadata: Any,
) -> None:
    """Fit ARIMA on a chronological training prefix and evaluate its held-out tail."""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    values, freq, proof = _time_series_values(
        series, time_index=time_index, cadence=cadence, frequency=frequency,
        ordered=ordered, regular=regular, minimum=40,
    )
    h = _method_positive_int(holdout, field="holdout")
    if h < 3 or h >= len(values) // 2:
        raise ValueError("ARIMA requires ordered regular data and a chronological holdout of 3..<n/2")
    parsed = _validate_arima_order(order)
    train, test = values[:-h], values[-h:]
    fit = ARIMA(train, order=parsed).fit()
    prediction = fit.get_forecast(steps=h)
    forecast = np.asarray(prediction.predicted_mean, dtype=float)
    interval = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
    metrics = _forecast_metrics(test, forecast, interval[:, 0], interval[:, 1])
    metrics.update(aic=float(fit.aic), bic=float(fit.bic))
    metrics.update(proof)
    lb_p, residual_status = _residual_autocorrelation(fit.resid)
    metrics["ljung_box_p_value"] = lb_p
    stationarity_values = np.diff(train, n=parsed[1]) if parsed[1] else train
    _, stationarity = _stationarity_metrics(stationarity_values)
    ar_roots = [abs(value) for value in getattr(fit, "arroots", ())]
    ma_roots = [abs(value) for value in getattr(fit, "maroots", ())]
    coverage = metrics["prediction_interval_coverage"]
    diagnostics = {
        **_time_diagnostics(),
        "stationarity": stationarity,
        "ar_stationarity": "pass" if not ar_roots or min(ar_roots) > 1.0 else "warn",
        "residual_autocorrelation": residual_status,
        "ma_invertibility": "pass" if not ma_roots or min(ma_roots) > 1.0 else "warn",
        "holdout_leakage": "pass",
        "prediction_interval_coverage": "pass" if coverage >= 0.8 else "warn",
    }
    from_method(
        "arima", n=len(values), diagnostics=diagnostics,
        estimates={"mean_holdout_forecast": metrics["mean_forecast"]},
        metrics=metrics, evaluation_split="held_out", frequency=freq,
        training_observations=len(train), test_observations=h,
        interval_method="model_based_gaussian", **metadata,
    )


def from_exponential_smoothing(
    series: Any, *, holdout: int, frequency: int,
    time_index: Any | None = None, cadence: Any = None,
    ordered: bool | None = None, regular: bool | None = None,
    trend: str | None = "add", seasonal: str | None = None,
    **metadata: Any,
) -> None:
    """Fit Holt-Winters on a chronological prefix and backtest Gaussian residual intervals."""
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    values, freq, proof = _time_series_values(
        series, time_index=time_index, cadence=cadence, frequency=frequency,
        ordered=ordered, regular=regular, minimum=40,
    )
    h = _method_positive_int(holdout, field="holdout")
    if h < 3 or h >= len(values) // 2:
        raise ValueError("exponential smoothing requires ordered regular data and chronological holdout")
    if trend not in {None, "add", "mul"} or seasonal not in {None, "add", "mul"}:
        raise ValueError("trend/seasonal must be None, 'add', or 'mul'")
    if seasonal is not None and freq < 2:
        raise ValueError("seasonal smoothing requires frequency >= 2")
    train, test = values[:-h], values[-h:]
    if trend == "mul" or seasonal == "mul":
        raise ValueError("typed ETS reference currently supports additive components only")
    fit = ETSModel(
        pd.Series(train, index=pd.RangeIndex(len(train))), error="add",
        trend="add" if trend == "add" else None,
        seasonal="add" if seasonal == "add" else None,
        seasonal_periods=freq if seasonal is not None else None,
    ).fit(disp=False)
    frame = fit.get_prediction(start=len(train), end=len(values)-1, method="exact").summary_frame(alpha=.05)
    forecast = np.asarray(frame["mean"], dtype=float)
    lower = np.asarray(frame["pi_lower"], dtype=float)
    upper = np.asarray(frame["pi_upper"], dtype=float)
    sigma = float(np.std(fit.resid, ddof=1))
    metrics = _forecast_metrics(test, forecast, lower, upper)
    metrics["residual_sd"] = sigma
    widths = upper - lower
    metrics.update(proof)
    metrics["first_interval_width"] = float(widths[0])
    metrics["last_interval_width"] = float(widths[-1])
    lb_p, residual_status = _residual_autocorrelation(fit.resid)
    metrics["ljung_box_p_value"] = lb_p
    diagnostics = {
        **_time_diagnostics(),
        "residual_autocorrelation": residual_status, "holdout_leakage": "pass",
        "prediction_interval_coverage": (
            "pass" if metrics["prediction_interval_coverage"] >= 0.8 else "warn"
        ),
    }
    from_method(
        "exponential_smoothing", n=len(values), diagnostics=diagnostics,
        estimates={"mean_holdout_forecast": metrics["mean_forecast"]},
        metrics=metrics, evaluation_split="held_out", frequency=freq,
        training_observations=len(train), test_observations=h,
        interval_method="ets_state_space_exact", **metadata,
    )


def from_interrupted_time_series(
    series: Any, *, intervention_index: int, frequency: int,
    time_index: Any | None = None, cadence: Any = None,
    ordered: bool | None = None, regular: bool | None = None,
    falsification_status: Any, **metadata: Any,
) -> None:
    """Fit segmented level/slope change with SARIMAX AR(1) errors."""
    import numpy as np
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    values, freq, proof = _time_series_values(
        series, time_index=time_index, cadence=cadence, frequency=frequency,
        ordered=ordered, regular=regular, minimum=40,
    )
    cut = _method_positive_int(intervention_index, field="intervention_index")
    if cut < 15 or len(values) - cut < 10:
        raise ValueError("ITS requires ordered regular data with >=15 pre and >=10 post observations")
    time: Any = np.arange(len(values), dtype=float)
    exog = pd.DataFrame({
        "time": time,
        "level_change": (time >= cut).astype(float),
        "slope_change": np.maximum(0.0, time - cut),
    })
    fit = SARIMAX(values, exog=exog, order=(1, 0, 0), trend="c",
                  enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
    names = list(fit.param_names); params = dict(zip(names, fit.params, strict=True))
    ses = dict(zip(names, fit.bse, strict=True)); pvals = dict(zip(names, fit.pvalues, strict=True))
    estimates = {key: float(params[key]) for key in ("level_change", "slope_change")}
    standard_errors = {key: float(ses[key]) for key in estimates}
    p_values = {key: float(pvals[key]) for key in estimates}
    ci_lower = {key: estimates[key] - 1.96 * standard_errors[key] for key in estimates}
    ci_upper = {key: estimates[key] + 1.96 * standard_errors[key] for key in estimates}
    lb_p, residual_status = _residual_autocorrelation(fit.resid)
    import statsmodels.api as sm
    pre_time = time[:cut]
    pre_mid = cut // 2
    pre_x = np.column_stack([
        np.ones(cut), pre_time, (pre_time >= pre_mid).astype(float),
        np.maximum(0.0, pre_time - pre_mid),
    ])
    pre_fit = sm.OLS(values[:cut], pre_x).fit(cov_type="HAC", cov_kwds={"maxlags": max(1, min(freq, cut // 5))})
    pre_test = pre_fit.wald_test(np.array([[0, 0, 1, 0], [0, 0, 0, 1]]), scalar=True)
    pretrend_p = float(pre_test.pvalue)
    diagnostics = {
        **_time_diagnostics(),
        "pre_intervention_trend": "pass" if pretrend_p > .05 else "warn",
        "intervention_timing": "pass",
        "residual_autocorrelation": residual_status,
        "design_specific_falsification": _causal_status(
            falsification_status, field="falsification_status"
        ),
    }
    metrics = {"aic": float(fit.aic), "bic": float(fit.bic), "ljung_box_p_value": lb_p,
               "pretrend_stability_p_value": pretrend_p, **proof}
    from_method(
        "interrupted_time_series", n=len(values), diagnostics=diagnostics,
        estimates=estimates, standard_errors=standard_errors, p_values=p_values,
        ci_lower=ci_lower, ci_upper=ci_upper, metrics=metrics,
        uncertainty_type="classical", frequency=freq,
        pre_periods=cut, post_periods=len(values)-cut, **metadata,
    )


def from_forecast_backtest(
    series: Any, *, order: Any, initial: int, frequency: int,
    time_index: Any | None = None, cadence: Any = None,
    ordered: bool | None = None, regular: bool | None = None, **metadata: Any,
) -> None:
    """Expanding-window one-step ARIMA backtest with naive baseline and intervals."""
    import numpy as np
    from statsmodels.tsa.arima.model import ARIMA

    values, freq, proof = _time_series_values(
        series, time_index=time_index, cadence=cadence, frequency=frequency,
        ordered=ordered, regular=regular, minimum=40,
    )
    start = _method_positive_int(initial, field="initial")
    parsed = _validate_arima_order(order)
    if start < max(30, 2 * freq) or len(values) - start < 5:
        raise ValueError("rolling-origin backtest requires ordered regular data, adequate initial window and >=5 origins")
    predictions=[]; actual=[]; lower=[]; upper=[]; baseline=[]
    for origin in range(start, len(values)):
        history = values[:origin]
        fit = ARIMA(history, order=parsed).fit()
        forecast = fit.get_forecast(steps=1)
        predictions.append(float(forecast.predicted_mean[0]))
        interval = np.asarray(forecast.conf_int(alpha=.05), dtype=float)[0]
        lower.append(float(interval[0])); upper.append(float(interval[1]))
        actual.append(float(values[origin])); baseline.append(float(values[origin-1]))
    metrics = _forecast_metrics(actual, predictions, lower, upper)
    baseline_rmse = float(np.sqrt(np.mean((np.asarray(actual)-np.asarray(baseline))**2)))
    metrics.update(baseline_rmse=baseline_rmse, origins=float(len(actual)))
    metrics.update(proof)
    diagnostics = {
        **_time_diagnostics(),
        "rolling_origin_backtest":"pass", "holdout_leakage":"pass",
        "prediction_interval_coverage": (
            "pass" if metrics["prediction_interval_coverage"] >= .8 else "warn"
        ),
        "baseline_comparison":"pass" if metrics["rmse"] <= baseline_rmse else "warn",
    }
    from_method(
        "forecast_backtest", n=len(values), folds=len(actual), diagnostics=diagnostics,
        metrics=metrics, evaluation_split="rolling_origin", frequency=freq,
        training_observations=start, test_observations=len(actual),
        interval_method="model_based_gaussian", **metadata,
    )


# ---------------------------------------------------------------------------
# Domain/design helpers — disclosure-safe aggregates only
# ---------------------------------------------------------------------------

def from_geospatial_moran(
    frame: Any, *, value: str, distance_threshold: float,
    permutations: int = 999, seed: int = 1729, **metadata: Any,
) -> None:
    """Distance-band Moran analysis with CRS/unit and permutation diagnostics."""
    import numpy as np
    import geopandas as gpd
    from scipy.spatial import cKDTree

    if not isinstance(frame, gpd.GeoDataFrame) or frame.crs is None:
        raise TypeError("frame must be a GeoDataFrame with a declared CRS")
    if not frame.crs.is_projected or frame.crs.to_epsg() is None:
        raise ValueError("geospatial distance analysis requires a projected EPSG CRS")
    if value not in frame.columns or frame.geometry.name == value:
        raise ValueError("value must name a non-geometry numeric column")
    if frame.geometry.isna().any() or frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ValueError("geometries must be present, non-empty, and valid")
    values = np.asarray(frame[value], dtype=float)
    n = len(values)
    if n < 20 or not np.all(np.isfinite(values)) or float(np.var(values)) <= 0:
        raise ValueError("Moran analysis requires at least 20 finite, non-constant values")
    threshold = float(distance_threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("distance_threshold must be finite and positive")
    reps = _method_positive_int(permutations, field="permutations")
    if reps < 199 or reps > 9999:
        raise ValueError("permutations must be between 199 and 9999")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    centroids = frame.geometry.centroid
    coordinates = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    pairs = sorted(cKDTree(coordinates).query_pairs(threshold))
    if not pairs:
        raise ValueError("distance threshold produces no spatial neighbor links")
    degree: Any = np.zeros(n, dtype=int)
    left_index = np.asarray([pair[0] for pair in pairs], dtype=int)
    right_index = np.asarray([pair[1] for pair in pairs], dtype=int)
    np.add.at(degree, left_index, 1); np.add.at(degree, right_index, 1)
    centered = values - float(np.mean(values))
    denominator = float(np.dot(centered, centered))
    expected_i = -1.0 / (n - 1)

    def moran(permuted: Any) -> float:
        numerator = 2.0 * float(np.dot(permuted[left_index], permuted[right_index]))
        return float(n * numerator / (2 * len(pairs) * denominator))

    observed = moran(centered)
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(reps):
        candidate = moran(rng.permutation(centered))
        extreme += abs(candidate - expected_i) >= abs(observed - expected_i)
    p_value = (extreme + 1) / (reps + 1)
    mcse = math.sqrt(p_value * (1 - p_value) / (reps + 1))
    unit_factor = float(frame.crs.axis_info[0].unit_conversion_factor)
    diagnostics = {
        "crs_validity": "pass", "spatial_weights": "pass",
        "spatial_autocorrelation": p_value, "privacy_aggregation": "pass",
    }
    metrics = {
        "moran_i": observed, "expected_moran_i": expected_i,
        "permutation_p_value": p_value, "permutation_mcse": mcse,
        "neighbor_links": float(len(pairs)), "mean_neighbors": float(np.mean(degree)),
        "island_fraction": float(np.mean(degree == 0)),
        "distance_threshold_crs_units": threshold,
        "distance_threshold_metres": threshold * unit_factor,
        "crs_linear_unit_to_metre": unit_factor,
    }
    from_method(
        "geospatial_analysis", n=n, diagnostics=diagnostics, metrics=metrics,
        crs_epsg=int(frame.crs.to_epsg()), seed=seed, replicates=reps,
        spatial_weight_rule="distance_band_binary", **metadata,
    )


def from_network_graph(
    nodes: Any, edges: Any, *, directed: bool = False, **metadata: Any,
) -> None:
    """Summarize a simple graph without emitting node identifiers or edge rows."""
    import numpy as np
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    if directed:
        raise ValueError("typed graph reference currently supports undirected simple graphs only")
    try:
        node_rows = list(nodes); rows = list(edges)
        if len(node_rows) < 10 or len(set(node_rows)) != len(node_rows):
            raise ValueError("nodes must declare at least 10 unique identifiers")
        labels = {label: index for index, label in enumerate(node_rows)}
    except TypeError as exc:
        raise ValueError("node identifiers must be hashable") from exc
    if not rows or any(not isinstance(edge, (tuple, list)) or len(edge) != 2 for edge in rows):
        raise ValueError("edges must contain two-item endpoint pairs")
    normalized: set[tuple[int, int]] = set()
    for left_label, right_label in rows:
        if left_label not in labels or right_label not in labels:
            raise ValueError("every edge endpoint must belong to the declared node universe")
        left_index, right_index = labels[left_label], labels[right_label]
        if left_index == right_index:
            raise ValueError("self-loops are not supported by the simple-graph helper")
        edge = (min(left_index, right_index), max(left_index, right_index))
        if edge in normalized:
            raise ValueError("duplicate edges are not supported")
        normalized.add(edge)
    n = len(labels)
    if n < 10:
        raise ValueError("network analysis requires at least 10 nodes")
    left_nodes: Any = np.asarray([edge[0] for edge in normalized], dtype=int)
    right_nodes: Any = np.asarray([edge[1] for edge in normalized], dtype=int)
    adjacency = sparse.csr_matrix(
        (
            np.ones(2 * len(left_nodes)),
            (np.r_[left_nodes, right_nodes], np.r_[right_nodes, left_nodes]),
        ),
        shape=(n, n),
    )
    components, component_labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(component_labels)
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    triangles = float((adjacency @ adjacency @ adjacency).diagonal().sum()) / 6.0
    connected_triples = float(np.sum(degrees * (degrees - 1) / 2))
    transitivity = 0.0 if connected_triples == 0 else 3 * triangles / connected_triples
    metrics = {
        "node_count": float(n), "edge_count": float(len(normalized)),
        "density": float(2 * len(normalized) / (n * (n - 1))),
        "mean_degree": float(np.mean(degrees)), "component_count": float(components),
        "largest_component_fraction": float(np.max(component_sizes) / n),
        "isolate_fraction": float(np.mean(degrees == 0)), "transitivity": transitivity,
    }
    diagnostics = {
        "graph_definition": "pass", "graph_symmetry": "pass",
        "dependence_aware_uncertainty": "not_applicable", "privacy_aggregation": "pass",
    }
    from_method("network_analysis", n=n, diagnostics=diagnostics, metrics=metrics, **metadata)


def from_text_stability(
    documents: Any, *, clusters: int = 2, seed: int = 1729,
    max_features: int = 5000, **metadata: Any,
) -> None:
    """TF-IDF clustering stability with no document, token, or feature-name output."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import adjusted_rand_score

    docs = list(documents)
    if len(docs) < 20 or any(not isinstance(doc, str) or not doc.strip() for doc in docs):
        raise ValueError("text analysis requires at least 20 non-empty documents")
    k = _method_positive_int(clusters, field="clusters")
    cap = _method_positive_int(max_features, field="max_features")
    if k < 2 or k > min(10, len(docs) // 5) or cap < 100 or cap > 10000:
        raise ValueError("clusters/max_features are outside supported privacy-safe bounds")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    matrix = TfidfVectorizer(
        lowercase=True, stop_words="english", min_df=2, max_df=.95,
        max_features=cap, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    ).fit_transform(docs)
    if matrix.shape[1] < k:
        raise ValueError("text corpus has insufficient repeated vocabulary")
    first = KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(matrix)
    second = KMeans(n_clusters=k, n_init=20, random_state=seed + 1).fit_predict(matrix)
    initialization_stability = float(adjusted_rand_score(first, second))
    rng = np.random.default_rng(seed)
    sample_n = max(k * 3, math.floor(.8 * len(docs)))
    sample_a = rng.choice(len(docs), size=sample_n, replace=False)
    sample_b = rng.choice(len(docs), size=sample_n, replace=False)
    resample_a = KMeans(n_clusters=k, n_init=20, random_state=seed + 2).fit(matrix[sample_a]).predict(matrix)
    resample_b = KMeans(n_clusters=k, n_init=20, random_state=seed + 3).fit(matrix[sample_b]).predict(matrix)
    stability = float(adjusted_rand_score(resample_a, resample_b))
    sizes = np.bincount(first, minlength=k)
    metrics = {
        "document_count": float(len(docs)), "vocabulary_size": float(matrix.shape[1]),
        "matrix_sparsity": float(1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "mean_document_norm": float(np.mean(np.sqrt(matrix.multiply(matrix).sum(axis=1)))),
        "initialization_stability_ari": initialization_stability,
        "resampling_stability_ari": stability, "cluster_count": float(k),
        "minimum_cluster_fraction": float(np.min(sizes) / len(docs)),
    }
    diagnostics = {
        "tokenization_specification": "pass", "held_out_or_stability_check": stability,
        "document_privacy": "pass", "vocabulary_privacy": "pass",
    }
    from_method("text_analysis", n=len(docs), diagnostics=diagnostics,
                metrics=metrics, seed=seed, stability_type="document_resampling", **metadata)


def from_arviz_posterior(
    inference_data: Any, *, observed_variable: str, **metadata: Any,
) -> None:
    """Emit Bayesian summaries only after ArviZ-computed MCMC and PPC diagnostics."""
    import numpy as np
    try:
        from arviz_stats import summary as arviz_summary
    except ImportError as exc:
        try:
            import arviz as az
            arviz_summary = az.summary
        except ImportError:
            raise RuntimeError("Bayesian qualification requires the maintained ArviZ package") from exc

    def group(name: str) -> Any:
        candidate = getattr(inference_data, name, None)
        if candidate is None:
            try:
                candidate = inference_data[name]
            except (KeyError, TypeError):
                return None
        return getattr(candidate, "dataset", candidate)

    posterior = group("posterior")
    observed_group = group("observed_data")
    predictive_group = group("posterior_predictive")
    sample_stats = group("sample_stats")
    if any(group is None for group in (posterior, observed_group, predictive_group, sample_stats)):
        raise ValueError("InferenceData must include posterior, sample_stats, observed_data, and posterior_predictive")
    if observed_variable not in observed_group or observed_variable not in predictive_group:
        raise ValueError("observed_variable must exist in observed and posterior-predictive groups")
    chains = int(posterior.sizes.get("chain", 0)); draws = int(posterior.sizes.get("draw", 0))
    if chains < 4 or draws < 100:
        raise ValueError("Bayesian diagnostics require at least four chains and 100 draws per chain")
    # Validate the safety-critical per-draw evidence before asking ArviZ to
    # summarize the fit.  Besides failing faster, this ensures malformed
    # InferenceData-like objects cannot surface a converter error in place of
    # the precise missing-divergence or PPC-alignment failure.
    if "diverging" not in sample_stats:
        raise ValueError("sample_stats must include per-draw divergence indicators")
    divergence_flags = np.asarray(sample_stats["diverging"])
    if divergence_flags.shape[:2] != (chains, draws) or divergence_flags.size != chains * draws:
        raise ValueError("divergence indicators must align exactly with posterior chain/draw axes")
    divergences = int(divergence_flags.sum())
    observed = np.asarray(observed_group[observed_variable], dtype=float)
    replicated = np.asarray(predictive_group[observed_variable], dtype=float)
    if (not np.all(np.isfinite(observed)) or not np.all(np.isfinite(replicated))
            or replicated.shape[:2] != (chains, draws)
            or replicated.shape[2:] != observed.shape):
        raise ValueError("posterior predictive draws and observed values must be finite and aligned")
    try:
        summary = arviz_summary(
            inference_data, kind="all", ci_prob=.95, ci_kind="hdi", round_to="none",
        )
    except TypeError:  # ArviZ <=0.22 compatibility
        summary = arviz_summary(  # type: ignore[call-arg]
            inference_data, kind="all", hdi_prob=.95, round_to=None,
        )
    if summary.empty or len(summary) > 20:
        raise ValueError("posterior must expose between 1 and 20 aggregate parameters")
    lower_columns = [name for name in summary.columns if name == "hdi_2.5%" or name.endswith("_lb")]
    upper_columns = [name for name in summary.columns if name == "hdi_97.5%" or name.endswith("_ub")]
    required = {"mean", "ess_bulk", "ess_tail", "r_hat"}
    if not required.issubset(summary.columns) or len(lower_columns) != 1 or len(upper_columns) != 1:
        raise ValueError("ArviZ summary is missing required convergence columns")
    estimates={};lower={};upper={}
    for index, (_, row) in enumerate(summary.iterrows(), 1):
        key=f"parameter_{index}";estimates[key]=float(row["mean"])
        lower[key]=float(row[lower_columns[0]]);upper[key]=float(row[upper_columns[0]])
    replicated_means = replicated.reshape(chains * draws, -1).mean(axis=1)
    ppc = float(np.mean(replicated_means >= float(np.mean(observed))))
    diagnostics = {"rhat": float(summary["r_hat"].max()),
                   "bulk_ess": float(summary["ess_bulk"].min()),
                   "tail_ess": float(summary["ess_tail"].min()),
                   "divergences": float(divergences),
                   "posterior_predictive_check": ppc}
    metrics = {"chains": float(chains), "draws_per_chain": float(draws),
               "parameter_count": float(len(summary)),
               "posterior_predictive_replicates": float(len(replicated_means))}
    from_method("bayesian_model", n=int(observed.size), diagnostics=diagnostics,
                estimates=estimates, ci_lower=lower, ci_upper=upper, metrics=metrics,
                uncertainty_type="posterior", **metadata)


def from_power_precision(
    effect_sizes: Any, *, alpha: float = .05, target_power: float = .8,
    allocation_ratio: float = 1.0, alternative: str = "two-sided", **metadata: Any,
) -> None:
    """Prospective two-sample t-test sample sizes over declared effect scenarios."""
    from statsmodels.stats.power import TTestIndPower

    effects = tuple(float(value) for value in effect_sizes)
    if not 1 <= len(effects) <= 12 or any(not math.isfinite(value) or not .01 <= value <= 5 for value in effects):
        raise ValueError("effect_sizes must contain 1..12 finite positive prospective scenarios")
    if effects != tuple(sorted(set(effects))):
        raise ValueError("effect_sizes must be unique and increasing")
    if not 0 < alpha < 1 or not .5 < target_power < 1 or not 0 < allocation_ratio <= 10:
        raise ValueError("alpha, target_power, or allocation_ratio is invalid")
    # Effect sizes are positive magnitudes. statsmodels may return a spurious
    # boundary solution for a positive magnitude with ``smaller``.
    if alternative not in {"two-sided", "larger"}:
        raise ValueError("alternative must be two-sided or larger for positive effect sizes")
    solver = TTestIndPower(); estimates={}; metrics={}
    max_total = 0
    for index, effect in enumerate(effects, 1):
        name = f"scenario_{index}"
        first_n = math.ceil(float(solver.solve_power(
            effect_size=effect, nobs1=None, alpha=alpha, power=target_power,
            ratio=allocation_ratio, alternative=alternative,
        )))
        second_n = math.ceil(first_n * allocation_ratio); total = first_n + second_n
        estimates[name] = float(total); metrics[f"effect_size#{name}"] = effect
        metrics[f"group1_n#{name}"] = float(first_n); metrics[f"group2_n#{name}"] = float(second_n)
        max_total = max(max_total, total)
    metrics.update(scenario_count=float(len(effects)), alpha=float(alpha),
                   target_power=float(target_power), allocation_ratio=float(allocation_ratio))
    diagnostics = {"effect_size_scenarios": "pass", "alpha_and_power": "pass",
                   "prospective_design": "pass"}
    payload = {
        "type": "method_result", "method_id": "power_precision", "n": max_total,
        "diagnostics": diagnostics, "estimates": estimates, "metrics": metrics,
        "test_alternative": (
            "two_sided" if alternative == "two-sided" else "larger"
        ), **metadata,
    }
    payload["_via_helper"] = "power_precision_v1"
    _write_result(payload)


def from_simulation_design(
    *, effect_size: float, group_n: int, replications: int = 2000,
    alpha: float = .05, seed: int = 1729, **metadata: Any,
) -> None:
    """Prospective Monte Carlo power for a standardized two-sample design."""
    import numpy as np
    from scipy.stats import ttest_ind
    from statsmodels.stats.proportion import proportion_confint
    from statsmodels.stats.power import TTestIndPower

    effect = float(effect_size); size = _method_positive_int(group_n, field="group_n")
    reps = _method_positive_int(replications, field="replications")
    if (not math.isfinite(effect) or not .01 <= effect <= 5
            or size < 5 or reps < 1000 or reps > 20000
            or reps * size > 5_000_000):
        raise ValueError("effect_size, group_n, or replications is outside supported bounds")
    if not 0 < alpha < 1 or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("alpha and seed must be valid")
    rng = np.random.default_rng(seed)
    control = rng.normal(size=(reps, size)); treated = rng.normal(loc=effect, size=(reps, size))
    p_values = ttest_ind(treated, control, axis=1, equal_var=True).pvalue
    rejected = int(np.sum(p_values < alpha)); power = rejected / reps
    mcse = math.sqrt(power * (1 - power) / reps)
    analytic = float(TTestIndPower().power(effect, size, alpha, ratio=1, alternative="two-sided"))
    lower, upper = proportion_confint(rejected, reps, alpha=.05, method="beta")
    diagnostics = {"seed_recorded": "pass", "replication_count": float(reps),
                   "monte_carlo_standard_error": mcse, "scenario_sensitivity": "warn"}
    metrics = {"effect_size": effect, "group_n": float(size), "alpha": float(alpha),
               "replications": float(reps), "rejection_count": float(rejected),
               "monte_carlo_standard_error": mcse, "analytic_power": analytic,
               "absolute_analytic_difference": abs(power - analytic)}
    payload = {
        "type": "method_result", "method_id": "simulation_design", "n": 2 * size,
        "diagnostics": diagnostics, "estimates": {"empirical_power": power},
        "ci_lower": {"empirical_power": float(lower)},
        "ci_upper": {"empirical_power": float(upper)}, "metrics": metrics,
        "seed": seed, "replicates": reps,
        "interval_method": "clopper_pearson_binomial", **metadata,
    }
    payload["_via_helper"] = "simulation_design_v1"
    _write_result(payload)


def _fitted_method_maps(
    fit: Any, *, fixed_effects_only: bool = False,
) -> tuple[
    dict[str, float], dict[str, float], dict[str, float],
    dict[str, float], dict[str, float],
]:
    """Extract aligned aggregate inference maps from a statsmodels fit."""
    import numpy as np

    model = getattr(fit, "model", None)
    if fixed_effects_only and hasattr(fit, "fe_params"):
        raw_params = getattr(fit, "fe_params")
    else:
        raw_params = getattr(fit, "params", None)
    if raw_params is None:
        raise TypeError("fitted result does not expose model coefficients")
    values = np.asarray(raw_params, dtype=float).reshape(-1)
    raw_index = getattr(raw_params, "index", None)
    names = list(raw_index) if raw_index is not None else list(
        getattr(model, "exog_names", ()) or ()
    )
    if len(names) < len(values):
        names.extend(f"term_{i + 1}" for i in range(len(names), len(values)))
    normalized_names: list[str] = []
    for raw_name in names[:len(values)]:
        # Patsy formula labels such as ``I(time ** 2)`` and
        # ``C(group)[T.case]`` are legitimate fitted terms but contain spaces
        # or brackets outside the result identifier alphabet. Canonicalize the
        # label deterministically; never substitute a raw data value.
        name = str(raw_name).replace("**", "^").replace("[", "(").replace("]", ")")
        name = re.sub(r"[^A-Za-z0-9_.():^#]", "", name)[:40]
        normalized_names.append(
            _method_quantity_name(name, field="coefficient name")
        )
    names = normalized_names
    if len(set(names)) != len(names):
        raise ValueError("fitted coefficient names must be unique")

    def vector(attribute: str) -> dict[str, float]:
        raw = getattr(fit, attribute, None)
        if raw is None:
            return {}
        vals = np.asarray(raw, dtype=float).reshape(-1)[:len(values)]
        return {
            name: float(value) for name, value in zip(names, vals, strict=True)
            if np.isfinite(value)
        }

    estimates = {
        name: float(value) for name, value in zip(names, values, strict=True)
        if np.isfinite(value)
    }
    if not estimates:
        raise ValueError("fitted model has no finite coefficients")
    standard_errors = vector("bse")
    p_values = {
        key: value for key, value in vector("pvalues").items()
        if 0.0 <= value <= 1.0
    }
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    try:
        interval = np.asarray(fit.conf_int(), dtype=float)
        if interval.ndim == 2 and interval.shape[1] >= 2:
            for name, row in zip(names, interval[:len(names)], strict=True):
                lo, hi = float(row[0]), float(row[1])
                if np.isfinite(lo) and np.isfinite(hi) and lo <= estimates.get(name, lo) <= hi:
                    lower[name], upper[name] = lo, hi
    except Exception:  # noqa: BLE001 - interval support differs by results class
        pass
    return estimates, standard_errors, p_values, lower, upper


def _longitudinal_structure(
    fit: Any,
) -> tuple[int, int, float, Any]:
    """Return records, cluster count, mean size, and group vector."""
    import numpy as np

    model = getattr(fit, "model", None)
    groups = np.asarray(getattr(model, "groups", ())).reshape(-1)
    n = _method_positive_int(getattr(model, "nobs", len(groups)), field="model.nobs")
    if len(groups) != n:
        raise ValueError("model.groups must contain one value per fitted record")
    _, counts = np.unique(groups, return_counts=True)
    if len(counts) < 2:
        raise ValueError("longitudinal models require at least two clusters")
    return n, int(len(counts)), float(np.mean(counts)), groups


def _lag1_residual_diagnostic(
    fit: Any, groups: Any, time_values: Any | None,
) -> str:
    """Conservative within-cluster lag-one residual-correlation check."""
    import numpy as np

    if time_values is None:
        return "warn"
    try:
        residuals = np.asarray(getattr(fit, "resid"), dtype=float).reshape(-1)
        times = np.asarray(time_values, dtype=float).reshape(-1)
        groups = np.asarray(groups).reshape(-1)
    except Exception:  # noqa: BLE001
        return "warn"
    if not (len(residuals) == len(times) == len(groups)):
        raise ValueError("time_values must align with fitted records")
    left: list[float] = []
    right: list[float] = []
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        order = positions[np.argsort(times[positions], kind="stable")]
        if len(order) > 1:
            left.extend(residuals[order[:-1]])
            right.extend(residuals[order[1:]])
    if len(left) < 8 or np.std(left) == 0 or np.std(right) == 0:
        return "warn"
    correlation = float(np.corrcoef(left, right)[0, 1])
    return "pass" if np.isfinite(correlation) and abs(correlation) < 0.3 else "warn"


def _mixed_converged(fit: Any) -> Any:
    import numpy as np

    value = getattr(fit, "converged", None)
    return bool(value) if isinstance(value, (bool, np.bool_)) else "warn"


def from_growth_curve(
    fit: Any, *, time_values: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit fixed growth parameters from a statsmodels ``MixedLM`` fit."""
    if "MixedLMResults" not in type(fit).__name__:
        raise TypeError("fit must be a fitted statsmodels MixedLM result")
    n, clusters, cluster_size, groups = _longitudinal_structure(fit)
    estimates, errors, p_values, lower, upper = _fitted_method_maps(
        fit, fixed_effects_only=True,
    )
    import numpy as np
    covariance: Any = np.asarray(getattr(fit, "cov_re", ()), dtype=float)
    random_ok = bool(covariance.size and np.all(np.isfinite(covariance)))
    diag = _method_diagnostics({
        "cluster_count": clusters,
        "cluster_size": cluster_size,
        "serial_correlation": _lag1_residual_diagnostic(fit, groups, time_values),
        "convergence": _mixed_converged(fit),
        "random_effect_structure": "pass" if random_ok else "warn",
    }, diagnostics)
    from_method(
        "growth_curve", n=n, diagnostics=diag, estimates=estimates,
        standard_errors=errors, p_values=p_values, ci_lower=lower,
        ci_upper=upper, clusters=clusters, records=n,
        uncertainty_type="classical", **metadata,
    )


def from_gee(
    fit: Any, *, time_values: Any | None = None,
    sensitivity_fit: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit a statsmodels GEE population-average coefficient result."""
    if "GEEResults" not in type(fit).__name__:
        raise TypeError("fit must be a fitted statsmodels GEE result")
    n, clusters, cluster_size, groups = _longitudinal_structure(fit)
    covariance_type = str(getattr(fit, "cov_type", "")).casefold()
    if covariance_type != "robust":
        raise ValueError(
            "GEE fit must use statsmodels cov_type='robust' sandwich covariance"
        )
    estimates, errors, p_values, lower, upper = _fitted_method_maps(fit)
    import numpy as np
    converged = getattr(fit, "converged", None)
    sensitivity: Any = "warn"
    metrics: dict[str, float] | None = None
    if sensitivity_fit is not None:
        if ("GEEResults" not in type(sensitivity_fit).__name__
                or str(getattr(sensitivity_fit, "cov_type", "")).casefold()
                != "robust"):
            raise ValueError("GEE sensitivity_fit must use robust covariance")
        comparison, *_ = _fitted_method_maps(sensitivity_fit)
        if set(comparison) != set(estimates):
            raise ValueError("GEE sensitivity fit must contain the same coefficients")
        sensitivity = max(
            abs(estimates[key] - comparison[key]) for key in estimates
        )
        metrics = {"working_correlation_max_abs_change": sensitivity}
    diag = _method_diagnostics({
        "cluster_count": clusters,
        "cluster_size": cluster_size,
        "serial_correlation": _lag1_residual_diagnostic(fit, groups, time_values),
        "convergence": (
            bool(converged)
            if isinstance(converged, (bool, np.bool_)) else "warn"
        ),
        # One fit cannot prove robustness to the working-correlation choice.
        "working_correlation_sensitivity": sensitivity,
    }, diagnostics)
    from_method(
        "gee", n=n, diagnostics=diag, estimates=estimates,
        standard_errors=errors, p_values=p_values, ci_lower=lower,
        ci_upper=upper, clusters=clusters, records=n,
        metrics=metrics, uncertainty_type="robust", **metadata,
    )


def _hausman_p_value(
    random_fit: Any,
    fixed_fit: Any,
    *,
    expected_groups: int | None = None,
) -> float:
    """Compute the classic coefficient/covariance Hausman statistic."""
    import numpy as np
    from scipy.stats import chi2

    re_params = getattr(random_fit, "fe_params", getattr(random_fit, "params", None))
    fe_params = getattr(fixed_fit, "params", None)
    re_index = getattr(re_params, "index", None)
    fe_index = getattr(fe_params, "index", None)
    re_names = list(
        re_index if re_index is not None else getattr(random_fit.model, "exog_names", ())
    )
    fe_names = list(
        fe_index if fe_index is not None else getattr(fixed_fit.model, "exog_names", ())
    )
    excluded = {"intercept", "const"}
    common = [
        name for name in re_names
        if name in fe_names and str(name).casefold() not in excluded
    ]
    if not common:
        raise ValueError("fixed and random models have no comparable slope terms")
    re_n = _safe_float(getattr(random_fit.model, "nobs", None))
    fe_n = _safe_float(getattr(fixed_fit.model, "nobs", None))
    if re_n is not None and fe_n is not None and re_n != fe_n:
        raise ValueError("Hausman models must use the same fitted records")
    if (expected_groups is not None
            and len(fe_names) - len(common) < expected_groups - 1):
        raise ValueError(
            "Hausman comparison model does not contain the required panel "
            "fixed effects"
        )

    def values(obj: Any, names: list[Any], selected: list[Any]) -> Any:
        raw = np.asarray(obj, dtype=float).reshape(-1)
        return np.asarray([raw[names.index(name)] for name in selected])

    difference = values(fe_params, fe_names, common) - values(re_params, re_names, common)
    re_cov = np.asarray(random_fit.cov_params(), dtype=float)
    fe_cov = np.asarray(fixed_fit.cov_params(), dtype=float)
    re_idx = [re_names.index(name) for name in common]
    fe_idx = [fe_names.index(name) for name in common]
    covariance_difference = (
        fe_cov[np.ix_(fe_idx, fe_idx)] - re_cov[np.ix_(re_idx, re_idx)]
    )
    covariance_difference = 0.5 * (
        covariance_difference + covariance_difference.T
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_difference)
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    tolerance = 1e-10 * scale
    if np.any(eigenvalues < -tolerance):
        raise ValueError(
            "Hausman covariance difference is indefinite; use a validated "
            "Mundlak/auxiliary-regression comparison instead"
        )
    supported = eigenvalues > tolerance
    rank = int(np.sum(supported))
    if rank < 1:
        raise ValueError("Hausman covariance difference is singular")
    projection = eigenvectors[:, supported].T @ difference
    statistic = float(np.sum((projection**2) / eigenvalues[supported]))
    if not np.isfinite(statistic) or statistic < -tolerance:
        raise ValueError("Hausman statistic is not finite and non-negative")
    return float(chi2.sf(max(0.0, statistic), rank))


def from_panel_random_effects(
    fit: Any, *, fixed_effects_fit: Any,
    time_values: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit a random-intercept panel fit plus a fitted Hausman comparison."""
    if "MixedLMResults" not in type(fit).__name__:
        raise TypeError("fit must be a fitted statsmodels MixedLM result")
    import numpy as np
    random_covariance: Any = np.asarray(
        getattr(fit, "cov_re", ()), dtype=float,
    )
    if random_covariance.shape != (1, 1) or not np.all(np.isfinite(random_covariance)):
        raise ValueError(
            "panel random-effects helper requires one finite random-intercept variance"
        )
    n, clusters, cluster_size, groups = _longitudinal_structure(fit)
    estimates, errors, p_values, lower, upper = _fitted_method_maps(
        fit, fixed_effects_only=True,
    )
    hausman_p = _hausman_p_value(
        fit, fixed_effects_fit, expected_groups=clusters,
    )
    diag = _method_diagnostics({
        "cluster_count": clusters,
        "cluster_size": cluster_size,
        "serial_correlation": _lag1_residual_diagnostic(fit, groups, time_values),
        "convergence": _mixed_converged(fit),
        "hausman": hausman_p,
    }, diagnostics)
    from_method(
        "panel_random_effects", n=n, diagnostics=diag,
        estimates=estimates, standard_errors=errors, p_values=p_values,
        ci_lower=lower, ci_upper=upper,
        metrics={"hausman_p_value": hausman_p}, clusters=clusters,
        records=n, uncertainty_type="classical", **metadata,
    )


def from_competing_risks(
    fit: Any, *, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Emit final cause-specific cumulative incidences from statsmodels."""
    if type(fit).__name__ != "CumIncidenceRight":
        raise TypeError("fit must be statsmodels CumIncidenceRight")
    import numpy as np
    status: Any = np.asarray(getattr(fit, "status", ())).reshape(-1)
    times: Any = np.asarray(getattr(fit, "times", ()), dtype=float).reshape(-1)
    curves = list(getattr(fit, "cinc", ()))
    n = _method_positive_int(len(status), field="records")
    if not curves or not len(times) or any(len(np.asarray(curve).reshape(-1)) != len(times) for curve in curves):
        raise ValueError("cumulative-incidence curves are missing or misaligned")
    events = int(np.sum(status > 0))
    cause_counts = [int(np.sum(status == cause)) for cause in range(1, len(curves) + 1)]
    estimates = {
        f"cause_{cause}_final": float(np.asarray(curve, dtype=float).reshape(-1)[-1])
        for cause, curve in enumerate(curves, 1)
    }
    if any(not 0.0 <= value <= 1.0 for value in estimates.values()):
        raise ValueError("cumulative incidences must be probabilities")
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    standard_errors: dict[str, float] = {}
    raw_se = getattr(fit, "cinc_se", None)
    if raw_se is not None and len(raw_se) == len(curves):
        for cause, (curve, se_curve) in enumerate(zip(curves, raw_se, strict=True), 1):
            key = f"cause_{cause}_final"
            estimate = estimates[key]
            se = float(np.asarray(se_curve, dtype=float).reshape(-1)[-1])
            if np.isfinite(se) and se >= 0:
                standard_errors[key] = se
                lower[key] = max(0.0, estimate - 1.96 * se)
                upper[key] = min(1.0, estimate + 1.96 * se)
    diag = _method_diagnostics({
        "subject_count": n,
        "event_count": events,
        "at_risk_support": "pass" if n - events > 0 else "warn",
        "cause_specific_events": (
            "pass" if cause_counts and all(value > 0 for value in cause_counts)
            else "warn"
        ),
    }, diagnostics)
    from_method(
        "competing_risks", n=n, diagnostics=diag, estimates=estimates,
        standard_errors=standard_errors or None, ci_lower=lower or None,
        ci_upper=upper or None, subjects=n, events=events, records=n,
        uncertainty_type="classical", **metadata,
    )


def _ph_interval_diagnostics(fit: Any) -> tuple[int, int, int, str, str]:
    """Validate counting-process intervals and derive safe PH diagnostics."""
    import numpy as np
    from scipy.stats import spearmanr

    model = getattr(fit, "model", None)
    stop: Any = np.asarray(getattr(model, "endog", ()), dtype=float).reshape(-1)
    start: Any = np.asarray(getattr(model, "entry", ()), dtype=float).reshape(-1)
    status: Any = np.asarray(getattr(model, "status", ()), dtype=float).reshape(-1)
    raw_groups = getattr(model, "groups", None)
    if raw_groups is None:
        raise ValueError(
            "counting-process PHReg fit must use fit(groups=subject_ids) "
            "for subject-clustered covariance"
        )
    groups = np.asarray(raw_groups).reshape(-1)
    n = len(stop)
    if not n or not (len(start) == len(status) == len(groups) == n):
        raise ValueError("PHReg fit must retain aligned start, stop, status, and groups")
    valid = bool(
        np.all(np.isfinite(start)) and np.all(np.isfinite(stop))
        and np.all(start >= 0) and np.all(stop > start)
        and np.all(np.isin(status, [0, 1]))
    )
    for group in np.unique(groups):
        positions = np.flatnonzero(groups == group)
        order = positions[np.argsort(start[positions], kind="stable")]
        if len(order) > 1 and np.any(start[order[1:]] < stop[order[:-1]]):
            valid = False
    subjects = int(len(np.unique(groups)))
    events = int(np.sum(status == 1))
    ph_status = "warn"
    try:
        residuals = np.asarray(fit.schoenfeld_residuals, dtype=float)
        event_rows = status == 1
        p_values = []
        for column in range(residuals.shape[1]):
            values = residuals[event_rows, column]
            keep = np.isfinite(values) & np.isfinite(stop[event_rows])
            if int(np.sum(keep)) >= 8:
                p_values.append(float(spearmanr(stop[event_rows][keep], values[keep]).pvalue))
        if p_values:
            ph_status = "pass" if min(p_values) >= 0.05 else "warn"
    except Exception:  # noqa: BLE001
        pass
    return n, subjects, events, "pass" if valid else "fail", ph_status


def _from_counting_process_phreg(
    fit: Any, *, method_id: str,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    if "PHRegResults" not in type(fit).__name__:
        raise TypeError("fit must be a fitted statsmodels PHReg result")
    import numpy as np
    n, subjects, events, interval_status, ph_status = _ph_interval_diagnostics(fit)
    estimates, errors, p_values, lower, upper = _fitted_method_maps(fit)
    model = fit.model
    groups = np.asarray(model.groups).reshape(-1)
    status = np.asarray(model.status).reshape(-1)
    event_counts = [int(np.sum(status[groups == group] == 1)) for group in np.unique(groups)]
    defaults = {
        "subject_count": subjects,
        "event_count": events,
        "at_risk_support": "pass" if events < n else "warn",
    }
    if method_id == "recurrent_events":
        defaults["within_subject_dependence"] = (
            "pass" if max(event_counts, default=0) > 1 else "warn"
        )
        defaults["proportional_hazards"] = ph_status
    elif method_id == "time_varying_survival":
        exog = np.asarray(model.exog, dtype=float)
        changes = any(
            len(np.unique(exog[groups == group], axis=0)) > 1
            for group in np.unique(groups)
        )
        defaults.update({
            "interval_integrity": interval_status,
            "proportional_hazards": ph_status,
        })
        if not changes:
            raise ValueError("time-varying survival fit has no within-subject covariate change")
    else:  # pragma: no cover - private caller allowlist
        raise ValueError("unsupported counting-process method")
    from_method(
        method_id, n=n, diagnostics=_method_diagnostics(defaults, diagnostics),
        estimates=estimates, standard_errors=errors, p_values=p_values,
        ci_lower=lower, ci_upper=upper, subjects=subjects, events=events,
        records=n, clusters=subjects, uncertainty_type="cluster_robust",
        **metadata,
    )


def from_recurrent_events(
    fit: Any, *, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Emit an Andersen–Gill PHReg fit with subject-clustered covariance."""
    _from_counting_process_phreg(
        fit, method_id="recurrent_events", diagnostics=diagnostics, **metadata,
    )


def from_time_varying_survival(
    fit: Any, *, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Emit a PHReg start/stop fit whose covariates vary within subject."""
    _from_counting_process_phreg(
        fit, method_id="time_varying_survival", diagnostics=diagnostics,
        **metadata,
    )


# ---------------------------------------------------------------------------
# Helpers — statsmodels / scipy / pandas convenience wrappers
# ---------------------------------------------------------------------------


def _special_regression_diagnostics(
    fit: Any, specific: dict[str, Any], overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Conservative shared diagnostics for typed non-Gaussian regressions."""
    convergence: Any = "warn"
    mle = _safe_attr(fit, "mle_retvals")
    if isinstance(mle, dict) and isinstance(mle.get("converged"), bool):
        convergence = bool(mle["converged"])
    elif isinstance(_safe_attr(fit, "converged"), bool):
        convergence = bool(_safe_attr(fit, "converged"))
    defaults = {
        "convergence": convergence,
        "specification": "warn",
        "influence": "warn",
        "multicollinearity": "warn",
        "heteroskedasticity": "not_applicable",
        "residual_distribution": "not_applicable",
        **specific,
    }
    return _method_diagnostics(defaults, overrides)


def _normal_interval_maps(
    estimates: dict[str, float], standard_errors: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for name, estimate in estimates.items():
        se = standard_errors.get(name)
        if se is not None and se >= 0:
            lower[name] = estimate - 1.959963984540054 * se
            upper[name] = estimate + 1.959963984540054 * se
    return lower, upper


def from_ordinal_model(
    fit: Any, *, diagnostics: dict[str, Any] | None = None,
    proportional_odds: Any = "warn", **metadata: Any,
) -> None:
    """Emit an aggregate statsmodels ``OrderedModel`` fit.

    Category labels and observations never leave the sandbox. Thresholds are
    transformed to the model's ordered cut-point scale and assigned synthetic
    names, preventing outcome labels from becoming a disclosure channel.
    """
    import numpy as np

    model = _safe_attr(fit, "model")
    if model is None or not callable(_safe_attr(model, "transform_threshold_params")):
        raise TypeError("fit must be a statsmodels OrderedModel result")
    k_levels = _safe_int(_safe_attr(model, "k_levels"))
    k_vars = _safe_int(_safe_attr(model, "k_vars"))
    params = np.asarray(_safe_attr(fit, "params"), dtype=float).reshape(-1)
    if k_levels is None or k_levels < 3 or k_vars is None or k_vars < 1:
        raise ValueError("ordinal regression requires at least three ordered categories and one predictor")
    exog_names = list(_safe_attr(model, "exog_names") or [])[:k_vars]
    if len(exog_names) != k_vars:
        exog_names = [f"predictor_{i + 1}" for i in range(k_vars)]
    names = [_method_quantity_name(name) for name in exog_names]
    if len(set(names)) != len(names) or any(
        name.startswith("threshold_") for name in names
    ):
        raise ValueError("predictor names collide with ordinal threshold identifiers")
    estimates = {
        names[i]: float(params[i]) for i in range(k_vars)
        if _safe_float(params[i]) is not None
    }
    thresholds = np.asarray(model.transform_threshold_params(params), dtype=float)[1:-1]
    for index, value in enumerate(thresholds, 1):
        estimates[f"threshold_{index}"] = float(value)
    bse_raw = np.asarray(_safe_attr(fit, "bse"), dtype=float).reshape(-1)
    p_raw = np.asarray(_safe_attr(fit, "pvalues"), dtype=float).reshape(-1)
    ses = {names[i]: float(bse_raw[i]) for i in range(k_vars) if _safe_float(bse_raw[i]) is not None}
    pvals = {names[i]: float(p_raw[i]) for i in range(k_vars) if _safe_float(p_raw[i]) is not None}
    lo, hi = _normal_interval_maps(estimates, ses)
    n = len(_safe_attr(model, "endog"))
    distribution = _safe_attr(model, "distr")
    distr = str(_safe_attr(distribution, "name") or distribution or "")
    link = "probit" if "norm" in distr.casefold() else "logit"
    if proportional_odds != "warn":
        raise ValueError(
            "proportional_odds cannot be promoted without an executable assumption test"
        )
    ordinal_diagnostics = _special_regression_diagnostics(
        fit, {"proportional_odds": "warn"}, diagnostics,
    )
    # OrderedModel fits the restriction but does not test it. A caller status
    # is not executable evidence of proportional odds.
    ordinal_diagnostics["proportional_odds"] = "warn"
    from_method(
        "ordinal_regression", n=n,
        diagnostics=ordinal_diagnostics,
        estimates=estimates, standard_errors=ses, p_values=pvals,
        ci_lower=lo, ci_upper=hi,
        metrics={
            "category_count": float(k_levels),
            "threshold_count": float(k_levels - 1),
            "log_likelihood": _method_float(
                _safe_attr(fit, "llf"), field="log likelihood",
            ),
            "aic": _method_float(_safe_attr(fit, "aic"), field="AIC"),
        },
        model_form=f"ordered_{link}", link=link,
        uncertainty_type="classical", **metadata,
    )


def from_multinomial_model(
    fit: Any, *, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Emit a statsmodels MNLogit fit with synthetic class identifiers."""
    import numpy as np

    model = _safe_attr(fit, "model")
    params = np.asarray(_safe_attr(fit, "params"), dtype=float)
    ses_raw = np.asarray(_safe_attr(fit, "bse"), dtype=float)
    p_raw = np.asarray(_safe_attr(fit, "pvalues"), dtype=float)
    if model is None or params.ndim != 2 or params.shape != ses_raw.shape:
        raise TypeError("fit must be a statsmodels MNLogit result")
    terms = list(_safe_attr(model, "exog_names") or [])
    if len(terms) != params.shape[0]:
        terms = [f"term_{i + 1}" for i in range(params.shape[0])]
    terms = [_method_quantity_name(term) for term in terms]
    if len(set(terms)) != len(terms):
        raise ValueError("multinomial predictor names must be unique")
    estimates: dict[str, float] = {}
    ses: dict[str, float] = {}
    pvals: dict[str, float] = {}
    for equation in range(params.shape[1]):
        for row, term in enumerate(terms):
            key = f"class_{equation + 1}#{term}"
            estimates[key] = float(params[row, equation])
            ses[key] = float(ses_raw[row, equation])
            pvals[key] = float(p_raw[row, equation])
    lo, hi = _normal_interval_maps(estimates, ses)
    endog = np.asarray(_safe_attr(model, "endog"))
    _values, counts = np.unique(endog, return_counts=True)
    categories = int(counts.size)
    if categories < 3:
        raise ValueError("multinomial regression requires at least three outcome categories")
    min_category_n = int(counts.min())
    from_method(
        "multinomial_regression", n=int(endog.size),
        diagnostics=_special_regression_diagnostics(
            fit, {"class_support": min_category_n >= 10}, diagnostics,
        ),
        estimates=estimates, standard_errors=ses, p_values=pvals,
        ci_lower=lo, ci_upper=hi,
        metrics={
            "category_count": float(categories),
            "equation_count": float(params.shape[1]),
            "min_category_n": float(min_category_n),
            "log_likelihood": _method_float(
                _safe_attr(fit, "llf"), field="log likelihood",
            ),
            "aic": _method_float(_safe_attr(fit, "aic"), field="AIC"),
        }, model_form="multinomial_logit", link="logit",
        uncertainty_type="classical", **metadata,
    )


def from_zero_inflated_model(
    fit: Any, *, diagnostics: dict[str, Any] | None = None,
    **metadata: Any,
) -> None:
    """Emit a fitted statsmodels ZIP or ZINB-P model, aggregate only."""
    import numpy as np

    model = _safe_attr(fit, "model")
    params = np.asarray(_safe_attr(fit, "params"), dtype=float).reshape(-1)
    names = list(_safe_attr(model, "exog_names") or []) if model is not None else []
    if model is None or len(names) != len(params) or not any(str(n).startswith("inflate_") for n in names):
        raise TypeError("fit must be a statsmodels zero-inflated count-model result")
    names = [_method_quantity_name(name) for name in names]
    bse = np.asarray(_safe_attr(fit, "bse"), dtype=float).reshape(-1)
    p_raw = np.asarray(_safe_attr(fit, "pvalues"), dtype=float).reshape(-1)
    estimates = {name: float(params[i]) for i, name in enumerate(names)}
    ses = {name: float(bse[i]) for i, name in enumerate(names)}
    pvals = {name: float(p_raw[i]) for i, name in enumerate(names)}
    lo, hi = _normal_interval_maps(estimates, ses)
    endog = np.asarray(_safe_attr(model, "endog"), dtype=float).reshape(-1)
    if endog.size == 0 or np.any(endog < 0) or np.any(endog != np.floor(endog)):
        raise ValueError("zero-inflated models require a non-negative integer outcome")
    mean = float(endog.mean())
    variance = float(endog.var(ddof=1)) if endog.size > 1 else 0.0
    cls = type(model).__name__.casefold()
    form = "zero_inflated_negative_binomial" if "negativebinomial" in cls else "zero_inflated_poisson"
    from_method(
        "zero_inflated_model", n=int(endog.size),
        diagnostics=_special_regression_diagnostics(fit, {
            "zero_process_specification": "warn",
            "overdispersion": variance / mean if mean > 0 else "warn",
        }, diagnostics),
        estimates=estimates, standard_errors=ses, p_values=pvals,
        ci_lower=lo, ci_upper=hi,
        metrics={
            "zero_fraction": float(np.mean(endog == 0)),
            "count_mean": mean,
            "variance_mean_ratio": variance / mean if mean > 0 else 0.0,
            "parameter_count": float(len(params)),
            "log_likelihood": _method_float(
                _safe_attr(fit, "llf"), field="log likelihood",
            ),
            "aic": _method_float(_safe_attr(fit, "aic"), field="AIC"),
        }, model_form=form, link="log", uncertainty_type="classical", **metadata,
    )


def from_spline_model(
    fit: Any, *, basis_df: int, basis: str = "bspline",
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit an aggregate statsmodels spline/non-linear regression fit."""
    import numpy as np

    if basis not in {"bspline", "natural_spline", "restricted_cubic_spline", "polynomial"}:
        raise ValueError("basis is not a supported non-linear basis")
    safe_df = _safe_int(basis_df)
    model = _safe_attr(fit, "model")
    params = np.asarray(_safe_attr(fit, "params"), dtype=float).reshape(-1)
    bse = np.asarray(_safe_attr(fit, "bse"), dtype=float).reshape(-1)
    p_raw = np.asarray(_safe_attr(fit, "pvalues"), dtype=float).reshape(-1)
    if model is None or safe_df is None or safe_df < 2 or not (len(params) == len(bse) == len(p_raw)):
        raise TypeError("fit must be a fitted statsmodels regression and basis_df must be at least two")
    design_info = _safe_attr(_safe_attr(model, "data"), "design_info")
    column_names = list(_safe_attr(design_info, "column_names") or [])
    term_slices = _safe_attr(design_info, "term_name_slices")
    if len(column_names) != len(params) or not hasattr(term_slices, "items"):
        raise TypeError(
            "spline fit must retain Patsy design metadata proving the nonlinear basis"
        )
    tokens = {
        "bspline": ("bs(",),
        "natural_spline": ("cr(",),
        "restricted_cubic_spline": ("cr(",),
        "polynomial": ("I(", "poly("),
    }[basis]
    basis_indices: set[int] = set()
    for term, column_slice in term_slices.items():
        if any(token in str(term) for token in tokens):
            basis_indices.update(range(column_slice.start, column_slice.stop))
    if len(basis_indices) != safe_df:
        raise ValueError(
            "declared basis_df does not match the fitted nonlinear design columns"
        )
    names: list[str] = []
    basis_counter = 0
    for index, column in enumerate(column_names):
        if index in basis_indices:
            basis_counter += 1
            names.append(f"basis_{basis_counter}")
        elif column in {"Intercept", "const"}:
            names.append("intercept")
        else:
            names.append(_method_quantity_name(column, field="covariate name"))
    if len(set(names)) != len(names):
        raise ValueError("covariate names collide with synthetic spline-basis identifiers")
    estimates = {names[i]: float(params[i]) for i in range(len(params))}
    ses = {names[i]: float(bse[i]) for i in range(len(params))}
    pvals = {names[i]: float(p_raw[i]) for i in range(len(params))}
    lo, hi = _normal_interval_maps(estimates, ses)
    n = _safe_int(_safe_attr(fit, "nobs"))
    if n is None or safe_df >= n:
        raise ValueError("basis degrees of freedom must be below the fitted sample size")
    from_method(
        "spline_regression", n=n,
        diagnostics=_special_regression_diagnostics(
            fit, {"degrees_of_freedom_sensitivity": "warn"}, diagnostics,
        ), estimates=estimates, standard_errors=ses, p_values=pvals,
        ci_lower=lo, ci_upper=hi,
        metrics={
            "basis_df": float(safe_df),
            "basis_parameter_count": float(len(basis_indices)),
            "parameter_count": float(len(params)),
            "r_squared": _method_float(
                _safe_attr(fit, "rsquared"), field="R-squared",
            ),
            "aic": _method_float(_safe_attr(fit, "aic"), field="AIC"),
            "bic": _method_float(_safe_attr(fit, "bic"), field="BIC"),
        }, model_form="polynomial_regression" if basis == "polynomial" else "regression_spline",
        basis=basis, uncertainty_type="classical", **metadata,
    )


def _survey_design_covariance(
    influence: Any, *, strata: Any | None, psu: Any | None,
    fpc: Any | None, fpc_mode: str, lonely_psu: str,
    stage1_inclusion_probabilities: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    """Taylor covariance for one- or two-stage stratified cluster samples."""
    import numpy as np

    u = np.asarray(influence, dtype=float)
    if u.ndim == 1:
        u = u[:, None]
    n = u.shape[0]
    if not np.all(np.isfinite(u)):
        raise ValueError("survey influence values must be finite")
    strata_values = np.zeros(n, dtype=object) if strata is None else np.asarray(strata, dtype=object)
    psu_values = np.arange(n, dtype=object)[:, None] if psu is None else np.asarray(psu, dtype=object)
    if psu_values.ndim == 1:
        psu_values = psu_values[:, None]
    if (strata_values.ndim != 1 or len(strata_values) != n
            or psu_values.ndim != 2 or psu_values.shape[0] != n
            or psu_values.shape[1] not in {1, 2}):
        raise ValueError("strata and one- or two-stage PSU arrays must match the data")
    stage_count = psu_values.shape[1]
    stage1_probabilities = (
        None if stage1_inclusion_probabilities is None
        else np.asarray(stage1_inclusion_probabilities, dtype=float)
    )
    if stage1_probabilities is not None and (
        stage1_probabilities.ndim != 1 or len(stage1_probabilities) != n
        or not np.all(np.isfinite(stage1_probabilities))
        or np.any(stage1_probabilities <= 0)
        or np.any(stage1_probabilities > 1)
    ):
        raise ValueError("stage-one inclusion probabilities must be finite in (0, 1]")
    if stage_count == 1 and stage1_probabilities is not None:
        raise ValueError("stage-one inclusion probabilities are only used for multistage designs")
    if lonely_psu not in {"fail", "adjust", "certainty"}:
        raise ValueError("lonely_psu must be fail, adjust, or certainty")
    if fpc_mode not in {"fraction", "population_size"}:
        raise ValueError("fpc_mode must be fraction or population_size")

    fpc_values = None if fpc is None else np.asarray(fpc, dtype=float)
    if fpc_values is not None and fpc_values.ndim == 1:
        fpc_values = fpc_values[:, None]
    if fpc_values is not None and (
        fpc_values.ndim != 2 or fpc_values.shape != (n, stage_count)
        or not np.all(np.isfinite(fpc_values))
    ):
        raise ValueError("FPC must have one finite column per sampling stage")

    def sampling_fraction(indexes: list[int], sampled: int, stage: int) -> float:
        if fpc_values is None:
            return 0.0
        values = fpc_values[indexes, stage]
        if np.max(values) - np.min(values) > 1e-10:
            raise ValueError("FPC must be constant within its parent sampling unit")
        declared = float(values[0])
        if fpc_mode == "fraction":
            if not 0 <= declared <= 1:
                raise ValueError("FPC sampling fractions must be in [0, 1]")
            return declared
        if declared < sampled or declared != math.floor(declared):
            raise ValueError(
                "FPC population size must be an integer at least as large as sampled units"
            )
        return sampled / declared

    groups: dict[object, dict[object, list[int]]] = {}
    for index, (stratum, cluster) in enumerate(zip(strata_values, psu_values[:, 0])):
        groups.setdefault(stratum, {}).setdefault(cluster, []).append(index)
    covariance = np.zeros((u.shape[1], u.shape[1]), dtype=float)
    # survey's ``lonely.psu = "adjust"`` centers a singleton PSU at the
    # grand mean PSU contribution, not at zero.  Compute that reference once
    # across all first-stage PSUs before visiting strata so iteration order
    # cannot affect the answer.
    first_stage_totals = [
        u[indexes].sum(axis=0)
        for stratum_clusters in groups.values()
        for indexes in stratum_clusters.values()
    ]
    first_stage_grand_mean = np.vstack(first_stage_totals).mean(axis=0)
    lonely = 0
    certainty = 0
    first_stage_certainty = 0
    adjusted = 0
    total_psus = 0
    noncertainty_strata = 0
    fpc_fractions: list[float] = []
    for stratum_clusters in groups.values():
        totals = np.vstack([
            u[indexes].sum(axis=0) for indexes in stratum_clusters.values()
        ])
        m = totals.shape[0]
        total_psus += m
        indexes = [index for rows in stratum_clusters.values() for index in rows]
        fraction = sampling_fraction(indexes, m, 0)
        fpc_fractions.append(fraction)
        if m == 1:
            lonely += 1
            if fraction == 1.0:
                certainty += 1
                first_stage_certainty += 1
                continue
            if lonely_psu == "certainty":
                raise ValueError(
                    "lonely_psu='certainty' requires an FPC proving a sampling fraction of one"
                )
            if lonely_psu == "fail":
                raise ValueError(
                    "a stratum has one PSU; choose an explicit defensible lonely-PSU policy"
                )
            deviation = totals[0] - first_stage_grand_mean
            covariance += (1.0 - fraction) * np.outer(deviation, deviation)
            adjusted += 1
            noncertainty_strata += 1
            continue
        centered = totals - totals.mean(axis=0)
        covariance += (
            (1.0 - fraction) * m / (m - 1.0) * centered.T @ centered
        )
        noncertainty_strata += 1
    secondary_psus = 0
    if stage_count == 2:
        # Add the conditional second-stage contribution inside each sampled
        # first-stage PSU. Full survey weights already contain both stages'
        # inverse inclusion probabilities, so these linearized totals are on
        # the correct population-total scale.
        second_stage_totals = [
            u[rows].sum(axis=0)
            for stratum_clusters in groups.values()
            for indexes in stratum_clusters.values()
            for rows in ({
                cluster: [index for index in indexes if psu_values[index, 1] == cluster]
                for cluster in dict.fromkeys(psu_values[indexes, 1])
            }).values()
        ]
        second_stage_grand_mean = np.vstack(second_stage_totals).mean(axis=0)
        for stratum_clusters in groups.values():
            for indexes in stratum_clusters.values():
                second_groups: dict[object, list[int]] = {}
                for index in indexes:
                    second_groups.setdefault(psu_values[index, 1], []).append(index)
                totals = np.vstack([
                    u[rows].sum(axis=0) for rows in second_groups.values()
                ])
                m = totals.shape[0]
                secondary_psus += m
                fraction = sampling_fraction(indexes, m, 1)
                fpc_fractions.append(fraction)
                if stage1_probabilities is not None:
                    parent_probabilities = stage1_probabilities[indexes]
                    if np.max(parent_probabilities) - np.min(parent_probabilities) > 1e-10:
                        raise ValueError(
                            "stage-one inclusion probability must be constant within each sampled PSU"
                        )
                    parent_probability = float(parent_probabilities[0])
                elif fpc_values is not None:
                    parent_probability = sampling_fraction(indexes, len(stratum_clusters), 0)
                else:
                    raise ValueError(
                        "multistage Taylor variance requires stage-one inclusion probabilities or an equal-probability first-stage FPC"
                    )
                if m == 1:
                    lonely += 1
                    if fraction == 1.0:
                        certainty += 1
                        continue
                    if lonely_psu == "certainty":
                        raise ValueError(
                            "lonely_psu='certainty' requires an FPC proving a sampling fraction of one"
                        )
                    if lonely_psu == "fail":
                        raise ValueError(
                            "a second-stage sampling unit is lonely; choose an explicit policy"
                        )
                    deviation = totals[0] - second_stage_grand_mean
                    covariance += parent_probability * (1.0 - fraction) * np.outer(deviation, deviation)
                    adjusted += 1
                    continue
                centered = totals - totals.mean(axis=0)
                covariance += (
                    parent_probability * (1.0 - fraction)
                    * m / (m - 1.0) * centered.T @ centered
                )
    design_df = total_psus - noncertainty_strata - first_stage_certainty
    if design_df < 1:
        raise ValueError("survey design has no residual design degrees of freedom")
    return covariance, {
        "strata_count": float(len(groups)), "psu_count": float(total_psus),
        "lonely_strata_count": float(lonely), "design_df": float(design_df),
        "lonely_certainty_count": float(certainty),
        "lonely_adjusted_count": float(adjusted),
        "fpc_fraction_min": min(fpc_fractions, default=0.0),
        "fpc_fraction_max": max(fpc_fractions, default=0.0),
        "stage_count": float(stage_count),
        "secondary_psu_count": float(secondary_psus),
    }


def _survey_replicate_covariance(
    full: Any, replicates: Any, *, method: str, fay_rho: float,
    mse: bool | None = None, scale: float | None = None,
    rscales: Any | None = None,
) -> tuple[Any, dict[str, float]]:
    import numpy as np

    full_values = np.asarray(full, dtype=float).reshape(-1)
    replicate_values = np.asarray(replicates, dtype=float)
    if replicate_values.ndim == 1:
        replicate_values = replicate_values[:, None]
    if replicate_values.ndim != 2 or replicate_values.shape[1] != len(full_values):
        raise ValueError("replicate estimates have an invalid shape")
    count = replicate_values.shape[0]
    if count < 2 or not np.all(np.isfinite(replicate_values)):
        raise ValueError("at least two finite replicate estimates are required")
    if mse is None:
        # Mirrors survey::svrVar defaults: bootstrap designs center at the
        # replicate mean unless MSE is requested; BRR/Fay/JK center at the
        # full-sample estimate under their standard design conventions.
        mse = method != "bootstrap"
    if not isinstance(mse, bool):
        raise TypeError("replicate_mse must be boolean")
    center = full_values if mse else replicate_values.mean(axis=0)
    centered = replicate_values - center
    if method == "brr":
        default_scale = 1.0 / count
    elif method == "fay":
        if not 0 <= fay_rho < 1:
            raise ValueError("Fay rho must be in [0, 1)")
        default_scale = 1.0 / (count * (1.0 - fay_rho) ** 2)
    elif method == "jackknife":
        default_scale = (count - 1.0) / count
    elif method == "bootstrap":
        default_scale = 1.0 / (count - 1.0)
    else:
        raise ValueError("replicate_method must be brr, fay, jackknife, or bootstrap")
    scale_value = default_scale if scale is None else float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0:
        raise ValueError("replicate scale must be finite and positive")
    if rscales is None:
        rscale_values = np.ones(count, dtype=float)
    else:
        rscale_values = np.asarray(rscales, dtype=float).reshape(-1)
        if (len(rscale_values) != count or not np.all(np.isfinite(rscale_values))
                or np.any(rscale_values <= 0)):
            raise ValueError("replicate rscales must be one finite positive value per replicate")
    covariance = scale_value * (centered * rscale_values[:, None]).T @ centered
    return covariance, {
        "replicate_mse": float(mse), "replicate_scale": scale_value,
        "replicate_rscale_min": float(rscale_values.min()),
        "replicate_rscale_max": float(rscale_values.max()),
    }


def _survey_critical_and_pvalue(estimate: float, se: float, df: int) -> tuple[float, float]:
    if se <= 0:
        return 1.959963984540054, 0.0 if estimate != 0 else 1.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df)), float(2 * t.sf(abs(estimate / se), df))
    except Exception:  # noqa: BLE001
        z = abs(estimate / se)
        return 1.959963984540054, math.erfc(z / math.sqrt(2.0))


def _survey_inputs(values: Any, weights: Any) -> tuple[Any, Any, int, float, float]:
    import numpy as np

    y = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if y.ndim != 1 or w.ndim != 1 or len(y) != len(w) or len(y) < 2:
        raise ValueError("survey values and weights must be matching one-dimensional arrays")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("survey values must be finite and probability weights strictly positive")
    total_weight = float(w.sum())
    effective_n = total_weight ** 2 / float(np.dot(w, w))
    weight_cv = float(w.std(ddof=1) / w.mean())
    return y, w, len(y), effective_n, weight_cv


def from_survey_mean(
    values: Any, weights: Any, *, proportion: bool = False,
    strata: Any | None = None, psu: Any | None = None,
    fpc: Any | None = None, fpc_mode: str = "fraction",
    replicate_weights: Any | None = None, replicate_method: str | None = None,
    fay_rho: float = 0.0, lonely_psu: str = "fail",
    replicate_mse: bool | None = None, replicate_scale: float | None = None,
    replicate_rscales: Any | None = None,
    stage1_inclusion_probabilities: Any | None = None,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Estimate a survey mean/proportion with Taylor or replicate variance."""
    import numpy as np

    y, w, n, effective_n, weight_cv = _survey_inputs(values, weights)
    if proportion and not np.all(np.isin(y, (0.0, 1.0))):
        raise ValueError("survey proportions require a binary 0/1 outcome")
    estimate = float(np.dot(w, y) / w.sum())
    influence = w * (y - estimate) / w.sum()
    if replicate_weights is not None:
        if fpc is not None or stage1_inclusion_probabilities is not None:
            raise ValueError("replicate weights cannot be combined with separate FPC or stage probabilities")
        rw = np.asarray(replicate_weights, dtype=float)
        if rw.ndim != 2 or rw.shape[0] != n or np.any(~np.isfinite(rw)) or np.any(rw < 0):
            raise ValueError("replicate weights must be a finite non-negative n by R matrix")
        denominators = rw.sum(axis=0)
        if np.any(denominators <= 0):
            raise ValueError("every replicate must have positive total weight")
        replicate_estimates = (rw.T @ y) / denominators
        method = replicate_method or ""
        covariance, replicate_metadata = _survey_replicate_covariance(
            [estimate], replicate_estimates[:, None], method=method,
            fay_rho=fay_rho, mse=replicate_mse, scale=replicate_scale,
            rscales=replicate_rscales,
        )
        design = {
            "strata_count": 0.0, "psu_count": 0.0,
            "lonely_strata_count": 0.0,
            "lonely_certainty_count": 0.0, "lonely_adjusted_count": 0.0,
            "design_df": float(rw.shape[1] - 1),
            "fpc_fraction_min": 0.0, "fpc_fraction_max": 0.0,
            "replicate_count": float(rw.shape[1]),
            "stage_count": 0.0, "secondary_psu_count": 0.0,
            **replicate_metadata,
        }
        variance_method = method
    else:
        if (replicate_method is not None or replicate_mse is not None
                or replicate_scale is not None or replicate_rscales is not None):
            raise ValueError("replicate settings require replicate_weights")
        covariance, design = _survey_design_covariance(
            influence, strata=strata, psu=psu, fpc=fpc,
            fpc_mode=fpc_mode, lonely_psu=lonely_psu,
            stage1_inclusion_probabilities=stage1_inclusion_probabilities,
        )
        design["replicate_count"] = 0.0
        variance_method = "taylor_linearization"
    variance = float(covariance[0, 0])
    if variance < 0 or not math.isfinite(variance):
        raise ValueError("survey variance is not finite and non-negative")
    se = math.sqrt(variance)
    # With-replacement weighted reference: each observation is its own PSU.
    reference_cov, _ = _survey_design_covariance(
        influence, strata=None, psu=None, fpc=None,
        fpc_mode="fraction", lonely_psu="fail",
    )
    reference_variance = float(reference_cov[0, 0])
    design_effect = variance / reference_variance if reference_variance > 0 else 1.0
    df = max(1, int(design["design_df"]))
    critical, p_value = _survey_critical_and_pvalue(estimate, se, df)
    name = "proportion" if proportion else "mean"
    lower = estimate - critical * se
    upper = estimate + critical * se
    if proportion:
        lower, upper = max(0.0, lower), min(1.0, upper)
    method_id = "survey_proportion" if proportion else "survey_mean"
    diag = _method_diagnostics({
        "weight_distribution": weight_cv,
        "design_effect": design_effect,
        "effective_sample_size": effective_n,
        "strata_psu_support": design["design_df"] >= 1,
        "variance_estimator": "pass",
        "lonely_psu": "pass" if design["lonely_strata_count"] == 0 else "warn",
    }, diagnostics)
    from_method(
        method_id, n=n, diagnostics=diag,
        estimates={name: estimate}, standard_errors={name: se},
        p_values={name: p_value}, ci_lower={name: lower}, ci_upper={name: upper},
        metrics={
            "variance": variance, "reference_variance": reference_variance,
            "design_effect": design_effect, "effective_sample_size": effective_n,
            "weight_cv": weight_cv, **design,
        }, weight_type="probability", variance_method=variance_method,
        lonely_psu_handling=lonely_psu,
        uncertainty_type="design_based", **metadata,
    )


def from_survey_regression(
    outcome: Any, design_matrix: Any, weights: Any, *,
    predictor_names: list[str], strata: Any | None = None,
    psu: Any | None = None, fpc: Any | None = None,
    fpc_mode: str = "fraction", replicate_weights: Any | None = None,
    replicate_method: str | None = None, fay_rho: float = 0.0,
    replicate_mse: bool | None = None, replicate_scale: float | None = None,
    replicate_rscales: Any | None = None,
    lonely_psu: str = "fail", diagnostics: dict[str, Any] | None = None,
    stage1_inclusion_probabilities: Any | None = None,
    **metadata: Any,
) -> None:
    """Fit probability-weighted linear regression with design covariance."""
    import numpy as np

    y, w, n, effective_n, weight_cv = _survey_inputs(outcome, weights)
    x = np.asarray(design_matrix, dtype=float)
    if x.ndim != 2 or x.shape[0] != n or x.shape[1] != len(predictor_names):
        raise ValueError("design_matrix and predictor_names do not match the survey rows")
    if not np.all(np.isfinite(x)) or x.shape[1] >= n:
        raise ValueError("survey regression design must be finite with fewer columns than rows")
    names = [_method_quantity_name(name, field="predictor name") for name in predictor_names]
    if len(set(names)) != len(names):
        raise ValueError("survey regression predictor names must be unique")
    bread_matrix = x.T @ (w[:, None] * x)
    if np.linalg.matrix_rank(bread_matrix) != x.shape[1]:
        raise ValueError("survey regression design matrix is rank deficient")
    # Solve directly instead of materializing the inverse. This is both less
    # work and more numerically stable for weighted designs near the edge of
    # estimability. The influence scores use the same stable solve below.
    beta = np.linalg.solve(bread_matrix, x.T @ (w * y))
    residual = y - x @ beta
    score = w[:, None] * residual[:, None] * x
    influence = np.linalg.solve(bread_matrix, score.T).T
    if replicate_weights is not None:
        if fpc is not None or stage1_inclusion_probabilities is not None:
            raise ValueError("replicate weights cannot be combined with separate FPC or stage probabilities")
        rw = np.asarray(replicate_weights, dtype=float)
        if rw.ndim != 2 or rw.shape[0] != n or np.any(~np.isfinite(rw)) or np.any(rw < 0):
            raise ValueError("replicate weights must be a finite non-negative n by R matrix")
        replicate_betas = []
        for column in range(rw.shape[1]):
            wr = rw[:, column]
            matrix = x.T @ (wr[:, None] * x)
            if wr.sum() <= 0 or np.linalg.matrix_rank(matrix) != x.shape[1]:
                raise ValueError("a replicate-weight regression design is singular")
            replicate_betas.append(np.linalg.solve(matrix, x.T @ (wr * y)))
        method = replicate_method or ""
        covariance, replicate_metadata = _survey_replicate_covariance(
            beta, replicate_betas, method=method, fay_rho=fay_rho,
            mse=replicate_mse, scale=replicate_scale,
            rscales=replicate_rscales,
        )
        design = {
            "strata_count": 0.0, "psu_count": 0.0,
            "lonely_strata_count": 0.0,
            "lonely_certainty_count": 0.0, "lonely_adjusted_count": 0.0,
            "design_df": float(rw.shape[1] - 1),
            "fpc_fraction_min": 0.0, "fpc_fraction_max": 0.0,
            "replicate_count": float(rw.shape[1]),
            "stage_count": 0.0, "secondary_psu_count": 0.0,
            **replicate_metadata,
        }
        variance_method = method
    else:
        if (replicate_method is not None or replicate_mse is not None
                or replicate_scale is not None or replicate_rscales is not None):
            raise ValueError("replicate settings require replicate_weights")
        covariance, design = _survey_design_covariance(
            influence, strata=strata, psu=psu, fpc=fpc,
            fpc_mode=fpc_mode, lonely_psu=lonely_psu,
            stage1_inclusion_probabilities=stage1_inclusion_probabilities,
        )
        design["replicate_count"] = 0.0
        variance_method = "taylor_linearization"
    reference_cov, _ = _survey_design_covariance(
        influence, strata=None, psu=None, fpc=None,
        fpc_mode="fraction", lonely_psu="fail",
    )
    estimates = {name: float(beta[i]) for i, name in enumerate(names)}
    ses: dict[str, float] = {}
    pvals: dict[str, float] = {}
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    metrics: dict[str, float] = {
        "effective_sample_size": effective_n, "weight_cv": weight_cv, **design,
    }
    df = max(1, int(design["design_df"]))
    design_effects: list[float] = []
    for index, name in enumerate(names):
        variance = float(covariance[index, index])
        reference = float(reference_cov[index, index])
        if variance < 0 or not math.isfinite(variance):
            raise ValueError("survey regression covariance is invalid")
        se = math.sqrt(variance)
        critical, p_value = _survey_critical_and_pvalue(float(beta[index]), se, df)
        ses[name] = se
        pvals[name] = p_value
        lower[name] = float(beta[index]) - critical * se
        upper[name] = float(beta[index]) + critical * se
        deff = variance / reference if reference > 0 else 1.0
        design_effects.append(deff)
        metrics[f"variance#{name}"] = variance
        metrics[f"deff#{name}"] = deff
    diag = _method_diagnostics({
        "weight_distribution": weight_cv,
        "design_effect": max(design_effects, default=1.0),
        "effective_sample_size": effective_n,
        "strata_psu_support": design["design_df"] >= 1,
        "variance_estimator": "pass",
        "lonely_psu": "pass" if design["lonely_strata_count"] == 0 else "warn",
    }, diagnostics)
    from_method(
        "survey_regression", n=n, diagnostics=diag,
        estimates=estimates, standard_errors=ses, p_values=pvals,
        ci_lower=lower, ci_upper=upper, metrics=metrics,
        weight_type="probability", variance_method=variance_method,
        lonely_psu_handling=lonely_psu,
        uncertainty_type="design_based", **metadata,
    )


def _reliability_statistics(items: Any) -> tuple[float, float, float]:
    """Standardized alpha, one-factor omega total, and min item-rest r.

    The fitted one-factor model is delegated to the maintained
    ``factor_analyzer`` implementation.  This helper returns aggregates only;
    loadings, scores, and row values are deliberately not exposed.
    """
    import numpy as np
    from factor_analyzer import FactorAnalyzer

    values = np.asarray(items, dtype=float)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("reliability requires a two-dimensional matrix with at least three items")
    if not np.all(np.isfinite(values)) or np.any(np.std(values, axis=0, ddof=1) <= 0):
        raise ValueError("reliability items must be finite and non-constant")
    standardized = (values - values.mean(axis=0)) / values.std(axis=0, ddof=1)
    correlation = np.corrcoef(standardized, rowvar=False)
    k = values.shape[1]
    mean_correlation = float((correlation.sum() - k) / (k * (k - 1)))
    denominator = 1.0 + (k - 1.0) * mean_correlation
    if denominator <= 0:
        raise ValueError("item covariance is incompatible with a reliability scale")
    alpha = float(k * mean_correlation / denominator)
    item_rest = []
    for index in range(k):
        rest = standardized[:, np.arange(k) != index].sum(axis=1)
        if np.std(rest, ddof=1) <= 0:
            raise ValueError("an item-rest score is constant")
        item_rest.append(float(np.corrcoef(standardized[:, index], rest)[0, 1]))
    minimum_item_rest = min(item_rest)
    if minimum_item_rest < 0:
        raise ValueError(
            "negative item-rest correlation detected; explicitly declare reverse_items before analysis"
        )
    fit = FactorAnalyzer(n_factors=1, rotation=None, method="minres")
    fit.fit(standardized)
    loadings = np.asarray(fit.loadings_, dtype=float).reshape(-1)
    uniqueness = np.asarray(fit.get_uniquenesses(), dtype=float).reshape(-1)
    if (len(loadings) != k or len(uniqueness) != k
            or not np.all(np.isfinite(loadings))
            or not np.all(np.isfinite(uniqueness)) or np.any(uniqueness < 0)):
        raise ValueError("one-factor omega fit is inadmissible")
    # A factor's global sign is arbitrary. Align it before summing loadings;
    # mixed signs have already been guarded by non-negative item-rest scores.
    if float(loadings.sum()) < 0:
        loadings = -loadings
    common = float(loadings.sum() ** 2)
    omega = common / (common + float(uniqueness.sum()))
    if not (0 <= alpha <= 1 and 0 <= omega <= 1):
        raise ValueError("reliability coefficient is outside [0, 1]")
    return alpha, omega, minimum_item_rest


def from_reliability(
    items: Any, *, reverse_items: list[int] | tuple[int, ...] | None = None,
    bootstrap_replicates: int = 500, seed: int = 20260822,
    diagnostics: dict[str, Any] | None = None, **metadata: Any,
) -> None:
    """Emit standardized alpha and one-factor omega with bootstrap intervals.

    Item reversal is never guessed.  Callers must explicitly identify zero-
    based columns; each selected numeric item is reversed around its observed
    min+max inside the sandbox.  Negative corrected item-rest correlations
    fail closed, which prevents an apparently low scale coefficient from
    silently masking a direction mistake.
    """
    import numpy as np

    values = np.asarray(items, dtype=float)
    if values.ndim != 2 or values.shape[0] < 10 or values.shape[1] < 3:
        raise ValueError("reliability requires at least 10 rows and three items")
    if not np.all(np.isfinite(values)):
        raise ValueError("reliability currently requires complete finite item data")
    reversed_indexes = list(reverse_items or ())
    if (len(set(reversed_indexes)) != len(reversed_indexes)
            or any(isinstance(index, bool) or not isinstance(index, int)
                   or index < 0 or index >= values.shape[1]
                   for index in reversed_indexes)):
        raise ValueError("reverse_items must contain unique zero-based item indexes")
    adjusted = values.copy()
    for index in reversed_indexes:
        column = adjusted[:, index]
        if float(column.max()) == float(column.min()):
            raise ValueError("a reversed item is constant")
        adjusted[:, index] = float(column.min() + column.max()) - column
    repetitions = _safe_int(bootstrap_replicates)
    safe_seed = _safe_int(seed)
    if repetitions is None or repetitions < 200 or repetitions > 5000:
        raise ValueError("bootstrap_replicates must be an integer from 200 to 5000")
    if safe_seed is None or safe_seed < 0:
        raise ValueError("seed must be a non-negative integer")
    alpha, omega, min_item_rest = _reliability_statistics(adjusted)
    generator = np.random.default_rng(safe_seed)
    bootstrap_alpha: list[float] = []
    bootstrap_omega: list[float] = []
    for _ in range(repetitions):
        indexes = generator.integers(0, adjusted.shape[0], size=adjusted.shape[0])
        try:
            alpha_rep, omega_rep, _ = _reliability_statistics(adjusted[indexes])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        bootstrap_alpha.append(alpha_rep)
        bootstrap_omega.append(omega_rep)
    successes = len(bootstrap_alpha)
    if successes < max(200, math.ceil(0.9 * repetitions)):
        raise ValueError("fewer than 90% of reliability bootstrap fits were admissible")
    alpha_bounds: Any = np.quantile(bootstrap_alpha, [0.025, 0.975])
    omega_bounds: Any = np.quantile(bootstrap_omega, [0.025, 0.975])
    # The boundary contract requires an interval to contain its estimate.
    # Percentile bootstrap intervals can very occasionally miss the original
    # point in skewed samples; taking the hull is conservative and preserves
    # the empirical percentile endpoints.
    alpha_bounds = np.array([min(float(alpha_bounds[0]), alpha),
                             max(float(alpha_bounds[1]), alpha)])
    omega_bounds = np.array([min(float(omega_bounds[0]), omega),
                             max(float(omega_bounds[1]), omega)])
    diag = _method_diagnostics({
        "sampling_adequacy": "pass" if adjusted.shape[0] >= 10 * adjusted.shape[1] else "warn",
        "fit_or_stability": "pass" if successes >= math.ceil(0.95 * repetitions) else "warn",
        "component_or_class_support": "pass",
        "item_count": float(adjusted.shape[1]),
        "omega_or_alpha_interval": "pass",
        "item_direction": "pass",
    }, diagnostics)
    from_method(
        "reliability", n=int(adjusted.shape[0]), diagnostics=diag,
        estimates={"alpha": alpha, "omega_total": omega},
        ci_lower={"alpha": float(alpha_bounds[0]), "omega_total": float(omega_bounds[0])},
        ci_upper={"alpha": float(alpha_bounds[1]), "omega_total": float(omega_bounds[1])},
        metrics={
            "item_count": float(adjusted.shape[1]),
            "reversed_item_count": float(len(reversed_indexes)),
            "min_item_rest_correlation": min_item_rest,
            "bootstrap_replicates": float(repetitions),
            "bootstrap_success_count": float(successes),
        }, seed=safe_seed, uncertainty_type="bootstrap", **metadata,
    )


def from_lm(model: Any, **extra: Any) -> None:
    """Emit a ``linear_regression`` payload from a fitted statsmodels
    result.

    Supports the regression-shape estimators statsmodels exposes:
    ``OLS``, ``GLM`` (Binomial / Poisson / Gaussian / Gamma / …),
    ``Logit``, ``Probit``, ``Poisson``, ``NegativeBinomial``,
    ``PHReg`` (Cox proportional hazards), and ``IV2SLS``. Each class
    needs a slightly different attribute mix:

      * OLS exposes ``rsquared`` / ``rsquared_adj`` / ``fvalue`` /
        ``f_pvalue`` / ``scale``; ``llf`` / ``aic`` / ``bic`` are
        also present.
      * GLM-family results (Logit, Probit, Poisson, NegBin, GLM)
        expose ``prsquared`` (McFadden's R²), ``llf``, ``aic``,
        ``bic``; ``rsquared`` is **not present** — the old helper
        would emit it as ``null`` and ship every GLM payload missing
        all fit metrics.
      * PHReg exposes ``llf`` and the censoring status via
        ``model.status``; ``aic`` / ``bic`` are not on the result
        wrapper (omit cleanly). ``nobs`` is also **not present** on
        ``PHRegResults`` — derive ``n`` from ``model.endog`` shape
        so the sanitizer's required-field check accepts the payload.

    Sklearn models don't expose any of these conventions; for
    sklearn use ``result(type="linear_regression", ...)`` directly.

    Also prints ``model.summary()`` to stdout so the researcher
    sees the conventional regression table in the TUI's raw log
    panel. Stdout never reaches the sanitizer, so this is purely
    for the researcher.
    """
    # Reserved marker: a caller cannot override the method identity
    # inferred below from the fitted object's maintained-library class.
    extra.pop("_registry_method_id", None)
    extra.pop("_via_helper", None)

    try:
        print(model.summary())
    except Exception:  # noqa: BLE001 — never let printing block the emit
        pass

    # Per-class dispatch by capability probe rather than ``isinstance``
    # (avoids importing statsmodels at module load — the runtime is
    # imported on every script, and pulling in statsmodels there
    # would charge the import cost on descriptive-only scripts too).
    inner = _safe_attr(model, "model")
    cls_name = type(model).__name__
    # PHReg: result is ``PHRegResults`` (no -Wrapper suffix); the
    # ``status`` attribute on ``model.model`` is the censoring flag.
    is_cox = cls_name.startswith("PHReg") or (
        inner is not None and hasattr(inner, "status")
    )
    # MixedLM detection. ``MixedLMResultsWrapper`` exposes ``cov_re``
    # (the random-effects covariance matrix) — no other result class
    # does. Goes BEFORE the GLM check so a future ``MixedGLM`` shape
    # doesn't get misclassified.
    is_mixed = (not is_cox) and _safe_attr(model, "cov_re") is not None
    # GLM family. Two paths:
    #   * ``Logit`` / ``Probit`` / ``Poisson`` / ``NegativeBinomial``
    #     result wrappers ship ``prsquared`` (McFadden's R²).
    #   * ``smf.glm(... family=Binomial())`` returns ``GLMResultsWrapper``
    #     which does NOT expose ``prsquared`` but ships ``deviance``
    #     and ``null_deviance`` — compute pseudo-R² from those.
    has_prsq = _safe_float(_safe_attr(model, "prsquared")) is not None
    has_deviance_pair = (
        _safe_float(_safe_attr(model, "deviance")) is not None
        and _safe_float(_safe_attr(model, "null_deviance")) is not None
    )
    is_glm = (not is_cox) and (not is_mixed) and (has_prsq or has_deviance_pair)
    # OLS-shape: anything with finite ``rsquared`` that isn't the
    # above. IV2SLS lands here too — it exposes ``rsquared`` but
    # NotImplementedError on ``llf`` / ``aic`` / ``bic``; ``_safe_attr``
    # absorbs those so the helper still emits the OLS fields it can.
    is_ols = (
        (not is_cox) and (not is_glm) and (not is_mixed)
        and _safe_float(_safe_attr(model, "rsquared")) is not None
    )

    # ``statsmodels`` exposes the design as ``model.model.exog_names``;
    # the response is ``model.model.endog_names``. The first column
    # is "Intercept" for formula-fit models and "const" for
    # ``add_constant(X)`` setups — we keep whichever name was used.
    response = getattr(inner, "endog_names", None) if inner is not None else None
    exog_names = list(getattr(inner, "exog_names", []) or []) if inner else []
    # predictor_variables = exog minus the intercept (sanitizer wants
    # the regressors of interest, not the intercept).
    predictors = [n for n in exog_names if n not in ("const", "Intercept")]

    # Coefficient table. PHReg ships these as bare ndarrays — pair
    # with ``exog_names`` rather than letting ``dict(ndarray)`` raise
    # the helper into silent oblivion.
    coefs = _to_dict(_safe_attr(model, "params"), names=exog_names)
    ses   = _to_dict(_safe_attr(model, "bse"),    names=exog_names)
    tvals = _to_dict(_safe_attr(model, "tvalues"), names=exog_names)
    pvals = _to_dict(_safe_attr(model, "pvalues"), names=exog_names)

    # Sample size. ``nobs`` on the result wrapper works for OLS / GLM
    # but is absent on ``PHRegResults``. Fall back to ``endog`` shape
    # so Cox payloads carry ``n`` instead of failing the sanitizer's
    # ``n`` required-int check.
    n = _safe_int(_safe_attr(model, "nobs"))
    if n is None and inner is not None:
        endog = getattr(inner, "endog", None)
        if endog is not None:
            try:
                n = int(getattr(endog, "shape", (len(endog),))[0])
            except (TypeError, AttributeError):
                n = None
    df_resid = _safe_int(_safe_attr(model, "df_resid"))

    fields: dict[str, Any] = {
        "n": n,
        "response_variable": response,
        "predictor_variables": predictors,
        "coefficients": coefs,
        "standard_errors": ses,
        "t_statistics": tvals,
        "p_values": pvals,
        "degrees_of_freedom": df_resid,
    }

    # Class-specific fit metrics — only emit fields meaningful for
    # this estimator. Shipping ``r_squared: null`` from a GLM (the
    # old behaviour) made every GLM payload trigger a sanitizer
    # transformation "dropped 'r_squared': not a finite number",
    # while leaving the actual fit metrics absent.
    if is_ols:
        for src, dst in (
            ("rsquared",     "r_squared"),
            ("rsquared_adj", "adj_r_squared"),
            ("fvalue",       "f_statistic"),
            ("f_pvalue",     "f_p_value"),
            ("llf",          "log_likelihood"),
            ("aic",          "aic"),
            ("bic",          "bic"),
        ):
            v = _safe_float(_safe_attr(model, src))
            if v is not None:
                fields[dst] = v
        # ``scale`` is the residual variance; sanitizer's
        # ``residual_std_error`` slot expects the standard deviation.
        sigma_sq = _safe_float(_safe_attr(model, "scale"))
        if sigma_sq is not None and sigma_sq >= 0:
            fields["residual_std_error"] = math.sqrt(sigma_sq)

    if is_glm:
        # Prefer ``prsquared`` when present; fall back to McFadden-
        # equivalent computed from deviance ratio for ``GLMResultsWrapper``
        # (``smf.glm(family=Binomial())`` and friends), which doesn't
        # expose ``prsquared``.
        pr2 = _safe_float(_safe_attr(model, "prsquared"))
        if pr2 is None:
            dev = _safe_float(_safe_attr(model, "deviance"))
            null_dev = _safe_float(_safe_attr(model, "null_deviance"))
            if dev is not None and null_dev is not None and null_dev > 0:
                pr2 = 1.0 - dev / null_dev
        if pr2 is not None:
            fields["pseudo_r_squared"] = pr2
        for src, dst in (
            ("llf", "log_likelihood"), ("aic", "aic"), ("bic", "bic"),
        ):
            v = _safe_float(_safe_attr(model, src))
            if v is not None:
                fields[dst] = v
        # Chi-squared LR test vs. the null model. ``llnull`` is the
        # log-likelihood of the intercept-only model; chi² =
        # 2 · (llf − llnull). Two sources by class:
        #   * Logit / Poisson / NegBin result wrappers compute it
        #     automatically and expose ``llnull``.
        #   * ``GLMResultsWrapper`` exposes the same via ``llf`` and
        #     the deviance pair: 2 · (llf - llnull) = null_dev - dev.
        llf = _safe_float(_safe_attr(model, "llf"))
        llnull = _safe_float(_safe_attr(model, "llnull"))
        if llf is not None and llnull is not None and llf >= llnull:
            fields["chi_squared"] = 2.0 * (llf - llnull)
        elif "chi_squared" not in fields:
            dev = _safe_float(_safe_attr(model, "deviance"))
            null_dev = _safe_float(_safe_attr(model, "null_deviance"))
            if dev is not None and null_dev is not None and null_dev >= dev:
                fields["chi_squared"] = null_dev - dev

    if is_mixed:
        # statsmodels MixedLM. Fixed-effects coefficient table is
        # already extracted above (Estimate / SE / z / P>|z|, since
        # MixedLM uses z-tests like a GLM). Mixed-specific fields:
        # variance components, per-level group counts, fit method,
        # ICC for the one-level intercept-only common case.
        #
        # statsmodels' single-grouping MixedLM stashes the column
        # values of the grouping factor in ``model.groups`` (an
        # ndarray, no name). The original column name isn't reachable
        # from the result, so the caller passes ``group_variable``
        # via kwargs. If omitted, default to "group" — the model
        # still gets the cardinality, just keyed by a generic name.
        group_var_name = str(extra.pop("group_variable", None) or "group")
        cov_re_attr = _safe_attr(model, "cov_re")
        re_var: dict[str, float] = {}
        if cov_re_attr is not None:
            try:
                # cov_re is a labelled DataFrame; diagonal entries
                # are variances. For random-intercept-only (k_re=1)
                # there's one diagonal entry. Random-slope adds more.
                if hasattr(cov_re_attr, "iloc"):
                    cov_arr = cov_re_attr.values
                    re_names = list(cov_re_attr.index)
                else:
                    import numpy as _np
                    cov_arr = _np.asarray(cov_re_attr)
                    re_names = [f"re_{i+1}" for i in range(cov_arr.shape[0])]
                for i, rn in enumerate(re_names):
                    v = float(cov_arr[i, i])
                    if not math.isfinite(v) or v < 0:
                        continue
                    # statsmodels labels the intercept random effect
                    # as "Group" by default; remap to bare group name.
                    if str(rn).lower() in ("group", "(intercept)", "intercept"):
                        key = group_var_name
                    else:
                        key = f"{group_var_name}.{rn}"
                    re_var[key] = v
            except Exception:  # noqa: BLE001
                pass
        scale = _safe_float(_safe_attr(model, "scale"))
        if scale is not None and scale >= 0:
            re_var["residual"] = scale
        if re_var:
            fields["random_effects_variance"] = re_var
        # n_groups_per_level: single-grouping statsmodels exposes
        # ``model.model.n_groups`` (the inner-model's int).
        if inner is not None:
            n_g = _safe_int(_safe_attr(inner, "n_groups"))
            if n_g is not None:
                fields["n_groups_per_level"] = {group_var_name: n_g}
        # Fit method.
        reml = _safe_attr(model, "reml")
        if isinstance(reml, bool):
            fields["fit_method"] = "REML" if reml else "ML"
        # ICC for the one-grouping intercept-only case.
        if "random_effects_variance" in fields:
            rev = fields["random_effects_variance"]
            if len(rev) == 2 and "residual" in rev:
                grp_keys = [k for k in rev if k != "residual"]
                if len(grp_keys) == 1:
                    s_u2 = rev[grp_keys[0]]
                    s_e2 = rev["residual"]
                    if s_u2 + s_e2 > 0:
                        fields["icc"] = s_u2 / (s_u2 + s_e2)
        for src, dst in (
            ("llf", "log_likelihood"), ("aic", "aic"), ("bic", "bic"),
        ):
            v = _safe_float(_safe_attr(model, src))
            if v is not None:
                fields[dst] = v

    if is_cox:
        # ``PHRegResults`` carries ``llf`` but not ``aic`` / ``bic`` on
        # the wrapper. Subject + failure counts come from the inner
        # ``model`` — ``model.endog`` is the observed time vector,
        # ``model.status`` the event indicator.
        llf = _safe_float(_safe_attr(model, "llf"))
        if llf is not None:
            fields["log_likelihood"] = llf
        if inner is not None:
            endog = getattr(inner, "endog", None)
            status = getattr(inner, "status", None)
            try:
                if endog is not None:
                    fields["n_subjects"] = int(
                        getattr(endog, "shape", (len(endog),))[0]
                    )
            except (TypeError, AttributeError):
                pass
            try:
                if status is not None:
                    fields["n_failures"] = int(sum(int(v != 0) for v in status))
            except (TypeError, AttributeError):
                pass

    # Cluster-robust SE metadata. statsmodels signals clustering via
    # ``cov_type == "cluster"`` and stashes the cluster assignment
    # vector in ``cov_kwds["groups"]``. We emit:
    #   * ``cluster_variables`` — the column name(s); for multi-way
    #     clustering ``groups`` is a 2-D array, treat each axis as
    #     a separate dimension.
    #   * ``n_clusters`` — cardinality per dimension (the same shape
    #     and disclosure profile as ``fixed_effects``).
    # Listing the cluster level identities is forbidden — only
    # cardinality and column names cross the boundary. Both are
    # already in the dataset schema the model saw.
    cov_type = _safe_attr(model, "cov_type")
    if isinstance(cov_type, str) and cov_type.lower() == "cluster":
        cov_kwds = _safe_attr(model, "cov_kwds") or {}
        groups = cov_kwds.get("groups") if isinstance(cov_kwds, dict) else None
        cluster_names, n_clusters = _extract_cluster_metadata(groups)
        if cluster_names:
            fields["cluster_variables"] = cluster_names
        if n_clusters:
            fields["n_clusters"] = n_clusters
        fields["robust_se_type"] = "cluster"
    else:
        # Non-cluster variance estimator. Map statsmodels' ``cov_type``
        # values onto the sanitizer's canonical robust_se_type enum
        # so the model can interpret the variance flavour at a glance
        # ("hc1" / "hac_newey_west" / "bootstrap") without parsing the
        # raw label. Helpers don't need to flag classical SEs explicitly
        # — absence of ``robust_se_type`` already implies model-based
        # SEs — but emitting it makes the choice legible on
        # ``expand_result(view="full")``.
        rse = _normalise_robust_se_type(cov_type)
        if rse is not None:
            fields["robust_se_type"] = rse

    # Aggregate diagnostics. These derive from the design matrix and
    # residual sums — pure aggregates, no per-observation leak. Add
    # only when computable; if numpy is missing or the model object
    # doesn't expose its design, omit silently rather than failing
    # the whole emit.
    vif = _compute_vif(model, predictors)
    if vif:
        fields["vif"] = vif
    cond = _compute_condition_number(model)
    if cond is not None:
        fields["condition_number"] = cond
    vcov = _compute_vcov(model)
    if vcov:
        fields["vcov"] = vcov
    fields.update(extra)
    registry_method_id = _registry_method_for_lm_fit(
        model, inner=inner, is_cox=is_cox, is_mixed=is_mixed,
        is_glm=is_glm, is_ols=is_ols,
    )
    # ``coefficient_table_with_fit_stats`` is the canonical bucket
    # name covering OLS / GLM / Cox / IV / fixest. ``linear_regression``
    # stays as a back-compat alias in the sanitizer for existing
    # stored payloads; new emissions use the descriptive name.
    payload = {"type": "coefficient_table_with_fit_stats", **fields}
    if registry_method_id is not None:
        payload["_registry_method_id"] = registry_method_id
    _write_result(payload)


def _registry_method_for_lm_fit(
    model: Any, *, inner: Any, is_cox: bool, is_mixed: bool,
    is_glm: bool, is_ols: bool,
) -> str | None:
    """Return an exact registry method for a fitted statsmodels object.

    The regression sanitizer is intentionally a structural bucket, not a
    method identity.  Workflow binding therefore uses this code-owned
    marker only when the fitted class/family/link identifies one registry
    method unambiguously. Unsupported GLM families/links and other
    regression-like objects still emit normally, but cannot satisfy a
    registry-approved workflow through this legacy bucket.
    """
    if is_cox:
        return "cox_proportional_hazards"
    if is_mixed:
        return "linear_mixed_effects"

    inner_name = type(inner).__name__.lower() if inner is not None else ""
    result_name = type(model).__name__.lower()
    class_surface = f"{inner_name} {result_name}"
    if "iv2sls" in class_surface:
        return "instrumental_variables"
    if "negativebinomial" in class_surface:
        return "negative_binomial_regression"
    if "probit" in class_surface:
        return "probit_regression"
    if "logit" in class_surface:
        return "logistic_regression"
    if "poisson" in class_surface:
        return "poisson_regression"

    if is_glm and inner is not None:
        family = _safe_attr(inner, "family")
        family_name = type(family).__name__.lower() if family is not None else ""
        link = _safe_attr(family, "link") if family is not None else None
        link_name = type(link).__name__.lower() if link is not None else ""
        if family_name == "binomial":
            if link_name == "logit":
                return "logistic_regression"
            if link_name == "probit":
                return "probit_regression"
            return None
        if family_name == "poisson":
            return "poisson_regression"
        if family_name == "negativebinomial":
            return "negative_binomial_regression"
        if family_name == "gaussian" and link_name in ("identity", ""):
            return "linear_regression"
        return None

    # OLS, WLS and GLS are all linear-regression estimators; IV2SLS was
    # split out above before this capability bucket.
    if is_ols and any(name in class_surface for name in ("ols", "wls", "gls")):
        return "linear_regression"
    return None


def _normalise_robust_se_type(cov_type: Any) -> str | None:
    """Map a statsmodels ``cov_type`` label onto the sanitizer's
    canonical robust_se_type enum.

    Returns ``None`` when the label isn't recognised (so the helper
    omits the field rather than smuggling a free-text value through),
    or when the label is the model-based default (``"nonrobust"``)
    where absence of the field already communicates "classical SEs".
    Cluster handling lives in the calling site — it needs the
    cov_kwds["groups"] payload alongside the label, so it stays
    inline rather than routing through here.
    """
    if not isinstance(cov_type, str):
        return None
    key = cov_type.strip().lower()
    if not key or key == "nonrobust":
        return None
    # Heteroskedasticity-consistent. statsmodels accepts both bare
    # ``"HC0"`` / ``"HC1"`` / ... and the prefixed ``"hc0"`` / etc.
    # depending on the call path.
    if key in ("hc0", "hc1", "hc2", "hc3"):
        return key
    # Newey-West HAC. statsmodels: ``"HAC"`` for kernel HAC,
    # ``"hac-panel"`` / ``"hac-groupsum"`` for panel-data flavours.
    # All collapse to ``hac_newey_west`` for the model — the gain
    # from distinguishing them at the wire-format level is small
    # compared to the cost of a wider enum.
    if key.startswith("hac"):
        return "hac_newey_west"
    # Bootstrap covariance — surfaces under names like
    # ``"bootstrap"`` or ``"clusterbootstrap"`` depending on package
    # version. ``cluster`` is handled separately above.
    if "bootstrap" in key:
        return "bootstrap"
    # Robust default in some packages — the typical mapping is HC1.
    # statsmodels' ``"robust"`` doesn't exist canonically; this
    # branch absorbs out-of-tree adapters that emit it.
    if key in ("robust", "sandwich"):
        return "hc1"
    return None


def _extract_cluster_metadata(
    groups: Any,
) -> tuple[list[str], dict[str, int]]:
    """Pull ``(cluster_variables, n_clusters)`` from a statsmodels
    ``cov_kwds["groups"]``.

    The shape is one of:
      * pandas Series — single-cluster, ``.name`` carries the column
        name (or empty if the caller passed a bare ndarray).
      * 1-D ndarray / list — single-cluster, no name available
        (caller used a raw array). Emit a positional label.
      * 2-D ndarray (rows × ndim) — multi-way clustering with
        ``ndim`` dimensions. Each column is one clustering axis.
      * pandas DataFrame — multi-way clustering with column names.

    Returns ``([], {})`` when the groups object isn't recognisable —
    the helper omits the fields rather than emitting incoherent
    metadata.
    """
    if groups is None:
        return [], {}
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return [], {}
    # pandas DataFrame: multi-column → multi-way.
    if hasattr(groups, "columns"):
        names: list[str] = [str(c) for c in groups.columns]
        counts: dict[str, int] = {}
        for c in groups.columns:
            try:
                counts[str(c)] = int(groups[c].nunique())
            except Exception:  # noqa: BLE001
                pass
        return names, counts
    # pandas Series: single-cluster with a name.
    name_attr = getattr(groups, "name", None)
    if name_attr is not None and not isinstance(groups, (list, tuple)):
        try:
            arr = np.asarray(groups)
        except Exception:  # noqa: BLE001
            return [], {}
        if arr.ndim == 1:
            return [str(name_attr)], {str(name_attr): int(np.unique(arr).size)}
    # ndarray / list. 2-D → multi-way (no names); 1-D → single.
    try:
        arr = np.asarray(groups)
    except Exception:  # noqa: BLE001
        return [], {}
    if arr.ndim == 1:
        return ["cluster"], {"cluster": int(np.unique(arr).size)}
    if arr.ndim == 2:
        names = [f"cluster_{i+1}" for i in range(arr.shape[1])]
        counts = {
            names[i]: int(np.unique(arr[:, i]).size)
            for i in range(arr.shape[1])
        }
        return names, counts
    return [], {}


def from_iv(
    model: Any,
    *,
    instrument_variables: list[str] | None = None,
    endogenous_variables: list[str] | None = None,
    first_stage_f: float | None = None,
    weak_instrument_p: float | None = None,
    hansen_j: float | None = None,
    hansen_j_p: float | None = None,
    endogeneity_p: float | None = None,
    **extra: Any,
) -> None:
    """Emit a regression-bucket payload from a 2SLS / IV fit, plus
    the IV-specific diagnostic scalars.

    Decision pinned in ``docs/architecture.md`` "IV as regression-bucket
    extension": 2SLS is structurally a regression-shape payload (the
    structural-equation coefficient table) with a handful of extra
    diagnostic scalars (first-stage F, Sargan / Hansen J,
    Wu-Hausman). It does NOT need a composite shape — that territory
    is reserved for genuine multi-stage estimators (3SLS, mediation,
    control-function corrections) where the model needs two
    independent coefficient tables.

    statsmodels' ``sandbox.regression.gmm.IV2SLS`` does not compute
    the first-stage F or Sargan / Wu-Hausman automatically — its
    sandbox status reflects that incomplete diagnostics surface.
    Compute them script-side and pass them through:

        from statsmodels.sandbox.regression.gmm import IV2SLS
        m = IV2SLS(y, exog, instruments).fit()
        first_stage = sm.OLS(endo, instruments).fit()
        sift.from_iv(
            m,
            instrument_variables=["z1", "z2"],
            endogenous_variables=["x_endo"],
            first_stage_f=float(first_stage.fvalue),
        )

    If you're using ``linearmodels.iv.IV2SLS`` (which DOES compute
    these), pass ``model.first_stage.diagnostics["f.stat"]``,
    ``model.sargan.stat`` / ``model.sargan.pval``, and
    ``model.wu_hausman().stat`` / ``model.wu_hausman().pval``.
    """
    iv_extra: dict[str, Any] = {}
    if instrument_variables is not None:
        iv_extra["instrument_variables"] = list(instrument_variables)
        iv_extra["n_instruments"] = len(instrument_variables)
    if endogenous_variables is not None:
        iv_extra["endogenous_variables"] = list(endogenous_variables)
        iv_extra["n_endogenous"] = len(endogenous_variables)
    for k, v in (
        ("first_stage_f", first_stage_f),
        ("weak_instrument_p", weak_instrument_p),
        ("hansen_j", hansen_j),
        ("hansen_j_p", hansen_j_p),
        ("endogeneity_p", endogeneity_p),
    ):
        vf = _safe_float(v)
        if vf is not None:
            iv_extra[k] = vf
    iv_extra.update(extra)
    from_lm(model, **iv_extra)


def from_marginal_effects(
    margeff: Any,
    *,
    variables: list[str] | None = None,
    method: str | None = None,
    outcome_variable: str | None = None,
    model_family: str | None = None,
    at_values: dict[str, float] | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``marginal_effects`` payload from a statsmodels
    ``DiscreteMargins`` / ``GenericMargins`` result.

    Wraps the output of ``fit.get_margeff(at=..., method=...)`` on a
    fitted statsmodels Logit / Probit / Poisson / GLM result. The
    ``DiscreteMargins`` object exposes:

      * ``margeff``           — per-variable marginal effects (ndarray)
      * ``margeff_se``        — delta-method standard errors
      * ``tvalues`` / ``pvalues`` — Wald-style test outputs
      * ``conf_int()``        — 95% CIs as a 2-D array
      * ``results``           — back-reference to the underlying fit;
                                ``.model.exog_names`` provides the
                                column labels.
      * ``margeff_options``   — dict carrying the ``at`` / ``method``
                                Stata-vocabulary choices the caller passed.

    Method mapping from statsmodels' ``at`` keyword onto the
    sanitizer's enum:

      * ``"overall"``  → ``"ame"``  (average over the sample)
      * ``"mean"``     → ``"mem"``  (evaluated at sample means)
      * ``"median"``   → ``"at_representative"`` (median is one
        specific representative covariate vector; the medians are
        passed through ``at_values``)
      * any explicit ``at`` dict → ``"at_representative"`` with the
        dict in ``at_values``

    Example:
        from statsmodels.formula.api import logit
        m = logit("y ~ age + female + income", data=df).fit()
        me = m.get_margeff(at="overall", method="dydx")
        sift.from_marginal_effects(
            me, outcome_variable="y", model_family="logit",
            label="AME from logit",
        )

    The helper is intentionally narrow — it reads from the
    ``DiscreteMargins`` shape statsmodels produces and routes onto
    the sanitizer's enum. R's ``marginaleffects::avg_slopes()``
    output is structurally different; that path uses
    ``sift$from_marginal_effects`` in the R runtime.

    **Disclosure note on ``at_values``** (relevant only for
    ``method="at_representative"``): the conditioning vector you
    pass is precision-clamped by the sample N before it reaches the
    model — at n=1000 you get ~4 sigfigs, at n=100 you get ~3. Pass
    interpretable summary points (means, medians, percentiles,
    round reference values from the literature). An exact-precision
    value pulled from a single row is gated by the precision floor;
    it won't cross as raw bytes, but the right interpretation is
    still "this is a representative point at this precision".
    """
    try:
        print(margeff.summary() if callable(getattr(margeff, "summary", None))
              else margeff)
    except Exception:  # noqa: BLE001
        pass

    # Duck-typed access — don't import statsmodels at module load.
    eff_arr = _safe_attr(margeff, "margeff")
    if eff_arr is None:
        raise TypeError(
            "sift.from_marginal_effects: ``margeff`` must expose "
            "``.margeff`` (statsmodels DiscreteMargins / GenericMargins "
            "shape). Try ``fit.get_margeff(at=..., method='dydx')``."
        )

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "sift.from_marginal_effects requires numpy"
        ) from e

    eff = np.asarray(eff_arr, dtype=float).ravel()
    se_attr = _safe_attr(margeff, "margeff_se")
    se = np.asarray(se_attr, dtype=float).ravel() if se_attr is not None else None
    t_attr = _safe_attr(margeff, "tvalues")
    tv = np.asarray(t_attr, dtype=float).ravel() if t_attr is not None else None
    p_attr = _safe_attr(margeff, "pvalues")
    pv = np.asarray(p_attr, dtype=float).ravel() if p_attr is not None else None
    ci_fn = _safe_attr(margeff, "conf_int")
    ci_arr = None
    if callable(ci_fn):
        try:
            ci_arr = np.asarray(ci_fn(), dtype=float)
        except Exception:  # noqa: BLE001
            ci_arr = None

    # Variable names. statsmodels' ``get_margeff`` drops the constant
    # automatically; the remaining ``margeff_options["exog_names"]``
    # carries the surviving column labels in order. If not present,
    # fall back to ``model.exog_names`` minus standard intercept
    # aliases.
    if variables is None:
        opts = _safe_attr(margeff, "margeff_options") or {}
        if isinstance(opts, dict) and isinstance(opts.get("exog_names"), list):
            variables = [str(v) for v in opts["exog_names"]]
        else:
            inner = _safe_attr(margeff, "results")
            inner_model = (
                _safe_attr(inner, "model") if inner is not None else None
            )
            exog_names = (
                list(getattr(inner_model, "exog_names", []) or [])
                if inner_model is not None else []
            )
            variables = [
                n for n in exog_names
                if n not in ("const", "Intercept", "(Intercept)", "intercept")
            ]
    variables = [str(v) for v in variables]
    if len(variables) != eff.size:
        raise ValueError(
            f"sift.from_marginal_effects: ``variables`` has "
            f"{len(variables)} entries but margeff has {eff.size}"
        )

    effects: dict[str, float] = {}
    ses: dict[str, float] = {}
    pvs: dict[str, float] = {}
    zs: dict[str, float] = {}
    los: dict[str, float] = {}
    his: dict[str, float] = {}
    for i, v in enumerate(variables):
        if i < eff.size and math.isfinite(float(eff[i])):
            effects[v] = float(eff[i])
        if se is not None and i < se.size and math.isfinite(float(se[i])):
            ses[v] = float(se[i])
        if tv is not None and i < tv.size and math.isfinite(float(tv[i])):
            zs[v] = float(tv[i])
        if pv is not None and i < pv.size and math.isfinite(float(pv[i])):
            pvs[v] = float(pv[i])
        if ci_arr is not None and i < ci_arr.shape[0] and ci_arr.shape[1] >= 2:
            lo, hi = float(ci_arr[i, 0]), float(ci_arr[i, 1])
            if math.isfinite(lo):
                los[v] = lo
            if math.isfinite(hi):
                his[v] = hi

    # Method resolution. Caller-supplied wins; otherwise infer from
    # the ``margeff_options["at"]`` value statsmodels stashes on the
    # result.
    if method is None:
        opts = _safe_attr(margeff, "margeff_options") or {}
        at = opts.get("at") if isinstance(opts, dict) else None
        if at == "overall" or at is None:
            method = "ame"
        elif at == "mean":
            method = "mem"
        else:
            method = "at_representative"
    method = str(method)

    # n: rows of the design statsmodels fit on. Pull from the inner
    # model — ``margeff`` itself doesn't carry a count directly.
    n_val: int | None = None
    inner = _safe_attr(margeff, "results")
    if inner is not None:
        n_val = _safe_int(_safe_attr(inner, "nobs"))
        if n_val is None:
            inner_model = _safe_attr(inner, "model")
            endog = (
                getattr(inner_model, "endog", None)
                if inner_model is not None else None
            )
            if endog is not None:
                try:
                    n_val = int(getattr(endog, "shape", (len(endog),))[0])
                except (TypeError, AttributeError):
                    n_val = None

    fields: dict[str, Any] = {
        "type": "marginal_effects",
        "method": method,
        "variables": variables,
        "effects": effects,
    }
    if ses:
        fields["standard_errors"] = ses
    if zs:
        fields["z_statistics"] = zs
    if pvs:
        fields["p_values"] = pvs
    if los:
        fields["ci_lower"] = los
    if his:
        fields["ci_upper"] = his
    if n_val is not None:
        fields["n"] = n_val
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if model_family is not None:
        fields["model_family"] = str(model_family)
    elif inner is not None:
        # Auto-detect: ``inner.model.__class__.__name__`` reveals
        # whether we're in Logit / Probit / Poisson / etc.
        cls = _safe_attr(_safe_attr(inner, "model"), "__class__")
        if cls is not None:
            name = getattr(cls, "__name__", "")
            if isinstance(name, str) and name:
                fields["model_family"] = name.lower()
    if at_values is not None and at_values:
        clean_at: dict[str, float] = {}
        for k, val in at_values.items():
            vf = _safe_float(val)
            if vf is not None:
                clean_at[str(k)] = vf
        if clean_at:
            fields["at_values"] = clean_at
    if label is not None:
        fields["label"] = str(label)
    fields.update(extra)
    result(**fields)


def from_cluster(
    fit: Any,
    X: Any = None,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    """Emit a ``cluster_analysis`` payload from a sklearn clustering
    fit. Dispatches on class:

      * ``KMeans`` — cluster centers and labels read directly from
        the fit; ``X`` not needed.
      * ``AgglomerativeClustering`` — sklearn agglomerative fits
        don't store cluster centers (they're not centroid-based),
        so ``X`` (the matrix the fit was built on) is required.
        Centroids and within-cluster SS computed post-hoc from
        ``X[fit.labels_ == k].mean(axis=0)``.

    DBSCAN and HDBSCAN are intentionally not supported. Their
    inference-adequacy story (density parameters, noise points, no
    centroids by construction) needs a separate design pass. The
    helper raises with a clear pointer to the generic
    ``sift.result(type="cluster_analysis", method="dbscan", ...)``
    path.

    Per-observation cluster assignments (``fit.labels_``) are NOT
    emitted on any path — per-row data, no allowlist slot.

    Examples:
        from sklearn.cluster import KMeans, AgglomerativeClustering
        X = df[["age", "income", "tenure"]].values
        sift.from_cluster(KMeans(n_clusters=4, random_state=42,
                                 n_init=10).fit(X),
                         variables=["age", "income", "tenure"])
        sift.from_cluster(AgglomerativeClustering(n_clusters=4,
                                                  linkage="ward").fit(X),
                         X=X, variables=["age", "income", "tenure"])
    """
    cls_name = type(fit).__name__
    if cls_name == "KMeans":
        _from_kmeans_impl(fit, variables=variables, label=label)
        return
    if cls_name == "AgglomerativeClustering":
        if X is None:
            raise ValueError(
                "sift.from_cluster: AgglomerativeClustering fits don't store "
                "centers — pass ``X`` (the matrix the fit was built on) so "
                "the helper can compute centroids from "
                "``X[fit.labels_ == k].mean(axis=0)``."
            )
        _from_agglomerative_impl(fit, X, variables=variables, label=label)
        return
    if cls_name in ("DBSCAN", "HDBSCAN"):
        raise TypeError(
            f"sift.from_cluster: dedicated {cls_name} helper not yet "
            f"shipped. Construct the payload via "
            f'`sift.result(type="cluster_analysis", method="dbscan", '
            f"cluster_sizes=..., n_noise_points=..., variables=..., ...)` "
            f"from the script — the cluster_analysis shape accepts dbscan "
            f"with centroids absent."
        )
    raise TypeError(
        f"sift.from_cluster: unknown clustering class {cls_name!r}. "
        f"Supported: KMeans, AgglomerativeClustering. DBSCAN-family: "
        f"use generic ``sift.result(type='cluster_analysis', ...)`` "
        f"until a dedicated helper ships."
    )


def _from_kmeans_impl(
    fit: Any,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    centers_attr = _safe_attr(fit, "cluster_centers_")
    labels_attr = _safe_attr(fit, "labels_")
    if centers_attr is None or labels_attr is None:
        raise RuntimeError(
            "sift.from_cluster: fit.cluster_centers_ or fit.labels_ missing"
        )
    centers = np.asarray(centers_attr)
    labels = np.asarray(labels_attr)
    n_clusters_actual, n_features = centers.shape

    if variables is None:
        variables = [f"feature_{i+1}" for i in range(n_features)]
    variables = [str(v) for v in variables]
    if len(variables) != n_features:
        raise ValueError(
            f"sift.from_cluster: variables has {len(variables)} entries "
            f"but KMeans was fit on {n_features} features"
        )

    cluster_labels = [f"cluster_{i+1}" for i in range(n_clusters_actual)]
    cluster_sizes: dict[str, int] = {}
    for i in range(n_clusters_actual):
        cluster_sizes[cluster_labels[i]] = int((labels == i).sum())

    centroids: dict[str, dict[str, float]] = {}
    for ci in range(n_clusters_actual):
        row: dict[str, float] = {}
        for fi, var in enumerate(variables):
            v = float(centers[ci, fi])
            if math.isfinite(v):
                row[var] = v
        if row:
            centroids[cluster_labels[ci]] = row

    inertia = _safe_float(_safe_attr(fit, "inertia_"))
    n_iter = _safe_int(_safe_attr(fit, "n_iter_"))
    n_obs = int(labels.size)
    fields: dict[str, Any] = {
        "type": "cluster_analysis",
        "method": "kmeans",
        "distance_metric": "euclidean",
        "n_observations": n_obs,
        "n_clusters": n_clusters_actual,
        "n_features": n_features,
        "variables": variables,
        "cluster_labels": cluster_labels,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
    }
    if inertia is not None:
        fields["total_within_ss"] = inertia
        fields["inertia"] = inertia
    if n_iter is not None:
        fields["n_iterations"] = n_iter
    if label is not None:
        fields["label"] = str(label)
    result(**fields)


def _from_agglomerative_impl(
    fit: Any,
    X: Any,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    """sklearn ``AgglomerativeClustering`` doesn't store centroids
    or within-cluster SS. Both are computed post-hoc from ``X``:
        centroid_k = X[labels == k].mean(axis=0)
        within_ss_k = ||X[labels == k] - centroid_k||²

    The dendrogram (``fit.children_``, ``fit.distances_``) is
    structurally absent from the payload — per-merge records over
    the data, researcher-only by construction.
    """
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError(
            "sift.from_cluster: ``X`` must be 2-D (n_observations × n_features)"
        )
    labels = np.asarray(_safe_attr(fit, "labels_"))
    n_clusters_actual = int(_safe_attr(fit, "n_clusters_") or labels.max() + 1)
    n_features = X_arr.shape[1]

    if variables is None:
        variables = [f"feature_{i+1}" for i in range(n_features)]
    variables = [str(v) for v in variables]
    if len(variables) != n_features:
        raise ValueError(
            f"sift.from_cluster: variables has {len(variables)} entries "
            f"but X has {n_features} columns"
        )

    cluster_labels = [f"cluster_{i+1}" for i in range(n_clusters_actual)]
    cluster_sizes: dict[str, int] = {}
    centroids: dict[str, dict[str, float]] = {}
    within_cluster_ss: dict[str, float] = {}
    total_within_ss = 0.0
    grand_mean = X_arr.mean(axis=0)
    for i in range(n_clusters_actual):
        cl = cluster_labels[i]
        mask = labels == i
        size = int(mask.sum())
        cluster_sizes[cl] = size
        if size == 0:
            continue
        sub = X_arr[mask]
        centroid = sub.mean(axis=0)
        centroids[cl] = {variables[fi]: float(centroid[fi]) for fi in range(n_features)}
        wss = float(((sub - centroid) ** 2).sum())
        within_cluster_ss[cl] = wss
        total_within_ss += wss
    total_ss = float(((X_arr - grand_mean) ** 2).sum())
    between_ss = total_ss - total_within_ss

    linkage_attr = _safe_attr(fit, "linkage")
    linkage = str(linkage_attr) if isinstance(linkage_attr, str) else None
    # sklearn distance defaults to euclidean for ward; the attribute
    # is ``metric`` (newer) or ``affinity`` (older).
    metric_attr = _safe_attr(fit, "metric") or _safe_attr(fit, "affinity")

    fields: dict[str, Any] = {
        "type": "cluster_analysis",
        "method": "hierarchical",
        "n_observations": int(X_arr.shape[0]),
        "n_clusters": n_clusters_actual,
        "n_features": n_features,
        "variables": variables,
        "cluster_labels": cluster_labels,
        "cluster_sizes": cluster_sizes,
        "centroids": centroids,
        "within_cluster_ss": within_cluster_ss,
        "total_within_ss": total_within_ss,
        "inertia": total_within_ss,
        "between_cluster_ss": between_ss,
        "total_ss": total_ss,
    }
    if total_ss > 0:
        fields["ss_ratio"] = between_ss / total_ss
    if linkage:
        fields["linkage"] = linkage
    if isinstance(metric_attr, str):
        fields["distance_metric"] = metric_attr
    if label is not None:
        fields["label"] = str(label)
    result(**fields)


# Back-compat alias. ``from_kmeans`` was the public name in earlier
# releases; ``from_cluster`` with class dispatch is the new
# canonical entry point.
def from_kmeans(
    fit: Any,
    *,
    variables: list[str] | None = None,
    label: str | None = None,
) -> None:
    """Back-compat alias for ``from_cluster`` on KMeans fits."""
    from_cluster(fit, variables=variables, label=label)


def from_pca(
    fit: Any,
    *,
    variables: list[str] | None = None,
    n_components: int | None = None,
    label: str | None = None,
) -> None:
    """Emit a ``factor_decomposition`` payload from a fitted
    ``sklearn.decomposition.PCA``.

    sklearn's PCA stores loadings transposed relative to R's prcomp:
    ``components_`` is ``(n_components, n_features)`` (each row is a
    component's loadings on the features). We pivot to the
    ``{variable: {component: value}}`` shape the sanitizer's
    ``loadings`` field expects.

    Caller passes ``variables`` (the column-name list matching the
    order of features the PCA was fit on). sklearn doesn't store
    column names — its design takes a 2-D array — so the names must
    come from the caller. If omitted, generic ``feature_1`` … fall-
    backs are used; less useful to the model.

    Privacy carve-out: ``fit.transform(X)`` (the per-observation
    factor scores) is researcher-only by structural absence —
    nothing in this helper emits it, and no field on the
    ``factor_decomposition`` allowlist would accept it.

    Example:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3).fit(df[["v1","v2","v3","v4","v5"]])
        sift.from_pca(pca, variables=["v1","v2","v3","v4","v5"],
                     label="five-variable PCA")
    """
    cls_name = type(fit).__name__
    if cls_name != "PCA":
        raise TypeError(
            "sift.from_pca: ``fit`` must be a sklearn.decomposition.PCA "
            f"instance; got {cls_name!r}"
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("sift.from_pca requires numpy") from e

    components_arr = _safe_attr(fit, "components_")
    if components_arr is None:
        raise RuntimeError(
            "sift.from_pca: fit.components_ missing — was the PCA fitted?"
        )
    components_arr = np.asarray(components_arr)  # (n_components, n_features)
    n_comp_full, n_feat = components_arr.shape

    n_obs = _safe_int(_safe_attr(fit, "n_samples_"))
    if n_obs is None:
        raise RuntimeError(
            "sift.from_pca: fit.n_samples_ missing — n_observations is required"
        )

    if variables is None:
        variables = [f"feature_{i+1}" for i in range(n_feat)]
    variables = [str(v) for v in variables]
    if len(variables) != n_feat:
        raise ValueError(
            f"sift.from_pca: variables has {len(variables)} entries but "
            f"PCA was fit on {n_feat} features"
        )

    n_comp = (
        min(n_components, n_comp_full) if isinstance(n_components, int)
        else n_comp_full
    )
    comp_labels = [f"PC{i+1}" for i in range(n_comp)]

    # Loadings: pivot from (n_comp, n_feat) to {variable: {component: value}}.
    loadings: dict[str, dict[str, float]] = {}
    for fi, var in enumerate(variables):
        row: dict[str, float] = {}
        for ci, c_lab in enumerate(comp_labels):
            v = float(components_arr[ci, fi])
            if math.isfinite(v):
                row[c_lab] = v
        if row:
            loadings[var] = row

    # Explained variance / ratio / cumulative — sklearn exposes
    # ``explained_variance_`` (eigenvalues) and
    # ``explained_variance_ratio_`` (normalized to sum=1 over
    # *retained* components). Compute cumulative from the ratio.
    # ``or []`` doesn't compose with ndarrays — ``bool(array)`` raises
    # for arrays with more than one element. Write the None check
    # explicitly.
    _ev = _safe_attr(fit, "explained_variance_")
    ev_attr = np.asarray(_ev if _ev is not None else [])
    _evr = _safe_attr(fit, "explained_variance_ratio_")
    evr_attr = np.asarray(_evr if _evr is not None else [])
    eigenvalues: dict[str, float] = {}
    explained_variance: dict[str, float] = {}
    explained_variance_ratio: dict[str, float] = {}
    cumulative_variance: dict[str, float] = {}
    cum = 0.0
    for ci, c_lab in enumerate(comp_labels):
        if ci < len(ev_attr) and math.isfinite(float(ev_attr[ci])):
            eigenvalues[c_lab] = float(ev_attr[ci])
            explained_variance[c_lab] = float(ev_attr[ci])
        if ci < len(evr_attr) and math.isfinite(float(evr_attr[ci])):
            r = float(evr_attr[ci])
            explained_variance_ratio[c_lab] = r
            cum += r
            cumulative_variance[c_lab] = cum

    # Communalities (PCA): sum of squared loadings across retained
    # components, per variable.
    communalities: dict[str, float] = {}
    for fi, var in enumerate(variables):
        h2 = float(np.sum(components_arr[:n_comp, fi] ** 2))
        if math.isfinite(h2):
            communalities[var] = h2

    fields: dict[str, Any] = {
        "type": "factor_decomposition",
        "method": "pca",
        "rotation": "none",
        "n_observations": n_obs,
        "n_variables": n_feat,
        "n_components": n_comp,
        "variables": variables,
        "components": comp_labels,
        "loadings": loadings,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "eigenvalues": eigenvalues,
        "communalities": communalities,
    }
    if label is not None:
        fields["label"] = str(label)
    result(**fields)


def from_factor_analyzer(
    fit: Any,
    *,
    variables: list[str] | None = None,
    method: str | None = None,
    rotation: str | None = None,
    n_observations: int | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``factor_decomposition`` payload from a fitted
    ``factor_analyzer.FactorAnalyzer``.

    factor_analyzer is the Python-side standard for exploratory
    factor analysis (the closest analogue to R ``psych::fa``).
    The fit object exposes:

      * ``loadings_``        — (n_features, n_factors) ndarray
      * ``get_uniquenesses()``
      * ``get_communalities()``
      * ``get_eigenvalues()`` — returns (original, common-factor)
      * ``get_factor_variance()`` — (variance, proportional,
                                     cumulative) per factor

    factor_analyzer doesn't store column names (it's fit on a
    bare ndarray or DataFrame), so ``variables`` must be passed
    explicitly when the fit was built from a numpy array. When
    fit from a DataFrame, factor_analyzer stashes the columns in
    ``.feature_names_`` (newer versions) — probed below as a
    fallback.

    Privacy carve-out: the per-row factor scores (the result of
    ``fit.transform(X)``) are structurally absent from the
    sanitizer's allowlist — no field accepts a 2-D array of
    per-observation values.

    Example:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=3, rotation="varimax", method="ml")
        fa.fit(df[["v1","v2","v3","v4","v5"]])
        sift.from_factor_analyzer(
            fa, variables=["v1","v2","v3","v4","v5"],
            n_observations=len(df), label="ML FA with varimax",
        )
    """
    cls_name = type(fit).__name__
    if cls_name != "FactorAnalyzer":
        raise TypeError(
            "sift.from_factor_analyzer: ``fit`` must be a "
            "factor_analyzer.FactorAnalyzer instance; got "
            f"{cls_name!r}"
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass
    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "sift.from_factor_analyzer requires numpy"
        ) from e

    loadings_arr = _safe_attr(fit, "loadings_")
    if loadings_arr is None:
        raise RuntimeError(
            "sift.from_factor_analyzer: fit.loadings_ missing — "
            "was the fit run?"
        )
    loadings_arr = np.asarray(loadings_arr)
    n_feat, n_factors = loadings_arr.shape

    # Variable names. Prefer the caller's list; fall back to
    # factor_analyzer's stash; default to feature_N for unnamed.
    if variables is None:
        fnames = _safe_attr(fit, "feature_names_")
        if fnames is not None:
            variables = [str(x) for x in fnames]
        else:
            variables = [f"feature_{i+1}" for i in range(n_feat)]
    variables = [str(v) for v in variables]
    if len(variables) != n_feat:
        raise ValueError(
            f"sift.from_factor_analyzer: variables has "
            f"{len(variables)} entries but FA was fit on {n_feat} features"
        )

    fac_labels = [f"factor{i+1}" for i in range(n_factors)]

    # Method: map factor_analyzer's ``method`` slot to the sanitizer
    # enum. The valid values on the fit are "minres" / "ml" /
    # "principal".
    if method is None:
        m_attr = _safe_attr(fit, "method")
        if isinstance(m_attr, str):
            m = m_attr.lower()
            if m == "ml":
                method = "maximum_likelihood"
            elif m == "minres":
                method = "minimum_residual"
            elif m == "principal":
                method = "principal_factor"
            else:
                method = "factor_analysis"
        else:
            method = "factor_analysis"
    method = str(method)

    if rotation is None:
        r_attr = _safe_attr(fit, "rotation")
        rotation = str(r_attr) if isinstance(r_attr, str) else "none"
    # Normalize None → "none" so the sanitizer's enum check passes.
    if rotation in (None, "None"):
        rotation = "none"

    loadings: dict[str, dict[str, float]] = {}
    for fi, var in enumerate(variables):
        row: dict[str, float] = {}
        for j, lab in enumerate(fac_labels):
            v = float(loadings_arr[fi, j])
            if math.isfinite(v):
                row[lab] = v
        if row:
            loadings[var] = row

    # Uniqueness / communalities via factor_analyzer's getters.
    uniqueness: dict[str, float] = {}
    communalities: dict[str, float] = {}
    try:
        u_arr = np.asarray(fit.get_uniquenesses())
        for fi, var in enumerate(variables):
            v = float(u_arr[fi])
            if math.isfinite(v):
                uniqueness[var] = v
    except Exception:  # noqa: BLE001
        pass
    try:
        c_arr = np.asarray(fit.get_communalities())
        for fi, var in enumerate(variables):
            v = float(c_arr[fi])
            if math.isfinite(v):
                communalities[var] = v
    except Exception:  # noqa: BLE001
        pass

    eigenvalues: dict[str, float] = {}
    explained_variance: dict[str, float] = {}
    explained_variance_ratio: dict[str, float] = {}
    cumulative_variance: dict[str, float] = {}
    # ``get_factor_variance()`` → tuple (variance, proportional,
    # cumulative), each an ndarray of length n_factors.
    try:
        variance_values, proportion_values, cumulative_values = (
            fit.get_factor_variance()
        )
        variance_values = np.asarray(variance_values)
        proportion_values = np.asarray(proportion_values)
        cumulative_values = np.asarray(cumulative_values)
        for j, lab in enumerate(fac_labels):
            v = float(variance_values[j])
            if math.isfinite(v):
                eigenvalues[lab] = v
                explained_variance[lab] = v
            p = float(proportion_values[j])
            if math.isfinite(p):
                explained_variance_ratio[lab] = p
            c = float(cumulative_values[j])
            if math.isfinite(c):
                cumulative_variance[lab] = c
    except Exception:  # noqa: BLE001
        pass

    fields: dict[str, Any] = {
        "type": "factor_decomposition",
        "method": method,
        "rotation": rotation,
        "n_variables": n_feat,
        "n_components": n_factors,
        "variables": variables,
        "components": fac_labels,
        "loadings": loadings,
    }
    if n_observations is not None:
        fields["n_observations"] = int(n_observations)
    if communalities:            fields["communalities"] = communalities
    if uniqueness:               fields["uniqueness"] = uniqueness
    if eigenvalues:              fields["eigenvalues"] = eigenvalues
    if explained_variance:       fields["explained_variance"] = explained_variance
    if explained_variance_ratio: fields["explained_variance_ratio"] = explained_variance_ratio
    if cumulative_variance:      fields["cumulative_variance"] = cumulative_variance

    # Goodness-of-fit scalars when available. factor_analyzer 0.4+
    # exposes a ``sufficiency`` test that ships chi² + p; not
    # universally present so probe quietly.
    suf_fn = _safe_attr(fit, "sufficiency")
    if callable(suf_fn):
        try:
            chi2, dof, pval = suf_fn(n_observations or 0)
            if math.isfinite(float(chi2)):
                fields["chi_squared"] = float(chi2)
            if math.isfinite(float(pval)):
                fields["chi_squared_p_value"] = float(pval)
            if int(dof) > 0:
                fields["degrees_of_freedom"] = int(dof)
        except Exception:  # noqa: BLE001
            pass

    if label is not None:
        fields["label"] = str(label)
    fields.update(extra)
    result(**fields)


def from_callaway_santanna(
    attgt: Any,
    fit_result: Any | None = None,
    *,
    outcome_variable: str | None = None,
    treatment_variable: str | None = None,
    aggregation_method: str = "event",
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``did_event_study`` payload from a Callaway-Sant'Anna
    fit produced by the ``differences`` package.

    Two-argument form keeps the ATTgt config (cohort column, data,
    anticipation, base_period) reachable alongside the per-(g, t)
    estimates the fitted result carries:

        from differences import ATTgt
        attgt = ATTgt(data=df.set_index(["id","period"]),
                      cohort_column="G", base_period="varying",
                      anticipation=0)
        result = attgt.fit(formula="y", control_group="never_treated")
        sift.from_callaway_santanna(attgt, result,
                                    outcome_variable="y",
                                    treatment_variable="G",
                                    label="headline DiD")

    If ``result`` is omitted, ``attgt`` is assumed to BE the fitted
    result (``differences`` happens to allow this fluent shape too).
    The helper pivots ATT(cohort, base_period, time) → ATT(cohort,
    event_time), pulls per-cohort treated counts from ``attgt.data``,
    and reads the aggregate ATT from
    ``result.aggregate(type_of_aggregation="simple")``.

    ``estimator`` is hard-coded to ``"callaway_santanna"`` for this
    helper; Sun-Abraham / de Chaisemartin land under their own
    helpers when those ship.
    """
    try:
        import numpy as np
        import pandas as pd
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "sift.from_callaway_santanna requires numpy + pandas"
        ) from e

    # If only one argument was passed, treat it as the fitted result
    # and try to recover the ATTgt config from it. Otherwise the
    # caller must pass both.
    if fit_result is None:
        fit_result = attgt
        attgt = None

    # Per-cell ATT(g, t) table. differences exposes a multi-indexed
    # DataFrame via ``result.to_pandas()`` with index levels
    # (cohort, base_period, time). The cell value lives under the
    # column tuple ('ATTgtElements', '', 'ATT'); SE under
    # ('ATTgtElements', 'analytic', 'std_error'). The pointwise
    # band columns are ('ATTgtElements', 'pointwise conf. band',
    # 'lower' | 'upper').
    if not hasattr(fit_result, "to_pandas"):
        raise TypeError(
            "sift.from_callaway_santanna: ``fit_result`` must be an "
            "ATTgtResult (returned by differences.ATTgt.fit(...))"
        )
    table = fit_result.to_pandas()
    # Robust column lookup — names rolled through the package version
    # could shift; match by trailing element.
    def _col(suffix: str) -> Any:
        for c in table.columns:
            if isinstance(c, tuple) and c[-1] == suffix:
                return c
        return None
    col_att = _col("ATT")
    col_se  = _col("std_error")
    col_lo  = _col("lower")
    col_hi  = _col("upper")
    if col_att is None:
        raise RuntimeError(
            "sift.from_callaway_santanna: ``ATT`` column not found in "
            "result.to_pandas() — package version mismatch?"
        )

    # Pivot to {cohort: {event_time: value}}. Index is
    # (cohort, base_period, time); event_time = time - cohort.
    att_dict: dict[str, dict[str, float]] = {}
    se_dict: dict[str, dict[str, float]] = {}
    ci_lo_dict: dict[str, dict[str, float]] = {}
    ci_hi_dict: dict[str, dict[str, float]] = {}
    cohorts_seen: set[str] = set()
    event_times_seen: set[int] = set()
    for idx, row in table.iterrows():
        if not isinstance(idx, tuple) or len(idx) < 3:
            continue
        cohort = idx[0]
        time   = idx[-1]
        try:
            event_time = int(time) - int(cohort)
        except (TypeError, ValueError):
            continue
        # The same (cohort, event_time) can appear under multiple
        # base periods when ``base_period="varying"``. Keep the
        # last one (the immediately-pre-treatment base period —
        # the conventional CS reporting).
        c_lab = str(int(cohort) if float(cohort).is_integer() else cohort)
        e_lab = str(event_time)
        cohorts_seen.add(c_lab)
        event_times_seen.add(event_time)
        att_dict.setdefault(c_lab, {})[e_lab] = float(row[col_att])
        if col_se is not None:
            se_dict.setdefault(c_lab, {})[e_lab] = float(row[col_se])
        if col_lo is not None:
            ci_lo_dict.setdefault(c_lab, {})[e_lab] = float(row[col_lo])
        if col_hi is not None:
            ci_hi_dict.setdefault(c_lab, {})[e_lab] = float(row[col_hi])

    # Per-cohort treated counts — number of distinct entity ids per
    # cohort in the input panel. ``attgt.data`` is the indexed
    # dataframe the fit was built from; entity id is the first level.
    n_treated_per_group: dict[str, int] = {}
    if attgt is not None and hasattr(attgt, "data") and hasattr(attgt, "cohort_column"):
        try:
            ent_name = attgt.data.index.names[0]
            sizes = (
                attgt.data.reset_index()
                .drop_duplicates(subset=[ent_name])
                .groupby(attgt.cohort_column)[ent_name].nunique()
            )
            for k, v in sizes.items():
                if pd.isna(k):
                    continue
                kk = str(int(k) if float(k).is_integer() else k)
                if kk in cohorts_seen:
                    n_treated_per_group[kk] = int(v)
        except Exception:  # noqa: BLE001
            pass

    fields: dict[str, Any] = {
        "type": "did_event_study",
        "estimator": "callaway_santanna",
        "groups": sorted(cohorts_seen, key=lambda x: float(x)),
        "event_times": sorted(event_times_seen),
        "att": att_dict,
        "standard_errors": se_dict,
        "ci_lower": ci_lo_dict,
        "ci_upper": ci_hi_dict,
        "n_treated_per_group": n_treated_per_group,
        "aggregation_method": str(aggregation_method),
    }
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if treatment_variable is not None:
        fields["treatment_variable"] = str(treatment_variable)
    if label is not None:
        fields["label"] = str(label)

    # Pass through CS configuration so the model knows the
    # identification assumptions the estimator ran under.
    if attgt is not None:
        bp = _safe_attr(attgt, "base_period_type")
        if isinstance(bp, str):
            fields["base_period"] = bp
        ant = _safe_attr(attgt, "anticipation")
        if isinstance(ant, int):
            fields["anticipation_periods"] = ant
    # control_group lives on ``attgt.estimation_details()`` (a method
    # returning a dict, populated after fit). Older versions exposed
    # it as an attribute; probe both shapes.
    if attgt is not None:
        det = _safe_attr(attgt, "estimation_details")
        if callable(det):
            try:
                det = det()
            except Exception:  # noqa: BLE001
                det = None
        if isinstance(det, dict):
            cg = det.get("control_group")
            if isinstance(cg, str):
                fields["comparison_group"] = cg

    # Aggregate scalars via ``aggregate(type_of_aggregation="simple")``.
    try:
        simple = fit_result.aggregate(type_of_aggregation="simple")
        # The aggregate is a DataFrame with one row.
        simple_pd = (
            simple.to_pandas() if hasattr(simple, "to_pandas") else simple
        )
        for c in simple_pd.columns:
            tail = c[-1] if isinstance(c, tuple) else c
            if tail == "ATT":
                fields["aggregate_att"] = float(simple_pd[c].iloc[0])
            elif tail == "std_error":
                v = float(simple_pd[c].iloc[0])
                fields["aggregate_se"] = v
                if "aggregate_att" in fields and v > 0:
                    z = abs(fields["aggregate_att"] / v)
                    fields["aggregate_p_value"] = float(
                        math.erfc(z / math.sqrt(2.0))
                    )
            elif tail == "lower":
                fields["aggregate_ci_lower"] = float(simple_pd[c].iloc[0])
            elif tail == "upper":
                fields["aggregate_ci_upper"] = float(simple_pd[c].iloc[0])
    except Exception:  # noqa: BLE001
        pass

    fields.update(extra)
    result(**fields)


def from_sun_abraham(
    fit: Any,
    n_treated: int,
    *,
    outcome_variable: str | None = None,
    treatment_variable: str | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``did_event_study`` payload from a Sun-Abraham
    interaction-weighted (IW) event study fit produced by
    ``pyfixest.event_study(..., estimator="saturated")``.

    The Sun-Abraham (2021) IW estimator solves the bias TWFE event
    studies pick up under treatment-effect heterogeneity. pyfixest's
    ``event_study`` with ``estimator="saturated"`` produces a
    cohort-saturated fit and binds an ``aggregate(agg, weighting)``
    method onto the returned Feols object that collapses the
    cohort × event-time grid to per-period IW estimates (the
    Sun-Abraham aggregate).

    Like the R helper, this emits a single synthetic cohort
    ``"all"`` because the aggregation happens inside the estimator;
    the model sees one ATT per event-time. ``n_treated`` is the
    total count of treated units across all cohorts — required,
    because the cohort-N gate has no input without it.

    Stata's ``eventstudyinteract`` is the SSC port of Sun-Abraham;
    a dedicated Stata helper is deferred (same SSC-auth + maintenance-
    lag posture as ``csdid``). For Stata-side Sun-Abraham today,
    emit via ``sift.result(type="did_event_study",
    estimator="sun_abraham", ...)``.

    Example:
        import pyfixest as pf
        # cohort variable ``g``; never-treated coded as a far-future
        # value (e.g. 10000) or as 0 per the package convention.
        fit = pf.event_study(
            df, yname="y", idname="id", tname="period", gname="g",
            estimator="saturated",
        )
        n_t = df.loc[df["g"] > 0, "id"].nunique()
        sift.from_sun_abraham(fit, n_treated=n_t,
                              outcome_variable="y",
                              treatment_variable="g",
                              label="Sun-Abraham IW")
    """
    if not isinstance(n_treated, int) or n_treated < 0:
        raise ValueError(
            "sift.from_sun_abraham: ``n_treated`` (total treated units) "
            "is required and must be a non-negative int"
        )
    aggregate_fn = getattr(fit, "aggregate", None)
    if not callable(aggregate_fn):
        raise TypeError(
            "sift.from_sun_abraham: ``fit`` must expose an "
            "``aggregate`` method (pyfixest saturated event study "
            "shape). Run ``pyfixest.event_study(..., "
            "estimator='saturated')`` first."
        )
    method_attr = getattr(fit, "_method", None)
    if isinstance(method_attr, str) and method_attr not in (
        "saturated", "sun_abraham"
    ):
        raise TypeError(
            "sift.from_sun_abraham: ``fit._method`` is "
            f"{method_attr!r} — expected 'saturated' (Sun-Abraham). "
            "Did you mean ``from_twfe_event_study`` for the TWFE fit?"
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass

    try:
        agg_df = aggregate_fn(agg="period", weighting="shares")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "sift.from_sun_abraham: fit.aggregate(agg='period', "
            f"weighting='shares') failed: {e}"
        ) from e
    if agg_df is None or not hasattr(agg_df, "iterrows"):
        raise RuntimeError(
            "sift.from_sun_abraham: aggregate() did not return a "
            "DataFrame — pyfixest version mismatch?"
        )

    cn = list(agg_df.columns)

    def _pick(candidates: tuple[str, ...]) -> str | None:
        for c in candidates:
            if c in cn:
                return c
        return None

    est_col = _pick(("Estimate", "estimate"))
    se_col  = _pick(("Std. Error", "std_error", "std.error", "se"))
    p_col   = _pick(("Pr(>|t|)", "Pr(>|z|)", "p_value", "p.value"))
    lo_col  = _pick(("2.5%", "conf_low", "conf.low"))
    hi_col  = _pick(("97.5%", "conf_high", "conf.high"))
    if est_col is None:
        raise RuntimeError(
            "sift.from_sun_abraham: aggregate() output missing an "
            "Estimate column — pyfixest version mismatch?"
        )

    att_all: dict[str, float] = {}
    se_all:  dict[str, float] = {}
    p_all:   dict[str, float] = {}
    ci_lo:   dict[str, float] = {}
    ci_hi:   dict[str, float] = {}
    event_times: list[int] = []
    for period_label, row in agg_df.iterrows():
        try:
            et = int(period_label)
        except (TypeError, ValueError):
            continue
        est_raw = row[est_col]
        try:
            est = float(est_raw)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(est):
            continue
        lab = str(et)
        att_all[lab] = est
        event_times.append(et)
        if se_col is not None:
            try:
                se = float(row[se_col])
            except (TypeError, ValueError):
                se = float("nan")
            if math.isfinite(se):
                se_all[lab] = se
        if p_col is not None:
            try:
                pv = float(row[p_col])
            except (TypeError, ValueError):
                pv = float("nan")
            if math.isfinite(pv):
                p_all[lab] = pv
        if lo_col is not None:
            try:
                lo = float(row[lo_col])
            except (TypeError, ValueError):
                lo = float("nan")
            if math.isfinite(lo):
                ci_lo[lab] = lo
        if hi_col is not None:
            try:
                hi = float(row[hi_col])
            except (TypeError, ValueError):
                hi = float("nan")
            if math.isfinite(hi):
                ci_hi[lab] = hi
        # Synthesize ±1.96 SE CI when pyfixest didn't ship explicit
        # CI columns and SE is present.
        if lab not in ci_lo and lab in se_all:
            ci_lo[lab] = est - 1.96 * se_all[lab]
        if lab not in ci_hi and lab in se_all:
            ci_hi[lab] = est + 1.96 * se_all[lab]

    if not event_times:
        raise RuntimeError(
            "sift.from_sun_abraham: aggregate() returned no rows with "
            "integer period labels"
        )

    fields: dict[str, Any] = {
        "type": "did_event_study",
        "estimator": "sun_abraham",
        "aggregation_method": "dynamic",
        "groups": ["all"],
        "event_times": sorted(event_times),
        "att": {"all": att_all},
        "standard_errors": {"all": se_all},
        "p_values": {"all": p_all},
        "ci_lower": {"all": ci_lo},
        "ci_upper": {"all": ci_hi},
        "n_treated_per_group": {"all": int(n_treated)},
    }
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if treatment_variable is not None:
        fields["treatment_variable"] = str(treatment_variable)
    if label is not None:
        fields["label"] = str(label)
    fields.update(extra)
    result(**fields)


def from_rdd(
    fit: Any,
    *,
    running_variable: str | None = None,
    outcome_variable: str | None = None,
    fuzzy_treatment_variable: str | None = None,
    first_stage_f: float | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit an ``rdd`` payload from an ``rdrobust`` fit.

    Wraps the rdrobust Python package (Calonico-Cattaneo-Titiunik
    2014) — the standard cross-language implementation. The payload
    carries the three-flavor τ table, bandwidth(s), kernel,
    polynomial order, effective N per side, and the bandwidth
    selector. For fuzzy RDD pass ``fuzzy_treatment_variable``; the
    estimator is tagged ``fuzzy_2sls`` and the caller may also
    supply ``first_stage_f``.

    Privacy carve-out is structural. The helper signature does not
    accept density / binscatter / mccrary keyword arguments — passing
    one raises. McCrary density and binscatter near the cutoff are
    visual diagnostics for the researcher; they have no field on
    the ``rdd`` shape's allowlist so even hand-crafted payloads
    through ``sift.result(type="rdd", ...)`` cannot smuggle them.

    Example:
        from rdrobust import rdrobust
        m = rdrobust(y=df["voted"], x=df["income"], c=50000)
        sift.from_rdd(m, running_variable="income",
                     outcome_variable="voted", label="headline RDD")

        # Fuzzy:
        m = rdrobust(y=df["voted"], x=df["income"], c=50000,
                     fuzzy=df["takeup"])
        sift.from_rdd(m, running_variable="income",
                     outcome_variable="voted",
                     fuzzy_treatment_variable="takeup",
                     first_stage_f=24.3)
    """
    # Privacy carve-out: reject density/binscatter kwargs that a
    # script might try to slip through ``**extra``.
    banned = (
        "mccrary_density_curve", "mccrary_density",
        "binscatter_bins", "binscatter", "density_curve",
    )
    for b in banned:
        if b in extra:
            raise ValueError(
                f"sift.from_rdd: ``{b}`` is a visual diagnostic for the "
                f"researcher and is not allowed on the rdd payload. The "
                f"model sees the analytical fields (tau / bandwidth / "
                f"effective N); ask the researcher qualitatively about "
                f"manipulation evidence if it bears on the design."
            )
    # rdrobust's result class is rdrobust_output. Duck-type rather
    # than importing rdrobust here (keeps the runtime import-light
    # for non-RDD scripts).
    cls = type(fit).__name__
    if cls != "rdrobust_output":
        raise TypeError(
            "sift.from_rdd: ``fit`` must be an rdrobust_output (returned "
            "by rdrobust.rdrobust(y, x, c=cutoff))."
        )
    try:
        print(fit)
    except Exception:  # noqa: BLE001
        pass

    fields: dict[str, Any] = {
        "type": "rdd",
        "estimator": (
            "fuzzy_2sls" if fuzzy_treatment_variable is not None
            else "local_polynomial"
        ),
    }
    if running_variable is not None:
        fields["running_variable"] = str(running_variable)
    if outcome_variable is not None:
        fields["outcome_variable"] = str(outcome_variable)
    if label is not None:
        fields["label"] = str(label)

    # Pull the three-flavor row indexes from the DataFrame-shaped
    # outputs. Python rdrobust uses the same row labels as R
    # (Conventional / Bias-Corrected / Robust).
    def _row(df: Any, label: str) -> float | None:
        try:
            v = df.loc[label].iloc[0]
            f = float(v)
            return f if math.isfinite(f) else None
        except Exception:  # noqa: BLE001
            return None

    coef = _safe_attr(fit, "coef")
    se   = _safe_attr(fit, "se")
    pv   = _safe_attr(fit, "pv")
    ci   = _safe_attr(fit, "ci")
    if coef is not None:
        for flavor, slot in (
            ("Conventional", "tau_conventional"),
            ("Bias-Corrected", "tau_bias_corrected"),
            ("Robust", "tau_robust"),
        ):
            v = _row(coef, flavor)
            if v is not None:
                fields[slot] = v
    if se is not None:
        for flavor, slot in (
            ("Conventional", "se_conventional"),
            ("Bias-Corrected", "se_bias_corrected"),
            ("Robust", "se_robust"),
        ):
            v = _row(se, flavor)
            if v is not None:
                fields[slot] = v
    if pv is not None:
        for flavor, slot in (
            ("Conventional", "p_conventional"),
            ("Bias-Corrected", "p_bias_corrected"),
            ("Robust", "p_robust"),
        ):
            v = _row(pv, flavor)
            if v is not None:
                fields[slot] = v
    if ci is not None:
        for flavor, lo_slot, hi_slot in (
            ("Conventional", "ci_lower_conventional", "ci_upper_conventional"),
            ("Bias-Corrected", "ci_lower_bias_corrected", "ci_upper_bias_corrected"),
            ("Robust", "ci_lower_robust", "ci_upper_robust"),
        ):
            try:
                lo = float(ci.loc[flavor].iloc[0])
                hi = float(ci.loc[flavor].iloc[1])
                if math.isfinite(lo): fields[lo_slot] = lo
                if math.isfinite(hi): fields[hi_slot] = hi
            except Exception:  # noqa: BLE001
                pass

    # Bandwidths: rdrobust stores ``h`` (main) and ``b`` (bias-
    # correction) as a DataFrame indexed by ["h", "b"] with columns
    # ["left", "right"].
    bws = _safe_attr(fit, "bws")
    if bws is not None:
        try:
            fields["bandwidth_left"]  = float(bws.loc["h", "left"])
            fields["bandwidth_right"] = float(bws.loc["h", "right"])
        except Exception:  # noqa: BLE001
            pass
        try:
            fields["bandwidth_bias_correction_left"]  = float(bws.loc["b", "left"])
            fields["bandwidth_bias_correction_right"] = float(bws.loc["b", "right"])
        except Exception:  # noqa: BLE001
            pass

    # Effective N inside the main bandwidth: ``N_h`` is a list /
    # array of [left, right].
    n_h = _safe_attr(fit, "N_h")
    if n_h is not None:
        try:
            fields["effective_n_left"]  = int(n_h[0])
            fields["effective_n_right"] = int(n_h[1])
        except Exception:  # noqa: BLE001
            pass

    p_order = _safe_attr(fit, "p")
    if p_order is not None:
        try:
            fields["polynomial_order"] = int(p_order)
        except (TypeError, ValueError):
            pass
    cutoff = _safe_attr(fit, "c")
    if cutoff is not None:
        cf = _safe_float(cutoff)
        if cf is not None:
            fields["cutoff"] = cf

    bwsel = _safe_attr(fit, "bwselect")
    if isinstance(bwsel, str):
        fields["bandwidth_selector"] = bwsel
    kernel = _safe_attr(fit, "kernel")
    if isinstance(kernel, str):
        # rdrobust reports kernel capitalized ("Triangular"); the
        # sanitizer accepts lowercase only.
        fields["kernel"] = kernel.lower()

    fs_f = _safe_float(first_stage_f)
    if fs_f is not None:
        fields["first_stage_f"] = fs_f

    fields.update(extra)
    result(**fields)


def from_kaplan_meier(
    fit: Any,
    horizons: dict[str, float] | None = None,
    *,
    time_variable: str | None = None,
    event_variable: str | None = None,
    group_variable: str | None = None,
    logrank_chi_squared: float | None = None,
    logrank_p_value: float | None = None,
    n_groups: int | None = None,
    label: str | None = None,
    **extra: Any,
) -> None:
    """Emit a ``kaplan_meier`` payload from a fitted survival curve.

    Duck-typed on the attributes statsmodels' ``SurvfuncRight``
    exposes; lifelines' ``KaplanMeierFitter`` works via the same
    shape if the caller wraps it (rare in practice — most lifelines
    users will already have ``KaplanMeierFitter.survival_function_``
    and can pass a thin adapter).

    Required attributes on ``fit``:
        time         — 1-D array of observed times (the input
                       duration vector)
        status       — 1-D event indicator (1 = event, 0 = censored)
        surv_times   — event-time array (post-fit, sorted)
        surv_prob    — survival probability array, same length
                       as ``surv_times``
    Optional:
        surv_prob_se — Greenwood SE per event time (enables CIs)
        quantile     — callable for median (``fit.quantile(0.5)``)
        quantile_ci  — callable for CI (``fit.quantile_ci(0.5)``)

    ``horizons`` maps canonical labels (``"1y"`` / ``"3y"`` /
    ``"5y"`` / ``"10y"`` — the only labels the sanitizer's
    kaplan_meier shape accepts) to numeric time values in whatever
    unit the fit was built in. The helper interpolates S(h) using
    a step-look-up (KM is a step function) and computes n_at_risk(h)
    from the original duration vector ``fit.time``.

    Log-rank inference across groups isn't computed by statsmodels'
    ``SurvfuncRight``. The caller computes it (manually or via
    lifelines / R) and passes the chi² + p-value as kwargs.
    """
    try:
        print(fit.summary() if callable(getattr(fit, "summary", None)) else fit)
    except Exception:  # noqa: BLE001
        pass

    try:
        import numpy as np
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "sift.from_kaplan_meier requires numpy in the runtime"
        ) from e

    time_arr = _safe_attr(fit, "time")
    status_arr = _safe_attr(fit, "status")
    surv_times = _safe_attr(fit, "surv_times")
    surv_prob = _safe_attr(fit, "surv_prob")
    if surv_times is None or surv_prob is None:
        raise TypeError(
            "sift.from_kaplan_meier: ``fit`` must expose ``surv_times`` "
            "and ``surv_prob`` (statsmodels.SurvfuncRight shape)"
        )

    surv_times = np.asarray(surv_times, dtype=float)
    surv_prob  = np.asarray(surv_prob,  dtype=float)
    surv_prob_se = _safe_attr(fit, "surv_prob_se")
    if surv_prob_se is not None:
        surv_prob_se = np.asarray(surv_prob_se, dtype=float)

    fields: dict[str, Any] = {}
    if time_variable is not None:   fields["time_variable"]  = str(time_variable)
    if event_variable is not None:  fields["event_variable"] = str(event_variable)
    if group_variable is not None:  fields["group_variable"] = str(group_variable)
    if label is not None:           fields["label"]          = str(label)

    if time_arr is not None and status_arr is not None:
        time_arr   = np.asarray(time_arr,   dtype=float)
        status_arr = np.asarray(status_arr, dtype=int)
        fields["n_subjects"] = int(time_arr.size)
        fields["n_failures"] = int((status_arr != 0).sum())
    else:
        # Fall back to derived counts when raw inputs aren't exposed
        # (e.g., a thin adapter that only ships the curve). Total
        # events = sum of n_events; total at-risk = n_risk at t=0.
        n_events = _safe_attr(fit, "n_events")
        n_risk   = _safe_attr(fit, "n_risk")
        if n_events is not None and n_risk is not None:
            n_events = np.asarray(n_events, dtype=int)
            n_risk   = np.asarray(n_risk,   dtype=int)
            if n_risk.size > 0:
                fields["n_subjects"] = int(n_risk[0])
                fields["n_failures"] = int(n_events.sum())

    # Median + CI. ``SurvfuncRight.quantile(0.5)`` raises on heavily-
    # censored curves where the median is undefined; absorb that.
    qfn = _safe_attr(fit, "quantile")
    if callable(qfn):
        try:
            med = float(qfn(0.5))
            if math.isfinite(med):
                fields["median_survival_time"] = med
        except Exception:  # noqa: BLE001
            pass
    qcifn = _safe_attr(fit, "quantile_ci")
    if callable(qcifn):
        try:
            lo, hi = qcifn(0.5)
            if lo is not None and math.isfinite(float(lo)):
                fields["median_survival_ci_lower"] = float(lo)
            if hi is not None and math.isfinite(float(hi)):
                fields["median_survival_ci_upper"] = float(hi)
        except Exception:  # noqa: BLE001
            pass

    # Per-horizon scalars. KM is a step function — S(h) is the
    # survival probability at the latest event time ≤ h. n_at_risk(h)
    # is the count of subjects whose observed time ≥ h, computed from
    # the original duration array (the SurvfuncRight's ``n_risk``
    # attribute is at-event-time, not at-arbitrary-horizon).
    if horizons:
        z_975 = 1.959963984540054  # 97.5th percentile of N(0,1) for 95% CI
        for label_str, h_time in horizons.items():
            if not isinstance(label_str, str):
                continue
            try:
                h = float(h_time)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(h):
                continue
            # Step look-up: largest event time ≤ h.
            idx = int(np.searchsorted(surv_times, h, side="right")) - 1
            if idx < 0:
                # h precedes the first event time — S(h) = 1.
                s_h = 1.0
                se_h: float | None = 0.0
            elif idx >= surv_prob.size:
                idx = surv_prob.size - 1
                s_h = float(surv_prob[idx])
                se_h = (
                    float(surv_prob_se[idx])
                    if surv_prob_se is not None and idx < surv_prob_se.size
                    else None
                )
            else:
                s_h = float(surv_prob[idx])
                se_h = (
                    float(surv_prob_se[idx])
                    if surv_prob_se is not None and idx < surv_prob_se.size
                    else None
                )
            if math.isfinite(s_h):
                fields[f"survival_at_{label_str}"] = s_h
            # n_at_risk from raw durations when accessible.
            if time_arr is not None:
                n_risk_h = int((time_arr >= h).sum())
                fields[f"n_at_risk_{label_str}"] = n_risk_h
            # Linear Greenwood CI (clamped to [0, 1]).
            if se_h is not None and math.isfinite(se_h) and math.isfinite(s_h):
                lo = max(0.0, s_h - z_975 * se_h)
                hi = min(1.0, s_h + z_975 * se_h)
                fields[f"survival_at_{label_str}_ci_lower"] = lo
                fields[f"survival_at_{label_str}_ci_upper"] = hi

    for k, v in (
        ("logrank_chi_squared", logrank_chi_squared),
        ("logrank_p_value", logrank_p_value),
    ):
        vf = _safe_float(v)
        if vf is not None:
            fields[k] = vf
    if isinstance(n_groups, int) and n_groups > 0:
        fields["n_groups"] = n_groups

    fields.update(extra)
    result(type="kaplan_meier", **fields)


def _compute_vcov(model: Any) -> dict[str, dict[str, float]] | None:
    """Variance-covariance matrix of the coefficient estimates.

    statsmodels exposes ``model.cov_params()`` returning a labelled
    DataFrame whose row + column index are the coefficient names.
    Diagonals are the squared SEs (so ``standard_errors[name]`` =
    ``sqrt(vcov[name][name])``); off-diagonals carry the
    coefficient covariances that drive Wald tests, joint
    significance, and linear-combination CIs the model can compute
    on its own.

    Pure aggregate from sigma^2 * (X'X)^-1 — no per-observation
    information. Returns None when the model object doesn't expose
    a parameter covariance (sklearn-shaped, custom estimators,
    etc.); the caller drops the field rather than emitting null.
    """
    fn = getattr(model, "cov_params", None)
    if fn is None or not callable(fn):
        return None
    try:
        cov = fn()
    except Exception:  # noqa: BLE001
        return None
    # statsmodels returns a pandas DataFrame for formula fits and a
    # numpy array for raw OLS(y, X). Handle both.
    to_dict = getattr(cov, "to_dict", None)
    if callable(to_dict):
        try:
            raw = to_dict()
        except Exception:  # noqa: BLE001
            return None
        out: dict[str, dict[str, float]] = {}
        for row_key, row_dict in raw.items():
            if not isinstance(row_dict, dict):
                continue
            inner: dict[str, float] = {}
            for col_key, val in row_dict.items():
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(fval):
                    inner[str(col_key)] = fval
            if inner:
                out[str(row_key)] = inner
        return out or None
    # numpy array path: pair with exog_names from the inner model.
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return None
    inner_model = getattr(model, "model", None)
    names = list(
        getattr(inner_model, "exog_names", []) or []
    ) if inner_model is not None else []
    if len(names) != arr.shape[0]:
        return None
    out_arr: dict[str, dict[str, float]] = {}
    for i, row_name in enumerate(names):
        array_inner: dict[str, float] = {}
        for j, col_name in enumerate(names):
            v = float(arr[i, j])
            if math.isfinite(v):
                array_inner[col_name] = v
        if array_inner:
            out_arr[row_name] = array_inner
    return out_arr or None


def _compute_vif(model: Any, predictors: list[str]) -> dict[str, float] | None:
    """Variance inflation factor per predictor.

    For each predictor x_i, fit an auxiliary OLS of x_i on the OTHER
    predictors (intercept handled by the design matrix). Return
    1 / (1 - R^2_aux). Pure aggregate over the design columns; no
    per-row data crosses back.

    Skipped silently if numpy isn't installed, the model lacks an
    accessible design matrix, or any predictor is perfectly collinear
    with the rest (R^2_aux >= 1) — the caller treats absence as
    "diagnostic unavailable" rather than "no collinearity".
    """
    try:
        import numpy as np
    except Exception:  # noqa: BLE001 — numpy missing → quietly omit
        return None
    inner = getattr(model, "model", None)
    X = getattr(inner, "exog", None) if inner is not None else None
    if X is None:
        return None
    try:
        X = np.asarray(X, dtype=float)
    except Exception:  # noqa: BLE001
        return None
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
        return None
    exog_names = list(getattr(inner, "exog_names", []) or [])
    if len(exog_names) != X.shape[1]:
        return None

    out: dict[str, float] = {}
    for i, name in enumerate(exog_names):
        if name in ("const", "Intercept", "(Intercept)"):
            continue
        if predictors and name not in predictors:
            # Only emit VIF for declared predictors so the sanitizer's
            # cross-field key validation accepts the result.
            continue
        xi = X[:, i]
        X_others = np.delete(X, i, axis=1)
        try:
            beta, *_ = np.linalg.lstsq(X_others, xi, rcond=None)
            xi_hat = X_others @ beta
            ss_res = float(np.sum((xi - xi_hat) ** 2))
            ss_tot = float(np.sum((xi - np.mean(xi)) ** 2))
        except Exception:  # noqa: BLE001
            continue
        if ss_tot <= 0 or ss_res < 0:
            continue
        r2_aux = 1.0 - ss_res / ss_tot
        if r2_aux >= 1.0 or r2_aux < 0.0:
            continue
        out[name] = 1.0 / (1.0 - r2_aux)
    return out or None


def _compute_condition_number(model: Any) -> float | None:
    """``kappa(X)`` — ratio of the largest to smallest singular
    value of the design matrix. High values flag near-collinearity
    that VIF can miss when it's spread across many predictors.

    Returns ``None`` if numpy is missing or the design isn't
    reachable; the caller drops the field rather than emitting a
    confusing ``null``.
    """
    try:
        import numpy as np
    except Exception:  # noqa: BLE001
        return None
    inner = getattr(model, "model", None)
    X = getattr(inner, "exog", None) if inner is not None else None
    if X is None:
        return None
    try:
        return float(np.linalg.cond(np.asarray(X, dtype=float)))
    except Exception:  # noqa: BLE001
        return None


def from_t_test(res: Any, *, n1: int, n2: int | None = None,
                mean1: float | None = None, mean2: float | None = None,
                test_type: str | None = None, **extra: Any) -> None:
    """Emit a ``t_test`` payload from a SciPy ``ttest_ind`` /
    ``ttest_rel`` / ``ttest_1samp`` result.

    SciPy's result object carries ``statistic``, ``pvalue``, and
    ``df`` (on newer versions); it does NOT carry sample sizes or
    group means, so callers must pass them explicitly. The
    docstring lists this requirement loudly because forgetting it
    is the #1 way scripts produce a payload the sanitizer rejects.

    ``test_type`` defaults to ``"one_sample"`` when only ``n1`` is
    given, ``"two_sample"`` when ``n2`` is set. Pass ``"welch"`` /
    ``"paired"`` explicitly when the underlying scipy call used those
    variants — scipy's result object doesn't carry that info itself.
    The field name is ``test_type`` (not ``subtype``) to match the R
    and Stata helpers and the sanitizer's ``_TTEST_REQUIRED`` set —
    drift here means the sanitizer drops every emit.
    """
    try:
        print(repr(res))
    except Exception:  # noqa: BLE001
        pass

    statistic = _safe_float(getattr(res, "statistic", None))
    pvalue = _safe_float(getattr(res, "pvalue", None))
    df = _safe_float(getattr(res, "df", None))

    if test_type is None:
        test_type = "one_sample" if n2 is None else "two_sample"

    # ``_safe_int`` instead of bare ``int(...)``: ``int(float('nan'))``
    # raises ``ValueError`` and crashes the helper before any payload
    # reaches disk. NaN sample sizes are unusual (you don't usually
    # have NaN counts of observations) but a careless caller passing
    # ``len(df_with_nans_in_index)`` or similar shouldn't lose the
    # entire result. Coerce to None and let the sanitizer reject the
    # field with a clear reason instead.
    fields: dict[str, Any] = {
        "test_type": test_type,
        "t_statistic": statistic,
        "p_value": pvalue,
        "degrees_of_freedom": df,
        "n1": _safe_int(n1),
        "mean1": mean1,
    }
    if n2 is not None:
        fields["n2"] = _safe_int(n2)
    if mean2 is not None:
        fields["mean2"] = mean2
    fields.update(extra)
    result(type="t_test", **fields)


def from_summarize(variable: str, *, n: int, mean: float, sd: float,
                   missing_count: int = 0,
                   distinct_count: int | None = None,
                   **extra: Any) -> None:
    """Emit a ``descriptive`` payload for a single numeric variable.
    Mirrors ``sift$from_summarize`` in the R library.

    ``min_value`` / ``max_value`` are no longer accepted: the
    sanitizer drops them in every payload because nothing in the
    payload binds the reported values to the named variable's actual
    column. Researchers who need a variable's range should use a
    Sift-owned path (request_data with a future bounds extension)
    rather than a script-emitted descriptive.

    ``distinct_count`` is the exact number of unique values. Unlike
    ``mean`` / ``sd`` (floats, which the sanitizer rounds to an
    N-appropriate number of significant figures), it is an allowed
    *integer* field and passes through unrounded — so this is the
    supported way to release an exact unique/cardinality count. The
    whole-payload ``n >= 10`` minimum still applies. Compute it from
    the data and pass it in, e.g.::

        sift.from_summarize(
            "ein", n=len(df),
            mean=df["ein"].mean(), sd=df["ein"].std(),
            missing_count=int(df["ein"].isna().sum()),
            distinct_count=int(df["ein"].nunique()),
        )
    """
    # ``_safe_int`` instead of bare ``int(...)``: avoid crashing the
    # whole helper on a NaN count that a careless caller forwarded
    # from a partial aggregation. The sanitizer will reject ``None``
    # integer fields with a clear reason; that's strictly better than
    # losing the entire result to a ``ValueError``.
    fields: dict[str, Any] = {
        "variable": variable,
        "n": _safe_int(n),
        "mean": _safe_float(mean),
        "sd": _safe_float(sd),
        "missing_count": _safe_int(missing_count),
    }
    # Only attach ``distinct_count`` when supplied AND coercible. Emitting
    # ``None`` would make the sanitizer drop it with a noisy "expected int"
    # transformation; omitting the key entirely is cleaner and equivalent.
    if distinct_count is not None:
        _dc = _safe_int(distinct_count)
        if _dc is not None:
            fields["distinct_count"] = _dc
    fields.update(extra)
    result(type="descriptive", **fields)


def from_table(variable: str, counts: Any, *, n: int | None = None,
               missing_count: int = 0, **extra: Any) -> None:
    """Emit a 1-D ``frequency_table`` payload.

    ``counts`` accepts a dict ``{"level": count}``, a pandas Series,
    or anything the encoder can normalise to that shape.
    """
    if hasattr(counts, "to_dict"):
        counts_dict = counts.to_dict()
    else:
        counts_dict = dict(counts)
    # Filter out non-finite values up front. ``int(float('nan'))``
    # raises ``ValueError`` and would crash the helper before the
    # payload reaches disk — pandas value_counts() doesn't usually
    # produce NaN cells, but a hand-built counts dict or a
    # ``pd.crosstab`` with all-missing combinations can. Drop the
    # NaN levels rather than coercing to 0 (which would lie about
    # observed-zero vs. unobserved).
    clean_counts: dict[str, int] = {}
    for k, v in counts_dict.items():
        iv = _safe_int(v)
        if iv is not None:
            clean_counts[str(k)] = iv
    # Default: auto-compute n from the clean counts. Branch on caller-
    # supplied n so a NaN / non-finite caller value flows through
    # ``_safe_int`` and lands as ``None`` (which the sanitizer
    # rejects), rather than getting silently coerced to 0 by
    # ``or 0``. ``n=0`` and a malformed/non-finite n both used to
    # serialize as ``"n": 0`` — a valid-looking sanitizer payload
    # that hid the upstream undefined-count problem.
    if n is None:
        safe_n: int | None = sum(clean_counts.values())
    else:
        safe_n = _safe_int(n)
    fields = {
        "variable": variable,
        "counts": clean_counts,
        "n": safe_n,
        "missing_count": _safe_int(missing_count),
    }
    fields.update(extra)
    result(type="frequency_table", **fields)


def from_crosstab(table: Any, *, row_variable: str | None = None,
                  col_variable: str | None = None,
                  missing_count: int = 0, **extra: Any) -> None:
    """Emit a 2-D ``crosstab`` payload from a pandas DataFrame
    produced by ``pd.crosstab(...)`` (or any 2-D table-like)."""
    try:
        import pandas as pd
    except ImportError:
        pd = None  # type: ignore[assignment]

    # NaN-tolerant cell coercion. ``pd.crosstab`` can produce NaN
    # cells when ``dropna=False`` leaves un-observed row/col
    # combinations, and bare ``int(float('nan'))`` raises
    # ``ValueError`` and crashes the helper before any payload
    # reaches disk. Drop NaN cells silently — "the cell was never
    # observed" is structurally different from "the cell has 0
    # observations" and conflating them via ``or 0`` would lie.
    def _cell(v: Any) -> int | None:
        return _safe_int(v)

    if pd is not None and isinstance(table, pd.DataFrame):
        counts: dict[str, dict[str, int]] = {}
        for row_label, row in table.iterrows():
            row_dict: dict[str, int] = {}
            for col in table.columns:
                iv = _cell(row[col])
                if iv is not None:
                    row_dict[str(col)] = iv
            counts[str(row_label)] = row_dict
        if row_variable is None:
            row_variable = str(table.index.name or "row")
        if col_variable is None:
            col_variable = str(table.columns.name or "column")
    else:
        # Caller passed a pre-built nested dict.
        counts = {}
        for rk, rv in dict(table).items():
            row_dict = {}
            for ck, cv in (rv or {}).items():
                iv = _cell(cv)
                if iv is not None:
                    row_dict[str(ck)] = iv
            counts[str(rk)] = row_dict
        row_variable = row_variable or "row"
        col_variable = col_variable or "column"

    # ``missing_count`` rides as ``None`` when the caller passed
    # a NaN / non-finite value so the sanitizer rejects it. The
    # previous ``or 0`` silently coerced bad inputs to a
    # valid-looking ``0``, hiding the upstream undefined-count
    # problem from disclosure-control review.
    fields = {
        "row_variable": row_variable,
        "col_variable": col_variable,
        "counts": counts,
        "missing_count": _safe_int(missing_count),
    }
    fields.update(extra)
    result(type="crosstab", **fields)


def from_magnitude_table(df: Any, group_var: str, value_var: str, *,
                         aggregation: str = "sum", **extra: Any) -> None:
    """Emit a ``magnitude_table`` payload (sum or mean of a numeric
    by group). Pre-aggregates here; the sanitizer applies the
    (1, 85%)-dominance rule on the per-cell ``max_share``.

    Field-name and dominance-metric contract MUST match the R / Stata
    helpers and the sanitizer's ``_MAGTAB_REQUIRED`` set:

    - top-level key is ``row_variable`` (not ``group_variable``).
    - per-cell ``max_share`` is the share-in-[0,1] dominance metric
      ``max(abs(values)) / sum(abs(values))`` — NOT the raw top
      absolute value. The sanitizer suppresses cells whose
      ``max_share`` exceeds the dominance threshold (default 0.85)
      and strips ``max_share`` from the visible payload before it
      reaches the model.

    Empty groups (no non-missing observations) emit ``{value: 0,
    n: 0, max_share: 0}`` so the sanitizer suppresses them on the
    n side rather than failing required-field validation. Same shape
    R uses.
    """
    if aggregation not in ("sum", "mean"):
        raise ValueError(
            f"aggregation must be 'sum' or 'mean', got {aggregation!r}"
        )
    grouped = df.groupby(group_var)[value_var]
    cells: dict[str, dict[str, Any]] = {}
    for key, group in grouped:
        clean = group.dropna()
        n_cell = int(len(clean))
        if n_cell == 0:
            cells[str(key)] = {"value": 0.0, "n": 0, "max_share": 0.0}
            continue
        agg_value = float(clean.sum()) if aggregation == "sum" else float(clean.mean())
        # Dominance metric: top contributor's share of the absolute
        # total. All-zero groups have undefined share — emit 0 (no
        # contributor dominates because there's no magnitude). Same
        # guard the R helper applies.
        abs_vals = clean.abs()
        total_abs = float(abs_vals.sum())
        max_share = (
            float(abs_vals.max()) / total_abs if total_abs > 0 else 0.0
        )
        cells[str(key)] = {
            "value": agg_value,
            "n": n_cell,
            "max_share": max_share,
        }
    # Reject ``**extra`` keys that would override fields the helper
    # computes from raw data. Without this guard, a caller could pass
    # ``cells={...}`` (or ``row_variable=...`` etc.) and the
    # ``fields.update(extra)`` below would clobber the helper's
    # computation. The ``_via_helper`` marker stamped at write time
    # would then authenticate attacker-supplied values, and the
    # sanitizer (which trusts the marker to skip recomputing
    # ``max_share``) would let a forged ``max_share=0`` bypass the
    # dominance gate. The marker is meant to prove the disclosure-
    # metric fields came from the helper, not just that the helper
    # was called.
    _reserved = {
        "type", "row_variable", "value_variable", "aggregation",
        "cells", "_via_helper",
    }
    forbidden = sorted(set(extra) & _reserved)
    if forbidden:
        raise ValueError(
            "from_magnitude_table: cannot override helper-computed "
            f"fields via keyword arguments: {forbidden}. These are "
            "computed from the DataFrame and bound to the "
            "_via_helper provenance marker."
        )
    fields = {
        "row_variable": group_var,
        "value_variable": value_var,
        "aggregation": aggregation,
        "cells": cells,
    }
    fields.update(extra)
    # Helper-provenance marker. The sanitizer requires this for
    # ``magnitude_table`` because cell-level ``max_share`` is
    # consulted-only and stripped; without proof that max_share
    # came from raw-data computation a malicious script could
    # publish a dominance-violating value with a forged
    # ``max_share=0`` and skip the dominance gate. Write directly
    # via _write_result, bypassing ``result()`` (which strips this
    # field from caller fields), so the marker can't be forged
    # through the generic API.
    payload = {
        "type": "magnitude_table",
        **fields,
        "_via_helper": "from_magnitude_table",
    }
    _write_result(payload)


def from_correlation(
    df: Any,
    *,
    variables: list[str] | None = None,
    method: str = "pearson",
    **extra: Any,
) -> None:
    """Emit a ``correlation_matrix`` payload from a pandas DataFrame.

    By default correlates every numeric column; pass ``variables`` to
    restrict to a named subset. ``method`` is one of ``'pearson'``,
    ``'spearman'``, ``'kendall'`` — anything else is rejected by the
    sanitizer.

    Sample size N is the number of *complete* rows over the chosen
    variables (pandas ``.dropna()`` semantics). Sub-threshold N is
    rejected by the sanitizer with a clear reason — at very low N a
    correlation of 0.99 between two columns is just "the three points
    are collinear" and could imply individual coordinates.

    Also prints the correlation matrix to stdout so the researcher
    sees the conventional view in the raw log panel.
    """
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(
            f"method must be 'pearson' / 'spearman' / 'kendall', "
            f"got {method!r}"
        )
    # Pick columns. Default to numeric columns if no list given;
    # respect the order the caller passed when they did.
    if variables is None:
        # Lazy: keep numeric + boolean (booleans correlate fine).
        try:
            import numpy as _np  # noqa: F401
        except ImportError:
            pass
        variables = [
            c for c in df.columns
            if str(df[c].dtype) not in ("object", "string", "category")
        ]
    if not variables:
        raise ValueError(
            "from_correlation: no numeric columns found and no "
            "``variables`` provided"
        )
    sub = df[variables]
    # Correlation matrix on rows where ALL chosen variables are
    # observed. Emitting N as `len(complete_rows)` is the honest
    # number — pairwise N-by-pair would be deceptive (each off-
    # diagonal would be a different sample).
    complete = sub.dropna()
    n = int(len(complete))
    missing_count = int(len(df) - n)
    corr = complete.corr(method=method)
    try:
        print(corr)
    except Exception:  # noqa: BLE001 — never let print block emit
        pass
    correlations: dict[str, dict[str, float]] = {}
    for row_var in variables:
        row_dict: dict[str, float] = {}
        for col_var in variables:
            try:
                v = float(corr.at[row_var, col_var])
                if math.isfinite(v):
                    row_dict[col_var] = v
            except Exception:  # noqa: BLE001
                continue
        if row_dict:
            correlations[row_var] = row_dict
    fields: dict[str, Any] = {
        "n": n,
        "variables": list(variables),
        "method": method,
        "correlations": correlations,
        "missing_count": missing_count,
    }
    fields.update(extra)
    result(type="correlation_matrix", **fields)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_dict(thing: Any, names: list[str] | None = None) -> dict[str, Any]:
    """Normalise a pandas Series / dict-like / ndarray to a plain
    ``{name: value}`` dict the encoder can serialise without further
    coercion.

    ``names`` is used as a fallback labelling when ``thing`` doesn't
    carry its own index — statsmodels ``PHRegResults`` ships params /
    bse / tvalues / pvalues as bare ndarrays without column names, so
    we pair them with the inner model's ``exog_names``. Without this,
    ``dict(ndarray)`` raises ``TypeError: cannot convert dictionary
    update sequence element #0 to a sequence`` and the helper aborts
    before the payload is written — the same silent-failure mode the
    R/Stata audits caught for Cox.
    """
    if hasattr(thing, "to_dict"):
        return {str(k): v for k, v in thing.to_dict().items()}
    # Index-less iterable (ndarray, list, tuple): require a names list.
    try:
        items = list(thing)
    except TypeError:
        return {str(k): v for k, v in dict(thing).items()}
    if names is not None and len(names) == len(items):
        return {str(names[i]): items[i] for i in range(len(items))}
    # Last-resort positional naming so an unnamed numeric vector at
    # least round-trips with deterministic keys rather than raising.
    return {f"x{i}": items[i] for i in range(len(items))}


def _safe_attr(obj: Any, name: str) -> Any:
    """Attribute fetch that survives properties raising
    ``NotImplementedError`` / ``ValueError`` etc. — statsmodels
    ``IV2SLS`` results define ``llf`` / ``aic`` / ``bic`` as
    properties that raise rather than missing, so a plain
    ``getattr(obj, name, None)`` propagates and aborts the helper."""
    try:
        return getattr(obj, name, None)
    except Exception:  # noqa: BLE001
        return None


def _safe_float(x: Any) -> float | None:
    """Coerce to float, returning None for None/NaN/Inf or
    non-coercible inputs. Keeps the payload JSON-clean."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _safe_int(x: Any) -> int | None:
    """Coerce to int, returning None for non-coercible / NaN inputs."""
    if x is None:
        return None
    try:
        f = float(x)
        if not math.isfinite(f):
            return None
        return int(f)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Plot helpers — model-output visualizations only
# ---------------------------------------------------------------------------
#
# Plots produced via these helpers are surfaced to the model on the
# next turn as image attachments. Raw-data plots (a histogram of an
# observed column, a scatter of all rows) are NOT covered on
# purpose — they would expose the data itself, which is the privacy
# line Sift is built to keep.
#
# Allowlist: only files written via these helpers (and registered
# in the manifest) are visible. ``plt.savefig(...)`` outside the
# helpers does NOT cross to the model — the file lands in the run
# dir for the researcher's eyes only.
#
# Mechanism mirrors the R library: write a PNG into
# ``<run_dir>/_sift_plots/`` and append a JSONL entry to
# ``manifest.jsonl``. The bridge reads only the manifest.


def _plots_dir() -> Any:
    """Return the ``_sift_plots`` directory beside the result file,
    creating it on first use. None when ``SIFT_RESULT_PATH`` isn't
    set (caller didn't go through the executor)."""
    if not _RESULT_PATH:
        return None
    from pathlib import Path
    d = Path(_RESULT_PATH).parent / "_sift_plots"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _unique_plot_name(d: Any, base: str) -> str:
    """Return ``base`` if no file by that name exists in ``d``;
    otherwise append ``_2``, ``_3``, ... before the extension.

    Without this, calling ``plot_coefficients(fit1)`` then
    ``plot_coefficients(fit2)`` in one script would:
      1. write ``coefficients.png`` (fit1) + manifest entry
      2. *overwrite* ``coefficients.png`` (now fit2) + a SECOND
         manifest entry pointing at the same path
    The bridge would read two manifest rows that point to the same
    image and the model would see two "different" plots that are
    in fact fit2 twice. ``plot_interaction`` already side-steps
    this by suffixing the variable name into the filename; the
    other helpers need a counter.
    """
    from pathlib import Path
    p = Path(base)
    stem, ext = p.stem, p.suffix
    candidate = base
    i = 2
    while (d / candidate).exists():
        candidate = f"{stem}_{i}{ext}"
        i += 1
    return candidate


def _append_plot_manifest(file: str, kind: str, label: str | None) -> None:
    d = _plots_dir()
    if d is None:
        return
    # Stamp every entry with the per-run token. The executor validates
    # this field after the script finishes and drops any entry whose
    # token is missing or wrong; that strips manifest rows a script
    # could otherwise have appended directly (saving a raw-data plot
    # under ``_sift_plots/`` and labeling it ``coefficients`` to slip
    # past the disclosure-control allowlist for vision attachment).
    # Same posture as the result-payload ``_token`` field — a
    # determined script can still reach into ``sift._RUN_TOKEN`` to
    # forge the value, but doing so requires obvious code in the
    # script the researcher reviews.
    entry: dict[str, Any] = {"file": file, "kind": kind, "_token": _RUN_TOKEN}
    if label:
        entry["label"] = label
    try:
        with (d / "manifest.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _append_plot_helper_error(helper: str, exc: BaseException) -> None:
    """Record a structured plot-helper failure so ``submit_script``
    can surface it in the tool result the MODEL receives. Without
    this, helper failures (matplotlib missing, etc.) only land in
    stderr, and the model says "thumbnail should be visible above"
    while the researcher sees nothing — the loop the user reported.

    The runner reads ``_sift_plots/helper_errors.jsonl`` after the
    run and includes a summary in the structured tool result so
    the model can react instead of guessing.
    """
    d = _plots_dir()
    if d is None:
        return
    error_kind = type(exc).__name__
    message = str(exc)
    fix: str | None = None
    lower = message.lower()
    if "matplotlib" in lower or "no module named 'matplotlib'" in lower:
        fix = "pip install matplotlib"
    elif "no module named 'scipy'" in lower:
        fix = "pip install scipy"
    elif "no module named 'statsmodels'" in lower:
        fix = "pip install statsmodels"
    entry: dict[str, Any] = {
        "helper": helper, "error": error_kind, "message": message,
    }
    if fix:
        entry["fix"] = fix
    try:
        with (d / "helper_errors.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _helper_failed(
    helper: str,
    message: str,
    exc: BaseException | None = None,
) -> None:
    """Record a graceful plot-helper failure: human-readable stderr
    line for the researcher's raw log AND a structured jsonl entry
    so the model's tool result surfaces the failure cause.

    The plot helpers wrap their bodies in ``try/except Exception``,
    but they ALSO have several early-return paths for shape problems
    (no ``.params``, malformed ``models`` dict, etc.). Those paths
    used to write only stderr — the model saw "no plots produced"
    with no hint why, then guessed and looped. Calling this helper
    at each early-return keeps both audiences informed without
    forcing the caller to raise (which would also unwind the
    surrounding analysis script's own bookkeeping).
    """
    sys.stderr.write(f"sift.{helper}: {message}\n")
    _append_plot_helper_error(
        helper, exc if exc is not None else RuntimeError(message)
    )


def plot_residuals(fitted: Any, label: str | None = None) -> None:
    """Write the four standard residual diagnostic panels for a
    statsmodels fit and register them with the plot manifest.

    Errors inside this helper print to stderr but never raise — a
    broken plot helper must not break the analysis script.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        # Force a non-interactive backend before importing pyplot:
        # the executor runs scripts headless and any default GUI
        # backend would either crash (no display) or pop a window
        # the researcher didn't ask for.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        resid = getattr(fitted, "resid", None)
        fitted_vals = getattr(fitted, "fittedvalues", None)
        if resid is None or fitted_vals is None:
            _helper_failed(
                "plot_residuals",
                "fitted object has no .resid / .fittedvalues; skipping",
            )
            return

        try:
            import numpy as _np
            resid_arr = _np.asarray(resid, dtype=float)
            fitted_arr = _np.asarray(fitted_vals, dtype=float)
        except ImportError as e:
            _helper_failed("plot_residuals", "numpy missing", exc=e)
            return

        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        # Residuals vs fitted
        axes[0, 0].scatter(fitted_arr, resid_arr, alpha=0.5, s=12)
        axes[0, 0].axhline(0, color="gray", lw=0.8)
        axes[0, 0].set_xlabel("Fitted values")
        axes[0, 0].set_ylabel("Residuals")
        axes[0, 0].set_title("Residuals vs Fitted")
        # Normal Q-Q
        try:
            from scipy import stats as _stats
            _stats.probplot(resid_arr, dist="norm", plot=axes[0, 1])
            axes[0, 1].set_title("Normal Q-Q")
        except ImportError:
            axes[0, 1].text(0.5, 0.5, "scipy not installed",
                            ha="center", va="center")
            axes[0, 1].set_title("Normal Q-Q")
        # Scale-Location
        sd = float(resid_arr.std() or 1.0)
        std_resid = (resid_arr - resid_arr.mean()) / sd
        sqrt_abs = (abs(std_resid)) ** 0.5
        axes[1, 0].scatter(fitted_arr, sqrt_abs, alpha=0.5, s=12)
        axes[1, 0].set_xlabel("Fitted values")
        axes[1, 0].set_ylabel(r"$\sqrt{|standardized\ resid|}$")
        axes[1, 0].set_title("Scale-Location")
        # Residual distribution
        axes[1, 1].hist(resid_arr, bins=20)
        axes[1, 1].set_xlabel("Residual")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Residual distribution")
        fig.tight_layout()

        fname = _unique_plot_name(d, "residuals.png")
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "residuals",
            label or "Residual diagnostics",
        )
    except Exception as e:  # noqa: BLE001 — never let plotting fail the script
        sys.stderr.write(f"sift.plot_residuals failed: {e}\n")
        _append_plot_helper_error("plot_residuals", e)


def plot_coefficients(fitted: Any, label: str | None = None) -> None:
    """Forest plot of coefficient point estimates with 95% CIs.

    Operates ONLY on the fit's ``params`` and ``conf_int()`` —
    pure functions of model output, never the raw data. The
    helper is the gate; there is no escape-hatch path that
    accepts an arbitrary file (that would let a histogram of
    raw rows pose as a coefficient plot — bypassing the
    privacy line the entire system rests on).

    Errors inside the helper print to stderr but never raise —
    a broken plot helper must not break the analysis around it.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np

        params = getattr(fitted, "params", None)
        if params is None:
            _helper_failed(
                "plot_coefficients",
                "fitted object has no .params; need a statsmodels-style fit",
            )
            return
        try:
            ci = fitted.conf_int(alpha=0.05)
        except Exception as e:  # noqa: BLE001
            _helper_failed(
                "plot_coefficients", f"conf_int failed: {e}", exc=e,
            )
            return

        # Drop intercept by default — researchers almost never want
        # it on the same scale as predictors. (If they do, they can
        # call this on a model fit without an intercept term.)
        names = list(getattr(params, "index", range(len(params))))
        ests = _np.asarray(params, dtype=float)
        try:
            import pandas as _pd
            if isinstance(ci, _pd.DataFrame):
                lows = ci.iloc[:, 0].to_numpy(dtype=float)
                highs = ci.iloc[:, 1].to_numpy(dtype=float)
            else:
                ci_arr = _np.asarray(ci, dtype=float)
                lows, highs = ci_arr[:, 0], ci_arr[:, 1]
        except ImportError:
            ci_arr = _np.asarray(ci, dtype=float)
            lows, highs = ci_arr[:, 0], ci_arr[:, 1]

        keep = [
            i for i, n in enumerate(names)
            if str(n).lower() not in ("intercept", "const", "_cons")
        ]
        if not keep:
            _helper_failed(
                "plot_coefficients",
                "nothing to plot after dropping intercept term",
            )
            return
        names = [str(names[i]) for i in keep]
        ests = ests[keep]
        lows = lows[keep]
        highs = highs[keep]

        fig, ax = plt.subplots(figsize=(8, max(2.5, 0.4 * len(names) + 1)))
        y = _np.arange(len(names))
        ax.hlines(y, lows, highs, lw=2, color="#4C78A8")
        ax.scatter(ests, y, s=60, color="#4C78A8", zorder=3)
        ax.axvline(0, color="gray", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Coefficient (95% CI)")
        ax.set_title("Coefficients")
        fig.tight_layout()

        fname = _unique_plot_name(d, "coefficients.png")
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "coefficients",
            label or "Coefficient estimates with 95% CIs",
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"sift.plot_coefficients failed: {e}\n")
        _append_plot_helper_error("plot_coefficients", e)


def plot_estimate_comparison(
    models: Any,
    coef: str,
    label: str | None = None,
) -> None:
    """Forest plot comparing one coefficient across multiple fits.

    ``models`` is a dict mapping label → fitted model (the keys
    become y-axis labels). Each fit must expose ``params`` and
    ``cov_params()`` (statsmodels) or ``params``/``bse``. ``coef``
    is the coefficient name to extract from each.

    Use case: "female gap before/after controls" — two regressions,
    one plot, no language switching to compose them. Same posture
    as the other ``plot_*`` helpers: produces a model-output plot
    only (point estimates + CIs from each fit's covariance), never
    raw rows.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        if not isinstance(models, dict) or len(models) < 2:
            _helper_failed(
                "plot_estimate_comparison",
                "`models` must be a dict of at least 2 fits keyed by label",
            )
            return
        if not isinstance(coef, str) or not coef:
            _helper_failed(
                "plot_estimate_comparison",
                "`coef` must be a coefficient name string",
            )
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np

        labels: list[str] = []
        ests: list[float] = []
        ses: list[float] = []
        for nm, fit in models.items():
            params = getattr(fit, "params", None)
            if params is None:
                _helper_failed(
                    "plot_estimate_comparison",
                    f"model {nm!r} has no .params; need a statsmodels-style fit",
                )
                return
            try:
                idx = list(params.index)
            except AttributeError:
                idx = [str(i) for i in range(len(params))]
            if coef not in idx:
                _helper_failed(
                    "plot_estimate_comparison",
                    f"coef {coef!r} not in model {nm!r}",
                )
                return
            est = float(params[coef])
            # SE: prefer .bse[coef]; fall back to sqrt of cov_params
            # diagonal. statsmodels exposes both.
            bse = getattr(fit, "bse", None)
            if bse is not None and coef in list(bse.index):
                se = float(bse[coef])
            else:
                cov = fit.cov_params()
                se = float(_np.sqrt(cov.loc[coef, coef]))
            labels.append(str(nm))
            ests.append(est)
            ses.append(se)

        ests_arr: Any = _np.asarray(ests, dtype=float)
        ses_arr: Any = _np.asarray(ses, dtype=float)
        lows = ests_arr - 1.96 * ses_arr
        highs = ests_arr + 1.96 * ses_arr

        n = len(labels)
        fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.5 * n + 1.5)))
        y = _np.arange(n)
        ax.hlines(y, lows, highs, lw=2, color="#4C78A8")
        ax.scatter(ests_arr, y, s=70, color="#4C78A8", zorder=3)
        ax.axvline(0, color="gray", lw=1, ls="--")
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{coef} (95% CI)")
        ax.set_title(f"Estimate comparison: {coef}")
        fig.tight_layout()

        fname = _unique_plot_name(d, "estimate_comparison.png")
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "estimate_comparison",
            label or f"Estimate comparison: {coef}",
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"sift.plot_estimate_comparison failed: {e}\n")
        _append_plot_helper_error("plot_estimate_comparison", e)


def plot_interaction(
    fitted: Any,
    var: str,
    data: Any | None = None,
    label: str | None = None,
    xlab: str | None = None,
    ylab: str | None = None,
    title: str | None = None,
) -> None:
    """Predicted-response curve across one predictor with the others
    held at their means (numeric) or first level (categorical).
    Bands are 1.96 * SE of the predicted mean.

    ``fitted`` is a statsmodels results object. ``data`` is the
    DataFrame the model was fit on (statsmodels doesn't reliably
    expose this back through the results object once formulae are
    involved). Optional ``xlab`` / ``ylab`` / ``title`` override
    defaults that fall back to the variable name and a generic
    "Predicted response" label.

    Falls through quietly with a stderr note when shape can't be
    derived.
    """
    try:
        d = _plots_dir()
        if d is None:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as _np
        import pandas as _pd

        if data is None:
            data = getattr(fitted.model, "data", None)
            data = getattr(data, "frame", None) if data is not None else None
        if data is None or var not in getattr(data, "columns", []):
            _helper_failed(
                "plot_interaction",
                f"pass data=... that contains column {var!r}; "
                f"couldn't derive it from the fit",
            )
            return

        # Build the prediction grid and the held-at-means template.
        # Disclosure-control note: the rendered PNG is allowlisted for
        # model vision (kind="interaction"), so anything legible on
        # the x-axis crosses the SDC boundary. The previous version
        # used col.min()/col.max() (numeric) and col.unique()
        # (categorical), which surfaced raw extrema and rare-level
        # identities the JSON sanitizer would have refused. We now
        # build a disclosure-safe grid:
        #   - numeric: mean ± 2*sd, which discloses no more than
        #     the descriptive sanitizer's already-allowed mean+sd
        #     pair. Tick labels are stripped so the model only sees
        #     curve SHAPE, not absolute x values.
        #   - categorical: drop levels whose observed count is
        #     below the SDC cell_suppression_threshold (matches
        #     frequency_table policy: rare-level identities are
        #     themselves disclosive). If that leaves nothing, refuse
        #     the plot rather than silently dropping back to all
        #     levels.
        col = data[var]
        is_numeric = _pd.api.types.is_numeric_dtype(col)
        # Mirror SDCConfig.cell_suppression_threshold default; the
        # runtime library has no direct access to the runner-side
        # config object.
        _CELL_SUPPRESSION_THRESHOLD = 10
        # Cap how many categorical bars can land on the plot — a
        # 200-level bar chart isn't readable AND multiplies the
        # data-channel surface through tick labels.
        _CATEGORICAL_LEVEL_CAP = 20
        suppression_note: str | None = None
        grid: Any
        if is_numeric:
            cleaned = col.dropna()
            if len(cleaned) < _CELL_SUPPRESSION_THRESHOLD:
                _helper_failed(
                    "plot_interaction",
                    f"variable {var!r} has fewer than "
                    f"{_CELL_SUPPRESSION_THRESHOLD} non-missing "
                    f"observations; below the disclosure threshold",
                )
                return
            mu = float(cleaned.mean())
            sd = float(cleaned.std(ddof=1)) if len(cleaned) > 1 else 0.0
            if not _np.isfinite(sd) or sd <= 0:
                # Constant variable or single observation — nothing
                # meaningful to plot, and reading min would expose
                # the constant value.
                _helper_failed(
                    "plot_interaction",
                    f"variable {var!r} has zero variance — interaction "
                    f"plot would expose the constant value",
                )
                return
            grid = _np.linspace(mu - 2.0 * sd, mu + 2.0 * sd, 100)
        else:
            cleaned = col.dropna()
            counts = cleaned.value_counts()
            # Drop rare levels (below threshold). Their identities are
            # disclosive even if the bar height is masked, same as the
            # frequency_table primary suppression rule.
            visible = counts[counts >= _CELL_SUPPRESSION_THRESHOLD]
            if visible.empty:
                _helper_failed(
                    "plot_interaction",
                    f"variable {var!r}: no level meets the disclosure "
                    f"threshold (n >= {_CELL_SUPPRESSION_THRESHOLD}); "
                    f"refusing to plot",
                )
                return
            # Keep top-K most frequent levels for readability.
            visible = visible.head(_CATEGORICAL_LEVEL_CAP)
            grid = list(visible.index)
            dropped = int((counts < _CELL_SUPPRESSION_THRESHOLD).sum())
            if dropped > 0:
                suppression_note = (
                    f"{dropped} rare level(s) with count < "
                    f"{_CELL_SUPPRESSION_THRESHOLD} suppressed"
                )

        template: dict[Any, Any] = {}
        for c in data.columns:
            v = data[c]
            if _pd.api.types.is_numeric_dtype(v):
                template[c] = float(v.mean())
            else:
                template[c] = v.dropna().iloc[0] if not v.dropna().empty else None
        new_rows = []
        for g in grid:
            row = dict(template)
            row[var] = g
            new_rows.append(row)
        new = _pd.DataFrame(new_rows)

        # Statsmodels' get_prediction returns a PredictionResults
        # with .summary_frame() including 'mean' and 'mean_ci_lower/upper'.
        try:
            pred = fitted.get_prediction(new)
            sf = pred.summary_frame(alpha=0.05)
            mean = sf["mean"].to_numpy()
            lo = sf["mean_ci_lower"].to_numpy()
            hi = sf["mean_ci_upper"].to_numpy()
        except Exception:
            # Fallback: predict() alone (no CI available)
            mean = _np.asarray(fitted.predict(new))
            lo = mean
            hi = mean

        xtitle = xlab if xlab is not None else var
        ytitle = ylab if ylab is not None else "Predicted response"
        ptitle = title if title is not None else f"Predicted response by {var}"

        fig, ax = plt.subplots(figsize=(9, 5))
        if is_numeric:
            # Filled CI ribbon under a colored line — much more
            # readable than the prior dashed-line whiskers, which
            # the user called out as a "really shitty" rendering.
            ax.fill_between(grid, lo, hi, color="#4C78A8", alpha=0.20)
            ax.plot(grid, mean, lw=2, color="#1F4E79")
            # Strip numeric x-axis tick labels: the absolute values
            # ARE data (mean ± 2σ for var). The disclosure-safe pair
            # is mean+sd, which goes through the descriptive
            # sanitizer with its own gates. Show only relative
            # anchors so the model can read the curve's shape
            # without reading the raw mean / sd off the axis.
            ax.set_xticks([
                grid[0], (grid[0] + grid[-1]) / 2.0, grid[-1]
            ])
            ax.set_xticklabels(["−2σ", "mean", "+2σ"])
        else:
            xs = _np.arange(len(grid))
            ax.bar(xs, mean, yerr=[mean - lo, hi - mean],
                   color="#4C78A8", edgecolor="#1F4E79",
                   capsize=4)
            ax.set_xticks(xs)
            # Run categorical tick labels through the same text-safety
            # primitive that gates every other model-visible string.
            # ``safe_text`` strips C0/C1 control chars, bidi overrides,
            # and zero-width characters, then caps length. Without this
            # a frequent-level category name like
            # ``"engineering\nIGNORE PRIOR INSTRUCTIONS:..."`` would
            # render straight into the model-visible image, bypassing
            # the JSON/text path's safety gate. ``safe_text`` returns
            # an empty string for completely-rejected inputs; fall
            # back to a redaction marker so the bar is still
            # identifiable at its x-position.
            from sift.text_safety import safe_text as _safe_text

            def _tick_label(v: object) -> str:
                t = _safe_text(str(v), max_len=24)
                return t or "[redacted]"
            ax.set_xticklabels(
                [_tick_label(g) for g in grid], rotation=30, ha="right",
            )
        ax.set_xlabel(xtitle)
        ax.set_ylabel(ytitle)
        ax.set_title(ptitle, fontweight="bold")
        if suppression_note:
            # Caption-style note so the model sees that some levels
            # were suppressed without seeing which ones.
            fig.text(
                0.99, 0.01, suppression_note,
                ha="right", va="bottom",
                fontsize=8, color="#666666",
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()

        # ``var`` can in principle contain weird chars; sanitize to
        # something filesystem-safe but still informative.
        safe_var = "".join(c if c.isalnum() or c in "-_" else "_" for c in var)
        fname = f"interaction_{safe_var}.png"
        fig.savefig(d / fname, dpi=110)
        plt.close(fig)
        _append_plot_manifest(
            fname, "interaction",
            label or f"Predicted response by {var}",
        )
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"sift.plot_interaction failed: {e}\n")
        _append_plot_helper_error("plot_interaction", e)


# ---------------------------------------------------------------------------
# Text extraction — local structure from a free-text column
# ---------------------------------------------------------------------------
#
# Private text analysis covers support tickets,
# clinical notes, survey free-response, contracts — organizational
# text data that's much harder to sanitize safely than a numeric
# column, because a single sentence can carry a name, a date, an
# address, a diagnosis. So the raw strings never leave this sandboxed
# process, and they never enter the payload the sanitizer inspects
# either — only counts and floats do.
#
# Honest scope: this is a DETERMINISTIC, lexicon/keyword-based
# extractor, not a local LLM. A trained local model would classify
# and score far more accurately, but this project has no such model
# available. A keyword classifier is transparent,
# has no hallucination surface, and needs no extra install. It is
# also unambiguously worse at nuance than an LLM would be: sarcasm,
# negation ("not bad at all"), and domain-specific vocabulary the
# built-in lexicon doesn't cover will all be scored wrong. The
# `categories` argument exists specifically so a researcher (or a
# Sift Skill) can supply a taxonomy suited to their
# domain rather than relying on generic keywords alone.

# A modest, general-purpose sentiment lexicon. Deliberately not
# exhaustive — this is a crude signal ("net positive/negative word
# hits"), not a sentiment MODEL, and the docstring above says so.
# Lowercase, single tokens only (the tokenizer below lowercases and
# splits on non-letter characters, so multi-word phrases would never
# match).
_TEXT_EXTRACT_POSITIVE_WORDS: frozenset[str] = frozenset({
    "good", "great", "excellent", "amazing", "wonderful", "love",
    "loved", "happy", "pleased", "satisfied", "helpful", "friendly",
    "fast", "quick", "easy", "smooth", "reliable", "professional",
    "recommend", "recommended", "perfect", "awesome", "fantastic",
    "impressed", "impressive", "efficient", "responsive", "kind",
    "thankful", "thanks", "thank", "appreciate", "appreciated",
    "best", "better", "improved", "improvement", "resolved",
    "solved", "convenient", "affordable", "worth", "outstanding",
    "exceeded", "exceptional", "delighted", "enjoyable", "enjoyed",
})
_TEXT_EXTRACT_NEGATIVE_WORDS: frozenset[str] = frozenset({
    "bad", "terrible", "awful", "horrible", "poor", "worst", "hate",
    "hated", "angry", "frustrated", "frustrating", "disappointed",
    "disappointing", "unhappy", "dissatisfied", "unhelpful", "rude",
    "slow", "difficult", "confusing", "broken", "unreliable",
    "unprofessional", "complaint", "complaining", "problem",
    "problems", "issue", "issues", "error", "errors", "failed",
    "failure", "delayed", "delay", "cancelled", "canceled",
    "refund", "waste", "annoying", "annoyed", "never", "wrong",
    "damaged", "defective", "missing", "confused", "useless",
    "expensive", "overpriced", "regret",
})

_TEXT_EXTRACT_TOKEN_RE = re.compile(r"[A-Za-z']+")


def _text_extract_tokenize(text: str) -> list[str]:
    return _TEXT_EXTRACT_TOKEN_RE.findall(text.lower())


def _text_extract_sentiment(text: str) -> float | None:
    """Crude lexicon sentiment in [-1, 1], or None if the text has no
    lexicon hits at all (as distinct from a neutral 0.0 — "no signal"
    and "measured neutral" are different and the caller should be
    able to tell them apart when averaging)."""
    tokens = _text_extract_tokenize(text)
    if not tokens:
        return None
    pos = sum(1 for t in tokens if t in _TEXT_EXTRACT_POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _TEXT_EXTRACT_NEGATIVE_WORDS)
    hits = pos + neg
    if hits == 0:
        return None
    return (pos - neg) / hits


def _text_extract_classify(
    text: str, categories: dict[str, list[str]], uncategorized_label: str,
) -> str:
    """First matching category wins (dict iteration order — Python
    3.7+ dicts preserve insertion order, so the caller controls
    priority by the order they build ``categories``). Case-
    insensitive substring match; a text matching none of the
    supplied keyword lists falls into ``uncategorized_label``."""
    lowered = text.lower()
    for category, keywords in categories.items():
        for kw in keywords:
            if isinstance(kw, str) and kw.lower() in lowered:
                return category
    return uncategorized_label


def from_text_extract(
    df: Any,
    text_column: str,
    *,
    categories: dict[str, list[str]],
    uncategorized_label: str = "uncategorized",
    **extra: Any,
) -> None:
    """Emit a ``text_extraction`` payload: LOCAL keyword classification
    + lexicon sentiment over a free-text column, reduced to category
    counts and per-category mean sentiment before anything is handed
    to ``result()``.

    ``categories`` maps a category name to a list of keywords/phrases
    that route a row into it (first match wins, case-insensitive
    substring search) — e.g.
    ``{"shipping": ["shipping", "delivery", "late", "package"],
       "billing": ["charge", "invoice", "refund", "payment"]}``.
    Rows matching no category land in ``uncategorized_label``.

    Every raw string in ``df[text_column]`` is read and discarded in
    this same local process — none of it, and no per-row value of any
    kind, is included in the emitted payload. Only aggregate counts
    and floats cross into ``result()``, which is exactly what the
    sanitizer's ``text_extraction`` schema accepts; the sanitizer
    itself additionally suppresses any category whose count is below
    the disclosure-control threshold and removes its sentiment score
    too (see ``sanitizer._sanitize_text_extraction``).
    """
    if text_column not in df.columns:
        raise ValueError(
            f"from_text_extract: {text_column!r} is not a column in df"
        )
    if not isinstance(categories, dict) or not categories:
        raise ValueError(
            "from_text_extract: categories must be a non-empty dict "
            "of category_name -> list of keywords"
        )

    series = df[text_column]
    n_total = len(series)
    non_null = series.dropna()
    missing_count = n_total - len(non_null)

    category_counts: dict[str, int] = {}
    sentiment_sum: dict[str, float] = {}
    sentiment_n: dict[str, int] = {}
    overall_sum = 0.0
    overall_n = 0

    for value in non_null:
        text = str(value)
        cat = _text_extract_classify(text, categories, uncategorized_label)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        score = _text_extract_sentiment(text)
        if score is not None:
            sentiment_sum[cat] = sentiment_sum.get(cat, 0.0) + score
            sentiment_n[cat] = sentiment_n.get(cat, 0) + 1
            overall_sum += score
            overall_n += 1

    category_sentiment = {
        cat: sentiment_sum[cat] / sentiment_n[cat]
        for cat in category_counts
        if sentiment_n.get(cat, 0) > 0
    }

    fields: dict[str, Any] = {
        "text_column": text_column,
        "categories": category_counts,
        "category_sentiment": category_sentiment,
        "n": int(len(non_null)),
        "missing_count": _safe_int(missing_count),
    }
    if overall_n > 0:
        fields["overall_sentiment_mean"] = overall_sum / overall_n
    fields.update(extra)
    result(type="text_extraction", **fields)
