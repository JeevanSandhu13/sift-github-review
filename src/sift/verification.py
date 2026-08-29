"""Sift — deterministic statistical verification of sanitized results.

An LLM asserting "this analysis looks valid" is an opinion. This
module is the code-level counterpart: a set of narrow, deterministic
checks computed from the *sanitized* payload (never from raw data),
each returning an explicit verdict. The output ships inline with
every ok result so both the model and the researcher see the same
verdicts, and the model is prompted to surface warnings rather than
bury them.

Honesty rules, enforced by construction:

- A check only appears when its inputs were actually present in the
  payload. There is no "assumed fine" default — absent inputs mean
  the check is simply not listed (callers can see what was and
  wasn't checked).
- Verdicts are ``pass`` / ``warn`` only. This module never blocks a
  result (the sanitizer owns hard rejects); it annotates.
- Thresholds are conventional and documented per check — sample-size
  rules of thumb, VIF 10, condition number 30, events-per-parameter
  10. They are heuristics, not guarantees, and the strings say what
  was found, not "verified correct".

Because every input is already sanitized (post-SDC), the verification
block is safe to cross the privacy boundary by construction — it is
a pure function of information the model was already given.
"""

from __future__ import annotations

import math
import re
from typing import Any

# Conventional thresholds. Deliberately few and deliberately standard;
# the point is deterministic, explainable flags, not a rulebook.
_MIN_COMFORTABLE_N = 30          # below: small-sample caution
_VIF_WARN = 10.0                 # Kutner et al. rule of thumb
_CONDITION_NUMBER_WARN = 30.0    # Belsley diagnostics
_R2_SUSPICIOUS = 0.999           # near-perfect fit → leakage/degeneracy
_OBS_PER_PARAM_WARN = 10         # events-per-variable heuristic
_FIRST_STAGE_F_WEAK = 10.0       # Staiger-Stock single-endogenous-regressor
                                  # rule of thumb; not a universal critical value
_BATCH_MULTIPLE_COMPARISONS = 5  # results per script before MC note

# --- shape-specific thresholds -------------------------------------
# Every constant below is a published convention, cited where it is
# used. They are heuristics for drawing attention, not decision rules:
# a warning means "a reader would ask about this", never "this result
# is wrong".
_PRE_TRENDS_P_MIN = 0.05     # DiD: pre-trend test rejecting → parallel
                             # trends is in question
_DID_MIN_COHORT = 10         # treated units per cohort worth reporting
_RDD_POLY_ORDER_MAX = 2      # Gelman & Imbens (2019): global high-order
                             # polynomials in RDD produce noisy,
                             # specification-driven estimates
_RDD_MIN_EFFECTIVE_N = 30    # per side, within bandwidth
_RDD_BANDWIDTH_ASYMMETRY = 3.0   # ratio between sides worth flagging
_KM_MIN_EVENTS = 10          # Peduzzi et al. (1995): survival power is
                             # driven by events, not subjects
_KM_MIN_AT_RISK = 10         # at-risk count behind a reported horizon
_DIAGNOSTIC_TEST_P_MAX = 0.05    # conventional alpha for the panel /
                                  # IV specification tests below —
                                  # hausman, panel breusch_pagan,
                                  # f_test_fe, wooldridge_ar1, hansen_j,
                                  # endogeneity (Wu-Hausman)

# Minimum-detectable-effect (MDE) constants for the t-test power note.
# This is a FORWARD-LOOKING precision calculation (Cohen, 1988) — the
# smallest true effect size this test's sample sizes could reliably
# detect at conventional alpha/power — not a "post-hoc power"
# computed from the observed effect. Post-hoc/observed power is a
# monotonic transform of the p-value and is widely regarded in the
# statistics literature as providing no information beyond the
# p-value itself (Hoenig and Heisey, 2001); this module deliberately
# does NOT compute that. MDE only needs the sample sizes, which are
# fixed in advance — exactly the property that makes it a legitimate
# thing to report regardless of what the test found.
_MDE_Z_ALPHA = 1.959964   # z_(0.975): two-sided alpha=0.05
_MDE_Z_POWER = 0.841621   # z_(0.80): 80% target power
_MDE_LARGE_D = 0.5        # Cohen's small/medium boundary — an MDE at
                          # or above this means only medium-or-larger
                          # effects were reliably detectable

# Extreme-coefficient-t-statistic threshold for the target-leakage /
# separation heuristic. |t| this large essentially never occurs with
# real, non-degenerate data at any realistic sample size — it is the
# classic signature of target leakage (a predictor that is a near-
# deterministic function of the outcome) or, for logistic/probit
# models, perfect or quasi-perfect separation.
_EXTREME_T_STAT = 50.0

# Generic modelling tokens excluded from the leakage-naming heuristic
# below — without this, virtually every regression would falsely
# flag on a shared "id"/"flag"/"score"/"count" token. Only a
# remaining, more SPECIFIC shared token (e.g. "fraud", "churn",
# "default") is treated as suspicious.
_LEAKAGE_NAMING_STOPWORDS: frozenset[str] = frozenset((
    "id", "flag", "value", "amount", "count", "total", "score",
    "rate", "num", "number", "is", "has", "the", "a", "an", "of",
    "in", "on", "at", "to", "for", "and", "or", "code", "type",
    "status", "level", "index", "key", "date", "time", "year",
    "log", "sqrt", "std", "avg", "mean", "pct", "percent",
))


def _name_tokens_for_leakage(name: Any) -> set[str]:
    """Lowercased, underscore/hyphen/space-split tokens of ``name``,
    with generic modelling words and short tokens (len < 3) dropped.
    Purely a heuristic string comparison over already-visible field
    names (``response_variable`` / ``predictor_variables`` are plain
    strings already in every sanitized regression payload) — no new
    data crosses any boundary to compute this.
    """
    if not isinstance(name, str) or not name:
        return set()
    raw = re.split(r"[^a-zA-Z0-9]+", name.lower())
    return {t for t in raw if len(t) >= 3 and t not in _LEAKAGE_NAMING_STOPWORDS}
_SILHOUETTE_WEAK = 0.25      # Kaufman & Rousseeuw (1990): below this,
                             # no substantial cluster structure found
_KMO_MIN = 0.60              # Kaiser (1974): below 0.6 is "mediocre to
                             # unacceptable" sampling adequacy for FA
_BARTLETT_P_MAX = 0.05       # Bartlett sphericity must reject for
                             # factoring to be justified
_RMSEA_POOR = 0.08           # Browne & Cudeck (1993): >0.08 is a poor
                             # approximate fit
_TLI_POOR = 0.90             # conventional acceptable-fit floor
_FACTOR_OBS_PER_VAR = 10     # classic 10:1 observations-per-variable
_MIN_CLUSTER_SIZE = 10       # clusters smaller than this are fragile


def _num(payload: dict[str, Any], key: str) -> float | None:
    """Return a finite numeric field, else None.

    Three values must be rejected:

    - **Booleans.** ``isinstance(True, int)`` is True in Python, so an
      unguarded read would treat a flag as the number 1.
    - **Strings.** Sanitized payloads carry suppression markers like
      ``"<10"`` in numeric slots; those are not quantities.
    - **NaN and infinity.** These reach payloads legitimately —
      ``sdc.round_to_sigfigs`` passes non-finite values through
      unchanged, so a degenerate fit (perfect separation, zero
      variance, empty subgroup) can put NaN in a field a check reads.
      Formatting one with ``int()`` raises, which would turn a
      already-degenerate analysis into a crashed tool call. A check
      that cannot be computed is simply not reported.
    """
    val = payload.get(key)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    try:
        fval = float(val)
    except (OverflowError, ValueError):
        # Python ints are unbounded, so ``float()`` on a sufficiently
        # large one raises rather than returning inf. A value that big
        # is not a statistic; treat it as uncheckable.
        return None
    return fval if math.isfinite(fval) else None


def _finite(val: Any) -> float | None:
    """Return a finite float for a raw value, else None.

    The value-level counterpart of :func:`_num` (which reads by key),
    used where a check inspects the entries of a payload dict. Both
    guard the same three hazards: booleans, non-numerics, and values
    that are non-finite or too large for ``float()`` to represent.
    """
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    try:
        fval = float(val)
    except (OverflowError, ValueError):
        return None
    return fval if math.isfinite(fval) else None


def _finite_values(mapping: Any) -> list[float]:
    """Finite numeric values from a payload dict, booleans excluded.

    ``isinstance(x, (int, float))`` accepts NaN and infinity, which
    then raise on ``int()`` during message formatting. Sanitized
    payloads really can contain them (see ``_num``), so every
    aggregate over a payload dict routes through here.
    """
    if not isinstance(mapping, dict):
        return []
    out: list[float] = []
    for val in mapping.values():
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            continue
        try:
            fval = float(val)
        except (OverflowError, ValueError):
            continue
        if math.isfinite(fval):
            out.append(fval)
    return out


def _mapping_field(payload: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a mapping-valued payload field without relying on repeated gets.

    Keeping this check in one place makes malformed optional fields consistently
    degrade to "not checkable" instead of leaking ``None`` into verification
    arithmetic or attribute access.
    """
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _check(checks: list[dict[str, str]], check_id: str,
           status: str, detail: str) -> None:
    checks.append({"id": check_id, "status": status, "detail": detail})


def verify_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a verification block for a sanitized payload, or None.

    ``None`` means "nothing checkable for this payload type" — the
    caller should omit the field entirely rather than fabricate an
    empty verdict.
    """
    if not isinstance(payload, dict):
        return None
    checks: list[dict[str, str]] = []

    n = _num(payload, "n")
    if n is not None:
        if n < _MIN_COMFORTABLE_N:
            _check(checks, "sample_size", "warn",
                   f"n={int(n)} is small; estimates and intervals are "
                   "fragile at this size")
        else:
            _check(checks, "sample_size", "pass",
                   f"n={int(n)} adequate for the reported statistics")

    ptype = payload.get("type")

    dispatch = {
        # Canonical regression-bucket type name (see sanitizer.py's
        # ``_REGRESSION_TYPE_CANONICAL``/``_REGRESSION_TYPE_LEGACY``)
        # plus its legacy alias, both routed to the same verifier.
        # Every regression result the current R / Python / Stata
        # helpers emit is stamped with the CANONICAL name -- without
        # this key, every current regression result silently got no
        # verification checks at all (heteroskedasticity, residuals,
        # convergence -- the entire point of this module), and only
        # results stored before the rename (using the legacy name)
        # were ever actually verified. See the architecture audit
        # finding this closes, and ``result_render.py``'s and
        # ``tools.py``'s dispatch tables for the same two-alias
        # pattern already applied correctly elsewhere.
        "coefficient_table_with_fit_stats": lambda: _verify_regression(payload, checks, n),
        "linear_regression": lambda: _verify_regression(payload, checks, n),
        "t_test": lambda: _verify_t_test(payload, checks),
        "frequency_table": lambda: _verify_counts(payload, checks),
        "crosstab": lambda: _verify_counts(payload, checks),
        "magnitude_table": lambda: _verify_magnitude(payload, checks),
        "did_event_study": lambda: _verify_did(payload, checks),
        "rdd": lambda: _verify_rdd(payload, checks),
        "kaplan_meier": lambda: _verify_kaplan_meier(payload, checks),
        "cluster_analysis": lambda: _verify_cluster(payload, checks),
        "factor_decomposition": lambda: _verify_factor(payload, checks),
        "marginal_effects": lambda: _verify_marginal_effects(payload, checks),
        # correlation_matrix: the sanitizer's min-N gate is the check;
        # the shared sample-size check above already covers ``n``.
        "correlation_matrix": lambda: None,
        "descriptive": lambda: _verify_descriptive(payload, checks),
        "text_extraction": lambda: _verify_text_extraction(payload, checks),
        "method_result": lambda: _verify_method_result(payload, checks),
    }
    handler = dispatch.get(str(ptype))
    if handler is not None:
        handler()

    if not checks:
        return None
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    block = {
        "checks": checks,
        "warnings": n_warn,
        "confidence": _confidence_level(checks),
        "note": (
            "deterministic checks computed from the sanitized result; "
            "absence of a check means its inputs were not present"
        ),
    }
    causality = _causality_label(payload)
    if causality is not None:
        block["causality"] = causality
    return block


# ---------------------------------------------------------------------------
# Causality label: what kind of claim this result's DESIGN supports,
# independent of whether the estimate itself looks clean.
# ---------------------------------------------------------------------------
#
# A perfectly well-specified OLS regression with zero warnings can
# still only support an associational claim — the analysis TYPE, not
# its diagnostics, determines whether causal language is warranted.
# This is keyed purely on ``payload.get("type")`` (plus, for the
# regression bucket, whether IV fields are present) — no new fields,
# no new sanitizer surface. Every label carries a caveat because the
# point is exactly that a result's fluency in the model's prose
# should not imply more than the design supports.

_CAUSALITY_ASSOCIATIONAL_CAVEAT = (
    "this is an associational analysis; it does not by itself "
    "establish that a predictor causes the outcome. Confounding, "
    "reverse causality, and omitted variables can all produce this "
    "pattern without a causal relationship."
)
_CAUSALITY_DESCRIPTIVE_CAVEAT = (
    "this is a descriptive summary of the data as observed; it makes "
    "no claim about what causes what."
)


def _causality_label(payload: dict[str, Any]) -> dict[str, str] | None:
    ptype = str(payload.get("type") or "")

    if ptype == "method_result":
        family = str(payload.get("method_family") or "")
        rule = str(payload.get("claim_rule") or "")
        if payload.get("method_id") == "causal_sensitivity":
            return {
                "label": "sensitivity_only",
                "design": "omitted_variable_sensitivity",
                "caveat": (
                    "this quantifies how strong an omitted confounder would need to be "
                    "to alter the reported association; it does not identify a causal effect"
                ),
            }
        if family == "causal":
            return {"label": "design_conditional_causal",
                    "design": str(payload.get("method_id") or ""),
                    "caveat": rule}
        if family == "predictive":
            return {"label": "predictive", "caveat": rule}
        if family in {"descriptive", "measurement", "domain"}:
            return {"label": "descriptive", "caveat": rule}
        return {"label": "associational", "caveat": rule}

    if ptype in ("linear_regression", "coefficient_table_with_fit_stats"):
        # IV / 2SLS fields present -> quasi-experimental, scoped to
        # the local average treatment effect the instrument
        # identifies; otherwise plain associational regression.
        if _num(payload, "first_stage_f") is not None:
            return {
                "label": "quasi_experimental",
                "design": "instrumental_variables",
                "caveat": (
                    "causal interpretation here relies on the "
                    "instrument(s) being valid (excluded from the "
                    "outcome equation except through the endogenous "
                    "regressor) and identifies a LOCAL average "
                    "treatment effect for units whose treatment status "
                    "is moved by the instrument — not necessarily the "
                    "average effect across the whole sample"
                ),
            }
        return {"label": "associational",
                "caveat": _CAUSALITY_ASSOCIATIONAL_CAVEAT}

    if ptype == "marginal_effects":
        # Marginal effects inherit the causal status of whatever base
        # model produced them; without visibility into that model
        # here, the conservative (associational) label applies.
        return {"label": "associational",
                "caveat": _CAUSALITY_ASSOCIATIONAL_CAVEAT}

    if ptype == "did_event_study":
        return {
            "label": "quasi_experimental",
            "design": "difference_in_differences",
            "caveat": (
                "causal interpretation relies on the PARALLEL TRENDS "
                "assumption: absent treatment, the treated and control "
                "groups would have followed the same trend, alongside no "
                "anticipation/interference and a valid comparison group. "
                "A non-rejecting pre-trends test is supportive but cannot "
                "prove parallel counterfactual trends"
            ),
        }

    if ptype == "rdd":
        return {
            "label": "quasi_experimental",
            "design": "regression_discontinuity",
            "caveat": (
                "causal interpretation is LOCAL to observations near "
                "the cutoff (a local average treatment effect at the "
                "discontinuity, not an average effect across the "
                "sample) and relies on continuity of potential outcomes "
                "and no precise manipulation of the running variable at "
                "the cutoff"
            ),
        }

    if ptype == "kaplan_meier":
        return {
            "label": "descriptive",
            "caveat": (
                "this describes observed survival patterns; group "
                "differences may reflect confounding rather than a "
                "causal effect of the modelled exposure or treatment "
                "unless the groups were randomly assigned"
            ),
        }

    if ptype == "correlation_matrix":
        return {"label": "associational",
                "caveat": (
                    "correlation does not imply causation — these "
                    "coefficients describe association only"
                )}

    if ptype in ("cluster_analysis", "factor_decomposition",
                 "frequency_table", "crosstab", "magnitude_table",
                 "descriptive"):
        return {"label": "descriptive",
                "caveat": _CAUSALITY_DESCRIPTIVE_CAVEAT}

    if ptype == "t_test":
        return {"label": "associational",
                "caveat": _CAUSALITY_ASSOCIATIONAL_CAVEAT}

    return None


def _verify_method_result(
    payload: dict[str, Any], checks: list[dict[str, str]],
) -> None:
    diagnostics = payload.get("diagnostics")
    if isinstance(diagnostics, dict):
        concerning = sorted(
            key for key, value in diagnostics.items()
            if value in {"warn", "fail", False}
        )
        if concerning:
            _check(checks, "method_diagnostics", "warn",
                   f"{len(concerning)} required diagnostic(s) raised concern: "
                   f"{', '.join(concerning)}")
        else:
            _check(checks, "method_diagnostics", "pass",
                   "all registry-required diagnostics were reported without a warning status")
    method_id = payload.get("method_id")
    p_values = payload.get("p_values")
    estimates = payload.get("estimates")
    lower = payload.get("ci_lower")
    upper = payload.get("ci_upper")
    if method_id == "descriptive_confidence_interval":
        complete = (
            isinstance(estimates, dict) and bool(estimates)
            and isinstance(lower, dict) and isinstance(upper, dict)
            and set(estimates) == set(lower) == set(upper)
        )
        _check(
            checks, "descriptive_interval", "pass" if complete else "warn",
            "descriptive estimate and confidence-interval endpoints are "
            "reported on matching quantity keys",
        )
    if method_id in {
        "nonparametric_test", "proportion_test", "anova", "ancova",
        "repeated_measures_test",
    }:
        _check(
            checks, "omnibus_inference",
            "pass" if isinstance(p_values, dict) and bool(p_values) else "warn",
            "the reference-method result reports at least one aggregate p-value",
        )
    if method_id == "repeated_measures_test":
        subjects = _num(payload, "subjects")
        records = _num(payload, "records")
        structure_ok = (
            subjects is not None and subjects > 1
            and records is not None and records >= subjects
        )
        _check(
            checks, "repeated_measure_structure",
            "pass" if structure_ok else "warn",
            "repeated-measures output distinguishes subject and record counts",
        )
    if method_id == "missingness_pattern":
        metrics = _mapping_field(payload, "metrics")
        complete_rate = _finite(metrics.get("complete_case_rate"))
        warning = diagnostics.get("complete_case_warning") if isinstance(diagnostics, dict) else None
        _check(
            checks, "complete_case_support",
            "warn" if warning == "warn" else "pass",
            (
                f"complete-case analysis would retain {100 * complete_rate:.1f}% of rows"
                if complete_rate is not None
                else "complete-case retention was not quantified"
            ),
        )
        _check(
            checks, "missing_mechanism_identification", "warn",
            "observed missingness patterns do not identify MCAR, MAR, or MNAR",
        )
    if method_id == "single_imputation":
        _check(
            checks, "single_imputation_scope", "warn",
            "single imputation is a declared preprocessing step and does not propagate missing-data uncertainty",
        )
    if method_id == "multiple_imputation":
        metrics = _mapping_field(payload, "metrics")
        fmi = _finite(metrics.get("max_fraction_missing_information"))
        drift = _finite(metrics.get("imputed_mean_trace_drift"))
        _check(
            checks, "rubin_pooling", "pass",
            f"Rubin pooling combined {payload.get('imputations')} imputed analyses with a recorded seed",
        )
        _check(
            checks, "imputation_trace_stability",
            "pass" if drift is not None and drift <= 0.5 else "warn",
            (
                f"maximum standardized split drift of imputed-value means was {drift:.3g}; this is a trace diagnostic, not proof of convergence"
                if drift is not None else "imputation trace stability was not quantified"
            ),
        )
        _check(
            checks, "missing_information", "warn" if fmi is not None and fmi >= 0.5 else "pass",
            (
                f"maximum fraction of missing information was {fmi:.1%}"
                if fmi is not None else "fraction of missing information was not quantified"
            ),
        )
    if method_id == "mnar_sensitivity":
        metrics = _mapping_field(payload, "metrics")
        stable = diagnostics.get("conclusion_stability") if isinstance(diagnostics, dict) else False
        _check(
            checks, "mnar_sensitivity_grid", "pass" if stable else "warn",
            (
                f"pooled conclusion classification was {'stable' if stable else 'not stable'} "
                f"across {int(metrics.get('scenario_count', 0))} declared delta scenarios"
            ),
        )
        _check(
            checks, "mnar_parameter_identification", "warn",
            "the delta range requires external scientific justification; observed data do not identify it",
        )
    if method_id in {"growth_curve", "gee", "panel_fixed_effects", "panel_random_effects"}:
        clusters = _num(payload, "clusters")
        records = _num(payload, "records")
        n = _num(payload, "n")
        structure_ok = (
            clusters is not None and clusters >= 2
            and records is not None and records == n
        )
        _check(
            checks, "longitudinal_structure",
            "pass" if structure_ok else "warn",
            "longitudinal output binds fitted records to repeated-measure clusters",
        )
    if method_id == "panel_random_effects" and isinstance(diagnostics, dict):
        hausman = _finite(diagnostics.get("hausman"))
        _check(
            checks, "panel_random_effects_hausman",
            "pass" if hausman is not None and hausman >= 0.05 else "warn",
            "Hausman comparison does not reject random-effects consistency"
            if hausman is not None and hausman >= 0.05
            else "Hausman comparison raises concern for random-effects consistency",
        )
    if method_id == "panel_fixed_effects" and isinstance(diagnostics, dict):
        within = _finite(diagnostics.get("within_variation"))
        fixed_p = _finite(diagnostics.get("fixed_effect_test"))
        clustered = diagnostics.get("clustered_uncertainty") == "pass"
        _check(
            checks, "panel_within_identification",
            "pass" if within is not None and within > 0 else "warn",
            "coefficients are identified from within-entity predictor variation in an exactly balanced panel",
        )
        _check(
            checks, "panel_clustered_inference", "pass" if clustered else "warn",
            "standard errors and intervals use entity-clustered covariance",
        )
        _check(
            checks, "panel_fixed_effect_relevance",
            "pass" if fixed_p is not None and fixed_p < 0.05 else "warn",
            "the pooled-versus-entity-effects comparison is diagnostic and does not make the associations causal",
        )
    if method_id == "gee" and isinstance(diagnostics, dict):
        sensitivity = _finite(
            diagnostics.get("working_correlation_sensitivity")
        )
        _check(
            checks, "gee_working_correlation_sensitivity",
            "pass" if sensitivity is not None and sensitivity >= 0 else "warn",
            (
                "working-correlation refit quantified a maximum absolute "
                f"coefficient change of {sensitivity:.4g}"
                if sensitivity is not None and sensitivity >= 0
                else "working-correlation sensitivity was not quantified by a refit"
            ),
        )
    if method_id == "competing_risks" and isinstance(estimates, dict):
        total_cif = sum(
            value for value in (_finite(item) for item in estimates.values())
            if value is not None
        )
        valid_cif = bool(estimates) and 0.0 <= total_cif <= 1.001
        _check(
            checks, "competing_risk_probability_mass",
            "pass" if valid_cif else "warn",
            "final cause-specific cumulative incidences occupy valid probability mass",
        )
    if method_id in {"recurrent_events", "time_varying_survival"}:
        subjects = _num(payload, "subjects")
        events = _num(payload, "events")
        records = _num(payload, "records")
        structure_ok = (
            subjects is not None and subjects >= 2
            and events is not None and events > 0
            and records is not None and records > subjects
            and payload.get("clusters") == payload.get("subjects")
            and payload.get("uncertainty_type") == "cluster_robust"
        )
        _check(
            checks, "counting_process_structure",
            "pass" if structure_ok else "warn",
            "counting-process output distinguishes subjects, events, and intervals with subject-clustered uncertainty",
        )
    if method_id == "multiple_testing_correction":
        correction = str(payload.get("multiple_testing") or "none")
        adjusted = p_values if isinstance(p_values, dict) else {}
        raw = estimates if isinstance(estimates, dict) else {}
        same_family = bool(raw) and set(raw) == set(adjusted)
        _check(
            checks, "multiplicity_family",
            "pass" if correction != "none" and same_family else "warn",
            f"{correction} correction reports raw and adjusted p-values for "
            f"the same {len(raw)} hypotheses",
        )
        expected = _adjust_p_values(raw, correction) if same_family else None
        # Sanitizer precision clamping occurs before verification.  A 0.002
        # absolute tolerance accepts that documented rounding at the smallest
        # supported N while still detecting unadjusted or wrongly adjusted
        # values in ordinary analyses.
        correct = expected is not None and all(
            abs(float(adjusted[key]) - expected[key]) <= 0.002
            for key in raw
        )
        _check(
            checks, "multiplicity_recalculation", "pass" if correct else "warn",
            "adjusted p-values were independently recalculated from the "
            "reported raw hypothesis family",
        )
    causal_ids = {
        "matching", "propensity_weighting", "synthetic_control",
        "treatment_effect_heterogeneity", "causal_sensitivity",
        "difference_in_differences",
    }
    if method_id in causal_ids:
        declared = bool(payload.get("estimand") and payload.get("design"))
        _check(
            checks, "causal_design_contract", "pass" if declared else "warn",
            "the result declares its estimand and design; causal interpretation "
            "remains conditional on the reported identifying assumptions",
        )
    metrics = _mapping_field(payload, "metrics")
    if method_id in {"matching", "propensity_weighting"}:
        overlap = _finite(metrics.get("overlap_fraction"))
        balance = _finite(metrics.get("max_abs_smd_after"))
        _check(
            checks, "propensity_overlap", "pass" if overlap is not None and overlap >= 0.8 else "warn",
            f"common-support overlap fraction is {overlap if overlap is not None else 'unreported'}",
        )
        _check(
            checks, "post_design_balance", "pass" if balance is not None and balance <= 0.1 else "warn",
            f"maximum absolute post-design standardized mean difference is "
            f"{balance if balance is not None else 'unreported'} (target <=0.1)",
        )
    if method_id in {
        "matching", "propensity_weighting", "synthetic_control",
        "treatment_effect_heterogeneity",
    }:
        _check(
            checks, "effect_uncertainty", "warn",
            "the typed helper reports a point estimate and design diagnostics "
            "only; it does not claim an analytic standard error or confidence interval",
        )
    if method_id == "synthetic_control":
        pre_rmse = _finite(metrics.get("pre_rmse"))
        placebo_p = _finite(metrics.get("placebo_p_value"))
        concentration = _finite(metrics.get("max_donor_weight"))
        _check(
            checks, "synthetic_pre_fit", "pass" if pre_rmse is not None else "warn",
            "pre-treatment RMSPE is reported; substantive adequacy is scale-dependent",
        )
        _check(
            checks, "synthetic_placebos",
            "pass" if placebo_p is not None and 0 <= placebo_p <= 1 else "warn",
            "the treated-unit post/pre RMSPE ratio is ranked against donor placebos",
        )
        _check(
            checks, "donor_concentration",
            "pass" if concentration is not None and concentration <= 0.8 else "warn",
            f"largest donor weight is {concentration if concentration is not None else 'unreported'}",
        )
    if method_id == "treatment_effect_heterogeneity":
        calibration = _finite(metrics.get("calibration_correlation"))
        _check(
            checks, "heterogeneity_calibration",
            "pass" if calibration is not None and calibration > 0 else "warn",
            "honest-split CATE predictions are compared with transformed held-out outcomes",
        )
    if method_id == "causal_sensitivity":
        rv = _finite(metrics.get("robustness_value_zero"))
        _check(
            checks, "omitted_variable_robustness",
            "pass" if rv is not None and rv >= 0.1 else "warn",
            f"partial-R2 robustness value for reducing the estimate to zero is "
            f"{rv if rv is not None else 'unreported'}; this is sensitivity, not identification",
        )
    if method_id == "difference_in_differences":
        raw_did = _finite(metrics.get("raw_did"))
        att = _finite(estimates.get("att")) if isinstance(estimates, dict) else None
        exact = raw_did is not None and att is not None and abs(raw_did - att) <= 0.01
        _check(
            checks, "did_two_by_two_contrast", "pass" if exact else "warn",
            "the interaction estimate was reconciled to the four-cell two-period difference-in-differences contrast",
        )
        _check(
            checks, "did_parallel_trends_boundary", "warn",
            "one pre-period cannot test parallel trends; causal interpretation requires external design justification",
        )
        _check(
            checks, "did_clustered_inference",
            "pass" if payload.get("uncertainty_type") == "cluster_robust" else "warn",
            "uncertainty is clustered by panel entity across the two repeated observations",
        )
    if method_id == "stationarity_diagnostic":
        adf_p = _finite(metrics.get("adf_p_value"))
        kpss_p = _finite(metrics.get("kpss_p_value"))
        adf_stat = _finite(metrics.get("adf_statistic"))
        adf_critical = _finite(metrics.get("adf_critical_05"))
        kpss_stat = _finite(metrics.get("kpss_statistic"))
        kpss_critical = _finite(metrics.get("kpss_critical_05"))
        agrees = (
            adf_p is not None and adf_p < 0.05 and (kpss_p is None or kpss_p > 0.05)
        ) or (
            adf_stat is not None and adf_critical is not None and adf_stat < adf_critical
            and kpss_stat is not None and kpss_critical is not None
            and kpss_stat < kpss_critical
        )
        _check(
            checks, "stationarity_evidence", "pass" if agrees else "warn",
            "ADF rejects a unit root and, when available, KPSS does not reject stationarity; "
            "structural breaks can invalidate either conclusion",
        )
    if method_id in {"arima", "exponential_smoothing", "forecast_backtest"}:
        coverage = _finite(metrics.get("prediction_interval_coverage"))
        nominal = _finite(metrics.get("nominal_coverage"))
        test_n = _num(payload, "test_observations")
        adequate = (
            coverage is not None and nominal is not None and test_n is not None
            and test_n >= 5 and abs(coverage - nominal) <= 0.2
        )
        _check(
            checks, "empirical_interval_coverage", "pass" if adequate else "warn",
            f"chronological evaluation observed interval coverage {coverage} against "
            f"nominal {nominal} over {int(test_n) if test_n is not None else 'unknown'} forecasts "
            f"using {payload.get('interval_method') or 'an undeclared method'}",
        )
        train_n = _num(payload, "training_observations")
        total_n = _num(payload, "n")
        split = payload.get("evaluation_split")
        leakage_free = (
            train_n is not None and test_n is not None and total_n is not None
            and int(train_n + test_n) == int(total_n)
            and split in {"held_out", "rolling_origin"}
        )
        _check(checks, "temporal_leakage", "pass" if leakage_free else "warn",
               "declared chronological training and test counts partition the series")
    if method_id == "forecast_backtest":
        rmse = _finite(metrics.get("rmse")); baseline = _finite(metrics.get("baseline_rmse"))
        _check(
            checks, "forecast_baseline",
            "pass" if rmse is not None and baseline is not None and rmse <= baseline else "warn",
            f"forecast RMSE {rmse} is compared with chronological naive baseline RMSE {baseline}",
        )
    if method_id == "interrupted_time_series":
        pretrend_p = _finite(metrics.get("pretrend_stability_p_value"))
        _check(
            checks, "pretrend_stability", "pass" if pretrend_p is not None and pretrend_p > .05 else "warn",
            f"a pre-intervention midpoint-break diagnostic reports p={pretrend_p}; "
            "failure to detect a break is not proof of a correct functional form",
        )
        _check(
            checks, "interrupted_series_identification", "warn",
            "segmented AR(1) regression estimates level and slope changes; causal interpretation "
            "still requires no concurrent intervention or time-varying confounding",
        )
    if method_id == "ordinal_regression":
        metrics = _mapping_field(payload, "metrics")
        threshold_rows = []
        if isinstance(estimates, dict):
            for key, value in estimates.items():
                if not isinstance(key, str) or not key.startswith("threshold_"):
                    continue
                try:
                    index = int(key.removeprefix("threshold_"))
                except ValueError:
                    continue
                if _finite(value) is not None:
                    threshold_rows.append((index, value))
        threshold_values = [value for _index, value in sorted(threshold_rows)]
        categories = _finite(metrics.get("category_count"))
        ordered = bool(threshold_values) and all(
            float(right) > float(left)
            for left, right in zip(threshold_values, threshold_values[1:])
        )
        coherent = categories is not None and len(threshold_values) == int(categories) - 1
        _check(
            checks, "ordinal_thresholds", "pass" if ordered and coherent else "warn",
            "ordered cut points were independently checked against the reported category count",
        )
    if method_id == "multinomial_regression":
        metrics = _mapping_field(payload, "metrics")
        categories = _finite(metrics.get("category_count"))
        equations = _finite(metrics.get("equation_count"))
        prefixes = {
            key.split("#", 1)[0] for key in (estimates or {})
            if isinstance(key, str) and "#" in key
        } if isinstance(estimates, dict) else set()
        coherent = (
            categories is not None and equations is not None
            and int(equations) == int(categories) - 1
            and len(prefixes) == int(equations)
        )
        _check(
            checks, "multinomial_equations", "pass" if coherent else "warn",
            "non-reference equations were independently matched to the outcome category count",
        )
    if method_id == "zero_inflated_model":
        metrics = _mapping_field(payload, "metrics")
        zero_fraction = _finite(metrics.get("zero_fraction"))
        ratio = _finite(metrics.get("variance_mean_ratio"))
        params = estimates if isinstance(estimates, dict) else {}
        coherent = (
            zero_fraction is not None and 0 <= zero_fraction <= 1
            and ratio is not None and ratio >= 0
            and any(key.startswith("inflate_") for key in params)
            and any(not key.startswith("inflate_") for key in params)
        )
        _check(
            checks, "zero_inflated_components", "pass" if coherent else "warn",
            "count and structural-zero components and count aggregates were independently checked",
        )
    if method_id == "spline_regression":
        metrics = _mapping_field(payload, "metrics")
        basis_df = _finite(metrics.get("basis_df"))
        basis_parameter_count = _finite(metrics.get("basis_parameter_count"))
        parameter_count = _finite(metrics.get("parameter_count"))
        coherent = (
            basis_df is not None and basis_df >= 2
            and basis_parameter_count == basis_df
            and parameter_count is not None
            and isinstance(estimates, dict)
            and int(parameter_count) == len(estimates)
        )
        _check(
            checks, "nonlinear_basis", "pass" if coherent else "warn",
            "basis degrees of freedom and emitted parameter count were independently reconciled",
        )
    if method_id in {"survey_mean", "survey_proportion", "survey_regression"}:
        metrics = _mapping_field(payload, "metrics")
        ses = _mapping_field(payload, "standard_errors")
        method = payload.get("variance_method")
        replicate_count = _finite(metrics.get("replicate_count"))
        strata_count = _finite(metrics.get("strata_count"))
        psu_count = _finite(metrics.get("psu_count"))
        design_df = _finite(metrics.get("design_df"))
        stage_count = _finite(metrics.get("stage_count"))
        secondary_psus = _finite(metrics.get("secondary_psu_count"))
        lonely = _finite(metrics.get("lonely_strata_count"))
        lonely_certainty = _finite(metrics.get("lonely_certainty_count"))
        lonely_adjusted = _finite(metrics.get("lonely_adjusted_count"))
        design_ok = (
            design_df is not None and design_df >= 1
            and replicate_count is not None and strata_count is not None
            and psu_count is not None and stage_count is not None
            and secondary_psus is not None
            and lonely is not None and lonely_certainty is not None
            and lonely_adjusted is not None
            and lonely_certainty + lonely_adjusted == lonely
            and (
                (method == "taylor_linearization" and replicate_count == 0
                 and strata_count >= 1 and psu_count >= strata_count
                 and stage_count in {1, 2}
                 and ((stage_count == 1 and secondary_psus == 0)
                      or (stage_count == 2 and secondary_psus >= psu_count)))
                or (method in {"brr", "fay", "jackknife", "bootstrap"}
                    and replicate_count >= 2 and strata_count == psu_count == 0
                    and stage_count == secondary_psus == 0)
            )
        )
        _check(
            checks, "survey_design_structure", "pass" if design_ok else "warn",
            "variance method, design degrees of freedom, strata/PSU counts, and replicate count were reconciled",
        )
        if method in {"brr", "fay", "jackknife", "bootstrap"}:
            mse = _finite(metrics.get("replicate_mse"))
            scale = _finite(metrics.get("replicate_scale"))
            rscale_min = _finite(metrics.get("replicate_rscale_min"))
            rscale_max = _finite(metrics.get("replicate_rscale_max"))
            replicate_contract = (
                mse in {0.0, 1.0} and scale is not None and scale > 0
                and rscale_min is not None and rscale_max is not None
                and 0 < rscale_min <= rscale_max
            )
            _check(
                checks, "replicate_variance_contract",
                "pass" if replicate_contract else "warn",
                "replicate centering mode, global scale, and per-replicate scale range were reported",
            )
        variance_ok = bool(ses)
        for key, se in ses.items():
            variance_key = "variance" if method_id != "survey_regression" else f"variance#{key}"
            variance = _finite(metrics.get(variance_key))
            se_value = _finite(se)
            if (variance is None or se_value is None
                    or abs(variance - se_value ** 2) > max(1e-8, abs(variance) * 0.01)):
                variance_ok = False
                break
        _check(
            checks, "survey_variance_identity", "pass" if variance_ok else "warn",
            "reported standard errors were independently squared and matched to design variances",
        )
        effective = _finite(metrics.get("effective_sample_size"))
        raw_n = _finite(payload.get("n"))
        _check(
            checks, "survey_effective_sample", "pass" if (
                effective is not None and raw_n is not None and 1 <= effective <= raw_n
            ) else "warn",
            "Kish effective sample size was bounded by the observed sample count",
        )
    if method_id == "reliability":
        metrics = _mapping_field(payload, "metrics")
        estimates_map = estimates if isinstance(estimates, dict) else {}
        interval_ok = (
            set(estimates_map) == {"alpha", "omega_total"}
            and isinstance(lower, dict) and isinstance(upper, dict)
            and set(lower) == set(upper) == set(estimates_map)
            and all(
                (value := _finite(estimates_map.get(key))) is not None
                and (lo := _finite(lower.get(key))) is not None
                and (hi := _finite(upper.get(key))) is not None
                and 0 <= lo <= value <= hi <= 1
                for key in estimates_map
            )
        )
        _check(
            checks, "reliability_interval", "pass" if interval_ok else "warn",
            "alpha and omega total were bounded and matched to bootstrap confidence intervals",
        )
        items = _finite(metrics.get("item_count"))
        reversed_items = _finite(metrics.get("reversed_item_count"))
        minimum_correlation = _finite(metrics.get("min_item_rest_correlation"))
        repetitions = _finite(metrics.get("bootstrap_replicates"))
        successes = _finite(metrics.get("bootstrap_success_count"))
        direction_ok = (
            items is not None and items >= 3
            and reversed_items is not None and 0 <= reversed_items <= items
            and minimum_correlation is not None and 0 <= minimum_correlation <= 1
            and repetitions is not None and repetitions >= 200
            and successes is not None and successes >= 0.9 * repetitions
        )
        _check(
            checks, "reliability_direction_stability",
            "pass" if direction_ok else "warn",
            "item-rest direction and admissible bootstrap-fit rate were independently checked",
        )
    if method_id == "confirmatory_factor_analysis":
        metrics = _mapping_field(payload, "metrics")
        factors = _finite(metrics.get("factor_count"))
        loadings = _finite(metrics.get("loading_count"))
        df = _finite(metrics.get("degrees_of_freedom"))
        cfi = _finite(metrics.get("cfi")); tli = _finite(metrics.get("tli"))
        rmsea = _finite(metrics.get("rmsea")); srmr = _finite(metrics.get("srmr"))
        coherent = (
            factors is not None and factors >= 1 and loadings is not None
            and loadings >= 3 * factors and df is not None and df > 0
            and cfi is not None and tli is not None and rmsea is not None
            and srmr is not None and 0 <= srmr <= 1
        )
        _check(checks, "cfa_fit_contract", "pass" if coherent else "warn",
               "identified CFA dimensions and maintained global fit indices were reconciled")
    if method_id == "measurement_invariance":
        metrics = _mapping_field(payload, "metrics")
        changes = estimates if isinstance(estimates, dict) else {}
        invariance_expected: dict[str, tuple[float | None, float | None]] = {
            "metric_delta_cfi": (_finite(metrics.get("configural_cfi")), _finite(metrics.get("metric_cfi"))),
            "metric_delta_rmsea": (_finite(metrics.get("metric_rmsea")), _finite(metrics.get("configural_rmsea"))),
            "scalar_delta_cfi": (_finite(metrics.get("metric_cfi")), _finite(metrics.get("scalar_cfi"))),
            "scalar_delta_rmsea": (_finite(metrics.get("scalar_rmsea")), _finite(metrics.get("metric_rmsea"))),
        }
        coherent = True
        for key, (left, right) in invariance_expected.items():
            change = _finite(changes.get(key))
            if (left is None or right is None or change is None
                    or abs(change - (left - right)) > 0.002):
                coherent = False
                break
        configural_df = _finite(metrics.get("configural_df"))
        metric_df = _finite(metrics.get("metric_df"))
        scalar_df = _finite(metrics.get("scalar_df"))
        coherent = coherent and (
            configural_df is not None and metric_df is not None
            and scalar_df is not None and configural_df < metric_df < scalar_df
        )
        _check(checks, "invariance_nested_models", "pass" if coherent else "warn",
               "configural, metric, and scalar degrees of freedom and fit-index changes were independently reconciled")
    if method_id == "latent_class":
        metrics = _mapping_field(payload, "metrics")
        proportions = estimates if isinstance(estimates, dict) else {}
        starts = _finite(metrics.get("start_count"))
        stable = _finite(metrics.get("stable_start_count"))
        entropy = _finite(metrics.get("normalized_entropy"))
        minimum = _finite(metrics.get("min_expected_class_n"))
        gap = _finite(metrics.get("second_best_gap"))
        tolerance = _finite(metrics.get("likelihood_tolerance"))
        coherent = (
            len(proportions) >= 2 and abs(sum(_finite_values(proportions)) - 1) <= 0.01
            and starts is not None and starts >= 5 and stable is not None and 2 <= stable <= starts
            and entropy is not None and 0 <= entropy <= 1
            and minimum is not None and minimum >= 10
            and gap is not None and tolerance is not None and 0 <= gap <= tolerance + 0.002
        )
        _check(checks, "latent_class_stability", "pass" if coherent else "warn",
               "class mass, entropy, minimum expected support, and reproduced multi-start optimum were checked")
    if method_id == "bayesian_model" and isinstance(diagnostics, dict):
        rhat = _finite(diagnostics.get("rhat"))
        bulk = _finite(diagnostics.get("bulk_ess"))
        tail = _finite(diagnostics.get("tail_ess"))
        divergences = _finite(diagnostics.get("divergences"))
        ppc = _finite(diagnostics.get("posterior_predictive_check"))
        if (rhat is None or bulk is None or tail is None
                or divergences is None or ppc is None):
            bad = True
        else:
            bad = (rhat >= 1.01 or bulk < 400 or tail < 400
                   or divergences != 0 or not .01 <= ppc <= .99)
        _check(checks, "bayesian_computation", "warn" if bad else "pass",
               "Bayesian convergence/efficiency diagnostics require R-hat <1.01, "
               "bulk/tail ESS support, and zero divergences; inspect reported diagnostics")
    if method_id == "geospatial_analysis":
        p_value = _finite(metrics.get("permutation_p_value"))
        islands = _finite(metrics.get("island_fraction"))
        _check(checks, "spatial_reference", "pass",
               f"projected EPSG:{payload.get('crs_epsg')} distances and a declared binary distance band define the weights")
        _check(checks, "spatial_dependence", "pass" if p_value is not None else "warn",
               f"Moran's I uses permutation p={p_value}; this diagnoses dependence and is not a causal effect")
        _check(checks, "spatial_support", "pass" if islands is not None and islands == 0 else "warn",
               f"spatial-weight island fraction is {islands}")
    if method_id == "network_analysis":
        _check(checks, "graph_privacy", "pass",
               "only whole-graph aggregates are released; node identifiers and edge rows are absent")
        _check(checks, "graph_dependence", "warn",
               "descriptive topology does not provide independent-observation uncertainty or causal effects")
    if method_id == "text_analysis":
        stability = _finite(metrics.get("resampling_stability_ari"))
        _check(checks, "text_privacy", "pass",
               "only corpus-level aggregates are released; documents, tokens, vocabulary, and assignments are absent")
        _check(checks, "text_stability", "pass" if stability is not None and stability >= .8 else "warn",
               f"independent document resamples have adjusted Rand stability {stability}; this is stability, not held-out prediction")
    if method_id == "power_precision":
        _check(checks, "prospective_power", "pass",
               "sample sizes are calculated from declared prospective effect-size scenarios, alpha, target power, allocation, and "
               f"{payload.get('test_alternative') or 'undeclared'} alternative")
    if (payload.get("method_family") == "predictive"
            and payload.get("evaluation_split") == "held_out"):
        conditional_interval = payload.get("interval_method") == "heldout_case_bootstrap"
        _check(
            checks, "predictive_interval_scope",
            "pass" if conditional_interval else "warn",
            "case-resampling interval quantifies held-out performance uncertainty conditional on the fitted workflow; it does not include model-refit uncertainty",
        )
    if method_id == "probability_calibration":
        brier = _finite(metrics.get("brier_score"))
        ece = _finite(metrics.get("expected_calibration_error"))
        minimum_bin = _finite(metrics.get("minimum_calibration_bin_count"))
        _check(
            checks, "calibration_aggregate_contract",
            "pass" if brier is not None and ece is not None and minimum_bin is not None and minimum_bin >= 5 else "warn",
            "held-out Brier score and equal-frequency reliability-bin error are aggregate-only; row probabilities are not released",
        )
        _check(
            checks, "calibration_claim_boundary", "warn",
            "calibration evidence is conditional on the fitted workflow and represented held-out population; discrimination is context, not the reported estimand",
        )
    if method_id == "simulation_design":
        mcse = _finite(metrics.get("monte_carlo_standard_error"))
        gap = _finite(metrics.get("absolute_analytic_difference"))
        _check(checks, "monte_carlo_precision", "pass" if mcse is not None and mcse <= .02 else "warn",
               f"recorded-seed simulation Monte Carlo SE is {mcse}")
        _check(checks, "simulation_reference", "pass" if gap is not None and gap <= .04 else "warn",
               f"simulation and analytic reference power differ by {gap}")
    if payload.get("method_family") == "predictive":
        split = payload.get("evaluation_split")
        _check(checks, "out_of_sample_evaluation",
               "pass" if split in {"held_out", "cross_validation", "grouped", "rolling_origin"} else "warn",
                   f"prediction performance evaluation used {split or 'no declared'} split")
        diagnostics = _mapping_field(payload, "diagnostics")
        metrics = _mapping_field(payload, "metrics")
        nested = (
            diagnostics.get("preprocessing_inside_split") == "pass"
            and diagnostics.get("split_integrity") == "pass"
        )
        _check(
            checks, "predictive_leakage_boundary", "pass" if nested else "warn",
            "preprocessing and any probability calibration were fit inside training partitions",
        )
        improvement = _finite(metrics.get("baseline_improvement"))
        _check(
            checks, "simple_baseline_comparison",
            "pass" if improvement is not None and improvement >= 0 else "warn",
            (
                f"out-of-sample improvement over the simple baseline was {improvement:.4g}"
                if improvement is not None
                else "simple baseline improvement was not quantified"
            ),
        )
        _check(
            checks, "predictive_calibration",
            "pass" if diagnostics.get("calibration") == "pass" else "warn",
            "out-of-sample calibration slope/intercept were checked separately from discrimination",
        )


def _adjust_p_values(
    raw: dict[str, Any], method: str,
) -> dict[str, float] | None:
    """Dependency-free Holm/Bonferroni/BH reference for verification."""
    try:
        values = {key: float(value) for key, value in raw.items()}
    except (TypeError, ValueError):
        return None
    if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
        return None
    m = len(values)
    if method == "bonferroni":
        return {key: min(1.0, value * m) for key, value in values.items()}
    ordered = sorted(values, key=lambda key: values[key])
    out: dict[str, float] = {}
    if method == "holm":
        running = 0.0
        for rank, key in enumerate(ordered):
            running = max(running, (m - rank) * values[key])
            out[key] = min(1.0, running)
        return out
    if method == "benjamini_hochberg":
        running = 1.0
        for rank in range(m, 0, -1):
            key = ordered[rank - 1]
            running = min(running, values[key] * m / rank)
            out[key] = min(1.0, running)
        return out
    return None


# ---------------------------------------------------------------------------
# Finding confidence: a single strong / moderate / weak rollup of the
# checks already computed above.
# ---------------------------------------------------------------------------
#
# Deliberately NOT a new independent judgement — it is a deterministic
# function of the check ids and statuses this module already produces,
# so it can never disagree with the detail a researcher would see by
# reading the checks list itself. The rule is simple by design (three
# tiers, one severity set) because a confidence label people will
# skim needs to be predictable from its inputs, not itself a model to
# audit.
#
# "Severe" check ids are ones where, on their own, a single warning
# already undermines trusting the point estimate at face value
# (non-convergence, evidence of leakage/separation, an unstable
# design matrix, too few observations, weak/invalid instruments, or
# an underpowered null result). Every other warning (SE flavour,
# specification-preference tests like Hausman/F-test-FE, obs/param
# ratio, incremental fit indices, ...) is a real caveat worth reading
# but doesn't by itself mean the finding shouldn't be trusted —
# accumulating several of them still degrades confidence, just one
# tier at a time rather than straight to "weak".
_SEVERE_CHECK_IDS: frozenset[str] = frozenset((
    "convergence", "suspicious_fit", "extreme_t_statistic",
    "target_leakage_naming", "conditioning", "sample_size",
    "instrument_strength", "overidentification", "power",
    "model_fit", "sampling_adequacy", "sphericity",
))


def _confidence_level(checks: list[dict[str, str]]) -> dict[str, str]:
    warn_ids = [c["id"] for c in checks if c["status"] == "warn"]
    if not warn_ids:
        return {"level": "strong",
                "scope": "reported_diagnostics_only",
                "reason": (
                    "no reported diagnostic raised a warning; diagnostics "
                    "that were not supplied remain unassessed"
                )}
    severe = [cid for cid in warn_ids if cid in _SEVERE_CHECK_IDS]
    if severe:
        return {"level": "weak",
                "scope": "reported_diagnostics_only",
                "reason": (
                    f"{len(severe)} check(s) flagged a serious concern "
                    f"({', '.join(sorted(severe))})"
                )}
    if len(warn_ids) <= 2:
        return {"level": "moderate",
                "scope": "reported_diagnostics_only",
                "reason": (
                    f"{len(warn_ids)} caveat(s) noted "
                    f"({', '.join(sorted(warn_ids))}) but nothing that "
                    "undermines the finding on its own"
                )}
    return {"level": "weak",
            "scope": "reported_diagnostics_only",
            "reason": (
                f"{len(warn_ids)} caveats accumulated "
                f"({', '.join(sorted(warn_ids))}); individually minor "
                "but the combination warrants caution"
            )}


def _verify_regression(payload: dict[str, Any],
                       checks: list[dict[str, str]],
                       n: float | None) -> None:
    coefs = payload.get("coefficients")
    k = len(coefs) if isinstance(coefs, dict) else None

    if n is not None and k:
        ratio = n / k
        if ratio < _OBS_PER_PARAM_WARN:
            _check(checks, "obs_per_parameter", "warn",
                   f"{ratio:.1f} observations per parameter "
                   f"(n={int(n)}, k={k}); risk of overfitting")
        else:
            _check(checks, "obs_per_parameter", "pass",
                   f"{ratio:.0f} observations per parameter")

    vif = payload.get("vif")
    if isinstance(vif, dict) and vif:
        numeric = _finite_values(vif)
        if numeric:
            worst = max(numeric)
            if worst > _VIF_WARN:
                offenders = sorted(
                    (
                        str(name) for name, value in vif.items()
                        if (finite := _finite(value)) is not None
                        and finite > _VIF_WARN
                    ),
                )[:5]
                _check(checks, "multicollinearity", "warn",
                       f"max VIF {worst:.1f} > {_VIF_WARN:g} "
                       f"({', '.join(offenders)}); coefficient SEs "
                       "for these variables are inflated")
            else:
                _check(checks, "multicollinearity", "pass",
                       f"max VIF {worst:.1f} ≤ {_VIF_WARN:g}")

    cond = _num(payload, "condition_number")
    if cond is not None:
        if cond > _CONDITION_NUMBER_WARN:
            _check(checks, "conditioning", "warn",
                   f"design-matrix condition number {cond:.0f} > "
                   f"{_CONDITION_NUMBER_WARN:g}; results may be "
                   "numerically unstable")
        else:
            _check(checks, "conditioning", "pass",
                   f"condition number {cond:.0f}")

    for key in ("r_squared", "pseudo_r_squared"):
        r2 = _num(payload, key)
        if r2 is not None and r2 > _R2_SUSPICIOUS:
            _check(checks, "suspicious_fit", "warn",
                   f"{key}={r2:.4f} is near-perfect; check for target "
                   "leakage, duplicated rows, or a degenerate model")

    # Extreme coefficient t-statistics — an independent leakage /
    # separation signal from the R² check above (a single dominant
    # leaked predictor can produce this even when R² itself isn't
    # near 1, e.g. in a multi-predictor model where the other
    # predictors add noise).
    t_stats = payload.get("t_statistics")
    if isinstance(t_stats, dict) and t_stats:
        magnitudes = {
            name: abs(v) for name, v in
            ((n, _finite(v)) for n, v in t_stats.items())
            if v is not None
        }
        if magnitudes:
            worst = max(magnitudes.values())
            if worst > _EXTREME_T_STAT:
                offenders = sorted(
                    name for name, mag in magnitudes.items()
                    if mag > _EXTREME_T_STAT
                )[:5]
                _check(checks, "extreme_t_statistic", "warn",
                       f"|t| up to {worst:.0f} (> {_EXTREME_T_STAT:g}) "
                       f"for {', '.join(offenders)}; unusually large "
                       "t-statistics like this can indicate target "
                       "leakage or, for logistic/probit models, "
                       "perfect or quasi-perfect separation")

    # Target-leakage naming heuristic: does any predictor's name
    # share a specific (non-generic) token with the response
    # variable's name? A heuristic only — flags naming SMELLS, not
    # proven leakage; the honest framing below reflects that.
    response_var = payload.get("response_variable")
    predictors = payload.get("predictor_variables")
    if isinstance(response_var, str) and isinstance(predictors, list):
        resp_tokens = _name_tokens_for_leakage(response_var)
        if resp_tokens:
            suspicious: list[tuple[str, set[str]]] = []
            for p in predictors:
                overlap = resp_tokens & _name_tokens_for_leakage(p)
                if overlap:
                    suspicious.append((p, overlap))
            if suspicious:
                described = ", ".join(
                    f"{p!r} (shares {'/'.join(sorted(ov))!r})"
                    for p, ov in suspicious[:5]
                )
                _check(checks, "target_leakage_naming", "warn",
                       f"predictor name(s) share a specific token with "
                       f"the response variable {response_var!r}: "
                       f"{described}; worth checking whether any of "
                       "these were measured after, or as a direct "
                       "consequence of, the outcome")
            else:
                _check(checks, "target_leakage_naming", "pass",
                       "no predictor name shares a suspicious token "
                       "with the response variable")

    se = payload.get("standard_errors")
    if isinstance(se, dict) and se:
        _check(checks, "uncertainty_reported", "pass",
               "standard errors present for all coefficients"
               if isinstance(coefs, dict) and set(se) >= set(coefs)
               else "standard errors present")

    robust = payload.get("robust_se_type")
    if isinstance(robust, str) and robust:
        if robust == "classical":
            _check(checks, "robust_se", "warn",
                   "classical (non-robust) standard errors; consider "
                   "a heteroskedasticity-robust specification")
        else:
            _check(checks, "robust_se", "pass",
                   f"robust standard errors in use ({robust})")

    fsf = _num(payload, "first_stage_f")
    if fsf is not None:
        if fsf < _FIRST_STAGE_F_WEAK:
            _check(checks, "instrument_strength", "warn",
                   f"first-stage F {fsf:.1f} < {_FIRST_STAGE_F_WEAK:g}; "
                   "the instruments may be weak under the conventional "
                   "Staiger-Stock rule of thumb. Use a design-specific "
                   "weak-instrument diagnostic/critical value when there "
                   "are multiple endogenous regressors or robust errors")
        else:
            _check(checks, "instrument_strength", "pass",
                   f"first-stage F {fsf:.1f} clears the conventional 10 "
                   "rule of thumb; this alone does not establish instrument "
                   "validity or strong identification")

    # Optimizer convergence. A non-converged fit's coefficients and
    # SEs can be numerically meaningless even though every other
    # field in the payload validates cleanly — this is the single
    # highest-priority check in this function when present, so it's
    # checked (and reported) regardless of what else is in the
    # payload. See ``sanitizer._OLS_VALID_CONVERGED`` for the enum.
    converged = payload.get("converged")
    if isinstance(converged, str) and converged:
        if converged == "not_converged":
            _check(checks, "convergence", "warn",
                   "the optimizer did not converge; coefficients and "
                   "standard errors may not be reliable — re-fit with "
                   "a different starting point, more iterations, or a "
                   "simpler specification before interpreting this "
                   "result")
        elif converged == "converged_with_warnings":
            _check(checks, "convergence", "warn",
                   "the optimizer converged but reported a warning "
                   "(e.g. a boundary or near-singular fit); treat "
                   "estimates near the flagged boundary with caution")
        else:
            _check(checks, "convergence", "pass", "the optimizer converged")

    # Panel-data specification tests (Hausman, panel Breusch-Pagan,
    # F-test for fixed effects, Wooldridge AR(1)) and IV diagnostics
    # (Hansen J overidentification, Wu-Hausman endogeneity) — all
    # already carried through the sanitizer's OLS-bucket schema
    # (see ``_OLS_ALLOWED_NUMERIC_FIELDS`` in sanitizer.py) but never
    # previously consulted here. Each is a real, standard
    # specification test with a conventional alpha and a directly
    # actionable reading; surfacing them costs nothing extra from the
    # researcher (the payload already has the numbers) and closes a
    # real verification gap.
    hausman_p = _num(payload, "hausman_p")
    if hausman_p is not None:
        if hausman_p < _DIAGNOSTIC_TEST_P_MAX:
            _check(checks, "fe_vs_re", "warn",
                   f"Hausman test rejects random effects "
                   f"(p={hausman_p:.3g}); the random-effects estimator "
                   "is likely inconsistent here — a fixed-effects "
                   "specification is preferred")
        else:
            _check(checks, "fe_vs_re", "pass",
                   f"Hausman test does not reject random effects "
                   f"(p={hausman_p:.3g})")

    bp_panel_p = _num(payload, "breusch_pagan_p")
    if bp_panel_p is not None:
        if bp_panel_p < _DIAGNOSTIC_TEST_P_MAX:
            _check(checks, "re_vs_pooled", "warn",
                   f"panel Breusch-Pagan LM test rejects pooled OLS "
                   f"(p={bp_panel_p:.3g}); random effects are "
                   "preferred over pooled OLS for this panel")
        else:
            _check(checks, "re_vs_pooled", "pass",
                   f"panel Breusch-Pagan LM test does not reject "
                   f"pooled OLS (p={bp_panel_p:.3g})")

    fe_p = _num(payload, "f_test_fe_p")
    if fe_p is not None:
        if fe_p < _DIAGNOSTIC_TEST_P_MAX:
            _check(checks, "fe_vs_pooled", "warn",
                   f"F-test rejects pooled OLS in favour of fixed "
                   f"effects (p={fe_p:.3g}); the fixed effects are "
                   "jointly significant")
        else:
            _check(checks, "fe_vs_pooled", "pass",
                   f"F-test does not reject pooled OLS "
                   f"(p={fe_p:.3g}); fixed effects add little here")

    ar1_p = _num(payload, "wooldridge_ar1_p")
    if ar1_p is not None:
        if ar1_p < _DIAGNOSTIC_TEST_P_MAX:
            _check(checks, "serial_correlation", "warn",
                   f"Wooldridge test finds first-order serial "
                   f"correlation in the panel residuals "
                   f"(p={ar1_p:.3g}); standard errors should be "
                   "clustered by panel unit (or the dynamics modelled "
                   "directly) — unclustered SEs here are "
                   "overconfident")
        else:
            _check(checks, "serial_correlation", "pass",
                   f"Wooldridge test finds no first-order serial "
                   f"correlation (p={ar1_p:.3g})")

    hansen_p = _num(payload, "hansen_j_p")
    if hansen_p is not None:
        if hansen_p < _DIAGNOSTIC_TEST_P_MAX:
            _check(checks, "overidentification", "warn",
                   f"Hansen J test rejects the overidentifying "
                   f"restrictions (p={hansen_p:.3g}); one or more "
                   "instruments may be invalid")
        else:
            _check(checks, "overidentification", "pass",
                   f"Hansen J test does not reject the overidentifying "
                   f"restrictions (p={hansen_p:.3g})")

    endog_p = _num(payload, "endogeneity_p")
    if endog_p is not None:
        if endog_p < _DIAGNOSTIC_TEST_P_MAX:
            _check(checks, "endogeneity", "warn",
                   f"Wu-Hausman test rejects exogeneity "
                   f"(p={endog_p:.3g}); OLS and IV estimates diverge "
                   "enough to justify using the IV specification")
        else:
            _check(checks, "endogeneity", "pass",
                   f"Wu-Hausman test does not reject exogeneity "
                   f"(p={endog_p:.3g}); OLS and IV estimates are not "
                   "meaningfully different here")

    icc = _num(payload, "icc")
    if icc is not None:
        # No warn threshold: ICC has no universal "too high/too low"
        # convention, it's descriptive of how much variance sits
        # between groups. Reported for context, always "pass".
        _check(checks, "intraclass_correlation", "pass",
               f"ICC={icc:.3f}; {icc * 100:.0f}% of total variance is "
               "between-group")


def _verify_t_test(payload: dict[str, Any],
                   checks: list[dict[str, str]]) -> None:
    # ``n1`` / ``n2`` are the only per-group counts the sanitizer
    # emits for this shape; earlier drafts also looked for
    # ``n_group1`` / ``n_group2``, which never appear.
    for key in ("n1", "n2"):
        gn = _num(payload, key)
        if gn is not None and gn < _MIN_COMFORTABLE_N:
            _check(checks, f"group_size_{key}", "warn",
                   f"{key}={int(gn)} is small for a t-test; normality "
                   "assumptions matter at this size")

    # Power / precision note (minimum detectable effect). See the
    # constants' docstring above for why this is a forward-looking
    # MDE calculation from the sample sizes alone, not a discredited
    # post-hoc power computed from the observed effect. Framed
    # differently depending on the result: a NULL result with a
    # large MDE is the case this genuinely protects against
    # ("absence of evidence" mis-reading); a significant result gets
    # the same number purely as context.
    n1 = _num(payload, "n1")
    n2 = _num(payload, "n2")
    test_type = payload.get("test_type")
    p_value = _num(payload, "p_value")
    mde: float | None = None
    design_label = ""
    if n1 and n1 > 0 and test_type in ("one_sample", "paired"):
        # Standardized one-sample effect (or standardized mean of paired
        # differences): one effective sample, not two independent groups.
        mde = (_MDE_Z_ALPHA + _MDE_Z_POWER) / math.sqrt(n1)
        design_label = "paired differences" if test_type == "paired" \
            else "one-sample"
    elif n1 and n2 and n1 > 0 and n2 > 0:
        # Preserve compatibility for older sanitized rows with no test_type:
        # two supplied group sizes unambiguously imply the independent-groups
        # approximation used before this branch became subtype-aware.
        mde = (_MDE_Z_ALPHA + _MDE_Z_POWER) * math.sqrt(1 / n1 + 1 / n2)
        design_label = "independent groups"

    if mde is not None:
        if p_value is not None and p_value >= _DIAGNOSTIC_TEST_P_MAX:
            if mde >= _MDE_LARGE_D:
                _check(checks, "power", "warn",
                       f"this null result (p={p_value:.3g}) cannot rule "
                       f"out a real effect smaller than about "
                       f"d≈{mde:.2f} (minimum detectable standardized "
                       f"effect at 80% power for {design_label}); do not "
                       "read this as "
                       "evidence of no effect without considering that")
            else:
                _check(checks, "power", "pass",
                       f"adequately powered to detect small-to-medium "
                       f"effects (minimum detectable standardized effect "
                       f"d≈{mde:.2f} at 80% power for {design_label})")
        else:
            _check(checks, "power", "pass",
                   f"this sample could reliably detect effects of "
                   f"standardized magnitude d≈{mde:.2f} or larger at "
                   f"80% power for "
                   f"{design_label}")


def _verify_counts(payload: dict[str, Any],
                   checks: list[dict[str, str]]) -> None:
    counts = payload.get("counts")
    if isinstance(counts, dict) and counts:
        suppressed = sum(
            1 for v in counts.values() if isinstance(v, str))
        total = len(counts)
        if suppressed:
            _check(checks, "suppression_extent", "warn",
                   f"{suppressed} of {total} cells suppressed for "
                   "disclosure control; distribution shape is partial")
        else:
            _check(checks, "suppression_extent", "pass",
                   f"no suppressed cells across {total} categories")


def _verify_text_extraction(payload: dict[str, Any],
                            checks: list[dict[str, str]]) -> None:
    """Suppression extent (reusing the same counts-shaped check as
    frequency_table/crosstab, just keyed on ``categories``) plus a
    taxonomy-coverage check: a classifier where most rows fall into
    "uncategorized" isn't telling the researcher much, and that's a
    fact about the KEYWORD LIST, not about the data -- worth flagging
    plainly rather than let a mostly-empty category breakdown read as
    a real finding.
    """
    categories = payload.get("categories")
    if isinstance(categories, dict) and categories:
        suppressed = sum(
            1 for v in categories.values() if isinstance(v, str))
        total = len(categories)
        if suppressed:
            _check(checks, "suppression_extent", "warn",
                   f"{suppressed} of {total} categories suppressed for "
                   "disclosure control; distribution shape is partial")
        else:
            _check(checks, "suppression_extent", "pass",
                   f"no suppressed categories across {total} categories")

        n = _num(payload, "n")
        uncategorized = categories.get("uncategorized")
        if (
            n is not None and n > 0
            and isinstance(uncategorized, int)
        ):
            share = 100.0 * uncategorized / n
            if share >= 40.0:
                _check(checks, "taxonomy_coverage", "warn",
                       f"{share:.0f}% of rows fell outside every "
                       "supplied category; the keyword taxonomy may "
                       "need broader terms, not that most text lacks "
                       "a theme")
            else:
                _check(checks, "taxonomy_coverage", "pass",
                       f"{share:.0f}% uncategorized — taxonomy covers "
                       "most of the text")


def batch_note(n_ok_results: int) -> str | None:
    """Envelope-level note for parameterized batches.

    Many results from one script usually means many hypothesis tests;
    flag the multiple-comparisons issue once, at the envelope level.
    """
    if n_ok_results >= _BATCH_MULTIPLE_COMPARISONS:
        return (
            f"{n_ok_results} results in this batch: p-values are not "
            "adjusted for multiple comparisons; consider "
            "Benjamini-Hochberg or Bonferroni before interpreting "
            "marginal significance"
        )
    return None


# ---------------------------------------------------------------------------
# Shape-specific verifiers
#
# Each reads only fields the sanitizer actually emits for that shape
# (see the ``_*_REQUIRED`` / ``_*_ALLOWED_*`` sets in ``sanitizer.py``).
# A check that references a field the sanitizer strips would never fire
# and would be silent dead weight, which is worse than no check because
# it looks like coverage.
# ---------------------------------------------------------------------------

def _verify_descriptive(payload: dict[str, Any],
                        checks: list[dict[str, str]]) -> None:
    """Missingness relative to the reported sample."""
    missing = _num(payload, "missing_count")
    n = _num(payload, "n")
    if missing is not None and n is not None and n > 0:
        share = 100.0 * missing / (n + missing) if (n + missing) else 0.0
        if share >= 20.0:
            _check(checks, "missingness", "warn",
                   f"{share:.0f}% of observations are missing this "
                   "variable; complete-case results may not represent "
                   "the full sample")
        else:
            _check(checks, "missingness", "pass",
                   f"{share:.0f}% missing")


def _verify_magnitude(payload: dict[str, Any],
                      checks: list[dict[str, str]]) -> None:
    cells = payload.get("cells")
    if isinstance(cells, dict) and cells:
        suppressed = sum(1 for v in cells.values() if isinstance(v, str))
        if suppressed:
            _check(checks, "suppression_extent", "warn",
                   f"{suppressed} of {len(cells)} cells suppressed for "
                   "disclosure control; totals do not decompose fully")
        else:
            _check(checks, "suppression_extent", "pass",
                   f"no suppressed cells across {len(cells)} groups")


def _verify_did(payload: dict[str, Any],
                checks: list[dict[str, str]]) -> None:
    """Difference-in-differences event study.

    The identifying assumption is parallel trends, and the pre-trend
    test is the one piece of evidence about it that the payload
    carries. Reporting an ATT while a pre-trend test rejects is the
    single most common way a DiD result misleads, so it is checked
    first and stated plainly.
    """
    pre_p = _num(payload, "pre_trends_p_value")
    if pre_p is not None:
        if pre_p < _PRE_TRENDS_P_MIN:
            _check(checks, "parallel_trends", "warn",
                   f"pre-treatment trend test rejects (p={pre_p:.3f}); "
                   "the parallel-trends assumption is questionable and "
                   "the estimate may reflect pre-existing divergence")
        else:
            _check(checks, "parallel_trends", "pass",
                   f"pre-trend test did not reject (p={pre_p:.3f}); this is "
                   "compatible with parallel pre-trends but does not prove "
                   "the identifying assumption, especially when power is low")
    else:
        _check(checks, "parallel_trends", "warn",
               "no pre-trend test reported; parallel trends is assumed "
               "but unexamined")

    cohorts = payload.get("n_treated_per_group")
    if isinstance(cohorts, dict) and cohorts:
        small = sorted(
            str(group) for group, value in cohorts.items()
            if (finite := _finite(value)) is not None
            and finite < _DID_MIN_COHORT
        )[:5]
        if small:
            _check(checks, "cohort_size", "warn",
                   f"{len(small)} treated cohort(s) below "
                   f"{_DID_MIN_COHORT} units ({', '.join(small)}); "
                   "cohort-specific effects there are imprecise")
        else:
            _check(checks, "cohort_size", "pass",
                   f"all {len(cohorts)} treated cohorts at or above "
                   f"{_DID_MIN_COHORT} units")

    if _num(payload, "aggregate_se") is not None or (
            _num(payload, "aggregate_ci_lower") is not None):
        _check(checks, "uncertainty_reported", "pass",
               "aggregate effect reported with uncertainty")
    elif _num(payload, "aggregate_att") is not None:
        _check(checks, "uncertainty_reported", "warn",
               "aggregate effect reported without a standard error or "
               "confidence interval")

    anticipation = _num(payload, "anticipation_periods")
    n_pre = _num(payload, "n_pre_treatment_periods")
    if n_pre is not None and n_pre < 2:
        _check(checks, "pre_periods", "warn",
               f"only {int(n_pre)} pre-treatment period(s); a pre-trend "
               "cannot be assessed meaningfully")
    if anticipation is not None and anticipation > 0:
        _check(checks, "anticipation", "pass",
               f"{int(anticipation)} anticipation period(s) excluded "
               "from the comparison window")


def _verify_rdd(payload: dict[str, Any],
                checks: list[dict[str, str]]) -> None:
    """Regression discontinuity.

    Effective sample size within the bandwidth — not the full N — is
    what the estimate rests on, and it is routinely far smaller than
    readers assume.
    """
    left = _num(payload, "effective_n_left")
    right = _num(payload, "effective_n_right")
    if left is not None and right is not None:
        smaller = min(left, right)
        if smaller < _RDD_MIN_EFFECTIVE_N:
            _check(checks, "effective_sample", "warn",
                   f"only {int(smaller)} effective observations on one "
                   f"side of the cutoff (left {int(left)}, right "
                   f"{int(right)}); the local estimate is fragile")
        else:
            _check(checks, "effective_sample", "pass",
                   f"effective N within bandwidth: {int(left)} left, "
                   f"{int(right)} right")
        if smaller > 0 and max(left, right) / smaller >= 2.0:
            _check(checks, "sample_balance", "warn",
                   "effective sample sizes differ by more than 2x "
                   "across the cutoff; the estimate leans on the "
                   "denser side")

    order = _num(payload, "polynomial_order")
    if order is not None:
        if order > _RDD_POLY_ORDER_MAX:
            _check(checks, "polynomial_order", "warn",
                   f"polynomial order {int(order)} exceeds "
                   f"{_RDD_POLY_ORDER_MAX}; high-order global "
                   "polynomials produce noisy, specification-driven "
                   "estimates (Gelman and Imbens, 2019)")
        else:
            _check(checks, "polynomial_order", "pass",
                   f"local polynomial of order {int(order)}")

    bw_l = _num(payload, "bandwidth_left")
    bw_r = _num(payload, "bandwidth_right")
    if bw_l and bw_r and min(bw_l, bw_r) > 0:
        ratio = max(bw_l, bw_r) / min(bw_l, bw_r)
        if ratio >= _RDD_BANDWIDTH_ASYMMETRY:
            _check(checks, "bandwidth_symmetry", "warn",
                   f"bandwidths differ {ratio:.1f}x across the cutoff; "
                   "check whether that is intended")

    tau_r = _num(payload, "tau_robust")
    tau_c = _num(payload, "tau_conventional")
    if tau_r is not None and tau_c is not None and abs(tau_c) > 1e-12:
        drift = abs(tau_r - tau_c) / abs(tau_c)
        if drift > 0.5:
            _check(checks, "estimator_agreement", "warn",
                   "robust and conventional estimates differ by more "
                   f"than 50% ({tau_c:.4g} vs {tau_r:.4g}); the result "
                   "is sensitive to bias correction")
        else:
            _check(checks, "estimator_agreement", "pass",
                   "robust and conventional estimates are comparable")

    fsf = _num(payload, "first_stage_f")
    if fsf is not None:
        if fsf < _FIRST_STAGE_F_WEAK:
            _check(checks, "instrument_strength", "warn",
                   f"fuzzy-RDD first-stage F {fsf:.1f} < "
                   f"{_FIRST_STAGE_F_WEAK:g}; the discontinuity is a "
                   "weak instrument for treatment")
        else:
            _check(checks, "instrument_strength", "pass",
                   f"fuzzy-RDD first-stage F {fsf:.1f}")


def _verify_kaplan_meier(payload: dict[str, Any],
                         checks: list[dict[str, str]]) -> None:
    """Survival analysis.

    Precision is governed by the number of *events*, and long-horizon
    survival estimates are governed by how many subjects remain at
    risk at that horizon — a 5-year estimate resting on 4 people is a
    number with no useful precision, and the KM curve gives no visual
    warning of it.
    """
    events = _num(payload, "n_failures")
    subjects = _num(payload, "n_subjects")
    if events is not None:
        if events < _KM_MIN_EVENTS:
            _check(checks, "event_count", "warn",
                   f"only {int(events)} event(s) observed; survival "
                   "estimates and any comparison are imprecise "
                   "regardless of how many subjects were followed")
        else:
            _check(checks, "event_count", "pass",
                   f"{int(events)} events observed")
    if events is not None and subjects is not None and subjects > 0:
        rate = 100.0 * events / subjects
        if rate < 10.0:
            _check(checks, "censoring", "warn",
                   f"only {rate:.0f}% of subjects experienced the "
                   "event; heavy censoring makes the tail of the curve "
                   "unstable")

    for horizon in ("1y", "3y", "5y", "10y"):
        at_risk = _num(payload, f"n_at_risk_{horizon}")
        estimate = _num(payload, f"survival_at_{horizon}")
        if estimate is None or at_risk is None:
            continue
        if at_risk < _KM_MIN_AT_RISK:
            _check(checks, f"at_risk_{horizon}", "warn",
                   f"the {horizon} survival estimate rests on "
                   f"{int(at_risk)} subject(s) still at risk; treat it "
                   "as indicative only")
        else:
            _check(checks, f"at_risk_{horizon}", "pass",
                   f"{int(at_risk)} at risk at {horizon}")

    if _num(payload, "median_survival_time") is not None:
        has_ci = (_num(payload, "median_survival_ci_lower") is not None
                  and _num(payload, "median_survival_ci_upper") is not None)
        _check(checks, "median_ci", "pass" if has_ci else "warn",
               "median survival reported with a confidence interval"
               if has_ci else
               "median survival reported without a confidence interval")


def _verify_cluster(payload: dict[str, Any],
                    checks: list[dict[str, str]]) -> None:
    """Cluster analysis.

    Clustering always returns clusters. Whether they reflect structure
    in the data or merely partition it is the question, so the
    separation diagnostic is the primary check.
    """
    sil = _num(payload, "silhouette_score")
    if sil is not None:
        if sil < _SILHOUETTE_WEAK:
            _check(checks, "cluster_separation", "warn",
                   f"silhouette {sil:.2f} is below {_SILHOUETTE_WEAK}; "
                   "no substantial cluster structure was found "
                   "(Kaufman and Rousseeuw, 1990) — the partition may "
                   "be arbitrary")
        else:
            _check(checks, "cluster_separation", "pass",
                   f"silhouette {sil:.2f} indicates separable clusters")
    else:
        _check(checks, "cluster_separation", "warn",
               "no separation diagnostic reported; whether these "
               "clusters reflect real structure is unassessed")

    sizes = payload.get("cluster_sizes")
    if isinstance(sizes, dict) and sizes:
        numeric = _finite_values(sizes)
        suppressed = [v for v in sizes.values() if isinstance(v, str)]
        if numeric:
            smallest = min(numeric)
            if smallest < _MIN_CLUSTER_SIZE:
                _check(checks, "cluster_sizes", "warn",
                       f"smallest cluster holds {int(smallest)} "
                       "observations; clusters that small are unstable "
                       "across reruns and reseeds")
            else:
                _check(checks, "cluster_sizes", "pass",
                       f"smallest cluster holds {int(smallest)} "
                       "observations")
        if suppressed:
            _check(checks, "cluster_suppression", "warn",
                   f"{len(suppressed)} cluster size(s) suppressed for "
                   "disclosure control")

    n_obs = _num(payload, "n_observations")
    k = _num(payload, "n_clusters")
    if n_obs and k and k > 0 and n_obs / k < 20:
        _check(checks, "clusters_vs_n", "warn",
               f"{int(k)} clusters over {int(n_obs)} observations "
               f"({n_obs / k:.0f} per cluster on average); consider "
               "whether the solution is over-partitioned")


def _verify_factor(payload: dict[str, Any],
                   checks: list[dict[str, str]]) -> None:
    """Factor analysis / PCA.

    Two questions precede interpreting any loading: is the correlation
    structure factorable at all (KMO, Bartlett), and does the retained
    solution fit (RMSEA, TLI).
    """
    kmo = _num(payload, "kmo")
    if kmo is not None:
        if kmo < _KMO_MIN:
            _check(checks, "sampling_adequacy", "warn",
                   f"KMO {kmo:.2f} is below {_KMO_MIN}; the correlation "
                   "structure is weak for factoring (Kaiser, 1974)")
        else:
            _check(checks, "sampling_adequacy", "pass",
                   f"KMO {kmo:.2f}")

    bartlett_p = _num(payload, "bartlett_p_value")
    if bartlett_p is not None:
        if bartlett_p > _BARTLETT_P_MAX:
            _check(checks, "sphericity", "warn",
                   f"Bartlett test does not reject (p={bartlett_p:.3f}); "
                   "the correlation matrix is not distinguishable from "
                   "identity, so factoring may not be justified")
        else:
            _check(checks, "sphericity", "pass",
                   f"Bartlett test rejects sphericity (p={bartlett_p:.3g})")

    rmsea = _num(payload, "rmsea")
    if rmsea is not None:
        if rmsea > _RMSEA_POOR:
            _check(checks, "model_fit", "warn",
                   f"RMSEA {rmsea:.3f} exceeds {_RMSEA_POOR}; the "
                   "retained solution fits poorly (Browne and Cudeck, "
                   "1993)")
        else:
            _check(checks, "model_fit", "pass", f"RMSEA {rmsea:.3f}")

    tli = _num(payload, "tli")
    if tli is not None and tli < _TLI_POOR:
        _check(checks, "incremental_fit", "warn",
               f"TLI {tli:.2f} is below {_TLI_POOR}")

    n_obs = _num(payload, "n_observations")
    n_var = _num(payload, "n_variables")
    if n_obs and n_var and n_var > 0:
        ratio = n_obs / n_var
        if ratio < _FACTOR_OBS_PER_VAR:
            _check(checks, "obs_per_variable", "warn",
                   f"{ratio:.1f} observations per variable; loadings "
                   "are unstable below roughly "
                   f"{_FACTOR_OBS_PER_VAR}:1")
        else:
            _check(checks, "obs_per_variable", "pass",
                   f"{ratio:.0f} observations per variable")


def _verify_marginal_effects(payload: dict[str, Any],
                             checks: list[dict[str, str]]) -> None:
    """Marginal effects from a non-linear fit."""
    effects = payload.get("effects")
    if not isinstance(effects, dict) or not effects:
        return
    ses = payload.get("standard_errors")
    ci_lo = payload.get("ci_lower")
    has_uncertainty = (
        (isinstance(ses, dict) and ses) or (isinstance(ci_lo, dict) and ci_lo))
    _check(checks, "uncertainty_reported",
           "pass" if has_uncertainty else "warn",
           "marginal effects reported with uncertainty" if has_uncertainty
           else "marginal effects reported without standard errors or "
                "confidence intervals")

    method = payload.get("method")
    if isinstance(method, str) and method:
        _check(checks, "estimand", "pass",
               f"marginal effects computed as {method}; the estimand "
               "depends on this choice")


# ---------------------------------------------------------------------------
# Session-level accounting
# ---------------------------------------------------------------------------
#
# Per-result checks cannot see the shape of an investigation. Two of
# the most consequential problems in applied statistics are only
# visible across results:
#
# - **Accumulated multiple comparisons.** A researcher who runs forty
#   hypothesis tests over an afternoon — each in its own script, each
#   individually clean — should expect roughly two "significant" results
#   at p<0.05 from pure noise. The per-batch note catches a 24-spec
#   loop; it cannot see forty separate calls. This does.
# - **Silent sample drift.** When results on the same dataset report
#   materially different N, a filter or a merge changed the population
#   mid-analysis. That is either intentional and worth stating in the
#   paper, or accidental and invalidating. Either way the researcher
#   should be told, because nothing else will tell them.
#
# Computed from the stored sanitized results — no new data crosses any
# boundary, and the accounting is arithmetic, not judgement.

_SESSION_TEST_WARN = 10          # tests before the MC note fires
_ALPHA = 0.05
_DRIFT_WARN_RATIO = 0.10         # 10% swing in N on one dataset

# Payload keys that carry a p-value, per shape.
_P_VALUE_KEYS = (
    "p_value", "p_values", "aggregate_p_value", "p_robust",
    "logrank_p_value", "chi_squared_p_value", "bartlett_p_value",
)


def _count_tests(payload: dict[str, Any]) -> tuple[int, int]:
    """Return ``(n_tests, n_significant)`` for one sanitized payload."""
    n_tests = n_sig = 0

    def p_value_leaves(value: Any):
        """Yield finite p-values from flat or nested result shapes.

        DiD stores p-values as cohort -> event-time -> value; the previous
        one-level walk silently counted zero tests for the entire event-study
        panel and understated session-wide multiplicity.
        """
        if isinstance(value, dict):
            for child in value.values():
                yield from p_value_leaves(child)
            return
        p = _finite(value)
        if p is not None and 0.0 <= p <= 1.0:
            yield p

    for key in _P_VALUE_KEYS:
        val = payload.get(key)
        for p in p_value_leaves(val):
            n_tests += 1
            n_sig += int(p < _ALPHA)
    return n_tests, n_sig


# ---------------------------------------------------------------------------
# Specification search / "garden of forking paths" detection
# ---------------------------------------------------------------------------
#
# ``challenge_summary`` catches an explicit batch: one script, several
# results the model itself grouped as alternative specifications of
# the same relationship. It says nothing about the far more common
# way specification search actually happens in practice: a dozen
# separate ``submit_script`` calls over an afternoon, each fitting the
# same response variable with a different control set, an outlier
# trim, or a subsample restriction -- never batched together, so
# ``challenge_summary`` never sees them as related. Each individual
# regression can be a clean, well-specified result on its own; the
# pattern is only visible across the session.
#
# This groups every regression-bucket result in the session by
# (dataset, response variable) and looks for two symptoms of forking:
#
#   - Many distinct specifications tried against the same outcome.
#     "Distinct" means a different ``predictor_variables`` set. Trying
#     five control-set variants for one outcome is not inherently
#     wrong -- that is what a robustness table is -- but the
#     researcher should know Sift counted five, not assume it went
#     unnoticed.
#   - A predictor that is significant (p < 0.05) in some
#     specifications tried for the same outcome and not in others.
#     This is the single most common tell of "kept adding controls
#     until it turned significant", and it is a different signal from
#     ``challenge_summary``'s sign check: a coefficient can keep its
#     sign while losing significance, which sign-agreement alone
#     cannot see.
#
# Both symptoms require the group to span at least two DISTINCT
# ``script_run_id`` values before either one fires. Without this
# gate, the mandatory robustness pass every finding is supposed to
# get (system_prompt.py: "re-estimate the key result under
# alternative specifications in the same script") would trip this
# detector on every single well-behaved batch -- three or four
# alternative specs, one script, one script_run_id, already
# transparently reported together and already given a code-computed
# ROBUST/FRAGILE verdict by ``challenge_summary``. That is the
# opposite of what this detector exists to catch: it is specifically
# about specifications that were NOT reported together, run across
# separate, disconnected ``submit_script`` calls. A single batch's
# internal spec count and significance pattern is challenge_summary's
# job, not this one's.
#
# Same posture as the rest of this module otherwise: computed only
# from stored sanitized payloads, arithmetic rather than judgement,
# reported as an advisory the researcher/model reads -- it never
# blocks or discards a result.

_SPEC_SEARCH_COUNT_WARN = 3      # distinct specs on one outcome before the count note fires
_SPEC_SEARCH_MIN_COMPARE = 2     # specs needed before a significance comparison means anything
_SPEC_SEARCH_MIN_RUNS = 2        # distinct script_run_ids required before this detector applies at all
_REGRESSION_SESSION_TYPES = frozenset((
    "linear_regression", "coefficient_table_with_fit_stats",
))


def _regression_spec_key(payload: dict[str, Any]) -> tuple[str, ...] | None:
    """Distinct-specification fingerprint: the sorted predictor set.

    Returns ``None`` for a payload with no usable predictor list (an
    intercept-only model, or a malformed/absent field) -- those are
    excluded from spec-search grouping rather than treated as a
    zero-predictor "specification" shared by every such result.
    """
    preds = payload.get("predictor_variables")
    if not isinstance(preds, list) or not preds:
        return None
    names = sorted({str(p) for p in preds if isinstance(p, str) and p})
    return tuple(names) if names else None


def _source_set_label(item: dict[str, Any]) -> str:
    """Stable label for the exact declared input set of one analysis."""
    raw = item.get("source_datasets")
    values = list(raw) if isinstance(raw, (list, tuple)) else []
    singular = item.get("source_dataset")
    if isinstance(singular, str) and singular:
        values.append(singular)
    clean = sorted({value for value in values if isinstance(value, str) and value})
    return " + ".join(clean)


def _detect_specification_search(
    results: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Advisory checks for garden-of-forking-paths patterns across a
    session's regression-bucket results. See module comment above."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for item in results or []:
        if item.get("analysis_type") not in _REGRESSION_SESSION_TYPES:
            continue
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        response = payload.get("response_variable")
        if not isinstance(response, str) or not response:
            continue
        spec = _regression_spec_key(payload)
        if spec is None:
            continue
        dataset = _source_set_label(item)
        coeffs = payload.get("coefficients")
        pvals = payload.get("p_values")
        run_id = item.get("script_run_id")
        groups.setdefault((dataset, response), []).append({
            "spec": spec,
            "coefficients": coeffs if isinstance(coeffs, dict) else {},
            "p_values": pvals if isinstance(pvals, dict) else {},
            "label": str(item.get("label") or ""),
            "script_run_id": run_id if isinstance(run_id, str) and run_id
                else None,
        })

    checks: list[dict[str, str]] = []
    for (dataset, response), members in groups.items():
        # Gate on the group as a whole, before looking at specs at
        # all: fewer than two distinct calls means whatever specs
        # exist here came from one script (or from rows with no
        # run-id recorded at all) -- either way, not the cross-call
        # pattern this detector targets. See the module comment for
        # why a single batch is deliberately left to
        # ``challenge_summary`` instead.
        distinct_runs = {m["script_run_id"] for m in members
                         if m["script_run_id"] is not None}
        if len(distinct_runs) < _SPEC_SEARCH_MIN_RUNS:
            continue

        distinct_specs = {m["spec"] for m in members}
        if len(distinct_specs) < _SPEC_SEARCH_MIN_COMPARE:
            continue

        where = f" on {dataset}" if dataset else ""
        tag = (f"specification_search::{dataset}::{response}" if dataset
               else f"specification_search::{response}")

        if len(distinct_specs) >= _SPEC_SEARCH_COUNT_WARN:
            _check(checks, tag, "warn",
                   f"{len(distinct_specs)} distinct specifications fit "
                   f"for {response}{where} across {len(distinct_runs)} "
                   f"separate script runs this session. Report every "
                   f"specification tried, not only the one presented, "
                   f"or pre-register the primary specification before "
                   f"further analysis")

        by_predictor: dict[str, list[tuple[str, float, str | None]]] = {}
        for m in members:
            for name, p in m["p_values"].items():
                pv = _finite(p)
                if pv is None or not (0.0 <= pv <= 1.0):
                    continue
                by_predictor.setdefault(str(name), []).append(
                    (m["label"], pv, m["script_run_id"]))

        # A predictor with mixed significance among its entries is a
        # "flip" in the loose sense, but this check only WARNS when
        # that disagreement is evidenced ACROSS runs -- entries that
        # disagree must themselves span multiple script_run_ids, not
        # just happen to sit in a group that contains more than one
        # run somewhere. Without that distinction, three specs
        # batched together in run A (already challenge_summary's
        # territory -- its internal instability is accounted for
        # there) plus one unrelated spec from run B elsewhere in the
        # same group would flag the run-A-only flip here too, purely
        # because run B's presence satisfied the group-level gate
        # without run B ever touching this predictor.
        #
        # A within-one-run flip is real, though, so "stable" would be
        # a false claim about it -- this check stays silent on that
        # predictor rather than asserting stability it cannot back up
        # OR warning about a pattern that isn't this check's to warn
        # about. Only a predictor with no disagreement of any kind
        # earns the "pass" verdict.
        cross_run_flips = sorted(
            name for name, entries in by_predictor.items()
            if len(entries) >= _SPEC_SEARCH_MIN_COMPARE
            and len({p < _ALPHA for _label, p, _rid in entries}) > 1
            and len({rid for _label, _p, rid in entries
                     if rid is not None}) >= _SPEC_SEARCH_MIN_RUNS
        )
        any_flip = any(
            len(entries) >= _SPEC_SEARCH_MIN_COMPARE
            and len({p < _ALPHA for _label, p, _rid in entries}) > 1
            for entries in by_predictor.values()
        )

        if cross_run_flips:
            _check(checks, f"{tag}::significance_stability", "warn",
                   f"{response}{where}: significance of "
                   f"{', '.join(cross_run_flips[:6])} changes across "
                   f"specifications run separately (p<{_ALPHA} in "
                   f"some, not others). State which specification is "
                   f"primary and report the others as robustness "
                   f"checks, not silently")
        elif by_predictor and not any_flip:
            _check(checks, f"{tag}::significance_stability", "pass",
                   f"{response}{where}: shared predictors' significance "
                   f"is stable across {len(distinct_specs)} "
                   f"specifications run separately")

    return checks


def session_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-result checks over a session.

    ``results`` is a list of ``{"analysis_type", "payload", "label",
    "source_dataset", "source_datasets"}`` dicts assembled by the caller
    from the store.
    Returns a block shaped like :func:`verify_payload` so both render
    the same way.
    """
    checks: list[dict[str, str]] = []
    total_tests = total_sig = 0
    n_by_dataset: dict[str, list[tuple[str, float]]] = {}

    for item in results or []:
        payload = item.get("payload")
        if not isinstance(payload, dict):
            continue
        tests, sig = _count_tests(payload)
        total_tests += tests
        total_sig += sig
        n = _num(payload, "n")
        dataset = _source_set_label(item)
        if n is not None and dataset:
            n_by_dataset.setdefault(dataset, []).append(
                (str(item.get("label") or ""), n))

    if total_tests:
        if total_tests >= _SESSION_TEST_WARN:
            expected = total_tests * _ALPHA
            _check(checks, "session_multiple_comparisons", "warn",
                   f"{total_tests} hypothesis tests across this session, "
                   f"{total_sig} significant at p<{_ALPHA}. About "
                   f"{expected:.1f} p-values below {_ALPHA} would be expected "
                   f"if every tested null were true and the tests valid; "
                   f"adjust (Benjamini-Hochberg or Bonferroni) or "
                   f"pre-register before treating marginal results as "
                   f"findings")
        else:
            _check(checks, "session_multiple_comparisons", "pass",
                   f"{total_tests} hypothesis tests recorded this "
                   f"session")

    for dataset, entries in n_by_dataset.items():
        if len(entries) < 2:
            continue
        sizes = [n for _label, n in entries]
        lo, hi = min(sizes), max(sizes)
        if hi <= 0:
            continue
        if (hi - lo) / hi >= _DRIFT_WARN_RATIO:
            _check(checks, f"sample_drift::{dataset}", "warn",
                   f"results on {dataset} use sample sizes from "
                   f"{int(lo):,} to {int(hi):,}; a filter, a merge or "
                   f"listwise deletion changed the population between "
                   f"analyses. State which sample each result refers "
                   f"to, or reconcile them")
        else:
            _check(checks, f"sample_drift::{dataset}", "pass",
                   f"consistent sample size across results on "
                   f"{dataset} ({int(hi):,})")

    checks.extend(_detect_specification_search(results or []))

    if not checks:
        return {"checks": [], "warnings": 0,
                "note": "no cross-result checks applicable yet"}
    return {
        "checks": checks,
        "warnings": sum(1 for c in checks if c["status"] == "warn"),
        "tests_run": total_tests,
        "significant": total_sig,
        "note": ("cross-result accounting computed from stored "
                 "sanitized results"),
    }


# ---------------------------------------------------------------------------
# Challenge Finding — deterministic agreement across alternative specs
# ---------------------------------------------------------------------------
#
# The mandatory robustness pass (system prompt) and the on-demand
# "Challenge" action both produce the same shape of evidence: one
# script, several results, each meant to be the SAME underlying
# relationship estimated a different way (different controls, outlier
# trimming, an alternate estimator, robust/clustered SE, a placebo
# window). Until now the ROBUST / SENSITIVE verdict on that batch was
# entirely the model's own reading of its own output — exactly the
# kind of claim this module exists to check instead of trust.
#
# ``challenge_summary`` computes the verdict from the numbers instead:
# for every named estimate common to the first ("baseline") result and
# each alternative, does the sign match? An alternative "agrees" when
# most of its shared estimates keep the baseline's sign. The count of
# agreeing alternatives out of the total is the ROBUST/FRAGILE count a
# researcher can actually check.
#
# Deliberately conservative about when it fires at all: if the batch's
# results don't share any named estimate with the baseline (an
# ordinary multi-result script producing unrelated tables — a
# regression, then a crosstab, then a descriptive summary), there is
# nothing to compare, and this returns ``None`` rather than fabricate
# a verdict across unrelated results. That single property is what
# makes it safe to compute automatically on every batch rather than
# needing the caller to first prove the batch is "a challenge" —
# structurally, only a genuine re-estimation batch has results that
# share coefficient/effect names.

# Payload fields, checked in order, that hold a name -> estimate dict.
# Regression coefficients and non-linear marginal effects are the two
# shapes Sift's runtime helpers emit today; a payload can carry at
# most one of these (the sanitizer's per-type schema is exclusive).
_ESTIMATE_DICT_FIELDS = ("coefficients", "effects")

# The payload does not identify which coefficient is the researcher's focal
# finding.  A majority rule can therefore call a batch ROBUST when the focal
# treatment estimate flips but two nuisance controls do not.  Require every
# shared named estimate to retain direction; callers that want a focal-only
# challenge should emit only that estimate in a marginal-effects result or a
# future explicit focal-estimate contract.
_CHALLENGE_AGREE_SHARE = 1.0


def _estimate_dict(payload: Any) -> dict[str, float] | None:
    """Pull the name -> finite-estimate mapping out of a sanitized
    payload, trying each known field in turn. Returns ``None`` when
    the payload carries none of them or the dict has no usable
    (finite, nonzero) values."""
    if not isinstance(payload, dict):
        return None
    for field_name in _ESTIMATE_DICT_FIELDS:
        raw = payload.get(field_name)
        if not isinstance(raw, dict) or not raw:
            continue
        cleaned: dict[str, float] = {}
        for key, val in raw.items():
            fv = _finite(val)
            # A zero estimate has no sign to compare — excluded
            # rather than arbitrarily counted either way.
            if fv is not None and fv != 0:
                cleaned[str(key)] = fv
        if cleaned:
            return cleaned
    return None


def challenge_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Deterministic ROBUST/FRAGILE verdict across a batch of results.

    ``results`` is an ordered list of sanitized payloads (or payload-
    bearing dicts) from one script run; the first entry is treated as
    the baseline / original specification and every later one as an
    alternative. Returns ``None`` when fewer than two results are
    given, or when no alternative shares a named estimate with the
    baseline (nothing to compare — not a challenge batch).
    """
    if not isinstance(results, list) or len(results) < 2:
        return None

    baseline = _estimate_dict(results[0])
    if baseline is None:
        return None

    comparisons: list[dict[str, Any]] = []
    for idx, alt_payload in enumerate(results[1:], start=1):
        alt = _estimate_dict(alt_payload)
        if alt is None:
            continue
        shared = sorted(set(baseline) & set(alt))
        if not shared:
            continue
        agree_names = [k for k in shared
                       if (baseline[k] > 0) == (alt[k] > 0)]
        share = len(agree_names) / len(shared)
        comparisons.append({
            "index": idx,
            "shared_estimates": len(shared),
            "agreeing_estimates": len(agree_names),
            "agrees": share >= _CHALLENGE_AGREE_SHARE,
            "disagreeing_names": sorted(set(shared) - set(agree_names)),
        })

    if not comparisons:
        return None

    n_total = len(comparisons)
    n_agree = sum(1 for c in comparisons if c["agrees"])
    verdict = "ROBUST" if n_agree == n_total else "FRAGILE"
    if verdict == "ROBUST":
        note = (
            f"ROBUST IN DIRECTION: all {n_total} alternative specification"
            f"{'s' if n_total != 1 else ''} checked agree in direction "
            f"with the original estimate. This checks sign stability only, "
            f"not magnitude, bias, or design validity."
        )
    else:
        flipped = sorted({
            name
            for c in comparisons if not c["agrees"]
            for name in c["disagreeing_names"]
        })
        note = (
            f"FRAGILE: only {n_agree} of {n_total} alternative "
            f"specifications agree with the original estimate."
        )
        if flipped:
            note += f" Sign changes on: {', '.join(flipped[:8])}."

    return {
        "verdict": verdict,
        "agreeing": n_agree,
        "total": n_total,
        "comparisons": comparisons,
        "scope": "direction_only",
        "note": note,
    }


def independent_challenge_pass(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently challenge stored outputs after the execution pipeline.

    Input rows contain only sanitized payloads plus local provenance.  This
    pass does not trust narrative text or execution success: it reruns the
    deterministic numerical verifier, checks declared output-schema status,
    compares the approved primary result with sensitivity analyses, and names
    sign contradictions across comparable estimates.
    """
    rows = [row for row in (results or []) if isinstance(row, dict)]
    checks: list[dict[str, Any]] = []
    contradictions: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        provenance = row.get("provenance")
        if not isinstance(payload, dict):
            checks.append({
                "result_id": row.get("result_id"), "status": "fail",
                "detail": "Stored result has no sanitized payload.",
            })
            continue
        verification = verify_payload(payload)
        warnings = [
            item for item in (verification or {}).get("checks", [])
            if item.get("status") == "warn"
        ]
        schema_ok = bool(
            isinstance(provenance, dict) and provenance.get("schema_verified") is True
        )
        checks.append({
            "result_id": row.get("result_id"),
            "status": "warn" if warnings or not schema_ok else "pass",
            "schema_verified": schema_ok,
            "numerical_warnings": len(warnings),
            "detail": (
                f"Declared schema {'verified' if schema_ok else 'not verified'}; "
                f"local numerical verifier reported {len(warnings)} warning(s)."
            ),
        })
        role = provenance.get("analysis_role") if isinstance(provenance, dict) else None
        if role == "primary":
            primary.append(row)
        elif role == "sensitivity":
            sensitivity.append(row)

    comparison = None
    if len(primary) == 1 and sensitivity:
        ordered = [primary[0]["payload"], *[row["payload"] for row in sensitivity]]
        comparison = challenge_summary(ordered)
    elif primary or sensitivity:
        comparison = {
            "verdict": "INCOMPLETE",
            "note": "A challenge pass needs exactly one primary result and at least one sensitivity result.",
            "scope": "workflow_roles",
        }

    estimates: list[tuple[str, dict[str, float]]] = []
    for row in rows:
        extracted = _estimate_dict(row.get("payload"))
        if extracted:
            estimates.append((str(row.get("result_id") or "unknown"), extracted))
    for index, (left_id, left) in enumerate(estimates):
        for right_id, right in estimates[index + 1:]:
            flipped = sorted(
                key for key in set(left) & set(right)
                if (left[key] > 0) != (right[key] > 0)
            )
            if flipped:
                contradictions.append({
                    "left_result_id": left_id, "right_result_id": right_id,
                    "kind": "estimate_direction_conflict",
                    "estimate_names": flipped[:20],
                })

    status = "pass"
    if contradictions or any(row["status"] != "pass" for row in checks):
        status = "warn"
    if comparison and comparison.get("verdict") in {"FRAGILE", "INCOMPLETE"}:
        status = "warn"
    return {
        "status": status,
        "scope": "sanitized_outputs_only",
        "checks": checks,
        "alternative_specification_comparison": comparison,
        "contradictions": contradictions,
        "limitations": [
            "This pass checks stored aggregates and declared diagnostics; it does not independently recreate the fit from raw data.",
            "Absence of a warning is not proof that assumptions or scientific interpretation are correct.",
        ],
    }


def workflow_challenge_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Run independent challenge passes within, never across, workflows."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in results or []:
        if not isinstance(row, dict):
            continue
        provenance = row.get("provenance")
        workflow_id = provenance.get("workflow_id") if isinstance(provenance, dict) else None
        if isinstance(workflow_id, str) and workflow_id:
            groups.setdefault(workflow_id, []).append(row)
    challenges = {
        workflow_id: independent_challenge_pass(rows)
        for workflow_id, rows in sorted(groups.items())
    }
    return {
        "status": (
            "warn" if any(row.get("status") != "pass" for row in challenges.values())
            else "pass"
        ),
        "workflows": challenges,
        "workflow_count": len(challenges),
    }
