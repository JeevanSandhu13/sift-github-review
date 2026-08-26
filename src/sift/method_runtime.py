"""Runtime-adapter guidance for registry-backed research methods.

The methodology registry answers *whether* a method fits a research design.
This module answers the separate execution question: which maintained Sift
adapter should generated Python or R code call after fitting the method (or,
for the newer adapters, which helper performs the fit itself).

Keeping this mapping explicit prevents a validated method from silently
falling back to a hand-assembled ``method_result`` when Sift has a typed,
executable-qualified adapter that can recompute the relevant diagnostics.
"""

from __future__ import annotations

from typing import Any


_PYTHON: dict[str, str] = {
    "descriptive_statistics": "sift.from_summarize",
    "frequency_table": "sift.from_table",
    "crosstab": "sift.from_crosstab",
    "magnitude_table": "sift.from_magnitude_table",
    "descriptive_confidence_interval": "sift.from_descriptive_confidence_interval",
    "t_test": "sift.from_t_test",
    "nonparametric_test": "sift.from_nonparametric_test",
    "proportion_test": "sift.from_proportion_test",
    "anova": "sift.from_anova",
    "ancova": "sift.from_anova",
    "repeated_measures_test": "sift.from_repeated_measures",
    "multiple_testing_correction": "sift.from_multiple_testing",
    "linear_regression": "sift.from_lm",
    "logistic_regression": "sift.from_lm",
    "probit_regression": "sift.from_lm",
    "poisson_regression": "sift.from_lm",
    "negative_binomial_regression": "sift.from_lm",
    "ordinal_regression": "sift.from_ordinal_model",
    "multinomial_regression": "sift.from_multinomial_model",
    "zero_inflated_model": "sift.from_zero_inflated_model",
    "spline_regression": "sift.from_spline_model",
    "marginal_effects": "sift.from_marginal_effects",
    "linear_mixed_effects": "sift.from_lm",
    "growth_curve": "sift.from_growth_curve",
    "gee": "sift.from_gee",
    "panel_fixed_effects": "sift.from_panel_fixed_effects",
    "panel_random_effects": "sift.from_panel_random_effects",
    "kaplan_meier": "sift.from_kaplan_meier",
    "cox_proportional_hazards": "sift.from_lm",
    "competing_risks": "sift.from_competing_risks",
    "recurrent_events": "sift.from_recurrent_events",
    "time_varying_survival": "sift.from_time_varying_survival",
    "matching": "sift.from_propensity_matching",
    "propensity_weighting": "sift.from_propensity_weighting",
    "difference_in_differences": "sift.from_difference_in_differences",
    "staggered_adoption": "sift.from_callaway_santanna",
    "event_study": "sift.from_sun_abraham",
    "synthetic_control": "sift.from_synthetic_control",
    "regression_discontinuity": "sift.from_rdd",
    "instrumental_variables": "sift.from_iv",
    "treatment_effect_heterogeneity": "sift.from_treatment_heterogeneity",
    "causal_sensitivity": "sift.from_causal_sensitivity",
    "survey_mean": "sift.from_survey_mean",
    "survey_proportion": "sift.from_survey_mean",
    "survey_regression": "sift.from_survey_regression",
    "missingness_pattern": "sift.from_missingness_pattern",
    "single_imputation": "sift.from_single_imputation",
    "multiple_imputation": "sift.from_multiple_imputation",
    "mnar_sensitivity": "sift.from_mnar_sensitivity",
    "stationarity_diagnostic": "sift.from_stationarity_diagnostic",
    "seasonal_decomposition": "sift.from_seasonal_decomposition",
    "arima": "sift.from_arima",
    "exponential_smoothing": "sift.from_exponential_smoothing",
    "interrupted_time_series": "sift.from_interrupted_time_series",
    "forecast_backtest": "sift.from_forecast_backtest",
    "pca": "sift.from_pca",
    "exploratory_factor_analysis": "sift.from_factor_analyzer",
    "reliability": "sift.from_reliability",
    "clustering": "sift.from_cluster",
    "predictive_regression": "sift.from_predictive_workflow",
    "predictive_classification": "sift.from_predictive_workflow",
    "probability_calibration": "sift.from_probability_calibration",
    "geospatial_analysis": "sift.from_geospatial_moran",
    "network_analysis": "sift.from_network_graph",
    "text_analysis": "sift.from_text_stability",
    "bayesian_model": "sift.from_arviz_posterior",
    "power_precision": "sift.from_power_precision",
    "simulation_design": "sift.from_simulation_design",
}

_R: dict[str, str] = {
    "descriptive_statistics": "sift$from_summarize",
    "frequency_table": "sift$from_table",
    "crosstab": "sift$from_crosstab",
    "magnitude_table": "sift$from_magnitude_table",
    "descriptive_confidence_interval": "sift$from_descriptive_confidence_interval",
    "t_test": "sift$from_t_test",
    "nonparametric_test": "sift$from_nonparametric_test",
    "proportion_test": "sift$from_proportion_test",
    "anova": "sift$from_anova",
    "ancova": "sift$from_anova",
    "repeated_measures_test": "sift$from_repeated_measures",
    "multiple_testing_correction": "sift$from_multiple_testing",
    "linear_regression": "sift$from_lm",
    "logistic_regression": "sift$from_lm",
    "probit_regression": "sift$from_lm",
    "poisson_regression": "sift$from_lm",
    "negative_binomial_regression": "sift$from_lm",
    "ordinal_regression": "sift$from_ordinal_model",
    "multinomial_regression": "sift$from_multinomial_model",
    "zero_inflated_model": "sift$from_zero_inflated_model",
    "spline_regression": "sift$from_spline_model",
    "marginal_effects": "sift$from_marginal_effects",
    "linear_mixed_effects": "sift$from_lm",
    "generalized_mixed_effects": "sift$from_lm",
    "kaplan_meier": "sift$from_kaplan_meier",
    "cox_proportional_hazards": "sift$from_lm",
    "matching": "sift$from_propensity_matching",
    "propensity_weighting": "sift$from_propensity_weighting",
    "staggered_adoption": "sift$from_callaway_santanna",
    "event_study": "sift$from_sun_abraham",
    "synthetic_control": "sift$from_synthetic_control",
    "regression_discontinuity": "sift$from_rdd",
    "treatment_effect_heterogeneity": "sift$from_treatment_heterogeneity",
    "causal_sensitivity": "sift$from_causal_sensitivity",
    "missingness_pattern": "sift$from_missingness_pattern",
    "single_imputation": "sift$from_single_imputation",
    "stationarity_diagnostic": "sift$from_stationarity_diagnostic",
    "seasonal_decomposition": "sift$from_seasonal_decomposition",
    "arima": "sift$from_arima",
    "exponential_smoothing": "sift$from_exponential_smoothing",
    "interrupted_time_series": "sift$from_interrupted_time_series",
    "forecast_backtest": "sift$from_forecast_backtest",
    "pca": "sift$from_pca",
    "exploratory_factor_analysis": "sift$from_fa",
    "confirmatory_factor_analysis": "sift$from_lavaan_cfa",
    "reliability": "sift$from_reliability",
    "measurement_invariance": "sift$from_lavaan_invariance",
    "clustering": "sift$from_cluster",
    "latent_class": "sift$from_polca",
    "power_precision": "sift$from_power_precision",
    "simulation_design": "sift$from_simulation_design",
}

# Shared adapters whose registry identity depends on one required, fixed
# argument. Returning this separately keeps ``preferred_helpers`` as stable
# callable names while preventing generated code from silently invoking the
# helper's default for a different method (for example ANCOVA as ANOVA or a
# survey proportion as a survey mean).
_REQUIRED_ARGUMENTS: dict[str, dict[str, dict[str, Any]]] = {
    "ancova": {
        "Python": {"method_id": "ancova"},
        "R": {"method_id": "ancova"},
    },
    "survey_proportion": {"Python": {"proportion": True}},
    "predictive_regression": {"Python": {"task": "regression"}},
    "predictive_classification": {"Python": {"task": "classification"}},
}


def runtime_guidance(method_id: str) -> dict[str, Any]:
    """Return bounded, code-owned execution guidance for one method ID."""
    helpers: dict[str, str] = {}
    if method_id in _PYTHON:
        helpers["Python"] = _PYTHON[method_id]
    if method_id in _R:
        helpers["R"] = _R[method_id]
    required_arguments = {
        language: dict(arguments)
        for language, arguments in _REQUIRED_ARGUMENTS.get(method_id, {}).items()
        if language in helpers
    }
    return {
        "preferred_helpers": helpers,
        "required_arguments": required_arguments,
        "typed_helper_available": bool(helpers),
        "instruction": (
            (
                "Use a preferred typed helper with every listed required "
                "argument; do not hand-assemble a method_result."
                if required_arguments else
                "Use a preferred typed helper; do not hand-assemble a method_result."
            )
            if helpers else
            "Fit the contract's maintained reference and emit only the required aggregate method_result fields."
        ),
    }


__all__ = ["runtime_guidance"]
