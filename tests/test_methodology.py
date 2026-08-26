from __future__ import annotations

import asyncio
import json

from sift.methodology import (
    CROSS_CUTTING_CONTROLS,
    DOMAIN_FLAGS,
    METHODS,
    METHODOLOGY_REGISTRY_VERSION,
    METHODOLOGY_SOURCES,
    OUTPUT_SCHEMAS,
    evaluate_method,
    recommend_methods,
    validate_registry,
    validate_research_specification,
)


def _complete_spec(**updates):
    spec = {
        "research_question": "How is x associated with y?",
        "unit_of_analysis": "person",
        "outcome": "y",
        "exposures": ["x"],
        "treatment": None,
        "predictors": ["x"],
        "controls": [],
        "target_population": "eligible adults",
        "estimand": "adjusted mean difference in y per unit x",
        "study_design": "cross-sectional observational",
        "goal": "associational",
        "repeated_measures": False,
        "clusters": None,
        "weights": None,
        "strata": None,
        "psu": None,
        "fpc": None,
        "replicate_weights": None,
        "time_ordering": "x measured before or with y; causal order unresolved",
        "missing_data_assumption": "MAR for adjusted analysis",
    }
    spec.update(updates)
    return spec


def test_registry_has_complete_contract_for_every_method() -> None:
    assert METHODOLOGY_REGISTRY_VERSION == 2
    assert validate_registry() == ()
    assert len(METHODS) >= 70
    for method in METHODS.values():
        assert method.assumptions
        assert method.diagnostics
        assert method.references
        assert method.output_schema
        assert method.claim_rule
        assert method.output_schema in OUTPUT_SCHEMAS
    assert all(source.startswith("https://") for source in METHODOLOGY_SOURCES)


def test_registry_covers_all_stage_10_method_families() -> None:
    expected = {
        "descriptive_statistics", "frequency_table", "crosstab", "magnitude_table",
        "descriptive_confidence_interval", "t_test", "nonparametric_test",
        "proportion_test", "anova", "ancova", "repeated_measures_test",
        "multiple_testing_correction", "linear_regression", "logistic_regression",
        "probit_regression", "poisson_regression", "negative_binomial_regression",
        "ordinal_regression", "multinomial_regression", "zero_inflated_model", "marginal_effects",
        "spline_regression", "linear_mixed_effects", "generalized_mixed_effects",
        "growth_curve", "gee", "panel_fixed_effects", "panel_random_effects",
        "kaplan_meier", "cox_proportional_hazards", "competing_risks",
        "recurrent_events", "time_varying_survival", "matching",
        "propensity_weighting", "difference_in_differences", "staggered_adoption",
        "event_study", "synthetic_control", "regression_discontinuity",
        "instrumental_variables", "treatment_effect_heterogeneity",
        "causal_sensitivity", "survey_mean", "survey_proportion", "survey_regression",
        "missingness_pattern", "single_imputation", "multiple_imputation",
        "mnar_sensitivity", "stationarity_diagnostic", "seasonal_decomposition",
        "arima", "exponential_smoothing", "interrupted_time_series",
        "forecast_backtest", "pca", "exploratory_factor_analysis",
        "confirmatory_factor_analysis", "reliability", "measurement_invariance",
        "clustering", "latent_class", "predictive_regression",
        "predictive_classification", "probability_calibration",
        "geospatial_analysis", "network_analysis", "text_analysis",
        "bayesian_model", "power_precision", "simulation_design",
    }
    assert expected <= set(METHODS)
    assert {"variance", "fixed_effects", "marginal_effects", "multiplicity",
            "survey_design", "missing_data", "prediction", "time_series", "influence"} <= set(CROSS_CUTTING_CONTROLS)


def test_research_spec_requires_every_material_choice() -> None:
    result = validate_research_specification({"research_question": "Does x affect y?"})
    assert result["valid"] is False
    joined = " ".join(result["clarifications"]).lower()
    for phrase in ("unit of analysis", "target population", "estimand", "study design",
                   "missing-data", "repeated measures", "primary sampling"):
        assert phrase in joined


def test_valid_method_contract_and_goal_compatibility() -> None:
    result = evaluate_method("linear_regression", _complete_spec())
    assert result["valid"] is True
    assert result["contract"]["output_schema"] == "method_result_v1"
    wrong = evaluate_method("matching", _complete_spec())
    assert wrong["valid"] is False
    assert any("does not support" in item for item in wrong["clarifications"])


def test_causal_and_predictive_specs_require_design_information() -> None:
    causal = _complete_spec(goal="causal", treatment=None, exposures=[], time_ordering="")
    result = evaluate_method("difference_in_differences", causal)
    assert result["valid"] is False
    assert any("treatment" in item.lower() for item in result["clarifications"])
    predictive = _complete_spec(goal="predictive", target="y", split_strategy=None)
    result = evaluate_method("predictive_classification", predictive)
    assert result["valid"] is False
    assert any("split" in item.lower() for item in result["clarifications"])


def test_required_roles_reject_explicit_none_placeholders() -> None:
    survey = _complete_spec(goal="descriptive", weights="none")
    result = evaluate_method("survey_mean", survey)
    assert result["valid"] is False
    assert "Identify the weights required by Survey-adjusted mean." in result["clarifications"]

    predictive = _complete_spec(
        goal="predictive", target="y", split_strategy="not applicable",
    )
    result = evaluate_method("predictive_regression", predictive)
    assert result["valid"] is False
    assert any("split strategy" in item for item in result["clarifications"])


def test_recommendations_report_complete_candidate_inventory_without_silent_cap() -> None:
    spec = _complete_spec(
        goal="inferential", event="event", subject_id="subject", panel_id="subject",
        time="time", weights="sampling_weight", indicators="i1,i2,i3",
        instrument="instrument", running_variable="score", cutoff=1.5,
        crs="EPSG:4326", nodes="nodes", edges="edges", text="text",
        group="group", target="target", split_strategy="cross-validation",
    )
    result = recommend_methods(spec, limit=len(METHODS))
    assert result["valid"] is True
    assert result["truncated"] is False
    assert result["candidate_count"] == len(result["candidates"])
    assert result["candidate_count"] > 50


def test_conditional_methods_are_not_silently_offered() -> None:
    spec = _complete_spec(
        goal="inferential", indicators=["i1", "i2", "i3"],
        outcome="class", missing_data_assumption="MAR",
    )
    result = evaluate_method("latent_class", spec)
    assert result["valid"] is False
    assert any("Conditional support" in item for item in result["clarifications"])
    spec["conditional_support_confirmed"] = True
    assert evaluate_method("latent_class", spec)["valid"] is True


def test_domain_flags_keep_conditional_domains_explicit() -> None:
    assert DOMAIN_FLAGS["bayesian"]["availability"] == "conditional"
    assert "R-hat" in DOMAIN_FLAGS["bayesian"]["condition"]
    assert METHODS["bayesian_model"].availability == "conditional"
    assert DOMAIN_FLAGS["bayesian"]["condition"] == METHODS["bayesian_model"].condition
    assert DOMAIN_FLAGS["latent_class"]["availability"] == "conditional"


def test_recommendations_exclude_conditional_methods_and_respect_goal() -> None:
    result = recommend_methods(_complete_spec())
    assert result["valid"] is True
    ids = {row["id"] for row in result["candidates"]}
    assert "linear_regression" in ids
    assert "latent_class" not in ids
    assert "bayesian_model" not in ids
    assert all(METHODS[method_id].availability == "supported" for method_id in ids)


def test_bayesian_method_requires_explicit_conditional_support_confirmation() -> None:
    spec = _complete_spec(goal="inferential")
    result = evaluate_method("bayesian_model", spec)
    assert result["valid"] is False
    assert any("Conditional support" in item for item in result["clarifications"])
    spec["conditional_support_confirmed"] = True
    assert evaluate_method("bayesian_model", spec)["valid"] is True


def test_validate_methodology_tool_can_recommend_before_selection() -> None:
    from sift.tools import validate_methodology
    response = asyncio.run(validate_methodology.handler({
        "research_specification": _complete_spec(),
    }))
    payload = json.loads(response["content"][0]["text"])
    assert payload["status"] == "ok"
    assert any(row["id"] == "linear_regression" for row in payload["candidates"])


def test_validate_methodology_tool_returns_fixed_contract() -> None:
    from sift.tools import validate_methodology
    response = asyncio.run(validate_methodology.handler({
        "method_id": "linear_regression",
        "research_specification": _complete_spec(),
    }))
    payload = json.loads(response["content"][0]["text"])
    assert payload["status"] == "ok"
    assert payload["contract"]["id"] == "linear_regression"
    assert payload["contract"]["diagnostics"]
