"""Machine-readable research-method contract and specification validator.

Sift does not reimplement statistical estimators. Generated code fits models
with maintained local R/Python/Stata libraries; this module defines when a
method may be proposed, which assumptions and aggregate diagnostics must be
reported, which reference implementation is intended, how its result crosses
the disclosure boundary, and what claim language the design supports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence


Goal = Literal["descriptive", "inferential", "associational", "predictive", "causal"]
Availability = Literal["supported", "conditional"]


@dataclass(frozen=True)
class MethodSpec:
    id: str
    family: str
    title: str
    goals: tuple[Goal, ...]
    required_roles: tuple[str, ...]
    assumptions: tuple[str, ...]
    diagnostics: tuple[str, ...]
    references: tuple[str, ...]
    output_schema: str
    claim_rule: str
    availability: Availability = "supported"
    condition: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_FAMILY: dict[str, dict[str, Any]] = {
    "descriptive": {
        "goals": ("descriptive", "inferential"), "roles": (),
        "assumptions": ("the observed sample and measurement process are defined",),
        "diagnostics": ("missingness", "effective_sample_size"),
        "schema": "aggregate_summary_v1",
        "claim": "Describe the observed sample; population claims require a defensible sampling design.",
    },
    "comparison": {
        "goals": ("inferential", "associational"), "roles": ("outcome", "exposure"),
        "assumptions": ("independent sampling or a declared dependence structure", "appropriate outcome scale"),
        "diagnostics": ("group_sample_sizes", "distribution_or_robustness_check"),
        "schema": "method_result_v1",
        "claim": "Report a group contrast, not a causal effect, unless treatment assignment identifies one.",
    },
    "regression": {
        "goals": ("associational", "inferential", "predictive"), "roles": ("outcome", "predictors"),
        "assumptions": ("model functional form is adequate", "dependence and variance structure are addressed"),
        "diagnostics": (
            "convergence", "specification", "influence", "multicollinearity",
            "heteroskedasticity", "residual_distribution",
        ),
        "schema": "method_result_v1",
        "claim": "Regression coefficients are conditional associations unless the study design identifies a causal estimand.",
    },
    "longitudinal": {
        "goals": ("associational", "inferential", "predictive"), "roles": ("outcome", "panel_id", "time"),
        "assumptions": ("within-unit dependence is modelled", "time ordering and missing waves are defined"),
        "diagnostics": ("cluster_count", "cluster_size", "serial_correlation", "convergence"),
        "schema": "method_result_v1",
        "claim": "Describe within/between-unit associations; do not imply intervention effects without identification.",
    },
    "survival": {
        "goals": ("descriptive", "associational", "inferential", "predictive"), "roles": ("time", "event"),
        "assumptions": ("censoring mechanism is stated", "subjects, records, and events are distinct"),
        "diagnostics": ("subject_count", "event_count", "at_risk_support"),
        "schema": "method_result_v1",
        "claim": "Report survival or hazard associations; hazard ratios are not risk ratios or causal effects by default.",
    },
    "causal": {
        "goals": ("causal",), "roles": ("outcome", "treatment", "estimand", "time_ordering"),
        "assumptions": ("the design-specific identifying assumptions are stated", "positivity and interference are addressed"),
        "diagnostics": ("overlap", "balance", "design_specific_falsification"),
        "schema": "method_result_v1",
        "claim": "Use causal language only for the declared estimand, population, and design under stated identifying assumptions.",
    },
    "survey": {
        "goals": ("descriptive", "inferential", "associational"), "roles": ("weights",),
        "assumptions": ("weights and sampling stages correspond to the target population",),
        "diagnostics": ("weight_distribution", "design_effect", "effective_sample_size"),
        "schema": "method_result_v1",
        "claim": "Population claims are limited to the frame and design represented by the supplied weights.",
    },
    "missing_data": {
        "goals": ("descriptive", "inferential", "associational", "predictive", "causal"), "roles": ("missing_data_assumption",),
        "assumptions": ("MCAR, MAR, or MNAR is stated and justified", "the imputation model preserves the analysis structure"),
        "diagnostics": ("missingness_pattern", "imputation_convergence", "between_imputation_variance"),
        "schema": "method_result_v1",
        "claim": "Inference is conditional on the declared missing-data mechanism and imputation specification.",
    },
    "time_series": {
        "goals": ("descriptive", "inferential", "predictive", "causal"), "roles": ("outcome", "time"),
        "assumptions": ("observations are correctly ordered and cadence is defined", "dependence and structural breaks are addressed"),
        "diagnostics": ("stationarity", "residual_autocorrelation", "rolling_origin_backtest"),
        "schema": "method_result_v1",
        "claim": "Forecast claims apply to the evaluated horizon and regime; interrupted-series causal claims require a credible counterfactual.",
    },
    "measurement": {
        "goals": ("descriptive", "inferential", "predictive"), "roles": ("indicators",),
        "assumptions": ("indicator scale and missingness are appropriate", "sample support is adequate for model complexity"),
        "diagnostics": ("sampling_adequacy", "fit_or_stability", "component_or_class_support"),
        "schema": "method_result_v1",
        "claim": "Components, factors, and classes are model-dependent summaries, not observed natural kinds.",
    },
    "predictive": {
        "goals": ("predictive",), "roles": ("target", "predictors", "split_strategy"),
        "assumptions": ("evaluation data are independent of fitting and tuning", "deployment population and loss are defined"),
        "diagnostics": ("held_out_performance", "baseline_comparison", "calibration", "split_integrity"),
        "schema": "method_result_v1",
        "claim": "Report out-of-sample prediction performance; feature importance is not a causal effect.",
    },
    "domain": {
        "goals": ("descriptive", "inferential", "associational", "predictive"), "roles": (),
        "assumptions": ("domain structure and sampling process are represented",),
        "diagnostics": ("domain_specific_validity", "sensitivity"),
        "schema": "method_result_v1",
        "claim": "Limit claims to the declared domain representation and validated sampling assumptions.",
    },
    "bayesian": {
        "goals": ("descriptive", "inferential", "associational", "predictive", "causal"), "roles": ("outcome",),
        "assumptions": ("priors and likelihood are justified", "posterior computation has converged"),
        "diagnostics": ("rhat", "bulk_ess", "tail_ess", "divergences", "posterior_predictive_check"),
        "schema": "method_result_v1",
        "claim": "Posterior claims are conditional on the model, priors, data, and satisfactory computation diagnostics.",
    },
    "design": {
        "goals": ("descriptive", "inferential", "predictive", "causal"), "roles": ("estimand",),
        "assumptions": ("effect scale, uncertainty target, and design constraints are declared",),
        "diagnostics": ("simulation_monte_carlo_error", "scenario_sensitivity"),
        "schema": "method_result_v1",
        "claim": "Power and precision are prospective design properties under stated effect and variance assumptions.",
    },
}


def _m(
    method_id: str, family: str, title: str, references: Sequence[str],
    *, roles: Sequence[str] = (), assumptions: Sequence[str] = (),
    diagnostics: Sequence[str] = (), availability: Availability = "supported",
    condition: str | None = None,
    diagnostic_contract: Sequence[str] | None = None,
    output_schema: str | None = None,
    claim_rule: str | None = None,
) -> MethodSpec:
    base = _FAMILY[family]
    return MethodSpec(
        id=method_id, family=family, title=title,
        goals=tuple(base["goals"]),
        required_roles=tuple(dict.fromkeys((*base["roles"], *roles))),
        assumptions=tuple(dict.fromkeys((*base["assumptions"], *assumptions))),
        diagnostics=tuple(dict.fromkeys(
            diagnostic_contract
            if diagnostic_contract is not None
            else (*base["diagnostics"], *diagnostics)
        )),
        references=tuple(references), output_schema=output_schema or base["schema"],
        claim_rule=claim_rule or base["claim"], availability=availability,
        condition=condition,
    )


_METHODS = (
    # Descriptive and inferential
    _m("descriptive_statistics", "descriptive", "Descriptive statistics", ("pandas.DataFrame.describe", "R summary")),
    _m("frequency_table", "descriptive", "Disclosure-controlled frequency table", ("pandas.Series.value_counts", "R table")),
    _m("crosstab", "descriptive", "Cross-tabulation", ("pandas.crosstab", "R xtabs")),
    _m("magnitude_table", "descriptive", "Magnitude table", ("pandas.DataFrame.groupby", "R aggregate")),
    _m(
        "descriptive_confidence_interval", "descriptive",
        "Confidence interval for descriptive quantity",
        ("statsmodels.stats.weightstats.DescrStatsW.tconfint_mean", "R stats::t.test"),
        roles=("outcome",),
        diagnostic_contract=("missingness", "effective_sample_size", "confidence_level"),
        output_schema="method_result_v1",
    ),
    _m("t_test", "comparison", "t-test", ("scipy.stats.ttest_ind/ttest_rel/ttest_1samp", "R t.test")),
    _m(
        "nonparametric_test", "comparison", "Rank-based non-parametric test",
        ("scipy.stats.mannwhitneyu/wilcoxon/kruskal", "R stats::wilcox.test/kruskal.test"),
        diagnostic_contract=("group_sample_sizes", "ties_and_zero_differences"),
    ),
    _m(
        "proportion_test", "comparison", "Proportion test",
        ("statsmodels.stats.proportion.proportions_ztest", "R stats::prop.test"),
        diagnostic_contract=("group_sample_sizes", "expected_cell_counts"),
    ),
    _m(
        "anova", "comparison", "ANOVA",
        ("statsmodels.stats.anova.anova_lm", "R stats::aov"),
        diagnostic_contract=("group_sample_sizes", "residual_distribution", "homogeneity_of_variance"),
    ),
    _m(
        "ancova", "comparison", "ANCOVA",
        ("statsmodels.stats.anova.anova_lm", "R stats::lm/aov"), roles=("predictors",),
        diagnostic_contract=(
            "group_sample_sizes", "residual_distribution",
            "homogeneity_of_variance", "parallel_slopes",
        ),
    ),
    _m(
        "repeated_measures_test", "longitudinal", "Repeated-measures test",
        ("statsmodels.stats.anova.AnovaRM", "R stats::friedman.test/nlme::lme"),
        diagnostic_contract=(
            "cluster_count", "cluster_size", "complete_cases",
            "sphericity_or_correction",
        ),
    ),
    _m(
        "multiple_testing_correction", "comparison", "Multiple-testing correction",
        ("statsmodels.stats.multitest.multipletests", "R stats::p.adjust"), roles=(),
        diagnostic_contract=("hypothesis_family", "correction_applied"),
    ),
    # Regression
    _m("linear_regression", "regression", "Linear regression", ("statsmodels.OLS", "R lm", "Stata regress")),
    _m("logistic_regression", "regression", "Logistic regression", ("statsmodels.Logit/GLM Binomial", "R glm(binomial)", "Stata logit")),
    _m("probit_regression", "regression", "Probit regression", ("statsmodels.Probit", "R glm(probit)", "Stata probit")),
    _m("poisson_regression", "regression", "Poisson regression", ("statsmodels.Poisson/GLM Poisson", "R glm(poisson)", "Stata poisson"), diagnostics=("overdispersion",)),
    _m("negative_binomial_regression", "regression", "Negative-binomial regression", ("statsmodels.NegativeBinomial", "R MASS::glm.nb", "Stata nbreg"), diagnostics=("overdispersion",)),
    _m("ordinal_regression", "regression", "Ordinal regression", ("statsmodels.miscmodels.OrderedModel", "R MASS::polr", "R ordinal::clm", "Stata ologit"), diagnostics=("proportional_odds",)),
    _m("multinomial_regression", "regression", "Multinomial regression", ("statsmodels.MNLogit", "R nnet::multinom", "Stata mlogit"), diagnostics=("class_support",)),
    _m("zero_inflated_model", "regression", "Zero-inflated count model", ("statsmodels.ZeroInflatedPoisson/ZeroInflatedNegativeBinomialP", "R pscl::zeroinfl", "Stata zip/zinb"), diagnostics=("zero_process_specification", "overdispersion")),
    _m("spline_regression", "regression", "Spline/non-linear regression", ("patsy.bs + statsmodels", "R splines::ns/bs", "Stata mkspline"), diagnostics=("degrees_of_freedom_sensitivity",)),
    _m("marginal_effects", "regression", "Marginal effects", ("statsmodels get_margeff", "R marginaleffects", "Stata margins"), diagnostics=("estimand_scale", "uncertainty_propagation")),
    # Hierarchical/longitudinal
    _m("linear_mixed_effects", "longitudinal", "Linear mixed-effects model", ("statsmodels.MixedLM", "R lme4::lmer", "Stata mixed")),
    _m("generalized_mixed_effects", "longitudinal", "Generalized mixed-effects model", ("R lme4::glmer", "Stata melogit/mepoisson")),
    _m("growth_curve", "longitudinal", "Growth-curve model", ("statsmodels.MixedLM", "R lme4::lmer", "Stata mixed"), diagnostics=("random_effect_structure",)),
    _m("gee", "longitudinal", "Generalized estimating equations", ("statsmodels.GEE", "R geepack::geeglm", "Stata xtgee"), diagnostics=("working_correlation_sensitivity",)),
    _m(
        "panel_fixed_effects", "longitudinal", "Panel fixed-effects model",
        ("statsmodels OLS on entity-within transformed data with entity-clustered covariance", "linearmodels.PanelOLS", "R fixest::feols", "Stata xtreg, fe"),
        diagnostic_contract=(
            "cluster_count", "cluster_size", "convergence",
            "balanced_panel", "within_variation", "fixed_effect_test",
            "clustered_uncertainty",
        ),
        claim_rule=(
            "Coefficients are entity fixed-effects associations identified only "
            "by within-entity change in the supplied balanced panel; they are not "
            "causal effects without a separately justified design."
        ),
    ),
    _m("panel_random_effects", "longitudinal", "Panel random-effects model", ("statsmodels.MixedLM random-intercept model + statsmodels OLS Hausman comparison", "linearmodels.RandomEffects", "R plm::plm", "Stata xtreg, re"), diagnostics=("hausman",)),
    # Survival
    _m("kaplan_meier", "survival", "Kaplan–Meier analysis", ("lifelines.KaplanMeierFitter", "R survival::survfit", "Stata sts")),
    _m("cox_proportional_hazards", "survival", "Cox proportional-hazards model", ("statsmodels.PHReg/lifelines.CoxPHFitter", "R survival::coxph", "Stata stcox"), roles=("predictors",), diagnostics=("proportional_hazards",)),
    _m("competing_risks", "survival", "Competing-risks model", ("statsmodels.duration.survfunc.CumIncidenceRight", "lifelines.AalenJohansenFitter", "R cmprsk::crr", "Stata stcrreg"), diagnostics=("cause_specific_events",)),
    _m("recurrent_events", "survival", "Recurrent-event model", ("statsmodels.PHReg counting-process entry + clustered covariance", "R survival::coxph counting-process", "Stata stcox counting-process"), roles=("subject_id",), diagnostics=("within_subject_dependence", "proportional_hazards")),
    _m("time_varying_survival", "survival", "Time-varying covariate survival model", ("statsmodels.PHReg counting-process entry", "lifelines.CoxTimeVaryingFitter", "R survival::coxph counting-process", "Stata stsplit/stcox"), roles=("subject_id",), diagnostics=("interval_integrity", "proportional_hazards")),
    # Causal inference
    _m(
        "matching", "causal", "Propensity-score nearest-neighbour matching",
        ("sklearn.linear_model.LogisticRegression + sklearn.neighbors.NearestNeighbors", "R stats::glm + nearest propensity neighbour"),
        diagnostic_contract=(
            "propensity_overlap", "standardized_mean_differences",
            "effective_matched_sample", "effect_uncertainty",
            "design_specific_falsification",
        ),
    ),
    _m(
        "propensity_weighting", "causal", "Propensity-score weighting",
        ("statsmodels.discrete.discrete_model.Logit", "R stats::glm(binomial)"),
        diagnostic_contract=(
            "propensity_overlap", "weight_extremes",
            "standardized_mean_differences", "effective_sample_size",
            "effect_uncertainty", "design_specific_falsification",
        ),
    ),
    _m(
        "difference_in_differences", "causal", "Difference-in-differences",
        ("statsmodels OLS two-by-two interaction with entity-clustered covariance", "R fixest::feols", "Stata xtdidregress"),
        roles=("panel_id", "time"),
        diagnostic_contract=(
            "parallel_pretrends", "treatment_timing",
            "balanced_two_period_panel", "clustered_uncertainty",
            "effect_uncertainty", "design_specific_falsification",
        ),
        claim_rule=(
            "The reported two-period ATT is causal only if untreated potential-outcome "
            "trends would have been parallel. A single pre-period cannot test that "
            "assumption, so external design justification is required."
        ),
    ),
    _m("staggered_adoption", "causal", "Staggered-adoption DiD", ("R did::att_gt", "R fixest::sunab", "Stata csdid"), roles=("panel_id", "time"), diagnostics=("cohort_support", "parallel_pretrends")),
    _m("event_study", "causal", "Event study", ("R fixest::sunab", "Stata eventstudyinteract"), roles=("panel_id", "time"), diagnostics=("reference_period", "parallel_pretrends")),
    _m(
        "synthetic_control", "causal", "Synthetic control",
        ("scipy.optimize.minimize constrained donor weights", "R stats::optim simplex donor weights"),
        roles=("panel_id", "time"),
        diagnostic_contract=(
            "pre_treatment_fit", "placebo_distribution",
            "donor_weight_concentration", "effect_uncertainty",
            "design_specific_falsification",
        ),
    ),
    _m("regression_discontinuity", "causal", "Regression discontinuity", ("Python rdrobust", "R rdrobust", "Stata rdrobust"), roles=("running_variable", "cutoff"), diagnostics=("bandwidth_sensitivity", "density_manipulation", "covariate_continuity")),
    _m("instrumental_variables", "causal", "Instrumental variables", ("linearmodels.IV2SLS", "R AER::ivreg", "Stata ivregress"), roles=("instrument",), diagnostics=("first_stage_strength", "overidentification", "endogeneity")),
    _m(
        "treatment_effect_heterogeneity", "causal", "Treatment-effect heterogeneity",
        ("sklearn.ensemble.RandomForestRegressor honest T-learner", "R rpart honest T-learner"),
        diagnostic_contract=(
            "propensity_overlap", "standardized_mean_differences",
            "honest_sample_splitting", "subgroup_multiplicity",
            "heterogeneity_calibration", "effect_uncertainty",
            "design_specific_falsification",
        ),
    ),
    _m(
        "causal_sensitivity", "causal", "Omitted-variable sensitivity analysis",
        ("Cinelli-Hazlett partial-R2 robustness value from statsmodels fit", "Cinelli-Hazlett partial-R2 robustness value from R stats::lm"),
        diagnostic_contract=(
            "robustness_value", "assumption_grid", "design_specific_falsification",
        ),
    ),
    # Survey
    _m("survey_mean", "survey", "Survey-adjusted mean", ("Python Taylor linearization/replicate-weight estimator", "R survey::svymean", "Stata svy: mean"), roles=("outcome",), diagnostics=("strata_psu_support", "variance_estimator", "lonely_psu")),
    _m("survey_proportion", "survey", "Survey-adjusted proportion", ("Python Taylor linearization/replicate-weight estimator", "R survey::svymean", "Stata svy: proportion"), roles=("outcome",), diagnostics=("strata_psu_support", "variance_estimator", "lonely_psu")),
    _m("survey_regression", "survey", "Survey-adjusted regression", ("Python probability-weighted estimating-equation sandwich", "R survey::svyglm", "Stata svy: regress/glm"), roles=("outcome", "predictors"), diagnostics=("strata_psu_support", "variance_estimator", "lonely_psu")),
    # Missing data
    _m(
        "missingness_pattern", "missing_data", "Missingness-pattern analysis",
        (
            "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html",
            "https://amices.org/mice/reference/md.pattern.html",
        ),
        roles=(),
        diagnostic_contract=(
            "missingness_pattern", "complete_case_rate", "complete_case_warning",
        ),
        claim_rule=(
            "Describe observed missingness only. Patterns do not identify MCAR, MAR, "
            "or MNAR and complete-case representativeness is not established."
        ),
    ),
    _m(
        "single_imputation", "missing_data", "Principled single imputation",
        (
            "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html",
            "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/median.html",
        ),
        availability="conditional",
        condition=(
            "Only deterministic nuisance-covariate or prediction preprocessing "
            "cases; never default inferential uncertainty."
        ),
        diagnostic_contract=(
            "missingness_pattern", "imputation_scope",
            "inferential_uncertainty_not_claimed",
        ),
        claim_rule=(
            "Single imputation is limited to the declared preprocessing scope. "
            "Do not use its output as inferential uncertainty or evidence that "
            "missing-data uncertainty was propagated."
        ),
    ),
    _m(
        "multiple_imputation", "missing_data", "Multiple imputation with Rubin pooling",
        (
            "https://www.statsmodels.org/stable/imputation.html",
            "https://amices.org/mice/",
        ),
        diagnostic_contract=(
            "missingness_pattern", "imputation_trace_stability",
            "between_imputation_variance", "seed_recorded",
            "fraction_missing_information", "rubin_pooling",
        ),
        claim_rule=(
            "Pooled inference is conditional on MAR and the declared chained-equation "
            "specification; diagnostics cannot prove the imputation model is correct."
        ),
    ),
    _m(
        "mnar_sensitivity", "missing_data", "MNAR sensitivity analysis",
        (
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6021473/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC5860630/",
        ),
        assumptions=(
            "the delta scale and range are scientifically justified rather than estimated from observed data",
        ),
        diagnostic_contract=(
            "delta_grid", "baseline_included", "conclusion_stability",
            "sensitivity_parameter_justification",
        ),
        claim_rule=(
            "Report how pooled conclusions vary over the declared MNAR delta grid. "
            "The data do not identify the sensitivity parameter, and stability over "
            "the grid does not prove MAR or MNAR."
        ),
    ),
    # Time series
    _m(
        "stationarity_diagnostic", "time_series", "Stationarity diagnostics",
        ("statsmodels.tsa.stattools.adfuller/kpss", "R urca::ur.df/ur.kpss (when installed)"), roles=(),
        diagnostic_contract=("temporal_order", "regular_frequency", "missingness", "stationarity_consensus"),
    ),
    _m(
        "seasonal_decomposition", "time_series", "Trend/seasonality decomposition",
        ("statsmodels.tsa.seasonal.STL", "R stats::stl"),
        diagnostic_contract=("temporal_order", "regular_frequency", "period_support", "residual_share"),
    ),
    _m(
        "arima", "time_series", "ARIMA-family model",
        ("statsmodels.tsa.arima.model.ARIMA", "R stats::arima/predict.Arima"),
        diagnostic_contract=(
            "temporal_order", "regular_frequency", "stationarity",
            "ar_stationarity", "ma_invertibility", "residual_autocorrelation", "holdout_leakage",
            "prediction_interval_coverage",
        ),
    ),
    _m(
        "exponential_smoothing", "time_series", "Exponential smoothing",
        ("statsmodels.tsa.exponential_smoothing.ets.ETSModel", "R stats::HoltWinters/predict.HoltWinters"),
        diagnostic_contract=(
            "temporal_order", "regular_frequency", "residual_autocorrelation",
            "holdout_leakage", "prediction_interval_coverage",
        ),
    ),
    _m(
        "interrupted_time_series", "time_series", "Interrupted time-series analysis",
        ("statsmodels.tsa.statespace.SARIMAX", "R stats::arima with segmented xreg"), roles=("treatment",),
        diagnostic_contract=(
            "temporal_order", "regular_frequency", "pre_intervention_trend",
            "intervention_timing", "residual_autocorrelation",
            "design_specific_falsification",
        ),
    ),
    _m(
        "forecast_backtest", "time_series", "Rolling-origin forecast backtest",
        ("statsmodels ARIMA expanding-window refits", "R stats::arima expanding-window refits"),
        diagnostic_contract=(
            "temporal_order", "regular_frequency", "rolling_origin_backtest",
            "holdout_leakage", "prediction_interval_coverage", "baseline_comparison",
        ),
    ),
    # Measurement/dimension reduction
    _m("pca", "measurement", "Principal component analysis", ("sklearn.decomposition.PCA", "R prcomp", "Stata pca")),
    _m("exploratory_factor_analysis", "measurement", "Exploratory factor analysis", ("factor_analyzer.FactorAnalyzer", "R psych::fa", "Stata factor"), diagnostics=("kmo", "bartlett_sphericity")),
    _m("confirmatory_factor_analysis", "measurement", "Confirmatory factor analysis", ("R lavaan::cfa/fitMeasures", "Stata sem estat gof"), diagnostics=("cfi", "tli", "rmsea", "srmr")),
    _m(
        "reliability", "measurement", "Reliability analysis",
        ("factor_analyzer.FactorAnalyzer + standardized Cronbach alpha bootstrap", "R psych::alpha/omega", "Stata alpha"),
        diagnostics=("item_count", "omega_or_alpha_interval", "item_direction"),
    ),
    _m("measurement_invariance", "measurement", "Measurement invariance", ("R lavaan::cfa", "Stata sem group()"), roles=("group",), diagnostics=("configural_fit", "metric_change", "scalar_change")),
    _m("clustering", "measurement", "Clustering", ("sklearn.cluster", "R stats::kmeans/hclust", "Stata cluster"), diagnostics=("silhouette", "stability", "cluster_sizes")),
    _m("latent_class", "measurement", "Latent-class model", ("R poLCA", "Stata gsem, lclass"), diagnostics=("class_sizes", "entropy", "solution_stability"), availability="conditional", condition="Offer only when expected class sizes, normalized entropy, and multi-start solution stability are adequate."),
    # Predictive workflows
    _m(
        "predictive_regression", "predictive", "Predictive regression workflow",
        (
            "https://scikit-learn.org/stable/modules/compose.html#pipeline",
            "https://scikit-learn.org/stable/modules/cross_validation.html",
            "https://workflows.tidymodels.org/",
        ),
        diagnostic_contract=(
            "held_out_performance", "baseline_comparison", "calibration",
            "split_integrity", "preprocessing_inside_split", "uncertainty",
        ),
    ),
    _m(
        "predictive_classification", "predictive", "Predictive classification workflow",
        (
            "https://scikit-learn.org/stable/modules/calibration.html",
            "https://scikit-learn.org/stable/modules/model_evaluation.html",
            "https://probably.tidymodels.org/",
        ),
        diagnostic_contract=(
            "held_out_performance", "baseline_comparison", "calibration",
            "split_integrity", "preprocessing_inside_split", "discrimination",
            "class_balance", "uncertainty",
        ),
    ),
    _m(
        "probability_calibration", "predictive", "Probability calibration",
        ("sklearn.calibration.CalibratedClassifierCV", "sklearn.metrics.brier_score_loss", "R probably"),
        diagnostic_contract=(
            "held_out_performance", "baseline_comparison", "calibration",
            "calibration_curve", "brier_score", "split_integrity",
            "preprocessing_inside_split", "calibration_nested",
            "class_balance", "uncertainty",
        ),
        claim_rule=(
            "Calibration claims apply only to held-out probability forecasts from "
            "the fitted workflow in the represented population; they do not establish "
            "classification utility, transportability, or causal effects."
        ),
    ),
    # Domain/design
    _m("geospatial_analysis", "domain", "Geospatial analysis", ("GeoPandas/Shapely + SciPy cKDTree", "R sf/spdep"), roles=("crs",),
       diagnostic_contract=("crs_validity", "spatial_weights", "spatial_autocorrelation", "privacy_aggregation")),
    _m("network_analysis", "domain", "Network/graph analysis", ("SciPy sparse.csgraph", "R igraph"), roles=("nodes", "edges"),
       diagnostic_contract=("graph_definition", "graph_symmetry", "dependence_aware_uncertainty", "privacy_aggregation")),
    _m("text_analysis", "domain", "Text analysis", ("scikit-learn TfidfVectorizer/KMeans", "R quanteda"), roles=("text",),
       diagnostic_contract=("tokenization_specification", "held_out_or_stability_check", "document_privacy", "vocabulary_privacy")),
    _m(
        "bayesian_model", "bayesian", "Bayesian model",
        ("CmdStanPy/PyMC", "R cmdstanr/brms/rstan"),
        availability="conditional",
        condition=(
            "Offer only when R-hat, bulk/tail effective sample sizes, "
            "divergences, and posterior-predictive diagnostics will be "
            "reported and validated."
        ),
    ),
    _m("power_precision", "design", "Power and precision calculation", ("statsmodels.stats.power.TTestIndPower", "R stats::power.t.test", "Stata power"),
       diagnostic_contract=("effect_size_scenarios", "alpha_and_power", "prospective_design")),
    _m("simulation_design", "design", "Simulation-based design evaluation", ("NumPy/SciPy/statsmodels", "R stats::t.test/power.t.test", "Stata simulate"),
       diagnostic_contract=("seed_recorded", "replication_count", "monte_carlo_standard_error", "scenario_sensitivity")),
)


METHODS: dict[str, MethodSpec] = {method.id: method for method in _METHODS}
# Version 2 makes Bayesian availability consistently conditional in both the
# method registry and the domain flags.  Clients caching method recommendations
# must not reuse v1 recommendations, which could silently offer Bayesian work
# before its mandatory computation diagnostics were confirmed.
METHODOLOGY_REGISTRY_VERSION = 2

# Primary maintained documentation consulted for the registry contract.
METHODOLOGY_SOURCES: tuple[str, ...] = (
    "https://www.statsmodels.org/stable/mixed_linear.html",
    "https://www.statsmodels.org/stable/gee.html",
    "https://www.statsmodels.org/stable/duration.html",
    "https://www.statsmodels.org/stable/generated/statsmodels.duration.hazard_regression.PHReg.html",
    "https://www.statsmodels.org/stable/discretemod.html",
    "https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html",
    "https://scikit-learn.org/stable/modules/cross_validation.html",
    "https://scikit-learn.org/stable/modules/compose.html",
    "https://scikit-learn.org/stable/modules/calibration.html",
    "https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svydesign.html",
    "https://r-survey.r-forge.r-project.org/pkgdown/docs/reference/svyglm.html",
    "https://amices.org/mice/reference/mice.html",
    "https://amices.org/mice/reference/pool.html",
    "https://bcallaway11.github.io/did/reference/att_gt.html",
    "https://rdpackages.github.io/rdrobust/",
    "https://bashtage.github.io/linearmodels/iv/iv/linearmodels.iv.results.FirstStageResults.diagnostics.html",
    "https://mc-stan.org/learn-stan/diagnostics-warnings.html",
)

OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "aggregate_summary_v1": {
        "result_type": ("descriptive", "frequency_table", "crosstab", "magnitude_table"),
        "privacy": "typed aggregate only; cell suppression and dominance rules apply",
    },
    "method_result_v1": {
        "result_type": "method_result",
        "required": ("method_id", "n", "diagnostics"),
        "optional_numeric_maps": (
            "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper", "metrics",
        ),
        "forbidden": (
            "rows", "observations", "predictions", "residuals", "fitted_values",
            "influence_by_case", "posterior_draws", "imputed_datasets",
        ),
        "privacy": "aggregate maps capped at 100 identifier-shaped keys; no observation-level arrays",
    },
}


CROSS_CUTTING_CONTROLS: dict[str, dict[str, Any]] = {
    "variance": {"options": ("classical", "heteroskedasticity_robust", "cluster_robust"),
                 "requires": ("declared dependence unit",)},
    "fixed_effects": {"requires": ("within-unit variation", "absorbed-variable reporting")},
    "marginal_effects": {"requires": ("scale and averaging estimand", "uncertainty propagation")},
    "multiplicity": {"options": ("holm", "bonferroni", "benjamini_hochberg"),
                     "requires": ("declared hypothesis family",)},
    "survey_design": {"roles": ("weights", "strata", "psu", "fpc", "replicate_weights")},
    "missing_data": {"requires": ("complete-case impact warning", "seed and imputation specification")},
    "prediction": {"requires": ("train_validation_test separation", "pipeline-fit only on training folds",
                                "simple baseline", "calibration/discrimination as applicable")},
    "time_series": {"forbids": ("random train/test split",), "requires": ("rolling-origin evaluation",)},
    "influence": {"privacy": "aggregate counts and maxima only; never observation identifiers or rows"},
}


DOMAIN_FLAGS: dict[str, dict[str, Any]] = {
    "geospatial": {"methods": ("geospatial_analysis",), "availability": "supported"},
    "network": {"methods": ("network_analysis",), "availability": "supported"},
    "text": {"methods": ("text_analysis",), "availability": "supported"},
    "bayesian": {"methods": ("bayesian_model",), "availability": "conditional",
                 "condition": METHODS["bayesian_model"].condition},
    "latent_class": {"methods": ("latent_class",), "availability": "conditional",
                     "condition": METHODS["latent_class"].condition},
}


_SPEC_KEYS = frozenset({
    "research_question", "unit_of_analysis", "outcome", "exposures", "treatment",
    "predictors", "controls", "target_population", "estimand", "study_design", "goal",
    "repeated_measures", "clusters", "weights", "strata", "time_ordering",
    "missing_data_assumption", "panel_id", "time", "event", "subject_id", "target",
    "split_strategy", "indicators", "instrument", "running_variable", "cutoff", "crs",
    "nodes", "edges", "text", "group",
    "psu", "fpc", "replicate_weights",
    "conditional_support_confirmed",
})


def _present(value: Any) -> bool:
    return bool(value.strip()) if isinstance(value, str) else bool(value)


def _role_present(value: Any) -> bool:
    """Return whether a value identifies an actual analysis role.

    The full research specification deliberately accepts explicit declarations
    such as ``"none"`` for optional fields.  A method-required role is
    different: placeholders such as ``none``/``not applicable`` cannot prove
    that weights, an outcome, an instrument, or another required variable was
    supplied.
    """
    if isinstance(value, str):
        normalized = " ".join(value.strip().lower().replace("_", " ").split())
        return bool(normalized) and normalized not in {
            "none", "n/a", "na", "not applicable", "not available",
            "unknown", "unspecified",
        }
    return bool(value)


def validate_research_specification(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate the choices that can materially change a research method."""
    if not isinstance(raw, Mapping):
        return {"valid": False, "clarifications": ["Provide a structured research specification."]}
    spec = {str(key): value for key, value in raw.items() if str(key) in _SPEC_KEYS}
    clarifications: list[str] = []
    for key, prompt in (
        ("research_question", "State the research question."),
        ("unit_of_analysis", "Identify the unit of analysis."),
        ("target_population", "Identify the target population."),
        ("estimand", "State the estimand or descriptive target."),
        ("study_design", "Identify the study design."),
        ("goal", "Classify the goal as descriptive, inferential, associational, predictive, or causal."),
        ("missing_data_assumption", "State the missing-data assumption."),
    ):
        if not _present(spec.get(key)):
            clarifications.append(prompt)
    for key, prompt in (
        ("exposures", "Identify exposures/treatments, or explicitly declare none."),
        ("predictors", "Identify predictors, or explicitly declare none."),
        ("controls", "Identify control variables, or explicitly declare none."),
        ("repeated_measures", "State whether observations are repeated measures."),
        ("clusters", "Identify clustering, or explicitly declare none."),
        ("weights", "Identify weights, or explicitly declare none."),
        ("strata", "Identify strata, or explicitly declare none."),
        ("psu", "Identify primary sampling units, or explicitly declare none."),
        ("fpc", "Identify finite-population corrections, or explicitly declare none."),
        ("replicate_weights", "Identify replicate weights, or explicitly declare none."),
        ("time_ordering", "State time ordering, or explicitly state that it is not applicable."),
    ):
        if key not in spec:
            clarifications.append(prompt)
    goal = spec.get("goal")
    if goal not in {"descriptive", "inferential", "associational", "predictive", "causal"}:
        clarifications.append("Use one supported goal classification.")
    if goal != "descriptive" and not _role_present(spec.get("outcome") or spec.get("target")):
        clarifications.append("Identify the outcome or prediction target.")
    if goal == "causal":
        if not _role_present(spec.get("treatment") or spec.get("exposures")):
            clarifications.append("Identify the treatment or exposure.")
        if not _role_present(spec.get("time_ordering")):
            clarifications.append("State treatment–outcome time ordering.")
    if goal == "predictive" and not _role_present(spec.get("split_strategy")):
        clarifications.append("Declare a train/validation/test or cross-validation split strategy.")
    if _present(spec.get("repeated_measures")) and not _role_present(spec.get("panel_id") or spec.get("clusters")):
        clarifications.append("Identify the subject/cluster key for repeated measures.")
    return {"valid": not clarifications, "clarifications": list(dict.fromkeys(clarifications)), "specification": spec}


def evaluate_method(method_id: str, raw_spec: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a method contract or precise clarification requirements."""
    method = METHODS.get(str(method_id))
    if method is None:
        return {"valid": False, "method_id": str(method_id),
                "clarifications": ["Choose a method present in the Sift methodology registry."]}
    validation = validate_research_specification(raw_spec)
    spec = validation.get("specification", {})
    clarifications = list(validation.get("clarifications", []))
    if spec.get("goal") not in method.goals:
        clarifications.append(
            f"{method.title} does not support the declared {spec.get('goal')!r} goal without changing the method or goal."
        )
    role_aliases = {
        "exposure": ("exposures", "treatment"), "predictors": ("predictors", "controls", "exposures"),
        "time": ("time", "time_ordering"), "event": ("event",), "weights": ("weights",),
        "estimand": ("estimand",), "time_ordering": ("time_ordering",), "target": ("target", "outcome"),
    }
    for role in method.required_roles:
        candidates = role_aliases.get(role, (role,))
        if not any(_role_present(spec.get(candidate)) for candidate in candidates):
            clarifications.append(f"Identify the {role.replace('_', ' ')} required by {method.title}.")
    if (method.availability == "conditional" and method.condition
            and spec.get("conditional_support_confirmed") is not True):
        clarifications.append(f"Conditional support: {method.condition}")
    return {
        "valid": not clarifications, "method_id": method.id,
        "clarifications": list(dict.fromkeys(clarifications)),
        "contract": method.as_dict(),
        "cross_cutting_controls": CROSS_CUTTING_CONTROLS,
    }


def recommend_methods(raw_spec: Mapping[str, Any] | None, *, limit: int = 20) -> dict[str, Any]:
    """List compatible supported methods without silently offering conditional ones."""
    validation = validate_research_specification(raw_spec)
    if not validation.get("valid"):
        return {"valid": False, "clarifications": validation.get("clarifications", []),
                "candidates": [], "candidate_count": 0, "truncated": False}
    candidates = []
    for method_id in sorted(METHODS):
        method = METHODS[method_id]
        if method.availability != "supported":
            continue
        evaluated = evaluate_method(method_id, raw_spec)
        if evaluated.get("valid"):
            candidates.append({
                "id": method.id, "title": method.title, "family": method.family,
                "required_roles": method.required_roles,
                "diagnostics": method.diagnostics,
            })
    bounded_limit = max(1, min(100, int(limit)))
    return {
        "valid": True, "clarifications": [],
        "candidates": candidates[:bounded_limit],
        "candidate_count": len(candidates),
        "truncated": len(candidates) > bounded_limit,
    }


def validate_registry() -> tuple[str, ...]:
    errors: list[str] = []
    if len(METHODS) != len(_METHODS):
        errors.append("duplicate method id")
    for method in _METHODS:
        for field in ("goals", "assumptions", "diagnostics", "references", "output_schema", "claim_rule"):
            if not getattr(method, field):
                errors.append(f"{method.id} missing {field}")
        if method.output_schema not in OUTPUT_SCHEMAS:
            errors.append(f"{method.id} references an unknown output schema")
        if method.availability == "conditional" and not method.condition:
            errors.append(f"{method.id} conditional without condition")
    return tuple(errors)


_REGISTRY_ERRORS = validate_registry()
if _REGISTRY_ERRORS:
    raise RuntimeError(f"invalid methodology registry: {_REGISTRY_ERRORS}")
