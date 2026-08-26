"""Target-leakage heuristics and t-test power/MDE notes.

Both additions are careful, honest heuristics rather than definitive
verdicts:

1. The power note computes a FORWARD-LOOKING minimum detectable
   effect (MDE) from sample sizes alone (Cohen, 1988) -- never a
   post-hoc/observed power derived from the p-value, which the
   statistics literature treats as uninformative. It only warns on a
   NULL result whose MDE is large (the "absence of evidence" trap);
   a well-powered null or any significant result gets a "pass" with
   the same number as context.
2. The leakage-naming heuristic flags a predictor name sharing a
   SPECIFIC (non-generic) token with the response variable's name.
   It must not fire on generic modelling tokens (id, flag, score,
   count, ...) or unrelated names, and must fire on a genuine overlap
   like "will_churn" / "churn_flag".
3. The extreme-t-statistic check is an independent leakage /
   separation signal from R² -- fires on |t| > 50 regardless of
   what R² looks like.
"""

from __future__ import annotations

from sift.verification import verify_payload


def _regression(**over):
    base = {
        "type": "linear_regression",
        "n": 5000,
        "coefficients": {"(Intercept)": 1.2, "tenure": -0.4, "spend": 0.1},
        "standard_errors": {"(Intercept)": 0.1, "tenure": 0.05, "spend": 0.02},
        "response_variable": "churn",
        "predictor_variables": ["tenure", "spend"],
    }
    base.update(over)
    return base


def _ttest(**over):
    base = {
        "type": "t_test",
        "n1": 30, "n2": 30,
        "mean1": 1.0, "mean2": 1.2,
        "p_value": 0.4,
    }
    base.update(over)
    return base


def _ids(block):
    return {c["id"]: c for c in block["checks"]}


# ---------------------------------------------------------------------------
# power / MDE
# ---------------------------------------------------------------------------


def test_power_absent_without_group_sizes():
    # n1 present (so SOME check fires and the block isn't None) but
    # n2 absent -> the MDE formula needs both, so "power" must not
    # appear even though "group_size_n1" does.
    block = verify_payload({
        "type": "t_test", "n1": 5, "mean1": 1.0, "mean2": 1.2, "p_value": 0.4,
    })
    assert block is not None
    assert "power" not in _ids(block)
    assert "group_size_n1" in _ids(block)


def test_power_warns_on_underpowered_null_result():
    block = verify_payload(_ttest(n1=15, n2=15, p_value=0.6))
    c = _ids(block)["power"]
    assert c["status"] == "warn"
    assert "cannot rule out" in c["detail"]


def test_power_passes_on_well_powered_null_result():
    block = verify_payload(_ttest(n1=5000, n2=5000, p_value=0.6))
    c = _ids(block)["power"]
    assert c["status"] == "pass"
    assert "adequately powered" in c["detail"]


def test_power_passes_and_contextualizes_significant_result():
    block = verify_payload(_ttest(n1=30, n2=30, p_value=0.01))
    c = _ids(block)["power"]
    assert c["status"] == "pass"
    assert "reliably detect effects of" in c["detail"]


def test_power_mde_shrinks_with_larger_samples():
    small = verify_payload(_ttest(n1=10, n2=10, p_value=0.01))
    large = verify_payload(_ttest(n1=1000, n2=1000, p_value=0.01))
    mde_small = _ids(small)["power"]["detail"]
    mde_large = _ids(large)["power"]["detail"]
    # Extract the d≈X figures crudely and compare magnitude via the
    # known monotonic relationship instead of string-parsing exactly.
    import re
    d_small = float(re.search(r"d.(\d+\.\d+)", mde_small).group(1))
    d_large = float(re.search(r"d.(\d+\.\d+)", mde_large).group(1))
    assert d_large < d_small


def test_paired_test_power_uses_one_effective_sample() -> None:
    block = verify_payload({
        "type": "t_test", "test_type": "paired", "n1": 100,
        "mean1": 1.0, "mean2": 1.1, "p_value": 0.4,
    })
    detail = _ids(block)["power"]["detail"]
    assert "paired differences" in detail
    # (1.96 + .84) / sqrt(100) ~= .28, not the independent-groups .40.
    assert "d≈0.28" in detail


def test_one_sample_power_does_not_require_n2() -> None:
    block = verify_payload({
        "type": "t_test", "test_type": "one_sample", "n1": 100,
        "mean1": 1.0, "p_value": 0.4,
    })
    assert "one-sample" in _ids(block)["power"]["detail"]


# ---------------------------------------------------------------------------
# target-leakage naming heuristic
# ---------------------------------------------------------------------------


def test_leakage_naming_absent_without_response_or_predictors():
    block = verify_payload(_regression(response_variable=None))
    assert "target_leakage_naming" not in _ids(block)


def test_leakage_naming_passes_on_unrelated_names():
    block = verify_payload(_regression(
        response_variable="satisfaction_score",
        predictor_variables=["age", "income"],
    ))
    assert _ids(block)["target_leakage_naming"]["status"] == "pass"


def test_leakage_naming_warns_on_specific_shared_token():
    block = verify_payload(_regression(
        response_variable="will_churn",
        predictor_variables=["churn_flag", "tenure"],
    ))
    c = _ids(block)["target_leakage_naming"]
    assert c["status"] == "warn"
    assert "churn_flag" in c["detail"]
    assert "churn" in c["detail"]


def test_leakage_naming_ignores_generic_shared_tokens():
    """'customer_id' and 'account_id' share only the generic 'id'
    token — must NOT be flagged, or every regression with an id
    column would false-positive."""
    block = verify_payload(_regression(
        response_variable="customer_id_purchased",
        predictor_variables=["account_id", "region"],
    ))
    # "id" is a stopword; "customer"/"purchased" vs "account"/"region"
    # share nothing specific.
    assert _ids(block)["target_leakage_naming"]["status"] == "pass"


def test_leakage_naming_case_and_separator_insensitive():
    block = verify_payload(_regression(
        response_variable="Will-Default",
        predictor_variables=["DEFAULT_flag"],
    ))
    assert _ids(block)["target_leakage_naming"]["status"] == "warn"


# ---------------------------------------------------------------------------
# extreme t-statistic
# ---------------------------------------------------------------------------


def test_extreme_t_statistic_absent_below_threshold():
    block = verify_payload(_regression(
        t_statistics={"tenure": -8.0, "spend": 5.0},
    ))
    assert "extreme_t_statistic" not in _ids(block)


def test_extreme_t_statistic_fires_above_threshold():
    block = verify_payload(_regression(
        t_statistics={"tenure": -8.0, "spend": 60.0},
    ))
    c = _ids(block)["extreme_t_statistic"]
    assert c["status"] == "warn"
    assert "spend" in c["detail"]
    assert "tenure" not in c["detail"].split(";")[0]  # only offenders named


def test_extreme_t_statistic_negative_values_use_magnitude():
    block = verify_payload(_regression(
        t_statistics={"tenure": -75.0, "spend": 1.0},
    ))
    c = _ids(block)["extreme_t_statistic"]
    assert c["status"] == "warn"
    assert "tenure" in c["detail"]
