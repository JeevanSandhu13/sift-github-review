"""Causality labels and automatic caveats.

A result's causal status is a function of the analysis TYPE (and, for
regression, whether IV fields are present) -- not of how clean its
diagnostics look. A perfectly well-specified OLS with zero warnings
still only supports an associational claim. Every label carries a
caveat; the point is that the model's fluent prose should never imply
more than the design supports.
"""

from __future__ import annotations

from sift.verification import verify_payload


def _regression(**over):
    base = {
        "type": "linear_regression",
        "n": 500,
        "coefficients": {"x": 0.5},
        "standard_errors": {"x": 0.1},
        "response_variable": "y",
        "predictor_variables": ["x"],
    }
    base.update(over)
    return base


def test_plain_ols_is_associational():
    block = verify_payload(_regression())
    assert block["causality"]["label"] == "associational"
    assert "does not by itself establish" in block["causality"]["caveat"]


def test_iv_regression_is_quasi_experimental():
    block = verify_payload(_regression(first_stage_f=15.0))
    c = block["causality"]
    assert c["label"] == "quasi_experimental"
    assert c["design"] == "instrumental_variables"
    assert "LOCAL average treatment effect" in c["caveat"]


def test_did_is_quasi_experimental_with_parallel_trends_caveat():
    block = verify_payload({"type": "did_event_study", "n": 500})
    c = block["causality"]
    assert c["label"] == "quasi_experimental"
    assert c["design"] == "difference_in_differences"
    assert "PARALLEL TRENDS" in c["caveat"]


def test_rdd_is_quasi_experimental_with_local_caveat():
    block = verify_payload({"type": "rdd", "n": 500})
    c = block["causality"]
    assert c["label"] == "quasi_experimental"
    assert c["design"] == "regression_discontinuity"
    assert "LOCAL" in c["caveat"]
    assert "manipulation" in c["caveat"]


def test_kaplan_meier_is_descriptive_with_confounding_caveat():
    block = verify_payload({"type": "kaplan_meier", "n": 500})
    c = block["causality"]
    assert c["label"] == "descriptive"
    assert "confounding" in c["caveat"]


def test_correlation_matrix_gets_correlation_is_not_causation_caveat():
    block = verify_payload({"type": "correlation_matrix", "n": 500})
    c = block["causality"]
    assert c["label"] == "associational"
    assert "does not imply causation" in c["caveat"]


def test_ttest_is_associational():
    block = verify_payload({
        "type": "t_test", "n1": 50, "n2": 50,
        "mean1": 1.0, "mean2": 1.3, "p_value": 0.01,
    })
    assert block["causality"]["label"] == "associational"


def test_cluster_analysis_is_descriptive():
    block = verify_payload({
        "type": "cluster_analysis", "n_observations": 500,
        "silhouette_score": 0.5,
    })
    assert block["causality"]["label"] == "descriptive"


def test_factor_decomposition_is_descriptive():
    block = verify_payload({
        "type": "factor_decomposition", "kmo": 0.7,
    })
    assert block["causality"]["label"] == "descriptive"


def test_marginal_effects_is_associational():
    block = verify_payload({
        "type": "marginal_effects",
        "effects": {"x": 0.3},
        "standard_errors": {"x": 0.05},
    })
    assert block["causality"]["label"] == "associational"


def test_causality_absent_when_no_checks_fire_at_all():
    # Matches verify_payload's existing "nothing checkable -> None"
    # contract (test_verification.py pins this for correlation_matrix
    # with no n) -- causality must not force a block into existence
    # on its own.
    assert verify_payload({"type": "correlation_matrix"}) is None


def test_causality_absent_for_unrecognised_type():
    block = verify_payload({"type": "something_new", "n": 500})
    # n alone produces a sample_size check, so block exists, but no
    # causality mapping is defined for an unrecognised type.
    assert block is not None
    assert "causality" not in block
