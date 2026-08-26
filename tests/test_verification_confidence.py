"""Finding confidence levels: strong, moderate, and weak.

The confidence rollup is a deterministic function of the checks
verify_payload() already computes -- pinned here as its own contract
so a future check addition can't silently change the tiering rule
without a test failing:

- Zero warnings -> strong.
- 1-2 non-severe warnings -> moderate.
- Any severe warning, or 3+ warnings regardless of severity -> weak.
- The set of "severe" ids is exercised individually so adding a new
  severe id later is a deliberate, visible decision.
"""

from __future__ import annotations

from sift.verification import verify_payload


def _regression(**over):
    base = {
        "type": "linear_regression",
        "n": 5000,
        "coefficients": {"x1": 0.5, "x2": 0.1},
        "standard_errors": {"x1": 0.1, "x2": 0.05},
        "response_variable": "y",
        "predictor_variables": ["x1", "x2"],
    }
    base.update(over)
    return base


def test_no_warnings_is_strong():
    block = verify_payload(_regression(robust_se_type="hc1"))
    assert block["warnings"] == 0
    assert block["confidence"]["level"] == "strong"


def test_one_non_severe_warning_is_moderate():
    block = verify_payload(_regression(robust_se_type="classical"))
    assert block["warnings"] == 1
    assert block["confidence"]["level"] == "moderate"
    assert "robust_se" in block["confidence"]["reason"]


def test_two_non_severe_warnings_is_still_moderate():
    block = verify_payload(_regression(
        robust_se_type="classical", hausman_p=0.01,
    ))
    assert block["warnings"] == 2
    assert block["confidence"]["level"] == "moderate"


def test_three_non_severe_warnings_becomes_weak():
    block = verify_payload(_regression(
        robust_se_type="classical", hausman_p=0.01, f_test_fe_p=0.001,
    ))
    assert block["warnings"] == 3
    assert block["confidence"]["level"] == "weak"
    assert "accumulated" in block["confidence"]["reason"]


def test_single_severe_warning_is_weak_even_alone():
    block = verify_payload(_regression(converged="not_converged"))
    assert block["warnings"] == 1
    assert block["confidence"]["level"] == "weak"
    assert "convergence" in block["confidence"]["reason"]


def test_severe_plus_nonsevere_is_weak_and_names_only_severe():
    block = verify_payload(_regression(
        converged="not_converged", robust_se_type="classical",
    ))
    assert block["confidence"]["level"] == "weak"
    assert "convergence" in block["confidence"]["reason"]
    assert "robust_se" not in block["confidence"]["reason"]


def test_confidence_absent_when_block_is_none():
    # A payload with no checkable fields at all yields None from
    # verify_payload -- there's no "confidence" to compute either.
    assert verify_payload({"type": "unknown_shape"}) is None


import pytest

@pytest.mark.parametrize("severe_id,kwargs", [
    ("convergence", {"converged": "not_converged"}),
    ("suspicious_fit", {"r_squared": 0.9999}),
    ("extreme_t_statistic", {"t_statistics": {"x1": 80.0}}),
    ("target_leakage_naming", {
        "response_variable": "will_churn",
        "predictor_variables": ["churn_flag"],
    }),
    ("conditioning", {"condition_number": 500.0}),
    ("sample_size", {"n": 5}),
    ("instrument_strength", {"first_stage_f": 2.0}),
    ("overidentification", {"hansen_j_p": 0.001}),
])
def test_each_severe_id_alone_forces_weak(severe_id, kwargs):
    block = verify_payload(_regression(**kwargs))
    assert block["confidence"]["level"] == "weak", (
        f"{severe_id} should force weak on its own; got "
        f"{block['confidence']}"
    )
    assert severe_id in block["confidence"]["reason"]


def test_power_severe_id_forces_weak_on_ttest():
    block = verify_payload({
        "type": "t_test", "n1": 10, "n2": 10,
        "mean1": 1.0, "mean2": 1.3, "p_value": 0.6,
    })
    assert block["confidence"]["level"] == "weak"
    assert "power" in block["confidence"]["reason"]
