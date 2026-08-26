"""Deterministic statistical verification — verdict honesty invariants.

The verification block's product claim is "checked by code, not by
the model". These tests pin the properties that keep the claim
honest: checks appear only when their inputs exist; thresholds fire
deterministically; the block never blocks or mutates a result; the
batch note appears only for genuine batches.
"""

from __future__ import annotations

from sift.verification import batch_note, verify_payload


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


def test_clean_regression_all_pass() -> None:
    block = verify_payload(_regression(
        vif={"tenure": 1.4, "spend": 1.4},
        condition_number=4.2,
        r_squared=0.31,
        robust_se_type="hc1",
    ))
    assert block is not None
    assert block["warnings"] == 0
    ids = {c["id"]: c["status"] for c in block["checks"]}
    assert ids["sample_size"] == "pass"
    assert ids["multicollinearity"] == "pass"
    assert ids["conditioning"] == "pass"
    assert ids["robust_se"] == "pass"


def test_thresholds_fire() -> None:
    block = verify_payload(_regression(
        n=25,
        vif={"tenure": 42.0, "spend": 1.2},
        condition_number=180.0,
        r_squared=0.9999,
        robust_se_type="classical",
    ))
    ids = {c["id"]: c for c in block["checks"]}
    assert ids["sample_size"]["status"] == "warn"
    assert ids["multicollinearity"]["status"] == "warn"
    assert "tenure" in ids["multicollinearity"]["detail"]
    assert ids["conditioning"]["status"] == "warn"
    assert ids["suspicious_fit"]["status"] == "warn"
    assert ids["robust_se"]["status"] == "warn"
    # obs/param: n=25, k=3 → 8.3 < 10 → warn
    assert ids["obs_per_parameter"]["status"] == "warn"
    assert block["warnings"] >= 5


def test_absent_inputs_produce_no_checks_not_passes() -> None:
    """A check must never claim 'pass' for something it couldn't see."""
    block = verify_payload(_regression())  # no vif / cond / r2 fields
    ids = {c["id"] for c in block["checks"]}
    assert "multicollinearity" not in ids
    assert "conditioning" not in ids
    assert "suspicious_fit" not in ids
    assert "robust_se" not in ids


def test_weak_instruments_flagged() -> None:
    block = verify_payload(_regression(first_stage_f=4.2))
    ids = {c["id"]: c["status"] for c in block["checks"]}
    assert ids["instrument_strength"] == "warn"
    block = verify_payload(_regression(first_stage_f=31.0))
    ids = {c["id"]: c["status"] for c in block["checks"]}
    assert ids["instrument_strength"] == "pass"


def test_suppression_extent_on_frequency_tables() -> None:
    block = verify_payload({
        "type": "frequency_table", "variable": "region", "n": 900,
        "counts": {"north": 400, "south": 480, "islands": "<10"},
    })
    ids = {c["id"]: c for c in block["checks"]}
    assert ids["suppression_extent"]["status"] == "warn"
    assert "1 of 3" in ids["suppression_extent"]["detail"]


def test_unknown_or_empty_payload_returns_none() -> None:
    assert verify_payload({}) is None
    assert verify_payload({"type": "correlation_matrix"}) is None
    assert verify_payload(None) is None  # type: ignore[arg-type]


def test_batch_note_only_for_batches() -> None:
    assert batch_note(1) is None
    assert batch_note(4) is None
    note = batch_note(24)
    assert note and "24" in note and "multiple comparisons" in note


def test_wired_into_result_entries(tmp_path) -> None:
    """End-to-end through _sanitize_and_store_payloads: an ok entry
    carries the verification block computed from its own sanitized
    payload."""
    from sift.store import ResultStore
    from sift.sanitizer import DEFAULT_CONFIG
    from sift.tools import _sanitize_and_store_payloads

    store = ResultStore(tmp_path / ".sift" / "results.db")
    raw = {
        "type": "linear_regression",
        "n": 25,
        "coefficients": {"(Intercept)": 1.0, "x": 2.0},
        "standard_errors": {"(Intercept)": 0.5, "x": 0.4},
        "response_variable": "y",
        "predictor_variables": ["x"],
        "sift_token": "t",
    }
    results, any_ok, _, _ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="test", language="Python",
        code="x", source_dataset=None, source_n=None,
        sdc_cfg=DEFAULT_CONFIG, run_dir=None,
        script_run_id="run1", store=store,
    )
    assert any_ok
    entry = results[0]
    assert entry["status"] == "ok"
    ver = entry.get("verification")
    assert ver is not None
    ids = {c["id"]: c["status"] for c in ver["checks"]}
    assert ids["sample_size"] == "warn"
    store.close()


def test_canonical_regression_type_name_gets_verified_too() -> None:
    """Regression test for architecture-audit finding I:
    ``verify_payload``'s dispatch table (and ``_causality_label``)
    keyed only on "linear_regression", the LEGACY alias. Every
    regression result the current R / Python / Stata helpers actually
    emit is stamped "coefficient_table_with_fit_stats" (see
    sanitizer.py's ``_REGRESSION_TYPE_CANONICAL`` and system_prompt.py's
    tool description) -- without this key, every CURRENT regression
    result silently got no verification checks and no causality label
    at all, and only pre-rename stored payloads (using the legacy
    name) were ever actually verified. This must produce the exact
    same verdict as the legacy-named payload.
    """
    payload = _regression(
        type="coefficient_table_with_fit_stats",
        vif={"tenure": 1.4, "spend": 1.4},
        condition_number=4.2,
        r_squared=0.31,
        robust_se_type="hc1",
    )
    block = verify_payload(payload)
    assert block is not None
    assert block["warnings"] == 0
    ids = {c["id"]: c["status"] for c in block["checks"]}
    assert ids["sample_size"] == "pass"
    assert ids["multicollinearity"] == "pass"
    assert ids["conditioning"] == "pass"
    # Causality labeling is the other half of the same dispatch gap.
    assert block.get("causality") is not None
    assert block["causality"]["label"] == "associational"
