from __future__ import annotations

from sift.result_render import render_table
from sift.sanitizer import sanitize, supported_types
from sift.verification import verify_payload


def _linear(**updates):
    row = {
        "type": "method_result", "method_id": "linear_regression", "n": 120,
        "diagnostics": {
            "convergence": "pass", "specification": "pass",
            "influence": "pass", "multicollinearity": "pass",
            "heteroskedasticity": "pass", "residual_distribution": "pass",
        },
        "estimates": {"x": 1.25}, "standard_errors": {"x": .2},
        "p_values": {"x": .01}, "ci_lower": {"x": .85}, "ci_upper": {"x": 1.65},
        "uncertainty_type": "robust", "multiple_testing": "none",
    }
    row.update(updates)
    return row


def test_method_result_is_registered_and_policy_fields_come_from_registry() -> None:
    assert "method_result" in supported_types()
    result = sanitize(_linear(claim_rule="attacker text", method_family="causal"))
    assert result.ok
    assert result.sanitized["method_family"] == "regression"
    assert "conditional associations" in result.sanitized["claim_rule"]
    assert "attacker" not in str(result.sanitized)


def test_method_result_requires_registry_diagnostics() -> None:
    payload = _linear(diagnostics={"convergence": "pass"})
    result = sanitize(payload)
    assert not result.ok
    assert "missing required methodology diagnostics" in result.rejection_reason


def test_method_result_rejects_invalid_intervals_and_p_values() -> None:
    assert not sanitize(_linear(ci_lower={"x": 2}, ci_upper={"x": 3})).ok
    result = sanitize(_linear(p_values={"x": 5}))
    assert result.ok
    assert "p_values" not in result.sanitized


def test_method_result_drops_unbounded_or_non_identifier_entries() -> None:
    result = sanitize(_linear(estimates={"x": 1, "raw row, secret": 123}))
    assert result.ok
    assert result.sanitized["estimates"] == {"x": 1.0}
    assert "raw row" not in str(result.sanitized)


def test_predictive_and_temporal_split_guards() -> None:
    diagnostics = {
        "held_out_performance": "pass", "baseline_comparison": "pass",
        "calibration": "pass", "split_integrity": "pass",
    }
    payload = {
        "type": "method_result", "method_id": "predictive_regression", "n": 100,
        "diagnostics": diagnostics, "metrics": {"rmse": 1.0},
    }
    assert not sanitize(payload).ok
    payload["evaluation_split"] = "cross_validation"
    # A declared split string alone is not executable evidence. Predictive
    # results must come through the typed workflow that nests preprocessing.
    assert not sanitize(payload).ok
    forecast = {
        "type": "method_result", "method_id": "forecast_backtest", "n": 100,
        "diagnostics": {
            "temporal_order": "pass", "regular_frequency": "pass",
            "rolling_origin_backtest": "pass", "prediction_interval_coverage": "pass",
            "baseline_comparison": "pass", "holdout_leakage": "pass",
        },
        "metrics": {
            "rmse": 2.5, "mae": 2.0, "prediction_interval_coverage": .95,
            "prediction_interval_mean_width": 8.0, "nominal_coverage": .95,
                "mean_forecast": 10.0, "mean_actual": 10.2,
                "baseline_rmse": 3.0, "origins": 20,
                "cadence_min_ratio": 1.0, "cadence_max_ratio": 1.0,
                "time_span_steps": 99,
        }, "evaluation_split": "held_out", "frequency": 1,
        "training_observations": 80, "test_observations": 20, "folds": 20,
        "interval_method": "model_based_gaussian",
    }
    assert not sanitize(forecast).ok
    forecast["evaluation_split"] = "rolling_origin"
    assert sanitize(forecast).ok


def test_imputation_requires_more_than_a_caller_asserted_seed() -> None:
    payload = {
        "type": "method_result", "method_id": "multiple_imputation", "n": 100,
        "diagnostics": {
            "missingness_pattern": "pass", "imputation_trace_stability": 0.1,
            "between_imputation_variance": "pass", "seed_recorded": "pass",
            "fraction_missing_information": "pass", "rubin_pooling": "pass",
        },
        "estimates": {"x": 1.0}, "imputations": 20,
    }
    assert not sanitize(payload).ok
    payload["seed"] = 42
    result = sanitize(payload)
    assert not result.ok
    assert "Rubin-pooled" in result.rejection_reason


def test_bayesian_diagnostics_drive_verifier_confidence() -> None:
    payload = {
        "type": "method_result", "method_id": "bayesian_model", "n": 500,
        "diagnostics": {
            "rhat": 1.0, "bulk_ess": 600, "tail_ess": 500,
            "divergences": 0, "posterior_predictive_check": .5,
        },
        "estimates": {"beta": .3},
        "ci_lower": {"beta": .1}, "ci_upper": {"beta": .5},
        "metrics": {
            "chains": 4, "draws_per_chain": 200, "parameter_count": 1,
            "posterior_predictive_replicates": 800,
        },
        "uncertainty_type": "posterior",
    }
    clean = sanitize(payload)
    assert clean.ok
    verification = verify_payload(clean.sanitized)
    assert any(row["id"] == "bayesian_computation" and row["status"] == "pass"
               for row in verification["checks"])

    payload["diagnostics"] = dict(payload["diagnostics"], divergences=1)
    assert not sanitize(payload).ok


def test_method_result_has_canonical_renderer_and_claim_boundary() -> None:
    clean = sanitize(_linear()).sanitized
    rendered = render_table(clean)
    assert "Quantity" in rendered and "Estimate / metric" in rendered and "95% CI" in rendered
    assert "Required diagnostic" in rendered
    assert "Claim boundary:" in rendered


def test_storage_pipeline_binds_emitted_result_to_prevalidated_method(tmp_path) -> None:
    from sift.sanitizer import DEFAULT_CONFIG
    from sift.store import ResultStore
    from sift.tools import _sanitize_and_store_payloads

    store = ResultStore(tmp_path / ".sift" / "results.db")
    raw = _linear()
    results, any_ok, *_ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="method", language="Python", code="# fit",
        source_dataset=None, source_n=None, sdc_cfg=DEFAULT_CONFIG, run_dir=None,
        script_run_id="run-method", store=store,
    )
    assert not any_ok
    assert "prevalidated" in results[0]["reason"]
    results, any_ok, *_ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="method", language="Python", code="# fit",
        source_dataset=None, source_n=None, sdc_cfg=DEFAULT_CONFIG, run_dir=None,
        script_run_id="run-method-2", store=store, expected_method_id="bayesian_model",
    )
    assert not any_ok
    assert "does not match" in results[0]["reason"]
    results, any_ok, *_ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="method", language="Python", code="# fit",
        source_dataset=None, source_n=None, sdc_cfg=DEFAULT_CONFIG, run_dir=None,
        script_run_id="run-method-3", store=store, expected_method_id="linear_regression",
    )
    assert any_ok and results[0]["status"] == "ok"
    store.close()


def test_storage_pipeline_binds_result_to_approved_analysis_and_provenance(tmp_path) -> None:
    from sift.sanitizer import DEFAULT_CONFIG
    from sift.store import ResultStore
    from sift.tools import _sanitize_and_store_payloads

    store = ResultStore(tmp_path / ".sift" / "results.db")
    workflow = {
        "workflow_id": "wf-test", "workflow_revision": 1,
        "approval_sha256": "a" * 64,
        "analyses": [{
            "id": "primary", "role": "primary", "seed": 42, "changes": [],
        }],
    }
    raw = _linear()
    results, any_ok, *_ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="method", language="Python", code="# fit",
        source_dataset=None, source_n=None, sdc_cfg=DEFAULT_CONFIG, run_dir=None,
        script_run_id="run-workflow-1", store=store,
        expected_method_id="linear_regression", workflow_context=workflow,
        provenance_base={"dataset_hashes": {"data.csv": "b" * 64}},
    )
    assert not any_ok
    assert "analysis_id" in results[0]["reason"]
    raw["analysis_id"] = "primary"
    raw["seed"] = 42
    results, any_ok, *_ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="method", language="Python", code="# fit",
        source_dataset=None, source_n=None, sdc_cfg=DEFAULT_CONFIG, run_dir=None,
        script_run_id="run-workflow-2", store=store,
        expected_method_id="linear_regression", workflow_context=workflow,
        provenance_base={"dataset_hashes": {"data.csv": "b" * 64}},
    )
    assert any_ok
    row = store.get(results[0]["result_id"])
    assert row.provenance["analysis_role"] == "primary"
    assert row.provenance["random_seed"] == 42
    assert row.provenance["dataset_hashes"]["data.csv"] == "b" * 64
    assert row.provenance["verification_outcome"]["status"] in {"pass", "warn"}
    store.close()
