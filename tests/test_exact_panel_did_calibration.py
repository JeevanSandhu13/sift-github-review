"""Executable qualification for the final exact registry adapters."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from sift.sanitizer import sanitize
from sift.verification import verify_payload


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runtime_module(tmp_path_factory):
    destination = tmp_path_factory.mktemp("exact_methods") / "results.jsonl"
    old_token = os.environ.get("SIFT_RUN_TOKEN")
    old_path = os.environ.get("SIFT_RESULT_PATH")
    os.environ["SIFT_RUN_TOKEN"] = "exact-method-token"
    os.environ["SIFT_RESULT_PATH"] = str(destination)
    try:
        spec = importlib.util.spec_from_file_location(
            "sift_exact_method_runtime", ROOT / "src" / "sift" / "runtime" / "sift.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old_token is None:
            os.environ.pop("SIFT_RUN_TOKEN", None)
        else:
            os.environ["SIFT_RUN_TOKEN"] = old_token
        if old_path is None:
            os.environ.pop("SIFT_RESULT_PATH", None)
        else:
            os.environ["SIFT_RESULT_PATH"] = old_path
    module._test_output = destination
    return module


def _last_payload(runtime_module) -> dict:
    payload = json.loads(runtime_module._test_output.read_text(encoding="utf-8").splitlines()[-1])
    payload.pop("_token", None)
    return payload


def test_panel_fixed_effects_recovers_within_signal_and_sanitizes(runtime_module) -> None:
    rng = np.random.default_rng(710)
    entities, periods = 30, 4
    entity = np.repeat(np.arange(entities), periods)
    time = np.tile(np.arange(periods), entities)
    entity_level = rng.normal(scale=2.0, size=entities)
    x = np.repeat(0.7 * entity_level, periods) + rng.normal(size=entities * periods)
    y = np.repeat(entity_level, periods) + 2.0 * x + rng.normal(scale=0.2, size=len(x))
    runtime_module.from_panel_fixed_effects(y, x[:, None], entity, time)
    raw = _last_payload(runtime_module)
    clean = sanitize(raw)
    assert clean.ok, clean.rejection_reason
    result = clean.sanitized
    assert abs(result["estimates"]["x1"] - 2.0) < 0.1
    assert result["clusters"] == entities
    assert result["records"] == entities * periods
    assert result["diagnostics"]["within_variation"] > 0
    checks = verify_payload(result)["checks"]
    assert any(row["id"] == "panel_clustered_inference" and row["status"] == "pass" for row in checks)


def test_panel_fixed_effects_rejects_unbalanced_and_time_invariant_predictors(runtime_module) -> None:
    entity = np.repeat(np.arange(12), 3)
    time = np.tile(np.arange(3), 12)
    with pytest.raises(ValueError, match="balanced panel"):
        runtime_module.from_panel_fixed_effects(
            np.arange(35.0), np.arange(35.0)[:, None], entity[:-1], time[:-1],
        )
    invariant = np.repeat(np.arange(12.0), 3)[:, None]
    with pytest.raises(ValueError, match="within-entity variation"):
        runtime_module.from_panel_fixed_effects(np.arange(36.0), invariant, entity, time)


def test_two_by_two_did_recovers_known_att_and_warns_on_parallel_trends(runtime_module) -> None:
    rng = np.random.default_rng(711)
    entities = 60
    panel_id = np.repeat(np.arange(entities), 2)
    post = np.tile([0, 1], entities)
    treated_entity = np.r_[np.zeros(30, dtype=int), np.ones(30, dtype=int)]
    treated = np.repeat(treated_entity, 2)
    intercept = rng.normal(scale=1.0, size=entities)
    noise = rng.normal(scale=0.08, size=entities * 2)
    outcome = np.repeat(intercept, 2) + 1.25 * post + 2.5 * treated * post + noise
    runtime_module.from_difference_in_differences(outcome, treated, post, panel_id)
    raw = _last_payload(runtime_module)
    clean = sanitize(raw)
    assert clean.ok, clean.rejection_reason
    result = clean.sanitized
    assert abs(result["estimates"]["att"] - 2.5) < 0.1
    assert result["diagnostics"]["parallel_pretrends"] == "not_applicable"
    checks = verify_payload(result)["checks"]
    assert any(row["id"] == "did_two_by_two_contrast" and row["status"] == "pass" for row in checks)
    assert any(row["id"] == "did_parallel_trends_boundary" and row["status"] == "warn" for row in checks)


def test_two_by_two_did_rejects_nonconstant_treatment(runtime_module) -> None:
    panel_id = np.repeat(np.arange(20), 2)
    post = np.tile([0, 1], 20)
    treated = np.repeat(np.r_[np.zeros(10), np.ones(10)], 2)
    treated[1] = 1
    with pytest.raises(ValueError, match="entity-invariant"):
        runtime_module.from_difference_in_differences(
            np.arange(40.0), treated, post, panel_id,
        )


class RecordingTransformer(BaseEstimator, TransformerMixin):
    fit_sizes: list[int] = []

    def fit(self, X, y=None):
        type(self).fit_sizes.append(len(X))
        self.fitted_ = True
        return self

    def transform(self, X):
        if not getattr(self, "fitted_", False):
            raise RuntimeError("transform before fit")
        return np.asarray(X)


def test_probability_calibration_is_nested_and_aggregate_only(runtime_module) -> None:
    rng = np.random.default_rng(712)
    X = rng.normal(size=(300, 3))
    score = 1.2 * X[:, 0] - 0.7 * X[:, 1] + rng.logistic(scale=1.0, size=300)
    y = (score > 0).astype(int)
    RecordingTransformer.fit_sizes.clear()
    runtime_module.from_probability_calibration(
        X, y, preprocessor=RecordingTransformer(), seed=712,
        calibration_folds=4, bootstrap_replicates=200,
        imbalance_strategy="none",
    )
    raw = _last_payload(runtime_module)
    clean = sanitize(raw)
    assert clean.ok, clean.rejection_reason
    result = clean.sanitized
    assert result["method_id"] == "probability_calibration"
    assert result["estimates"] == {"brier_score": result["metrics"]["brier_score"]}
    assert {"predictions", "probabilities", "rows", "labels"}.isdisjoint(result)
    assert sorted(RecordingTransformer.fit_sizes) == [180, 180, 180, 180, 240]
    assert 300 not in RecordingTransformer.fit_sizes
    checks = verify_payload(result)["checks"]
    assert any(row["id"] == "calibration_aggregate_contract" and row["status"] == "pass" for row in checks)


@pytest.mark.parametrize(
    ("helper", "mutation"),
    [
        ("panel_fixed_effects_v1", lambda row: row["metrics"].__setitem__("entity_count", 29.0)),
        ("difference_in_differences_v1", lambda row: row["metrics"].__setitem__("raw_did", 99.0)),
        ("probability_calibration_v1", lambda row: row["metrics"].__setitem__("brier_score", 0.99)),
    ],
)
def test_exact_sanitizers_reject_aggregate_forgery(runtime_module, helper, mutation) -> None:
    rows = [json.loads(line) for line in runtime_module._test_output.read_text(encoding="utf-8").splitlines()]
    payload = next(copy.deepcopy(row) for row in reversed(rows) if row.get("_via_helper") == helper)
    payload.pop("_token", None)
    mutation(payload)
    assert not sanitize(payload).ok


def test_exact_sanitizers_reject_missing_provenance_and_method_mismatch(runtime_module) -> None:
    rows = [json.loads(line) for line in runtime_module._test_output.read_text(encoding="utf-8").splitlines()]
    calibration = next(copy.deepcopy(row) for row in reversed(rows) if row.get("method_id") == "probability_calibration")
    calibration.pop("_token", None)
    calibration.pop("_via_helper")
    assert not sanitize(calibration).ok

    mismatched = copy.deepcopy(calibration)
    mismatched["method_id"] = "predictive_classification"
    mismatched["_via_helper"] = "probability_calibration_v1"
    assert not sanitize(mismatched).ok


def test_generic_emitter_cannot_forge_exact_helper_provenance(runtime_module) -> None:
    runtime_module.from_method(
        "probability_calibration", n=100,
        diagnostics={name: "pass" for name in (
            "held_out_performance", "baseline_comparison", "calibration",
            "calibration_curve", "brier_score", "split_integrity",
            "preprocessing_inside_split", "calibration_nested", "class_balance", "uncertainty",
        )},
        metrics={"brier_score": 0.2}, _via_helper="probability_calibration_v1",
    )
    payload = _last_payload(runtime_module)
    assert "_via_helper" not in payload
    assert not sanitize(payload).ok
