"""Executable and adversarial qualification for predictive workflows."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from sift.sanitizer import sanitize
from sift.verification import verify_payload


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "audit_predictive_workflows.py"


@pytest.fixture(scope="module")
def predictive_rows(tmp_path_factory) -> dict[str, dict]:
    destination = tmp_path_factory.mktemp("predictive") / "results.jsonl"
    environment = os.environ.copy()
    environment["SIFT_RUN_TOKEN"] = "predictive-qualification-token"
    environment["SIFT_RESULT_PATH"] = str(destination)
    process = subprocess.run(
        [sys.executable, str(AUDIT)], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stderr
    rows = {}
    for line in destination.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload.pop("_token", None)
        rows[payload["analysis_id"]] = payload
    assert set(rows) == {
        "regression_holdout", "regression_cv",
        "classification_holdout", "classification_cv",
    }
    return rows


@pytest.fixture(scope="module")
def runtime_module(tmp_path_factory):
    destination = tmp_path_factory.mktemp("predictive_runtime") / "results.jsonl"
    old_token, old_path = os.environ.get("SIFT_RUN_TOKEN"), os.environ.get("SIFT_RESULT_PATH")
    os.environ["SIFT_RUN_TOKEN"] = "predictive-runtime-token"
    os.environ["SIFT_RESULT_PATH"] = str(destination)
    try:
        spec = importlib.util.spec_from_file_location(
            "sift_predictive_runtime", ROOT / "src" / "sift" / "runtime" / "sift.py",
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


@pytest.mark.parametrize(
    "analysis_id",
    ["regression_holdout", "regression_cv", "classification_holdout", "classification_cv"],
)
def test_real_workflows_are_valid_aggregate_only(
    predictive_rows: dict[str, dict], analysis_id: str,
) -> None:
    clean = sanitize(predictive_rows[analysis_id])
    assert clean.ok, clean.rejection_reason
    assert {"predictions", "probabilities", "rows", "fold_assignments"}.isdisjoint(clean.sanitized)
    assert clean.sanitized["diagnostics"]["preprocessing_inside_split"] == "pass"


def test_train_validation_test_counts_and_bootstrap_are_honest(
    predictive_rows: dict[str, dict],
) -> None:
    for analysis_id in ("regression_holdout", "classification_holdout"):
        result = sanitize(predictive_rows[analysis_id]).sanitized
        assert result["training_observations"] == 216
        assert result["validation_observations"] == 72
        assert result["test_observations"] == 72
        assert result["uncertainty_type"] == "bootstrap"
        assert result["interval_method"] == "heldout_case_bootstrap"
        assert result["bootstrap_replicates"] == 300
        key = "roc_auc" if "classification" in analysis_id else "rmse"
        assert result["ci_lower"][key] <= result["estimates"][key] <= result["ci_upper"][key]


def test_cross_validation_is_out_of_fold_and_does_not_claim_naive_uncertainty(
    predictive_rows: dict[str, dict],
) -> None:
    for analysis_id in ("regression_cv", "classification_cv"):
        result = sanitize(predictive_rows[analysis_id]).sanitized
        assert result["evaluation_split"] == "cross_validation"
        assert result["evaluated_observations"] == result["n"] == 360
        assert result["folds"] == 5
        assert result["bootstrap_replicates"] == 0
        assert "uncertainty_type" not in result
        assert result["diagnostics"]["uncertainty"] == "not_applicable"


def test_regression_recovers_strong_known_signal_and_beats_mean_baseline(
    predictive_rows: dict[str, dict],
) -> None:
    result = sanitize(predictive_rows["regression_holdout"]).sanitized
    assert result["metrics"]["r2"] > 0.8
    assert result["metrics"]["rmse"] < result["metrics"]["baseline_rmse"] / 2
    assert result["diagnostics"]["baseline_comparison"] == "pass"


def test_classification_reports_discrimination_calibration_imbalance_and_baseline(
    predictive_rows: dict[str, dict],
) -> None:
    result = sanitize(predictive_rows["classification_holdout"]).sanitized
    assert result["metrics"]["roc_auc"] > 0.75
    assert result["metrics"]["brier_score"] < result["metrics"]["baseline_brier_score"]
    assert result["imbalance_strategy"] == "balanced_weight"
    assert result["calibration_method"] == "nested_sigmoid"
    assert result["metrics"]["minority_fraction"] < 0.4
    checks = verify_payload(result)["checks"]
    assert any(row["id"] == "out_of_sample_evaluation" and row["status"] == "pass" for row in checks)


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


def test_preprocessor_is_cloned_and_fit_only_inside_training_partitions(runtime_module) -> None:
    rng = np.random.default_rng(9)
    X = rng.normal(size=(120, 3))
    y = 1 + X[:, 0] + rng.normal(scale=0.2, size=120)
    prefit = RecordingTransformer().fit(X[:7])
    RecordingTransformer.fit_sizes.clear()
    runtime_module.from_predictive_workflow(
        X, y, task="regression", evaluation="train_validation_test",
        preprocessor=prefit, seed=9, bootstrap_replicates=200,
    )
    assert sorted(RecordingTransformer.fit_sizes) == [72, 96]
    assert 120 not in RecordingTransformer.fit_sizes
    RecordingTransformer.fit_sizes.clear()
    runtime_module.from_predictive_workflow(
        X, y, task="regression", evaluation="cross_validation",
        preprocessor=prefit, seed=9, folds=5,
    )
    assert RecordingTransformer.fit_sizes == [96] * 5


def test_probability_calibration_nests_preprocessing_inside_development_folds(
    runtime_module,
) -> None:
    rng = np.random.default_rng(21)
    X = rng.normal(size=(120, 2))
    score = X[:, 0] - 0.4 * X[:, 1]
    y = np.zeros(120, dtype=int)
    y[np.argsort(score)[-60:]] = 1
    RecordingTransformer.fit_sizes.clear()
    runtime_module.from_predictive_workflow(
        X, y, task="classification", evaluation="train_validation_test",
        preprocessor=RecordingTransformer(), seed=21,
        bootstrap_replicates=200, imbalance_strategy="none",
    )
    # Three inner calibration folds on train (3 x 48) and then on the
    # train+validation development set (3 x 64); never the 24-row test set.
    assert sorted(RecordingTransformer.fit_sizes) == [48, 48, 48, 64, 64, 64]
    assert 120 not in RecordingTransformer.fit_sizes


def test_class_labels_and_unhandled_imbalance_fail_closed(runtime_module) -> None:
    rng = np.random.default_rng(12)
    X = rng.normal(size=(160, 2))
    labels = np.array(["case"] * 60 + ["control"] * 100)
    with pytest.raises(ValueError, match="0/1 labels"):
        runtime_module.from_predictive_workflow(X, labels, task="classification")
    binary = np.array([1] * 50 + [0] * 110)
    with pytest.raises(ValueError, match="requires balanced_weight"):
        runtime_module.from_predictive_workflow(
            X, binary, task="classification", imbalance_strategy="none",
        )
    with pytest.raises(ValueError, match="cannot override"):
        runtime_module.from_predictive_workflow(
            X, binary.astype(float), task="regression", baseline_model="forged",
        )


@pytest.mark.parametrize(
    ("analysis_id", "mutation"),
    [
        ("regression_holdout", lambda row: row.__setitem__("test_observations", 71)),
        ("regression_cv", lambda row: row.__setitem__("bootstrap_replicates", 200)),
        ("classification_holdout", lambda row: row["metrics"].__setitem__("baseline_auc", 0.8)),
        ("classification_holdout", lambda row: row.__setitem__("calibration_method", "not_applicable")),
        ("classification_cv", lambda row: row["diagnostics"].__setitem__("split_integrity", "warn")),
    ],
)
def test_predictive_sanitizer_rejects_split_and_metric_forgeries(
    predictive_rows: dict[str, dict], analysis_id: str, mutation,
) -> None:
    payload = copy.deepcopy(predictive_rows[analysis_id])
    mutation(payload)
    clean = sanitize(payload)
    assert not clean.ok
