"""Deterministic executable-reference audit for predictive workflows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sift" / "runtime"))
import sift as sift_runtime  # noqa: E402


rng = np.random.default_rng(20260822)
n = 360
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
x3 = rng.normal(size=n)
X = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
X.loc[rng.choice(n, 35, replace=False), "x2"] = np.nan

y_regression = 2.0 + 1.8 * x1 - 0.9 * x2 + 0.4 * x3 + rng.normal(scale=0.65, size=n)
# Preserve the true outcome before feature missingness was introduced.
y_regression = np.nan_to_num(y_regression, nan=2.0 + 1.8 * x1 + 0.4 * x3)
sift_runtime.from_predictive_workflow(
    X, y_regression, task="regression", evaluation="train_validation_test",
    seed=20260822, bootstrap_replicates=300, analysis_id="regression_holdout",
)
sift_runtime.from_predictive_workflow(
    X, y_regression, task="regression", evaluation="cross_validation",
    seed=20260822, folds=5, analysis_id="regression_cv",
)

linear = -1.9 + 1.6 * x1 - 1.0 * np.nan_to_num(x2) + 0.5 * x3
probability = 1 / (1 + np.exp(-linear))
y_classification = rng.binomial(1, probability)
sift_runtime.from_predictive_workflow(
    X, y_classification, task="classification",
    evaluation="train_validation_test", seed=20260822,
    bootstrap_replicates=300, imbalance_strategy="balanced_weight",
    analysis_id="classification_holdout",
)
sift_runtime.from_predictive_workflow(
    X, y_classification, task="classification", evaluation="cross_validation",
    seed=20260822, folds=5, imbalance_strategy="balanced_weight",
    analysis_id="classification_cv",
)
