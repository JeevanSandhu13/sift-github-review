"""Deterministic executable-reference audit for Stage 10 missing-data methods."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sift" / "runtime"))
import sift as sift_runtime  # noqa: E402


SEED = 20260822
rng = np.random.default_rng(SEED)
n = 240
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
y = 1.0 + 1.45 * x1 - 0.55 * x2 + rng.normal(scale=0.55, size=n)
data = pd.DataFrame({"y": y, "x1": x1, "x2": x2})

# MAR-like covariate omissions create multiple joint patterns and meaningful
# complete-case attrition without changing the known data-generating slope.
p_x1 = 1 / (1 + np.exp(-(-1.1 + 0.8 * x2)))
miss_x1 = rng.random(n) < p_x1
miss_x2 = rng.random(n) < 0.12
data.loc[miss_x1, "x1"] = np.nan
data.loc[miss_x2, "x2"] = np.nan
sift_runtime.from_missingness_pattern(data)

# Single imputation is qualified only as deterministic preprocessing.  No
# inferential model is fit or emitted on this branch.
source_features = data[["x1", "x2"]]
simple = SimpleImputer(strategy="median")
completed_features = simple.fit_transform(source_features)
sift_runtime.from_single_imputation(
    simple, source_features, completed_features,
    scope="prediction_preprocessing",
)

# Maintained statsmodels MICE/Predictive Mean Matching is fitted inside the
# helper, binding the emitted seed and fitting specification to the result.
sift_runtime.from_multiple_imputation(
    data, formula="y ~ x1 + x2", seed=SEED, burn_in=12,
    imputations=16, matching_donors=10,
)

# Delta-adjusted pattern-mixture sensitivity for an incomplete outcome.  Each
# scenario contains multiple independently completed-data fits, and every
# scenario is Rubin-pooled by the helper.
mnar = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
outcome_missing = rng.random(n) < (1 / (1 + np.exp(-(-1.25 + 0.65 * x1))))
mnar.loc[outcome_missing, "y"] = np.nan
deltas = (-0.8, 0.0, 0.8)
sift_runtime.from_mnar_sensitivity(
    mnar, incomplete_outcome="y", formula="y ~ x1 + x2",
    parameter="Intercept", deltas=deltas, seed=SEED + 1, burn_in=12,
    imputations=16, matching_donors=10,
)
