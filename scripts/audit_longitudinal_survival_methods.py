"""Deterministic executable-reference audit for six Stage 10 methods."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import CumIncidenceRight
from statsmodels.genmod.cov_struct import Exchangeable, Independence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sift" / "runtime"))
import sift as sift_runtime  # noqa: E402


rng = np.random.default_rng(20260822)

# Quadratic population growth with a random subject intercept.
groups, waves = 32, 5
growth = pd.DataFrame({
    "id": np.repeat(np.arange(groups), waves),
    "time": np.tile(np.arange(waves), groups),
})
random_intercept = rng.normal(0, 0.8, groups)
growth["y"] = (
    2 + 0.7 * growth.time - 0.08 * growth.time**2
    + np.repeat(random_intercept, waves)
    + rng.normal(0, 0.35, len(growth))
)
growth_fit = smf.mixedlm(
    "y ~ time + I(time ** 2)", growth, groups=growth["id"],
).fit(reml=False, method="lbfgs")
sift_runtime.from_growth_curve(growth_fit, time_values=growth.time)

# Population-average binary response with exchangeable working correlation.
gee_data = growth.copy()
linear_predictor = (
    -0.6 + 0.45 * gee_data.time
    + np.repeat(rng.normal(0, 0.7, groups), waves)
)
gee_data["event"] = rng.binomial(
    1, 1 / (1 + np.exp(-linear_predictor)),
)
gee_fit = sm.GEE.from_formula(
    "event ~ time", groups="id", data=gee_data,
    family=sm.families.Binomial(), cov_struct=Exchangeable(),
).fit()
gee_independence = sm.GEE.from_formula(
    "event ~ time", groups="id", data=gee_data,
    family=sm.families.Binomial(), cov_struct=Independence(),
).fit()
sift_runtime.from_gee(
    gee_fit, time_values=gee_data.time, sensitivity_fit=gee_independence,
)

# Exogenous random-intercept panel; FE and RE slopes should be compatible.
subjects, periods = 48, 6
panel = pd.DataFrame({
    "id": np.repeat(np.arange(subjects), periods),
    "time": np.tile(np.arange(periods), subjects),
    "x": rng.normal(size=subjects * periods),
})
panel["y"] = (
    1.2 + 0.65 * panel.x
    + np.repeat(rng.normal(0, 0.9, subjects), periods)
    + rng.normal(0, 0.45, len(panel))
)
random_fit = smf.mixedlm("y ~ x", panel, groups=panel.id).fit(
    reml=False, method="lbfgs",
)
fixed_fit = smf.ols("y ~ x + C(id)", panel).fit()
sift_runtime.from_panel_random_effects(
    random_fit, fixed_effects_fit=fixed_fit, time_values=panel.time,
)

# Two independent causes with administrative censoring preserve probability
# mass below one at the final reported horizon.
n_competing = 240
cause_one = rng.exponential(5.0, n_competing)
cause_two = rng.exponential(7.0, n_competing)
administrative_censor = 5.0
competing_time = np.minimum(np.minimum(cause_one, cause_two), administrative_censor)
competing_status = np.where(
    (cause_one < cause_two) & (cause_one < administrative_censor), 1,
    np.where(
        (cause_two < cause_one) & (cause_two < administrative_censor), 2, 0,
    ),
)
competing_fit = CumIncidenceRight(competing_time, competing_status)
sift_runtime.from_competing_risks(competing_fit)

# Andersen-Gill recurrent-event process with subject-clustered covariance.
subjects, intervals = 60, 4
recurrent_id: Any = np.repeat(np.arange(subjects), intervals)
recurrent_start = np.tile(np.arange(intervals), subjects).astype(float)
recurrent_stop = recurrent_start + 1.0
recurrent_x: Any = np.repeat(rng.binomial(1, 0.5, subjects), intervals)
event_probability = 1 / (1 + np.exp(-(-1.0 + 0.7 * recurrent_x)))
recurrent_status = rng.binomial(1, event_probability)
recurrent_fit = PHReg(
    recurrent_stop, recurrent_x[:, None], status=recurrent_status,
    entry=recurrent_start,
).fit(groups=recurrent_id)
sift_runtime.from_recurrent_events(recurrent_fit)

# Staggered within-subject covariate changes and at most one terminal event.
tv_id: list[int] = []
tv_start: list[float] = []
tv_stop: list[float] = []
tv_x: list[float] = []
tv_status: list[int] = []
for subject in range(subjects):
    exposure = int(rng.binomial(1, 0.5))
    for interval in range(intervals):
        if interval and rng.random() < 0.4:
            exposure = 1 - exposure
        event = int(rng.random() < (0.08 + 0.16 * exposure))
        tv_id.append(subject)
        tv_start.append(float(interval))
        tv_stop.append(float(interval + 1))
        tv_x.append(float(exposure))
        tv_status.append(event)
        if event:
            break
tv_fit = PHReg(
    np.asarray(tv_stop), np.asarray(tv_x)[:, None],
    status=np.asarray(tv_status), entry=np.asarray(tv_start),
).fit(groups=np.asarray(tv_id))
sift_runtime.from_time_varying_survival(tv_fit)
