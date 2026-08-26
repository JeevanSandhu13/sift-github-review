---
name: Predictive model validation
description: Leakage-resistant validation guide for classification, regression, calibration, and forecasting.
when_to_use: The goal is prediction, classification, risk scoring, forecasting, ranking, or deployment performance rather than explanation.
---
# Predictive model validation

## Fix the prediction problem

Declare the prediction time, target window, eligible population, available
features, action horizon, and loss or utility. Features must be genuinely
available at prediction time. Split by person, site, or time whenever records
share a source; random row splits can leak near-duplicates and future information.

## Assumptions and diagnostics

Keep tuning and preprocessing inside each training fold. Compare against a simple
baseline, report out-of-sample discrimination and calibration, and evaluate class
imbalance with metrics suited to the decision. Check temporal and site transport,
subgroup performance, missingness drift, and uncertainty around performance.

## Validation checklist

- Prove split integrity and remove cross-split duplicates.
- Record every seed, fold definition, package version, and dataset hash.
- Fit preprocessing only on training data.
- Compare a simple baseline and a calibrated model.
- Use rolling-origin evaluation for forecasts.
- Never interpret feature importance as a causal effect.
