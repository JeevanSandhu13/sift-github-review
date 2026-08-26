---
name: Longitudinal and panel analysis
description: Validation guide for repeated measures, panels, growth curves, and temporal dependence.
when_to_use: The same units appear at multiple times, the researcher mentions panels, waves, repeated measures, trajectories, fixed effects, or growth.
---
# Longitudinal and panel analysis

## Declare the panel

Identify the unit key, time variable, cadence, duplicated unit-time rows, entry
and exit rules, and whether the panel is balanced. A record count is not a unit
count. Decide whether the estimand is within-unit, between-unit, or population-
average before selecting fixed effects, random effects, GEE, or a growth model.

## Assumptions and diagnostics

Check serial correlation, time-varying confounding, missing waves, informative
attrition, and sufficient within-unit variation. Random-effects models require a
defensible relationship between unit effects and covariates; otherwise compare a
fixed-effects specification. Cluster uncertainty at the dependence unit unless a
stronger design-specific estimator applies.

## Validation checklist

- Verify unit-time uniqueness and chronological sorting.
- Report units, records, waves, and cluster-size distribution separately.
- Compare within-unit support with the intended estimand.
- Use grouped or temporal validation, never a random row split.
- Challenge cadence, lag, time trend, and covariance assumptions.
- Mark attrition and observation-window limitations explicitly.
