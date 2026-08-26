---
name: Survey-weighted analysis
description: Judgment guide for analyzing survey data with sampling weights, strata, or clusters correctly.
when_to_use: The dataset came from a survey with sampling weights, a complex design (strata/PSU/clusters), or the researcher mentions "survey weights", "design effect", or a named survey (a governmental household or health survey, an omnibus panel, etc.).
---
# Survey-weighted analysis

This is judgment guidance on getting survey-design mechanics right,
not a new payload shape — every result here still crosses through the
same helpers (`sift.from_lm`, `sift.from_summarize`, etc.) as any
other analysis. The judgment is in HOW the script computes the
estimate before handing it to a helper.

## The core mistake to catch

Running an unweighted `mean()` or an unweighted OLS on survey data
when a weight variable exists in the dataset silently answers a
different question than the one being asked — the SAMPLE's average,
not the POPULATION's. If the codebook or schema shows a variable
named something like `weight`, `wt`, `pweight`, `svy_weight`, or a
documented sampling-weight column, ask (or infer from context) whether
the researcher wants a population-representative estimate before
writing any script. Most of the time for survey data, the answer is
yes, and skipping the weight is the single most common analysis error
in applied survey work.

## Three different things people call "weights"

- **Probability weights (design/sampling weights).** Inverse
  selection probability, used to make sample statistics represent the
  population. This is almost always what a `weight`/`pweight` column
  in survey data means. In Python, statsmodels' `WLS` accepts these as
  `weights=`, but note statsmodels' OLS/WLS standard errors under
  simple `weights=` do NOT automatically account for the survey design
  (clustering, stratification) — see below. In R, `survey::svyglm`
  with a declared `svydesign` object is the correct tool and handles
  both weighting and design-based SEs together; a plain `lm(weights=)`
  does not.
- **Frequency weights.** Each row represents `w` identical observations
  (common after aggregating microdata). `statsmodels` `freq_weights=`
  is the right argument here — different from `var_weights=`, which
  assumes each row is already an average of `w` observations with
  variance scaled accordingly. Getting `freq_weights` vs `var_weights`
  backwards silently changes the standard errors, not the point
  estimate — a script that runs without error can still be wrong.
- **Post-stratification / raking weights.** Adjust the sample to match
  known population margins (age × sex × region, etc.) on top of base
  design weights. Treat these like probability weights for estimation
  purposes; the distinction mostly matters for how the weights were
  CONSTRUCTED, not how they're used in `WLS`/`svyglm`.

If it's unclear which kind of weight column the dataset has, say so
and ask, rather than guessing — the three give different SEs from the
same weight VALUES.

## Standard errors: the part people skip

A population estimate computed with the right weight but the wrong
standard error is still wrong for inference. Complex survey designs
typically need at least one of:

- **Clustering by PSU (primary sampling unit).** Observations within
  the same cluster (household, school, village) are correlated;
  ignoring this understates SEs, sometimes drastically. If a PSU or
  cluster-ID column exists, use it — `cov_type="cluster"` with
  `cov_kwds={"groups": psu_id}` in statsmodels, or a declared
  `svydesign(ids=~psu, ...)` in R's `survey` package.
- **Stratification.** If a strata variable exists, R's `survey`
  package folds it into `svydesign(strata=~stratum, ...)` and adjusts
  SEs accordingly; a plain clustered-SE regression in Python/statsmodels
  has no direct strata argument — note this limitation to the
  researcher rather than silently dropping stratification.
- **Replicate weights (BRR, jackknife).** Some survey products (many
  national statistical agency releases) ship replicate weight columns
  instead of a strata/PSU design. If present, the correct SE comes
  from the replicate-weight variance formula for that survey's
  documented method, not from a generic clustered/robust SE on the
  main weight. Flag this explicitly — it needs survey-specific code,
  not a generic `sift.from_lm` call.

## What to tell the researcher

- Name which weight variable was used and what kind (probability /
  frequency / post-stratification) in the findings, not just "weighted
  regression."
- If the analysis is unweighted because no design information was
  available or usable, say that plainly — "no sampling weight was
  applied; this describes the achieved sample, not necessarily the
  target population" — rather than presenting an unweighted estimate
  as if it answers the population question.
- If a design effect (ratio of the design-based variance to the
  simple-random-sample variance) is computable and large, mention it —
  it's a useful signal for how much the complex design matters here,
  independent of the point estimate itself.

## Validation checklist

- Confirm the weight type against the survey documentation.
- Verify strata, PSU, FPC, and replicate-weight roles before fitting.
- Report weighted sample support, effective sample size, and design effect.
- Check lonely-PSU handling and whether certainty strata are represented.
- Compare the design-based result with a clearly labelled unweighted sensitivity.
