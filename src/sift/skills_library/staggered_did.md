---
name: Staggered-adoption difference-in-differences
description: Judgment guide for choosing and defending a DiD estimator when treatment timing varies across units.
when_to_use: The researcher asks for a DiD, event study, or "effect of policy X" and units adopted treatment at different times (not a single clean cutoff date for everyone).
---
# Staggered-adoption difference-in-differences

This is judgment guidance, not mechanics. For how to actually emit a
`did_event_study` payload from a fitted model, see the tool
documentation for `sift.from_callaway_santanna`,
`sift.from_sun_abraham`, and the `did_event_study` shape — this skill
is about which estimator to reach for and why, and the pitfalls that
make naive two-way fixed effects (TWFE) actively misleading here.

## Why staggered timing breaks naive TWFE

A standard TWFE regression (`y ~ treated + unit_FE + time_FE`) with
staggered adoption implicitly uses ALREADY-TREATED units as part of
the control group for later-treated units' comparisons. If treatment
effects change over time (grow, shrink, or reverse), those
"forbidden comparisons" contaminate the pooled coefficient — sometimes
enough to flip its sign relative to the true average effect (Goodman-
Bacon 2021; de Chaisemartin & D'Haultfœuille 2020). This is not a
minor technicality: a negative estimated effect can coexist with every
single unit-level effect being positive. Do not fit staggered-adoption
DiD as a wide-coefficient OLS and report the treatment coefficient at
face value. Flag this risk explicitly if the researcher's dataset has
staggered timing and they're heading toward a plain TWFE spec.

## Choosing an estimator

- **Callaway & Sant'Anna (2021)** — the default first choice for most
  applied staggered-DiD questions. Computes group-time ATT(g,t) for
  each treatment cohort against a clean control group (never-treated,
  or not-yet-treated), then aggregates. Handles heterogeneous and
  dynamic treatment effects without the forbidden-comparison problem.
  Ask the researcher whether a never-treated group exists in their
  data — if not, `comparison_group="notyettreated"` is the fallback,
  but note it's a slightly weaker identification story (still-untreated
  units may differ systematically from never-treated ones).
- **Sun & Abraham (2021)** — an interaction-weighted event-study
  estimator, useful when the researcher specifically wants a clean
  per-relative-time coefficient path (an event-study plot) rather
  than group-time cells. Similar robustness to CS; different
  aggregation mechanics.
- **de Chaisemartin & D'Haultfœuille** — another heterogeneity-robust
  estimator, useful as a robustness cross-check against CS or when CS's
  parallel-trends variant doesn't fit the design. No first-class Python
  helper ships yet (see the `did_event_study` shape documentation for
  the current per-language coverage) — if the researcher wants this
  specifically, say so plainly rather than silently substituting CS.

## Design questions to ask before fitting anything

1. **Is there a genuine never-treated group?** If everyone is treated
   eventually, the comparison group must be "not yet treated" units,
   which is a weaker assumption (their future treatment could
   correlate with time-varying confounders). Say this out loud in the
   findings, don't bury it in a footnote.
2. **Could treatment timing itself be endogenous?** Units that adopt
   early are rarely a random subset. If earlier adopters differ
   systematically (bigger firms, richer regions, sicker patients),
   that's a threat to parallel trends independent of the estimator
   chosen. This is a design problem, not something CS or Sun-Abraham
   fixes for you.
3. **Anticipation.** Do units change behavior BEFORE the recorded
   treatment date (e.g., a policy announced before it takes effect)?
   If plausible, use `anticipation_periods` (or the base-period
   analogue) rather than silently assuming zero anticipation.
4. **Event-time window.** Cohorts observed for only 1-2 post-periods
   contribute little to a dynamic aggregation and their pre-period
   coefficients are noisy. Ask whether the researcher wants a
   balanced-panel restriction (same event-time window for every
   cohort) — it trades sample size for cleaner comparability across
   the event-time path.

## Reporting

- Report the AGGREGATE ATT (simple or dynamic/event-time, per the
  researcher's question) alongside the per-event-time path, not just
  a single pooled number — the whole point of these estimators is
  that heterogeneity is visible, not averaged away by construction.
- Always name the estimator in the findings text (Callaway-Sant'Anna,
  Sun-Abraham, etc.) — "a difference-in-differences model" alone
  hides a real methodological choice from the reader.
- If a plain TWFE result is ALSO available (e.g. the researcher ran
  one first), report both and explain the direction/size of the gap
  in terms of the forbidden-comparison mechanism above — that
  contrast is itself informative, not just a robustness footnote.
- Cohort sizes below the disclosure threshold are dropped whole by
  Sift's sanitizer (the `did_event_study` shape's cohort-N gate) —
  if the researcher has many small cohorts, expect several to vanish
  from the reported table. Say so rather than letting a sparse table
  look like a data problem.
