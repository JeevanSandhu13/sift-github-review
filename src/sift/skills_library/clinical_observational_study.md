---
name: Clinical observational study
description: Design and validation guide for observational clinical and health-record research.
when_to_use: The researcher is analyzing patient, encounter, registry, claims, treatment, diagnosis, or clinical outcome data without randomized assignment.
---
# Clinical observational study

## Design before estimation

Define time zero, eligibility, treatment assignment, follow-up, outcome window,
and censoring before examining effects. Check that predictors are measured at or
before time zero; post-treatment features create leakage and can induce bias.
Distinguish people, encounters, and records, and never treat repeated encounters
as independent people.

## Assumptions and failure modes

State the causal or associational estimand. For causal work, identify exchangeability,
positivity, consistency, and interference assumptions. Check immortal-time bias,
informative censoring, treatment switching, prevalent-user bias, and missingness.
Claims and billing codes are proxies for care, not direct clinical truth.

## Validation checklist

- Verify unique patient and encounter keys and temporal ordering.
- Report people, records, outcomes, and censoring separately.
- Inspect overlap and covariate balance for treatment comparisons.
- Use patient-level grouping for splits and variance estimation.
- Challenge results with outcome definitions, grace periods, and censoring rules.
- Keep every conclusion within the approved design and evidence citation.
