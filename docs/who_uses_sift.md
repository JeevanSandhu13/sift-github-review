# Who Sift is for

Sift is for researchers and analysts who want help from a capable model but
cannot treat a confidential dataset as ordinary chat input.

The common requirement is not a particular profession. It is the need to keep
source records and computation inside a trusted local environment while still
using a model to plan, write, and interpret an analysis.

## Research and organizational settings

Sift is designed for work such as:

- **Health and clinical research:** observational studies, outcomes research,
  epidemiology, quality improvement, and trial-support analysis.
- **Social science and public policy:** surveys, administrative records,
  panels, program evaluation, and causal inference.
- **Education:** student outcomes, institutional research, assessment, and
  learning analytics.
- **Government and nonprofit research:** labour, housing, health,
  humanitarian, census, and monitoring-and-evaluation data.
- **Regulated or proprietary analysis:** finance, insurance, workforce,
  manufacturing, customer, market, risk, and operational data.
- **Confidential text and mixed-method work:** local extraction followed by
  deliberately bounded summaries.

These examples do not establish compliance with a law, contract, ethics
protocol, or institutional policy. The researcher and organization remain
responsible for authorizing the dataset, model provider, endpoint, analysis,
and disclosure settings.

## What a typical workflow looks like

A researcher might use Sift to:

1. profile a dataset and inspect its quality without exposing rows to a model;
2. develop an analysis in Python, R, or licensed Stata under native
   operating-system confinement;
3. review statistical checks alongside the model's interpretation;
4. compare alternative specifications without losing the analytical record;
5. produce a codebook, report, disclosure record, AI-use statement, or
   replication package from the stored session.

Sift works with ordinary files as well as user-approved, read-only extracts
from databases, warehouses, object stores, and research services. In every
case, the selected material is brought into the local workspace before
analysis.

## When to choose something else

Sift is not the right tool when:

- policy prohibits any external model use, including sanitized results;
- the selected provider is considered deliberately malicious;
- a workflow requires sending raw records to a remote model;
- the task depends on a method or data format that has not been reviewed for
  the current release;
- the work requires a compliance certification that Sift does not hold.

Sift also does not replace statistical judgment, domain review, ethics
approval, data-governance review, or independent security assessment.

Read [Sift overview](overview.md) for the workflow,
[Architecture](architecture.md) for the enforced boundary, and
[Security policy](../SECURITY.md) for the documented limitations.
