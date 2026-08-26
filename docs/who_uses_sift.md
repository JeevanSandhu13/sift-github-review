# Who Sift is for

Sift is designed for researchers and analysts who want model-assisted analysis
without directly uploading raw datasets to a model provider. It is most useful
when data is confidential, regulated, contractually restricted, proprietary,
or simply too sensitive for a general-purpose chat interface.

## Representative settings

- Health and clinical research: observational studies, outcomes research,
  epidemiology, quality improvement, and trial-support analysis.
- Social science and public policy: surveys, administrative records, panels,
  program evaluation, and causal-inference workflows.
- Education: student outcomes, institutional research, assessment, and learning
  analytics.
- Government and nonprofit research: census, labor, health, housing,
  humanitarian, and monitoring-and-evaluation data.
- Regulated and proprietary industry work: financial risk, insurance, people
  analytics, manufacturing quality, market research, and operational data.
- Confidential qualitative work: documents and free-text fields where local
  extraction and tightly bounded summaries are appropriate.

These examples describe possible uses, not automatic compliance with a law,
contract, ethics protocol, or institutional policy. The organization and
researcher remain responsible for authorizing the selected data, model
provider, endpoint, and analysis.

## Common workflows

Sift supports source files commonly used by researchers, including CSV, Excel,
Stata, SPSS, SAS, R, Parquet, and line-delimited JSON. It can also materialize
reviewed, read-only extracts from local databases, database servers, cloud
warehouses, and approved object-storage sources.

Typical workflows include:

1. Inspect a dataset's schema and quality without exposing rows to the model.
2. Develop and execute an analysis in Python, R, or licensed Stata under native
   OS confinement.
3. Review deterministic statistical checks alongside the model's explanation.
4. Challenge a finding with alternative specifications and compare results.
5. Export codebooks, disclosure records, reports, and replication materials
   from the stored local session.

## When Sift is not the right tool

Sift is not appropriate when a policy forbids any external model use, when the
selected provider is considered deliberately malicious, or when raw records
must be sent to a remote model. It also does not replace domain review,
statistical judgment, ethics approval, or independent security assessment.

See [the architecture](architecture.md), [security model](security_threat_model.json),
and [verification guide](verification.md) for the exact guarantees and limits.
