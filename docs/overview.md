# Sift overview

Sift is a desktop research assistant for working with sensitive data. It gives
a researcher access to model-assisted analysis without giving the model direct
access to the underlying dataset.

The application runs on macOS, Windows, and Linux. Data remains in a local
workspace, analysis code runs on the researcher's computer, and only
policy-approved statistical results can return to the selected model.

## The problem

Clinical records, survey responses, administrative data, student outcomes,
financial histories, and proprietary business data are often governed by
ethics approvals, contracts, law, or institutional policy. Uploading those
records to a general-purpose model may be prohibited or unnecessary.

At the same time, a capable model can help frame a question, choose an
analysis, write code, interpret diagnostics, and challenge a result. Sift is
designed to separate that reasoning work from unrestricted access to the data.

## A Sift session

1. **The researcher selects the data.** Sift opens local files and folders or
   materializes a user-approved extract from a database, cloud store, or
   research service.
2. **Sift profiles it locally.** The application records structure, variable
   types, missingness, ranges, and likely identifiers.
3. **The researcher sets the permission level.** This determines which schema
   fields and bounded summaries may enter model context.
4. **The model proposes and writes the analysis.** It works through a fixed
   Sift tool set rather than a general shell, filesystem, database, or web
   interface.
5. **The script runs beside the data.** macOS uses a deny-by-default sandbox,
   Windows uses AppContainer, and Linux uses Bubblewrap. Network access is
   denied and filesystem access is restricted.
6. **Sift reviews the result before disclosure.** The output must match a
   registered statistical shape and pass suppression, precision, dominance,
   size, and text-safety checks.

The researcher can inspect the raw data, complete local output, generated code,
plots, verification results, and disclosure history. The model receives the
researcher's messages, the permitted schema, sanitized statistical results,
approved aggregate figures, and redacted errors.

## Models and credentials

Sift can use Anthropic, OpenAI, Gemini, selected enterprise model deployments,
and OpenAI-compatible local or remote endpoints. The researcher supplies and
pays for the chosen account. Sift does not operate a model service.

Credentials are stored in macOS Keychain, Windows Credential Manager, or a
Freedesktop Secret Service-compatible vault. They are not placed in the model
prompt, generated-code environment, or a plaintext Sift settings file.

## Data and methods

Common research formats include CSV, Excel, Stata, SPSS, SAS, R, Parquet,
Arrow, ORC, and JSON. Optional format packs cover scientific, geospatial,
clinical, and genomic data. Sift can also create read-only local extracts from
major relational databases, warehouses, object stores, and research
repositories.

The maintained method library covers descriptive and inferential statistics,
regression, longitudinal and mixed models, survival analysis, survey
estimation, missing data, time series, prediction, measurement, Bayesian
workflows, study design, and causal inference. Deterministic checks accompany
results so missing diagnostics and fragile assumptions are visible rather than
silently treated as passed.

## Records and outputs

The local session preserves the analysis plan, scripts, results, verification
verdicts, source provenance, model usage, and each disclosure made to the
model. Sift can turn that record into reports, codebooks, governance summaries,
AI-use statements, and replication packages that exclude the raw data.

## The boundary and its limit

Sift's controls are designed for accidental disclosure, ordinary model errors,
and data-origin prompt injection. The current generated-code runtime is not a
defence against a model provider that is deliberately trying to encode raw
values into an aggregate-shaped result. Generated code and Sift's result
helpers share an interpreter, so the application cannot cryptographically
attest the meaning of every calculation.

Researchers whose threat model includes a malicious provider should use a
fully local model in the same trusted environment or leave generated-code
execution disabled. See [Security policy](../SECURITY.md) and
[Architecture](architecture.md) for the exact contract.

## Availability

Sift 0.1 is a public beta:

- macOS: signed and notarized for Apple silicon (`Sift.dmg`)
- Windows: unsigned x64 installer (`Sift-Windows-x64-Setup.exe`)
- Linux x86_64: `Sift-Linux-x86_64.tar.gz`
- Linux ARM64: `Sift-Linux-aarch64.tar.gz`

Start with [Installing Sift](install.md). The broader motivation and design are
described in
[What If an AI Analyst Never Saw Your Raw Data?](https://sapieninstitute.org/projects/sift).
