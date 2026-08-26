# Product direction

## The problem

Many valuable research datasets cannot be pasted into a general-purpose chat
tool. Patient records, student data, linked administrative records, private
documents, and proprietary panels may be constrained by ethics approval,
contract, regulation, or institutional policy. Researchers still benefit from
model assistance with study design, analysis code, diagnostics, and
interpretation—provided the data boundary remains explicit and enforceable.

## The product contract

Sift keeps raw-data access and computation local while a researcher-selected
model plans analyses and interprets disclosure-controlled results.

```text
SELECTED MODEL        plans, requests, and interprets
       |
PRIVACY BOUNDARY      policy, disclosure control, and accounting
       |
LOCAL EXECUTION       native confinement and deterministic checks
       |
RESEARCH DATA         files, databases, and approved cloud sources
```

The model receives researcher messages, policy-permitted schema, sanitized
statistical results, and explicitly permitted media. It does not receive raw
rows, credentials, unrestricted files, raw process output, or a general shell.
The precise same-interpreter limitation for generated analysis code is stated
in the [architecture](architecture.md) and [security policy](../SECURITY.md).

## Current foundation

- One enumerated tool surface shared across supported model providers.
- Researcher-supplied Anthropic, OpenAI, Gemini, enterprise-cloud, local, and
  OpenAI-compatible endpoints.
- Native fail-closed script confinement on macOS, Windows, and Linux, with
  execution refused when the platform probe cannot establish the boundary.
- Per-result statistical disclosure control, injection hardening, release
  accounting, and deterministic methodological checks.
- Python, R, and Stata runtime helpers for the maintained analysis contracts.
- Local files, reviewed database extracts, cloud-source materialization, and
  out-of-core DuckDB/Arrow paths.
- Reports, codebooks, disclosure records, checkpoints, and replication
  packages generated from local session state.
- A native desktop interface with the same product language and interaction
  model across supported platforms.

Support in source code is not the same as native or vendor certification.
Platform-specific behavior must pass on the target operating system, and live
services require disposable infrastructure owned or authorized by the tester.
The [verification guide](verification.md) and
[qualification inventory](qualification_inventory.json) keep those claims
separate.

## Priorities

### Strengthen cumulative disclosure controls

The release ledger records what crossed the boundary, and session-level checks
detect several repeated-query patterns. Further work should make cumulative
disclosure limits easier to understand and administer without calling ordinary
accounting formal differential privacy.

### Improve institutional deployment

Enterprise policy already supplies an administrator-controlled floor. The next
step is simpler deployment, reviewed policy templates, identity integration,
and export-approval workflows suitable for managed research environments.

### Expand deterministic analysis plans

Generated code is flexible but has an explicit semantic-attestation limit.
More host-owned, typed analysis plans can reduce that reliance for common
methods while preserving reproducibility and provider independence.

### Deepen long-running research workflows

Sift should make assumptions, decisions, contradictions, robustness checks,
and provenance easier to follow across a project—not merely within one chat
turn.

### Tighten the path to publication

Replication packages and structured exports reduce transcription. Continued
work should improve journal-quality figures, methods text bound to executed
operations, and reproducibility checks that clearly report drift.

## Non-goals

- Claiming privacy or compliance that the implementation cannot demonstrate.
- Weakening the local boundary to make a feature more convenient.
- Presenting model judgment as deterministic verification.
- Hiding provider disclosures, costs, generated code, or local execution logs
  from the researcher.
- Bundling or funding model access; researchers choose and pay their provider.

Success means that a researcher and their reviewer can understand exactly what
Sift did, what left the machine, which checks ran, and which claims still depend
on external qualification.
