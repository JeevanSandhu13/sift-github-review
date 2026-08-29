# Sift architecture

Sift is a local desktop research assistant. A researcher-selected model plans
and authors analyses, while Sift controls data access, executes code locally,
and decides what may cross the model boundary.

This document defines the stable architecture and its security invariants. See
[`overview.md`](overview.md) for product behavior,
[`extending_analysis_shapes.md`](extending_analysis_shapes.md) for extension
mechanics, and [`verification.md`](verification.md) for qualification.

## System boundary

The model does not receive a shell, filesystem, database connection, or general
network tool. It can call only Sift's registered operations. Provider-native
web search, code execution, computer use, URL retrieval, and hosted tool
surfaces are disabled and checked when each request is built.

Generated R, Stata, or Python code runs locally under the native confinement
backend:

- macOS: `sandbox-exec` with a deny-by-default profile;
- Windows: AppContainer with Job Objects and a mandatory live denial probe;
- Linux: `bubblewrap` with namespace isolation and a mandatory baseline probe.

Every backend denies network access, exposes reviewed read paths, grants writes
only to the workspace and current run, hides private `.sift` state, strips the
subprocess environment to an allowlist, and enforces resource limits. If the
backend or its probe is missing, failed, or inconclusive, generated-code
execution is refused.

## Data flow

1. The researcher opens files, a folder, or a reviewed external-data result.
2. Sift canonicalizes supported inputs locally and records source lineage.
3. The active dataset policy limits schema depth, banned variables, export
   behavior, disclosure thresholds, and optional differential privacy.
4. The model proposes work through the fixed tool interface.
5. Sift validates methodology and workflow state before consequential analyses.
6. Generated code runs in the platform sandbox and writes framed result files.
7. Sift validates framing, sanitizes the declared result shape, computes
   deterministic verification checks, stores the result, and records any
   model-visible disclosure.
8. The provider receives only the approved prompt material, sanitized result,
   and allowlisted plot attachments.

Database, warehouse, object-store, and research-service connections remain
host-side and researcher initiated. Credentials never enter the generated-code
sandbox or provider prompt. Database results are materialized into the local
session before ordinary policy and disclosure controls apply.

## Runtime-library contract

The files under `src/sift/runtime/` are the only maintained adapters from an
analysis runtime into Sift result envelopes. Python, R, and Stata helpers must
agree on field meaning, missing-value behavior, sample-size semantics, and
analysis identifiers.

A result is admitted only when all of the following hold:

1. It is framed for the current run and carries the expected per-run integrity
   token.
2. Its result type is registered and has an explicit field allowlist.
3. Required fields are present after type filtering.
4. Shape-specific invariants, size limits, and disclosure-control rules pass.
5. Every data-origin string passes the text-safety boundary.
6. The payload is bounded before storage and before provider delivery.

The integrity token rejects stale helpers and trivial direct writes. It is not
semantic attestation: generated code shares an interpreter with the helper and
could deliberately recover the token or mislabel a calculation. Sift therefore
does not claim protection from a malicious selected model provider.

Adding a helper file also requires adding it to the executor's staging surface.
Tests enforce that the runtime directory and staged helper inventory stay in
sync.

## Statistical-method contract

The methodology registry defines supported methods, required research roles,
assumptions, limitations, runtime guidance, and required diagnostics. The
workflow layer binds the researcher's intent, estimand, primary analysis,
sensitivity analyses, approvals, and deterministic seeds before execution.

Typed runtime helpers are preferred over generic result construction because
they bind maintained fit objects or raw aggregates to required diagnostics. A
missing diagnostic is reported as missing; it is never treated as a pass.

### Shared coefficient result shapes

Instrumental-variable fits use the maintained linear-regression result envelope
with explicit IV fields such as instrument names, first-stage strength,
endogeneity tests, and over-identification diagnostics. They do not introduce a
parallel result shape. The sanitizer admits only the bounded IV aggregates, and
verification recognizes the IV fields to apply the appropriate checks and
causal-language constraints.

The same extension pattern applies to supported GLM, Cox, fixed-effects, and
mixed-effects fits: share the coefficient-table core, add only registered
bounded diagnostics, and keep method-specific semantics in the registry and
verification layers.

## Provider architecture

Provider adapters implement one internal event contract for assistant text,
thinking, tool calls, tool results, authentication failures, usage, and turn
completion. Tool schemas are generated from one provider-neutral inventory.
Each adapter must prove that provider-native capabilities remain disabled and
that context compaction cannot silently remove the system boundary.

Researchers bring their own provider account or compatible endpoint. Sift does
not bundle, resell, or fund model access. Secrets are stored through the
operating system credential store and are never written to Sift settings in
cleartext.

## Desktop architecture

The native application hosts local HTML, CSS, and JavaScript through pywebview.
The interface communicates with Python through the in-process bridge; there is
no localhost API server and no remote frontend asset dependency. Native file
dialogs, credential stores, web engines, installers, and confinement backends
remain platform specific while the interaction model stays shared.

See [`desktop_interface.md`](desktop_interface.md) for interaction,
accessibility, and cross-platform UI requirements.

## Local state and provenance

Each session owns its working directory and private `.sift` state. That state
contains sanitized results, chat history, workflow state, checkpoints,
provenance, and a hash-chained disclosure ledger. Concurrent sessions use
task-local working-directory context and independent provider sessions so a UI
focus change cannot redirect an in-flight turn.

Replication packages and reports are derived from stored sanitized material.
Raw datasets are never copied into those exports. Export approval and
institutional policy are evaluated locally before a file is produced.

## Security invariants

- Raw rows, raw free text, raw stdout, and raw stderr never reach the model.
- Credentials, connection strings, and unrestricted local paths never reach
  generated code or model-visible diagnostics.
- Provider-native tools stay disabled on every request.
- Generated code never runs without a positively verified native sandbox.
- Every model-visible result passes the sanitizer; no caller bypasses it by
  constructing a provider response directly.
- Unknown policy values and malformed security state fail closed.
- Data-origin labels cannot become instructions merely because they appear in
  JSON, Markdown, a plot manifest, an error, or a filename.
- Researcher-only plots built from row-level values are never attached to the
  model.
- No result is described as verified unless its deterministic check actually
  ran.
- A platform-neutral unit test is not native release evidence. macOS, Windows,
  and Linux artifacts pass their own build and installation lanes.

## Limits and deployment assumptions

Sift's controls reduce accidental disclosure and constrain a non-adversarial
provider whose generated code may be mistaken or prompt-injected. They do not
cryptographically prove the meaning of code running in the same interpreter.

The release ledger and adaptive suppression make cumulative exposure visible
and increasingly conservative, but they are not a formal privacy proof.
Differential privacy is a separate, explicit opt-in mechanism. Deployments that
include a malicious provider, a malicious analyst, or cross-user access require
additional governance and a narrower host-owned execution language.
