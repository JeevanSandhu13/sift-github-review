# Security policy

Sift's core claim is narrower: it does not directly upload raw dataset
files or rows, generated analysis is confined without network access, and
model-visible results pass through disclosure controls. This assumes the
selected provider and its generated code are not deliberately adversarial.
Reports that break the boundary under that documented assumption—or affect
the tool interface, OS confinement, sanitizer, or credential storage—take
priority.

## Trust boundary

Generated analysis code shares an interpreter with Sift's runtime helpers.
The per-run token detects stale/missing runtime framing and trivial direct
writes, but it is not cryptographic attestation of a calculation's meaning.
Code intentionally reading the token can fabricate aggregate-shaped payloads
or mislabel source variables. Consequently, Sift is not certified to defend
confidential data from a malicious model provider that deliberately authors
bypass code. Researchers with that threat model must not enable generated-
code execution with that provider; a future constrained, host-owned analysis
plan/execution engine would be required to close this boundary.

## Reporting a vulnerability

Preferred channel: GitHub's private vulnerability reporting. Open the
**Security** tab on this repo and choose **Report a vulnerability**.
This routes a private advisory to the maintainer; the issue stays out
of public view until disclosure is coordinated.

Alternatively, email the maintainer at jeevan@sapieninstitute.org. Please do
not file public issues for security reports.

Please include:

- A minimal reproduction (steps, environment, Sift version).
- Which layer the issue affects (tool interface, sandbox, sanitizer,
  packaging, auth).
- Whether the issue is exploitable against the documented threat model
  (a non-adversarial provider whose generated code may be mistaken or exposed
  to data-origin prompt injection), requires a deliberately malicious
  provider/generated script, or requires a researcher to undermine safeguards.

## What is in scope

- Anything that lets the model observe raw data values, free-text
  values, or low-count cells past the sanitizer.
- Anything that lets the model issue arbitrary filesystem, network, or
  shell operations from inside the sandbox.
- Auth or keyring handling that exposes API keys to the model or
  writes them to disk in cleartext.
- Packaging issues (signing, notarization, supply chain) that affect
  the integrity of a downloaded native installer or archive.

## What is out of scope

- Native GUI packaging and release certification on a platform that has not
  completed its platform-specific release lane. The backend contains macOS,
  Windows, and Linux confinement implementations, but each must pass its real
  kernel probe and release tests on that operating system.
- A deliberately malicious selected model/provider authoring code to forge
  aggregate-shaped outputs or helper plot metadata; this is the explicit
  same-interpreter limitation above, not an implemented security guarantee.
- A researcher willingly pasting their own raw data into a chat
  message. The model can read what the researcher types; this is by
  design.
- Vulnerabilities in upstream dependencies that do not affect Sift's
  privacy invariants.

## Disclosure timeline

Best-effort acknowledgment within seven days. Fixes are prioritized by
severity. Coordinated disclosure is preferred so a fix can ship before
public detail.

## Supported versions

The current beta line (`0.1.x`) receives security fixes. After the first
stable release (`1.0.0`), this section will document the support window for
the previous minor.
