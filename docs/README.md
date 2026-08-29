# Sift documentation

The README is the quickest introduction to Sift. This directory contains the
longer references for researchers, institutional reviewers, contributors, and
release maintainers.

## Using Sift

- [Installing Sift](install.md) covers downloads, checksums, setup on macOS,
  Windows, and Linux, first-run configuration, upgrades, uninstalling, and
  troubleshooting.
- [Sift overview](overview.md) explains the product and privacy model without
  requiring knowledge of the codebase.
- [Who Sift is for](who_uses_sift.md) describes representative research and
  organizational settings, along with cases where another tool is a better
  choice.

## Understanding the system

- [Architecture](architecture.md) defines the data flow, trust boundary,
  provider interface, execution runtime, and non-negotiable security
  invariants.
- [Desktop interface](desktop_interface.md) records the interaction,
  accessibility, and cross-platform behavior expected of the desktop app.
- [Product direction](vision.md) describes the current product contract,
  priorities, and explicit non-goals.

## Contributing and verification

- [Contributing to Sift](../CONTRIBUTING.md) covers environment setup,
  repository structure, engineering conventions, and test expectations.
- [Extending analysis coverage](extending_analysis_shapes.md) is the
  implementation guide for adding result shapes without weakening disclosure
  controls.
- [Verification](verification.md) contains maintainer-run security, method,
  integration, and native release checks.
- [Windows AppContainer backend](windows_appcontainer.md) documents the
  Windows confinement design and native qualification boundary.

## Structured assurance records

The JSON files in this directory are inputs to Sift's automated assurance and
qualification tooling. They are maintained as machine-readable records rather
than narrative documentation:

- `security_threat_model.json`
- `live_database_certification.json`
- `qualification_inventory.json`
- `beta_study_protocol.json`

Schemas for independent security assessments are under `docs/security/`.
Generated qualification reports, installers, and local evidence belong in
`dist/` and are not committed to the repository.

For the motivation behind the project, read
[What If an AI Analyst Never Saw Your Raw Data?](https://sapieninstitute.org/projects/sift).
