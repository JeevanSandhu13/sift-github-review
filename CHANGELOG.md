# Changelog

Notable changes to Sift are recorded here. The project uses
[Semantic Versioning](https://semver.org/). During the 0.x beta, minor releases
may still include compatibility changes.

## Unreleased

- No unreleased changes.

## 0.1.0-beta.1 — 2026-08-28

The first public beta of Sift.

### Highlights

- Local-first model-assisted analysis with a fixed tool interface, native
  operating-system confinement, and statistical disclosure controls.
- Researcher-supplied Anthropic, OpenAI, Gemini, enterprise, and
  OpenAI-compatible model connections.
- Python, R, and Stata analysis workflows with registered result shapes and
  deterministic verification.
- Local files, read-only database extracts, cloud objects, and research-source
  integrations.
- Session provenance, disclosure history, codebooks, reports, AI-use
  statements, and replication exports.

### Desktop releases

- macOS for Apple silicon: Developer ID signed, Apple notarized, stapled, and
  checked by Gatekeeper.
- Windows 11 x64: installer and portable archive released as an explicitly
  unsigned beta. SmartScreen or managed-device policy may block the build.
- Linux x86_64 and ARM64: per-user archives with Bubblewrap confinement and
  Secret Service credential storage.

### Release records

Each downloadable artifact is accompanied by a SHA-256 checksum, a CycloneDX
software bill of materials, and a Sift Ed25519 release statement. Native
artifacts are published through GitHub Releases rather than committed to the
source tree.
