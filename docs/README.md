# Sift documentation

## Start here

- [`overview.md`](overview.md) — product behavior and privacy model in plain
  language.
- [`install.md`](install.md) — source setup, native installation, upgrades,
  local state, and platform prerequisites.
- [`architecture.md`](architecture.md) — architecture, design decisions, and
  non-negotiable invariants.
- [`verification.md`](verification.md) — local, integration, and release
  qualification procedures.

## Research and analysis

- [`extending_analysis_shapes.md`](extending_analysis_shapes.md) — how to add
  a result shape without weakening disclosure control.
- [`vision.md`](vision.md) — product direction and explicit non-goals.
- [`who_uses_sift.md`](who_uses_sift.md) — intended research settings and
  representative workflows.
- [`beta_study_protocol.json`](beta_study_protocol.json) — structured beta
  evaluation protocol.

## Security and privacy

- [`security_threat_model.json`](security_threat_model.json) — machine-readable
  threats, controls, and linked tests.
- [`security/`](security/) — schemas for independent assessment attestations.
- [`windows_appcontainer.md`](windows_appcontainer.md) — Windows
  AppContainer design and native qualification boundary.

## Desktop and integrations

- [`desktop_interface.md`](desktop_interface.md) — desktop interaction,
  accessibility, and cross-platform interface contract.
- [`live_database_certification.json`](live_database_certification.json) —
  bring-your-own-database compatibility scenarios and evidence format.
- [`qualification_inventory.json`](qualification_inventory.json) —
  machine-readable qualification inventory used by the evaluation pipeline.

Generated qualification reports and native installers belong in `dist/` and
are intentionally excluded from the source repository.
