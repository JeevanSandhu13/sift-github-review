# Changelog

## Sift 0.1.0 Windows beta distribution update (2026-08-28)

The Windows installer and portable archive are available as explicitly
labelled unsigned beta artifacts. Their SHA-256 checksums, CycloneDX SBOMs,
and Sift Ed25519 release signatures remain published for integrity checking.
The installation guide documents expected SmartScreen and managed-device
behavior without asking users to disable Windows security controls.

## Sift 0.1.0 macOS distribution update (2026-08-28)

The public beta macOS artifact is now produced with hardened-runtime
Developer ID signing, Apple notarization, ticket stapling, Gatekeeper
assessment, and a launch smoke test. Release binaries are published as
GitHub Release assets rather than committed to the source repository.

## Sift 0.1.0 (2026-08-20)

Initial public beta release. The tool interface, sandbox contract, and
sanitizer allowlist are documented for outside contributors, while the
`0.1.x` line remains eligible for compatibility changes before 1.0.
