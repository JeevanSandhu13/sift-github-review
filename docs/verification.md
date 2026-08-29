# Verifying Sift

Sift separates tests by what they can establish. A unit test can verify a
parser or state machine. It cannot certify kernel confinement on another
operating system, validate a live database driver, or prove that a signed
installer opens on a clean computer.

This guide is for contributors and release maintainers. End users should use
the packaged [platform checks](install.md#platform-checks).

## Evidence levels

| Level | What it covers | Typical environment |
| --- | --- | --- |
| Automated tests | Application logic, disclosure controls, providers, connectors, methods, packaging contracts | Local development and hosted CI |
| Hosted platform qualification | Native build and compatibility checks available on GitHub-hosted systems | macOS, Ubuntu, Windows Server |
| Native client qualification | Kernel, renderer, installer, upgrade, uninstall, and confinement behavior on the supported desktop target | Clean macOS, Windows 11, and Linux clients |
| Live-service qualification | Authentication, transport, native types, cancellation, and read-only behavior against a real provider or database | Disposable accounts and infrastructure |
| Independent assessment | External penetration testing and remediation confirmation | Qualified third party |

A lower level is not a substitute for a higher one. Unavailable evidence is
reported as unavailable or skipped, never converted into a pass.

## Routine contributor checks

Install the complete development environment:

```sh
uv sync --locked --all-extras --group dev
```

Run the narrow test related to the change, then the complete suite:

```sh
uv run pytest tests/test_relevant_area.py -q
uv run pytest -q
```

The full suite includes sanitizer boundaries, generated-code framing,
credential behavior, connector policy, methods, exports, update trust,
packaging gates, and platform-planning tests. Some tests require R, licensed
Stata, a native operating system, or live credentials and will skip when their
documented dependency is unavailable.

Review every skip. A skip can be expected; it is not evidence that the skipped
behavior passed.

## Security-boundary checks

Changes to execution, sanitization, credentials, policies, connectors, release
trust, or provider tools require the related adversarial tests in addition to
the full suite. The principal test areas include:

- `tests/test_executor_sandbox.py`
- `tests/test_executor_sandbox_stata.py`
- `tests/test_win_appcontainer.py`
- `tests/test_private_state_security.py`
- `tests/test_security_assurance.py`
- `tests/test_release_manifest.py`
- `tests/test_manage_release_signing.py`

A native confinement check must demonstrate:

- permitted reads and writes succeed;
- an unrelated file read is denied;
- private `.sift` state is denied;
- an outside write is denied;
- outbound and listening network access are denied;
- timeouts and cancellation terminate descendants;
- temporary permissions, profiles, and run state are cleaned up.

If a denial probe returns real file content or reaches the network, stop the
release. Redacting the test output does not turn a failed boundary into a pass.

## Method and scientific checks

Method changes require:

1. sanitizer and shape-validation tests;
2. real fits against the maintained library or estimator;
3. deterministic verification checks;
4. cross-language comparison where more than one runtime implements the same
   method;
5. synthetic data with a known data-generating process;
6. a documented skip when a licensed runtime is unavailable.

The maintainer entry points under `scripts/` produce structured evidence:

- `method_qualification_evidence.py`
- `scientific_qualification.py`
- `performance_qualification.py`

Licensed Stata comparison must run on a machine with a valid Stata
installation. Sift reads Stata files without Stata, but that does not qualify
Stata code execution or cross-language numerical agreement.

## Provider, connector, and database checks

Provider and live-service tests use researcher- or maintainer-supplied
credentials. Sift does not include model usage or disposable vendor
infrastructure.

The structured entry points are:

- `provider_qualification.py`
- `database_qualification.py`
- `security_qualification.py`

Live tests must use disposable data and least-privilege credentials. Never
place a credential, connection string, account identifier, or private result in
CI logs or committed evidence.

Database qualification covers transport verification, supported
authentication paths, reviewed native types, read-only enforcement, query
cancellation, and vendor-specific behavior. The scenario inventory is
`live_database_certification.json`. A connector implementation can be
available before every vendor scenario has live certification; public wording
must preserve that distinction.

## Native desktop qualification

The repository defines three release lanes:

- `.github/workflows/platform-qualification.yml` for hosted macOS, Ubuntu,
  and Windows Server compatibility;
- `.github/workflows/linux-arm64-native-qualification.yml` for the ARM64
  Linux baseline;
- `.github/workflows/windows-11-native-qualification.yml` for a self-hosted
  Windows 11 x64 client.

Each native lane builds the artifact on its target operating system and checks
the installed layout, bundled assets, credential integration, platform report,
confinement, upgrade, and uninstall behavior available to that host.

Windows Server compatibility does not certify Windows 11 AppContainer
behavior. A macOS build does not establish Linux or Windows packaging. The
release notes must state the evidence actually available for each artifact.

## Clean-install smoke test

Use a clean user account or disposable machine that has not run Sift before.
Do not preinstall optional analysis packages unless the test specifically
requires them.

For each platform:

1. verify the artifact checksum and release statement;
2. install through the documented user path;
3. confirm the expected platform trust prompt;
4. open Sift and reach provider setup;
5. configure a test provider credential and run its connection check;
6. open locally generated sample data;
7. run a baseline descriptive result and regression;
8. inspect code, local output, sanitized result, verification, and disclosure
   record;
9. create one report and one replication package;
10. quit, reopen, and resume the session;
11. install the same version again or upgrade from the previous beta;
12. uninstall and confirm that research state is retained.

Also exercise one missing optional runtime or package. The application should
name the missing dependency, request approval before installation, and never
invent a result after execution fails.

## Platform trust checks

### macOS

The release pipeline must verify the nested signatures, hardened runtime,
notarization result, stapled ticket, and Gatekeeper assessment. The final
`Sift.dmg`, not an earlier copy, is the asset uploaded to GitHub.

### Windows

A general Windows release requires an Authenticode signature from a trusted
publisher and timestamping. An explicitly unsigned beta may be published with
the expected SmartScreen limitation, but it must not be described as signed or
native-qualified without the corresponding evidence.

### Linux

Verify both processor archives on their documented glibc baselines. The
installed platform check must confirm Qt WebEngine, Secret Service, Bubblewrap,
and the supported namespace policy. Do not disable Qt sandboxing or system-wide
security controls to make a release pass.

## Release checklist

Before creating a GitHub Release:

1. confirm the intended version and changelog;
2. require a clean source state;
3. run the complete automated suite and review skips;
4. complete each available native qualification lane;
5. record unavailable external evidence without presenting it as passed;
6. build artifacts on their target systems;
7. verify checksums, SBOM bindings, release statements, and the aggregate
   release manifest;
8. verify macOS or Windows platform signatures where required;
9. complete a clean-install smoke test;
10. upload immutable artifacts and checksums to a GitHub prerelease or release;
11. compare the uploaded asset hashes with the locally qualified files;
12. install once from the downloaded GitHub asset.

Release artifacts and generated evidence belong in `dist/` and are excluded
from source control.
