# Contributing to Sift

Thank you for taking the time to improve Sift. The project handles workflows
that may involve confidential research data, so changes are reviewed for
correctness, reproducibility, and their effect on the privacy boundary.

## Before you start

For a small bug fix, documentation correction, or focused test improvement,
open a pull request directly. For a new provider, data connector, analysis
shape, or change to the security model, open an issue first so the contract can
be agreed before implementation.

Do not include real research data, credentials, session directories, native
build outputs, or private qualification evidence in an issue or pull request.
Use synthetic fixtures with known properties.

## Development setup

Sift requires Python 3.10 or later and uses
[uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync --locked --all-extras --group dev
uv run pytest -q
uv run sift
```

Platform prerequisites and packaged-app instructions are in
[Installing Sift](docs/install.md).

## Repository guide

| Path | Purpose |
| --- | --- |
| `src/sift/` | Application code, provider adapters, workflow, privacy controls, and desktop bridge |
| `src/sift/runtime/` | Python, R, and Stata helpers that produce registered result envelopes |
| `src/sift/web/` | Dependency-free desktop interface |
| `tests/` | Unit, integration, security, method, and packaging coverage |
| `packaging/` | Native builds, installers, signing, and platform qualification |
| `scripts/` | Maintainer-run qualification and differential tests |
| `siftbench/` | Deterministic benchmark cases with known synthetic truth |
| `docs/` | User, architecture, verification, and maintainer documentation |

## Engineering rules

- Keep provider-specific behavior behind the provider interfaces.
- Never place raw data or credentials in a provider payload.
- Treat every model-visible string, number, error, and image as a disclosure
  boundary.
- Give every new result shape an explicit sanitizer allowlist, cross-field
  validation, and adversarial tests.
- Fail closed when confinement, credential storage, policy loading, or an
  integrity check is unavailable or inconclusive.
- Explain security assumptions and non-obvious trade-offs in comments. Do not
  use source comments as a development diary.
- Name tests for the behavior they protect rather than a ticket number or
  implementation chronology.
- Keep generated files and machine-specific state out of source control.

The detailed result-shape process is documented in
[Extending analysis coverage](docs/extending_analysis_shapes.md).

## Testing

Run the smallest relevant test while developing, then run the complete suite:

```bash
uv run pytest tests/test_relevant_area.py -q
uv run pytest -q
```

Changes to the executor, sanitizer, providers, credentials, database
connectors, or packaging also require the matching manual or native
qualification in [Verification](docs/verification.md). A test that requires a
licensed runtime, third-party account, or specific operating system must skip
with a clear reason when that dependency is unavailable. A skip is not release
evidence.

Before opening a pull request:

- add or update tests for the behavior you changed;
- run the complete test suite;
- update user or maintainer documentation where the contract changed;
- confirm no credential, dataset, session, or build artifact is included;
- describe any check you could not run and why.

## Pull requests

Keep a pull request focused enough to review as one change. Explain the user or
research need, the approach, the tests run, and any effect on privacy,
compatibility, or release packaging.

Security vulnerabilities should not be reported through a public issue or pull
request. Follow [SECURITY.md](SECURITY.md) for private reporting.
