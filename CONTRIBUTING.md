# Contributing to Sift

Sift handles confidential research data, so changes are reviewed against both
correctness and the privacy boundary. Small, focused changes with explicit
tests are easier to evaluate than broad rewrites.

## Set up a development environment

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --locked --all-extras --group dev
uv run pytest -q
```

Launch the desktop app with `uv run sift`. Platform prerequisites and native
build instructions are in [`docs/install.md`](docs/install.md).

## Repository map

- `src/sift/` contains the application, provider adapters, privacy boundary,
  research workflow, and desktop bridge.
- `src/sift/runtime/` contains the Python, R, and Stata result helpers that
  produce sanitizer-compatible result envelopes.
- `src/sift/web/` contains the dependency-free desktop interface.
- `tests/` contains unit, integration, security-boundary, scientific-method,
  and packaging qualification coverage.
- `packaging/` contains native build, signing, installation, and qualification
  tooling for macOS, Windows, and Linux.
- `scripts/` contains maintainer-run qualification and differential tests.
- `siftbench/` contains deterministic synthetic benchmark cases.

The [documentation index](docs/README.md) points to the architecture, privacy,
methodology, verification, and packaging references.

## Engineering conventions

- Use descriptive module and test names. Regression tests should name the
  behavior they protect, not a ticket, review batch, or chronology.
- Comments should explain invariants, security assumptions, non-obvious
  tradeoffs, or external constraints. Do not use comments as a change log.
- Keep provider-specific logic behind the provider interfaces and keep raw
  data out of provider payloads.
- Treat every model-visible string and numeric result as crossing a security
  boundary. New result shapes require an explicit sanitizer allowlist and
  adversarial tests.
- Fail closed when a sandbox, credential store, policy file, or integrity check
  is unavailable or inconclusive.
- Keep generated files, local environments, credentials, session state, and
  native build outputs out of source control. The root `.gitignore` lists the
  expected local artifacts.

## Testing changes

Run the narrowest relevant tests while developing, then the complete suite:

```bash
uv run pytest tests/test_relevant_area.py -q
uv run pytest -q
```

Changes to the executor, sanitizer, provider adapters, credential handling,
database connectors, or native packaging require the corresponding manual or
platform qualification in [`docs/verification.md`](docs/verification.md).
Tests that require a licensed runtime, third-party account, or native operating
system must skip with an explicit reason; a skip is not release evidence.

## Security reports

Do not open a public issue for a suspected vulnerability. Follow
[`SECURITY.md`](SECURITY.md) for private reporting.

