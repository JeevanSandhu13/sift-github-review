# Sift

**Private intelligence for your data.**

Sift is a local-first AI data analyst. A researcher-selected model
(Anthropic, OpenAI, Google, or an OpenAI-compatible endpoint) drives
statistical analyses in R, Stata, or Python without Sift directly
uploading the raw dataset. The model works through a
narrow tool interface: no shell, no filesystem, no network. Scripts
run under a fail-closed OS confinement backend with network denied. Every result
passes through a statistical-disclosure-control sanitizer before the
model sees it. You see the raw output; the model only ever sees
sanitized summaries and plot images permitted by the active policy.

This boundary assumes the selected provider and its generated code are
not deliberately adversarial. Generated code shares an interpreter with
Sift's runtime helpers, so the per-run integrity token catches stale
framing and trivial bypasses but cannot cryptographically attest what a
calculation means. Sift is not certified to protect a dataset from a
provider intentionally writing code to encode raw values as apparently
valid aggregates. Researchers whose threat model includes a malicious
provider must not enable generated-code execution with that provider.

## How the privacy boundary works

Three independent layers, none of which depends on which model is
driving:

1. **Tool interface** — the model has a fixed, enumerated set of
   tools (schema inspection, bounded data requests, sandboxed script
   submission, result recall, analysis-plan updates). Provider
   built-ins (bash, filesystem, web) are disabled and verified
   per-request.
2. **Sandbox** — scripts run under the native fail-closed confinement
   backend (`sandbox-exec` on macOS, AppContainer on Windows, and
   `bubblewrap` on Linux): network denied, tight read/write subpath
   allowlists, and an environment-variable allowlist. Sift refuses to
   run scripts if the required backend or its live probe is unavailable.
3. **Sanitizer** — per-analysis-type field allowlists plus SDC rules
   (N-scaled precision clamping, small-cell suppression, dominance
   checks, minimum-N hard rejects) and injection-hardening of every
   data-origin string. Results carry a per-run integrity token that
   rejects missing/stale runtime framing and trivial direct writes; it
   is not semantic attestation against code in the same interpreter.

What crosses to the model: your messages, dataset schema up to a
per-dataset ceiling you control, sanitized statistical summaries,
redacted error framing, and model-output plots produced by
allowlisted helpers. What never crosses: raw rows, raw stdout/stderr,
file contents, credentials, and raw-data plots when generated code follows
the runtime contract. The adversarial-generated-code limitation above is
load-bearing.

See [`docs/vision.md`](docs/vision.md) for where this is going.

## Key capabilities

- **Release ledger** — a per-session, hash-chained, append-only
  record of every disclosure that crossed to the model (operation,
  variables, sample size, transformations, payload hash). This is
  disclosure *accounting*, not differential privacy; it makes the
  cumulative surface auditable rather than provably bounded.
- **Deterministic verification** — sanitized results are checked by
  code, not by the model, across every analysis shape Sift supports.
  Regressions get sample-size, observations-per-parameter, VIF,
  conditioning, suspicious-fit and robust-SE checks; difference-in-
  differences gets the pre-trend test and cohort sizes; RDD gets
  effective sample size, bandwidth symmetry and polynomial order;
  survival gets event counts and at-risk depth behind each horizon;
  clustering gets separation; factor analysis gets KMO, sphericity
  and fit. Thresholds are published conventions, cited in the source.
  Verdicts are shown per result; nothing is labeled verified unless a
  check actually ran, and a missing diagnostic is reported as missing
  rather than passed.

- **Across-result checks** — some of the most consequential
  statistical problems are invisible within a single result. Sift
  counts hypothesis tests across the whole session (forty
  individually-clean tests should produce about two "significant"
  results from noise alone) and flags sample-size drift between
  results on the same dataset, which means a filter or merge moved
  the population mid-analysis.

- **Bounded repair** — when scripts fail repeatedly, Sift tells the
  model to stop retrying and consult you, rather than burning turns
  on the same broken approach.
- **Analysis plans** — for substantial analyses the model maintains
  a visible plan (pending / active / done per step) so you can see
  what it intends to do and where it is.
- **Replication packages** — one click produces a folder with the
  scripts Sift ran, the disclosure-controlled results, Markdown and
  LaTeX (`booktabs`) tables, a methods document with verification
  verdicts, software versions, and the disclosure record. Contains no
  raw data. Intended to be attached to a paper.
- **AI-use disclosure statement** — a paste-ready paragraph for
  journal disclosure forms and methods sections, generated from the
  session's own records (models used, interaction count, what
  categorically did and did not cross to the provider). Included in
  the analysis report and the replication package's METHODS.md.

- **Analysis reports** — one click renders the session as a
  shareable document: findings with their tables, verification
  verdicts (warnings first), helper-produced figures, and the
  disclosure record, as a single self-contained HTML file plus
  Markdown. Assembled from stored results by code — no model-written
  prose, and the only figures embedded are ones already cleared for
  crossing to the model.

- **Disclosure reports** — a reviewable record of everything released
  to the model in a session, written for an IRB or data-governance
  office, including an explicit statement of what it does not
  establish.
- **Codebook export** — a data dictionary for every dataset in the
  session (variables, labels, value labels, types, missingness,
  ranges) as Markdown + CSV, built locally from the files' own
  metadata. IRB submissions and data archives ask for exactly this.

- **Dataset profile** — row and variable counts, missingness,
  distinct counts, ranges, and likely identifier / constant /
  all-missing columns. Computed and displayed locally; never sent to
  the model, which remains governed by the per-dataset Permission
  tier.

## Trying it without your own data

Sift ships three synthetic datasets. On the landing screen, choose
**Try Sift with sample data** — everything is generated on your
machine and contains no real records. `sample_customers.csv` carries a
recoverable churn effect, informative missingness, an identifier, a
constant column, duplicates, and a category rare enough that the
disclosure-control layer visibly suppresses it (plus a second cell,
because one hidden cell is recoverable from the total).
`sample_trial.csv` is a two-arm survival study with a real treatment
effect and heavy censoring, so the survival verification checks have
something true to say. `sample_panel.csv` is a staggered-adoption
firm panel whose treatment effect is +2.0 by construction, so an
event-study estimate can be judged against known truth.

## Cost accounting

Each session records its token usage. Token counts are exact and come
from the provider. Spend is shown as the provider's own figure where
one is supplied, and otherwise as an estimate against a dated
rate table — labelled as an estimate, with the date. When no rate is
known for a model, no cost is shown at all rather than a misleading
zero. Open the **Ledger** panel to see both.

## Databases and warehouses

Most institutional data is not a file. Sift connects to SQLite,
DuckDB, and anything with a SQLAlchemy URI — PostgreSQL, MySQL/
MariaDB, SQL Server, Oracle, Snowflake, BigQuery, Redshift — provided
your environment has that backend's driver.

The connector is built so it does **not** weaken the sandbox:

- The query runs **on the host**, never inside the sandbox, so the
  sandbox keeps its total network denial. Generated code still cannot
  reach a database, or anything else.
- **Credentials never enter the sandbox** and are never shown to the
  model; connection strings are redacted wherever they surface.
- **The model cannot issue a query.** It can propose SQL for you to
  run; executing one is a researcher action in the interface. A
  connector the model could drive would be an exfiltration primitive.
- **Read-only, single statement.** `SELECT` / `WITH` / `SHOW` /
  `DESCRIBE` / `EXPLAIN` only. Stacked statements, comment-hidden
  writes, and anything that can modify a database are refused.
- Results are materialized to Parquet inside the session, where the
  schema-depth policy, disclosure control, profiling and size guards
  all apply unchanged. Each materialization is recorded in the ledger
  as a local ingestion, so an extract's provenance is auditable.

## Resource limits

Two guards keep a large file or a runaway script from taking down the
session. Both are fail-safe: an unparseable override falls back to the
default rather than switching the guard off.

| Setting | Default | Effect |
|---|---|---|
| `SIFT_MAX_LOAD_BYTES` | 512 MB | Ceiling on loading a dataset fully into memory. Above it, bounded fact requests are refused with a pointer to running a script instead, and the profile panel falls back to a labelled sample. |
| `SIFT_SCRIPT_MAX_MEMORY_BYTES` | 8 GiB | Address-space ceiling per script. Exceeding it surfaces inside the script as a normal memory error the model can repair, instead of the OS killing the app. `0` disables. |
| `SIFT_SCRIPT_TIMEOUT_SECONDS` | 300 | Per-script wall-clock timeout. |

## Platforms and installation

The native release targets are Apple-silicon macOS 11 or later, 64-bit
Windows 11, and 64-bit glibc-based Linux. Ubuntu 22.04 is the x86_64
qualification baseline and Ubuntu 24.04 is the ARM64 baseline. Use
`Sift.dmg`, `Sift-Windows-x64-Setup.exe`,
`Sift-Linux-x86_64.tar.gz`, or `Sift-Linux-aarch64.tar.gz` for the
matching system. See [the complete installation guide](docs/install.md)
for prerequisites, upgrades, uninstall behavior, local state paths,
institutional settings, and platform checks.

Release artifacts include the maintained Python analysis runtime. R and
licensed Stata are optional and are used only when a researcher explicitly
chooses to execute code in those languages. Reading R/Stata data files and
using Sift's standard bundled analyses do not require either product.

Sift includes no model account. Researchers supply an Anthropic, OpenAI,
Gemini, approved enterprise-cloud, or OpenAI-compatible endpoint credential.
Secrets entered in Sift are stored in macOS Keychain, Windows Credential
Manager, or a Freedesktop Secret Service-compatible vault on Linux—never in a
Sift plaintext settings file. A ChatGPT or Claude consumer subscription is not
an API credential and does not fund Sift usage.

Analysis-side packages (installable from chat via the
`install_packages` tool, behind an Approve/Deny modal):

```bash
python3 -m pip install pandas numpy statsmodels scipy matplotlib \
                       scikit-learn rdrobust differences
```

```r
install.packages(c("haven", "ggplot2", "lme4", "fixest", "survival",
                   "did", "rdrobust"))
```

Stata helpers ship with Sift; no SSC installs required.

## Run from source

```bash
uv sync --locked --all-extras --group dev
uv run pytest            # full suite

uv run sift              # landing screen: drop files or pick folder
uv run sift /path/to/data
```

Drop `.csv`, `.tsv`, `.xlsx` (first worksheet), `.dta` (Stata),
`.sav`/`.zsav` (SPSS), `.sas7bdat`/`.xpt` (SAS), `.rds` (R),
`.parquet`, `.jsonl`, or `.ndjson` files onto the landing screen.
SPSS and SAS variable labels and value labels flow into the schema,
the dataset profile, and the codebook export. CSV dialect handling is
automatic: Latin-1/Windows and UTF-16 encodings, BOMs,
semicolon-separated files, and decimal commas (the European Excel
default) are detected from the file rather than assumed, and the
profile flags conventional coded-missing values such as `-999` when
they sit at a column's extreme. Files are staged into
`~/.sift-sessions/<timestamp>_<id>/`, which becomes the sandbox root
for that session. (A pre-existing `~/.sift-sessions/` tree from Sift
keeps working.)

## Build native applications

Build each artifact on its target operating system; a build from one system
does not certify another.

macOS:

```bash
bash packaging/build_app.sh      # → dist/Sift.app
bash packaging/build_dmg.sh      # → dist/Sift.dmg
open dist/Sift.app
```

A bare local build is unsigned (fine for the same machine). For a
signed + notarized release set `SIFT_SIGN_IDENTITY` before
`build_app.sh` and `SIFT_NOTARIZE_PROFILE` before `build_dmg.sh`.

Windows PowerShell:

```powershell
.\packaging\build_windows.ps1 -SkipSign
```

Linux:

```bash
bash packaging/build_linux.sh
```

Development signing skips are never accepted by the production release mode.
The native pipelines verify branding, runtime completeness, confinement,
installation, in-place upgrade, and uninstall behavior before producing their
final artifacts.

## Feedback

Sift ships with the in-app feedback channel **disabled** — no
third-party endpoint receives anything unless a build is explicitly
configured with `SIFT_FEEDBACK_ACCESS_KEY`. Otherwise, email
jeevan@sapieninstitute.org.

## Repository layout

Start with the [contributor guide](CONTRIBUTING.md), the
[documentation index](docs/README.md), and the
[architecture reference](docs/architecture.md).

- `src/sift/` — application code (tools, executor, sanitizer, SDC,
  release ledger, verification, providers, web UI)
- `src/sift/runtime/` — R / Python / Stata result and plot helpers
- `tests/` — unit, integration, security-boundary, scientific-method, and
  packaging qualification coverage
- `packaging/` — shared PyInstaller spec and native macOS, Windows, and Linux
  build/install qualification scripts
- `scripts/` — explicit qualification and differential-testing entry points
- `siftbench/` — deterministic benchmark cases with known synthetic truth

## Security

Report vulnerabilities per [`SECURITY.md`](SECURITY.md). Known,
documented residual risks: cumulative inference across many queries
is *recorded* (release ledger) but not yet *bounded*; a
small-bandwidth covert channel exists in script-authored regression
metadata; `install_packages` reaches the network by design, behind
explicit user approval. Claims beyond this are not made.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).
