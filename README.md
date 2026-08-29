# Sift

**Local-first AI data analysis for sensitive research.**

The datasets that matter most are often the datasets an AI system should not
receive. Sift is a desktop research assistant built for that constraint. It
lets a model plan an analysis, write Python, R, or Stata, and interpret the
results while the source data and computation remain on your computer.

You choose the model and provide the account. Sift works with Anthropic,
OpenAI, Gemini, selected enterprise deployments, and OpenAI-compatible local
or remote endpoints. It does not bundle a model subscription or send data to a
Sift-operated service.

Available integrations include:

- **Models:** Anthropic, OpenAI, Google Gemini, Azure OpenAI, Gemini and Claude
  on Google Vertex AI, Claude on Amazon Bedrock, and OpenAI-compatible local or
  remote endpoints such as Ollama, vLLM, LM Studio, and OpenRouter.
- **Databases and warehouses:** SQLite, DuckDB, PostgreSQL, MySQL, MariaDB,
  Microsoft SQL Server, Oracle, Snowflake, Google BigQuery, Amazon Redshift,
  and Databricks SQL.
- **Storage:** Amazon S3, Google Cloud Storage, Azure Blob Storage, signed HTTPS
  downloads, and SFTP.
- **Research and collaboration services:** Zotero, OSF, Dataverse, Zenodo,
  Figshare, Dryad, Google Drive, OneDrive, SharePoint, Box, Dropbox, REDCap,
  Qualtrics, KoboToolbox, and OpenClinica.

The project is open source under Apache 2.0. For the longer account of the
problem and design, read
[What If an AI Analyst Never Saw Your Raw Data?](https://sapieninstitute.org/projects/sift).

## Download

Sift is in public beta. Installers and checksums are published on the
[GitHub Releases page](https://github.com/JeevanSandhu13/sift-github-review/releases).

| Operating system | Supported computer | Download | Release status |
| --- | --- | --- | --- |
| macOS | Apple silicon, macOS 11 or later | `Sift.dmg` | Developer ID signed and Apple notarized |
| Windows | x64, Windows 11 | `Sift-Windows-x64-Setup.exe` | Unsigned beta |
| Linux | x86_64, glibc 2.35 or later | `Sift-Linux-x86_64.tar.gz` | Beta |
| Linux | ARM64, glibc 2.39 or later | `Sift-Linux-aarch64.tar.gz` | Beta |

The Windows beta has not yet received a CA-backed Authenticode signature.
SmartScreen may warn that the publisher is unknown, and Smart App Control or
an institution's device policy may block it. Do not turn off a security control
to install Sift. Wait for the signed Windows release if your computer does not
offer an approved way to continue.

### Install on macOS

1. Download `Sift.dmg` from GitHub Releases.
2. Open the disk image and drag Sift into **Applications**.
3. Eject the disk image.
4. Open Sift from **Applications**.

The macOS beta is signed, notarized, and stapled. It should open through the
normal Gatekeeper flow without a command-line workaround.

### Install on Windows

1. Download `Sift-Windows-x64-Setup.exe` and its `.sha256` file.
2. Verify the checksum if you are able to do so.
3. Run the installer. It installs for the current user and does not require
   administrator access.
4. If SmartScreen appears, read the warning and choose **More info > Run
   anyway** only if that option is available and the checksum matches.
5. Open **Start > Sift > Sift**.

Windows 11 normally includes the Microsoft Edge WebView2 Runtime that Sift
uses for its interface. If it is missing or too old, the installer stops before
making changes and identifies the official Microsoft component you need.

The portable archive, `Sift-Windows-x64.zip`, is available for testers who
cannot use the installer. Extract the complete archive and run `Sift.exe`
without moving it away from the adjacent `_internal` directory. The portable
build is also unsigned and remains subject to Windows security policy.

### Install on Linux

1. Download the archive that matches your processor.
2. Extract the complete archive to a local folder.
3. On Ubuntu 24.04, run `sudo ./prepare_ubuntu_host.sh`. The helper first
   checks the current host and changes nothing when the required policy is
   already present.
4. Run `./install.sh`.
5. Open Sift from the applications menu.

Sift requires a desktop X11 or Wayland session, Bubblewrap, and a running
Freedesktop Secret Service-compatible credential store. The installer checks
these requirements and reports what is missing. It never disables the Qt
sandbox or the operating system's namespace protections.

See the [installation guide](docs/install.md) for upgrades, uninstalling,
state locations, institutional policy, and platform checks.

## Your first session

1. Open **Manage providers** and add an API key or local endpoint. Credentials
   are saved in the operating system's protected credential store.
2. Choose **Try Sift with sample data** if you want to explore the application
   before opening your own files.
3. Add a file, folder, database extract, or supported cloud object.
4. Choose how much schema and summary information the model may receive.
5. Ask a research question in ordinary language.
6. Review the analysis plan, generated code, local output, verification
   results, and disclosure record as the work proceeds.

A ChatGPT, Claude, or Gemini consumer subscription does not include API usage.
Use an API credential from the provider, an approved enterprise deployment, or
an OpenAI-compatible model endpoint you control.

## How Sift works

A Sift session begins with data selected by the researcher. Profiling happens
locally: Sift identifies the structure of the dataset, variable types,
missingness, ranges, and likely identifiers. The active permission level
determines which parts of that profile can be shown to the model.

The model works through a small, fixed set of Sift tools. It can inspect the
permitted schema, maintain an analysis plan, submit a script, and read an
approved result. It does not receive a general shell, filesystem browser,
database connection, or web-search tool.

Generated scripts run beside the data under the native isolation mechanism for
the operating system:

- macOS uses a deny-by-default sandbox.
- Windows uses AppContainer.
- Linux uses Bubblewrap.

Network access is denied, readable and writable locations are allowlisted, and
credentials remain outside the script process. Sift refuses to run generated
code when the required confinement check does not pass.

Before a result returns to the model, it must match a registered statistical
shape and pass disclosure controls. These controls include minimum sample
requirements, small-cell and complementary suppression, bounded precision,
dominance checks, output-size limits, and hardening of text that originated in
the data.

The researcher and the model therefore see different views:

| The researcher can inspect | The model may receive |
| --- | --- |
| Raw files and locally materialized extracts | Researcher messages |
| Complete local script output | The permitted portion of the schema |
| Generated Python, R, or Stata | Sanitized statistical results |
| Local plots and dataset profiles | Approved aggregate figures |
| Verification and disclosure records | Redacted execution errors |

Raw rows, credentials, file contents, and unrestricted standard output are not
part of the model context when generated code follows Sift's runtime contract.

### An important security limit

Sift protects against accidental disclosure, ordinary model mistakes, and
data-origin prompt injection within its documented boundary. It does not claim
to protect a dataset from a deliberately malicious model provider.

Generated code and Sift's analysis helpers currently share an interpreter. The
runtime can reject missing or stale framing and many direct bypasses, but it
cannot cryptographically prove that an aggregate-shaped result was calculated
honestly. If your threat model includes a provider intentionally writing code
to encode raw values into a permitted result, do not enable generated-code
execution with that provider. The full boundary is documented in
[SECURITY.md](SECURITY.md) and [the architecture reference](docs/architecture.md).

## Data and analysis

Sift reads the formats researchers already use, including CSV and TSV, Excel,
Stata, SPSS, SAS, R, Parquet, Arrow, ORC, and JSON. Optional format packs add
scientific, geospatial, clinical, and genomic formats such as HDF5, NetCDF,
MATLAB, FITS, GeoPackage, Shapefile, GeoTIFF, VCF, NIfTI, DICOM, and FHIR.

Database connectors cover SQLite, DuckDB, PostgreSQL, MySQL, MariaDB, SQL
Server, Oracle, Snowflake, BigQuery, Redshift, and Databricks SQL. The
researcher selects and runs a read-only query; the model cannot browse the
database or execute SQL itself. The selected result is materialized locally
before it enters the normal analysis and disclosure pipeline.

Selection-based connectors are also available for common object stores,
research repositories, and collaboration services. The user chooses a
specific object. Sift does not give the model permission to browse the
underlying account.

The maintained method library covers descriptive and inferential statistics,
regression, longitudinal and mixed models, survival analysis, survey
estimation, missing data, time series, prediction, measurement models,
Bayesian workflows, study design, and a broad set of causal methods. Method
selection and result checks are deterministic parts of the application rather
than instructions left solely to the model.

## Records and exports

Sift keeps a local record of the analytical process: plans, scripts, sanitized
results, verification verdicts, source provenance, model usage, and each
disclosure made to the model. From that record it can produce:

- HTML, Markdown, PDF, and PowerPoint reports
- codebooks and data dictionaries
- disclosure reports for governance review
- AI-use statements for papers and submission forms
- replication packages containing code, results, methods, and software
  versions without the raw data

The disclosure ledger is an audit record, not a differential-privacy budget.
It makes cumulative access reviewable; it does not prove that repeated queries
cannot reveal additional information.

## Troubleshooting

### The application opens without styling

Do not continue with an unstyled page. Quit Sift and install a current release
from GitHub. A missing interface bundle is a packaging failure, not a browser
setting.

### A provider key is rejected

Confirm that it is an API credential for the provider and not a consumer-chat
subscription login. Remove the saved entry in **Manage providers**, add it
again, and run the connection check. Sift stores credentials in macOS
Keychain, Windows Credential Manager, or a Freedesktop Secret Service vault.

### R or Stata is not found

R and licensed Stata are optional. Sift can read R and Stata data files and run
its standard bundled analyses without either application. Install the external
runtime only when you explicitly want Sift to execute code in that language.

### Linux reports that confinement is unavailable

Install Bubblewrap and confirm that the distribution permits its supported
user-namespace policy. On Ubuntu 24.04, use the included
`prepare_ubuntu_host.sh` helper. Do not disable the Qt sandbox or system-wide
security policy.

### Windows blocks the beta

Verify the published checksum. If Windows offers **More info > Run anyway**,
you may use that path for the unsigned beta. If Smart App Control or an
institutional policy blocks the file without an override, wait for the signed
Windows release.

More detailed checks are in [Installing Sift](docs/install.md). Bugs and
ordinary support questions belong in
[GitHub Issues](https://github.com/JeevanSandhu13/sift-github-review/issues).
Send security reports privately as described in [SECURITY.md](SECURITY.md).

## Run from source

Sift uses Python 3.10 or later and [uv](https://docs.astral.sh/uv/) for the
development environment:

```bash
uv sync --locked --all-extras --group dev
uv run pytest -q
uv run sift
```

Native release artifacts must be built and qualified on their target operating
system. See [CONTRIBUTING.md](CONTRIBUTING.md) for the repository map and
development workflow, and [docs/verification.md](docs/verification.md) for the
qualification procedures.

## Documentation

- [Overview](docs/overview.md)
- [Installation and troubleshooting](docs/install.md)
- [Architecture and privacy boundary](docs/architecture.md)
- [Who Sift is for](docs/who_uses_sift.md)
- [Verification](docs/verification.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Contact

Questions and feedback: [js.sandhu@mail.utoronto.ca](mailto:js.sandhu@mail.utoronto.ca)

Sift is a project of the [Sapien Institute](https://sapieninstitute.org).

## License

Copyright 2026 Jeevan Sandhu.

Licensed under the [Apache License 2.0](LICENSE).
