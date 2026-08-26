# Sift overview

Plain-language description of what Sift is and how it works.
Current version: **0.1.0** (August 2026).

## What Sift is

Sift is a macOS, Windows, and Linux app that sits on a researcher's own computer and
lets them use an AI assistant to analyze sensitive data without
sending that data to any third party. Medical records, HR data,
survey responses, IRB-restricted research. The data stays on the
researcher's machine.

The problem it solves. AI assistants are useful for data analysis,
but using one normally means uploading the data or describing it
in detail. Both expose the data. Researchers with confidential
datasets often can't do that, legally or ethically. Sift is a
thin local layer that lets the assistant help with the analysis
while the actual values stay on disk.

## How it works

The researcher drags data files into Sift's window, points it at a directory,
or imports a reviewed result from a database, cloud store, or research service.
The in-app source catalog shows the supported file formats and connectors.

A chat starts inside Sift. The researcher can use Anthropic, OpenAI, Gemini,
an approved enterprise-cloud model, or an OpenAI-compatible endpoint using
their own account. The model has no filesystem access,
no shell, no network. It reaches the data only through a narrow
set of operations: ask about the schema, submit an R / Stata /
Python script, read sanitized results, compose multiple results
into a comparison table.

When the model submits a script, Sift runs it locally in a
sandbox. The sandbox blocks network access and restricts which
files the script can read (the researcher's data directory plus
the paths R, Stata, or Python need to start up). The script's
output then passes through a sanitizer that applies statistical
disclosure control rules. It rounds coefficients based on sample
size, suppresses cells with fewer than 10 observations, never
reveals individual observations like min or max, and never
forwards raw text values from the data.

## What you see vs what the model sees

The researcher sees the full raw script output in the chat
window, in a result panel under each script run. The model sees
only the sanitized version.

Conversations persist across restarts. Every turn is saved to a
per-session log on disk. When the researcher reopens, the recent
turns plus a list of stored analytical results are loaded back
so the conversation picks up where it left off. Older turns that
have fallen out of that window can be retrieved on demand.

## The privacy guarantee

The model never directly touches the data. It writes questions
about the data (as code) and gets back privacy-filtered answers.
The boundary is enforced by three independent layers:

1. **The tool interface.** No filesystem, no shell, no network.
2. **The sandbox.** Native fail-closed confinement on macOS, Windows, and
   Linux; network denied and file reads restricted to the researcher's data
   directory and the runtime's startup paths.
3. **The sanitizer.** Statistical disclosure rules applied to
   every output before the model sees it.

## Install and run

Choose the native production artifact for the computer: `Sift.dmg` on Apple
silicon macOS, `Sift-Windows-x64-Setup.exe` on 64-bit Windows 11,
`Sift-Linux-x86_64.tar.gz` on x86_64 Linux, or
`Sift-Linux-aarch64.tar.gz` on ARM64 Linux. Follow
[`docs/install.md`](install.md); it includes prerequisites, platform checks,
upgrades, and uninstall behavior.

First launch asks for the researcher's own model credential or local endpoint.
Secrets are stored in macOS Keychain, Windows Credential Manager, or a
Freedesktop Secret Service-compatible vault. The maintained Python analysis
runtime is bundled. R and licensed Stata are optional and are needed only when
the researcher explicitly chooses to run code in those languages; importing
their data formats does not require either product.

## Where to read more

For implementation details, the supported analysis shapes, and
the contributor-facing architecture, see
[`architecture.md`](architecture.md) and
[`extending_analysis_shapes.md`](extending_analysis_shapes.md). For the full change history,
see [`CHANGELOG.md`](../CHANGELOG.md).
