# Installing Sift

Sift is a local desktop research assistant for macOS, Windows, and Linux. The
application includes its maintained Python analysis runtime. Researchers add
their own model API key, cloud identity, local model endpoint, or database
connection; Sift does not include a paid model account and does not require a
Stata licence for ordinary data import or bundled analysis.

## Choose the correct download

| System | Supported release target | Recommended artifact |
| --- | --- | --- |
| macOS | Apple silicon (M1 or later), macOS 11 or later | `Sift.dmg` |
| Windows | 64-bit Windows 11 | `Sift-Windows-x64-Setup.exe` |
| Linux x86_64 | 64-bit glibc-based desktop; Ubuntu 22.04 is the qualification baseline | `Sift-Linux-x86_64.tar.gz` |
| Linux ARM64 | 64-bit glibc-based desktop; Ubuntu 24.04 is the qualification baseline | `Sift-Linux-aarch64.tar.gz` |

Use only an artifact whose checksum and release signature match the published
release manifest. Production macOS and Windows downloads must also carry their
platform-native publisher signature. Signing and Apple notarization happen in
the final release pipeline; an unsigned local development build is not a
public release.

## macOS

1. Open `Sift.dmg`.
2. Drag Sift into Applications.
3. Eject the Sift disk image, then open Sift from Applications.

The public release is expected to be Developer ID-signed, notarized, and
stapled so Gatekeeper can verify it without a network connection. Do not tell
users to bypass Gatekeeper for a public release. An unsigned build may be used
only for local development on a machine that created or explicitly trusts it.

To upgrade, quit Sift and replace the existing application in Applications.
To uninstall, quit Sift and move `/Applications/Sift.app` to the Trash.

## Windows

1. Run `Sift-Windows-x64-Setup.exe`.
2. Complete the per-user installer; administrator access is not required.
3. Open Start, choose Sift, then choose Sift.

Sift requires the Microsoft Edge WebView2 Evergreen Runtime. Windows 11
normally includes it. If it is absent, the installer stops before changing the
computer and identifies the missing official Microsoft component. The setup
program supports in-place upgrades and registers Sift in Windows Settings.

To uninstall, open **Settings > Apps > Installed apps**, find Sift, and choose
**Uninstall**. The portable ZIP is also supported: extract the entire archive
to a writable local folder and run `Sift.exe` without separating it from its
adjacent `_internal` directory.

## Linux

Sift requires a desktop X11 or Wayland session, glibc 2.35 or newer on x86_64
or glibc 2.39 or newer on ARM64,
`bubblewrap` with working unprivileged namespace confinement, and a running
Freedesktop Secret Service-compatible credential vault. Minimal Ubuntu desktop
images must also install Qt's runtime libraries, including
`libwayland-server0`; the app fails closed with a precise platform check when a
native renderer dependency is absent.

1. Extract the matching `Sift-Linux-x86_64.tar.gz` or
   `Sift-Linux-aarch64.tar.gz` without moving individual files out of
   the extracted Sift directory.
2. Open a terminal in that directory.
3. On Ubuntu 24.04, run `sudo ./prepare_ubuntu_host.sh`. It first tests the
   current policy and does nothing when confinement already works. If needed,
   it installs Ubuntu's official bubblewrap AppArmor profile; it never disables
   Ubuntu's system-wide user-namespace protection.
4. Run `./install.sh`.
5. Open Sift from the applications menu, or run `~/.local/bin/sift`.

The installer is per-user and honours `XDG_DATA_HOME` and `XDG_BIN_HOME`. It
performs an atomic in-place upgrade and does not require administrator access.
The separate Ubuntu host-policy helper requires administrator access only when
the host needs the official AppArmor policy installed.
To uninstall, run:

```sh
"${XDG_DATA_HOME:-$HOME/.local/share}/sift/uninstall.sh"
```

## First launch and settings

On first launch, choose **Manage providers** and configure at least one model:
Anthropic, OpenAI, Gemini, an approved enterprise-cloud model, or an
OpenAI-compatible local endpoint. Provider usage and billing belong to the
researcher's account. Keys entered in Sift are stored only in the operating
system's protected credential vault:

- macOS Keychain
- Windows Credential Manager
- a Freedesktop Secret Service-compatible vault on Linux

Use **Add data source** to choose local files or a folder, a supported database,
cloud storage, or a research service. A connection is made by Sift on the local
machine. The selected result is copied into the private local workspace; the
connected account is not exposed to the model as a tool. Each connector shows
the exact credentials and permissions it needs and supports a connection test
before import.

Sift supports CSV/TSV, Excel, Stata, SPSS, SAS, R, Parquet, Arrow/Feather,
JSON/JSONL, SQLite, DuckDB, and the connectors presented in the data-source
catalog. Reading a `.dta` file does not require Stata. If a researcher
explicitly asks Sift to execute their own R or Stata code, a compatible local R
or licensed Stata installation is required for that optional execution path.

The **How Sift works** walkthrough remains available from the application. It
explains local workspaces, permission tiers, model disclosure, execution,
verification, exports, and how to inspect evidence.

## Where local state is kept

Imported sessions are stored in `~/.sift-sessions`. A project folder opened in
place keeps its private state in that folder's `.sift` directory. Removing the
application deliberately retains these research directories and stored
credentials so an uninstall cannot silently destroy research work.

Diagnostics are local, bounded, and redacted. Their default locations are:

- macOS: `~/Library/Logs/Sift`
- Windows: `%LOCALAPPDATA%\Sift\Logs`
- Linux: `$XDG_STATE_HOME/sift/log`, or `~/.local/state/sift/log`

Sift does not check for updates at startup. Network access to the configured
update channel occurs only after the researcher chooses **Check for updates**.

Institutional administrators may install a protected policy at:

- macOS: `/Library/Application Support/Sift/enterprise_policy.yaml`
- Windows: `%ProgramData%\Sift\enterprise_policy.yaml`
- Linux: `/etc/sift/enterprise_policy.yaml`

That optional policy can restrict providers, connectors, destinations,
disclosure, diagnostics, exports, and resource limits. An invalid configured
policy fails closed instead of silently falling back to unrestricted behavior.

## Data disclosure model

Raw datasets remain in the selected local workspace. The model initially sees
only metadata allowed by the active permission tier. The default tier permits
names, coarse types, labels, and bounded completeness/distinct-count summaries;
it does not expose raw values, row-level records, minima, maxima, or medians.
More detailed data requires an explicit bounded request and passes through
Sift's disclosure controls. Analysis outputs are sanitized before they can be
returned to the model, while the researcher can inspect the local evidence and
execution record.

## Platform checks

Before relying on a new installation, run the packaged check for that system:

```sh
# macOS
/Applications/Sift.app/Contents/Resources/sift/sift --platform-check

# Linux
~/.local/bin/sift --platform-check
```

On Windows PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\Sift\Sift.exe" --platform-check
```

The check verifies the supported architecture, native renderer, protected
credential store, bundled assets, analysis runtime, and required execution
confinement. A failure is a compatibility or security failure to resolve, not
a warning to bypass.

## Running from this source tree

Maintainers can run the same application without installing a release artifact:

```sh
uv sync --locked --all-extras --group dev
uv run pytest -q
uv run sift
```

macOS, Windows, and Linux artifacts must each be built and exercised on their
native qualification host. A successful source-tree test on one operating
system does not certify an installer for another.
