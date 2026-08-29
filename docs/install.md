# Installing Sift

This guide covers the packaged desktop application. If you want to change the
code or build Sift yourself, skip to [Run from source](#run-from-source).

Sift includes its maintained Python analysis environment. You do not need to
install Python to use the desktop app. R and licensed Stata are optional and
are needed only when you explicitly ask Sift to execute code in those
languages.

## 1. Choose the right download

Download Sift from the repository's
[Releases page](https://github.com/JeevanSandhu13/sift-github-review/releases).

| Computer | Minimum supported system | File |
| --- | --- | --- |
| Apple silicon Mac | macOS 11 | `Sift.dmg` |
| Windows x64 PC | Windows 11 | `Sift-Windows-x64-Setup.exe` |
| Linux x86_64 | glibc 2.35; Ubuntu 22.04 baseline | `Sift-Linux-x86_64.tar.gz` |
| Linux ARM64 | glibc 2.39; Ubuntu 24.04 baseline | `Sift-Linux-aarch64.tar.gz` |

The macOS release is Developer ID signed, Apple notarized, and stapled. The
current Windows build is an unsigned beta. Linux releases include x86_64 and
ARM64 archives.

## 2. Verify the download

Each release asset has a matching `.sha256` file. Compare the published value
with the file you downloaded before installing.

On macOS:

```sh
shasum -a 256 ~/Downloads/Sift.dmg
```

On Windows PowerShell:

```powershell
Get-FileHash "$HOME\Downloads\Sift-Windows-x64-Setup.exe" -Algorithm SHA256
```

On Linux:

```sh
sha256sum ~/Downloads/Sift-Linux-x86_64.tar.gz
```

The value must match the corresponding checksum shown on GitHub. A mismatch
usually means the file is incomplete or has changed. Delete it and download it
again; do not run it.

Sift also publishes a CycloneDX software bill of materials and an Ed25519
release statement for each artifact. These are primarily for institutional
review and automated release verification.

## 3. Install the application

### macOS

1. Open `Sift.dmg`.
2. Drag **Sift** into **Applications**.
3. Eject the Sift disk image.
4. Open Sift from **Applications**.

Gatekeeper should identify the application as software from Jeevan Sandhu and
open it through the normal macOS flow. The release does not require a
command-line exception.

To upgrade, quit Sift, open the newer disk image, and replace the existing copy
in **Applications**.

To uninstall, quit Sift and move `/Applications/Sift.app` to the Trash.
Sessions and credentials are retained so removing the app cannot silently
delete research work. See [Remove retained data](#remove-retained-data) if you
want to delete those separately.

### Windows

1. Run `Sift-Windows-x64-Setup.exe`.
2. Complete the per-user installer. Administrator access is not required.
3. Open **Start > Sift > Sift**.

The beta does not yet have a CA-backed Authenticode publisher signature.
Windows Defender SmartScreen may display **Windows protected your PC**. After
verifying the checksum, choose **More info > Run anyway** only when Windows
offers that option. Smart App Control and organization-managed devices may
block unsigned applications without an override. Do not disable those
protections; wait for the signed Windows release.

Sift uses the Microsoft Edge WebView2 Evergreen Runtime for its interface.
Windows 11 normally includes it. If the runtime is missing or too old, the
installer stops before changing the computer and identifies the official
Microsoft component to install.

The setup program supports in-place upgrades. To uninstall, open **Settings >
Apps > Installed apps**, find Sift, and choose **Uninstall**.

For portable testing, extract the complete `Sift-Windows-x64.zip` archive to
a writable local folder and run `Sift.exe`. Keep `Sift.exe` beside its
`_internal` directory. The portable build is also unsigned and is subject to
the same Windows security policy.

### Linux

Sift needs:

- an X11 or Wayland desktop session
- Bubblewrap with working unprivileged namespace confinement
- a running Freedesktop Secret Service-compatible credential store
- glibc 2.35 or later on x86_64, or glibc 2.39 or later on ARM64

To install:

1. Extract the archive that matches your processor.
2. Open a terminal in the extracted Sift directory.
3. On Ubuntu 24.04, run `sudo ./prepare_ubuntu_host.sh`. It checks the current
   policy first and changes nothing when the host is already compatible. When
   needed, it installs Ubuntu's Bubblewrap AppArmor profile; it does not turn
   off the system-wide user-namespace policy.
4. Run `./install.sh`.
5. Open Sift from the applications menu or run `~/.local/bin/sift`.

The installer is per-user, honours `XDG_DATA_HOME` and `XDG_BIN_HOME`, and
performs atomic in-place upgrades. The Ubuntu policy helper is the only step
that may require administrator access.

To uninstall:

```sh
"${XDG_DATA_HOME:-$HOME/.local/share}/sift/uninstall.sh"
```

## 4. Complete first-run setup

Open **Manage providers** and configure at least one model. Sift accepts:

- Anthropic, OpenAI, or Gemini API credentials
- approved enterprise model deployments
- OpenAI-compatible local or remote endpoints

Provider access and billing remain with your account. A consumer ChatGPT,
Claude, or Gemini subscription is not an API credential.

Sift stores credentials in the protected service provided by the operating
system:

- macOS Keychain
- Windows Credential Manager
- a Freedesktop Secret Service-compatible vault on Linux

The first request to save or read a key may trigger an operating-system
permission prompt. Approve it only when the prompt identifies the Sift
application you just installed.

Next, choose **Try Sift with sample data** or **Add data source**. Sample data
is generated locally and contains no real records. Data-source connections are
made by Sift on your computer. You select a file, object, or read-only query
result; the model cannot browse the connected account.

The in-app **How Sift works** walkthrough explains workspaces, permission
levels, model disclosure, script execution, verification, and exports.

## Troubleshooting

### The macOS app is reported as damaged or cannot be verified

Confirm that you downloaded the current `Sift.dmg` from GitHub Releases and
that its SHA-256 value matches. Delete any older or incomplete copy and
download it again. Do not use `xattr`, disable Gatekeeper, or create a local
signing exception for the public build. A current release that fails normal
Gatekeeper verification should be reported as a packaging problem.

### Windows says the publisher is unknown

That is expected for the unsigned Windows beta. Verify the SHA-256 checksum
before choosing **More info > Run anyway**. If no override is offered, the
computer's security policy is blocking unsigned software; wait for the signed
release.

### Windows reports that WebView2 is missing

Install the Microsoft Edge WebView2 Evergreen Runtime from Microsoft, then run
the Sift installer again. Do not download a WebView2 installer from a third
party.

### Linux opens no window

Run the [platform check](#platform-checks). Confirm that you are in a graphical
X11 or Wayland session and that the required Qt libraries are installed.
Minimal Ubuntu installations may be missing `libwayland-server0`.

### Linux reports that confinement is unavailable

Install Bubblewrap using the distribution's package manager. On Ubuntu 24.04,
run the included `prepare_ubuntu_host.sh` helper. Sift intentionally refuses
to run generated code without a working confinement backend.

### Sift cannot store an API key on Linux

Confirm that a Secret Service provider, such as the desktop environment's
keyring, is installed, unlocked, and available in the current session. Sift
does not fall back to a plaintext credential file.

### A provider connection fails

Check that you supplied an API key rather than a consumer-subscription login,
that the account has access to the selected model, and that provider billing is
active. Use **Manage providers** to forget the saved credential, add it again,
and run the connection check.

### R or Stata is unavailable

No action is required for ordinary Sift analysis or for reading `.rds` and
`.dta` files. Install R or licensed Stata only when you want Sift to execute
code in that language. The external runtime must be available on the normal
application path.

### The interface appears as an unstyled web page

Quit Sift and install a current packaged release. An unstyled page means the
desktop interface bundle is missing or unreadable; it is not an appearance
setting. Include the Sift version and operating system when reporting it.

## Platform checks

The packaged application includes a local diagnostic that checks the processor,
renderer, credential store, bundled assets, analysis runtime, and confinement
backend.

On macOS:

```sh
/Applications/Sift.app/Contents/Resources/sift/sift --platform-check
```

On Windows PowerShell:

```powershell
& "$env:LOCALAPPDATA\Programs\Sift\Sift.exe" --platform-check
```

On Linux:

```sh
~/.local/bin/sift --platform-check
```

A failed platform check identifies a compatibility or security requirement to
resolve. It is not a warning to bypass.

## Local state and institutional policy

Sift keeps imported sessions in `~/.sift-sessions`. A project folder opened
in place keeps its private state in that folder's `.sift` directory.

Redacted local diagnostics are stored in:

- macOS: `~/Library/Logs/Sift`
- Windows: `%LOCALAPPDATA%\Sift\Logs`
- Linux: `$XDG_STATE_HOME/sift/log`, or `~/.local/state/sift/log`

Sift does not check for updates at startup. It contacts the configured update
channel only after you choose **Check for updates**.

An institution may install a protected policy at:

- macOS: `/Library/Application Support/Sift/enterprise_policy.yaml`
- Windows: `%ProgramData%\Sift\enterprise_policy.yaml`
- Linux: `/etc/sift/enterprise_policy.yaml`

The policy can restrict providers, connectors, destinations, disclosure,
diagnostics, exports, and resource limits. If a configured policy cannot be
parsed or verified, Sift fails closed.

### Remove retained data

Uninstalling Sift retains sessions and credentials by design. To remove them:

1. Use **Manage providers > Forget** for saved model credentials.
2. Remove database or source credentials from Sift or the operating system's
   credential manager.
3. Archive or delete the researcher's `~/.sift-sessions` directory and any
   project `.sift` directories only after confirming they are no longer
   needed.
4. Remove the platform diagnostic directory if desired.

These steps delete research state and cannot be undone. Review the exact
folders before removing them.

## What is disclosed to the model

Raw datasets remain in the selected local workspace. The active permission
level determines which schema and summary fields the model may receive.
Detailed requests are bounded and pass through disclosure controls. Analysis
results are sanitized before entering model context, while the researcher can
inspect the full local evidence and execution record.

Read [Sift architecture](architecture.md) for the complete boundary and its
documented limitation, and [Security policy](../SECURITY.md) before using Sift
for a threat model that includes a deliberately malicious provider.

## Run from source

Install [uv](https://docs.astral.sh/uv/) and run:

```sh
uv sync --locked --all-extras --group dev
uv run pytest -q
uv run sift
```

Native artifacts must be built and exercised on their target operating system.
A successful source run on one platform does not qualify an installer for
another. Build and release procedures are documented in
[CONTRIBUTING.md](../CONTRIBUTING.md) and
[Verification](verification.md).
