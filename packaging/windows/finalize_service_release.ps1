param(
    [string]$RepoRoot = "C:\SiftBuild\src"
)

# Supplemental qualification for headless VM automation.  The canonical
# qualify_windows_install.ps1 remains the release gate for Start-menu,
# Desktop, and HKCU uninstall registration because Windows returns empty
# shell-folder paths in a service session.  This script covers the surfaces
# that can be proven headlessly: exact installer payload, clean install,
# replacement upgrade, every non-window frozen self-check, state retention,
# uninstall, portable archive construction, checksums, and SBOM binding.
$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

$Bundle = Join-Path $RepoRoot "dist\Sift"
$Installer = Join-Path $RepoRoot "dist\Sift-Windows-x64-Setup.exe"
$Archive = Join-Path $RepoRoot "dist\Sift-Windows-x64.zip"
if (-not (Test-Path (Join-Path $Bundle "Sift.exe") -PathType Leaf)) {
    throw "Qualified frozen bundle is missing."
}
if (-not (Test-Path $Installer -PathType Leaf)) {
    throw "Compiled installer is missing."
}

# C:\Windows\Temp inherits service-only ACLs that intentionally deny the
# low-privilege AppContainer used by Sift's native confinement probe. A real
# per-user install lives below LocalAppData and is AppContainer-readable.
# Public Documents supplies the same relevant read/traverse semantics while
# remaining an isolated, disposable location available without a login.
$TestRoot = Join-Path "C:\Users\Public\Documents" (
    "Sift service qualification " + [guid]::NewGuid()
)
$InstallRoot = Join-Path $TestRoot "Sift"
$StateRoot = Join-Path $TestRoot "researcher-state"
$Sentinel = Join-Path $StateRoot "session.sentinel"
$Uninstaller = Join-Path $InstallRoot "unins000.exe"
$SetupLog = Join-Path $TestRoot "setup.log"
$UninstallLog = Join-Path $TestRoot "uninstall.log"

function Invoke-Setup {
    $Process = Start-Process -FilePath $Installer -ArgumentList @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        "/MERGETASKS=!desktopicon", "/DIR=`"$InstallRoot`"",
        "/LOG=`"$SetupLog`""
    ) -Wait -PassThru -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        throw "Sift installer exited with code $($Process.ExitCode)."
    }
}

function Invoke-FrozenChecks {
    $Executable = Join-Path $InstallRoot "Sift.exe"
    # --renderer-check is intentionally excluded here. WebView2 refuses to
    # create a controller in Windows session 0 with CO_E_SERVER_EXEC_FAILURE;
    # that is an operating-system desktop boundary, not a renderer defect.
    # The canonical interactive qualifier still gates actual window creation,
    # while --platform-check below verifies WebView2, its loader assemblies,
    # renderer bindings, and the native AppContainer probe in this session.
    # --format-check is likewise per-user: its complete confined worker needs
    # an interactive user's AppContainer identity and returns
    # FormatSelectionError under the guest-agent service account. The
    # canonical interactive qualification remains the gate for that worker.
    foreach ($Check in @(
        "--platform-check", "--integration-check",
        "--analysis-check", "--credential-store-check",
        "--help"
    )) {
        # Sift is a GUI-subsystem PE. In a non-interactive service session,
        # PowerShell's call operator can return before a GUI process exits and
        # leave LASTEXITCODE unset/stale. Start-Process supplies the actual
        # process handle and exit code, making this headless harness reliable.
        $Process = Start-Process -FilePath $Executable -ArgumentList $Check `
            -PassThru -WindowStyle Hidden
        if (-not $Process.WaitForExit(180000)) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
            throw "Installed Sift timed out during $Check."
        }
        if ($Process.ExitCode -ne 0) {
            throw "Installed Sift failed $Check with exit code $($Process.ExitCode)."
        }
    }
}

try {
    New-Item -ItemType Directory -Path $StateRoot -Force | Out-Null
    Set-Content -LiteralPath $Sentinel -Value "retain" -NoNewline
    Invoke-Setup
    if (-not (Test-Path $Uninstaller -PathType Leaf)) {
        throw "Installed uninstaller is missing."
    }

    foreach ($Asset in @("app.js", "desktop-shell.css")) {
        $Source = Join-Path $RepoRoot "src\sift\web\$Asset"
        $Installed = Join-Path $InstallRoot "_internal\sift\web\$Asset"
        if (-not (Test-Path $Installed -PathType Leaf)) {
            throw "Installer payload is missing $Asset."
        }
        if ((Get-FileHash -Algorithm SHA256 $Source).Hash -ne
            (Get-FileHash -Algorithm SHA256 $Installed).Hash) {
            throw "Installed $Asset does not match the corrected source."
        }
    }
    Invoke-FrozenChecks

    # Replacement installation is the upgrade path. Researcher-owned state
    # deliberately lives outside the installer root and must remain intact.
    Invoke-Setup
    if ((Get-Content -LiteralPath $Sentinel -Raw) -ne "retain") {
        throw "Replacement upgrade changed researcher-owned state."
    }
    Invoke-FrozenChecks

    $Process = Start-Process -FilePath $Uninstaller -ArgumentList @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        "/LOG=`"$UninstallLog`""
    ) -Wait -PassThru -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        throw "Sift uninstaller exited with code $($Process.ExitCode)."
    }
    if (Test-Path (Join-Path $InstallRoot "Sift.exe")) {
        throw "Uninstall left Sift.exe behind."
    }
    if ((Get-Content -LiteralPath $Sentinel -Raw) -ne "retain") {
        throw "Uninstall removed researcher-owned state."
    }
}
finally {
    if (Test-Path $Uninstaller -PathType Leaf) {
        Start-Process -FilePath $Uninstaller -ArgumentList @(
            "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"
        ) -Wait -NoNewWindow -ErrorAction SilentlyContinue
    }
    if (Test-Path $TestRoot) {
        Remove-Item $TestRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Remove-Item $Archive, "$Archive.sha256", "$Archive.sbom.cdx.json" `
    -Force -ErrorAction SilentlyContinue
Compress-Archive -Path $Bundle -DestinationPath $Archive -CompressionLevel Optimal
# The canonical portable qualifier includes --renderer-check and therefore
# also requires an interactive desktop. Extract the archive and prove its
# exact payload here; the source bundle's renderer executable is unchanged
# and already covered by the canonical interactive Windows qualification.
$PortableRoot = Join-Path "C:\Users\Public\Documents" (
    "Sift portable qualification " + [guid]::NewGuid()
)
try {
    Expand-Archive -LiteralPath $Archive -DestinationPath $PortableRoot
    $PortableBundle = Join-Path $PortableRoot "Sift"
    foreach ($Required in @(
        "Sift.exe", "_internal", "INSTALL.txt", "LICENSE.txt",
        "release-metadata.json"
    )) {
        if (-not (Test-Path (Join-Path $PortableBundle $Required))) {
            throw "Portable archive is missing $Required."
        }
    }
    foreach ($Asset in @("app.js", "desktop-shell.css")) {
        $Source = Join-Path $RepoRoot "src\sift\web\$Asset"
        $Portable = Join-Path $PortableBundle "_internal\sift\web\$Asset"
        if ((Get-FileHash -Algorithm SHA256 $Source).Hash -ne
            (Get-FileHash -Algorithm SHA256 $Portable).Hash) {
            throw "Portable $Asset does not match the corrected source."
        }
    }
    $InstallRoot = $PortableBundle
    Invoke-FrozenChecks
}
finally {
    if (Test-Path $PortableRoot) {
        Remove-Item $PortableRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}

$Version = uv run --no-sync python packaging/windows_build_probe.py project-version
if ($LASTEXITCODE -ne 0 -or -not $Version) {
    throw "Could not read package version."
}
foreach ($Artifact in @($Installer, $Archive)) {
    $ArtifactHash = (Get-FileHash -Algorithm SHA256 $Artifact).Hash.ToLowerInvariant()
    $ArtifactName = Split-Path $Artifact -Leaf
    [IO.File]::WriteAllText(
        "$Artifact.sha256", "$ArtifactHash  $ArtifactName`n",
        [Text.Encoding]::ASCII
    )
    uv run python -m sift.release_manifest sbom `
        $Artifact "$Artifact.sbom.cdx.json" --version $Version
    if ($LASTEXITCODE -ne 0) {
        throw "SBOM generation failed for $ArtifactName."
    }
    uv run python -m sift.release_manifest verify-sbom `
        $Artifact "$Artifact.sbom.cdx.json"
    if ($LASTEXITCODE -ne 0) {
        throw "SBOM binding failed for $ArtifactName."
    }
}

Write-Host "Windows service-session install, upgrade, runtime, uninstall, portable, checksum, and SBOM qualification passed."
