param(
    [switch]$SkipVendor,
    [switch]$SkipSign,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "This release must be built and tested on Windows."
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is required."
}
$PythonTarget = if ($env:SIFT_WINDOWS_PYTHON_TARGET) {
    $env:SIFT_WINDOWS_PYTHON_TARGET
} else {
    "cpython-3.12.11-windows-x86_64-none"
}
if ($PythonTarget -ne "cpython-3.12.11-windows-x86_64-none") {
    throw "Sift's Windows release target must be cpython-3.12.11-windows-x86_64-none."
}
$env:UV_PYTHON = $PythonTarget
if (-not $env:UV_PROJECT_ENVIRONMENT) {
    $env:UV_PROJECT_ENVIRONMENT = Join-Path $RepoRoot ".venv-windows-x64"
}
uv python install $PythonTarget
if ($LASTEXITCODE -ne 0) { throw "The pinned Windows x64 Python could not be installed." }
uv sync --locked --all-extras
if ($LASTEXITCODE -ne 0) { throw "Database connector dependency sync failed." }
$PythonTargetJson = uv run --no-sync python packaging/windows_build_probe.py python-target
if ($LASTEXITCODE -ne 0 -or -not $PythonTargetJson) {
    throw "Could not inspect the selected Windows Python target."
}
$PythonTargetReport = $PythonTargetJson | ConvertFrom-Json
if ($PythonTargetReport.platform -ne "win-amd64" -or $PythonTargetReport.pointer_bits -ne 64) {
    throw "The selected Python is not the required Windows x64 target: $PythonTargetJson"
}

function Get-PortableExecutableMachine([string]$Path) {
    $Reader = $null
    $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try {
        $Reader = [IO.BinaryReader]::new($Stream)
        if ($Reader.ReadUInt16() -ne 0x5A4D) { throw "Missing DOS executable signature: $Path" }
        $Stream.Position = 0x3C
        $PeOffset = $Reader.ReadUInt32()
        if ($PeOffset -gt ($Stream.Length - 6)) { throw "Invalid PE header offset: $Path" }
        $Stream.Position = $PeOffset
        if ($Reader.ReadUInt32() -ne 0x00004550) { throw "Missing PE signature: $Path" }
        return $Reader.ReadUInt16()
    } finally {
        if ($Reader) { $Reader.Dispose() } else { $Stream.Dispose() }
    }
}
$InnoCompiler = $null
if ($env:SIFT_INNO_SETUP) {
    if (-not (Test-Path $env:SIFT_INNO_SETUP -PathType Leaf)) {
        throw "SIFT_INNO_SETUP does not point to ISCC.exe."
    }
    $InnoCompiler = (Resolve-Path $env:SIFT_INNO_SETUP).Path
} else {
    $InnoCommand = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($InnoCommand) {
        $InnoCompiler = $InnoCommand.Source
    } else {
        $ProgramFilesX86 = [Environment]::GetFolderPath("ProgramFilesX86")
        $ProgramFiles64 = [Environment]::GetFolderPath("ProgramFiles")
        $LocalPrograms = Join-Path $env:LOCALAPPDATA "Programs"
        $InnoCandidates = @(
            (Join-Path $ProgramFiles64 "Inno Setup 7\ISCC.exe"),
            (Join-Path $ProgramFilesX86 "Inno Setup 7\ISCC.exe"),
            (Join-Path $LocalPrograms "Inno Setup 7\ISCC.exe"),
            (Join-Path $LocalPrograms "InnoSetup7\ISCC.exe"),
            (Join-Path $ProgramFiles64 "Inno Setup 6\ISCC.exe"),
            (Join-Path $ProgramFilesX86 "Inno Setup 6\ISCC.exe"),
            (Join-Path $LocalPrograms "Inno Setup 6\ISCC.exe"),
            (Join-Path $LocalPrograms "InnoSetup6\ISCC.exe")
        )
        foreach ($Candidate in $InnoCandidates) {
            if (Test-Path $Candidate -PathType Leaf) {
                $InnoCompiler = $Candidate
                break
            }
        }
    }
}
if (-not $InnoCompiler) {
    throw "Inno Setup 6.3 or newer is required to build the branded x64 per-user installer. Set SIFT_INNO_SETUP to its ISCC.exe if it is installed in a custom location."
}
$ReleaseMode = if ($env:SIFT_RELEASE_MODE) { $env:SIFT_RELEASE_MODE } else { "development" }
$ReleaseChannel = if ($env:SIFT_RELEASE_CHANNEL) { $env:SIFT_RELEASE_CHANNEL } else { "stable" }
if ($ReleaseMode -notin "development", "production") {
    throw "SIFT_RELEASE_MODE must be development or production."
}
if ($ReleaseChannel -notin "stable", "beta") {
    throw "SIFT_RELEASE_CHANNEL must be stable or beta."
}
if ($ReleaseMode -eq "production") {
    if ($SkipVendor) { throw "Production mode cannot skip the bundled analysis runtime." }
    if ($SkipSign) { throw "Production mode cannot use -SkipSign." }
    if (-not $env:SIFT_WINDOWS_CERT_SHA1) {
        throw "Production mode requires SIFT_WINDOWS_CERT_SHA1."
    }
    if (-not $env:SIFT_RELEASE_PRIVATE_KEY_B64) {
        throw "Production mode requires SIFT_RELEASE_PRIVATE_KEY_B64."
    }
    if (-not $env:SIFT_RELEASE_KEY_ID) {
        throw "Production mode requires SIFT_RELEASE_KEY_ID."
    }
    $env:PYTHONPATH = "src"
    uv run --no-sync python packaging/windows_build_probe.py production-update-policy
    if ($LASTEXITCODE -ne 0) { throw "production update policy is not configured." }
}

if (-not $SkipVendor) {
    uv run python packaging/vendor_python.py
    if ($LASTEXITCODE -ne 0) { throw "Analysis runtime vendoring failed." }
}

uv run python packaging/generate_brand_assets.py --check
if ($LASTEXITCODE -ne 0) { throw "Native brand asset verification failed." }
uv run python packaging/verify_database_drivers.py
if ($LASTEXITCODE -ne 0) { throw "Database connector verification failed." }
uv run python -m sift --platform-check
if ($LASTEXITCODE -ne 0) { throw "Windows desktop runtime qualification failed." }
$IntegrationReport = uv run python -m sift --integration-check
if ($LASTEXITCODE -ne 0) {
    Write-Host $IntegrationReport
    throw "Windows integration runtime qualification failed."
}
$FormatReport = uv run python -m sift --format-check
if ($LASTEXITCODE -ne 0) {
    Write-Host $FormatReport
    throw "Windows confined format-worker qualification failed."
}

if (-not $SkipTests -and $env:SIFT_SKIP_TESTS -ne "1") {
    uv run pytest -q
    if ($LASTEXITCODE -ne 0) { throw "Windows release tests failed." }
}

$Version = uv run --no-sync python packaging/windows_build_probe.py project-version
if ($LASTEXITCODE -ne 0 -or -not $Version) { throw "Could not read package version." }
if ($Version -notmatch '^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$') {
    throw "Package version '$Version' cannot be represented in Windows version metadata."
}
$NumericVersion = "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"
$WindowsVersionInfo = Join-Path $RepoRoot "packaging\generated\windows-version-info.txt"
uv run python packaging/write_windows_version_info.py `
    $WindowsVersionInfo --version $Version
if ($LASTEXITCODE -ne 0) { throw "Windows executable version-resource generation failed." }

uv run pyinstaller packaging/sift.spec --clean --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }
$Bundle = Join-Path $RepoRoot "dist\Sift"
$Executable = Join-Path $Bundle "Sift.exe"
if (-not (Test-Path $Executable)) { throw "Missing $Executable" }
if ((Get-PortableExecutableMachine $Executable) -ne 0x8664) {
    throw "Frozen Sift.exe is not an x64 PE image."
}
& $Executable --platform-check | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Frozen Windows desktop runtime qualification failed." }
$FrozenIntegrationReport = & $Executable --integration-check
if ($LASTEXITCODE -ne 0) {
    Write-Host $FrozenIntegrationReport
    throw "Frozen Windows integration runtime qualification failed."
}
$FrozenFormatReport = & $Executable --format-check
if ($LASTEXITCODE -ne 0) {
    Write-Host $FrozenFormatReport
    throw "Frozen Windows confined format-worker qualification failed."
}
& $Executable --renderer-check | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Frozen Windows WebView2 renderer qualification failed." }
$env:PYTHONDONTWRITEBYTECODE = "1"
& $Executable --analysis-check | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Frozen Windows analysis runtime qualification failed." }
& $Executable --credential-store-check | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Frozen Windows Credential Manager qualification failed." }
& $Executable --help | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Frozen executable smoke test failed." }
uv run python packaging/verify_frozen_bundle.py $Bundle
if ($LASTEXITCODE -ne 0) { throw "Frozen bundle surface verification failed." }

$ExecutableVersion = (Get-Item $Executable).VersionInfo
if ($ExecutableVersion.ProductName -ne "Sift" -or
    $ExecutableVersion.CompanyName -ne "Sapien Institute" -or
    $ExecutableVersion.OriginalFilename -ne "Sift.exe" -or
    $ExecutableVersion.ProductVersion -ne $Version) {
    throw "Frozen Windows executable branding/version resources are incomplete."
}

# PyInstaller's work tree, the source vendoring tree, and uv's download cache
# are no longer needed once the frozen bundle has passed its runtime, surface,
# and branding checks. Releasing them here keeps enough disk available to build
# and lifecycle-test both the installer and portable archive on clean CI hosts.
foreach ($Workspace in @(
    (Join-Path $RepoRoot "build"),
    (Join-Path $RepoRoot "packaging\vendor")
)) {
    if (Test-Path -LiteralPath $Workspace) {
        Remove-Item -LiteralPath $Workspace -Recurse -Force
    }
}
uv cache clean
if ($LASTEXITCODE -ne 0) { throw "uv cache cleanup failed." }

$ShouldSign = (-not $SkipSign -and [bool]$env:SIFT_WINDOWS_CERT_SHA1)
if ($ShouldSign) {
    $SignTool = (Get-Command signtool.exe -ErrorAction Stop).Source
    $Signables = @(Get-ChildItem $Bundle -Recurse -File |
        Where-Object { $_.Extension -in ".exe", ".dll", ".pyd" })
    foreach ($File in $Signables) {
        & $SignTool sign /sha1 $env:SIFT_WINDOWS_CERT_SHA1 /fd SHA256 `
            /tr http://timestamp.digicert.com /td SHA256 $File.FullName
        if ($LASTEXITCODE -ne 0) { throw "Signing failed: $($File.FullName)" }
    }
    foreach ($File in $Signables) {
        & $SignTool verify /pa /all $File.FullName
        if ($LASTEXITCODE -ne 0) {
            throw "Signature verification failed: $($File.FullName)"
        }
    }
}

uv run python packaging/write_package_metadata.py `
    (Join-Path $Bundle "release-metadata.json") `
    --version $Version --platform windows --architecture x86_64
if ($LASTEXITCODE -ne 0) { throw "Package metadata generation failed." }
Copy-Item packaging\windows\INSTALL.txt (Join-Path $Bundle "INSTALL.txt") -Force
Copy-Item LICENSE (Join-Path $Bundle "LICENSE.txt") -Force

$Archive = Join-Path $RepoRoot "dist\Sift-Windows-x64.zip"
$Installer = Join-Path $RepoRoot "dist\Sift-Windows-x64-Setup.exe"
@($Archive, $Installer, "$Archive.sha256", "$Archive.sbom.cdx.json", "$Archive.sig.json", `
    "$Installer.sha256", "$Installer.sbom.cdx.json", "$Installer.sig.json") |
    Where-Object { Test-Path $_ } |
    ForEach-Object { Remove-Item $_ -Force }

$InnoArguments = @(
    "/Qp",
    "/DMyAppVersion=$Version",
    "/DMyAppNumericVersion=$NumericVersion",
    "/DSourceDir=$Bundle",
    "/DOutputDir=$(Join-Path $RepoRoot 'dist')"
)
if ($ShouldSign) {
    # Inno creates the uninstaller while compiling Setup.  Supplying its named
    # SignTool here ensures that embedded uninstaller is Authenticode-signed;
    # signing Setup.exe afterward alone would leave Add/Remove Programs with
    # an unsigned uninstaller.
    $InnoSignCommand = '$q' + $SignTool + '$q sign /sha1 ' + `
        $env:SIFT_WINDOWS_CERT_SHA1 + `
        ' /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 $f'
    $InnoArguments += "/DSignInstaller=1"
    $InnoArguments += "/Ssiftsign=$InnoSignCommand"
}
$InnoArguments += (Join-Path $RepoRoot "packaging\windows\Sift.iss")
$QuotedInnoArguments = @($InnoArguments | ForEach-Object {
    if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ }
})
$InnoProcess = Start-Process -FilePath $InnoCompiler `
    -ArgumentList $QuotedInnoArguments -Wait -PassThru -NoNewWindow
if ($InnoProcess.ExitCode -ne 0 -or -not (Test-Path $Installer)) {
    throw "Branded Windows installer creation failed."
}
if ($ShouldSign) {
    & $SignTool sign /sha1 $env:SIFT_WINDOWS_CERT_SHA1 /fd SHA256 `
        /tr http://timestamp.digicert.com /td SHA256 $Installer
    if ($LASTEXITCODE -ne 0) { throw "Installer signing failed." }
    & $SignTool verify /pa /all $Installer
    if ($LASTEXITCODE -ne 0) { throw "Installer signature verification failed." }
}

# Create the portable artifact while the verified source bundle is available,
# then release that multi-gigabyte tree before exercising Setup. Keeping the
# bundle and Setup's installed copy at the same time can exhaust a clean hosted
# Windows runner even though a researcher machine has adequate install space.
Compress-Archive -Path $Bundle -DestinationPath $Archive -CompressionLevel Optimal
if (-not (Test-Path $Archive)) { throw "Archive creation failed." }
Remove-Item -LiteralPath $Bundle -Recurse -Force

# Exercise what researchers receive, not only the pre-installer bundle. This
# performs a silent per-user clean install, an in-place reinstall/upgrade,
# frozen runtime checks, and an uninstall while proving external user state is
# retained.
& (Join-Path $RepoRoot "packaging\qualify_windows_install.ps1") `
    -Installer $Installer
if ($LASTEXITCODE -ne 0) { throw "Installed Windows lifecycle qualification failed." }

& (Join-Path $RepoRoot "packaging\qualify_windows_portable.ps1") -Archive $Archive
if ($LASTEXITCODE -ne 0) { throw "Portable Windows archive qualification failed." }

# Preserve the historical build output for downstream frozen-executable checks
# and local release inspection. At this point Setup's temporary installation
# has been removed, so restoring the bundle does not recreate the peak.
Expand-Archive -LiteralPath $Archive -DestinationPath (Join-Path $RepoRoot "dist") -Force
if (-not (Test-Path $Executable -PathType Leaf)) {
    throw "Portable archive did not restore the qualified frozen bundle."
}

foreach ($Artifact in @($Installer, $Archive)) {
    $ArtifactHash = (Get-FileHash -Algorithm SHA256 $Artifact).Hash.ToLowerInvariant()
    $ArtifactName = Split-Path $Artifact -Leaf
    # Use an explicit LF terminator so the standard checksum sidecar works
    # unchanged with sha256sum/shasum on Windows, macOS, and Linux. PowerShell's
    # Set-Content writes the platform newline (CRLF on Windows), which causes
    # Unix checksum tools to treat the trailing carriage return as part of the
    # artifact filename.
    [System.IO.File]::WriteAllText(
        "$Artifact.sha256",
        "$ArtifactHash  $ArtifactName`n",
        [System.Text.Encoding]::ASCII
    )
    uv run python -m sift.release_manifest sbom `
        $Artifact "$Artifact.sbom.cdx.json" --version $Version
    if ($LASTEXITCODE -ne 0) { throw "SBOM generation failed for $ArtifactName." }
    uv run python -m sift.release_manifest verify-sbom `
        $Artifact "$Artifact.sbom.cdx.json"
    if ($LASTEXITCODE -ne 0) { throw "SBOM binding failed for $ArtifactName." }
    if (-not (Test-Path "$Artifact.sha256") -or -not (Test-Path "$Artifact.sbom.cdx.json")) {
        throw "Checksum or SBOM sidecar is missing for $ArtifactName."
    }
    if ($ReleaseMode -eq "production") {
        $SignedAt = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
        uv run python -m sift.release_manifest sign-file `
            $Artifact "$Artifact.sig.json" --version $Version `
            --channel $ReleaseChannel --signed-at $SignedAt `
            --key-id $env:SIFT_RELEASE_KEY_ID
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path "$Artifact.sig.json")) {
            throw "Detached release signature generation failed for $ArtifactName."
        }
    }
}

if ($ReleaseMode -eq "production") {
    Write-Host "Built, Authenticode-signed, checksummed, SBOM-recorded, and detached-signed: $Installer"
    Write-Host "Portable fallback: $Archive"
} else {
    Write-Host "Built branded installer, checksums, and SBOMs (development; signing optional): $Installer"
    Write-Host "Portable fallback: $Archive"
}
