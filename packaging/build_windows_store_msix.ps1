param(
    [string]$Bundle = "",
    [string]$Output = "",
    [switch]$SkipRuntimeChecks
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $RepoRoot
if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "The Microsoft Store package must be built and checked on Windows."
}

# Partner Center assigns both values after the publisher reserves the product
# name. Guessing either produces a package that cannot be submitted, so this
# builder fails before copying the multi-gigabyte application bundle.
if (-not $env:SIFT_MSIX_IDENTITY_NAME) {
    throw "SIFT_MSIX_IDENTITY_NAME is required (copy Package/Identity/Name from Partner Center)."
}
if (-not $env:SIFT_MSIX_PUBLISHER) {
    throw "SIFT_MSIX_PUBLISHER is required (copy Package/Identity/Publisher from Partner Center)."
}
if ($env:SIFT_MSIX_IDENTITY_NAME -notmatch '^[A-Za-z0-9][A-Za-z0-9.-]{2,49}$') {
    throw "SIFT_MSIX_IDENTITY_NAME is not a valid Store package identity."
}
if ($env:SIFT_MSIX_PUBLISHER -notmatch '^CN=[^\x00-\x1f<>]{1,240}$') {
    throw "SIFT_MSIX_PUBLISHER is not a valid Partner Center publisher subject."
}
$PublisherDisplayName = if ($env:SIFT_MSIX_PUBLISHER_DISPLAY_NAME) {
    $env:SIFT_MSIX_PUBLISHER_DISPLAY_NAME
} else {
    "Sapien Institute"
}
if ($PublisherDisplayName -match '[\x00-\x1f<>]' -or $PublisherDisplayName.Length -gt 256) {
    throw "SIFT_MSIX_PUBLISHER_DISPLAY_NAME contains unsafe manifest characters."
}

$BundlePath = if ($Bundle) { $Bundle } else { Join-Path $RepoRoot "dist\Sift" }
$OutputPath = if ($Output) {
    $Output
} else {
    Join-Path $RepoRoot "dist\Sift-Windows-x64.msix"
}
if (-not (Test-Path $BundlePath -PathType Container)) {
    throw "The frozen Windows bundle is missing: $BundlePath"
}
$Executable = Join-Path $BundlePath "Sift.exe"
if (-not (Test-Path $Executable -PathType Leaf)) {
    throw "The frozen Windows executable is missing."
}
foreach ($Required in "_internal", "INSTALL.txt", "LICENSE.txt", "release-metadata.json") {
    if (-not (Test-Path (Join-Path $BundlePath $Required))) {
        throw "The Windows bundle is incomplete: $Required"
    }
}

function Find-WindowsSdkTool([string]$Name) {
    $Command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($Command) { return $Command.Source }
    $KitsRoot = Join-Path ([Environment]::GetFolderPath("ProgramFilesX86")) `
        "Windows Kits\10"
    if (Test-Path $KitsRoot -PathType Container) {
        $Candidate = Get-ChildItem $KitsRoot -Recurse -File -Filter $Name `
            -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '\\x64\\' } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($Candidate) { return $Candidate.FullName }
    }
    return $null
}

$MakeAppx = Find-WindowsSdkTool "makeappx.exe"
if (-not $MakeAppx) {
    throw "MakeAppx is missing. Install the free Windows 11 SDK, then rerun this builder."
}
if (-not $SkipRuntimeChecks) {
    & $Executable --platform-check | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Frozen Windows platform check failed." }
    & $Executable --integration-check | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Frozen Windows integration check failed." }
    & $Executable --analysis-check | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Frozen Windows analysis check failed." }
}

$PinnedPython = Join-Path $RepoRoot ".uv-python\cpython-3.12.11-windows-x86_64-none\python.exe"
$Version = ""
if (Test-Path $PinnedPython -PathType Leaf) {
    $Version = (& $PinnedPython `
        -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
} else {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "Could not read the Sift version; uv or the pinned Windows Python is required."
    }
    $Version = uv run --no-sync python packaging/windows_build_probe.py project-version
}
if ($Version -notmatch '^(\d+)\.(\d+)\.(\d+)(?:[-+].*)?$') {
    throw "Sift version '$Version' cannot be represented by MSIX."
}
$MsixVersion = "$($Matches[1]).$($Matches[2]).$($Matches[3]).0"

function Escape-Xml([string]$Value) {
    return [System.Security.SecurityElement]::Escape($Value)
}

$StagingParent = Join-Path $RepoRoot "dist\.Sift-Windows-MSIX.staging"
$Staging = Join-Path $StagingParent "package"
if (Test-Path $StagingParent) { Remove-Item $StagingParent -Recurse -Force }
New-Item -ItemType Directory -Path $Staging | Out-Null
try {
    Copy-Item (Join-Path $BundlePath "*") $Staging -Recurse -Force
    Copy-Item (Join-Path $RepoRoot "packaging\windows\msix\Assets") `
        (Join-Path $Staging "Assets") -Recurse -Force
    $Template = Get-Content `
        (Join-Path $RepoRoot "packaging\windows\msix\AppxManifest.xml.in") -Raw
    $Manifest = $Template.Replace(
        "@@IDENTITY_NAME@@", (Escape-Xml $env:SIFT_MSIX_IDENTITY_NAME)
    ).Replace(
        "@@PUBLISHER@@", (Escape-Xml $env:SIFT_MSIX_PUBLISHER)
    ).Replace(
        "@@PUBLISHER_DISPLAY_NAME@@", (Escape-Xml $PublisherDisplayName)
    ).Replace(
        "@@VERSION@@", $MsixVersion
    )
    if ($Manifest -match '@@[A-Z_]+@@') {
        throw "The MSIX manifest still contains an unresolved value."
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $Staging "AppxManifest.xml"),
        $Manifest,
        [System.Text.UTF8Encoding]::new($false)
    )

    if (Test-Path $OutputPath) { Remove-Item $OutputPath -Force }
    & $MakeAppx pack /d $Staging /p $OutputPath /o
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $OutputPath -PathType Leaf)) {
        throw "MakeAppx failed to create the Store package."
    }
    # MakeAppx performs semantic package validation by default. Never pass /nv,
    # which would suppress the exact manifest/file checks needed here.

    # The Windows App Certification Kit is installed with the SDK on release
    # hosts. Its command location varies, so run it when discoverable and make
    # the absence explicit rather than pretending certification happened.
    $AppCert = Find-WindowsSdkTool "appcert.exe"
    if ($AppCert) {
        $Report = "$OutputPath.wack.xml"
        & $AppCert reset
        if ($LASTEXITCODE -ne 0) { throw "Windows App Certification Kit reset failed." }
        & $AppCert test -appxpackagepath $OutputPath -reportoutputpath $Report
        if ($LASTEXITCODE -ne 0) {
            throw "Windows App Certification Kit rejected the Store package."
        }
    } else {
        Write-Warning "Windows App Certification Kit was not found; Partner Center certification remains required."
    }
} finally {
    if (Test-Path $StagingParent) { Remove-Item $StagingParent -Recurse -Force }
}

$Hash = (Get-FileHash -Algorithm SHA256 $OutputPath).Hash.ToLowerInvariant()
[System.IO.File]::WriteAllText(
    "$OutputPath.sha256",
    "$Hash  $(Split-Path $OutputPath -Leaf)`n",
    [System.Text.Encoding]::ASCII
)
Write-Host "Built unsigned Microsoft Store submission package: $OutputPath"
Write-Host "Partner Center will replace the package signature with Microsoft's trusted signature."
