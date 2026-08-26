param(
    [Parameter(Mandatory = $true)]
    [string]$Archive
)

$ErrorActionPreference = "Stop"
if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "Windows portable qualification must run on Windows."
}

$ResolvedArchive = (Resolve-Path -LiteralPath $Archive).Path
# Keep both a space and a literal percent sign in the extraction path, but do
# not let the qualification harness itself push valid bundled entries beyond
# the legacy MAX_PATH boundary used by Windows PowerShell's Expand-Archive.
# The archive's longest current entry is ~170 characters; a compact random
# suffix keeps the complete extracted path below 260 on a normal user profile.
$RandomSuffix = [guid]::NewGuid().ToString("N").Substring(0, 12)
$TestRoot = Join-Path ([IO.Path]::GetTempPath()) ("Sift portable % " + $RandomSuffix)

try {
    New-Item -ItemType Directory -Path $TestRoot | Out-Null
    Expand-Archive -LiteralPath $ResolvedArchive -DestinationPath $TestRoot

    $Bundle = Join-Path $TestRoot "Sift"
    $Executable = Join-Path $Bundle "Sift.exe"
    $Internal = Join-Path $Bundle "_internal"
    if (-not (Test-Path $Executable -PathType Leaf)) {
        throw "Portable archive did not contain Sift\Sift.exe."
    }
    if (-not (Test-Path $Internal -PathType Container)) {
        throw "Portable archive did not contain Sift's adjacent runtime."
    }
    foreach ($RequiredFile in "INSTALL.txt", "LICENSE.txt", "release-metadata.json") {
        if (-not (Test-Path (Join-Path $Bundle $RequiredFile) -PathType Leaf)) {
            throw "Portable archive is missing $RequiredFile."
        }
    }

    $MetadataPath = Join-Path $Bundle "release-metadata.json"
    $Metadata = Get-Content -LiteralPath $MetadataPath -Raw | ConvertFrom-Json
    if ($Metadata.format -ne "sift-package-metadata" -or
        $Metadata.platform -ne "windows" -or
        $Metadata.architecture -ne "x86_64" -or
        $Metadata.executable -ne "Sift.exe" -or
        $Metadata.install_scope -ne "per-user" -or
        $Metadata.requires_administrator -ne $false) {
        throw "Portable archive release metadata is inconsistent."
    }

    $VersionInfo = (Get-Item $Executable).VersionInfo
    if ($VersionInfo.ProductName -ne "Sift" -or
        $VersionInfo.CompanyName -ne "Sapien Institute" -or
        $VersionInfo.OriginalFilename -ne "Sift.exe" -or
        $VersionInfo.ProductVersion -ne $Metadata.version) {
        throw "Portable Sift.exe branding or version metadata is incomplete."
    }

    $env:PYTHONDONTWRITEBYTECODE = "1"
    foreach ($Check in "--platform-check", "--renderer-check", "--integration-check", "--format-check", "--analysis-check", "--credential-store-check", "--help") {
        & $Executable $Check | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Portable Sift failed $Check."
        }
    }
    Write-Host "Portable Windows archive qualification passed."
}
finally {
    if (Test-Path $TestRoot) {
        Remove-Item $TestRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
