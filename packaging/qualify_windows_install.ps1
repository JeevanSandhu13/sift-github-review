param(
    [Parameter(Mandatory = $true)]
    [string]$Installer
)

$ErrorActionPreference = "Stop"
if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
    throw "Windows installer qualification must run on Windows."
}
$ResolvedInstaller = (Resolve-Path $Installer).Path
# Keep the qualification location representative of the real per-user default
# while retaining a space and percent sign to catch quoting bugs. The bundled
# scientific runtime contains legitimate deep package paths, so an artificially
# long test prefix would cross legacy MAX_PATH even though the default install
# location does not.
$RandomSuffix = [guid]::NewGuid().ToString("N").Substring(0, 8)
$TestRoot = Join-Path ([IO.Path]::GetTempPath()) ("Sift % $RandomSuffix")
$InstallRoot = Join-Path $TestRoot "Sift"
$ExpectedRoot = [IO.Path]::GetFullPath(
    $InstallRoot
).TrimEnd([IO.Path]::DirectorySeparatorChar)
$SetupLog = Join-Path $TestRoot "setup.log"
$UninstallLog = Join-Path $TestRoot "uninstall.log"
$Programs = [Environment]::GetFolderPath([Environment+SpecialFolder]::Programs)
$Desktop = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
$StartMenuGroup = Join-Path $Programs "Sift"
$StartMenuShortcut = Join-Path $StartMenuGroup "Sift.lnk"
$DesktopShortcut = Join-Path $Desktop "Sift.lnk"
$DesktopShortcutExisted = Test-Path $DesktopShortcut -PathType Leaf
$DefaultExecutable = Join-Path $env:LOCALAPPDATA "Programs\Sift\Sift.exe"
$UninstallRegistryPaths = @(
    'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{D5A030DA-6B36-4C28-A901-5146A39A71FD}_is1',
    'HKCU:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\{D5A030DA-6B36-4C28-A901-5146A39A71FD}_is1'
)
$Uninstaller = $null

# The real installer AppId is intentionally exercised.  On a non-clean host
# that would replace the researcher's registered Sift installation even when
# /DIR points at a temporary folder.  Fail before changing anything instead.
if ((Test-Path $DefaultExecutable -PathType Leaf) -or
    (Test-Path $StartMenuGroup) -or
    @($UninstallRegistryPaths | Where-Object { Test-Path $_ }).Count -gt 0) {
    throw "Windows installer qualification requires a clean host with no registered Sift installation or Start-menu group."
}

function Invoke-Installer([string[]]$Arguments, [string]$LogPath) {
    $Process = Start-Process -FilePath $ResolvedInstaller -ArgumentList $Arguments `
        -Wait -PassThru -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $LogPath -PathType Leaf) {
            Write-Host "Last 300 non-rollback lines of the Sift installer log:"
            Get-Content -LiteralPath $LogPath |
                Where-Object { $_ -notmatch ' Deleting (file|directory): ' } |
                Select-Object -Last 300 |
                Write-Host
        } else {
            Write-Host "The Sift installer did not create its requested log: $LogPath"
        }
        throw "Sift installer exited with code $($Process.ExitCode)."
    }
}

try {
    New-Item -ItemType Directory -Path $TestRoot | Out-Null
    $InstallArguments = @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        "/MERGETASKS=!desktopicon", "/DIR=`"$InstallRoot`"",
        "/LOG=`"$SetupLog`""
    )
    Invoke-Installer $InstallArguments $SetupLog
    $Executable = Join-Path $InstallRoot "Sift.exe"
    # Set this immediately after Setup returns so every subsequent failure
    # can use the product's own uninstaller from the finally block.
    $Uninstaller = Join-Path $InstallRoot "unins000.exe"
    if (-not (Test-Path $Executable -PathType Leaf)) {
        throw "Installed Sift.exe is missing."
    }
    if (-not (Test-Path $StartMenuShortcut -PathType Leaf)) {
        throw "Installed Sift Start-menu shortcut is missing."
    }
    if (-not $DesktopShortcutExisted -and (Test-Path $DesktopShortcut -PathType Leaf)) {
        throw "Silent installation created the opt-in desktop shortcut."
    }
    foreach ($Documentation in "INSTALL.txt", "LICENSE.txt", "release-metadata.json") {
        if (-not (Test-Path (Join-Path $InstallRoot $Documentation) -PathType Leaf)) {
            throw "Installed Sift documentation/metadata is missing: $Documentation"
        }
    }
    $VersionInfo = (Get-Item $Executable).VersionInfo
    if ($VersionInfo.ProductName -ne "Sift" -or
        $VersionInfo.CompanyName -ne "Sapien Institute" -or
        $VersionInfo.OriginalFilename -ne "Sift.exe") {
        throw "Installed Sift.exe has incomplete Windows branding resources."
    }
    $Shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($StartMenuShortcut)
    if ($Shortcut.TargetPath -ne $Executable -or $Shortcut.WorkingDirectory -ne $InstallRoot) {
        throw "Installed Sift Start-menu shortcut has an incorrect target or working directory."
    }
    $Registrations = @($UninstallRegistryPaths | Where-Object { Test-Path $_ })
    if ($Registrations.Count -ne 1) {
        throw "Sift installer did not create exactly one per-user uninstall registration."
    }
    $Registration = Get-ItemProperty $Registrations[0]
    if (-not $Registration.InstallLocation) {
        throw "Sift uninstall registration has no installation location."
    }
    $RegisteredRoot = [IO.Path]::GetFullPath(
        [string]$Registration.InstallLocation
    ).TrimEnd([IO.Path]::DirectorySeparatorChar)
    $RegisteredUninstaller = [string]$Registration.UninstallString
    if ($Registration.DisplayName -ne "Sift" -or
        $Registration.Publisher -ne "Sapien Institute" -or
        $RegisteredRoot -ne $ExpectedRoot -or
        $RegisteredUninstaller.IndexOf(
            $Uninstaller, [StringComparison]::OrdinalIgnoreCase
        ) -lt 0) {
        throw "Sift uninstall registration is incomplete or points at the wrong installation."
    }
    $env:PYTHONDONTWRITEBYTECODE = "1"
    foreach ($Check in "--platform-check", "--renderer-check", "--integration-check", "--format-check", "--analysis-check", "--credential-store-check", "--help") {
        & $Executable $Check | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Installed Sift failed $Check." }
    }

    # Reinstall over the same location to exercise the upgrade path. An
    # unrelated user-state sentinel must survive because installers never own
    # or remove research sessions.
    $UserState = Join-Path $TestRoot "researcher-state"
    New-Item -ItemType Directory -Path $UserState | Out-Null
    Set-Content -Path (Join-Path $UserState "session.sentinel") -Value "retain" -NoNewline
    Invoke-Installer $InstallArguments $SetupLog
    if ((Get-Content (Join-Path $UserState "session.sentinel") -Raw) -ne "retain") {
        throw "Upgrade changed researcher-owned state."
    }
    foreach ($Check in "--platform-check", "--renderer-check", "--integration-check", "--format-check", "--analysis-check", "--credential-store-check", "--help") {
        & $Executable $Check | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Upgraded Sift failed $Check." }
    }

    if (-not (Test-Path $Uninstaller -PathType Leaf)) {
        throw "Windows uninstaller is missing."
    }
    $Process = Start-Process -FilePath $Uninstaller -ArgumentList @(
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
        "/LOG=`"$UninstallLog`""
    ) -Wait -PassThru -NoNewWindow
    if ($Process.ExitCode -ne 0) {
        if (Test-Path -LiteralPath $UninstallLog -PathType Leaf) {
            Write-Host "Last 300 non-cleanup lines of the Sift uninstaller log:"
            Get-Content -LiteralPath $UninstallLog |
                Where-Object { $_ -notmatch ' Deleting (file|directory): ' } |
                Select-Object -Last 300 |
                Write-Host
        }
        throw "Sift uninstaller exited with code $($Process.ExitCode)."
    }
    if (Test-Path $Executable) { throw "Uninstall left the application executable behind." }
    if (Test-Path $StartMenuShortcut) { throw "Uninstall left the Start-menu shortcut behind." }
    if (@($UninstallRegistryPaths | Where-Object { Test-Path $_ }).Count -ne 0) {
        throw "Uninstall left the per-user application registration behind."
    }
    if ((Get-Content (Join-Path $UserState "session.sentinel") -Raw) -ne "retain") {
        throw "Uninstall removed researcher-owned state."
    }
    Write-Host "Windows clean install, upgrade, execution, and uninstall qualification passed."
}
finally {
    # If a check failed after Setup completed, use the product uninstaller
    # before deleting the test directory so Add/Remove Programs is not left
    # with an orphaned qualification entry.
    if ($Uninstaller -and (Test-Path $Uninstaller -PathType Leaf)) {
        Start-Process -FilePath $Uninstaller -ArgumentList @(
            "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"
        ) -Wait -NoNewWindow -ErrorAction SilentlyContinue
    }
    # A catastrophically incomplete Setup could omit its uninstaller after
    # writing registration. Remove only a registration proven to point at
    # this unique temporary qualification root; never touch another install.
    foreach ($RegistryPath in $UninstallRegistryPaths) {
        if (-not (Test-Path $RegistryPath)) { continue }
        $Candidate = Get-ItemProperty $RegistryPath -ErrorAction SilentlyContinue
        if (-not $Candidate -or -not $Candidate.InstallLocation) { continue }
        $CandidateRoot = [IO.Path]::GetFullPath(
            [string]$Candidate.InstallLocation
        ).TrimEnd([IO.Path]::DirectorySeparatorChar)
        if ($CandidateRoot -eq $ExpectedRoot) {
            Remove-Item $RegistryPath -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    if (Test-Path $StartMenuGroup) {
        Remove-Item $StartMenuGroup -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $TestRoot) {
        Remove-Item $TestRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
}
