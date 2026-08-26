#define MyAppName "Sift"
#define MyAppPublisher "Sapien Institute"
#define MyAppExeName "Sift.exe"
#ifndef MyAppVersion
  #error MyAppVersion must be supplied by build_windows.ps1
#endif
#ifndef SourceDir
  #error SourceDir must be supplied by build_windows.ps1
#endif
#ifndef OutputDir
  #error OutputDir must be supplied by build_windows.ps1
#endif
#ifndef MyAppNumericVersion
  #error MyAppNumericVersion must be supplied by build_windows.ps1
#endif

[Setup]
AppId={{D5A030DA-6B36-4C28-A901-5146A39A71FD}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppCopyright=Copyright (C) 2026 {#MyAppPublisher}
MinVersion=10.0.22000
SetupArchitecture=x64
DefaultDirName={localappdata}\Programs\Sift
DefaultGroupName=Sift
DisableProgramGroupPage=yes
; The program-group page is intentionally hidden, so researchers cannot opt
; out there. Keep this explicit: inheriting a prior beta's "no icons" state
; would otherwise produce an installed app with no Start-menu entry.
AllowNoIcons=no
UsePreviousGroup=no
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#OutputDir}
OutputBaseFilename=Sift-Windows-x64-Setup
SetupIconFile=Sift.ico
UninstallDisplayIcon={app}\Sift.exe
UninstallDisplayName=Sift
LicenseFile={#SourceDir}\LICENSE.txt
InfoBeforeFile={#SourceDir}\INSTALL.txt
WizardStyle=modern
WizardImageFile=installer-wizard.bmp
WizardSmallImageFile=installer-small.bmp
Compression=lzma2/ultra64
SolidCompression=yes
CloseApplications=yes
RestartApplications=no
UsedUserAreasWarning=no
SetupLogging=yes
VersionInfoVersion={#MyAppNumericVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoDescription=Sift installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
#ifdef SignInstaller
SignTool=siftsign
SignedUninstaller=yes
#else
SignedUninstaller=no
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Sift"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; AppUserModelID: "org.sapieninstitute.sift"
Name: "{userdesktop}\Sift"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\{#MyAppExeName}"; AppUserModelID: "org.sapieninstitute.sift"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch Sift"; Flags: nowait postinstall skipifsilent

[Code]
const
  WebView2Client = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';
  MinimumWebView2Version = '86.0.616.0';

function HasWebView2At(RootKey: Integer; KeyPath: String): Boolean;
var
  Version: String;
  InstalledVersion: Int64;
  MinimumVersion: Int64;
begin
  Result := RegQueryStringValue(RootKey, KeyPath, 'pv', Version) and
    StrToVersion(Version, InstalledVersion) and
    StrToVersion(MinimumWebView2Version, MinimumVersion) and
    (ComparePackedVersion(InstalledVersion, MinimumVersion) >= 0);
end;

function IsWebView2Installed: Boolean;
var
  NativePath: String;
  WowPath: String;
begin
  NativePath := 'SOFTWARE\Microsoft\EdgeUpdate\Clients\' + WebView2Client;
  WowPath := 'SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\' + WebView2Client;
  Result :=
    HasWebView2At(HKCU, NativePath) or HasWebView2At(HKCU, WowPath) or
    HasWebView2At(HKLM, NativePath) or HasWebView2At(HKLM, WowPath);
end;

function InitializeSetup: Boolean;
begin
  Result := IsWebView2Installed;
  if not Result then
  begin
    Log('Sift installation blocked: Microsoft Edge WebView2 Evergreen Runtime is missing or too old.');
    if not WizardSilent then
      MsgBox(
        'Sift needs a supported Microsoft Edge WebView2 Evergreen Runtime.' + #13#10 + #13#10 +
        'Install it from Microsoft or ask your administrator to deploy it, then run this installer again.' + #13#10 +
        'https://developer.microsoft.com/microsoft-edge/webview2/',
        mbError, MB_OK);
  end;
end;
