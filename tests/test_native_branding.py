"""Cross-platform branding and native installer contracts."""

from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
APP_ID = "org.sapieninstitute.sift"


def test_committed_native_brand_assets_match_the_canonical_master() -> None:
    subprocess.run(
        [sys.executable, str(PACKAGING / "generate_brand_assets.py"), "--check"],
        cwd=ROOT,
        check=True,
    )
    manifest = json.loads((PACKAGING / "brand-assets.json").read_text(encoding="utf-8"))
    assert manifest["canonical_source"] == "packaging/icon-source.png"
    assert set(manifest["assets"]) == {
        "packaging/Sift.icns",
        "packaging/macos/installer-background.png",
        "packaging/windows/Sift.ico",
        "packaging/windows/installer-small.bmp",
        "packaging/windows/installer-wizard.bmp",
        "packaging/windows/msix/Assets/Square150x150Logo.png",
        "packaging/windows/msix/Assets/Square44x44Logo.png",
        "packaging/windows/msix/Assets/StoreLogo.png",
        "src/sift/web/app-icon.png",
        *{
            f"packaging/linux/icons/hicolor/{size}x{size}/apps/"
            f"{APP_ID}.png"
            for size in (16, 24, 32, 48, 64, 128, 256, 512)
        },
    }
    assert all("\\" not in relative for relative in manifest["assets"])
    generator = (PACKAGING / "generate_brand_assets.py").read_text(encoding="utf-8")
    assert "path.relative_to(ROOT).as_posix()" in generator


def test_windows_uses_native_icon_at_every_shell_size() -> None:
    with Image.open(PACKAGING / "windows" / "Sift.ico") as icon:
        sizes = set(icon.info["sizes"])
    assert {(n, n) for n in (16, 20, 24, 32, 40, 48, 64, 128, 256)} <= sizes
    spec = (PACKAGING / "sift.spec").read_text(encoding="utf-8")
    assert '"windows" / "Sift.ico"' in spec
    installer = (PACKAGING / "windows" / "Sift.iss").read_text(encoding="utf-8")
    for contract in (
        "SetupIconFile=Sift.ico",
        "MinVersion=10.0.22000",
        "ArchitecturesAllowed=x64compatible",
        "ArchitecturesInstallIn64BitMode=x64compatible",
        "UninstallDisplayIcon={app}\\Sift.exe",
        "WizardImageFile=installer-wizard.bmp",
        "WizardSmallImageFile=installer-small.bmp",
        "PrivilegesRequired=lowest",
        "AllowNoIcons=no",
        "UsePreviousGroup=no",
        "SignedUninstaller=yes",
        "function IsWebView2Installed",
        "Microsoft Edge WebView2 Evergreen Runtime",
        "MinimumWebView2Version = '86.0.616.0'",
        'Name: "{group}\\Sift"',
        'AppUserModelID: "org.sapieninstitute.sift"',
        "LicenseFile={#SourceDir}\\LICENSE.txt",
        "InfoBeforeFile={#SourceDir}\\INSTALL.txt",
    ):
        assert contract in installer
    # SetupArchitecture is exclusive to Inno Setup 7. Keeping the bootstrapper
    # compatible with 6.3+ lets the stable compiler used by Windows CI create
    # the same x64-only, 64-bit-install-mode package.
    assert "SetupArchitecture=" not in installer
    assert "version=WINDOWS_VERSION_INFO" in spec
    assert "console=IS_WINDOWS" in spec
    assert 'hide_console="hide-early" if IS_WINDOWS else None' in spec
    build = (PACKAGING / "build_windows.ps1").read_text(encoding="utf-8")
    assert "write_windows_version_info.py" in build
    assert build.index("write_windows_version_info.py") < build.index("pyinstaller")
    assert "Start-Process -FilePath $InnoCompiler" in build
    assert "-Wait -PassThru -NoNewWindow" in build
    assert "sift.release_manifest verify-sbom" in build
    ui = (ROOT / "src" / "sift" / "ui.py").read_text(encoding="utf-8")
    assert "SetCurrentProcessExplicitAppUserModelID" in ui
    qualify = (PACKAGING / "qualify_windows_install.ps1").read_text(encoding="utf-8")
    assert "requires a clean host with no registered Sift installation" in qualify
    assert "exactly one per-user uninstall registration" in qualify
    assert "incorrect target or working directory" in qualify
    assert "use the product uninstaller" in qualify
    assert qualify.index('$Uninstaller = Join-Path $InstallRoot') < qualify.index(
        'if (-not (Test-Path $Executable'
    )
    assert '/DIR=`"$InstallRoot`"' in qualify
    # DisableProgramGroupPage=yes makes Inno ignore /GROUP by design. The
    # lifecycle test must verify the real fixed group rather than a fictional
    # command-line override.
    assert '/GROUP=' not in qualify
    assert 'Join-Path $Programs "Sift"' in qualify
    assert "no registered Sift installation or Start-menu group" in qualify
    assert "CandidateRoot -eq $ExpectedRoot" in qualify
    assert '("Sift % $RandomSuffix")' in qualify
    assert '.Substring(0, 8)' in qualify
    assert "Upgraded Sift failed $Check" in qualify
    portable = (PACKAGING / "qualify_windows_portable.ps1").read_text(
        encoding="utf-8"
    )
    assert "Portable Windows archive qualification passed" in portable
    assert '"--platform-check", "--renderer-check", "--integration-check"' in portable
    assert '"Sift portable % "' in portable
    assert 'Substring(0, 12)' in portable
    assert "legacy MAX_PATH boundary" in portable


def test_macos_launcher_preserves_command_line_arguments() -> None:
    launcher = (PACKAGING / "launcher.sh").read_text(encoding="utf-8")
    assert 'exec "$SIFT_BIN" "$@"' in launcher
    assert "do not depend on npm" in launcher


def test_linux_desktop_metadata_and_all_hicolor_sizes_are_valid() -> None:
    desktop_path = PACKAGING / "linux" / f"{APP_ID}.desktop.in"
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str  # type: ignore[method-assign]
    parser.read(desktop_path, encoding="utf-8")
    entry = parser["Desktop Entry"]
    assert entry["Type"] == "Application"
    assert entry["Name"] == "Sift"
    assert entry["Icon"] == APP_ID
    assert entry["Terminal"] == "false"
    assert entry["Exec"] == '"__SIFT_EXECUTABLE__"'

    metainfo = ET.parse(PACKAGING / "linux" / f"{APP_ID}.metainfo.xml").getroot()
    assert metainfo.findtext("id") == APP_ID
    assert metainfo.findtext("name") == "Sift"
    assert metainfo.find("launchable").text == f"{APP_ID}.desktop"  # type: ignore[union-attr]

    for size in (16, 24, 32, 48, 64, 128, 256, 512):
        path = (
            PACKAGING / "linux" / "icons" / "hicolor" / f"{size}x{size}"
            / "apps" / f"{APP_ID}.png"
        )
        with Image.open(path) as icon:
            assert icon.size == (size, size)
            assert icon.mode == "RGBA"
    ui = (ROOT / "src" / "sift" / "ui.py").read_text(encoding="utf-8")
    assert 'icon=str(web_dir / "app-icon.png")' in ui


@pytest.mark.skipif(os.name == "nt", reason="executes the POSIX Linux installer")
def test_linux_per_user_install_and_uninstall_are_isolated(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "app").mkdir(parents=True)
    executable = bundle / "app" / "sift"
    executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    shutil.copy(PACKAGING / "linux" / "install.sh", bundle / "install.sh")
    shutil.copy(PACKAGING / "linux" / "uninstall.sh", bundle / "uninstall.sh")
    shutil.copy(PACKAGING / "linux" / "INSTALL.txt", bundle / "INSTALL.txt")
    shutil.copy(ROOT / "LICENSE", bundle / "LICENSE.txt")
    shutil.copytree(PACKAGING / "linux" / "icons", bundle / "share" / "icons")
    (bundle / "share" / "applications").mkdir(parents=True)
    (bundle / "share" / "metainfo").mkdir(parents=True)
    shutil.copy(
        PACKAGING / "linux" / f"{APP_ID}.desktop.in",
        bundle / "share" / "applications",
    )
    shutil.copy(
        PACKAGING / "linux" / f"{APP_ID}.metainfo.xml",
        bundle / "share" / "metainfo",
    )
    (bundle / "release-metadata.json").write_text(
        '{"format":"sift-package-metadata"}\n', encoding="utf-8",
    )
    (bundle / "install.sh").chmod(0o755)
    (bundle / "uninstall.sh").chmod(0o755)

    home = tmp_path / 'home with spaces % $and `ticks` "quotes"'
    data_home = home / ".local" / "share"
    bin_home = home / ".local" / "bin"
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_BIN_HOME": str(bin_home),
    }
    applications = data_home / "applications"
    applications.mkdir(parents=True)
    unrelated = home / "unrelated-desktop-file"
    unrelated.write_text("do not replace\n", encoding="utf-8")
    (applications / f"{APP_ID}.desktop").symlink_to(unrelated)
    subprocess.run([str(bundle / "install.sh")], env=env, check=True)
    assert unrelated.read_text(encoding="utf-8") == "do not replace\n"
    assert not (applications / f"{APP_ID}.desktop").is_symlink()
    desktop = (data_home / "applications" / f"{APP_ID}.desktop").read_text(encoding="utf-8")
    assert "__SIFT_EXECUTABLE__" not in desktop
    installed_parser = configparser.ConfigParser(interpolation=None)
    installed_parser.read_string(desktop)
    exec_field = installed_parser["Desktop Entry"]["Exec"]
    assert exec_field.startswith('"') and exec_field.endswith('"')
    encoded_path = exec_field[1:-1]
    decoded_path = ""
    index = 0
    while index < len(encoded_path):
        if (
            encoded_path[index] == "\\"
            and index + 1 < len(encoded_path)
            and encoded_path[index + 1] in '\\`"$'
        ):
            decoded_path += encoded_path[index + 1]
            index += 2
        else:
            decoded_path += encoded_path[index]
            index += 1
    decoded_path = decoded_path.replace("%%", "%")
    assert decoded_path == str(data_home / "sift" / "app" / "sift")
    subprocess.run([str(bin_home / "sift")], env=env, check=True)
    installed_uninstaller = data_home / "sift" / "uninstall.sh"
    assert installed_uninstaller.stat().st_mode & 0o111
    assert (data_home / "sift" / "INSTALL.txt").is_file()
    assert (data_home / "sift" / "LICENSE.txt").is_file()
    assert (data_home / "sift" / "share" / "applications" / f"{APP_ID}.desktop.in").is_file()

    # The installed copy remains a complete installer source.  Re-running it
    # exercises the unusual but valid case where BUNDLE_ROOT == APP_HOME.
    subprocess.run([str(data_home / "sift" / "install.sh")], env=env, check=True)
    subprocess.run([str(bin_home / "sift")], env=env, check=True)

    retained = home / "research-session.txt"
    retained.write_text("keep", encoding="utf-8")
    subprocess.run([str(installed_uninstaller)], env=env, check=True)
    assert not (data_home / "sift").exists()
    assert not (bin_home / "sift").exists()
    assert retained.read_text(encoding="utf-8") == "keep"


@pytest.mark.skipif(os.name == "nt", reason="executes the POSIX Linux installer")
def test_linux_failed_upgrade_restores_previous_app_and_integration(
    tmp_path: Path,
) -> None:
    def make_bundle(name: str, marker: str) -> Path:
        bundle = tmp_path / name
        (bundle / "app").mkdir(parents=True)
        executable = bundle / "app" / "sift"
        executable.write_text(
            f"#!/usr/bin/env bash\nprintf '%s' '{marker}'\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        for filename in ("install.sh", "uninstall.sh", "INSTALL.txt"):
            shutil.copy(PACKAGING / "linux" / filename, bundle / filename)
        shutil.copy(ROOT / "LICENSE", bundle / "LICENSE.txt")
        shutil.copytree(PACKAGING / "linux" / "icons", bundle / "share" / "icons")
        (bundle / "share" / "applications").mkdir(parents=True)
        (bundle / "share" / "metainfo").mkdir(parents=True)
        shutil.copy(
            PACKAGING / "linux" / f"{APP_ID}.desktop.in",
            bundle / "share" / "applications",
        )
        shutil.copy(
            PACKAGING / "linux" / f"{APP_ID}.metainfo.xml",
            bundle / "share" / "metainfo",
        )
        (bundle / "release-metadata.json").write_text(
            '{"format":"sift-package-metadata"}\n', encoding="utf-8",
        )
        (bundle / "install.sh").chmod(0o755)
        (bundle / "uninstall.sh").chmod(0o755)
        return bundle

    first = make_bundle("first", "first")
    second = make_bundle("second", "second")
    home = tmp_path / "home"
    data_home = home / ".local" / "share"
    bin_home = home / ".local" / "bin"
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_BIN_HOME": str(bin_home),
    }
    subprocess.run([str(first / "install.sh")], env=env, check=True)
    assert subprocess.check_output([str(bin_home / "sift")], env=env, text=True) == "first"

    # A directory at a managed icon path forces a failure after the new app
    # has been swapped in.  The installer must restore both the old app and
    # every integration file it touched before the failure.
    blocked_icon = (
        data_home / "icons" / "hicolor" / "48x48" / "apps"
        / f"{APP_ID}.png"
    )
    blocked_icon.unlink()
    blocked_icon.mkdir()
    desktop_before = (
        data_home / "applications" / f"{APP_ID}.desktop"
    ).read_bytes()
    completed = subprocess.run(
        [str(second / "install.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Cannot replace a directory" in completed.stderr
    assert subprocess.check_output([str(bin_home / "sift")], env=env, text=True) == "first"
    assert (
        data_home / "applications" / f"{APP_ID}.desktop"
    ).read_bytes() == desktop_before


@pytest.mark.skipif(os.name == "nt", reason="executes the POSIX Linux installer")
def test_linux_installer_refuses_to_replace_unowned_install_path(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "app").mkdir(parents=True)
    executable = bundle / "app" / "sift"
    executable.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    for filename in ("install.sh", "uninstall.sh", "INSTALL.txt"):
        shutil.copy(PACKAGING / "linux" / filename, bundle / filename)
    shutil.copy(ROOT / "LICENSE", bundle / "LICENSE.txt")
    shutil.copytree(PACKAGING / "linux" / "icons", bundle / "share" / "icons")
    (bundle / "share" / "applications").mkdir(parents=True)
    (bundle / "share" / "metainfo").mkdir(parents=True)
    shutil.copy(
        PACKAGING / "linux" / f"{APP_ID}.desktop.in",
        bundle / "share" / "applications",
    )
    shutil.copy(
        PACKAGING / "linux" / f"{APP_ID}.metainfo.xml",
        bundle / "share" / "metainfo",
    )
    (bundle / "release-metadata.json").write_text(
        '{"format":"sift-package-metadata"}\n', encoding="utf-8",
    )
    (bundle / "install.sh").chmod(0o755)

    home = tmp_path / "home"
    data_home = home / ".local" / "share"
    occupied = data_home / "sift"
    occupied.mkdir(parents=True)
    sentinel = occupied / "researcher-owned.txt"
    sentinel.write_text("retain", encoding="utf-8")
    env = {
        **os.environ,
        "HOME": str(home),
        "XDG_DATA_HOME": str(data_home),
        "XDG_BIN_HOME": str(home / ".local" / "bin"),
    }
    completed = subprocess.run(
        [str(bundle / "install.sh")], env=env, capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "Refusing to replace an unrecognized" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "retain"


@pytest.mark.skipif(os.name == "nt", reason="executes the POSIX Linux installer")
def test_linux_uninstaller_refuses_an_unowned_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    app_home = home / ".local" / "share" / "sift"
    app_home.mkdir(parents=True)
    sentinel = app_home / "researcher-owned.txt"
    sentinel.write_text("retain", encoding="utf-8")
    completed = subprocess.run(
        ["bash", str(PACKAGING / "linux" / "uninstall.sh")],
        env={**os.environ, "HOME": str(home)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "Refusing to remove an unrecognized directory" in completed.stderr
    assert sentinel.read_text(encoding="utf-8") == "retain"


def test_macos_app_and_disk_image_use_the_canonical_brand_assets() -> None:
    app = (PACKAGING / "build_app.sh").read_text(encoding="utf-8")
    assert "<string>Sift.icns</string>" in app
    assert "<key>CFBundleIconName</key>" not in app
    assert "<key>CFBundleInfoDictionaryVersion</key>" in app
    dmg = (PACKAGING / "build_dmg.sh").read_text(encoding="utf-8")
    assert ".VolumeIcon.icns" in dmg
    assert "installer-background.png" in dmg
    assert "set background picture" in dmg
    assert 'set position of item "Sift.app"' in dmg
    assert "repeat 20 times" in dmg
    assert "if installerWindow is missing value then error" in dmg
    assert "close installerWindow" in dmg


def test_every_native_builder_fails_closed_on_stale_brand_assets() -> None:
    for relative in (
        "build_app.sh", "build_dev_app.sh", "build_windows.ps1",
        "build_linux.sh", "build_dmg.sh",
    ):
        source = (PACKAGING / relative).read_text(encoding="utf-8")
        assert "generate_brand_assets.py" in source and "--check" in source, relative


def test_windows_version_resource_is_deterministic_and_complete(tmp_path: Path) -> None:
    output = tmp_path / "version-info.txt"
    subprocess.run(
        [
            sys.executable,
            str(PACKAGING / "write_windows_version_info.py"),
            str(output),
            "--version", "1.2.3-beta.1",
        ],
        check=True,
    )
    version_info = output.read_text(encoding="utf-8")
    for contract in (
        "filevers=(1, 2, 3, 0)",
        "prodvers=(1, 2, 3, 0)",
        "Sapien Institute",
        "Sift research assistant",
        "Sift.exe",
        "1.2.3-beta.1",
    ):
        assert contract in version_info
