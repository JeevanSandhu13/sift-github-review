"""Cross-platform desktop renderer and release-preflight contracts."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sift.platform_support import (
    WINDOWS_11_MINIMUM_BUILD,
    WINDOWS_WEBVIEW2_CLIENT,
    WINDOWS_WEBVIEW2_MINIMUM_VERSION,
    credential_store_roundtrip,
    desktop_runtime_report,
    preferred_webview_gui,
    runtime_architecture,
    windows_x64_emulation,
    windows_11_or_newer,
    windows_build_number,
    windows_webview2_runtime_supported,
    windows_webview2_runtime_version,
)


ROOT = Path(__file__).resolve().parents[1]


def test_each_os_has_one_reviewed_renderer() -> None:
    assert preferred_webview_gui("darwin") == "cocoa"
    assert preferred_webview_gui("win32") == "edgechromium"
    assert preferred_webview_gui("linux") == "qt"


def test_windows_runtime_architecture_follows_x64_process_not_arm_host() -> None:
    assert runtime_architecture(
        platform_name="win32",
        machine="ARM64",
        python_platform="win-amd64",
    ) == "amd64"
    assert runtime_architecture(
        platform_name="win32",
        machine="AMD64",
        python_platform="win-arm64",
    ) == "arm64"
    assert windows_x64_emulation(
        platform_name="win32",
        machine="ARM64",
        python_platform="win-amd64",
    )
    assert not windows_x64_emulation(
        platform_name="win32",
        machine="AMD64",
        python_platform="win-amd64",
    )
    assert not windows_x64_emulation(
        platform_name="win32",
        machine="ARM64",
        python_platform="win-arm64",
    )


def test_webview2_probe_checks_user_and_machine_installs_and_uses_newest() -> None:
    class Key:
        def __init__(self, version: str):
            self.version = version

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class Registry:
        HKEY_CURRENT_USER = "user"
        HKEY_LOCAL_MACHINE = "machine"

        @staticmethod
        def OpenKey(root: str, path: str):
            assert WINDOWS_WEBVIEW2_CLIENT in path
            versions = {
                ("user", rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WINDOWS_WEBVIEW2_CLIENT}"): "150.0.1.0",
                ("machine", rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WINDOWS_WEBVIEW2_CLIENT}"): "151.0.2.0",
            }
            if (root, path) not in versions:
                raise OSError("missing")
            return Key(versions[(root, path)])

        @staticmethod
        def QueryValueEx(key: Key, name: str):
            assert name == "pv"
            return key.version, 1

    assert windows_webview2_runtime_version(Registry) == "151.0.2.0"


def test_windows_runtime_support_floors_are_explicit() -> None:
    assert WINDOWS_WEBVIEW2_MINIMUM_VERSION == (86, 0, 616, 0)
    assert windows_webview2_runtime_supported("86.0.616.0") is True
    assert windows_webview2_runtime_supported("86.0.615.999") is False
    assert windows_webview2_runtime_supported("not-a-version") is False
    assert WINDOWS_11_MINIMUM_BUILD == 22_000

    class Version:
        def __init__(self, build: int) -> None:
            self.build = build

    assert windows_build_number(Version(22_000)) == 22_000
    assert windows_11_or_newer(Version(22_000)) is True
    assert windows_11_or_newer(Version(21_999)) is False


def test_current_host_report_is_content_free_and_structurally_complete() -> None:
    report = desktop_runtime_report(require_sandbox=False)
    assert report["schema_version"] == 1
    assert report["renderer"] in {"cocoa", "edgechromium", "qt"}
    assert all(set(item) == {"name", "ok", "detail"} for item in report["checks"])
    assert not any("/Users/" in str(item) for item in report["checks"])


def test_credential_store_probe_roundtrips_and_removes_random_canary() -> None:
    class SecureBackend:
        priority = 1

    class FakeKeyring:
        def __init__(self) -> None:
            self.values: dict[tuple[str, str], str] = {}

        @staticmethod
        def get_keyring() -> SecureBackend:
            return SecureBackend()

        def set_password(self, service: str, account: str, value: str) -> None:
            self.values[(service, account)] = value

        def get_password(self, service: str, account: str) -> str | None:
            return self.values.get((service, account))

        def delete_password(self, service: str, account: str) -> None:
            del self.values[(service, account)]

    ring = FakeKeyring()
    assert credential_store_roundtrip(ring) == (
        True,
        "secure OS credential-store round-trip passed",
    )
    assert ring.values == {}


def test_credential_store_probe_reports_and_contains_cleanup_failure() -> None:
    class SecureBackend:
        priority = 1

    class DeleteFailureKeyring:
        value = ""

        @staticmethod
        def get_keyring() -> SecureBackend:
            return SecureBackend()

        def set_password(self, _service: str, _account: str, value: str) -> None:
            self.value = value

        def get_password(self, _service: str, _account: str) -> str | None:
            return self.value or None

        @staticmethod
        def delete_password(_service: str, _account: str) -> None:
            raise RuntimeError("simulated vault cleanup failure")

    assert credential_store_roundtrip(DeleteFailureKeyring()) == (
        False,
        "OS credential-store canary cleanup failed",
    )


def test_platform_check_cli_emits_machine_readable_report() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "sift", "--platform-check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout.strip())
    assert report["platform"] in {"darwin", "win32", "linux"}
    assert isinstance(report["ok"], bool)
    assert completed.returncode == (0 if report["ok"] else 1)


def test_every_native_build_runs_source_and_frozen_platform_checks() -> None:
    for relative in (
        "packaging/build_app.sh",
        "packaging/build_linux.sh",
        "packaging/build_windows.ps1",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert source.count("--platform-check") >= 2, relative


def test_every_native_release_verifies_frozen_analysis_runtime() -> None:
    for relative in (
        "packaging/build_app.sh",
        "packaging/build_linux.sh",
        "packaging/build_windows.ps1",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "--analysis-check" in source, relative
        assert "cannot skip the bundled analysis runtime" in source, relative


def test_linux_release_executes_renderer_and_complete_elf_checks() -> None:
    source = (ROOT / "packaging" / "build_linux.sh").read_text(encoding="utf-8")
    assert "xvfb-run -a dist/sift/sift --renderer-check" in source
    assert "verify_linux_elf_dependencies.py dist/sift" in source
    assert "desktop-file-validate" in source
    assert "appstreamcli validate" in source
    assert "qualify_credential_store.sh" in source


def test_linux_workflow_qualifies_final_archive_after_staging_cleanup() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "platform-qualification.yml"
    ).read_text(encoding="utf-8")
    linux_step = workflow.split(
        "- name: Qualify finalized Linux release artifact", maxsplit=1
    )[1].split("- name: Qualify frozen Windows executable", maxsplit=1)[0]
    assert 'ARCHIVE="dist/Sift-Linux-x86_64.tar.gz"' in linux_step
    assert (
        "(cd dist && sha256sum --check Sift-Linux-x86_64.tar.gz.sha256)"
        in linux_step
    )
    assert 'verify-sbom \\\n            "$ARCHIVE" "$ARCHIVE.sbom.cdx.json"' in linux_step
    assert 'qualify_linux_install.sh "$ARCHIVE"' in linux_step
    assert "dist/sift/sift" not in linux_step


def test_ubuntu_2404_artifact_job_prepares_confinement_policy() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "platform-qualification.yml"
    ).read_text(encoding="utf-8")
    newer_job = workflow.split("qualify-linux-newer:", maxsplit=1)[1]
    preparation = "sudo bash packaging/linux/prepare_ubuntu_host.sh"
    qualification = (
        "bash packaging/qualify_linux_install.sh "
        "dist/Sift-Linux-x86_64.tar.gz"
    )
    assert preparation in newer_job
    assert qualification in newer_job
    assert newer_job.index(preparation) < newer_job.index(qualification)


def test_windows_release_executes_real_webview2_renderer_check() -> None:
    build = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    spec = (ROOT / "packaging" / "sift.spec").read_text(encoding="utf-8")
    qualify = (ROOT / "packaging" / "qualify_windows_install.ps1").read_text(
        encoding="utf-8"
    )
    assert "--renderer-check" in build
    assert "--renderer-check" in qualify
    assert "--credential-store-check" in build
    assert "--credential-store-check" in qualify
    # Inno Setup 6.3+ supports an x64-only app in 64-bit install mode. Accept
    # both stable 6.x and 7.x compiler layouts, including their per-user names.
    assert '$InnoCandidates = @(' in build
    assert '"Inno Setup 7\\ISCC.exe"' in build
    assert '"InnoSetup7\\ISCC.exe"' in build
    assert '"Inno Setup 6\\ISCC.exe"' in build
    assert '"InnoSetup6\\ISCC.exe"' in build
    assert "Inno Setup 6.3 or newer is required" in build
    # A windowed-only PyInstaller executable has no standard streams on
    # Windows. Sift keeps a console only when launched from an existing
    # console, while normal shell launches hide the console immediately.
    assert "console=IS_WINDOWS" in spec
    assert 'hide_console="hide-early" if IS_WINDOWS else None' in spec
    assert '"cpython-3.12.11-windows-x86_64-none"' in build
    assert 'platform -ne "win-amd64"' in build
    assert "Get-PortableExecutableMachine" in build
    assert "0x8664" in build


def test_linux_build_uses_stable_archive_architecture_and_glibc_baseline() -> None:
    source = (ROOT / "packaging" / "build_linux.sh").read_text(encoding="utf-8")
    assert "glibc 2.39 baseline" in source
    assert "glibc 2.35 or older" in source
    assert "command -v cc" in source
    assert "build-essential" in source
    assert "--no-binary-package cryptography" in source
    assert "Rust 1.83 or newer" in source
    assert "AESGCM.generate_key" in source
    assert 'Sift-Linux-${MANIFEST_ARCH}.tar.gz' in source
    # Static release-contract checks above remain cross-platform.  The Bash
    # parser is a native Linux/macOS tool and is covered by those release
    # hosts; a stock Windows installation correctly has no `bash` executable.
    if sys.platform != "win32":
        subprocess.run(
            ["bash", "-n", str(ROOT / "packaging" / "build_linux.sh")],
            check=True,
        )


def test_linux_qt_wheels_honor_the_glibc_baseline() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    marker = "sys_platform == 'linux'"
    for requirement in (
        "pyqt6==6.11.0",
        "pyqt6-qt6==6.11.2",
        "pyqt6-webengine==6.11.0",
        "pyqt6-webengine-qt6==6.11.2",
    ):
        assert f'"{requirement}; {marker}"' in project


def test_native_qualification_workflow_covers_all_three_operating_systems() -> None:
    workflow = (ROOT / ".github" / "workflows" / "platform-qualification.yml").read_text(encoding="utf-8")
    for runner in ("macos-15", "windows-2025", "ubuntu-22.04"):
        assert runner in workflow
    assert "Windows Server 2025 x64 compatibility" in workflow
    assert "uv sync --locked --all-extras" in workflow
    assert "uv run pytest -q" in workflow
    assert "packaging/vendor_python.py" in workflow
    # The source check runs on every matrix host. macOS and Windows retain a
    # directly addressable frozen tree, while Linux moves its verified tree
    # into the final archive to avoid holding a multi-gigabyte duplicate.
    assert workflow.count("--platform-check") >= 3
    assert "Qualify finalized Linux release artifact" in workflow
    assert workflow.count("--analysis-check") == 2
    linux_build = (ROOT / "packaging" / "build_linux.sh").read_text(
        encoding="utf-8"
    )
    assert "dist/sift/sift --analysis-check" in linux_build
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "20cfd1bf945f4377ade1205e4dbc17946fc9a30d" in workflow
    assert "ubuntu-24.04" in workflow
    assert "qualify_credential_store.sh" in linux_build
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow


def test_windows_11_has_a_real_client_os_qualification_route() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "windows-11-native-qualification.yml"
    ).read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, Windows, X64, sift-windows-11]" in workflow
    assert "Win32_OperatingSystem" in workflow
    assert "Windows 11" in workflow
    assert "ProductType -ne 1" in workflow
    assert ".\\packaging\\build_windows.ps1 -SkipSign" in workflow
    assert "windows-11-qualification.json" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow


def test_linux_arm64_has_a_native_baseline_qualification_route() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "linux-arm64-native-qualification.yml"
    ).read_text(encoding="utf-8")
    assert "runs-on: [self-hosted, Linux, ARM64, sift-ubuntu-24-04-arm64]" in workflow
    assert 'test "$(uname -m)" = "aarch64"' in workflow
    assert 'test "$VERSION_ID" = "24.04"' in workflow
    assert 'test "$(getconf GNU_LIBC_VERSION)" = "glibc 2.39"' in workflow
    assert "bash packaging/build_linux.sh" in workflow
    assert "build-essential" in workflow
    assert "dtolnay/rust-toolchain@f8be11a05b1d4f3fcebe6410cc16743212b999b0" in workflow
    assert "libssl-dev" in workflow
    assert "libffi-dev" in workflow
    assert "pkg-config" in workflow
    assert "Sift-Linux-aarch64.tar.gz" in workflow
    assert "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1" in workflow
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in workflow


def test_development_app_build_is_atomic_and_disk_guarded() -> None:
    source = (ROOT / "packaging" / "build_dev_app.sh").read_text(encoding="utf-8")
    assert ".Sift.app.staging" in source
    assert "SIFT_DEV_APP_MIN_FREE_KIB:-524288" in source
    assert source.index("plutil -lint") < source.index('rm -rf "$APP_BUNDLE"')
    assert "trap 'rm -rf \"$STAGING_BUNDLE\"' EXIT" in source
