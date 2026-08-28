"""Source-level release gates that are exercised on their native OS in CI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_windows_production_cannot_skip_signing_or_sidecars() -> None:
    source = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    assert '$ReleaseMode -eq "production"' in source
    assert 'if ($SkipSign) { throw' in source
    assert "SIFT_WINDOWS_CERT_SHA1" in source
    assert "SIFT_RELEASE_PRIVATE_KEY_B64" in source
    assert "SIFT_RELEASE_KEY_ID" in source
    assert "Get-FileHash -Algorithm SHA256" in source
    assert "[System.IO.File]::WriteAllText(" in source
    assert '"$ArtifactHash  $ArtifactName`n"' in source
    assert 'Set-Content -Path "$Artifact.sha256"' not in source
    assert "sift.release_manifest sbom" in source
    assert "sift.release_manifest sign-file" in source
    assert "signtool.exe" in source and "verify /pa /all" in source


def test_windows_build_releases_redundant_trees_before_installer_lifecycle() -> None:
    source = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    bundle_ready = 'throw "Frozen Sift.exe is not an x64 PE image."'
    frozen_format = "$FrozenFormatReport = & $Executable --format-check"
    verified = "Frozen Windows executable branding/version resources are incomplete."
    build_cleanup = 'Join-Path $RepoRoot "build"'
    vendor_cleanup = 'Join-Path $RepoRoot "packaging\\vendor"'
    source_environment_cleanup = 'Join-Path $RepoRoot ".venv"'
    build_environment_cleanup = "$env:UV_PROJECT_ENVIRONMENT"
    cache_cleanup = "uv cache clean"
    free_space_gate = "$QualificationFreeBytes -lt 512MB"
    installer = "Start-Process -FilePath $InnoCompiler"
    archive = "Compress-Archive -Path $Bundle"
    bundle_cleanup = "Remove-Item -LiteralPath $Bundle -Recurse -Force"
    sidecars = "uv run python -m sift.release_manifest sbom"
    lifecycle = "qualify_windows_install.ps1"
    portable = "qualify_windows_portable.ps1"
    restore = "Expand-Archive -LiteralPath $Archive"
    assert (
        source.index(bundle_ready)
        < source.index(build_cleanup)
        < source.index(vendor_cleanup)
        < source.index(source_environment_cleanup)
        < source.index(cache_cleanup)
        < source.index(free_space_gate)
        < source.index(frozen_format)
        < source.index(verified)
        < source.index(installer)
        < source.index(archive)
        < source.index(bundle_cleanup)
        < source.index(sidecars)
        < source.rindex(build_environment_cleanup)
        < source.index(lifecycle)
        < source.index(portable)
        < source.index(restore)
    )
    assert source.count(sidecars) == 1
    assert "uv run " not in source[source.rindex(build_environment_cleanup) :]
    assert "Release artifact changed during qualification" in source


def test_windows_installer_qualification_emits_logs_before_cleanup() -> None:
    source = (ROOT / "packaging" / "qualify_windows_install.ps1").read_text(
        encoding="utf-8"
    )
    assert "Invoke-Installer $InstallArguments $SetupLog" in source
    assert "Last 300 non-rollback lines" in source
    assert "Last 300 non-cleanup lines" in source
    assert "Select-Object -Last 300" in source
    assert "Deleting (file|directory)" in source
    assert source.index("Select-Object -Last 300") < source.index(
        "if (Test-Path $TestRoot)"
    )


def test_windows_store_msix_is_free_store_signed_and_fails_closed() -> None:
    source = (ROOT / "packaging" / "build_windows_store_msix.ps1").read_text(
        encoding="utf-8"
    )
    manifest = (
        ROOT / "packaging" / "windows" / "msix" / "AppxManifest.xml.in"
    ).read_text(encoding="utf-8")
    assert "SIFT_MSIX_IDENTITY_NAME is required" in source
    assert "SIFT_MSIX_PUBLISHER is required" in source
    assert "makeappx.exe" in source
    assert "Windows Kits\\10" in source
    assert "Install the free Windows 11 SDK" in source
    assert "& $MakeAppx pack /d $Staging /p $OutputPath /o" in source
    assert "pack /nv" not in source
    assert "appcert.exe" in source
    assert "& $AppCert reset" in source
    assert "-appxpackagepath" in source
    assert "Partner Center will replace" in source
    assert "Windows.FullTrustApplication" in manifest
    assert 'ProcessorArchitecture="x64"' in manifest
    assert '<rescap:Capability Name="runFullTrust" />' in manifest
    assert "@@IDENTITY_NAME@@" in manifest and "@@PUBLISHER@@" in manifest
    for asset in (
        "StoreLogo.png", "Square44x44Logo.png", "Square150x150Logo.png",
    ):
        assert (ROOT / "packaging" / "windows" / "msix" / "Assets" / asset).is_file()


def test_linux_production_requires_signature_checksum_and_sbom() -> None:
    source = (ROOT / "packaging" / "build_linux.sh").read_text(encoding="utf-8")
    assert 'RELEASE_MODE="${SIFT_RELEASE_MODE:-development}"' in source
    assert 'if [[ "$RELEASE_MODE" == "production" ]]' in source
    assert "SIFT_RELEASE_PRIVATE_KEY_B64" in source
    assert "SIFT_RELEASE_KEY_ID" in source
    assert 'sha256sum "$ARCHIVE"' in source
    assert "sift.release_manifest sbom" in source
    assert "sift.release_manifest sign-file" in source
    if sys.platform != "win32":
        subprocess.run(
            ["bash", "-n", str(ROOT / "packaging" / "build_linux.sh")],
            check=True,
        )


def test_linux_build_releases_redundant_trees_before_archive_lifecycle() -> None:
    source = (ROOT / "packaging" / "build_linux.sh").read_text(encoding="utf-8")
    assert 'mv dist/sift "$STAGE_ROOT/app"' in source
    assert 'cp -R dist/sift/. "$STAGE_ROOT/app/"' not in source
    verified = "uv run python packaging/verify_linux_elf_dependencies.py dist/sift"
    build_cleanup = "rm -rf -- build packaging/vendor"
    cache_cleanup = "uv cache clean"
    archive = 'tar -C "$STAGE_PARENT" -czf "$ARCHIVE" Sift'
    staging_cleanup = 'rm -rf -- "$STAGE_PARENT" dist/sift'
    qualify = '"$REPO_ROOT/packaging/qualify_linux_install.sh" "$ARCHIVE"'
    assert (
        source.index(verified)
        < source.index(build_cleanup)
        < source.index(cache_cleanup)
        < source.index(archive)
        < source.index(staging_cleanup)
        < source.index(qualify)
    )


def test_macos_release_emits_same_portable_trust_sidecars() -> None:
    source = (ROOT / "packaging" / "release.sh").read_text(encoding="utf-8")
    assert "load_update_policy" in source
    assert "Production update policy is not configured" in source
    assert "SIFT_RELEASE_PRIVATE_KEY_B64" in source
    assert "SIFT_RELEASE_KEY_ID" in source
    assert "sift.release_manifest sbom" in source
    assert "sift.release_manifest sign-file" in source
    assert '"$DMG.sha256"' in source or 'SHA256_FILE="$DMG.sha256"' in source
    if sys.platform != "win32":
        subprocess.run(
            ["bash", "-n", str(ROOT / "packaging" / "release.sh")],
            check=True,
        )


def test_portable_package_metadata_is_explicit_and_deterministic(tmp_path) -> None:
    output = tmp_path / "release-metadata.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging" / "write_package_metadata.py"),
            str(output),
            "--version", "1.2.3",
            "--platform", "linux",
            "--architecture", "x86_64",
        ],
        check=True,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema_version"] == 2
    assert document["package_kind"] == "portable_archive_with_per_user_installer"
    assert document["silent_installer_available"] is True
    assert document["requires_administrator"] is False
    assert document["executable"] == "app/sift"
    assert document["installer"]["filename"] == "install.sh"
    assert "glibc 2.35 or newer" in document["runtime_requirements"][0]
    assert "$XDG_DATA_HOME/sift/uninstall.sh" in document["uninstall"]

    arm_output = tmp_path / "release-metadata-arm64.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging" / "write_package_metadata.py"),
            str(arm_output),
            "--version", "1.2.3",
            "--platform", "linux",
            "--architecture", "aarch64",
        ],
        check=True,
    )
    arm_document = json.loads(arm_output.read_text(encoding="utf-8"))
    assert "glibc 2.39 or newer" in arm_document["runtime_requirements"][0]


def test_portable_archives_include_human_installation_guidance() -> None:
    windows = (ROOT / "packaging" / "windows" / "INSTALL.txt").read_text(
        encoding="utf-8"
    )
    linux = (ROOT / "packaging" / "linux" / "INSTALL.txt").read_text(
        encoding="utf-8"
    )
    assert "Windows Settings > Apps > Installed apps" in windows
    assert "Windows Credential Manager" in windows
    assert "build 22000 or newer" in windows
    assert "86.0.616.0 or newer" in windows
    assert "%ProgramData%\\Sift\\enterprise_policy.yaml" in windows
    assert "Do not copy Sift.exe out of its folder" in windows
    assert "sift/uninstall.sh" in linux
    assert "Freedesktop Secret Service" in linux
    assert "/etc/sift/enterprise_policy.yaml" in linux
    assert "bubblewrap" in linux
    assert "sudo ./prepare_ubuntu_host.sh" in linux
    assert "Qt WebEngine renderer profile" in linux
    assert "does not disable" in linux
    desktop = (
        ROOT / "packaging" / "linux" / "org.sapieninstitute.sift.desktop.in"
    ).read_text(encoding="utf-8")
    metainfo = (
        ROOT / "packaging" / "linux" / "org.sapieninstitute.sift.metainfo.xml"
    ).read_text(encoding="utf-8")
    assert "Categories=Science;DataVisualization;" in desktop
    assert '<url type="homepage">https://github.com/JeevanSandhu13/Sift</url>' in metainfo
    windows_build = (ROOT / "packaging" / "build_windows.ps1").read_text(
        encoding="utf-8"
    )
    linux_build = (ROOT / "packaging" / "build_linux.sh").read_text(
        encoding="utf-8"
    )
    assert "packaging\\windows\\INSTALL.txt" in windows_build
    assert "packaging/linux/INSTALL.txt" in linux_build
    assert "appstreamcli validate --no-net" in linux_build
    assert "Production cannot reuse a previously frozen application" in linux_build


def test_primary_guidance_matches_the_three_platform_product() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    overview = (ROOT / "docs" / "overview.md").read_text(encoding="utf-8")
    install = (ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    for document in (readme, overview, install):
        assert "Sift.dmg" in document
        assert "Sift-Windows-x64-Setup.exe" in document
        assert "Sift-Linux-x86_64.tar.gz" in document
        assert "Sift-Linux-aarch64.tar.gz" in document
    assert "macOS only at this stage" not in readme
    assert "Claude subscription via" not in readme
    normalized_readme = " ".join(readme.split())
    assert "Windows Credential Manager" in normalized_readme
    assert "Freedesktop Secret Service" in normalized_readme


def test_native_builds_qualify_installed_lifecycle_and_updates_fail_closed() -> None:
    mac = (ROOT / "packaging" / "build_app.sh").read_text(encoding="utf-8")
    windows = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    linux = (ROOT / "packaging" / "build_linux.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "platform-qualification.yml").read_text(
        encoding="utf-8"
    )
    assert "production update policy is not configured" in mac
    assert "production update policy is not configured" in windows
    assert "production update policy is not configured" in linux
    assert 'export PATH="$(dirname -- "$UV_BIN"):' in mac
    assert "qualify_windows_install.ps1" in windows
    assert "qualify_windows_portable.ps1" in windows
    assert "qualify_linux_install.sh" in linux
    assert "qualify_macos_install.sh" in workflow


def test_windows_metadata_names_webview2_requirement(tmp_path) -> None:
    output = tmp_path / "release-metadata.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "packaging" / "write_package_metadata.py"),
            str(output),
            "--version", "1.2.3",
            "--platform", "windows",
            "--architecture", "x86_64",
        ],
        check=True,
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    assert any(
        requirement.startswith("Microsoft Edge WebView2 Evergreen Runtime 86.0.616.0")
        for requirement in document["runtime_requirements"]
    )
    assert any(
        requirement.startswith("64-bit Windows 11 (build 22000")
        for requirement in document["runtime_requirements"]
    )
    assert document["package_kind"] == "installer_and_portable_archive"
    assert document["installer"]["filename"] == "Sift-Windows-x64-Setup.exe"
    assert document["installer"]["silent_arguments"] == [
        "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
    ]


def test_release_finalizer_requires_pinned_trust_and_all_three_platforms() -> None:
    source = (ROOT / "packaging" / "finalize_release.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--trust-store", type=Path, required=True)' in source
    assert 'parser.add_argument("--trust-store-sha256", required=True)' in source
    assert "macos,arm64" in source
    assert "windows,x86_64" in source
    assert '"x86_64": "Sift-Linux-x86_64.tar.gz"' in source
    assert '"aarch64": "Sift-Linux-aarch64.tar.gz"' in source
    assert 'action="append"' in source
    assert "architectures or list(LINUX_ARTIFACT_FILENAMES)" in source
    assert "load_trusted_json" in source
    assert "verify_release" in source


def test_release_finalizer_defaults_to_both_linux_architectures() -> None:
    import importlib.util

    module_path = ROOT / "packaging" / "finalize_release.py"
    spec = importlib.util.spec_from_file_location("sift_finalize_release", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    arguments = module.linux_artifact_arguments(Path("dist"), None)
    assert arguments == [
        "--artifact",
        f"linux,x86_64,{Path('dist') / 'Sift-Linux-x86_64.tar.gz'},application/gzip",
        "--artifact",
        f"linux,aarch64,{Path('dist') / 'Sift-Linux-aarch64.tar.gz'},application/gzip",
    ]
    assert module.linux_artifact_arguments(Path("dist"), ["aarch64", "aarch64"]) == [
        "--artifact",
        f"linux,aarch64,{Path('dist') / 'Sift-Linux-aarch64.tar.gz'},application/gzip",
    ]


def test_windows_headless_finalizer_preserves_native_security_gates() -> None:
    source = (
        ROOT / "packaging" / "windows" / "finalize_service_release.ps1"
    ).read_text(encoding="utf-8")
    wrapper = (
        ROOT / "packaging" / "windows" / "run_finalize_service_release.cmd"
    ).read_text(encoding="utf-8")
    assert "Supplemental qualification" in source
    assert "qualify_windows_install.ps1 remains the release gate" in source
    assert 'Join-Path "C:\\Users\\Public\\Documents"' in source
    assert "Get-FileHash -Algorithm SHA256 $Source" in source
    assert "Get-FileHash -Algorithm SHA256 $Installed" in source
    assert "--renderer-check is intentionally excluded" in source
    assert "CO_E_SERVER_EXEC_FAILURE" in source
    assert "--format-check is likewise per-user" in source
    assert "FormatSelectionError" in source
    assert '"--credential-store-check"' in source
    assert "Start-Process -FilePath $Executable" in source
    assert "Expand-Archive -LiteralPath $Archive" in source
    assert "Portable $Asset does not match the corrected source" in source
    assert "verify-sbom" in source
    assert "service-finalize.exit" in wrapper
    assert "POLARS_SKIP_CPU_CHECK=1" in wrapper


def test_windows_vm_receiver_is_private_atomic_and_filename_limited() -> None:
    source = (
        ROOT / "packaging" / "receive_windows_artifacts.py"
    ).read_text(encoding="utf-8")
    for filename in (
        "Sift-Windows-x64-Setup.exe",
        "Sift-Windows-x64-Setup.exe.sha256",
        "Sift-Windows-x64-Setup.exe.sbom.cdx.json",
        "Sift-Windows-x64.zip",
        "Sift-Windows-x64.zip.sha256",
        "Sift-Windows-x64.zip.sbom.cdx.json",
    ):
        assert f'"{filename}"' in source
    assert 'default="192.168.64.1"' in source
    assert "MAX_ARTIFACT_BYTES" in source
    assert "os.fsync" in source
    assert "os.replace(temporary, destination)" in source
