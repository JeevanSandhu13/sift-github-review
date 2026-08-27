"""Release artifacts must include every database integration they advertise."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_os_release_installs_and_verifies_all_database_extras() -> None:
    for relative in (
        "packaging/build_app.sh",
        "packaging/build_linux.sh",
        "packaging/build_windows.ps1",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "sync --locked --all-extras" in text, relative
        assert "packaging/verify_database_drivers.py" in text, relative
        assert source_and_frozen_check_count(text) >= 2, relative


def test_linux_release_fails_early_when_required_native_libraries_are_absent() -> None:
    text = (ROOT / "packaging/build_linux.sh").read_text(encoding="utf-8")
    for soname in (
        "libodbc.so.2",
        "libxcb-shape.so.0",
        "libxcb-icccm.so.4",
        "libxcb-keysyms.so.1",
        "libxcb-dri3.so.0",
        "libxcb-image.so.0",
        "libxcb-randr.so.0",
        "libxcb-render-util.so.0",
        "libxcb-sync.so.1",
        "libxcb-util.so.1",
        "libxcb-xfixes.so.0",
        "libxcb-xkb.so.1",
        "libpulse.so.0",
        "libtbb.so.12",
        "libsnappy.so.1",
        "libwebpdemux.so.2",
        "libwebpmux.so.3",
        "libasound.so.2",
        "libminizip.so.1",
        "libgbm.so.1",
        "libEGL.so.1",
        "libnspr4.so",
        "libnss3.so",
        "libXdamage.so.1",
        "libxkbfile.so.1",
    ):
        assert f'require_shared_library "{soname}"' in text


def source_and_frozen_check_count(text: str) -> int:
    return text.count("--integration-check")


def test_frozen_bundle_collects_dynamic_database_drivers() -> None:
    spec = (ROOT / "packaging/sift.spec").read_text(encoding="utf-8")
    for module in (
        "psycopg",
        "pymysql",
        "oracledb",
        "snowflake",
        "sqlalchemy_bigquery",
        "sqlalchemy_redshift",
        "redshift_connector",
        "databricks",
    ):
        assert f'runtime_submodules("{module}")' in spec
    assert '"pyodbc"' in spec
    for distribution in (
        "snowflake-sqlalchemy",
        "sqlalchemy-bigquery",
        "sqlalchemy-redshift",
        "databricks-sqlalchemy",
    ):
        assert f'optional_metadata("{distribution}")' in spec


def test_frozen_bundle_collects_cloud_source_sdks() -> None:
    spec = (ROOT / "packaging/sift.spec").read_text(encoding="utf-8")
    for module in (
        "boto3",
        "botocore",
        "google.cloud.storage",
        "azure.identity",
        "azure.storage.blob",
        "paramiko",
    ):
        assert f'runtime_submodules("{module}")' in spec


def test_frozen_bundle_collects_dynamic_provider_sdks() -> None:
    spec = (ROOT / "packaging/sift.spec").read_text(encoding="utf-8")
    for module in ("openai", "google.genai", "anthropic"):
        assert f'runtime_submodules("{module}")' in spec


def test_frozen_bundle_collects_gdal_and_proj_runtime_registries() -> None:
    spec = (ROOT / "packaging/sift.spec").read_text(encoding="utf-8")
    assert 'collect_data_files(\n        "pyogrio"' in spec
    assert 'collect_data_files(\n        "rasterio"' in spec
    assert '"gdal_data/**", "proj_data/**"' in spec
    assert "+ GEOSPATIAL_RUNTIME_DATAS" in spec


def test_every_native_installer_qualification_rechecks_integrations() -> None:
    for relative in (
        "packaging/qualify_macos_install.sh",
        "packaging/qualify_linux_install.sh",
        "packaging/qualify_windows_install.ps1",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "--integration-check" in text, relative
        assert "--format-check" in text, relative


def test_frozen_linux_bundle_collects_secure_secret_service_backend() -> None:
    spec = (ROOT / "packaging/sift.spec").read_text(encoding="utf-8")
    assert 'runtime_submodules("keyring.backends")' in spec
    assert 'runtime_submodules("secretstorage")' in spec
    assert 'runtime_submodules("jeepney")' in spec
