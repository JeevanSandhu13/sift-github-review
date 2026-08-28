"""Frozen-integration gate stays offline, complete, and content-free."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sift.integration_runtime import integration_runtime_report
from sift.format_selection import _worker_command, worker_main


ROOT = Path(__file__).resolve().parents[1]
_NUMPY_CYTHON_SIZE_WARNING = (
    r"ignore:^numpy\.ndarray size changed, may indicate binary "
    r"incompatibility\.:RuntimeWarning"
)


@pytest.mark.filterwarnings(_NUMPY_CYTHON_SIZE_WARNING)
def test_integration_runtime_report_imports_every_advertised_surface() -> None:
    report = integration_runtime_report()
    assert report["ok"] is True, [
        row for row in report["checks"]
        if row["required"] and not row["ok"]
    ]
    assert report["check_count"] == len(report["checks"])
    categories = {row["category"] for row in report["checks"]}
    assert categories == {
        "database_driver",
        "database_dialect",
        "cloud_source",
        "provider_implementation",
        "provider_sdk",
        "core_data_format",
        "selected_data_format",
        "runtime_data",
    }
    assert all(set(row) == {
        "category", "id", "module", "required", "ok", "detail",
    }
               for row in report["checks"])
    optional = [row for row in report["checks"] if not row["required"]]
    if sys.platform == "win32":
        assert [(row["category"], row["id"], row["module"]) for row in optional] == [
            ("selected_data_format", "bcf", "pysam"),
        ]
    else:
        assert optional == []


@pytest.mark.filterwarnings(_NUMPY_CYTHON_SIZE_WARNING)
def test_netcdf4_compiled_runtime_round_trip() -> None:
    """Exercise the compiled netCDF4 boundary, not only its Python import."""
    from netCDF4 import Dataset

    dataset = Dataset("in-memory.nc", mode="w", diskless=True, persist=False)
    try:
        dataset.createDimension("observation", 3)
        values = dataset.createVariable("measurement", "f8", ("observation",))
        values[:] = [1.25, 2.5, 5.0]
        dataset.sync()
        assert values[:].tolist() == [1.25, 2.5, 5.0]
    finally:
        dataset.close()


def test_platform_optional_import_is_visible_without_failing_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift import integration_runtime

    real_import = integration_runtime.importlib.import_module

    def import_without_pysam(name: str):
        if name == "pysam":
            raise ModuleNotFoundError(name)
        return real_import(name)

    monkeypatch.setattr(integration_runtime.sys, "platform", "win32")
    monkeypatch.setattr(
        integration_runtime.importlib,
        "import_module",
        import_without_pysam,
    )
    report = integration_runtime.integration_runtime_report()
    bcf = next(
        row for row in report["checks"]
        if row["category"] == "selected_data_format" and row["id"] == "bcf"
    )
    assert bcf == {
        "category": "selected_data_format",
        "id": "bcf",
        "module": "pysam",
        "required": False,
        "ok": False,
        "detail": "not packaged on win32",
    }
    assert integration_runtime._required_checks_pass([bcf]) is True


def test_integration_check_cli_is_machine_readable_and_offline() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "sift", "--integration-check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    # The check must never disclose machine paths in either success or
    # failure diagnostics.
    assert str(Path.home()) not in completed.stdout


def test_frozen_format_workers_route_through_application_flags(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    command = _worker_command(
        "materialize", Path("input.xml"), Path("selection.json"),
        Path("output.parquet"), Path("metadata.json"),
    )
    assert command == [
        sys.executable, "--format-worker", "input.xml", "selection.json",
        "output.parquet", "metadata.json",
    ]


def test_format_worker_refuses_direct_unconfined_invocation(
    monkeypatch, capsys,
) -> None:
    monkeypatch.delenv("SIFT_PROCESS_TREE_MARKER", raising=False)
    assert worker_main(["--list-worker", "input.xml", "output.json"]) == 1
    assert "parent marker is absent" in capsys.readouterr().err


def test_runtime_data_lookup_uses_materialized_frozen_root(
    monkeypatch, tmp_path: Path,
) -> None:
    from sift import integration_runtime

    resource = tmp_path / "pyogrio" / "gdal_data" / "gdalvrt.xsd"
    resource.parent.mkdir(parents=True)
    resource.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert integration_runtime._package_resource_check(
        "geospatial", "pyogrio", "gdal_data/gdalvrt.xsd",
    )["ok"] is True


def test_format_runtime_check_cli_materializes_in_confined_child() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "sift", "--format-check"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    report = json.loads(completed.stdout)
    if (
        completed.returncode != 0
        and str(report.get("detail", "")).startswith("materialize:backend-health")
    ):
        from sift.executor import cached_environment
        from sift.format_selection import FormatSelectionError, _require_parser_backend

        try:
            _require_parser_backend(cached_environment(), sys.platform)
        except FormatSelectionError:
            pytest.skip(
                "the host sandbox backend cannot be applied inside this test environment"
            )
    assert completed.returncode == 0, report
    assert report == {
        "schema_version": 1,
        "ok": True,
        "check": "confined_complex_format_worker",
        "detail": "synthetic XML materialization passed",
    }
