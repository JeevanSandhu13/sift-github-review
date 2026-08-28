from __future__ import annotations

import gzip
import importlib.util
import json
import os
import pickle
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from sift import schema
from sift.format_selection import (
    FormatSelectionError,
    format_capabilities,
    list_container_objects,
    materialize_selected_format,
    recognize_omop_tables,
    reject_unsafe_serialization,
)
import sift.format_selection as format_selection_module


def _live_parser_backend_ready() -> bool:
    from sift.executor import cached_environment
    try:
        format_selection_module._require_parser_backend(
            cached_environment(), sys.platform,
        )
    except FormatSelectionError:
        return False
    return True


requires_live_parser_backend = pytest.mark.skipif(
    not _live_parser_backend_ready(),
    reason="the host sandbox backend cannot be applied inside this test environment",
)


def _materialize(session: Path, source: Path, selection: dict | None = None) -> tuple[pd.DataFrame, dict]:
    output = materialize_selected_format(
        session, source=source, selection=selection or {},
        output_name=f"{source.stem.replace('.', '_')}.parquet",
    )
    metadata = json.loads(output.with_suffix(".parquet.metadata.json").read_text(encoding="utf-8"))
    assert metadata["parser_pid"] != os.getpid()
    frame = schema.load_data(output)
    assert schema.row_count(output) == len(frame)
    assert schema.extract(output, "names_only")["observation_count"] == len(frame)
    from sift.dataset_profile import profile_dataset
    profile = profile_dataset(output)
    assert profile["ok"] is True and profile["rows_profiled"] == len(frame)
    return frame, metadata


def test_parser_read_paths_include_virtualenv_base_runtime(tmp_path: Path) -> None:
    source = tmp_path / "input.zip"
    source.write_bytes(b"fixture")

    paths = format_selection_module._parser_read_paths(source)

    assert str(Path(sys.executable).resolve()) in paths
    assert str(Path(sys.prefix).resolve()) in paths
    assert str(Path(sys.base_prefix).resolve()) in paths
    assert str(source.resolve()) in paths


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            "isolated format parser AppContainer cleanup failed; "
            "the confinement state is no longer trusted",
            "materialize:appcontainer-cleanup",
        ),
        (
            "isolated format parser could not start inside confinement",
            "materialize:confinement-launch",
        ),
        (
            "isolated format parser rejected the input",
            "materialize:parser-exit",
        ),
    ],
)
def test_format_self_check_failure_detail_is_actionable_without_host_data(
    message: str,
    expected: str,
) -> None:
    secret_path = "/confidential/researcher/patient-8472.xml"
    detail = format_selection_module._self_check_failure_detail(
        "materialize",
        FormatSelectionError(f"{message}: {secret_path}"),
    )
    assert detail == expected
    assert secret_path not in detail


class _ParserProcessStub:
    def __init__(self) -> None:
        self.pid = 424242
        self.returncode = 0
        self.stdout = object()
        self.stderr = object()
        self.communicate_calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        self.communicate_calls += 1
        return "", ""

    def poll(self) -> int | None:
        return self.returncode if self.communicate_calls else None

    def kill(self) -> None:
        self.returncode = -9


@pytest.mark.parametrize(
    ("platform_name", "environment"),
    [
        ("darwin", SimpleNamespace(sandbox_exec=None, bwrap=None, appcontainer_support=False)),
        ("linux", SimpleNamespace(sandbox_exec=None, bwrap=None, appcontainer_support=False)),
        ("win32", SimpleNamespace(sandbox_exec=None, bwrap=None, appcontainer_support=False)),
        ("freebsd14", SimpleNamespace(sandbox_exec=None, bwrap=None, appcontainer_support=False)),
    ],
)
def test_parser_never_falls_back_to_an_unsandboxed_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_name: str,
    environment: SimpleNamespace,
) -> None:
    source = tmp_path / "input.xml"
    source.write_text("<root />")
    monkeypatch.setattr(format_selection_module.sys, "platform", platform_name)
    monkeypatch.setattr("sift.executor.cached_environment", lambda: environment)
    monkeypatch.setattr(
        format_selection_module.subprocess,
        "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("plain parser process was launched"),
        ),
    )
    with pytest.raises(FormatSelectionError, match="refuses to run a parser unsandboxed"):
        format_selection_module._run_parser_worker(
            ["python", "worker.py"],
            staging=tmp_path,
            source=source,
            timeout_seconds=10,
        )


@pytest.mark.parametrize("platform_name", ["darwin", "linux"])
def test_parser_posix_launch_uses_sandbox_limits_bounded_capture_and_tree_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    platform_name: str,
) -> None:
    import sift.executor as executor
    import sift.process_tree as process_tree

    source = tmp_path / "input.xml"
    source.write_text("<root />")
    staging = tmp_path / "stage"
    staging.mkdir()
    environment = SimpleNamespace(
        sandbox_exec="/usr/bin/sandbox-exec",
        bwrap="/usr/bin/bwrap",
        appcontainer_support=False,
    )
    monkeypatch.setattr(format_selection_module.sys, "platform", platform_name)
    monkeypatch.setattr(format_selection_module, "_require_parser_backend", lambda *args: None)
    monkeypatch.setattr(executor, "cached_environment", lambda: environment)
    monkeypatch.setattr(
        format_selection_module,
        "_trusted_posix_launch_shell",
        lambda: Path("/bin/bash"),
    )
    monkeypatch.setattr(executor, "script_min_free_disk_bytes", lambda: 0)
    monkeypatch.setattr(
        executor, "_write_sandbox_profile", lambda *args, **kwargs: staging / "sandbox.sb",
    )
    monkeypatch.setattr(executor, "_bwrap_argv", lambda *args, **kwargs: ["--unshare-net"])
    monkeypatch.setattr(
        executor, "_resource_limited_argv", lambda command, platform: ["limit-wrapper", *command],
    )
    observed: dict[str, object] = {}
    proc = _ParserProcessStub()

    def fake_popen(command: list[str], **kwargs: object) -> _ParserProcessStub:
        observed["command"] = command
        observed["popen"] = kwargs
        return proc

    def fake_communicate(process: object, **kwargs: object) -> tuple[str, str]:
        observed["limits"] = kwargs
        return "bounded stdout", "bounded stderr"

    cleanup: list[object] = []
    monkeypatch.setattr(format_selection_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(executor, "_communicate_with_memory_guard", fake_communicate)
    monkeypatch.setattr(process_tree, "attach_posix_descendant_tracker", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        process_tree, "tracked_process_identities", lambda process: (object(),),
    )
    monkeypatch.setattr(
        process_tree, "terminate_tracked_process_tree",
        lambda process: cleanup.append(process) or True,
    )

    result = format_selection_module._run_parser_worker(
        ["python", "worker.py"],
        staging=staging,
        source=source,
        timeout_seconds=17,
    )
    command = observed["command"]
    assert isinstance(command, list) and "limit-wrapper" in command
    assert (
        "/usr/bin/sandbox-exec" in command
        if platform_name == "darwin"
        else "/usr/bin/bwrap" in command and "--unshare-net" in command
    )
    assert observed["popen"]["start_new_session"] is True  # type: ignore[index]
    assert observed["limits"] == {
        "timeout_seconds": 17,
        "memory_limit_bytes": format_selection_module.PARSER_MEMORY_LIMIT_BYTES,
        "process_limit": format_selection_module.PARSER_PROCESS_LIMIT,
        "cpu_limit_seconds": 17.0,
        "disk_directory": staging,
        "disk_reserve_bytes": 0,
    }
    assert result.stdout == "bounded stdout" and result.stderr == "bounded stderr"
    assert cleanup == [proc]


def test_parser_windows_launch_uses_appcontainer_job_limits_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import sift.executor as executor
    import sift.win_appcontainer as win_appcontainer

    source = tmp_path / "input.xml"
    source.write_text("<root />")
    staging = tmp_path / "stage"
    staging.mkdir()
    environment = SimpleNamespace(
        sandbox_exec=None, bwrap=None, appcontainer_support=True,
    )
    proc = _ParserProcessStub()
    observed: dict[str, object] = {}

    class FakeAppContainerRun:
        def __init__(self, *args: object, **kwargs: object) -> None:
            observed["args"] = args
            observed["kwargs"] = kwargs

        def __enter__(self) -> _ParserProcessStub:
            observed["entered"] = True
            return proc

        def __exit__(self, *args: object) -> None:
            observed["exited"] = True

    monkeypatch.setattr(format_selection_module.sys, "platform", "win32")
    monkeypatch.setattr(format_selection_module, "_require_parser_backend", lambda *args: None)
    monkeypatch.setattr(executor, "cached_environment", lambda: environment)
    monkeypatch.setattr(executor, "script_min_free_disk_bytes", lambda: 1234)
    monkeypatch.setattr(executor, "_disk_reserve_preflight_error", lambda *args: None)
    monkeypatch.setattr(executor, "script_file_size_limit_bytes", lambda: 5678)
    monkeypatch.setattr(win_appcontainer, "AppContainerRun", FakeAppContainerRun)
    monkeypatch.setattr(
        format_selection_module.subprocess, "Popen",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Windows parser bypassed AppContainer"),
        ),
    )

    result = format_selection_module._run_parser_worker(
        ["python", "worker.py"],
        staging=staging,
        source=source,
        timeout_seconds=19,
    )
    assert result.returncode == 0
    assert observed["entered"] is True and observed["exited"] is True
    assert observed["kwargs"]["memory_bytes"] == format_selection_module.PARSER_MEMORY_LIMIT_BYTES  # type: ignore[index]
    assert observed["kwargs"]["max_processes"] == format_selection_module.PARSER_PROCESS_LIMIT  # type: ignore[index]
    assert observed["kwargs"]["cpu_seconds"] == 19  # type: ignore[index]
    assert observed["kwargs"]["max_file_size_bytes"] == 5678  # type: ignore[index]
    assert observed["kwargs"]["min_free_disk_bytes"] == 1234  # type: ignore[index]
    assert str(source) in observed["kwargs"]["extra_read_paths"]  # type: ignore[index]
    assert observed["args"][2] == staging / "astropy"  # type: ignore[index]
    worker_environment = observed["args"][3]  # type: ignore[index]
    assert worker_environment["HOME"] == str(staging / "astropy")
    assert worker_environment["TEMP"] == str(staging / "astropy")
    assert worker_environment["USERPROFILE"] == str(staging / "astropy")
    assert worker_environment["XDG_CONFIG_HOME"] == str(staging)


def test_parser_stderr_cannot_disclose_source_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    archive = tmp_path / "secret.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.csv", "private_value\nresearch-secret-8472\n")
    monkeypatch.setattr(
        format_selection_module,
        "_run_parser_worker",
        lambda *args, **kwargs: format_selection_module.subprocess.CompletedProcess(
            args=["parser"],
            returncode=1,
            stdout="research-secret-8472",
            stderr="native parser rejected value research-secret-8472",
        ),
    )
    with pytest.raises(FormatSelectionError) as caught:
        list_container_objects(archive)
    assert "research-secret-8472" not in str(caught.value)
    assert str(archive) not in str(caught.value)
    assert str(caught.value) == "isolated container inspection rejected the input"


def test_worker_metadata_reader_rejects_symlinks_and_oversized_files(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * 33)
    with pytest.raises(FormatSelectionError, match="malformed metadata"):
        format_selection_module._read_worker_json(oversized, max_bytes=32)

    target = tmp_path / "private.json"
    target.write_text('{"secret":"must-not-be-followed"}')
    link = tmp_path / "worker.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(FormatSelectionError, match="malformed metadata"):
        format_selection_module._read_worker_json(link)


@pytest.mark.parametrize(("name", "body", "rows"), [
    ("table.csv.gz", "a,b\n1,x\n2,y\n", 2),
    ("table.tsv.gz", "a\tb\n1\tx\n2\ty\n", 2),
    ("table.jsonl.gz", '{"a":1,"b":"x"}\n{"a":2,"b":"y"}\n', 2),
])
def test_gzip_tables_schema_rows_profile_and_load_agree(
    tmp_path: Path, name: str, body: str, rows: int,
) -> None:
    path = tmp_path / name
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(body)
    loaded = schema.load_data(path)
    assert len(loaded) == rows
    assert schema.row_count(path) == rows
    extracted = schema.extract(path, "names_types_labels_summary")
    assert extracted["observation_count"] == rows
    assert {row["name"] for row in extracted["variables"]} == {"a", "b"}


def test_gzip_expansion_guard(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "bomb.csv.gz"
    with gzip.open(path, "wb") as handle:
        handle.write(b"a\n" + b"1\n" * 10_000)
    monkeypatch.setattr(schema, "full_load_max_bytes", lambda: 1024)
    with pytest.raises(schema.DatasetTooLargeError, match="expands"):
        schema.load_data(path)


@requires_live_parser_backend
def test_zip_requires_exact_safe_member_and_runs_isolated(tmp_path: Path) -> None:
    archive = tmp_path / "tables.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("one.csv", "id,value\n1,a\n")
        handle.writestr("two.csv", "id,value\n2,b\n")
    assert [row["id"] for row in list_container_objects(archive)] == ["one.csv", "two.csv"]
    with pytest.raises(FormatSelectionError, match="explicit member"):
        materialize_selected_format(tmp_path, source=archive)
    frame, metadata = _materialize(tmp_path, archive, {"member": "two.csv"})
    assert frame["id"].tolist() == [2]
    assert metadata["archive_member"] == "two.csv"


def test_zip_worker_places_scratch_inside_parent_confined_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    archive = tmp_path / "tables.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.csv", "id\n1\n")
    staging = tmp_path / "staging"
    staging.mkdir()
    monkeypatch.setattr(
        format_selection_module.tempfile,
        "mkdtemp",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("confined worker attempted to create a scratch directory"),
        ),
    )
    frame, metadata = format_selection_module._parse_to_frame(
        archive,
        {"member": "data.csv"},
        scratch_dir=staging,
    )
    assert frame["id"].tolist() == [1]
    assert metadata["archive_member"] == "data.csv"
    extracted = list(staging.glob("*.csv"))
    assert len(extracted) == 1 and extracted[0].is_file()


@requires_live_parser_backend
def test_container_inspection_itself_runs_out_of_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    archive = tmp_path / "one.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.csv", "id\n1\n")
    monkeypatch.setattr(
        "sift.format_selection._list_container_objects_direct",
        lambda path: (_ for _ in ()).throw(AssertionError("ran in desktop process")),
    )
    assert list_container_objects(archive)[0]["id"] == "data.csv"


def test_worker_container_listing_does_not_require_parent_directory_traversal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A Windows AppContainer receives the selected file, not its siblings."""
    archive = tmp_path / "one.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.csv", "id\n1\n")

    def forbidden_resolve(*args: object, **kwargs: object) -> Path:
        raise PermissionError("parent directory traversal was attempted")

    monkeypatch.setattr(Path, "resolve", forbidden_resolve)
    rows = format_selection_module._list_container_objects_direct(archive)
    assert rows == [{"id": "data.csv", "bytes": 5}]


@requires_live_parser_backend
def test_zip_traversal_and_high_ratio_are_rejected(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.zip"
    with zipfile.ZipFile(traversal, "w") as handle:
        handle.writestr("../escape.csv", "x\n1\n")
    # Worker diagnostics can contain source values, so the host intentionally
    # exposes only the fixed rejection class across this trust boundary.
    with pytest.raises(FormatSelectionError, match="rejected"):
        list_container_objects(traversal)
    ratio = tmp_path / "ratio.zip"
    with zipfile.ZipFile(ratio, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("zeros.csv", b"0" * 1_000_000)
    with pytest.raises(FormatSelectionError, match="rejected"):
        list_container_objects(ratio)


@requires_live_parser_backend
def test_avro_xml_and_dbf_real_parsers(tmp_path: Path) -> None:
    from fastavro import writer
    avro = tmp_path / "rows.avro"
    with avro.open("wb") as handle:
        writer(handle, {"type": "record", "name": "Row", "fields": [
            {"name": "id", "type": "long"}, {"name": "label", "type": "string"},
        ]}, [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}])
    session = tmp_path / "avro-session"; session.mkdir()
    frame, _ = _materialize(session, avro)
    assert frame.dtypes["id"].kind in "iu" and frame["label"].tolist() == ["a", "b"]

    xml = tmp_path / "rows.xml"
    xml.write_text("<root><row><id>1</id><label>a</label></row><row><id>2</id><label>b</label></row></root>")
    session = tmp_path / "xml-session"; session.mkdir()
    with pytest.raises(FormatSelectionError, match="record_path"):
        materialize_selected_format(session, source=xml)
    frame, metadata = _materialize(session, xml, {"record_path": "root/row"})
    assert len(frame) == 2 and metadata["record_path"] == "row"

    import geopandas as gpd
    from shapely.geometry import Point
    bundle = tmp_path / "shape"
    gpd.GeoDataFrame({"id": [1], "name": ["a"]}, geometry=[Point(0, 0)], crs="EPSG:4326").to_file(bundle.with_suffix(".shp"))
    dbf = bundle.with_suffix(".dbf")
    session = tmp_path / "dbf-session"; session.mkdir()
    frame, _ = _materialize(session, dbf)
    assert frame["id"].tolist() == [1]


@requires_live_parser_backend
def test_scientific_containers_require_and_preserve_selection(tmp_path: Path) -> None:
    import h5py
    h5 = tmp_path / "arrays.h5"
    with h5py.File(h5, "w") as handle:
        data = handle.create_dataset("group/table", data=np.array([[1, 2], [3, 4]], dtype=np.int16))
        data.attrs["units"] = "mg"
        handle.create_dataset("private", data=np.array([99]))
    assert {row["id"] for row in list_container_objects(h5)} == {"group/table", "private"}
    session = tmp_path / "h5-session"; session.mkdir()
    frame, metadata = _materialize(session, h5, {"dataset": "group/table"})
    assert frame.shape == (2, 2) and metadata["attributes"]["units"] == "mg"
    assert all(dtype == np.dtype("int16") for dtype in frame.dtypes)

    import xarray as xr
    nc = tmp_path / "climate.nc"
    xr.Dataset({"temperature": (("time", "site"), [[10.0, 11.0], [12.0, 13.0]], {"units": "degC"}),
                "private": (("time",), [1, 2])}, coords={"time": [2020, 2021], "site": ["a", "b"]}).to_netcdf(nc, engine="scipy")
    session = tmp_path / "nc-session"; session.mkdir()
    frame, metadata = _materialize(session, nc, {"variable": "temperature"})
    assert len(frame) == 4
    assert metadata["dimensions"] == ["time", "site"]
    assert set(metadata["coordinates"]) == {"time", "site"}
    assert metadata["units"] == "degC"

    from scipy.io import savemat
    mat = tmp_path / "workspace.mat"
    savemat(mat, {"chosen": np.array([[1.0, 2.0]]), "other": np.array([[99.0]])})
    assert {row["id"] for row in list_container_objects(mat)} == {"chosen", "other"}
    session = tmp_path / "mat-session"; session.mkdir()
    frame, metadata = _materialize(session, mat, {"variable": "chosen"})
    assert frame.shape == (1, 2) and metadata["variable"] == "chosen"


@requires_live_parser_backend
def test_fits_requires_domain_flag_and_selected_hdu(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from astropy.io import fits
    path = tmp_path / "sky.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.array([[1.0, 2.0]]), name="SCI")]).writeto(path)
    with pytest.raises(FormatSelectionError, match="capability flag"):
        list_container_objects(path)
    monkeypatch.setenv("SIFT_DOMAIN_CAPABILITIES", "astronomy")
    assert [row["id"] for row in list_container_objects(path)] == ["0", "1"]
    session = tmp_path / "fits-session"; session.mkdir()
    frame, metadata = _materialize(session, path, {"hdu": "1"})
    assert frame.shape == (1, 2) and metadata["hdu"] == 1


@requires_live_parser_backend
def test_geospatial_formats_preserve_crs_validate_geometry_and_sidecars(tmp_path: Path) -> None:
    geojson = tmp_path / "points.geojson"
    geojson.write_text(json.dumps({"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"id": 1},
         "geometry": {"type": "Point", "coordinates": [1, 2]}},
    ]}))
    session = tmp_path / "geojson-session"; session.mkdir()
    frame, metadata = _materialize(session, geojson)
    assert frame["id"].tolist() == [1] and metadata["crs"]

    import geopandas as gpd
    from shapely.geometry import Point
    geo = gpd.GeoDataFrame({"id": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs="EPSG:4326")
    gpkg = tmp_path / "layers.gpkg"
    geo.to_file(gpkg, layer="chosen", driver="GPKG")
    session = tmp_path / "gpkg-session"; session.mkdir()
    frame, metadata = _materialize(session, gpkg, {"layer": "chosen"})
    assert len(frame) == 2 and "4326" in metadata["crs"] and metadata["invalid_geometries"] == 0

    shp = tmp_path / "points.shp"
    geo.to_file(shp)
    session = tmp_path / "shp-session"; session.mkdir()
    frame, metadata = _materialize(session, shp)
    assert len(frame) == 2 and "4326" in metadata["crs"]
    shp.with_suffix(".shx").unlink()
    failed_session = tmp_path / "shp-session-2"; failed_session.mkdir()
    with pytest.raises(FormatSelectionError, match="rejected"):
        materialize_selected_format(failed_session, source=shp)


@requires_live_parser_backend
def test_bounded_raster_workflow_preserves_crs(tmp_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_origin
    raster = tmp_path / "small.tif"
    with rasterio.open(raster, "w", driver="GTiff", width=2, height=2, count=1,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(0, 2, 1, 1)) as output:
        output.write(np.array([[1, 2], [3, 4]], dtype=np.uint8), 1)
    session = tmp_path / "raster-session"; session.mkdir()
    frame, metadata = _materialize(session, raster)
    assert frame.loc[0, "mean"] == 2.5 and metadata["crs"] == "EPSG:4326"


@requires_live_parser_backend
def test_invalid_geojson_geometry_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "invalid.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"id": 1},
        "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [1, 1],
            [1, 0], [0, 1], [0, 0]]]},
    }]}))
    session = tmp_path / "invalid-geo-session"; session.mkdir()
    _, metadata = _materialize(session, path)
    assert metadata["invalid_geometries"] == 1


@requires_live_parser_backend
def test_domain_formats_are_flagged_and_bounded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vcf = tmp_path / "variants.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n##contig=<ID=1>\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n1\t10\trs1\tA\tG\t50\tPASS\t.\n")
    with pytest.raises(FormatSelectionError, match="genomics"):
        materialize_selected_format(tmp_path, source=vcf)
    monkeypatch.setenv("SIFT_DOMAIN_CAPABILITIES", "genomics,medical_imaging,clinical")
    session = tmp_path / "vcf-session"; session.mkdir()
    frame, _ = _materialize(session, vcf)
    assert frame.loc[0, "id"] == "rs1"
    if importlib.util.find_spec("pysam") is not None:
        import pysam
        bcf = tmp_path / "variants.bcf"
        with pysam.VariantFile(vcf) as source_variants:
            with pysam.VariantFile(bcf, "wb", header=source_variants.header) as output_variants:
                for record in source_variants:
                    output_variants.write(record)
        session = tmp_path / "bcf-session"; session.mkdir()
        bcf_frame, _ = _materialize(session, bcf)
        assert bcf_frame.loc[0, "id"] == "rs1"

    import nibabel as nib
    nii = tmp_path / "brain.nii"
    nib.save(nib.Nifti1Image(np.arange(8, dtype=np.float32).reshape(2, 2, 2), np.eye(4)), nii)
    session = tmp_path / "nii-session"; session.mkdir()
    frame, metadata = _materialize(session, nii)
    assert len(frame) == 1 and metadata["shape"] == [2, 2, 2]

    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    dcm = tmp_path / "image.dcm"
    meta = FileMetaDataset(); meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(dcm), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = generate_uid(); dataset.SOPInstanceUID = generate_uid()
    dataset.PatientName = "Sensitive^Name"; dataset.Modality = "MR"
    dataset.save_as(dcm)
    session = tmp_path / "dcm-session"; session.mkdir()
    frame, metadata = _materialize(session, dcm)
    assert not bool(frame.loc[0, "deidentified"])
    assert "PatientName" in metadata["deidentification_issues"]
    assert "Sensitive" not in json.dumps(metadata)

    fhir = tmp_path / "records.fhir"
    fhir.write_text(json.dumps({"resourceType": "Bundle", "type": "collection", "entry": [
        {"resource": {"resourceType": "Patient", "id": "p1", "active": True}},
        {"resource": {"resourceType": "Observation", "id": "o1", "status": "final"}},
    ]}))
    session = tmp_path / "fhir-session"; session.mkdir()
    frame, metadata = _materialize(session, fhir)
    assert len(frame) == 2 and metadata["resource_types"] == ["Observation", "Patient"]


@requires_live_parser_backend
def test_dicom_deidentification_requires_positive_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIFT_DOMAIN_CAPABILITIES", "medical_imaging")
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    def write_image(name: str, *, removed: str | None, burned: str | None) -> Path:
        path = tmp_path / name
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        dataset.SOPClassUID = generate_uid()
        dataset.SOPInstanceUID = generate_uid()
        dataset.Modality = "MR"
        dataset.Rows = 1
        dataset.Columns = 1
        dataset.BitsAllocated = 8
        dataset.PixelData = b"\0"
        if removed is not None:
            dataset.PatientIdentityRemoved = removed
        if burned is not None:
            dataset.BurnedInAnnotation = burned
        dataset.save_as(path)
        return path

    unknown = write_image("unknown.dcm", removed=None, burned=None)
    unknown_session = tmp_path / "unknown-session"
    unknown_session.mkdir()
    frame, metadata = _materialize(unknown_session, unknown)
    assert not bool(frame.loc[0, "deidentified"])
    assert metadata["patient_identity_removed"] == "UNKNOWN"
    assert metadata["burned_in_annotation"] == "UNKNOWN"
    assert metadata["pixel_data_status"] == "NOT_INSPECTED"
    assert metadata["pixels_read"] is False

    certified = write_image("certified.dcm", removed="YES", burned="NO")
    certified_session = tmp_path / "certified-session"
    certified_session.mkdir()
    frame, metadata = _materialize(certified_session, certified)
    assert bool(frame.loc[0, "deidentified"])
    assert metadata["deidentification_issues"] == []


@requires_live_parser_backend
def test_dicom_private_elements_prevent_deidentification_certification(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("SIFT_DOMAIN_CAPABILITIES", "medical_imaging")
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    path = tmp_path / "private.dcm"
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = generate_uid()
    dataset.SOPInstanceUID = generate_uid()
    dataset.PatientIdentityRemoved = "YES"
    dataset.BurnedInAnnotation = "NO"
    dataset.add_new((0x0011, 0x1010), "LO", "opaque private metadata")
    dataset.save_as(path)

    session = tmp_path / "private-session"
    session.mkdir()
    frame, metadata = _materialize(session, path)
    assert not bool(frame.loc[0, "deidentified"])
    assert metadata["deidentification_issues"] == ["PrivateDataElements"]
    assert "opaque private metadata" not in json.dumps(metadata)


@requires_live_parser_backend
def test_plink_bed_bim_fam_bundle(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SIFT_DOMAIN_CAPABILITIES", "genomics")
    prefix = tmp_path / "toy"
    prefix.with_suffix(".fam").write_text("F1 I1 0 0 1 -9\nF1 I2 0 0 2 -9\n")
    prefix.with_suffix(".bim").write_text("1 rs1 0 100 A G\n")
    prefix.with_suffix(".bed").write_bytes(bytes([0x6C, 0x1B, 0x01, 0b00001000]))
    session = tmp_path / "plink-session"; session.mkdir()
    frame, metadata = _materialize(session, prefix.with_suffix(".bed"))
    assert metadata["samples"] == 2 and metadata["variants"] == 1
    assert metadata["sidecars"] == ["toy.bim", "toy.fam"]
    assert frame["sample_id"].tolist() == ["I1", "I2"]


def test_unsafe_python_objects_and_ambiguous_containers_fail_closed(tmp_path: Path) -> None:
    pickled = tmp_path / "renamed.bin"
    pickled.write_bytes(pickle.dumps({"payload": 1}))
    with pytest.raises(FormatSelectionError, match="pickle"):
        reject_unsafe_serialization(pickled)
    legacy = tmp_path / "legacy-renamed.bin"
    legacy.write_bytes(pickle.dumps({"payload": 1}, protocol=0))
    with pytest.raises(FormatSelectionError, match="pickle"):
        reject_unsafe_serialization(legacy)
    for name in ("unsafe.pkl", "unsafe.joblib", "unsafe.pickle"):
        path = tmp_path / name; path.write_bytes(b"not even valid")
        with pytest.raises(FormatSelectionError, match="never accepted"):
            reject_unsafe_serialization(path)

    h5 = tmp_path / "ambiguous.h5"
    import h5py
    with h5py.File(h5, "w") as handle:
        handle["one"] = [1]; handle["two"] = [2]
    with pytest.raises(FormatSelectionError, match="explicit dataset"):
        materialize_selected_format(tmp_path, source=h5)


def test_symlink_sources_and_output_collisions_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source.geojson"
    source.write_text('{"type":"FeatureCollection","features":[]}')
    link = tmp_path / "linked.geojson"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(FormatSelectionError, match="regular local"):
        materialize_selected_format(tmp_path, source=link)

    session = tmp_path / "collision-session"; session.mkdir()
    sidecar = session / "chosen.parquet.metadata.json"
    sidecar.write_text("owner data")
    with pytest.raises(FormatSelectionError, match="already exists"):
        materialize_selected_format(
            session, source=source, output_name="chosen.parquet",
        )
    assert sidecar.read_text(encoding="utf-8") == "owner data"
    assert not (session / "chosen.parquet").exists()


def test_parser_errors_do_not_echo_source_values(tmp_path: Path) -> None:
    source = tmp_path / "broken.xml"
    source.write_text("<root><row>SECRET-PATIENT-VALUE</root>")
    session = tmp_path / "broken-session"; session.mkdir()
    with pytest.raises(FormatSelectionError) as caught:
        materialize_selected_format(
            session, source=source, selection={"record_path": "root/row"},
        )
    assert "SECRET-PATIENT-VALUE" not in str(caught.value)
    assert not list(session.glob("*.parquet"))


@pytest.mark.parametrize("payload", [b"", b"not a zip", os.urandom(128)])
def test_malformed_complex_files_fail_without_publishing(tmp_path: Path, payload: bytes) -> None:
    source = tmp_path / "malformed.zip"
    source.write_bytes(payload)
    with pytest.raises((FormatSelectionError, zipfile.BadZipFile)):
        materialize_selected_format(tmp_path, source=source, selection={"member": "x.csv"})
    assert not list(tmp_path.glob("*.parquet"))


@settings(max_examples=20, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(payload=st.binary(min_size=0, max_size=512).filter(
    lambda value: not value.startswith(b"\x1f\x8b")))
def test_malformed_gzip_fuzzing_never_loads(payload: bytes, tmp_path: Path) -> None:
    source = tmp_path / "fuzz.csv.gz"
    source.write_bytes(payload)
    with pytest.raises(Exception):
        schema.load_data(source)


def test_omop_table_recognition_requires_core_shape() -> None:
    recognized = recognize_omop_tables([
        "person.csv", "visit_occurrence.parquet", "measurement.csv", "notes.txt",
    ])
    assert recognized["recognized"] is True
    assert recognized["matched_tables"] == ["measurement", "person", "visit_occurrence"]


def test_format_registry_is_cross_platform_and_domain_gated(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    monkeypatch.delenv("SIFT_DOMAIN_CAPABILITIES", raising=False)
    rows = {row["id"]: row for row in format_capabilities()}
    assert rows["avro"]["ready"] is True
    assert rows["hdf5"]["ready"] is True
    geospatial_installed = all(
        importlib.util.find_spec(name) is not None
        for name in ("geopandas", "pyogrio", "shapely")
    )
    assert rows["geopackage"]["installed"] is geospatial_installed
    assert rows["geopackage"]["ready"] is geospatial_installed
    assert rows["vcf"]["installed"] is True
    assert rows["bcf"]["installed"] is (
        importlib.util.find_spec("pysam") is not None
    )
    assert rows["vcf"]["enabled"] is False
    assert rows["bcf"]["enabled"] is False
    assert rows["dicom"]["enabled"] is False


def test_worker_environment_preserves_windows_appcontainer_location(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CreateProcessW needs LOCALAPPDATA for an AppContainer child, while
    unrelated user/profile and credential variables must remain scrubbed."""
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\Researcher\AppData\Local")
    monkeypatch.setenv("USERPROFILE", r"C:\Users\Researcher")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "not-a-real-secret")
    environment = format_selection_module._worker_environment()
    assert environment["LOCALAPPDATA"] == r"C:\Users\Researcher\AppData\Local"
    assert "USERPROFILE" not in environment
    assert "AWS_SECRET_ACCESS_KEY" not in environment


def test_vcf_stdlib_parser_validates_required_fields(tmp_path: Path) -> None:
    valid = tmp_path / "variants.vcf"
    valid.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tSAMPLE\n"
        "chr1\t11\t.\tA\tC,G\t.\tq10;s50\tDP=4\t0/1\n",
        encoding="utf-8",
    )
    frame = format_selection_module._vcf_frame(valid)
    assert frame.to_dict(orient="records") == [{
        "chrom": "chr1", "pos": 11, "id": None, "ref": "A",
        "alt": "C,G", "qual": None, "filter": "q10,s50",
    }]

    malformed = tmp_path / "malformed.vcf"
    malformed.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\tzero\t.\tA\tC\t50\tPASS\t.\n",
        encoding="utf-8",
    )
    with pytest.raises(FormatSelectionError, match="invalid POS"):
        format_selection_module._vcf_frame(malformed)


def test_recognized_complex_formats_require_materialization_in_schema(tmp_path: Path) -> None:
    path = tmp_path / "many.h5"
    path.write_bytes(b"not hdf5")
    with pytest.raises(schema.SchemaExtractError, match="explicit"):
        schema.load_data(path)
    with pytest.raises(schema.SchemaExtractError, match="explicit"):
        schema.extract(path, "names_only")
    assert schema.row_count(path) is None


@requires_live_parser_backend
def test_format_selection_ui_bridge_never_accepts_an_outside_path(tmp_path: Path) -> None:
    from sift.ui import SiftBridge

    session = tmp_path / "session"; session.mkdir()
    archive = session / "tables.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("data.csv", "id\n1\n")
    bridge = SiftBridge(); bridge.cwd = session
    inspected = bridge.inspect_data_container("tables.zip")
    assert inspected["ok"] is True
    assert inspected["explicit_selection_required"] is True
    materialized = bridge.materialize_data_format_selection(
        "tables.zip", {"member": "data.csv"}, "selected.parquet",
    )
    assert materialized["ok"] is True
    assert (session / "selected.parquet").is_file()
    outside = bridge.inspect_data_container("../tables.zip")
    assert outside["ok"] is False
