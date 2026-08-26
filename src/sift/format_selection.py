"""Isolated, explicit-selection materialization for complex data formats.

Ordinary flat tables are handled by :mod:`sift.schema`. Containers and domain
formats enter here: the researcher names the member/dataset/variable/layer,
an offline child process parses it under resource limits, and the result is an
ordinary Parquet table plus a bounded metadata sidecar. No parser runs in the
desktop process and no parser can use the network.
"""

from __future__ import annotations

import gzip
import importlib.metadata
import importlib.util
import json
import math
import os
import pickletools
import re
import secrets
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from sift.text_safety import safe_text


class FormatSelectionError(Exception):
    pass


MAX_CONTAINER_OBJECTS = 10_000
MAX_ARCHIVE_MEMBERS = 10_000
MAX_ARCHIVE_EXPANDED_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_RATIO = 200
MAX_RASTER_PIXELS = 100_000_000
MAX_MEDICAL_FILES = 10_000
PARSER_TIMEOUT_SECONDS = 300
PARSER_MEMORY_LIMIT_BYTES = 4 * 1024 * 1024 * 1024
PARSER_PROCESS_LIMIT = 32
_SELECTION_KEY = re.compile(r"^[A-Za-z0-9_./:@+ -]{1,1024}$")
_UNSAFE_EXTENSIONS = {".pkl", ".pickle", ".joblib", ".cloudpickle", ".dill"}


@dataclass(frozen=True)
class FormatCapability:
    id: str
    extensions: tuple[str, ...]
    domain: str
    flag: str | None
    dependencies: tuple[str, ...]
    selection: str

    def as_dict(self) -> dict[str, Any]:
        ready = all(importlib.util.find_spec(name) is not None for name in self.dependencies)
        enabled = self.flag is None or self.flag in _enabled_flags()
        return {**asdict(self), "installed": ready, "enabled": enabled,
                "ready": ready and enabled}


FORMAT_CAPABILITIES: tuple[FormatCapability, ...] = (
    FormatCapability("compressed_tables", (".csv.gz", ".tsv.gz", ".jsonl.gz", ".ndjson.gz"), "general", None, (), "none"),
    FormatCapability("zip", (".zip",), "general", None, (), "member"),
    FormatCapability("avro", (".avro",), "general", None, ("fastavro",), "none"),
    FormatCapability("xml_table", (".xml",), "general", None, ("defusedxml",), "record_path"),
    FormatCapability("dbf", (".dbf",), "general", None, ("dbfread",), "none"),
    FormatCapability("r_workspace", (".rda", ".rdata"), "general", None, ("pyreadr",), "r_object"),
    FormatCapability("hdf5", (".h5", ".hdf5"), "scientific", None, ("h5py",), "dataset"),
    FormatCapability("netcdf", (".nc", ".netcdf"), "scientific", None, ("xarray", "netCDF4"), "variable"),
    FormatCapability("matlab", (".mat",), "scientific", None, ("scipy",), "variable"),
    FormatCapability("fits", (".fits", ".fit", ".fts"), "astronomy", "astronomy", ("astropy",), "hdu"),
    FormatCapability("geojson", (".geojson",), "geospatial", None, ("shapely",), "none"),
    FormatCapability("geopackage", (".gpkg",), "geospatial", None, ("geopandas", "pyogrio", "shapely"), "layer"),
    FormatCapability("shapefile", (".shp",), "geospatial", None, ("geopandas", "pyogrio", "shapely"), "none"),
    FormatCapability("raster", (".tif", ".tiff", ".vrt"), "geospatial", None, ("rasterio", "numpy"), "bounded_summary"),
    # Text VCF has a small, auditable stdlib parser so it works identically on
    # Windows, macOS, and Linux.  BCF remains a separate capability because
    # its htslib-backed parser does not publish Windows wheels; keeping these
    # rows separate prevents one unavailable binary parser from hiding VCF.
    FormatCapability("vcf", (".vcf", ".vcf.gz"), "genomics", "genomics", (), "none"),
    FormatCapability("bcf", (".bcf",), "genomics", "genomics", ("pysam",), "none"),
    FormatCapability("plink", (".bed",), "genomics", "genomics", ("numpy",), "dataset_prefix"),
    FormatCapability("nifti", (".nii", ".nii.gz"), "neuroimaging", "medical_imaging", ("nibabel", "numpy"), "bounded_summary"),
    FormatCapability("dicom", (".dcm",), "medical", "medical_imaging", ("pydicom",), "metadata_only"),
    FormatCapability("fhir", (".fhir",), "clinical", "clinical", (), "bundle"),
)


def _enabled_flags() -> frozenset[str]:
    return frozenset(
        value.strip().casefold() for value in
        os.environ.get("SIFT_DOMAIN_CAPABILITIES", "").split(",") if value.strip()
    )


def format_capabilities() -> list[dict[str, Any]]:
    return [row.as_dict() for row in FORMAT_CAPABILITIES]


def _parser_provenance(capability: FormatCapability) -> dict[str, Any]:
    packages = []
    for name in capability.dependencies:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages.append({"name": name, "version": version})
    return {
        "adapter": capability.id,
        "adapter_version": 1,
        "packages": packages,
    }


def _capability(path: Path) -> FormatCapability:
    name = path.name.casefold()
    for row in FORMAT_CAPABILITIES:
        if any(name.endswith(extension) for extension in row.extensions):
            return row
    raise FormatSelectionError(f"no complex-format adapter for {path.suffix!r}")


def _format_suffix(path: Path) -> str:
    name = path.name.casefold()
    for compound in (".nii.gz", ".vcf.gz"):
        if name.endswith(compound):
            return compound
    return path.suffix.casefold()


def _require_ready(path: Path) -> FormatCapability:
    row = _capability(path)
    state = row.as_dict()
    if not state["enabled"]:
        raise FormatSelectionError(
            f"{row.domain} format support requires the {row.flag!r} domain capability flag"
        )
    if not state["installed"]:
        missing = [name for name in row.dependencies if importlib.util.find_spec(name) is None]
        raise FormatSelectionError(
            f"{row.id} support is not installed; missing: {', '.join(missing)}"
        )
    return row


def reject_unsafe_serialization(path: Path) -> None:
    suffix = path.suffix.casefold()
    if suffix in _UNSAFE_EXTENSIONS:
        raise FormatSelectionError("executable Python object serialization is never accepted")
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError as e:
        raise FormatSelectionError("data file is unreadable") from e
    if len(head) >= 2 and head[0] == 0x80 and 0 <= head[1] <= 5:
        raise FormatSelectionError("pickle-compatible executable object payload is never accepted")
    # Protocol 0/1 pickles do not carry the modern 0x80 protocol marker. The
    # pickle opcode disassembler is non-executing, so use it only as a bounded
    # signature check and fail closed when a complete STOP-terminated stream
    # is recognized.
    try:
        with path.open("rb") as handle:
            candidate = handle.read(1024 * 1024 + 1)
        if len(candidate) <= 1024 * 1024 and any(
            opcode.name == "STOP" for opcode, _argument, _position
            in pickletools.genops(candidate)
        ):
            raise FormatSelectionError(
                "pickle-compatible executable object payload is never accepted"
            )
    except FormatSelectionError:
        raise
    except (ValueError, EOFError):
        pass


def _selection(value: str | int | None, label: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not _SELECTION_KEY.fullmatch(text) or ".." in PurePosixPath(text).parts:
        raise FormatSelectionError(f"invalid explicit {label} selection")
    return text


def _safe_archive_members(path: Path) -> list[zipfile.ZipInfo]:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
        raise FormatSelectionError("ZIP archive has an invalid member count")
    expanded = 0
    for info in infos:
        member = PurePosixPath(info.filename)
        if member.is_absolute() or ".." in member.parts or "\\" in info.filename:
            raise FormatSelectionError("ZIP archive contains an unsafe member path")
        mode = (info.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise FormatSelectionError("ZIP archive symbolic links are not accepted")
        expanded += int(info.file_size)
        if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
            raise FormatSelectionError("ZIP archive exceeds the expansion limit")
        if info.file_size and info.file_size > max(1, info.compress_size) * MAX_ARCHIVE_RATIO:
            raise FormatSelectionError("ZIP archive contains a suspicious compression ratio")
    return infos


def _list_container_objects_direct(path: Path) -> list[dict[str, Any]]:
    """Return bounded object names only; never values or an account listing."""
    # The trusted parent already resolved and validated this exact source
    # before granting its file ACL to the AppContainer.  Resolving it again
    # inside the child is not only redundant on Windows: ``Path.resolve``
    # asks the kernel to traverse the parent directory, while the least-
    # privilege ACL deliberately grants the child this file and *not* a
    # listing of the confidential directory beside it.  Validate the named
    # object without following reparse points, then retain the absolute path
    # the parent supplied.
    source = Path(os.path.abspath(os.fspath(path)))
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise FormatSelectionError("format source must be a regular local file") from exc
    if (
        stat.S_ISLNK(source_stat.st_mode)
        or not stat.S_ISREG(source_stat.st_mode)
        or (
            int(getattr(source_stat, "st_file_attributes", 0))
            & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        )
    ):
        raise FormatSelectionError("format source must be a regular local file")
    reject_unsafe_serialization(source)
    cap = _require_ready(source)
    suffix = _format_suffix(source)
    rows: list[dict[str, Any]] = []
    if suffix == ".zip":
        rows = [{"id": info.filename, "bytes": int(info.file_size)}
                for info in _safe_archive_members(source) if not info.is_dir()]
    elif suffix in {".h5", ".hdf5"}:
        import h5py
        with h5py.File(source, "r") as handle:
            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    rows.append({"id": name, "shape": list(obj.shape), "dtype": str(obj.dtype)})
            handle.visititems(visit)
    elif suffix == ".mat":
        from scipy.io import whosmat
        rows = [{"id": name, "shape": list(shape), "dtype": kind}
                for name, shape, kind in whosmat(source)]
    elif suffix in {".rda", ".rdata"}:
        import pyreadr
        result = pyreadr.read_r(str(source))
        rows = [
            {"id": str(name), "shape": list(value.shape)}
            for name, value in result.items() if hasattr(value, "columns")
        ]
    elif suffix in {".nc", ".netcdf"}:
        import xarray as xr
        with xr.open_dataset(source, decode_cf=True) as dataset:
            rows = [{"id": name, "dimensions": list(value.dims),
                     "shape": list(value.shape), "units": value.attrs.get("units")}
                    for name, value in dataset.data_vars.items()]
    elif suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits
        with fits.open(source, memmap=True, lazy_load_hdus=True) as hdus:
            rows = [{"id": str(index), "name": hdu.name,
                     "shape": list(getattr(getattr(hdu, "data", None), "shape", ()))}
                    for index, hdu in enumerate(hdus)]
    elif suffix == ".gpkg":
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as connection:
            names = connection.execute(
                "SELECT table_name FROM gpkg_contents WHERE data_type IN ('features','attributes') ORDER BY table_name"
            ).fetchall()
        rows = [{"id": str(name[0])} for name in names]
    else:
        rows = [{"id": source.name, "selection": cap.selection}]
    if len(rows) > MAX_CONTAINER_OBJECTS:
        raise FormatSelectionError("container exposes too many objects")
    return rows


def _worker_environment() -> dict[str, str]:
    environment = {
        key: value for key, value in os.environ.items()
        if key in {"PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP",
                   "LOCALAPPDATA", "SIFT_DOMAIN_CAPABILITIES",
                   "SIFT_FULL_LOAD_MAX_BYTES"}
    }
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    return environment


def _worker_command(mode: str, *paths: Path) -> list[str]:
    """Build a worker command that also functions inside PyInstaller.

    A frozen executable is an application entry point, not a general Python
    interpreter. Passing it ``format_selection.py --worker`` makes the Sift
    UI parser consume the source path as its workspace and reject the worker
    arguments. Route frozen workers through hidden application flags instead;
    source installs continue to use Python's normal ``-m`` execution.
    """
    flags = {
        "list": ("--format-list-worker", "--list-worker"),
        "materialize": ("--format-worker", "--worker"),
    }
    if mode not in flags:
        raise ValueError("unknown isolated format worker mode")
    frozen_flag, source_flag = flags[mode]
    string_paths = [str(path) for path in paths]
    if getattr(sys, "frozen", False):
        return [sys.executable, frozen_flag, *string_paths]
    return [
        sys.executable,
        "-m",
        "sift.format_selection",
        source_flag,
        *string_paths,
    ]


def _trusted_posix_launch_shell() -> Path | None:
    """Return a fixed system shell used only for the parser launch gate."""
    return next(
        (
            candidate
            for candidate in (Path("/bin/bash"), Path("/usr/bin/bash"))
            if candidate.is_file()
        ),
        None,
    )


def _parser_read_paths(source: Path) -> tuple[str, ...]:
    """Return the narrow, read-only host paths needed by a parser.

    The Python installation and Sift package tree are executable inputs.  A
    source is granted as one file, not as its parent directory.  Formats with
    conventional sidecars get only those same-stem files.  This prevents a
    malformed parser input from turning "open this dataset" into permission
    to inspect every other confidential dataset beside it.
    """
    candidates: list[Path] = [
        Path(sys.executable).resolve(),
        Path(sys.prefix).resolve(),
        # A virtual environment may point at a relocatable/standalone base
        # interpreter whose shared library is outside ``sys.prefix``.  On
        # macOS dyld resolves that library beside ``sys.base_prefix`` before
        # Python code starts; omitting the canonical base made the sandboxed
        # worker abort even though the venv itself was allowlisted.
        Path(sys.base_prefix).resolve(),
        Path(__file__).resolve().parents[1],
        source,
    ]
    suffix = _format_suffix(source)
    if suffix == ".shp":
        candidates.extend(
            source.with_suffix(extension)
            for extension in (".shx", ".dbf", ".prj", ".cpg", ".qix", ".sbn", ".sbx")
        )
    elif suffix == ".dbf":
        candidates.extend(
            source.with_suffix(extension)
            for extension in (".cpg", ".dbt", ".fpt")
        )
    elif suffix == ".bed":
        candidates.extend((source.with_suffix(".bim"), source.with_suffix(".fam")))
    # Resolve only paths which exist.  ``strict=True`` also prevents a
    # dangling sidecar symlink from becoming a sandbox grant.
    resolved: list[str] = []
    for candidate in candidates:
        try:
            path = candidate.resolve(strict=True)
        except OSError:
            continue
        value = str(path)
        if value not in resolved:
            resolved.append(value)
    return tuple(resolved)


def _read_worker_json(path: Path, *, max_bytes: int = 1_000_000) -> Any:
    """Read one regular, no-follow worker result under a hard byte cap."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        before = os.lstat(path)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or (
                int(getattr(before, "st_file_attributes", 0))
                & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
            )
        ):
            raise FormatSelectionError(
                "isolated format parser returned malformed metadata"
            )
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise FormatSelectionError("isolated format parser returned malformed metadata") from exc
    try:
        metadata = os.fstat(descriptor)
        before_identity = (int(before.st_dev), int(before.st_ino))
        opened_identity = (int(metadata.st_dev), int(metadata.st_ino))
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > max_bytes
            or (
                before_identity != (0, 0)
                and opened_identity != before_identity
            )
        ):
            raise FormatSelectionError("isolated format parser returned malformed metadata")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            raw = handle.read(max_bytes + 1)
    except OSError as exc:
        raise FormatSelectionError("isolated format parser returned malformed metadata") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(raw) > max_bytes:
        raise FormatSelectionError("isolated format parser returned malformed metadata")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise FormatSelectionError("isolated format parser returned malformed metadata") from exc


def _require_parser_backend(environment: Any, platform_name: str) -> None:
    """Require a present *and live-proven* confinement backend."""
    if platform_name == "darwin":
        if environment.sandbox_exec is None:
            raise FormatSelectionError(
                "isolated format parsing requires sandbox-exec; Sift refuses "
                "to run a parser unsandboxed"
            )
        from sift.env_detect import sandbox_baseline_result
        ok, detail = sandbox_baseline_result()
        if not ok:
            raise FormatSelectionError(
                "isolated format parsing requires a working sandbox-exec "
                f"backend; health check failed: {safe_text(detail, max_len=300)}"
            )
        return
    if platform_name.startswith("linux"):
        if environment.bwrap is None:
            raise FormatSelectionError(
                "isolated format parsing requires bubblewrap; Sift refuses "
                "to run a parser unsandboxed"
            )
        from sift.env_detect import bwrap_baseline_result
        ok, detail = bwrap_baseline_result()
        if not ok:
            raise FormatSelectionError(
                "isolated format parsing requires a working bubblewrap "
                f"backend; health check failed: {safe_text(detail, max_len=300)}"
            )
        return
    if platform_name.startswith("win"):
        if not environment.appcontainer_support:
            raise FormatSelectionError(
                "isolated format parsing requires Windows AppContainer; Sift "
                "refuses to run a parser unsandboxed"
            )
        from sift.env_detect import appcontainer_probe_result
        ok, detail = appcontainer_probe_result()
        if not ok:
            raise FormatSelectionError(
                "isolated format parsing requires a working Windows "
                f"AppContainer backend; health check failed: {safe_text(detail, max_len=300)}"
            )
        return
    raise FormatSelectionError(
        f"isolated format parsing is unavailable on platform {platform_name!r}; "
        "Sift refuses to run a parser unsandboxed"
    )


def _run_parser_worker(
    command: list[str], *, staging: Path, source: Path, timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    """Run one parser under Sift's production confinement controls.

    This deliberately shares the script executor's backend builders, output
    cap, identity-safe POSIX descendant tracker, parent-side RSS/process/CPU/
    disk monitors, and Windows Job Object implementation.  There is no plain
    subprocess fallback on any platform or failure path.
    """
    from sift.executor import (
        _CpuLimitExceeded,
        _DiskReserveExceeded,
        _MemoryLimitExceeded,
        _ProcessLimitExceeded,
        _ResourceMonitorUnavailable,
        _bwrap_argv,
        _communicate_with_memory_guard,
        _disk_reserve_preflight_error,
        _merge_bounded_capture,
        _resource_limited_argv,
        _write_sandbox_profile,
        cached_environment,
        script_file_size_limit_bytes,
        script_min_free_disk_bytes,
    )
    from sift.process_tree import (
        ProcessTreeSnapshotUnavailable,
        attach_posix_descendant_tracker,
        tracked_process_identities,
        terminate_tracked_process_tree,
    )

    staging = Path(staging).resolve(strict=True)
    source = Path(source).resolve(strict=True)
    platform_name = sys.platform
    environment = cached_environment()
    _require_parser_backend(environment, platform_name)
    timeout = max(1.0, min(3600.0, float(timeout_seconds)))
    disk_reserve = script_min_free_disk_bytes()
    disk_error = _disk_reserve_preflight_error(staging, disk_reserve)
    if disk_error is not None:
        raise FormatSelectionError(f"isolated format parser was not started: {disk_error}")

    extra_read_paths = _parser_read_paths(source)
    process_environment = _worker_environment()
    # Create archive scratch in the trusted parent. Windows AppContainer can
    # write files inside a parent-granted directory, but creating a new
    # directory from the low-privilege token can block in the kernel on some
    # supported Windows builds. The child never needs that authority.
    # Astropy appends a literal ``astropy`` directory to each XDG base. Name
    # the directly granted private directory accordingly and point the XDG
    # bases at its parent, so Astropy uses this exact existing directory and
    # never needs to create a child directory from the low-privilege token.
    parser_scratch = staging / "astropy"
    parser_scratch.mkdir(mode=0o700)
    # Several scientific readers (notably Astropy) resolve configuration or
    # cache locations while importing. Point every private-home/temp surface
    # at the exact scratch directory that receives a direct Windows ACL,
    # instead of sibling directories whose inherited ACE propagation varies
    # across supported Windows builds.
    process_environment.update({
        "HOME": str(parser_scratch),
        "XDG_CONFIG_HOME": str(staging),
        "XDG_CACHE_HOME": str(staging),
        "TMPDIR": str(parser_scratch),
        "TMP": str(parser_scratch),
        "TEMP": str(parser_scratch),
    })
    if platform_name.startswith("win"):
        home_drive, home_path = os.path.splitdrive(str(parser_scratch))
        process_environment.update({
            "USERPROFILE": str(parser_scratch),
            "HOMEDRIVE": home_drive,
            "HOMEPATH": home_path,
        })
    marker = secrets.token_urlsafe(24)
    process_environment["SIFT_PROCESS_TREE_MARKER"] = marker

    appcontainer_context: Any = None
    parser_command = list(command)
    if platform_name == "darwin":
        if not environment.sandbox_exec:
            raise FormatSelectionError("macOS sandbox-exec is unavailable")
        profile = _write_sandbox_profile(
            staging, staging, extra_read_paths=extra_read_paths,
        )
        parser_command = [environment.sandbox_exec, "-f", str(profile), *parser_command]
    elif platform_name.startswith("linux"):
        if not environment.bwrap:
            raise FormatSelectionError("Linux bubblewrap is unavailable")
        # The shared builder masks <cwd>/.sift before re-exposing the current
        # run.  Materialize the mount point explicitly for bwrap versions
        # which do not create a missing tmpfs destination.
        (staging / ".sift").mkdir(mode=0o700)
        parser_command = [
            environment.bwrap,
            *_bwrap_argv(
                staging, staging, Path.home(),
                extra_read_paths=extra_read_paths,
            ),
            *parser_command,
        ]
    elif platform_name.startswith("win"):
        from sift.win_appcontainer import AppContainerRun
        appcontainer_context = AppContainerRun(
            parser_command,
            staging,
            parser_scratch,
            process_environment,
            extra_read_paths=extra_read_paths,
            cpu_seconds=max(1, int(math.ceil(timeout))),
            memory_bytes=PARSER_MEMORY_LIMIT_BYTES,
            max_processes=PARSER_PROCESS_LIMIT,
            max_file_size_bytes=script_file_size_limit_bytes(),
            min_free_disk_bytes=disk_reserve,
        )
    else:  # pragma: no cover - rejected by _require_parser_backend
        raise AssertionError("unreachable parser platform")

    if not platform_name.startswith("win"):
        try:
            parser_command = _resource_limited_argv(parser_command, platform_name)
        except RuntimeError as exc:
            raise FormatSelectionError(
                f"isolated format parser was not started: {exc}"
            ) from exc
        # Hold the trusted launcher (and no parser code) until the parent has
        # positively captured its birth identity. Very short parsers can
        # otherwise exit during the first macOS ``ps`` snapshot, producing a
        # false "monitor unavailable" even though their result is complete.
        # The gate is fixed shell code; every path/command remains argv data.
        launch_gate = staging / ".parser-launch-ready"
        shell = _trusted_posix_launch_shell()
        if shell is None:
            raise FormatSelectionError(
                "isolated format parser cannot establish its resource monitor"
            )
        parser_command = [
            str(shell),
            "-c",
            'while [ ! -f "$1" ]; do sleep 0.01; done; shift; exec "$@"',
            "sift-parser-launch-gate",
            str(launch_gate),
            *parser_command,
        ]
    else:
        launch_gate = None

    proc: Any = None
    context_entered = False

    def terminate_posix_tree() -> None:
        if proc is None:
            return
        if terminate_tracked_process_tree(proc):
            return
        # Tracker setup itself can fail before its attribute is attached.
        # The new session still provides a deterministic cleanup boundary for
        # the direct child and every descendant which has not deliberately
        # detached.  Falling back again to proc.kill covers a launch which
        # died before the process group became observable.
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001 - best-effort after group kill
                pass

    try:
        try:
            if appcontainer_context is not None:
                proc = appcontainer_context.__enter__()
                context_entered = True
            else:
                proc = subprocess.Popen(
                    parser_command,
                    cwd=str(staging),
                    env=process_environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                )
                attach_posix_descendant_tracker(
                    proc, marker=("SIFT_PROCESS_TREE_MARKER", marker),
                )
                # Fail closed before releasing the parser if ownership cannot
                # be verified. A few bounded retries tolerate a transient
                # process-table snapshot failure without ever running parser
                # code outside a working monitor.
                monitor_ready = False
                monitor_deadline = time.monotonic() + 1.0
                while time.monotonic() < monitor_deadline:
                    try:
                        identities = tracked_process_identities(proc)
                    except ProcessTreeSnapshotUnavailable:
                        identities = None
                    if identities:
                        monitor_ready = True
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(0.01)
                if not monitor_ready or launch_gate is None:
                    raise FormatSelectionError(
                        "isolated format parser cannot establish its resource monitor"
                    )
                launch_gate.touch(mode=0o600)
        except Exception as exc:
            raise FormatSelectionError(
                "isolated format parser could not start inside confinement"
            ) from exc

        try:
            if platform_name.startswith("win"):
                stdout, stderr = proc.communicate(timeout=timeout)
            else:
                stdout, stderr = _communicate_with_memory_guard(
                    proc,
                    timeout_seconds=int(math.ceil(timeout)),
                    memory_limit_bytes=PARSER_MEMORY_LIMIT_BYTES,
                    process_limit=PARSER_PROCESS_LIMIT,
                    cpu_limit_seconds=timeout,
                    disk_directory=staging,
                    disk_reserve_bytes=disk_reserve,
                )
        except (
            subprocess.TimeoutExpired,
            _MemoryLimitExceeded,
            _ProcessLimitExceeded,
            _CpuLimitExceeded,
            _DiskReserveExceeded,
            _ResourceMonitorUnavailable,
        ) as stopped:
            if platform_name.startswith("win"):
                proc.kill()
            else:
                terminate_posix_tree()
            try:
                tail_out, tail_err = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                tail_out, tail_err = "", ""
            stdout = _merge_bounded_capture(
                getattr(stopped, "_sift_captured_stdout", getattr(stopped, "stdout", None)),
                bool(getattr(stopped, "_sift_stdout_truncated", False)),
                tail_out,
            )
            stderr = _merge_bounded_capture(
                getattr(stopped, "_sift_captured_stderr", getattr(stopped, "stderr", None)),
                bool(getattr(stopped, "_sift_stderr_truncated", False)),
                tail_err,
            )
            if isinstance(stopped, subprocess.TimeoutExpired):
                reason = "timed out"
            elif isinstance(stopped, _ResourceMonitorUnavailable):
                reason = "lost a required resource monitor"
            else:
                reason = "exceeded a parser resource limit"
            # Parser diagnostics are untrusted data.  Native libraries often
            # include a rejected cell value, XML text, DICOM tag value, or
            # filesystem path in stderr.  Even bounded/safe_text-normalized
            # output is not disclosure-safe, so retain it only in the local
            # process object and expose a fixed classification to callers.
            raise FormatSelectionError(f"isolated format parser {reason}") from stopped

        return subprocess.CompletedProcess(
            parser_command, int(proc.returncode), stdout or "", stderr or "",
        )
    finally:
        if proc is not None and not platform_name.startswith("win"):
            terminate_posix_tree()
        if appcontainer_context is not None and context_entered:
            try:
                appcontainer_context.__exit__(None, None, None)
            except Exception as exc:
                raise FormatSelectionError(
                    "isolated format parser AppContainer cleanup failed; "
                    "the confinement state is no longer trusted"
                ) from exc


def list_container_objects(path: Path) -> list[dict[str, Any]]:
    """List object metadata in the same offline parser isolation boundary."""
    original = Path(path).expanduser()
    if original.is_symlink() or not original.is_file():
        raise FormatSelectionError("format source must be a regular local file")
    source = original.resolve(strict=True)
    reject_unsafe_serialization(source)
    _require_ready(source)
    with tempfile.TemporaryDirectory(prefix="sift-format-list-") as folder:
        # macOS exposes the per-user temporary tree through both /var and
        # /private/var.  sandbox-exec matches canonical kernel paths, so use
        # the resolved spelling consistently in the profile *and* worker argv.
        staging = Path(folder).resolve(strict=True)
        output = staging / "objects.json"
        command = _worker_command("list", source, output)
        completed = _run_parser_worker(
            command, staging=staging, source=source, timeout_seconds=60,
        )
        if completed.returncode != 0:
            raise FormatSelectionError("isolated container inspection rejected the input")
        payload = _read_worker_json(output)
        if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
            raise FormatSelectionError("isolated container inspection returned malformed metadata")
        return payload["objects"]


def _frame_from_xml(path: Path, record_path: str) -> tuple[Any, dict[str, Any]]:
    from defusedxml import ElementTree as ET
    import pandas as pd

    wanted = [part for part in _selection(record_path, "XML record path").split("/") if part]
    root = ET.parse(path).getroot()
    nodes = [root]
    if wanted and root.tag.rsplit("}", 1)[-1] == wanted[0]:
        wanted = wanted[1:]
    for component in wanted:
        nodes = [child for node in nodes for child in list(node)
                 if child.tag.rsplit("}", 1)[-1] == component]
    if not nodes:
        raise FormatSelectionError("XML record path selected no records")
    records = []
    for node in nodes:
        row = {f"@{key}": value for key, value in node.attrib.items()}
        for child in list(node):
            key = child.tag.rsplit("}", 1)[-1]
            if list(child):
                raise FormatSelectionError("XML records must be flat at the selected path")
            if key in row:
                raise FormatSelectionError("XML record contains duplicate child names")
            row[key] = child.text
        records.append(row)
    return pd.DataFrame.from_records(records), {"record_path": "/".join(wanted)}


def _geojson_frame(path: Path) -> tuple[Any, dict[str, Any]]:
    import pandas as pd
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("type") != "FeatureCollection":
        raise FormatSelectionError("GeoJSON must be one FeatureCollection")
    features = payload.get("features")
    if not isinstance(features, list) or len(features) > MAX_CONTAINER_OBJECTS * 100:
        raise FormatSelectionError("GeoJSON feature count is invalid or too large")
    rows, invalid = [], 0
    for feature in features:
        if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
            raise FormatSelectionError("GeoJSON contains a malformed feature")
        geometry = feature.get("geometry")
        if geometry is not None:
            if not isinstance(geometry, dict) or not geometry.get("type"):
                invalid += 1
            else:
                try:
                    from shapely.geometry import shape
                    if not shape(geometry).is_valid:
                        invalid += 1
                except (ImportError, ValueError, TypeError):
                    # A malformed coordinate tree is invalid even when Shapely
                    # is absent; valid geometry is conservatively left unknown.
                    try:
                        json.dumps(geometry["coordinates"], allow_nan=False)
                    except (KeyError, TypeError, ValueError):
                        invalid += 1
        row = dict(feature["properties"])
        row["geometry"] = json.dumps(geometry, separators=(",", ":")) if geometry else None
        rows.append(row)
    crs = payload.get("crs") or {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
    return pd.DataFrame.from_records(rows), {"crs": crs, "invalid_geometries": invalid}


def _vcf_frame(path: Path) -> Any:
    """Read the non-sample VCF fields without a platform-specific binary.

    Sift intentionally materializes the same bounded variant-level table the
    previous pysam adapter produced; sample genotype columns are not copied
    into the dataframe.  Strict UTF-8, the required header, numeric position
    and quality fields, and a per-record size bound make malformed input fail
    closed inside the already-confined parser process.
    """
    import pandas as pd

    opener = gzip.open if path.name.casefold().endswith(".vcf.gz") else open
    rows: list[dict[str, Any]] = []
    header_seen = False
    required_header = ("#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO")
    try:
        with opener(path, "rt", encoding="utf-8-sig", errors="strict", newline="") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if len(raw_line) > 64 * 1024 * 1024:
                    raise FormatSelectionError("VCF record exceeds the bounded row workflow")
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith("##"):
                    if header_seen:
                        raise FormatSelectionError("VCF metadata appears after the column header")
                    continue
                if line.startswith("#"):
                    fields = tuple(line.split("\t", 8)[:8])
                    if header_seen or fields != required_header:
                        raise FormatSelectionError("VCF column header is missing or malformed")
                    header_seen = True
                    continue
                if not header_seen:
                    raise FormatSelectionError("VCF column header is missing or malformed")
                fields = line.split("\t", 8)
                if len(fields) < 8:
                    raise FormatSelectionError(f"VCF record {line_number} has fewer than eight fields")
                chrom, position_text, identifier, reference, alternate, quality_text, filters, _info = fields[:8]
                if not chrom or chrom == "." or not reference or reference == ".":
                    raise FormatSelectionError(f"VCF record {line_number} has invalid CHROM or REF")
                try:
                    position = int(position_text)
                except ValueError as e:
                    raise FormatSelectionError(f"VCF record {line_number} has an invalid POS") from e
                if position <= 0:
                    raise FormatSelectionError(f"VCF record {line_number} has an invalid POS")
                if quality_text == ".":
                    quality = None
                else:
                    try:
                        quality = float(quality_text)
                    except ValueError as e:
                        raise FormatSelectionError(f"VCF record {line_number} has an invalid QUAL") from e
                    if not math.isfinite(quality):
                        raise FormatSelectionError(f"VCF record {line_number} has an invalid QUAL")
                rows.append({
                    "chrom": chrom,
                    "pos": position,
                    "id": None if identifier == "." else identifier,
                    "ref": reference,
                    "alt": "" if alternate == "." else alternate,
                    "qual": quality,
                    "filter": ",".join(value for value in filters.split(";") if value != "."),
                })
                if len(rows) > MAX_CONTAINER_OBJECTS * 100:
                    raise FormatSelectionError("variant table exceeds the bounded row workflow")
    except (OSError, UnicodeError, gzip.BadGzipFile) as e:
        raise FormatSelectionError("VCF input is unreadable or malformed") from e
    if not header_seen:
        raise FormatSelectionError("VCF column header is missing or malformed")
    return pd.DataFrame.from_records(
        rows,
        columns=("chrom", "pos", "id", "ref", "alt", "qual", "filter"),
    )


def _parse_to_frame(
    path: Path,
    selection: dict[str, Any],
    *,
    scratch_dir: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    import pandas as pd

    suffix = _format_suffix(path)
    if suffix == ".zip":
        cap = _require_ready(path)
        metadata: dict[str, Any] = {
            "format": cap.id, "parser_pid": os.getpid(),
            "source_parser": _parser_provenance(cap),
        }
        member = _selection(selection.get("member"), "ZIP member")
        infos = {info.filename: info for info in _safe_archive_members(path)}
        if member not in infos or infos[member].is_dir():
            raise FormatSelectionError("selected ZIP member was not found")
        # The Windows AppContainer has create/write rights in its private
        # staging tree but deliberately has no DELETE right over that host
        # tree.  A child-side TemporaryDirectory therefore parses correctly
        # and then fails while trying to remove its own scratch directory.
        # Leave this bounded scratch subtree for the trusted parent to remove
        # after the AppContainer has exited and its ACL has been reverted.
        if scratch_dir is None:
            folder = Path(tempfile.mkdtemp(prefix="sift-archive-"))
        else:
            folder = Path(scratch_dir)
            try:
                scratch_stat = folder.lstat()
            except OSError as exc:
                raise FormatSelectionError("archive scratch directory is unavailable") from exc
            if (
                not stat.S_ISDIR(scratch_stat.st_mode)
                or stat.S_ISLNK(scratch_stat.st_mode)
                or (
                    int(getattr(scratch_stat, "st_file_attributes", 0))
                    & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
                )
            ):
                raise FormatSelectionError("archive scratch directory is unsafe")
        with zipfile.ZipFile(path) as archive:
            safe_member_name = re.sub(
                r"[^A-Za-z0-9._-]",
                "_",
                Path(member).name,
            )[-180:] or "member"
            target = folder / f"{secrets.token_hex(16)}-{safe_member_name}"
            with archive.open(infos[member]) as source, target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
            nested = {key: value for key, value in selection.items() if key != "member"}
            frame, child = _parse_to_frame(
                target,
                nested,
                scratch_dir=scratch_dir,
            )
            return frame, {**child, "archive_member": member, "parser_pid": os.getpid()}
    if suffix in {".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".gz",
                  ".dta", ".rds", ".rda", ".rdata", ".parquet", ".feather",
                  ".arrow", ".ipc", ".orc", ".sav", ".zsav", ".por",
                  ".sas7bdat", ".xpt", ".xlsx", ".xls", ".ods"}:
        from sift.schema import load_data
        direct_metadata: dict[str, Any] = {
            "format": suffix.removeprefix("."), "parser_pid": os.getpid(),
        }
        if suffix in {".rda", ".rdata"}:
            direct_metadata["source_parser"] = _parser_provenance(
                _require_ready(path),
            )
        return load_data(
            path, sheet=selection.get("sheet"),
            r_object=selection.get("r_object"),
        ), direct_metadata
    cap = _require_ready(path)
    metadata = {
        "format": cap.id, "parser_pid": os.getpid(),
        "source_parser": _parser_provenance(cap),
    }
    if suffix == ".avro":
        from fastavro import reader
        with path.open("rb") as handle:
            rows = list(reader(handle))
        return pd.DataFrame.from_records(rows), metadata
    if suffix == ".xml":
        frame, extra = _frame_from_xml(path, str(selection.get("record_path") or ""))
        return frame, {**metadata, **extra}
    if suffix == ".dbf":
        from dbfread import DBF
        # The default ``ignorecase=True`` scans the entire parent directory to
        # rediscover the already-exact filename.  Parser confinement grants
        # only this dataset and its explicit sidecars, so avoid both that
        # unnecessary directory disclosure and the resulting sandbox denial.
        return pd.DataFrame(iter(DBF(
            path,
            load=False,
            ignorecase=False,
            char_decode_errors="strict",
        ))), metadata
    if suffix in {".h5", ".hdf5"}:
        import h5py
        import numpy as np
        dataset_name = _selection(selection.get("dataset"), "HDF5 dataset")
        with h5py.File(path, "r") as handle:
            if dataset_name not in handle or not isinstance(handle[dataset_name], h5py.Dataset):
                raise FormatSelectionError("selected HDF5 dataset was not found")
            dataset = handle[dataset_name]
            if dataset.ndim > 2:
                raise FormatSelectionError("HDF5 table selection must be one- or two-dimensional")
            values = dataset[...]
            attrs = {str(k): safe_text(str(v), max_len=500) for k, v in dataset.attrs.items()}
        frame = pd.DataFrame(values if values.ndim == 2 else {dataset_name: np.asarray(values)})
        return frame, {**metadata, "dataset": dataset_name, "shape": list(values.shape), "attributes": attrs}
    if suffix in {".nc", ".netcdf"}:
        import xarray as xr
        variable = _selection(selection.get("variable"), "NetCDF variable")
        with xr.open_dataset(path, decode_cf=True) as dataset:
            if variable not in dataset.data_vars:
                raise FormatSelectionError("selected NetCDF variable was not found")
            value = dataset[variable]
            frame = value.to_dataframe(name=variable).reset_index()
            extra = {"variable": variable, "dimensions": list(value.dims),
                     "coordinates": list(value.coords),
                     "attributes": {str(k): safe_text(str(v), max_len=500) for k, v in value.attrs.items()},
                     "units": safe_text(str(value.attrs.get("units", "")), max_len=200)}
        return frame, {**metadata, **extra}
    if suffix == ".mat":
        import numpy as np
        from scipy.io import loadmat
        variable = _selection(selection.get("variable"), "MAT variable")
        values = loadmat(path, variable_names=[variable], squeeze_me=False,
                         struct_as_record=True)
        if variable not in values:
            raise FormatSelectionError("selected MAT variable was not found")
        array = np.asarray(values[variable])
        if array.ndim > 2 or array.dtype.kind == "O":
            raise FormatSelectionError("selected MAT variable is not a plain one- or two-dimensional array")
        frame = pd.DataFrame(array if array.ndim == 2 else {variable: array.reshape(-1)})
        return frame, {**metadata, "variable": variable, "shape": list(array.shape), "dtype": str(array.dtype)}
    if suffix in {".fits", ".fit", ".fts"}:
        from astropy.io import fits
        hdu_index = int(_selection(selection.get("hdu"), "FITS HDU"))
        with fits.open(path, memmap=True) as hdus:
            if hdu_index < 0 or hdu_index >= len(hdus):
                raise FormatSelectionError("selected FITS HDU was not found")
            hdu = hdus[hdu_index]
            data = hdu.data
            if data is None:
                raise FormatSelectionError("selected FITS HDU contains no data")
            import numpy as np
            data = np.asarray(data).astype(data.dtype.newbyteorder("="), copy=False)
            if getattr(data.dtype, "names", None):
                frame = pd.DataFrame.from_records(data)
            elif data.ndim <= 2:
                frame = pd.DataFrame(data)
            else:
                raise FormatSelectionError("selected FITS image exceeds two dimensions")
            header = {str(k): safe_text(str(v), max_len=300) for k, v in list(hdu.header.items())[:500]}
        return frame, {**metadata, "hdu": hdu_index, "header": header}
    if suffix == ".geojson":
        frame, extra = _geojson_frame(path)
        return frame, {**metadata, **extra}
    if suffix in {".gpkg", ".shp"}:
        import geopandas as gpd
        if suffix == ".shp":
            missing = [ext for ext in (".shx", ".dbf") if not path.with_suffix(ext).is_file()]
            if missing:
                raise FormatSelectionError(f"Shapefile is missing required sidecars: {', '.join(missing)}")
            layer = None
        else:
            layer = _selection(selection.get("layer"), "GeoPackage layer")
        geo = gpd.read_file(path, layer=layer)
        invalid = int((~geo.geometry.is_valid & geo.geometry.notna()).sum())
        crs = geo.crs.to_string() if geo.crs else None
        frame = pd.DataFrame(geo.drop(columns=[geo.geometry.name]))
        frame["geometry"] = geo.geometry.to_wkt()
        return frame, {**metadata, "layer": layer, "crs": crs, "invalid_geometries": invalid}
    if suffix in {".tif", ".tiff", ".vrt"}:
        import rasterio
        with rasterio.open(path) as raster:
            pixels = raster.width * raster.height * raster.count
            if pixels > MAX_RASTER_PIXELS:
                raise FormatSelectionError("raster exceeds the bounded pixel workflow")
            rows = []
            for band in range(1, raster.count + 1):
                values = raster.read(band, masked=True)
                rows.append({"band": band, "count": int(values.count()),
                             "min": float(values.min()), "max": float(values.max()),
                             "mean": float(values.mean()), "std": float(values.std())})
            extra = {"crs": str(raster.crs) if raster.crs else None,
                     "transform": list(raster.transform), "width": raster.width,
                     "height": raster.height, "bands": raster.count}
        return pd.DataFrame.from_records(rows), {**metadata, **extra}
    if suffix in {".vcf", ".vcf.gz"}:
        return _vcf_frame(path), metadata
    if suffix == ".bcf":
        import pysam
        rows = []
        with pysam.VariantFile(path) as variant_file:
            for record in variant_file:
                alternate_alleles = tuple(str(value) for value in (record.alts or ()))
                rows.append({"chrom": record.chrom, "pos": record.pos, "id": record.id,
                             "ref": record.ref, "alt": ",".join(alternate_alleles),
                             "qual": record.qual,
                             "filter": ",".join(str(value) for value in record.filter.keys())})
                if len(rows) > MAX_CONTAINER_OBJECTS * 100:
                    raise FormatSelectionError("variant table exceeds the bounded row workflow")
        return pd.DataFrame.from_records(rows), metadata
    if suffix == ".bed":
        import numpy as np
        bim, fam = path.with_suffix(".bim"), path.with_suffix(".fam")
        missing = [candidate.suffix for candidate in (bim, fam) if not candidate.is_file()]
        if missing:
            raise FormatSelectionError(f"PLINK BED is missing required sidecars: {', '.join(missing)}")
        fam_rows = [line.split() for line in fam.read_text(encoding="utf-8").splitlines() if line.strip()]
        bim_rows = [line.split() for line in bim.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(len(row) < 2 for row in fam_rows) or any(len(row) < 6 for row in bim_rows):
            raise FormatSelectionError("PLINK sidecar rows are malformed")
        samples, variant_count = len(fam_rows), len(bim_rows)
        if samples == 0 or variant_count == 0 or samples * variant_count > 10_000_000:
            raise FormatSelectionError("PLINK genotype matrix exceeds the bounded workflow")
        raw = path.read_bytes()
        if raw[:3] != b"\x6c\x1b\x01":
            raise FormatSelectionError("PLINK BED must use SNP-major binary encoding")
        bytes_per_variant = (samples + 3) // 4
        if len(raw) != 3 + variant_count * bytes_per_variant:
            raise FormatSelectionError("PLINK BED size does not match BIM/FAM dimensions")
        matrix: Any = np.empty((samples, variant_count), dtype=np.float32)
        mapping = {0: 0.0, 1: np.nan, 2: 1.0, 3: 2.0}
        for variant in range(variant_count):
            block = raw[3 + variant * bytes_per_variant:3 + (variant + 1) * bytes_per_variant]
            for sample in range(samples):
                code = (block[sample // 4] >> ((sample % 4) * 2)) & 0b11
                matrix[sample, variant] = mapping[code]
        frame = pd.DataFrame(matrix, columns=[row[1] for row in bim_rows])
        frame.insert(0, "sample_id", [row[1] for row in fam_rows])
        return frame, {**metadata, "samples": samples,
                       "variants": variant_count,
                       "sidecars": [bim.name, fam.name]}
    if suffix in {".nii", ".nii.gz"}:
        import nibabel as nib
        import numpy as np
        image: Any = nib.load(path)
        if math.prod(image.shape) > MAX_RASTER_PIXELS:
            raise FormatSelectionError("NIfTI image exceeds the bounded voxel workflow")
        data = image.get_fdata(dtype=np.float32, caching="unchanged")
        volumes = data.reshape((-1, data.shape[-1])) if data.ndim == 4 else data.reshape((-1, 1))
        frame = pd.DataFrame({"volume": range(volumes.shape[1]),
                              "mean": volumes.mean(axis=0), "std": volumes.std(axis=0),
                              "min": volumes.min(axis=0), "max": volumes.max(axis=0)})
        return frame, {**metadata, "shape": list(image.shape),
                       "zooms": list(image.header.get_zooms()),
                       "affine": image.affine.tolist()}
    if suffix == ".dcm":
        import pydicom
        # Never load bulk pixels merely to inspect a research file. Because
        # that means we intentionally cannot prove whether pixels exist, an
        # explicit BurnedInAnnotation=NO is required below for every DICOM
        # object before Sift will make a positive de-identification claim.
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        identifying_keywords = {
            "AccessionNumber", "AdditionalPatientHistory", "InstitutionAddress",
            "InstitutionName", "InstitutionalDepartmentName", "MedicalRecordLocator",
            "NameOfPhysiciansReadingStudy", "OperatorsName", "OtherPatientIDs",
            "OtherPatientNames", "PatientAddress", "PatientBirthDate",
            "PatientBirthName", "PatientID", "PatientInsurancePlanCodeSequence",
            "PatientMotherBirthName", "PatientName", "PatientTelephoneNumbers",
            "PerformingPhysicianName", "PhysiciansOfRecord", "ReferringPhysicianName",
            "RequestingPhysician", "StudyID",
        }
        phi: set[str] = set()
        private_elements_present = False
        for element in dataset.iterall():
            keyword = str(getattr(element, "keyword", "") or "")
            if bool(getattr(element.tag, "is_private", False)):
                private_elements_present = True
            if keyword in identifying_keywords:
                try:
                    populated = bool(str(element.value).strip())
                except Exception:  # noqa: BLE001 - unreadable PHI fails closed
                    populated = True
                if populated:
                    phi.add(keyword)
        if private_elements_present:
            phi.add("PrivateDataElements")
        burned = str(getattr(dataset, "BurnedInAnnotation", "")).strip().upper()
        identity_removed = str(
            getattr(dataset, "PatientIdentityRemoved", ""),
        ).strip().upper()
        # Absence of known PHI is not positive proof of de-identification.
        # Require both the DICOM de-identification assertion and an explicit
        # statement that no identifying text is burned into pixels. Since
        # pixels are deliberately not inspected, unknown/missing evidence
        # remains unclassified rather than becoming a false certification.
        deidentified = (
            not phi
            and identity_removed == "YES"
            and burned == "NO"
        )
        frame = pd.DataFrame([{"modality": str(getattr(dataset, "Modality", "")),
                               "sop_class_uid": str(getattr(dataset, "SOPClassUID", "")),
                               "rows": getattr(dataset, "Rows", None),
                               "columns": getattr(dataset, "Columns", None),
                               "deidentified": deidentified}])
        return frame, {
            **metadata,
            "deidentification_issues": sorted(phi),
            "patient_identity_removed": identity_removed or "UNKNOWN",
            "burned_in_annotation": burned or "UNKNOWN",
            "pixel_data_status": "NOT_INSPECTED",
            "pixels_read": False,
        }
    if suffix == ".fhir":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("resourceType") != "Bundle":
            raise FormatSelectionError("FHIR input must be a Bundle resource")
        entries = payload.get("entry", [])
        if not isinstance(entries, list) or len(entries) > MAX_CONTAINER_OBJECTS * 100:
            raise FormatSelectionError("FHIR Bundle entry count is invalid or too large")
        resources: list[dict[str, Any]] = []
        for row in entries:
            if not isinstance(row, dict):
                continue
            resource_value = row.get("resource")
            if isinstance(resource_value, dict):
                resources.append(resource_value)
        frame = pd.json_normalize(resources, sep=".")
        return frame, {**metadata, "bundle_type": payload.get("type"),
                       "resource_types": sorted({str(r.get("resourceType")) for r in resources})}
    raise FormatSelectionError(f"format parser is not implemented for {suffix!r}")


def _bounded_json(value: Any, *, max_bytes: int = 1_000_000) -> bytes:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    if len(raw) > max_bytes:
        raise FormatSelectionError("format metadata exceeds the bounded sidecar limit")
    return raw


class _OfflineSocket(socket.socket):
    """Socket type that preserves constructor compatibility but never connects."""

    def connect(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("network disabled in Sift format parser")

    def connect_ex(self, *args: Any, **kwargs: Any) -> Any:
        raise PermissionError("network disabled in Sift format parser")


def _activate_parser_isolation() -> None:
    # Defense in depth: parsers receive no usable network socket even if a
    # third-party library unexpectedly attempts remote resolution.
    setattr(socket, "socket", _OfflineSocket)

    def denied(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("network disabled in Sift format parser")

    socket.create_connection = denied  # type: ignore[assignment]
    socket.getaddrinfo = denied  # type: ignore[assignment]
    try:
        import resource
        memory = 4 * 1024 * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_CPU, (PARSER_TIMEOUT_SECONDS, PARSER_TIMEOUT_SECONDS + 5))
    except (ImportError, OSError, ValueError):
        pass


def _worker(source: Path, selection_path: Path, output: Path, metadata_path: Path) -> None:
    _activate_parser_isolation()
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise FormatSelectionError("format selection must be an object")
    frame, metadata = _parse_to_frame(
        source,
        selection,
        scratch_dir=output.parent / "astropy",
    )
    if len(frame) > 100_000_000 or len(frame.columns) > 100_000:
        raise FormatSelectionError("materialized table exceeds the row or column safety limit")
    frame.to_parquet(output, index=False)
    metadata_path.write_bytes(_bounded_json(metadata))


def materialize_selected_format(
    cwd: Path, *, source: Path, selection: dict[str, Any] | None = None,
    output_name: str | None = None, timeout_seconds: float = PARSER_TIMEOUT_SECONDS,
) -> Path:
    """Parse one selected object offline and atomically publish a Parquet table."""
    cwd = Path(cwd).resolve(strict=True)
    original_source = Path(source).expanduser()
    if original_source.is_symlink() or not original_source.is_file():
        raise FormatSelectionError("format source must be a regular local file")
    source = original_source.resolve(strict=True)
    reject_unsafe_serialization(source)
    cap = _require_ready(source)
    chosen = dict(selection or {})
    required = {
        "member": "member", "record_path": "record_path", "dataset": "dataset",
        "variable": "variable", "hdu": "hdu", "layer": "layer",
        "r_object": "r_object",
    }.get(cap.selection)
    if required and not chosen.get(required):
        raise FormatSelectionError(f"{cap.id} requires explicit {required} selection")
    if len(chosen) > 16:
        raise FormatSelectionError("format selection contains too many fields")
    name = output_name or f"{source.stem}_selected.parquet"
    if Path(name).name != name or not name.casefold().endswith(".parquet"):
        raise FormatSelectionError("materialized output name must be a plain .parquet filename")
    target = cwd / name
    sidecar = target.with_suffix(target.suffix + ".metadata.json")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    reserved: list[Path] = []
    try:
        for candidate in (target, sidecar):
            descriptor = os.open(candidate, flags, 0o600)
            os.close(descriptor)
            reserved.append(candidate)
    except OSError as e:
        for candidate in reserved:
            candidate.unlink(missing_ok=True)
        if isinstance(e, FileExistsError):
            raise FormatSelectionError("materialized output already exists") from e
        raise FormatSelectionError("could not reserve materialized output safely") from e
    succeeded = False
    try:
        with tempfile.TemporaryDirectory(prefix=".sift-format-", dir=cwd) as folder:
            staging = Path(folder)
            selection_path = staging / "selection.json"
            output = staging / "result.parquet"
            metadata_path = staging / "metadata.json"
            selection_path.write_bytes(_bounded_json(chosen, max_bytes=64 * 1024))
            command = _worker_command(
                "materialize", source, selection_path, output, metadata_path,
            )
            completed = _run_parser_worker(
                command,
                staging=staging,
                source=source,
                timeout_seconds=timeout_seconds,
            )
            if completed.returncode != 0:
                raise FormatSelectionError("isolated format parser rejected the input")
            if (
                output.is_symlink()
                or metadata_path.is_symlink()
                or not output.is_file()
                or not metadata_path.is_file()
            ):
                raise FormatSelectionError("isolated format parser produced incomplete output")
            from sift.schema import full_load_max_bytes
            if output.stat().st_size > full_load_max_bytes():
                raise FormatSelectionError("materialized table exceeds the safe analysis-load ceiling")
            import pyarrow.parquet as pq
            pq.ParquetFile(output)
            metadata = _read_worker_json(metadata_path)
            if not isinstance(metadata, dict):
                raise FormatSelectionError("isolated format parser returned malformed metadata")
            if int(metadata.get("parser_pid", os.getpid())) == os.getpid():
                raise FormatSelectionError("complex parser did not execute in isolation")
            os.replace(output, target)
            os.replace(metadata_path, sidecar)
        succeeded = True
    finally:
        if not succeeded:
            target.unlink(missing_ok=True)
            sidecar.unlink(missing_ok=True)
    try:
        from sift.canonical_dataset import (
            create_manifest, snapshot_source_artifact,
        )
        source_artifact = snapshot_source_artifact(cwd, source)
        create_manifest(
            cwd,
            target,
            selection=chosen,
            dataset_kind="derived",
            parents=(source_artifact["fingerprint"],),
            transformations=({
                "operation": "isolated_format_materialization",
                "runtime": cap.id,
                "source_artifact_fingerprint": source_artifact["fingerprint"],
            },),
        )
    except Exception as e:
        target.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise FormatSelectionError(
            f"could not establish canonical materialization identity: {type(e).__name__}"
        ) from e
    from sift import release_ledger
    if not release_ledger.record_release(
        cwd, kind="local_ingestion", tool="(isolated format selection)",
        extra={"dataset": target.name, "source": source.name,
               "source_format": cap.id, "selection": chosen,
               "metadata_sidecar": sidecar.name},
    ):
        target.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise FormatSelectionError("could not record format materialization provenance")
    return target


OMOP_CORE_TABLES = frozenset({
    "person", "observation_period", "visit_occurrence", "condition_occurrence",
    "drug_exposure", "procedure_occurrence", "measurement", "observation",
    "death", "note", "specimen", "device_exposure", "location", "care_site",
    "provider", "payer_plan_period", "cost", "concept", "concept_relationship",
})


def recognize_omop_tables(names: list[str]) -> dict[str, Any]:
    normalized = {Path(name).stem.casefold() for name in names[:10_000]}
    matched = sorted(normalized & OMOP_CORE_TABLES)
    return {"recognized": "person" in matched and len(matched) >= 3,
            "matched_tables": matched, "coverage": len(matched) / len(OMOP_CORE_TABLES)}


def format_runtime_self_check() -> dict[str, Any]:
    """Exercise the real confined parser process using synthetic local data."""
    try:
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory(prefix="sift-format-check-") as folder:
            session = Path(folder).resolve(strict=True)
            source = session / "synthetic.xml"
            source.write_text(
                "<records><row><id>1</id><value>2.5</value></row>"
                "<row><id>2</id><value>3.5</value></row></records>",
                encoding="utf-8",
            )
            output = materialize_selected_format(
                session,
                source=source,
                selection={"record_path": "row"},
                output_name="selected.parquet",
                timeout_seconds=60,
            )
            table = parquet.read_table(output)
            ok = table.num_rows == 2 and table.num_columns == 2
    except Exception as exc:  # noqa: BLE001 - bounded release evidence
        return {
            "schema_version": 1,
            "ok": False,
            "check": "confined_complex_format_worker",
            # Never leak a host path from a confinement/SDK exception.
            "detail": type(exc).__name__,
        }
    return {
        "schema_version": 1,
        "ok": ok,
        "check": "confined_complex_format_worker",
        "detail": "synthetic XML materialization passed" if ok else "shape mismatch",
    }


def worker_main(argv: list[str] | None = None) -> int:
    """Run one child-side parser operation after parent confinement setup."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        # The trusted parent sets a random marker before creating its process
        # monitor and releases the child only after that monitor is attached.
        # Hidden flags invoked directly must never become an unsandboxed parser
        # backdoor.
        if not os.environ.get("SIFT_PROCESS_TREE_MARKER"):
            raise FormatSelectionError("isolated parser parent marker is absent")
        if len(arguments) == 3 and arguments[0] == "--list-worker":
            _activate_parser_isolation()
            rows = _list_container_objects_direct(Path(arguments[1]))
            Path(arguments[2]).write_bytes(_bounded_json(
                {"objects": rows, "parser_pid": os.getpid()},
            ))
        elif len(arguments) == 5 and arguments[0] == "--worker":
            _worker(
                Path(arguments[1]), Path(arguments[2]),
                Path(arguments[3]), Path(arguments[4]),
            )
        else:
            raise FormatSelectionError("invalid isolated parser invocation")
    except Exception as e:  # noqa: BLE001 - bounded worker error only
        message = str(e) if isinstance(e, FormatSelectionError) else "parser rejected malformed input"
        print(f"{type(e).__name__}: {safe_text(message, max_len=300)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by parent tests
    raise SystemExit(worker_main())
