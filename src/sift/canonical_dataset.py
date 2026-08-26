"""Canonical, session-private dataset semantics and content identity.

This module is the single local contract between ingestion and trusted
analysis surfaces.  A source file is not identified by its filename: it is
identified by source bytes, an explicit object/worksheet selection, parser
versions, and a canonical schema.  The durable manifest contains metadata
only; observation values never enter it.

The on-disk store is deliberately rooted below ``<session>/.sift/datasets``.
Nothing is shared between session directories, even when two sessions open
the same absolute source path.  Writes are locked and atomic, source
snapshots are content addressed and read-only, and cached tables are reused
only after both their manifest and content hash verify.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sift.config import ensure_private_sift_dir
from sift.file_lock import exclusive_file_lock


MANIFEST_VERSION = 1
MAX_COLLECTION_MEMBERS = 10_000
MAX_METADATA_BYTES = 2 * 1024 * 1024
_STORE_DIR = "datasets"
_SAFE_METADATA_SCALARS = frozenset({
    "format", "dataset", "variable", "layer", "archive_member", "hdu",
    "record_path", "bundle_type", "crs", "units", "width", "height",
    "bands", "samples", "variants", "pixels_read",
    "extraction_scope", "query_sha256", "backend", "row_limit",
    "source_kind", "remote_version",
})
_SAFE_METADATA_LISTS = frozenset({
    "shape", "dimensions", "coordinates", "sidecars", "resource_types",
    "deidentification_issues", "zooms",
})
_ID_NAME = re.compile(
    r"(^|_)(id|identifier|uuid|guid|mrn|subject|participant|patient|person)($|_)",
    re.IGNORECASE,
)
_WEIGHT_NAME = re.compile(
    r"(^|_)(weight|weights|wt|pweight|fweight|aweight|sampling_weight)($|_)",
    re.IGNORECASE,
)
_TIME_NAME = re.compile(
    r"(^|_)(date|datetime|timestamp|time|when|dt|year|month|quarter|wave|visit|period)($|_)",
    re.IGNORECASE,
)
_GEO_NAME = re.compile(
    r"(^|_)(lat|latitude|lon|lng|longitude|geometry|geom|wkt|crs|postal|postcode|zip)($|_)",
    re.IGNORECASE,
)


class CanonicalDatasetError(ValueError):
    """A safe, Sift-authored canonicalization failure."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    raw = _canonical_json(value)
    if len(raw) > MAX_METADATA_BYTES:
        raise CanonicalDatasetError("canonical dataset manifest exceeds its metadata limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".canonical-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _store_root(cwd: Path) -> Path:
    root = Path(cwd).resolve(strict=True)
    store = ensure_private_sift_dir(root) / _STORE_DIR
    store.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(store, 0o700)
    except OSError:
        pass
    return store


def _resolve_source(cwd: Path, source: Path) -> tuple[Path, str]:
    root = Path(cwd).resolve(strict=True)
    candidate = Path(source).expanduser()
    if candidate.is_symlink():
        raise CanonicalDatasetError("canonical dataset sources may not be symbolic links")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise CanonicalDatasetError("canonical dataset source must be a regular file")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise CanonicalDatasetError("canonical dataset source must remain inside the session") from exc
    return resolved, relative


def _logical_suffix(path: Path) -> str:
    name = path.name.casefold()
    for suffix in (".jsonl.gz", ".ndjson.gz", ".csv.gz", ".tsv.gz", ".nii.gz", ".vcf.gz"):
        if name.endswith(suffix):
            return suffix
    return path.suffix.casefold()


def _source_components(path: Path) -> list[Path]:
    """Return files whose bytes jointly define one source dataset."""
    suffix = _logical_suffix(path)
    rows = [path]
    if suffix == ".shp":
        rows.extend(candidate for candidate in (
            path.with_suffix(".shx"), path.with_suffix(".dbf"),
            path.with_suffix(".prj"), path.with_suffix(".cpg"),
        ) if candidate.is_file() and not candidate.is_symlink())
    elif suffix == ".bed":
        rows.extend(candidate for candidate in (
            path.with_suffix(".bim"), path.with_suffix(".fam"),
        ) if candidate.is_file() and not candidate.is_symlink())
    return rows


def _source_hashes(path: Path) -> tuple[str, list[dict[str, Any]]]:
    components = []
    combined = hashlib.sha256()
    for candidate in _source_components(path):
        before = candidate.stat()
        digest = _sha256_file(candidate)
        after = candidate.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns,
        ):
            raise CanonicalDatasetError("source changed while its content hash was being computed")
        size = after.st_size
        row = {"name": candidate.name, "sha256": digest, "size_bytes": size}
        components.append(row)
        combined.update(_canonical_json(row))
    # A single-file dataset's source hash is exactly the familiar file hash.
    # Bundles (Shapefile/PLINK) use a deterministic hash of every named
    # component while retaining each individual file hash in ``components``.
    source_sha256 = (
        str(components[0]["sha256"])
        if len(components) == 1 else combined.hexdigest()
    )
    return source_sha256, components


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _parser_identity(path: Path) -> dict[str, Any]:
    suffix = _logical_suffix(path)
    if suffix in {".dta", ".sav", ".zsav", ".por", ".sas7bdat", ".xpt"}:
        package = "pyreadstat"
    elif suffix in {".rds", ".rda", ".rdata"}:
        package = "pyreadr"
    elif suffix in {".parquet", ".feather", ".arrow", ".ipc", ".orc"}:
        package = "pyarrow"
    elif suffix in {".xlsx"}:
        package = "openpyxl"
    elif suffix == ".xls":
        package = "xlrd"
    elif suffix == ".ods":
        package = "odfpy"
    else:
        package = "pandas"
    return {
        "adapter": f"sift.schema:{suffix}",
        "adapter_version": MANIFEST_VERSION,
        "package": package,
        "package_version": _package_version(package),
    }


def _normalized_names(names: Sequence[Any]) -> list[str]:
    output: list[str] = []
    used: set[str] = set()
    for index, raw in enumerate(names):
        value = unicodedata.normalize("NFKC", str(raw)).strip()
        value = re.sub(r"\s+", "_", value)
        value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
        if not value:
            value = f"column_{index + 1}"
        if value[0].isdigit():
            value = "column_" + value
        base = value
        counter = 2
        while value.casefold() in used:
            value = f"{base}__{counter}"
            counter += 1
        used.add(value.casefold())
        output.append(value)
    return output


def _logical_type(series: Any, name: str = "") -> tuple[str, float]:
    import pandas as pd

    dtype = series.dtype
    if isinstance(dtype, pd.CategoricalDtype):
        return "categorical", 1.0
    if pd.api.types.is_bool_dtype(dtype):
        return "boolean", 1.0
    if pd.api.types.is_integer_dtype(dtype):
        return "integer", 1.0
    if pd.api.types.is_float_dtype(dtype):
        return "number", 1.0
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime", 1.0
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "duration", 1.0
    # Pandas 3 reads text columns as ``StringDtype`` by default.  A named
    # time column still needs the bounded value probe below; returning here
    # would make the same CSV classify as datetime under pandas 2 and string
    # under pandas 3.
    if (pd.api.types.is_string_dtype(dtype)
            and not pd.api.types.is_object_dtype(dtype)
            and not _TIME_NAME.search(name)):
        return "string", 1.0
    sample = series.dropna().head(256)
    if sample.empty:
        return "unknown", 0.0
    kinds = {type(value) for value in sample}
    if all(isinstance(value, Decimal) for value in sample):
        return "decimal", 0.98
    if all(isinstance(value, (bytes, bytearray, memoryview)) for value in sample):
        return "binary", 0.98
    if all(isinstance(value, Mapping) for value in sample):
        return "struct", 0.95
    if all(isinstance(value, (list, tuple)) for value in sample):
        return "list", 0.95
    if all(isinstance(value, str) for value in sample):
        if _TIME_NAME.search(name):
            try:
                parsed = pd.to_datetime(sample, errors="coerce", utc=False)
                if float(parsed.notna().mean()) >= 0.9:
                    return "datetime", 0.85
            except (TypeError, ValueError, OverflowError):
                pass
        try:
            distinct = int(series.nunique(dropna=True))
            if len(series) >= 20 and distinct <= min(50, max(2, len(series) // 20)):
                return "categorical", 0.75
        except (TypeError, ValueError):
            pass
        return "string", 0.9
    return ("mixed", 0.5) if len(kinds) > 1 else ("unknown", 0.4)


def _decimal_metadata(series: Any, logical_type: str) -> dict[str, int] | None:
    if logical_type != "decimal":
        return None
    precision = 1
    scale = 0
    for value in series.dropna().head(10_000):
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
        sign, digits, exponent = decimal.as_tuple()
        del sign
        if not isinstance(exponent, int):
            # Decimal NaN/Infinity use a symbolic exponent and cannot have
            # meaningful fixed-point precision metadata.
            continue
        precision = max(precision, len(digits))
        scale = max(scale, max(0, -exponent))
    return {"precision": precision, "scale": scale}


def _role_flags(series: Any, name: str, logical_type: str) -> list[str]:
    roles: list[str] = []
    non_null = int(series.notna().sum())
    unique = 0
    if non_null:
        try:
            unique = int(series.nunique(dropna=True))
        except (TypeError, ValueError):
            unique = 0
    identifier_name = bool(_ID_NAME.search(name))
    if identifier_name:
        roles.append("identifier_like")
    if non_null >= 20 and unique / non_null >= 0.98 and (identifier_name or logical_type in {"string", "integer"}):
        roles.append("identifier")
    if identifier_name and 0 < unique < non_null:
        roles.append("repeated_measures_identifier")
    if _WEIGHT_NAME.search(name) and logical_type in {"integer", "number", "decimal"}:
        try:
            if bool((series.dropna() >= 0).all()):
                roles.append("weight")
        except (TypeError, ValueError):
            pass
    if logical_type == "datetime" or _TIME_NAME.search(name):
        roles.append("time_index")
    if _GEO_NAME.search(name):
        roles.append("geospatial")
    return roles


def _metadata_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not sidecar.is_file() or sidecar.is_symlink() or sidecar.stat().st_size > MAX_METADATA_BYTES:
        return {}
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _bounded_source_metadata(sidecar: Mapping[str, Any]) -> dict[str, Any]:
    """Keep structural metadata, never arbitrary observation-like values."""
    output: dict[str, Any] = {}
    for key in sorted(_SAFE_METADATA_SCALARS):
        value = sidecar.get(key)
        if value is None or isinstance(value, (str, int, float, bool)):
            output[key] = value
    for key in sorted(_SAFE_METADATA_LISTS):
        value = sidecar.get(key)
        if isinstance(value, list) and len(value) <= 10_000 and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        ):
            output[key] = value
    for key in ("attributes", "header"):
        value = sidecar.get(key)
        if isinstance(value, dict):
            output[f"{key}_keys"] = sorted(str(item) for item in value)[:10_000]
    source_parser = sidecar.get("source_parser")
    if isinstance(source_parser, dict):
        packages = source_parser.get("packages")
        if isinstance(packages, list) and len(packages) <= 100:
            clean_packages = []
            for row in packages:
                if isinstance(row, dict) and isinstance(row.get("name"), str):
                    clean_packages.append({
                        "name": row["name"][:200],
                        "version": str(row.get("version"))[:200]
                        if row.get("version") is not None else None,
                    })
            output["source_parser"] = {
                "adapter": str(source_parser.get("adapter", ""))[:200],
                "adapter_version": source_parser.get("adapter_version"),
                "packages": clean_packages,
            }
    return output


def _manifest_parser(path: Path, sidecar: Mapping[str, Any]) -> dict[str, Any]:
    parser = _parser_identity(path)
    source_parser = _bounded_source_metadata(sidecar).get("source_parser")
    if source_parser:
        parser["source_parser"] = source_parser
    return parser


def _readstat_metadata(path: Path) -> dict[str, Any]:
    suffix = _logical_suffix(path)
    if suffix not in {".dta", ".sav", ".zsav", ".por", ".sas7bdat", ".xpt"}:
        return {}
    try:
        import pyreadstat
        reader: Any = {
            ".dta": pyreadstat.read_dta,
            ".sav": pyreadstat.read_sav,
            ".zsav": pyreadstat.read_sav,
            ".por": pyreadstat.read_por,
            ".sas7bdat": pyreadstat.read_sas7bdat,
            ".xpt": pyreadstat.read_xport,
        }[suffix]
        try:
            _frame, meta = reader(str(path), metadataonly=True, user_missing=True)
        except TypeError:
            _frame, meta = reader(str(path), metadataonly=True)
    except Exception:
        return {}
    names = [str(name) for name in (getattr(meta, "column_names", None) or [])]
    labels = getattr(meta, "column_names_to_labels", None) or {}
    if not labels:
        labels = dict(zip(names, getattr(meta, "column_labels", None) or []))
    variable_to_label = getattr(meta, "variable_to_label", None) or {}
    value_label_sets = getattr(meta, "value_labels", None) or {}
    values = {
        str(name): {str(code): str(label) for code, label in value_label_sets.get(set_name, {}).items()}
        for name, set_name in variable_to_label.items() if set_name in value_label_sets
    }
    return {
        "variable_labels": {str(k): str(v) for k, v in labels.items() if v not in (None, "")},
        "value_labels": values,
        "declared_missing_values": getattr(meta, "missing_ranges", None) or {},
        "variable_formats": getattr(meta, "original_variable_types", None) or {},
    }


def _arrow_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if _logical_suffix(path) != ".parquet":
        return {}
    try:
        import pyarrow.parquet as pq
        schema = pq.ParquetFile(path).schema_arrow
    except Exception:
        return {}
    output: dict[str, dict[str, Any]] = {}
    for field in schema:
        row: dict[str, Any] = {}
        if field.metadata:
            decoded = {
                key.decode("utf-8", "replace"): value.decode("utf-8", "replace")
                for key, value in field.metadata.items()
            }
            for wanted in ("unit", "units", "timezone", "crs", "label"):
                if wanted in decoded:
                    row[wanted] = decoded[wanted]
            row["metadata_keys"] = sorted(decoded)
        try:
            import pyarrow as pa
            if pa.types.is_decimal(field.type):
                row["decimal"] = {"precision": field.type.precision, "scale": field.type.scale}
            if pa.types.is_timestamp(field.type) and field.type.tz:
                row["timezone"] = field.type.tz
        except Exception:
            pass
        output[str(field.name)] = row
    return output


def _original_column_names(
    path: Path, frame: Any, selection: Mapping[str, Any] | None,
) -> list[str]:
    """Recover pre-normalization header names where the source preserves them."""
    suffix = _logical_suffix(path)
    try:
        if suffix in {".csv", ".tsv"}:
            from sift.schema import _csv_has_header, text_table_params
            encoding, separator, _decimal = text_table_params(path, suffix)
            if _csv_has_header(path, separator):
                with path.open("r", encoding=encoding, newline="") as handle:
                    names = next(csv.reader(handle, delimiter=separator))
                if len(names) == len(frame.columns):
                    return [str(name) for name in names]
        if suffix in {".csv.gz", ".tsv.gz"}:
            from sift.schema import _gzip_has_header
            separator = "\t" if suffix == ".tsv.gz" else ","
            if _gzip_has_header(path, separator):
                with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
                    names = next(csv.reader(handle, delimiter=separator))
                if len(names) == len(frame.columns):
                    return [str(name) for name in names]
        if suffix in {".xlsx", ".xls", ".ods"}:
            import pandas as pd
            first = pd.read_excel(
                path,
                sheet_name=(selection or {}).get("worksheet", 0),
                header=None,
                nrows=1,
            )
            if len(first.columns) == len(frame.columns) and len(first):
                return [
                    "" if pd.isna(value) else str(value)
                    for value in first.iloc[0].tolist()
                ]
    except Exception:
        pass
    return [str(name) for name in frame.columns]


def _column_manifest(
    frame: Any, path: Path, sidecar: Mapping[str, Any],
    selection: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    readstat = _readstat_metadata(path)
    arrow = _arrow_metadata(path)
    labels = readstat.get("variable_labels", {})
    value_labels = readstat.get("value_labels", {})
    missing_values = readstat.get("declared_missing_values", {})
    source_names = _original_column_names(path, frame, selection)
    normalized = _normalized_names(list(frame.columns))
    original_by_position: dict[int, str] = {}
    raw_renames = (selection or {}).get("column_renames")
    if isinstance(raw_renames, list):
        for item in raw_renames:
            if (
                isinstance(item, dict)
                and isinstance(item.get("position"), int)
                and isinstance(item.get("original"), str)
            ):
                original_by_position[item["position"]] = item["original"]
    rows: list[dict[str, Any]] = []
    global_crs = sidecar.get("crs")
    global_units = sidecar.get("units")
    for index, (raw_name, normal_name) in enumerate(zip(frame.columns, normalized)):
        name = str(raw_name)
        series = frame.iloc[:, index]
        logical, confidence = _logical_type(series, normal_name)
        dtype = series.dtype
        category = None
        try:
            import pandas as pd
            if isinstance(dtype, pd.CategoricalDtype):
                category = {
                    "ordered": bool(dtype.ordered),
                    "levels": [str(value) for value in dtype.categories[:10_000]],
                    "levels_truncated": len(dtype.categories) > 10_000,
                }
        except Exception:
            pass
        timezone_name = arrow.get(name, {}).get("timezone")
        if timezone_name is None:
            timezone_name = str(getattr(dtype, "tz", "")) or None
        decimal = arrow.get(name, {}).get("decimal") or _decimal_metadata(series, logical)
        row = {
            "position": index,
            "original_name": original_by_position.get(index, source_names[index]),
            "materialized_name": name,
            "normalized_name": normal_name,
            "storage_type": str(dtype),
            "logical_type": logical,
            "type_confidence": confidence,
            "nullable": bool(series.isna().any()),
            "variable_label": labels.get(name) or arrow.get(name, {}).get("label"),
            "value_labels": value_labels.get(name) or None,
            "declared_missing_values": missing_values.get(name) or None,
            "measurement_unit": arrow.get(name, {}).get("units") or arrow.get(name, {}).get("unit") or (
                global_units if len(frame.columns) == 1 or name == str(sidecar.get("variable")) else None
            ),
            "timezone": timezone_name,
            "coordinate_system": arrow.get(name, {}).get("crs") or (
                global_crs if "geospatial" in _role_flags(series, normal_name, logical) else None
            ),
            "categorical": category,
            "decimal": decimal,
            "roles": _role_flags(series, normal_name, logical),
            "source_metadata_keys": arrow.get(name, {}).get("metadata_keys", []),
        }
        rows.append(row)
    return rows


def _update_digest_scalar(digest: Any, value: Any) -> None:
    if value is None:
        digest.update(b"n;")
        return
    try:
        import pandas as pd
        if pd.isna(value):
            digest.update(b"n;")
            return
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        digest.update(b"b1;" if value else b"b0;")
    elif isinstance(value, int):
        digest.update(f"i{value};".encode("ascii"))
    elif isinstance(value, float):
        if math.isnan(value):
            digest.update(b"n;")
        elif math.isinf(value):
            digest.update(b"f+inf;" if value > 0 else b"f-inf;")
        else:
            digest.update(b"f" + value.hex().encode("ascii") + b";")
    elif isinstance(value, Decimal):
        digest.update(b"d" + str(value.normalize()).encode("ascii") + b";")
    elif isinstance(value, (datetime, date, time)):
        digest.update(b"t" + value.isoformat().encode("utf-8") + b";")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(b"x" + base64.b64encode(bytes(value)) + b";")
    elif isinstance(value, Mapping):
        digest.update(b"m" + _canonical_json(value) + b";")
    elif isinstance(value, (list, tuple)):
        digest.update(b"l" + _canonical_json(value) + b";")
    else:
        raw = str(value).encode("utf-8", "surrogatepass")
        digest.update(b"s" + str(len(raw)).encode("ascii") + b":" + raw + b";")


def _content_hash(frame: Any, normalized_names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json({"columns": list(normalized_names), "rows": len(frame)}))
    for row in frame.itertuples(index=False, name=None):
        digest.update(b"[")
        for value in row:
            _update_digest_scalar(digest, value)
        digest.update(b"]")
    return digest.hexdigest()


def _selection_scope(path: Path, selection: Mapping[str, Any] | None, sidecar: Mapping[str, Any]) -> dict[str, Any]:
    chosen = dict(selection or {})
    if _logical_suffix(path) in {".xlsx", ".xls", ".ods"}:
        chosen.setdefault("worksheet", 0)
    for key in (
        "dataset", "variable", "layer", "archive_member", "hdu", "record_path",
        "extraction_scope", "query_sha256", "backend", "row_limit",
        "column_renames", "source_kind", "remote_version", "remote_identifiers",
    ):
        if key in sidecar and key not in chosen:
            chosen[key] = sidecar[key]
    if len(chosen) > 32:
        raise CanonicalDatasetError("dataset selection contains too many fields")
    return {str(key): value for key, value in sorted(chosen.items())}


def _structure(columns: Sequence[Mapping[str, Any]], sidecar: Mapping[str, Any]) -> dict[str, Any]:
    nested = [row["normalized_name"] for row in columns if row["logical_type"] in {"list", "struct", "mixed"}]
    selected = any(key in sidecar for key in ("dataset", "variable", "layer", "archive_member", "hdu"))
    return {
        "kind": "selected_table" if selected else "table",
        "nested_columns": nested,
        "multi_table_source": bool(selected or sidecar.get("resource_types")),
    }


def _snapshot(cwd: Path, path: Path, source_hash: str, components: Sequence[Mapping[str, Any]]) -> list[str]:
    root = _store_root(cwd) / "snapshots" / source_hash[:2] / source_hash
    root.mkdir(parents=True, exist_ok=True)
    published: list[str] = []
    names = {str(row["name"]): str(row["sha256"]) for row in components}
    for source_file in _source_components(path):
        target = root / source_file.name
        expected = names[source_file.name]
        if target.is_file():
            if _sha256_file(target) != expected:
                raise CanonicalDatasetError("immutable source snapshot failed integrity verification")
        else:
            fd, temporary = tempfile.mkstemp(prefix=".snapshot-", suffix=".tmp", dir=root)
            os.close(fd)
            try:
                shutil.copyfile(source_file, temporary)
                if _sha256_file(Path(temporary)) != expected:
                    raise CanonicalDatasetError("source changed while its immutable snapshot was being created")
                try:
                    # Publish without replacing a snapshot another Sift
                    # process may have won concurrently.  Do not mark the
                    # temporary name read-only before unlinking it: on
                    # Windows ``chmod(..., 0o400)`` sets the read-only file
                    # attribute and Windows then refuses that unlink.  The
                    # parser child is already gone and this private store is
                    # not exposed to it, so hard-link, remove the temporary
                    # name, and only then freeze the surviving target.
                    os.link(temporary, target)
                    Path(temporary).unlink(missing_ok=True)
                    os.chmod(target, 0o400)
                except FileExistsError:
                    Path(temporary).unlink(missing_ok=True)
            except Exception:
                Path(temporary).unlink(missing_ok=True)
                raise
        if target.is_symlink() or not target.is_file() or _sha256_file(target) != expected:
            raise CanonicalDatasetError("immutable source snapshot failed integrity verification")
        try:
            os.chmod(target, 0o400)
        except OSError:
            pass
        published.append(target.relative_to(Path(cwd).resolve()).as_posix())
    return published


def snapshot_source_artifact(cwd: Path, source: Path) -> dict[str, Any]:
    """Content-address and immutably snapshot a non-tabular source artifact."""
    root = Path(cwd).resolve(strict=True)
    original = Path(source).expanduser()
    if original.is_symlink():
        raise CanonicalDatasetError("source artifacts may not be symbolic links")
    path = original.resolve(strict=True)
    if not path.is_file():
        raise CanonicalDatasetError("source artifact must be a regular file")
    try:
        relative: str | None = path.relative_to(root).as_posix()
    except ValueError:
        # File-picker workflows legitimately select a source outside the
        # session before materializing it inward. Never persist its absolute
        # path; the immutable snapshot and source hash are the durable identity.
        relative = None
    source_hash, components = _source_hashes(path)
    body = {
        "kind": "source_artifact",
        "session_relative_path": relative,
        "source_name": path.name,
        "origin": "session" if relative is not None else "external_selected_file",
        "format": _logical_suffix(path).removeprefix("."),
        "source_sha256": source_hash,
        "components": components,
        "snapshot_paths": _snapshot(root, path, source_hash, components),
    }
    body["fingerprint"] = hashlib.sha256(_canonical_json(body)).hexdigest()
    return body


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(value, dict) or value.get("manifest_version") != MANIFEST_VERSION:
        return None
    fingerprint = value.get("fingerprint")
    body = {key: item for key, item in value.items() if key not in {"fingerprint", "created_at"}}
    if not isinstance(fingerprint, str) or hashlib.sha256(_canonical_json(body)).hexdigest() != fingerprint:
        return None
    return value


def _path_index_path(cwd: Path, relative: str) -> Path:
    return _store_root(cwd) / "paths" / (
        hashlib.sha256(relative.encode("utf-8")).hexdigest() + ".json"
    )


def _current_cached_manifest(
    cwd: Path, path: Path, relative: str, scope: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a verified manifest without parsing observations on a cache hit."""
    index_path = _path_index_path(cwd, relative)
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or not isinstance(index.get("fingerprint"), str):
        return None
    current_hash, _components = _source_hashes(path)
    if index.get("source_sha256") != current_hash:
        return None
    manifest = _read_manifest(manifest_path(cwd, index["fingerprint"]))
    if manifest is None:
        return None
    if manifest.get("selection") != dict(scope):
        return None
    if manifest.get("source", {}).get("session_relative_path") != relative:
        return None
    return manifest


def _publish_manifest(
    root: Path, path: Path, relative: str, source_hash: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    fingerprint = hashlib.sha256(_canonical_json(body)).hexdigest()
    manifest = {
        **body,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    store = _store_root(root)
    target = store / "manifests" / f"{fingerprint}.json"
    index_path = _path_index_path(root, relative)
    with exclusive_file_lock(store / "canonical.lock"):
        existing = _read_manifest(target)
        if existing is None:
            _atomic_json(target, manifest)
        else:
            manifest = existing
        previous = None
        try:
            previous = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        stat = path.stat()
        _atomic_json(index_path, {
            "session_relative_path": relative,
            "fingerprint": fingerprint,
            "source_sha256": source_hash,
            "stat": {
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            },
            "previous_fingerprint": (
                previous.get("fingerprint") if isinstance(previous, dict)
                and previous.get("fingerprint") != fingerprint else None
            ),
        })
        # Invalidate only stale derived cache entries for this path. Manifests
        # and snapshots remain immutable provenance history.
        if isinstance(previous, dict):
            old = previous.get("fingerprint")
            if isinstance(old, str) and old != fingerprint:
                stale = store / "cache" / f"{old}.parquet"
                stale.unlink(missing_ok=True)
                stale.with_suffix(".integrity.json").unlink(missing_ok=True)
    return manifest


def _clean_lineage(
    parents: Sequence[str], transformations: Sequence[Mapping[str, Any]],
) -> tuple[list[str], list[dict[str, Any]]]:
    parent_rows = sorted(set(str(parent) for parent in parents))
    if any(not re.fullmatch(r"[0-9a-f]{64}", parent) for parent in parent_rows):
        raise CanonicalDatasetError("lineage parent fingerprints must be SHA-256 values")
    clean_transformations: list[dict[str, Any]] = []
    allowed_keys = frozenset({
        "operation", "code_sha256", "parameters_sha256", "runtime",
        "runtime_version", "source_artifact_fingerprint", "query_sha256",
        "accepted_finding_ids", "correction_count",
    })
    for transformation in transformations:
        if not isinstance(transformation, Mapping):
            raise CanonicalDatasetError("dataset transformations must be metadata objects")
        row = {
            str(key): value for key, value in transformation.items()
            if str(key) in allowed_keys
            and (value is None or isinstance(value, (str, int, float, bool)))
        }
        if not row.get("operation"):
            raise CanonicalDatasetError("every dataset transformation needs an operation")
        if any(len(str(value)) > 500 for value in row.values() if value is not None):
            raise CanonicalDatasetError("dataset transformation metadata is too long")
        clean_transformations.append(row)
    return parent_rows, clean_transformations


def create_manifest(
    cwd: Path,
    source: Path,
    *,
    selection: Mapping[str, Any] | None = None,
    dataset_kind: str = "source",
    parents: Sequence[str] = (),
    transformations: Sequence[Mapping[str, Any]] = (),
    snapshot: bool = True,
    frame: Any | None = None,
) -> dict[str, Any]:
    """Create or return the exact canonical manifest for one local table."""
    if dataset_kind not in {"source", "derived"}:
        raise CanonicalDatasetError("dataset_kind must be 'source' or 'derived'")
    root = Path(cwd).resolve(strict=True)
    path, relative = _resolve_source(root, source)
    sidecar = _metadata_sidecar(path)
    scope = _selection_scope(path, selection, sidecar)
    parent_rows, clean_transformations = _clean_lineage(
        parents, transformations,
    )
    source_hash, components = _source_hashes(path)
    if frame is None:
        from sift.schema import load_data
        sheet = scope.get("worksheet")
        frame = load_data(
            path, sheet=sheet, r_object=scope.get("r_object"),
        )
    columns = _column_manifest(frame, path, sidecar, scope)
    normalized = [row["normalized_name"] for row in columns]
    content_hash = _content_hash(frame, normalized)
    body: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "source": {
            "session_relative_path": relative,
            "kind": dataset_kind,
            "format": _logical_suffix(path).removeprefix("."),
            "source_sha256": source_hash,
            "components": components,
            "snapshot_paths": [],
        },
        "content_sha256": content_hash,
        "parser": _manifest_parser(path, sidecar),
        "selection": scope,
        "shape": {"rows": int(len(frame)), "columns": int(len(frame.columns))},
        "columns": columns,
        "source_specific_metadata": _bounded_source_metadata(sidecar),
        "structure": _structure(columns, sidecar),
        "lineage": {
            "parents": parent_rows,
            "transformations": clean_transformations,
        },
    }
    if snapshot:
        body["source"]["snapshot_paths"] = _snapshot(root, path, source_hash, components)
    final_source_hash, final_components = _source_hashes(path)
    if final_source_hash != source_hash or final_components != components:
        raise CanonicalDatasetError("source changed while its canonical manifest was being created")
    return _publish_manifest(root, path, relative, source_hash, body)


def _metadata_only_manifest(
    cwd: Path, path: Path, relative: str, scope: Mapping[str, Any],
    *, snapshot: bool, dataset_kind: str = "source",
    parents: Sequence[str] = (),
    transformations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Canonical identity for data too large for a trusted full-memory load."""
    source_hash, components = _source_hashes(path)
    sidecar = _metadata_sidecar(path)
    sample = None
    sampled = False
    try:
        from sift.dataset_profile import _read_frame
        sample, sampled = _read_frame(
            path, True,
            sheet=scope.get("worksheet"),
            session_root=cwd,
        )
    except Exception:
        sample = None
    columns: list[dict[str, Any]] = []
    if sample is not None:
        columns = _column_manifest(sample, path, sidecar, scope)
    else:
        try:
            from sift.schema import extract
            schema = extract(path, "names_types_labels", sheet=scope.get("worksheet"))
            raw_variables = schema.get("variables", [])
            normalized = _normalized_names([row.get("name", "") for row in raw_variables])
            for index, (row, name) in enumerate(zip(raw_variables, normalized)):
                original = str(row.get("name", f"column_{index + 1}"))
                roles = []
                if _ID_NAME.search(name):
                    roles.append("identifier_like")
                if _WEIGHT_NAME.search(name):
                    roles.append("weight")
                if _TIME_NAME.search(name):
                    roles.append("time_index")
                if _GEO_NAME.search(name):
                    roles.append("geospatial")
                columns.append({
                    "position": index,
                    "original_name": original,
                    "normalized_name": name,
                    "storage_type": None,
                    "logical_type": row.get("type", "unknown"),
                    "type_confidence": 0.7 if row.get("type") else 0.0,
                    "nullable": None,
                    "variable_label": row.get("label"),
                    "value_labels": row.get("value_labels"),
                    "declared_missing_values": None,
                    "measurement_unit": None,
                    "timezone": None,
                    "coordinate_system": None,
                    "categorical": None,
                    "decimal": None,
                    "roles": roles,
                    "source_metadata_keys": [],
                })
        except Exception:
            columns = []
    try:
        from sift.schema import row_count
        rows = row_count(path)
    except Exception:
        rows = None
    parent_rows, clean_transformations = _clean_lineage(
        parents, transformations,
    )
    body: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "source": {
            "session_relative_path": relative,
            "kind": dataset_kind,
            "format": _logical_suffix(path).removeprefix("."),
            "source_sha256": source_hash,
            "components": components,
            "snapshot_paths": _snapshot(cwd, path, source_hash, components) if snapshot else [],
        },
        "content_sha256": source_hash,
        "content_hash_basis": "source_bytes",
        "parser": _manifest_parser(path, sidecar),
        "selection": dict(scope),
        "shape": {
            "rows": int(rows) if rows is not None else (int(len(sample)) if sample is not None else None),
            "rows_exact": rows is not None,
            "columns": len(columns),
            "columns_exact": bool(columns),
        },
        "columns": columns,
        "source_specific_metadata": _bounded_source_metadata(sidecar),
        "structure": _structure(columns, sidecar),
        "lineage": {
            "parents": parent_rows,
            "transformations": clean_transformations,
        },
        "profiling_scope": "bounded_sample" if sampled else "metadata_only",
    }
    final_hash, final_components = _source_hashes(path)
    if final_hash != source_hash or final_components != components:
        raise CanonicalDatasetError("source changed while its canonical manifest was being created")
    return _publish_manifest(cwd, path, relative, source_hash, body)


def ensure_manifest(
    cwd: Path,
    source: Path,
    *,
    selection: Mapping[str, Any] | None = None,
    snapshot: bool = True,
    dataset_kind: str = "source",
    parents: Sequence[str] = (),
    transformations: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Return a current manifest, parsing observations only on a cache miss."""
    root = Path(cwd).resolve(strict=True)
    path, relative = _resolve_source(root, source)
    scope = _selection_scope(path, selection, _metadata_sidecar(path))
    current = _current_cached_manifest(root, path, relative, scope)
    if current is not None:
        # A prior caller may have opted out of snapshots. A later trusted
        # analysis requesting one must not silently inherit that weaker state.
        requested_parents, requested_transformations = _clean_lineage(
            parents, transformations,
        )
        if (
            current.get("source", {}).get("kind") == dataset_kind
            and current.get("lineage", {}).get("parents") == requested_parents
            and current.get("lineage", {}).get("transformations")
            == requested_transformations
            and (not snapshot or current.get("source", {}).get("snapshot_paths"))
        ):
            return current
    try:
        return create_manifest(
            root, path, selection=scope, snapshot=snapshot,
            dataset_kind=dataset_kind, parents=parents,
            transformations=transformations,
        )
    except Exception as exc:
        from sift.schema import DatasetTooLargeError
        if not isinstance(exc, DatasetTooLargeError):
            raise
        return _metadata_only_manifest(
            root, path, relative, scope, snapshot=snapshot,
            dataset_kind=dataset_kind, parents=parents,
            transformations=transformations,
        )


def current_manifest(
    cwd: Path, source: Path, *, selection: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Return the current verified identity, or ``None`` without parsing rows."""
    root = Path(cwd).resolve(strict=True)
    path, relative = _resolve_source(root, source)
    scope = _selection_scope(path, selection, _metadata_sidecar(path))
    return _current_cached_manifest(root, path, relative, scope)


def manifest_path(cwd: Path, fingerprint: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise CanonicalDatasetError("invalid canonical dataset fingerprint")
    return _store_root(Path(cwd)) / "manifests" / f"{fingerprint}.json"


def discard_uncommitted_manifest(
    cwd: Path,
    source: Path,
    fingerprint: str,
) -> bool:
    """Remove an ingestion manifest that was never provenance-committed.

    Ingestion has to establish canonical identity before it can append the
    release record that commits the dataset.  If that append fails, retaining
    the immutable snapshot would silently preserve a second copy of a
    confidential dataset even though the import itself was reported as
    failed.  This narrowly scoped rollback removes only the exact current
    path/fingerprint pair and refuses to act when another path, manifest, or
    lineage entry references it.

    The caller is responsible for using this only before publishing any
    release/audit record that names the fingerprint.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        return False
    root = Path(cwd).resolve(strict=True)
    try:
        resolved, relative = _resolve_source(root, source)
    except (CanonicalDatasetError, OSError):
        return False
    store = _store_root(root)
    index_path = _path_index_path(root, relative)
    target_manifest = store / "manifests" / f"{fingerprint}.json"
    with exclusive_file_lock(store / "canonical.lock"):
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        manifest = _read_manifest(target_manifest)
        if (
            not isinstance(index, dict)
            or index.get("fingerprint") != fingerprint
            or index.get("previous_fingerprint") is not None
            or manifest is None
            or manifest.get("source", {}).get("session_relative_path") != relative
        ):
            return False

        # A fingerprint normally belongs to one path because the path is part
        # of its canonical body.  Verify that invariant before removing it.
        for candidate in (store / "paths").glob("*.json"):
            if candidate == index_path:
                continue
            try:
                other = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if isinstance(other, dict) and other.get("fingerprint") == fingerprint:
                return False
        for candidate in (store / "manifests").glob("*.json"):
            if candidate == target_manifest:
                continue
            other = _read_manifest(candidate)
            if other is None:
                return False
            parents = other.get("lineage", {}).get("parents", [])
            if isinstance(parents, list) and fingerprint in parents:
                return False

        snapshot_paths = manifest.get("source", {}).get("snapshot_paths", [])
        referenced_snapshots: set[str] = set()
        for candidate in (store / "manifests").glob("*.json"):
            if candidate == target_manifest:
                continue
            other = _read_manifest(candidate)
            if other is None:
                return False
            values = other.get("source", {}).get("snapshot_paths", [])
            if isinstance(values, list):
                referenced_snapshots.update(
                    value for value in values if isinstance(value, str)
                )

        # Remove every confidential byte-bearing artifact *before* removing
        # the metadata that makes it discoverable and retryable.  If an OS
        # lock or permission error prevents deletion, retain the index and
        # manifest and report failure to the caller; an untracked orphan is
        # worse than a failed rollback that can be retried deliberately.
        cache = store / "cache" / f"{fingerprint}.parquet"
        try:
            cache.unlink(missing_ok=True)
            cache.with_suffix(".integrity.json").unlink(missing_ok=True)
        except OSError:
            return False
        if isinstance(snapshot_paths, list):
            for value in snapshot_paths:
                if not isinstance(value, str):
                    return False
                if value in referenced_snapshots:
                    continue
                candidate = (root / value).resolve()
                try:
                    candidate.relative_to(store.resolve())
                except ValueError:
                    return False
                try:
                    # Snapshot publication intentionally removes write bits.
                    # Windows maps that to a read-only file attribute which
                    # must be cleared before deletion; POSIX unlink does not
                    # require this but the chmod is harmless.
                    if candidate.exists():
                        os.chmod(candidate, 0o600)
                    candidate.unlink(missing_ok=True)
                except OSError:
                    return False
                for directory in (candidate.parent, candidate.parent.parent):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
        try:
            target_manifest.unlink(missing_ok=True)
            index_path.unlink(missing_ok=True)
        except OSError:
            return False
        return not index_path.exists() and not target_manifest.exists()


def load_canonical_dataset(
    cwd: Path,
    source: Path,
    *,
    selection: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a frame and its manifest through the verified local cache."""
    root = Path(cwd).resolve(strict=True)
    path, _relative = _resolve_source(root, source)
    scope = dict(selection or {})
    if columns is not None:
        scope["columns"] = [str(column) for column in columns]
    from sift.schema import load_data
    sheet = scope.get("worksheet", scope.get("sheet"))
    _resolved, relative = _resolve_source(root, path)
    manifest = _current_cached_manifest(root, path, relative, scope)
    frame = None
    if manifest is None:
        # One parse produces both the content identity and the cache on a miss.
        frame = load_data(
            path, sheet=sheet, columns=list(columns) if columns else None,
            r_object=scope.get("r_object"),
        )
        manifest = create_manifest(root, path, selection=scope, frame=frame)
    cache = _store_root(root) / "cache" / f"{manifest['fingerprint']}.parquet"
    integrity = cache.with_suffix(".integrity.json")
    cache.parent.mkdir(parents=True, exist_ok=True)
    lock = cache.with_suffix(".lock")
    with exclusive_file_lock(lock):
        valid = False
        if cache.is_file() and not cache.is_symlink() and integrity.is_file() and not integrity.is_symlink():
            try:
                integrity_row = json.loads(integrity.read_text(encoding="utf-8"))
                valid = (
                    isinstance(integrity_row, dict)
                    and integrity_row.get("fingerprint") == manifest["fingerprint"]
                    and integrity_row.get("content_sha256") == manifest["content_sha256"]
                    and integrity_row.get("parquet_sha256") == _sha256_file(cache)
                )
                if valid:
                    return load_data(cache), manifest
            except Exception:
                valid = False
        if not valid:
            cache.unlink(missing_ok=True)
            integrity.unlink(missing_ok=True)
            if frame is None:
                frame = load_data(
                    path, sheet=sheet, columns=list(columns) if columns else None,
                    r_object=scope.get("r_object"),
                )
                if _content_hash(frame, [row["normalized_name"] for row in manifest["columns"]]) != manifest["content_sha256"]:
                    raise CanonicalDatasetError("source content changed after canonical identity verification")
            fd, temporary = tempfile.mkstemp(prefix=".dataset-cache-", suffix=".parquet", dir=cache.parent)
            os.close(fd)
            try:
                frame.to_parquet(temporary, index=False)
                os.chmod(temporary, 0o600)
                os.replace(temporary, cache)
                import pyarrow.parquet as pq
                cached_frame = pq.read_table(cache).to_pandas()
                # The frame was already content-hashed while the manifest was
                # created. Re-encoding every scalar after the Parquet round
                # trip doubled cold-load CPU time on wide data. DataFrame.equals
                # is the stronger check needed here: it compares shape, labels,
                # dtypes, values, and missing-value positions in vectorized
                # code. The exact Parquet bytes are SHA-256-bound immediately
                # below, so the durable cache boundary remains tamper evident.
                if not cached_frame.equals(frame):
                    cache.unlink(missing_ok=True)
                    return frame, manifest
                _atomic_json(integrity, {
                    "fingerprint": manifest["fingerprint"],
                    "content_sha256": manifest["content_sha256"],
                    "parquet_sha256": _sha256_file(cache),
                })
            except Exception:
                Path(temporary).unlink(missing_ok=True)
                cache.unlink(missing_ok=True)
                integrity.unlink(missing_ok=True)
                raise
    return frame, manifest


def load_canonical_data(
    cwd: Path,
    source: Path,
    *,
    selection: Mapping[str, Any] | None = None,
    columns: Sequence[str] | None = None,
) -> Any:
    """Load a frame through the canonical contract and verified local cache."""
    frame, _manifest = load_canonical_dataset(
        cwd, source, selection=selection, columns=columns,
    )
    return frame


def compare_schemas(previous: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    """Classify schema evolution without reading or retaining observations."""
    old = {row["normalized_name"]: row for row in previous.get("columns", [])}
    new = {row["normalized_name"]: row for row in current.get("columns", [])}
    added = sorted(new.keys() - old.keys())
    removed = sorted(old.keys() - new.keys())
    changed = sorted(
        name for name in old.keys() & new.keys()
        if old[name].get("logical_type") != new[name].get("logical_type")
    )
    reordered = [name for name in new if name in old] != [name for name in old if name in new]
    return {
        "added": added,
        "removed": removed,
        "type_changed": changed,
        "reordered": reordered,
        "backward_compatible": not removed and not changed,
    }


def create_collection_manifest(
    cwd: Path,
    sources: Iterable[Path],
    *,
    partitioned: bool = False,
    snapshot: bool = True,
) -> dict[str, Any]:
    """Describe a bounded multi-table or partitioned dataset collection."""
    rows = list(sources)
    if not rows or len(rows) > MAX_COLLECTION_MEMBERS:
        raise CanonicalDatasetError("dataset collection member count is invalid")
    manifests = [create_manifest(cwd, source, snapshot=snapshot) for source in rows]
    evolution = [compare_schemas(manifests[0], current) for current in manifests[1:]]
    members = [{
        "path": item["source"]["session_relative_path"],
        "fingerprint": item["fingerprint"],
        "rows": item["shape"]["rows"],
    } for item in manifests]
    body = {
        "manifest_version": MANIFEST_VERSION,
        "kind": "partitioned_dataset" if partitioned else "dataset_collection",
        "members": members,
        "shape": {
            "rows": sum(item["shape"]["rows"] for item in manifests),
            "tables": len(manifests),
        },
        "schema": manifests[0]["columns"],
        "schema_evolution": evolution,
        "partition_keys": _partition_keys([item["path"] for item in members]) if partitioned else [],
    }
    body["content_sha256"] = hashlib.sha256(_canonical_json({
        "member_content": [item["content_sha256"] for item in manifests],
        "partition_keys": body["partition_keys"],
    })).hexdigest()
    fingerprint = hashlib.sha256(_canonical_json(body)).hexdigest()
    result = {**body, "fingerprint": fingerprint,
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    target = _store_root(Path(cwd)) / "collections" / f"{fingerprint}.json"
    with exclusive_file_lock(target.with_suffix(".lock")):
        if not target.exists():
            _atomic_json(target, result)
    return result


def _partition_keys(paths: Sequence[str]) -> list[str]:
    keys: set[str] = set()
    for value in paths:
        for part in Path(value).parts[:-1]:
            if "=" in part:
                key, _partition_value = part.split("=", 1)
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    keys.add(key)
    return sorted(keys)


def load_dataset_collection(
    cwd: Path, sources: Iterable[Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a multi-table collection without silently flattening its tables."""
    root = Path(cwd).resolve(strict=True)
    paths = list(sources)
    manifest = create_collection_manifest(root, paths, partitioned=False)
    tables = {
        path.resolve().relative_to(root).as_posix(): load_canonical_data(root, path)
        for path in paths
    }
    return tables, manifest


def load_partitioned_data(
    cwd: Path, root: Path,
) -> tuple[Any, dict[str, Any]]:
    """Load a Hive-style partition tree under one explicit collection contract."""
    import pandas as pd

    session = Path(cwd).resolve(strict=True)
    partition_root = Path(root).resolve(strict=True)
    try:
        partition_root.relative_to(session)
    except ValueError as exc:
        raise CanonicalDatasetError("partition root must remain inside the session") from exc
    paths = discover_partition_files(partition_root)
    manifest = create_collection_manifest(session, paths, partitioned=True)
    breaking = [
        row for row in manifest["schema_evolution"]
        if row["removed"] or row["type_changed"]
    ]
    if breaking:
        raise CanonicalDatasetError(
            "partition schemas contain removed columns or incompatible type changes"
        )
    from sift.schema import full_load_max_bytes
    total_bytes = sum(path.stat().st_size for path in paths)
    if total_bytes > full_load_max_bytes():
        raise CanonicalDatasetError(
            "partitioned dataset exceeds the safe combined in-memory load ceiling"
        )
    frames = []
    for path in paths:
        frame = load_canonical_data(session, path)
        partition_values: dict[str, str] = {}
        for part in path.relative_to(partition_root).parts[:-1]:
            if "=" in part:
                key, value = part.split("=", 1)
                if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                    partition_values[key] = value
        for key, value in partition_values.items():
            if key in frame.columns:
                non_null = frame[key].dropna().astype(str)
                if not non_null.empty and not bool((non_null == value).all()):
                    raise CanonicalDatasetError(
                        f"partition key {key!r} conflicts with values stored in the table"
                    )
            else:
                frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False), manifest


def discover_partition_files(root: Path) -> list[Path]:
    """Boundedly discover supported, non-symlink files below one directory."""
    from sift.schema import DATA_EXTENSIONS
    directory = Path(root).resolve(strict=True)
    if not directory.is_dir():
        raise CanonicalDatasetError("partition root must be a directory")
    rows: list[Path] = []
    for candidate in directory.rglob("*"):
        if candidate.is_symlink():
            continue
        if ".sift" in candidate.relative_to(directory).parts:
            continue
        if candidate.name.casefold().endswith(".metadata.json"):
            continue
        if candidate.is_file() and any(candidate.name.casefold().endswith(ext) for ext in DATA_EXTENSIONS):
            rows.append(candidate)
            if len(rows) > MAX_COLLECTION_MEMBERS:
                raise CanonicalDatasetError("partitioned dataset exposes too many files")
    return sorted(rows, key=lambda path: path.as_posix())


__all__ = [
    "CanonicalDatasetError", "MANIFEST_VERSION", "compare_schemas",
    "create_collection_manifest", "create_manifest", "current_manifest",
    "discover_partition_files",
    "discard_uncommitted_manifest",
    "ensure_manifest",
    "load_canonical_data", "load_canonical_dataset",
    "load_dataset_collection", "load_partitioned_data", "manifest_path",
    "snapshot_source_artifact",
]
