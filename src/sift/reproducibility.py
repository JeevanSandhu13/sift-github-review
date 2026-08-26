"""Model-free reproducibility, provenance audit, and numerical comparison.

The replication bundle is intentionally executable without a provider.  It
contains exact scripts, expected sanitized payloads, source identities, parser
and runtime metadata, policy hashes, and a manifest over every exported file.
This module verifies that evidence, reports environment drift, and can rerun
the scripts against researcher-supplied source files using Sift's local
sandbox/executor only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, Iterable

from sift.file_lock import exclusive_file_lock
from sift.secure_file import append_bytes_no_follow


AUDIT_FILENAME = "reproducibility_audit.jsonl"
AUDIT_VERSION = 1
BUNDLE_MANIFEST = "bundle_manifest.json"
REPRODUCE_MANIFEST = "reproduce.json"
# A rerun report is derived from a bundle after publication.  It must not be
# part of the immutable source contract, otherwise the first rerun would make
# an otherwise untouched bundle fail its next integrity check.
DERIVED_REPORT = "reproduction_report.json"
_AUDIT_TIP_CACHE: dict[str, tuple[int, int, int, int, int, str, str | None]] = {}


def _audit_signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_metadata(value: dict[str, Any]) -> dict[str, Any]:
    """Keep the audit value-free and bounded by a closed metadata shape."""
    allowed = {
        "script_run_id", "result_id", "result_ids", "workflow_id",
        "analysis_id", "status", "verification_status", "warning_count",
        "bundle_manifest_sha256", "export_kind", "supersedes_result_id",
        "superseded_by", "reason_sha256", "challenge_status",
    }
    clean: dict[str, Any] = {}
    for key, raw in value.items():
        if key not in allowed:
            continue
        if isinstance(raw, (str, int, float, bool)) or raw is None:
            clean[key] = raw if not isinstance(raw, str) else raw[:500]
        elif isinstance(raw, list) and len(raw) <= 100 and all(
            isinstance(item, (str, int)) for item in raw
        ):
            clean[key] = [str(item)[:200] for item in raw]
    return clean


def append_audit_event(
    cwd: Path,
    event_type: str,
    metadata_row: dict[str, Any],
) -> dict[str, Any]:
    """Append one fsync'd, hash-linked, value-free provenance event."""
    if not event_type or len(event_type) > 80:
        raise ValueError("invalid reproducibility audit event type")
    from sift.config import ensure_private_sift_dir

    sift_dir = ensure_private_sift_dir(Path(cwd))
    path = sift_dir / AUDIT_FILENAME
    lock = path.with_suffix(path.suffix + ".lock")
    with exclusive_file_lock(lock):
        previous = "0" * 64
        previous_timestamp: str | None = None
        sequence = 1
        if path.is_file():
            try:
                signature = _audit_signature(path)
                cached = _AUDIT_TIP_CACHE.get(str(path))
                if cached is not None and cached[:4] == signature:
                    sequence = cached[4] + 1
                    previous = cached[5]
                    previous_timestamp = cached[6]
                else:
                    payload = path.read_bytes()
                    health = verify_audit_bytes(payload)
                    if not health.get("valid"):
                        raise RuntimeError("reproducibility audit chain is corrupt")
                    last = payload.decode("utf-8").splitlines()[-1]
                    decoded = json.loads(last)
                    previous = str(decoded["event_sha256"])
                    previous_timestamp = str(decoded.get("timestamp") or "") or None
                    sequence = int(decoded["sequence"]) + 1
            except Exception as exc:
                raise RuntimeError("reproducibility audit chain is corrupt") from exc
        from sift.reliability import clock_safe_timestamp
        body = {
            "version": AUDIT_VERSION,
            "sequence": sequence,
            "timestamp": clock_safe_timestamp(previous_timestamp),
            "event_type": event_type,
            "metadata": _safe_metadata(metadata_row),
            "previous_sha256": previous,
        }
        event = {**body, "event_sha256": hashlib.sha256(_canonical(body)).hexdigest()}
        append_bytes_no_follow(
            path, _canonical(event) + b"\n", mode=0o600, sync=True,
        )
        try:
            _AUDIT_TIP_CACHE[str(path)] = (
                *_audit_signature(path), sequence,
                str(event["event_sha256"]), str(event["timestamp"]),
            )
        except OSError:
            _AUDIT_TIP_CACHE.pop(str(path), None)
    return event


def verify_audit_bytes(payload: bytes) -> dict[str, Any]:
    """Verify an exact audit snapshot without a second filesystem read."""
    if not payload:
        return {"valid": True, "events": 0, "last_sha256": "0" * 64}
    previous = "0" * 64
    count = 0
    try:
        for count, line in enumerate(payload.decode("utf-8").splitlines(), 1):
            row = json.loads(line)
            digest = row.pop("event_sha256")
            if row.get("sequence") != count or row.get("previous_sha256") != previous:
                raise ValueError("sequence or previous hash mismatch")
            if hashlib.sha256(_canonical(row)).hexdigest() != digest:
                raise ValueError("event hash mismatch")
            previous = digest
        return {"valid": True, "events": count, "last_sha256": previous}
    except Exception as exc:
        return {"valid": False, "events": count, "reason": str(exc)}


def verify_audit_chain(cwd: Path) -> dict[str, Any]:
    path = Path(cwd) / ".sift" / AUDIT_FILENAME
    if not path.exists():
        return verify_audit_bytes(b"")
    try:
        return verify_audit_bytes(path.read_bytes())
    except OSError as exc:
        return {"valid": False, "events": 0, "reason": type(exc).__name__}


def _runtime_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    distributions = {
        "sift": "sift", "pandas": "pandas", "numpy": "numpy",
        "pyarrow": "pyarrow", "pyreadstat": "pyreadstat", "pyreadr": "pyreadr",
        "openpyxl": "openpyxl", "xlrd": "xlrd", "odfpy": "odfpy",
        "statsmodels": "statsmodels", "scipy": "scipy",
        "sklearn": "scikit-learn", "duckdb": "duckdb",
        "sqlalchemy": "sqlalchemy", "sqlglot": "sqlglot",
    }
    for package, distribution in distributions.items():
        try:
            versions[package] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return versions


def build_bundle_manifest(bundle_dir: Path) -> dict[str, Any]:
    """Hash every regular bundle file and publish the manifest atomically."""
    root = bundle_dir.resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"bundle cannot contain symbolic links: {relative}")
        if (
            relative in {BUNDLE_MANIFEST, DERIVED_REPORT}
            or not path.is_file()
        ):
            continue
        rows.append({
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    body = {
        "format": "sift-reproducibility-bundle",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generator": {"python": sys.version.split()[0], "packages": _runtime_versions()},
        "files": rows,
    }
    body["manifest_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    target = root / BUNDLE_MANIFEST
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=root, prefix=".bundle-manifest-", delete=False
    ) as handle:
        temp = Path(handle.name)
        json.dump(body, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp, 0o600)
    os.replace(temp, target)
    return body


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    root = bundle_dir.resolve()
    try:
        manifest = json.loads((root / BUNDLE_MANIFEST).read_text(encoding="utf-8"))
        digest = manifest.pop("manifest_sha256")
        if hashlib.sha256(_canonical(manifest)).hexdigest() != digest:
            raise ValueError("bundle manifest self-hash mismatch")
        expected = {row["path"]: row for row in manifest["files"]}
        actual: dict[str, dict[str, Any]] = {}
        unsafe: list[str] = []
        for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                unsafe.append(relative)
                continue
            if (
                relative in {BUNDLE_MANIFEST, DERIVED_REPORT}
                or not path.is_file()
            ):
                continue
            actual[relative] = {
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        if actual != expected or unsafe:
            missing = sorted(set(expected) - set(actual))
            added = sorted(set(actual) - set(expected))
            changed = sorted(
                key for key in set(actual) & set(expected) if actual[key] != expected[key]
            )
            return {
                "valid": False, "missing": missing, "added": added,
                "changed": changed, "unsafe": unsafe,
            }
        return {"valid": True, "files": len(expected), "manifest_sha256": digest}
    except Exception as exc:
        return {"valid": False, "reason": str(exc)}


def environment_drift(bundle_dir: Path) -> dict[str, Any]:
    root = bundle_dir.resolve()
    reproduce = json.loads((root / REPRODUCE_MANIFEST).read_text(encoding="utf-8"))
    expected = reproduce.get("environment", {}).get("packages", {})
    current = _runtime_versions()
    differences: list[dict[str, Any]] = []
    for name in sorted(set(expected) | set(current)):
        if expected.get(name) != current.get(name):
            differences.append({
                "package": name,
                "expected": expected.get(name),
                "current": current.get(name),
            })
    return {
        "drift": bool(differences),
        "differences": differences,
        "platform_expected": reproduce.get("environment", {}).get("platform"),
        "platform_current": platform.system(),
    }


def _compare(
    expected: Any,
    actual: Any,
    path: str,
    differences: list[dict[str, Any]],
    *,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            differences.append({"path": path, "expected": expected, "actual": actual})
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        e, a = float(expected), float(actual)
        if not (math.isfinite(e) and math.isfinite(a)) or not math.isclose(
            e, a, rel_tol=rtol, abs_tol=atol
        ):
            differences.append({
                "path": path,
                "expected": expected,
                "actual": actual,
                "absolute_error": abs(a - e) if math.isfinite(a) and math.isfinite(e) else None,
            })
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else str(key)
            if key not in expected or key not in actual:
                differences.append({
                    "path": child,
                    "expected": expected.get(key, "<missing>"),
                    "actual": actual.get(key, "<missing>"),
                })
            else:
                _compare(expected[key], actual[key], child, differences, rtol=rtol, atol=atol)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append({"path": path + ".length", "expected": len(expected), "actual": len(actual)})
        for index, (left, right) in enumerate(zip(expected, actual)):
            _compare(left, right, f"{path}[{index}]", differences, rtol=rtol, atol=atol)
        return
    if expected != actual:
        differences.append({"path": path, "expected": expected, "actual": actual})


def compare_payloads(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    rtol: float = 1e-7,
    atol: float = 1e-10,
) -> dict[str, Any]:
    differences: list[dict[str, Any]] = []
    _compare(expected, actual, "", differences, rtol=rtol, atol=atol)
    return {
        "match": not differences,
        "rtol": rtol,
        "atol": atol,
        "differences": differences[:500],
        "differences_truncated": len(differences) > 500,
    }


def _dataset_statuses(
    datasets: Iterable[dict[str, Any]], data_root: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        relative = str(dataset.get("path", ""))
        target = (data_root / relative).resolve()
        if data_root.resolve() not in target.parents:
            rows.append({"path": relative, "status": "unsafe_path"})
        elif not target.is_file():
            rows.append({"path": relative, "status": "missing"})
        else:
            actual = sha256_file(target)
            expected = dataset.get("source_sha256")
            rows.append({
                "path": relative,
                "status": (
                    "identity_unavailable" if not expected
                    else "match" if actual == expected
                    else "hash_mismatch"
                ),
                "expected_sha256": expected,
                "actual_sha256": actual,
            })
    return rows


def rerun_bundle(
    bundle_dir: Path,
    *,
    data_root: Path,
    executor_fn: Callable[..., Any] | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Rerun a verified bundle locally without importing any provider."""
    root = bundle_dir.resolve()
    integrity = verify_bundle(root)
    if not integrity.get("valid"):
        return {"status": "blocked", "reason": "bundle_integrity_failed", "integrity": integrity}
    manifest = json.loads((root / REPRODUCE_MANIFEST).read_text(encoding="utf-8"))
    dataset_rows = _dataset_statuses(manifest.get("datasets", []), data_root.resolve())
    if any(row["status"] != "match" for row in dataset_rows):
        return {"status": "blocked", "reason": "source_dataset_mismatch", "datasets": dataset_rows}
    if executor_fn is None:
        from sift.executor import run_script

        executor_fn = run_script
    from sift.sanitizer import sanitize

    run_reports: list[dict[str, Any]] = []
    overall = True
    for run in manifest.get("runs", []):
        if not run.get("runnable") or not run.get("script"):
            overall = False
            run_reports.append({
                "script_run_id": run.get("script_run_id"),
                "status": "not_runnable",
            })
            continue
        script = root / run["script"]
        if sha256_file(script) != run["script_sha256"]:
            return {"status": "blocked", "reason": "script_hash_mismatch", "script": run["script"]}
        outcome = executor_fn(
            run["language"],
            script.read_text(encoding="utf-8"),
            data_root.resolve(),
            timeout_seconds=timeout_seconds,
        )
        if not getattr(outcome, "ok", False):
            overall = False
            run_reports.append({"script_run_id": run["script_run_id"], "status": "execution_failed"})
            continue
        actual_payloads: list[dict[str, Any]] = []
        settings = run.get("privacy_configuration", {}).get(
            "disclosure_settings", {}
        )
        from sift.sanitizer import DEFAULT_CONFIG, SDCConfig
        try:
            replay_config = SDCConfig(
                min_n_regression=int(settings["min_n_regression"]),
                min_n_descriptive=int(settings["min_n_descriptive"]),
                min_n_ttest_group=int(settings["min_n_ttest_group"]),
                cell_suppression_threshold=int(
                    settings["cell_suppression_threshold"]
                ),
                min_n_did_cohort=int(settings["min_n_did_cohort"]),
                dominance_threshold=float(settings["dominance_threshold"]),
            )
        except (KeyError, TypeError, ValueError):
            replay_config = DEFAULT_CONFIG
        for payload in getattr(outcome, "result_payloads", []):
            cleaned = sanitize(payload, replay_config)
            if cleaned.ok and cleaned.sanitized:
                actual_payloads.append(cleaned.sanitized)
        expected_payloads = [
            json.loads((root / item["result"]).read_text(encoding="utf-8"))["payload"]
            for item in run.get("expected_results", [])
        ]
        comparison_policy = manifest.get("comparison", {})
        comparisons = [
            compare_payloads(
                expected,
                actual,
                rtol=float(comparison_policy.get("rtol", 1e-7)),
                atol=float(comparison_policy.get("atol", 1e-10)),
            )
            for expected, actual in zip(expected_payloads, actual_payloads)
        ]
        count_match = len(expected_payloads) == len(actual_payloads)
        matched = count_match and all(row["match"] for row in comparisons)
        overall = overall and matched
        run_reports.append({
            "script_run_id": run["script_run_id"],
            "status": "match" if matched else "different",
            "expected_count": len(expected_payloads),
            "actual_count": len(actual_payloads),
            "comparisons": comparisons,
        })
    report = {
        "status": "match" if overall else "different",
        "model_contacted": False,
        "bundle_integrity": integrity,
        "environment_drift": environment_drift(root),
        "datasets": dataset_rows,
        "runs": run_reports,
    }
    destination = root / DERIVED_REPORT
    # The report is deliberately outside the signed/hash manifest contract:
    # generating it must not make the source bundle fail its own verification.
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=root, prefix=".reproduction-report-", delete=False
    ) as handle:
        temp_report = Path(handle.name)
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(temp_report, 0o600)
    os.replace(temp_report, destination)
    return report
