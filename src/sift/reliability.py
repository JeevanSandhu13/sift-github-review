"""Crash-safe persistence, capacity checks, and session recovery diagnostics."""

from __future__ import annotations

import json
import hashlib
import re
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sift.file_lock import exclusive_file_lock


DEFAULT_FREE_SPACE_RESERVE = 512 * 1024 * 1024
STALE_ARTIFACT_SECONDS = 24 * 60 * 60


class ReliabilityError(RuntimeError):
    pass


def capacity_status(
    directory: Path, *, incoming_bytes: int = 0,
    reserve_bytes: int = DEFAULT_FREE_SPACE_RESERVE,
) -> dict[str, Any]:
    """Report whether a bounded write can complete without consuming reserve."""
    root = Path(directory)
    probe = root if root.exists() else root.parent
    try:
        usage = shutil.disk_usage(probe)
    except OSError as exc:
        return {
            "ok": False, "reason": "capacity_unavailable",
            "error": type(exc).__name__,
        }
    required = max(0, int(incoming_bytes)) + max(0, int(reserve_bytes))
    return {
        "ok": usage.free >= required,
        "free_bytes": usage.free,
        "required_bytes": required,
        "reserve_bytes": max(0, int(reserve_bytes)),
        "incoming_bytes": max(0, int(incoming_bytes)),
        "reason": None if usage.free >= required else "insufficient_free_space",
    }


def atomic_write_bytes(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    reserve_bytes: int = 0,
    use_lock: bool = True,
) -> None:
    """Write one complete file with fsync + same-directory replace.

    The old file remains intact on every failure before ``os.replace``. A
    sidecar lock serializes threads and processes; callers already holding a
    stronger transaction lock can opt out to avoid redundant lock files.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    health = capacity_status(
        target.parent, incoming_bytes=len(payload), reserve_bytes=reserve_bytes,
    )
    if not health.get("ok"):
        raise ReliabilityError(str(health.get("reason")))

    def _write() -> None:
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, target)
            # Persist the directory entry where the platform supports it.
            if os.name != "nt":
                directory_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    if use_lock:
        lock_root = Path(tempfile.gettempdir()) / "sift-file-locks"
        lock_name = hashlib.sha256(
            str(target.resolve()).encode("utf-8")
        ).hexdigest() + ".lock"
        with exclusive_file_lock(lock_root / lock_name):
            _write()
    else:
        _write()


def atomic_write_text(
    path: Path, text: str, *, mode: int = 0o600,
    reserve_bytes: int = 0, use_lock: bool = True,
) -> None:
    atomic_write_bytes(
        path, text.encode("utf-8"), mode=mode,
        reserve_bytes=reserve_bytes, use_lock=use_lock,
    )


def atomic_write_json(
    path: Path, value: Any, *, mode: int = 0o600,
    reserve_bytes: int = 0, use_lock: bool = True,
) -> None:
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False,
    ) + "\n"
    atomic_write_text(
        path, encoded, mode=mode, reserve_bytes=reserve_bytes, use_lock=use_lock,
    )


def clock_safe_timestamp(
    previous: str | None = None, *, now: datetime | None = None,
) -> str:
    """Return UTC ISO time strictly after ``previous`` despite wall-clock skew."""
    candidate = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if previous:
        try:
            prior = datetime.fromisoformat(previous.replace("Z", "+00:00"))
            if prior.tzinfo is None:
                prior = prior.replace(tzinfo=timezone.utc)
            prior = prior.astimezone(timezone.utc)
            if candidate <= prior:
                candidate = prior + timedelta(microseconds=1)
        except (TypeError, ValueError, OverflowError):
            pass
    return candidate.isoformat(timespec="microseconds")


def sqlite_integrity(db_path: Path) -> dict[str, Any]:
    """Read-only SQLite quick/integrity checks that never mutate a store."""
    path = Path(db_path)
    if not path.exists():
        return {"ok": True, "status": "absent"}
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=2.0)
        try:
            quick = [row[0] for row in connection.execute("PRAGMA quick_check")]
            foreign = [tuple(row) for row in connection.execute("PRAGMA foreign_key_check")]
        finally:
            connection.close()
        return {
            "ok": quick == ["ok"] and not foreign,
            "status": "ok" if quick == ["ok"] and not foreign else "corrupt",
            "quick_check": quick[:20],
            "foreign_key_issues": len(foreign),
        }
    except (OSError, sqlite3.Error) as exc:
        return {"ok": False, "status": "unreadable", "error": type(exc).__name__}


def _stale_candidates(cwd: Path, cutoff: float) -> list[Path]:
    roots = [Path(cwd), Path(cwd) / "exports", Path(cwd) / ".sift"]
    prefixes = (
        ".sift-cloud-", ".extract-metadata-", ".zotero-selection-",
        ".zotero-attachment-", ".quality-correction-", ".dataset-cache-",
    )
    rows: list[Path] = []
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        try:
            children = list(root.iterdir())
        except OSError:
            continue
        for path in children:
            name = path.name
            recognized = name.startswith(prefixes) or (
                root.name == "exports" and name.startswith(".") and ".building-" in name
            )
            try:
                if (
                    recognized and not path.is_symlink()
                    and path.stat().st_mtime < cutoff
                ):
                    rows.append(path)
            except OSError:
                continue
    return sorted(rows, key=lambda item: item.as_posix())


def _corrupt_optional_indexes(cwd: Path) -> list[Path]:
    directory = cwd / ".sift" / "datasets" / "paths"
    if not directory.is_dir() or directory.is_symlink():
        return []
    corrupt: list[Path] = []
    for path in sorted(directory.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(row, dict):
                raise ValueError("not an object")
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("fingerprint", ""))):
                raise ValueError("bad fingerprint")
            if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("source_sha256", ""))):
                raise ValueError("bad source hash")
        except (OSError, ValueError, TypeError):
            corrupt.append(path)
    return corrupt


def session_recovery_report(
    cwd: Path, *, clean_stale: bool = False,
    repair_optional_indexes: bool = False,
    stale_after_seconds: int = STALE_ARTIFACT_SECONDS,
) -> dict[str, Any]:
    """Inspect recoverability and optionally remove only recognized stale staging."""
    root = Path(cwd).resolve(strict=True)
    cutoff = time.time() - max(3600, int(stale_after_seconds))
    candidates = _stale_candidates(root, cutoff)
    removed: list[str] = []
    failures: list[str] = []
    if clean_stale:
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(relative)
            except OSError:
                failures.append(relative)

    corrupt_indexes = _corrupt_optional_indexes(root)
    quarantined: list[str] = []
    if repair_optional_indexes:
        for path in corrupt_indexes:
            relative = path.relative_to(root).as_posix()
            quarantine = path.with_suffix(
                path.suffix + f".corrupt-{time.time_ns()}"
            )
            try:
                os.replace(path, quarantine)
                quarantined.append(relative)
            except OSError:
                failures.append(relative)

    from sift.chat_history import history_health
    from sift.reproducibility import verify_audit_chain
    from sift.release_ledger import verify_chain

    history = history_health(root)
    ledger_ok, ledger_records, ledger_detail = verify_chain(root)
    report: dict[str, Any] = {
        "ok": False,
        "checked_at": clock_safe_timestamp(),
        "capacity": capacity_status(root),
        "result_store": sqlite_integrity(root / ".sift" / "results.db"),
        "chat_history": {
            "valid_records": history.valid_events,
            "invalid_records": history.invalid_lines,
            "unrecorded_events": history.unrecorded_events,
        },
        "release_ledger": {
            "valid": ledger_ok, "records": ledger_records, "detail": ledger_detail,
        },
        "reproducibility_audit": verify_audit_chain(root),
        "stale_artifacts": [p.relative_to(root).as_posix() for p in candidates],
        "removed": removed,
        "cleanup_failures": failures,
        "corrupt_optional_indexes": [
            path.relative_to(root).as_posix() for path in corrupt_indexes
        ],
        "quarantined_optional_indexes": quarantined,
    }
    report["ok"] = bool(
        report["capacity"].get("ok")
        and report["result_store"].get("ok")
        and report["chat_history"]["invalid_records"] == 0
        and report["release_ledger"]["valid"]
        and report["reproducibility_audit"].get("valid")
        and not failures
        and (
            not corrupt_indexes or len(quarantined) == len(corrupt_indexes)
        )
    )
    return report


__all__ = [
    "DEFAULT_FREE_SPACE_RESERVE", "ReliabilityError", "atomic_write_bytes",
    "atomic_write_json", "atomic_write_text", "capacity_status",
    "clock_safe_timestamp", "session_recovery_report", "sqlite_integrity",
]
