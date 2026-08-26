"""Secret-free, tamper-evident audit events for host integrations."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sift.file_lock import exclusive_file_lock
from sift.secure_file import append_bytes_no_follow, read_bytes_no_follow

AUDIT_SCHEMA_VERSION = 1
AUDIT_RELATIVE_PATH = Path(".sift") / "integration_audit.jsonl"
_ALLOWED_METADATA = frozenset({
    "auth_method", "bytes", "cancelled", "columns", "duration_ms",
    "input_tokens", "output_tokens", "region", "retry_count", "rows",
    "status_code", "truncated",
})


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _safe_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key not in _ALLOWED_METADATA:
            continue
        if isinstance(value, bool) or value is None:
            out[key] = value
        elif isinstance(value, int) and not isinstance(value, bool):
            out[key] = value
        elif isinstance(value, float) and value == value and abs(value) != float("inf"):
            out[key] = value
        elif key in {"auth_method", "region"} and isinstance(value, str):
            cleaned = value.strip()[:80]
            if cleaned and all(ord(char) >= 32 for char in cleaned):
                out[key] = cleaned
    return out


def _read_last_hash(path: Path) -> str:
    try:
        lines = read_bytes_no_follow(path, max_bytes=64 * 1024 * 1024).decode(
            "utf-8",
        ).splitlines()
    except FileNotFoundError:
        return "0" * 64
    except OSError:
        raise
    for line in reversed(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("event_hash") if isinstance(row, dict) else None
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("integration audit tail is malformed")
        return value
    return "0" * 64


def record_integration_event(
    cwd: Path,
    *,
    integration_id: str,
    kind: Literal["model", "database", "object_storage", "research_service"],
    action: str,
    outcome: Literal["success", "failure", "cancelled", "denied"],
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Append an allowlisted event; unknown metadata is discarded."""
    if not all(
        isinstance(value, str) and value and len(value) <= 100
        for value in (integration_id, kind, action, outcome)
    ):
        return False
    path = Path(cwd) / AUDIT_RELATIVE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path.parent, 0o700)
        except OSError:
            pass
        with exclusive_file_lock(path.with_suffix(".lock")):
            previous = _read_last_hash(path)
            event = {
                "schema_version": AUDIT_SCHEMA_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "integration_id": integration_id,
                "kind": kind,
                "action": action,
                "outcome": outcome,
                "metadata": _safe_metadata(metadata),
                "previous_hash": previous,
            }
            event["event_hash"] = hashlib.sha256(_canonical(event)).hexdigest()
            append_bytes_no_follow(
                path, _canonical(event) + b"\n", mode=0o600, sync=True,
            )
        return True
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def read_and_verify(cwd: Path) -> tuple[bool, list[dict[str, Any]]]:
    path = Path(cwd) / AUDIT_RELATIVE_PATH
    try:
        lines = read_bytes_no_follow(
            path, max_bytes=64 * 1024 * 1024,
        ).decode("utf-8").splitlines()
    except FileNotFoundError:
        return True, []
    except (OSError, UnicodeError):
        return False, []
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    try:
        for line in lines:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("previous_hash") != previous:
                return False, rows
            claimed = row.pop("event_hash", None)
            actual = hashlib.sha256(_canonical(row)).hexdigest()
            row["event_hash"] = claimed
            if claimed != actual:
                return False, rows
            previous = claimed
            rows.append(row)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, rows
    return True, rows


__all__ = [
    "AUDIT_RELATIVE_PATH", "AUDIT_SCHEMA_VERSION", "read_and_verify",
    "record_integration_event",
]
