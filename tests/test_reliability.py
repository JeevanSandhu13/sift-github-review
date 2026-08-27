from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sift import enterprise_policy
from sift.reliability import (
    ReliabilityError,
    atomic_write_json,
    atomic_write_text,
    capacity_status,
    clock_safe_timestamp,
    session_recovery_report,
    sqlite_integrity,
)
from sift.store import ResultStore


def test_atomic_write_preserves_old_file_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "state.json"
    target.write_text('{"old":true}\n')
    real_replace = os.replace

    def fail_replace(source, destination):
        if Path(destination) == target:
            raise OSError("fault injected")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="fault injected"):
        atomic_write_json(target, {"new": True})
    assert target.read_text(encoding="utf-8") == '{"old":true}\n'
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_atomic_write_is_whole_under_concurrent_threads(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    payloads = [{"writer": index, "text": "x" * 1000} for index in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: atomic_write_json(target, value), payloads))
    assert json.loads(target.read_text(encoding="utf-8")) in payloads


def test_migration_failure_rolls_back_every_schema_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    db = tmp_path / ".sift" / "results.db"
    db.parent.mkdir()
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE results (id TEXT PRIMARY KEY, label TEXT NOT NULL, "
        "analysis_type TEXT NOT NULL, sanitized_payload TEXT NOT NULL, "
        "language TEXT NOT NULL, script_code TEXT NOT NULL, transformations TEXT NOT NULL, "
        "raw_log_path TEXT, created_at TEXT NOT NULL)"
    )
    connection.commit()
    connection.close()

    def broken(self, cols):
        self._conn.execute("ALTER TABLE results ADD COLUMN should_rollback TEXT")
        self._conn.execute("THIS IS NOT SQL")

    monkeypatch.setattr(ResultStore, "_apply_migrations", broken)
    with pytest.raises(sqlite3.Error):
        ResultStore(db)
    check = sqlite3.connect(db)
    try:
        columns = {row[1] for row in check.execute("PRAGMA table_info(results)")}
    finally:
        check.close()
    assert "should_rollback" not in columns


def test_store_integrity_checks_healthy_and_corrupt_databases(tmp_path: Path) -> None:
    db = tmp_path / ".sift" / "results.db"
    store = ResultStore(db)
    assert store.integrity_report()["ok"] is True
    store.close()
    assert sqlite_integrity(db)["ok"] is True
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"not sqlite")
    assert sqlite_integrity(broken)["ok"] is False


def test_capacity_preflight_and_read_only_failure_are_explicit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr("sift.reliability.shutil.disk_usage", lambda _p: Usage(100, 99, 1))
    status = capacity_status(tmp_path, incoming_bytes=2, reserve_bytes=0)
    assert status["ok"] is False and status["reason"] == "insufficient_free_space"
    with pytest.raises(ReliabilityError, match="insufficient_free_space"):
        atomic_write_text(tmp_path / "blocked.txt", "xx", reserve_bytes=0)


def test_unwritable_target_does_not_destroy_existing_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    target = tmp_path / "state.txt"
    target.write_text("old")

    def denied(*args, **kwargs):
        raise PermissionError("read-only filesystem")

    monkeypatch.setattr("sift.reliability.tempfile.mkstemp", denied)
    with pytest.raises(PermissionError, match="read-only"):
        atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "old"


def test_clock_safe_timestamp_never_moves_backwards() -> None:
    previous = "2030-01-01T00:00:00.000000+00:00"
    skewed = datetime(2020, 1, 1, tzinfo=timezone.utc)
    result = clock_safe_timestamp(previous, now=skewed)
    assert result == "2030-01-01T00:00:00.000001+00:00"


def test_duplicate_export_requests_converge_under_concurrency(tmp_path: Path) -> None:
    barrier = threading.Barrier(12)

    def request(_index: int):
        barrier.wait()
        return enterprise_policy.request_export_approval(tmp_path, "analysis_report")

    with ThreadPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(request, range(12)))
    assert len({row["id"] for row in rows}) == 1
    assert len(enterprise_policy.list_export_requests(tmp_path)) == 1


def test_concurrent_duplicate_turn_registration_is_idempotent(tmp_path: Path) -> None:
    from sift.runner import SessionRunner

    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-4-6",
    )
    assert runner.register_pending_turn("turn-a", "same") == "turn-a"
    assert runner.register_pending_turn("turn-b", "same") == "turn-a"
    runner.discard_pending_turn("turn-a")
    assert runner.register_pending_turn("turn-c", "same") == "turn-c"
    runner.discard_pending_turn("turn-c")


def test_session_recovery_quarantines_optional_index_and_cleans_stale_staging(
    tmp_path: Path,
) -> None:
    index_dir = tmp_path / ".sift" / "datasets" / "paths"
    index_dir.mkdir(parents=True)
    corrupt = index_dir / ("a" * 64 + ".json")
    corrupt.write_text("not json")
    stale = tmp_path / ".sift-cloud-interrupted.csv"
    stale.write_text("partial")
    os.utime(stale, (1, 1))

    preview = session_recovery_report(tmp_path)
    assert preview["ok"] is False
    assert preview["corrupt_optional_indexes"]
    assert preview["stale_artifacts"] == [stale.name]
    assert corrupt.is_file() and stale.is_file()

    repaired = session_recovery_report(
        tmp_path, clean_stale=True, repair_optional_indexes=True,
    )
    assert repaired["ok"] is True
    assert repaired["removed"] == [stale.name]
    assert not corrupt.exists() and not stale.exists()
    assert list(index_dir.glob("*.corrupt-*"))


def test_long_running_audit_history_remains_valid(tmp_path: Path) -> None:
    from sift.reproducibility import append_audit_event, verify_audit_chain

    started = time.perf_counter()
    for index in range(1_000):
        append_audit_event(tmp_path, "script_execution", {
            "script_run_id": f"run-{index}", "status": "ok",
        })
    health = verify_audit_chain(tmp_path)
    elapsed = time.perf_counter() - started
    assert health["valid"] is True and health["events"] == 1_000
    # Each event is individually locked, hash-linked, and fsync'd.  Do not
    # weaken that durability contract to make a shared runner look faster.
    # Local machines retain the strict interactive ceiling; hosted Windows
    # runners get a wider scheduler/storage-noise allowance while still
    # bounding the average durable append to 45 ms.
    if os.name == "nt" and os.environ.get("CI"):
        limit_seconds = 45.0
    else:
        limit_seconds = 10.0 if os.name == "nt" else 5.0
    assert elapsed < limit_seconds


def test_many_concurrent_sessions_keep_stores_isolated_and_consistent(
    tmp_path: Path,
) -> None:
    def populate(index: int) -> tuple[int, bool]:
        root = tmp_path / f"session-{index}"
        store = ResultStore(root / ".sift" / "results.db")
        for row in range(25):
            store.insert(
                label=f"s{index}-{row}", analysis_type="descriptive",
                sanitized_payload={"type": "descriptive", "n": 20},
                language="Python", script_code="# exact", transformations=[],
            )
        count = store.count()
        healthy = store.integrity_report()["ok"]
        store.close()
        return count, healthy

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(populate, range(32)))
    assert outcomes == [(25, True)] * 32
