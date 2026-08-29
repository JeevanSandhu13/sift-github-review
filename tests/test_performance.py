from __future__ import annotations

import asyncio
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from sift.performance import (
    PerformanceBudgets,
    create_representative_fixtures,
    evaluate_performance_budgets,
)
from sift.schema import load_data, scan_arrow_batches


def test_lazy_arrow_scan_projects_and_pushes_predicate(tmp_path: Path) -> None:
    path = tmp_path / "rows.parquet"
    pd.DataFrame({
        "id": range(10_000),
        "group": ["a"] * 5_000 + ["b"] * 5_000,
        "unused": ["x" * 100] * 10_000,
    }).to_parquet(path, index=False, row_group_size=1_000)
    batches = scan_arrow_batches(
        path, columns=["id", "group"],
        predicates=[("id", ">=", 9_000), ("group", "==", "b")],
        batch_size=257,
    )
    rows = list(batches)
    assert sum(batch.num_rows for batch in rows) == 1_000
    assert all(batch.schema.names == ["id", "group"] for batch in rows)
    assert max(batch.num_rows for batch in rows) <= 257

    filtered = load_data(
        path, columns=["id"], filters=[("id", ">=", 9_900)],
    )
    assert list(filtered.columns) == ["id"] and len(filtered) == 100
    with pytest.raises(Exception, match="unsupported predicate operator"):
        list(scan_arrow_batches(path, predicates=[("id", "contains", 1)]))

    csv_path = tmp_path / "rows.csv"
    pd.DataFrame({"id": range(10)}).to_csv(csv_path, index=False)
    with pytest.raises(Exception, match="supported only for Parquet"):
        load_data(csv_path, filters=[("id", ">=", 5)])


def test_provider_backpressure_bounds_global_concurrency(
    monkeypatch,
) -> None:
    import sift.runner as runner

    monkeypatch.setenv("SIFT_MAX_PROVIDER_CONCURRENCY", "2")
    active = 0
    maximum = 0
    waited = 0

    async def stream():
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.02)
        yield "done"
        active -= 1

    async def consume():
        nonlocal waited
        async for _ in runner._provider_events_with_backpressure(
            "test", stream(), 2, on_wait=lambda: _waited(),
        ):
            pass

    def _waited():
        nonlocal waited
        waited += 1

    async def drive():
        runner._PROVIDER_SEMAPHORES.clear()
        await asyncio.gather(*(consume() for _ in range(8)))

    asyncio.run(drive())
    assert maximum == 2
    assert waited >= 1


def test_representative_fixture_and_budgeted_qualification(tmp_path: Path) -> None:
    fixture = create_representative_fixtures(tmp_path / "fixture", rows=2_000)
    assert fixture["rows"] == 2_000
    assert Path(fixture["left"]).is_file() and Path(fixture["database"]).is_file()

    # Throughput budgets need enough payload to dominate fixed process,
    # connection, provenance, and Parquet-footer costs.  A 2,000-row extract
    # is a useful functional smoke test but is not a meaningful throughput
    # sample on slower CI/virtualized Windows hosts.
    # Run the release benchmark in a fresh interpreter. Running it after
    # thousands of tests makes retained native allocations from unrelated
    # libraries part of the sample and produces a machine-order-dependent
    # memory result.
    output = tmp_path / "qualification.json"
    process = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[1] / "scripts" / "performance_qualification.py"),
            "--rows", "20000",
            "--output", str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert process.returncode == 0, report["violations"]
    assert report["status"] == "pass", report["violations"]
    assert report["measurements"]["arrow_scan"]["predicate_pushdown"] is True
    assert (
        report["measurements"]["database_extract"]["throughput_basis"]
        == "uncompressed_parquet_row_group_bytes"
    )
    assert report["measurements"]["profile"]["cache_speedup"] > 1
    assert report["measurements"]["model_tokens"]["measurement"].startswith("provider")
    assert report["measurements"]["memory"]["peak_incremental_process_mib"] >= 0
    assert "native allocations" in report["measurements"]["memory"]["scope"]


def test_performance_regression_beyond_budget_fails() -> None:
    measurements = {
        "startup": {"seconds_median": 99.0},
        "schema": {"seconds": 99.0},
        "profile": {
            "cold_seconds": 99.0,
            "warm_seconds": 99.0,
            "cache_speedup": 0.0,
        },
        "linkage": {"seconds": 99.0},
        "arrow_scan": {"seconds": 99.0},
        "database_extract": {"mib_per_second": 0.0},
        "parquet_conversion": {"mib_per_second": 0.0},
        "memory": {"peak_incremental_process_mib": 9999.0},
        "local_workflow": {"seconds": 999.0},
    }
    violations = evaluate_performance_budgets(
        measurements, PerformanceBudgets(),
    )
    assert {row["metric"] for row in violations} == {
        "startup_seconds", "schema_seconds", "profile_seconds",
        "warm_profile_seconds", "profile_cache_speedup",
        "linkage_seconds", "arrow_scan_seconds", "database_mib_per_second",
        "parquet_mib_per_second", "peak_incremental_process_mib",
        "local_workflow_seconds",
    }
