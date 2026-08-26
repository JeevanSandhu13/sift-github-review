"""Repeatable local performance fixtures, measurements, and release budgets."""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from sift.subprocess_safety import run_bounded_capture


@dataclass(frozen=True)
class PerformanceBudgets:
    startup_seconds: float = 2.0
    schema_seconds: float = 0.5
    profile_seconds: float = 5.0
    warm_profile_seconds: float = 0.5
    profile_cache_speedup: float = 2.0
    linkage_seconds: float = 5.0
    arrow_scan_seconds: float = 0.5
    database_mib_per_second: float = 0.1
    parquet_mib_per_second: float = 25.0
    peak_incremental_process_mib: float = 256.0
    local_workflow_seconds: float = 12.0


DEFAULT_BUDGETS = PerformanceBudgets()


def create_representative_fixtures(root: Path, *, rows: int = 20_000) -> dict[str, Any]:
    """Create deterministic mixed numeric/categorical/time/linkage fixtures."""
    if not isinstance(rows, int) or not 1_000 <= rows <= 1_000_000:
        raise ValueError("fixture rows must be between 1,000 and 1,000,000")
    import numpy as np
    import pandas as pd

    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260821)
    ids: Any = np.arange(rows, dtype="int64")
    frame = pd.DataFrame({
        "participant_id": ids,
        "group": np.where(ids % 4 == 0, "treated", "comparison"),
        "site": np.array([f"site-{value % 25:02d}" for value in ids]),
        "event_date": pd.Timestamp("2020-01-01") + pd.to_timedelta(ids % 1460, unit="D"),
        "outcome": rng.normal(0, 1, rows) + (ids % 4 == 0) * 0.25,
        "exposure": rng.normal(10, 3, rows),
        "weight": rng.uniform(0.5, 2.0, rows),
        "flag": ids % 7 == 0,
    })
    left = directory / "cohort.parquet"
    right = directory / "outcomes.parquet"
    frame.to_parquet(left, index=False, row_group_size=2_000)
    frame[["participant_id", "outcome", "event_date"]].to_parquet(
        right, index=False, row_group_size=2_000,
    )
    database = directory / "research.sqlite"
    connection = sqlite3.connect(database)
    try:
        frame.to_sql("observations", connection, if_exists="replace", index=False)
        connection.execute("CREATE INDEX idx_observations_id ON observations(participant_id)")
        connection.commit()
    finally:
        connection.close()
    return {
        "rows": rows,
        "columns": len(frame.columns),
        "left": str(left), "right": str(right), "database": str(database),
        "left_bytes": left.stat().st_size,
    }


def _measure(operation: Callable[[], Any]) -> tuple[Any, float, int]:
    # Allocation tracing can multiply the run time of scalar-heavy pandas
    # operations and therefore makes a speed qualification measure the tracer
    # rather than the product. Sample the process resident set instead. This is
    # cross-platform and, unlike tracemalloc, includes native Arrow allocations.
    import psutil

    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = baseline
    stopped = threading.Event()

    def sample_rss() -> None:
        nonlocal peak
        while not stopped.wait(0.005):
            try:
                peak = max(peak, process.memory_info().rss)
            except (psutil.Error, OSError):
                return

    sampler = threading.Thread(
        target=sample_rss, name="sift-performance-rss", daemon=True,
    )
    sampler.start()
    started = time.perf_counter()
    try:
        value = operation()
        elapsed = time.perf_counter() - started
    finally:
        stopped.set()
        sampler.join(timeout=0.1)
        try:
            peak = max(peak, process.memory_info().rss)
        except (psutil.Error, OSError):
            pass
    return value, elapsed, max(0, peak - baseline)


def measure_startup(*, samples: int = 3) -> dict[str, Any]:
    """Measure a fresh interpreter importing Sift's application bridge."""
    source_root = Path(__file__).resolve().parents[1]
    code = (
        "import sys,time;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "t=time.perf_counter();import sift.ui;"
        "print(time.perf_counter()-t)"
    )
    timings: list[float] = []
    for _ in range(max(1, min(7, samples))):
        outcome = run_bounded_capture(
            [sys.executable, "-I", "-c", code], timeout=30, check=True,
        )
        timings.append(float(outcome.stdout.strip().splitlines()[-1]))
    return {
        "seconds_median": statistics.median(timings),
        "seconds_samples": timings,
        "samples": len(timings),
    }


def run_performance_qualification(
    root: Path, *, rows: int = 100_000,
    budgets: PerformanceBudgets = DEFAULT_BUDGETS,
) -> dict[str, Any]:
    """Run the real local paths and return a machine-readable budget verdict."""
    from sift.dataset_profile import _clear_profile_cache, profile_dataset
    from sift.linkage import analyze_pair
    from sift.schema import extract, scan_arrow_batches
    from sift.usage_meter import summarize

    session = Path(root)
    fixtures = create_representative_fixtures(session, rows=rows)
    left, right = Path(fixtures["left"]), Path(fixtures["right"])
    measurements: dict[str, Any] = {"startup": measure_startup()}
    peaks: list[int] = []

    schema_result, elapsed, peak = _measure(lambda: extract(left, "names_only"))
    peaks.append(peak)
    measurements["schema"] = {
        "seconds": elapsed, "variables": len(schema_result.get("variables", [])),
    }

    _clear_profile_cache()
    profile, cold, peak = _measure(
        lambda: profile_dataset(left, session_root=session)
    )
    peaks.append(peak)
    _cached, warm, warm_peak = _measure(
        lambda: profile_dataset(left, session_root=session)
    )
    peaks.append(warm_peak)
    measurements["profile"] = {
        "cold_seconds": cold, "warm_seconds": warm,
        "cache_speedup": cold / max(warm, 1e-9),
        "rows_profiled": profile.get("rows_profiled"),
    }

    links, elapsed, peak = _measure(
        lambda: analyze_pair(left, right, session_root=session)
    )
    peaks.append(peak)
    measurements["linkage"] = {
        "seconds": elapsed, "candidate_keys": len(links),
    }

    selected_rows = 0
    started = time.perf_counter()
    for batch in scan_arrow_batches(
        left, columns=["participant_id", "outcome"],
        predicates=[("participant_id", ">=", rows // 2)], batch_size=2_048,
    ):
        selected_rows += batch.num_rows
    measurements["arrow_scan"] = {
        "seconds": time.perf_counter() - started,
        "rows": selected_rows, "projected_columns": 2,
        "predicate_pushdown": True, "lazy_batches": True,
    }

    from sift.connectors import run_extract
    with tempfile.TemporaryDirectory(prefix="sift-perf-db-", dir=session) as temp_dir:
        extract_root = Path(temp_dir)
        benchmark_db = extract_root / "research.sqlite"
        shutil.copyfile(fixtures["database"], benchmark_db)
        result, elapsed, peak = _measure(lambda: run_extract(
            extract_root,
            connection=f"sqlite:///{benchmark_db}",
            sql="SELECT participant_id, outcome, exposure FROM observations",
            dataset_name="benchmark_extract", row_limit=rows,
        ))
        peaks.append(peak)
        # Measure logical bytes processed, not compressed bytes written.
        # Using the on-disk Parquet size perversely reports *lower* throughput
        # when compression improves and is especially noisy for small fixtures,
        # where the fixed footer is a large fraction of the file.  Parquet's
        # row-group metadata records the uncompressed encoded column bytes and
        # gives one comparable basis across OSes and compression ratios.
        import pyarrow.parquet as pq

        parquet_metadata = pq.read_metadata(result.dataset_path)
        logical_bytes = sum(
            parquet_metadata.row_group(index).total_byte_size
            for index in range(parquet_metadata.num_row_groups)
        )
        extracted_mib = logical_bytes / (1024 * 1024)
        measurements["database_extract"] = {
            "seconds": elapsed, "rows": result.rows,
            "mib_per_second": extracted_mib / max(elapsed, 1e-9),
            "throughput_basis": "uncompressed_parquet_row_group_bytes",
        }

    import pandas as pd
    frame = pd.read_parquet(left)
    with tempfile.TemporaryDirectory(prefix="sift-perf-parquet-", dir=session) as temp_dir:
        output = Path(temp_dir) / "converted.parquet"
        _value, elapsed, peak = _measure(lambda: frame.to_parquet(output, index=False))
        peaks.append(peak)
        source_mib = float(frame.memory_usage(index=True, deep=True).sum()) / (1024 * 1024)
        measurements["parquet_conversion"] = {
            "seconds": elapsed,
            "mib_per_second": source_mib / max(elapsed, 1e-9),
        }

    usage = summarize(session)
    measurements["model_tokens"] = {
        "input": usage.get("input_tokens", 0),
        "output": usage.get("output_tokens", 0),
        "cache_read": usage.get("cache_read_tokens", 0),
        "measurement": "provider_reported_session_totals",
    }
    local_workflow = sum((
        measurements["schema"]["seconds"],
        measurements["profile"]["cold_seconds"],
        measurements["linkage"]["seconds"],
        measurements["arrow_scan"]["seconds"],
    ))
    measurements["local_workflow"] = {"seconds": local_workflow}
    measurements["memory"] = {
        "peak_incremental_process_mib": max(peaks, default=0) / (1024 * 1024),
        "scope": (
            "Peak increase in resident process memory sampled with psutil, "
            "including native allocations"
        ),
    }

    fixture_summary = {
        "rows": fixtures["rows"], "columns": fixtures["columns"],
        "left_bytes": fixtures["left_bytes"],
        "formats": ["parquet", "sqlite"],
    }
    violations = evaluate_performance_budgets(measurements, budgets)
    limits = asdict(budgets)
    return {
        "format": "sift-performance-qualification", "version": 2,
        "platform": {"python": sys.version.split()[0], "os": os.name},
        "fixtures": fixture_summary, "measurements": measurements,
        "budgets": limits, "violations": violations,
        "status": "pass" if not violations else "fail",
    }


def evaluate_performance_budgets(
    measurements: dict[str, Any],
    budgets: PerformanceBudgets = DEFAULT_BUDGETS,
) -> list[dict[str, Any]]:
    """Return exact budget violations; any returned row fails qualification."""
    limits = asdict(budgets)
    violations: list[dict[str, Any]] = []
    checks = (
        ("startup_seconds", measurements["startup"]["seconds_median"], "max"),
        ("schema_seconds", measurements["schema"]["seconds"], "max"),
        ("profile_seconds", measurements["profile"]["cold_seconds"], "max"),
        ("warm_profile_seconds", measurements["profile"]["warm_seconds"], "max"),
        ("profile_cache_speedup", measurements["profile"]["cache_speedup"], "min"),
        ("linkage_seconds", measurements["linkage"]["seconds"], "max"),
        ("arrow_scan_seconds", measurements["arrow_scan"]["seconds"], "max"),
        ("database_mib_per_second", measurements["database_extract"]["mib_per_second"], "min"),
        ("parquet_mib_per_second", measurements["parquet_conversion"]["mib_per_second"], "min"),
        (
            "peak_incremental_process_mib",
            measurements["memory"]["peak_incremental_process_mib"], "max",
        ),
        ("local_workflow_seconds", measurements["local_workflow"]["seconds"], "max"),
    )
    for name, actual, direction in checks:
        limit = limits[name]
        failed = actual > limit if direction == "max" else actual < limit
        if failed:
            violations.append({
                "metric": name, "actual": actual, "limit": limit,
                "direction": direction,
            })
    return violations


def write_performance_qualification(
    root: Path, output: Path, *, rows: int = 100_000,
) -> dict[str, Any]:
    report = run_performance_qualification(root, rows=rows)
    from sift.reliability import atomic_write_json
    atomic_write_json(output, report)
    return report


__all__ = [
    "DEFAULT_BUDGETS", "PerformanceBudgets", "create_representative_fixtures",
    "evaluate_performance_budgets",
    "measure_startup", "run_performance_qualification",
    "write_performance_qualification",
]
