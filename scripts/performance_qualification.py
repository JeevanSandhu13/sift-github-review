#!/usr/bin/env python3
"""Run Sift's deterministic local performance release gate."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sift.performance import write_performance_qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "dist" / "performance" / "qualification.json",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="sift-performance-") as directory:
        report = write_performance_qualification(
            Path(directory), args.output, rows=args.rows,
        )
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
