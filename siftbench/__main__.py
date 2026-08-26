"""Run SiftBench from the command line: ``python -m siftbench``.

Runs every seed case in a fresh temp directory and prints a small
report to stdout. Exits non-zero if any case fails, so it's usable
as a CI gate as well as an interactive check.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from siftbench.cases import SEED_CASES
from siftbench.runner import run_all, summarize


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="siftbench-") as tmp:
        runs = run_all(SEED_CASES, Path(tmp))
        report = summarize(runs)

    print(f"SiftBench: {report['passed']}/{report['total']} passed "
          f"(score {report['score']:.2f})\n")
    for c in report["cases"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['id']}: {c['message']}")
    print()
    print(json.dumps(report, indent=2))

    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    sys.exit(main())
