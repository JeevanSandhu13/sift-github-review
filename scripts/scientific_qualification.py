#!/usr/bin/env python3
"""Run Sift's deterministic scientific correctness and privacy release gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sift.evaluation import write_scientific_qualification


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "dist" / "evaluation" / "qualification.json",
    )
    parser.add_argument(
        "--agent-report", type=Path,
        help=(
            "Optional provider-evaluation JSON produced by "
            "scripts/provider_qualification.py. Credentials and raw responses "
            "are never read from this artifact."
        ),
    )
    args = parser.parse_args()
    agent_report = None
    if args.agent_report is not None:
        agent_report = json.loads(args.agent_report.read_text(encoding="utf-8"))
    report = write_scientific_qualification(
        args.output.parent, args.output, agent_report=agent_report,
    )
    print(f"Scientific qualification: {report['status']}")
    print(
        "Confidential release qualification: "
        f"{report['confidential_release_gate']['status']}"
    )
    for check in report["checks"]:
        print(f"[{check['status'].upper()}] {check['id']}: {check['detail']}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
