"""Fail a release build that ships upstream test or benchmark trees."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


EXCLUDED_DIRECTORIES = frozenset({
    "test", "tests", "testing", "benchmark", "benchmarks", "__pycache__",
})


def prohibited_bundle_entries(root: Path) -> list[str]:
    """Return bounded, relative QA-only entries found in a frozen bundle."""
    base = Path(root).resolve()
    if not base.is_dir() or base.is_symlink():
        raise ValueError("frozen bundle must be a real directory")
    findings: list[str] = []
    for directory, names, files in os.walk(base, topdown=True, followlinks=False):
        current = Path(directory)
        rejected = [
            name for name in names if name.casefold() in EXCLUDED_DIRECTORIES
        ]
        for name in rejected:
            findings.append((current / name).relative_to(base).as_posix() + "/")
        names[:] = [name for name in names if name not in rejected]
        for name in files:
            folded = name.casefold()
            if folded == "conftest.py":
                findings.append((current / name).relative_to(base).as_posix())
        if len(findings) >= 100:
            return sorted(findings[:100])
    return sorted(findings)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        findings = prohibited_bundle_entries(args.bundle)
    except ValueError as exc:
        parser.error(str(exc))
    if findings:
        print("Frozen bundle contains prohibited upstream QA artifacts:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print("Frozen bundle contains no upstream test or benchmark trees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
