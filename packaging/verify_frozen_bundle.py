"""Verify the bounded surface and required runtimes of a frozen bundle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


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


def required_runtime_failures(root: Path) -> list[str]:
    """Return required frozen-runtime files that are absent or unusable."""
    base = Path(root).resolve()
    if not base.is_dir() or base.is_symlink():
        raise ValueError("frozen bundle must be a real directory")
    cli_name = "claude.exe" if os.name == "nt" else "claude"
    cli = base / "_internal" / "claude_agent_sdk" / "_bundled" / cli_name
    failures: list[str] = []
    if not cli.is_file():
        failures.append(f"missing Anthropic agent runtime: {cli.relative_to(base)}")
    elif os.name != "nt" and not os.access(cli, os.X_OK):
        failures.append(
            f"Anthropic agent runtime is not executable: {cli.relative_to(base)}"
        )
    return failures


def provider_runtime_probe_failures(root: Path) -> list[str]:
    """Start required provider runtimes without contacting provider APIs."""
    base = Path(root).resolve()
    if not base.is_dir() or base.is_symlink():
        raise ValueError("frozen bundle must be a real directory")
    cli_name = "claude.exe" if os.name == "nt" else "claude"
    cli = base / "_internal" / "claude_agent_sdk" / "_bundled" / cli_name
    if not cli.is_file():
        return []  # The presence check reports this with a clearer message.
    try:
        result = subprocess.run(
            [str(cli), "--version"],
            check=False,
            capture_output=True,
            text=True,
            # A first launch directly from a compressed macOS disk image can
            # spend tens of seconds in decompression and Gatekeeper scanning.
            # Installed copies normally respond in under a second, but the
            # release verifier must not mistake that one-time I/O cost for a
            # corrupt provider runtime.
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"Anthropic agent runtime could not start: {exc}"]
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        suffix = f": {detail}" if detail else ""
        return [
            "Anthropic agent runtime version probe failed "
            f"with exit code {result.returncode}{suffix}"
        ]
    if not (result.stdout or result.stderr).strip():
        return ["Anthropic agent runtime version probe returned no output"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    try:
        findings = prohibited_bundle_entries(args.bundle)
        runtime_failures = required_runtime_failures(args.bundle)
        if not runtime_failures:
            runtime_failures.extend(provider_runtime_probe_failures(args.bundle))
    except ValueError as exc:
        parser.error(str(exc))
    if findings:
        print("Frozen bundle contains prohibited upstream QA artifacts:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    if runtime_failures:
        print("Frozen bundle is missing required provider runtimes:")
        for failure in runtime_failures:
            print(f"- {failure}")
        return 1
    print(
        "Frozen bundle contains no upstream QA trees and includes required "
        "provider runtimes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
