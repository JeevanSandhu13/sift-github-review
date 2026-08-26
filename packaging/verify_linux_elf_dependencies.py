"""Fail a Linux release when any bundled ELF has an unresolved dependency.

Checking only the launcher misses native modules and Qt plugins that are
loaded later.  This walks the complete frozen tree and asks the platform
loader to resolve every executable/shared object before the archive ships.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ELF_MAGIC = b"\x7fELF"


def _is_elf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == ELF_MAGIC
    except OSError:
        return False


def _unresolved(output: str) -> list[str]:
    return [line.strip() for line in output.splitlines() if "=> not found" in line]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        raise SystemExit("ELF dependency verification must run on Linux")

    bundle = args.bundle.resolve()
    if not bundle.is_dir():
        raise SystemExit(f"bundle is not a directory: {bundle}")

    checked = 0
    failures: list[str] = []
    seen: set[tuple[int, int]] = set()
    # PyInstaller's Linux bootloader prepends the bundle's _internal
    # directory to LD_LIBRARY_PATH before loading Python or any extension.
    # Reproduce that runtime contract for each direct ldd probe; otherwise
    # wheels' content-addressed libraries are falsely reported missing even
    # though the bootloader resolves them from the frozen bundle root.
    internal = bundle / "_internal"
    loader_paths = [str(internal)] if internal.is_dir() else []
    inherited_loader_path = os.environ.get("LD_LIBRARY_PATH")
    if inherited_loader_path:
        loader_paths.append(inherited_loader_path)
    loader_environment = {
        **os.environ,
        "LC_ALL": "C",
        "LD_LIBRARY_PATH": os.pathsep.join(loader_paths),
    }
    for candidate in sorted(bundle.rglob("*")):
        if not candidate.is_file() or not _is_elf(candidate):
            continue
        try:
            stat = candidate.stat()
        except OSError as exc:
            failures.append(f"{candidate.relative_to(bundle)}: {type(exc).__name__}")
            continue
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen:
            continue
        seen.add(identity)
        checked += 1
        completed = subprocess.run(
            ["ldd", str(candidate)],
            capture_output=True,
            text=True,
            timeout=30,
            env=loader_environment,
            check=False,
        )
        output = completed.stdout + completed.stderr
        missing = _unresolved(output)
        static = "statically linked" in output or "not a dynamic executable" in output
        if missing:
            failures.append(
                f"{candidate.relative_to(bundle)}: " + "; ".join(missing)
            )
        elif completed.returncode != 0 and not static:
            detail = " ".join(output.split())[:300] or f"ldd exit {completed.returncode}"
            failures.append(f"{candidate.relative_to(bundle)}: {detail}")

    if checked == 0:
        raise SystemExit("no ELF files were found in the frozen Linux bundle")
    if failures:
        print("Unresolved or unreadable Linux native dependencies:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print(f"Linux native dependency closure verified for {checked} ELF files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
