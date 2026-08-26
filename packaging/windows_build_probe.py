"""Stable machine-readable probes for the PowerShell Windows build.

Windows PowerShell 5.1 applies native-command quote processing that can alter
embedded Python ``-c`` source.  Keeping these probes in a file avoids that
shell boundary and lets both development and production builds use identical,
testable logic.
"""

from __future__ import annotations

import argparse
import json
import struct
import sysconfig
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def python_target() -> None:
    print(json.dumps({
        "platform": sysconfig.get_platform().lower(),
        "pointer_bits": struct.calcsize("P") * 8,
    }, sort_keys=True))


def project_version() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
        import tomli as tomllib  # type: ignore[no-redef]
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        value = tomllib.load(handle)["project"]["version"]
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("project version is missing")
    print(value.strip())


def production_update_policy() -> None:
    from sift.update_config import load_update_policy

    if load_update_policy().get("configured") is not True:
        raise SystemExit("production update policy is not configured")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=(
        "python-target", "project-version", "production-update-policy",
    ))
    args = parser.parse_args(argv)
    if args.probe == "python-target":
        python_target()
    elif args.probe == "project-version":
        project_version()
    else:
        production_update_policy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
