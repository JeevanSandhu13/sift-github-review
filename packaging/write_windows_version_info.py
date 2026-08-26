"""Generate deterministic Windows executable version resources for PyInstaller."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", args.version)
    if match is None:
        parser.error("version must contain three numeric components")
    components = tuple(int(value) for value in match.groups()) + (0,)
    if any(value > 65_535 for value in components):
        parser.error("Windows version components must not exceed 65535")
    numeric = ".".join(str(value) for value in components)
    text = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={components!r},
    prodvers={components!r},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'Sapien Institute'),
        StringStruct('FileDescription', 'Sift research assistant'),
        StringStruct('FileVersion', '{numeric}'),
        StringStruct('InternalName', 'Sift'),
        StringStruct('LegalCopyright', 'Copyright (C) 2026 Sapien Institute'),
        StringStruct('OriginalFilename', 'Sift.exe'),
        StringStruct('ProductName', 'Sift'),
        StringStruct('ProductVersion', '{args.version}')
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
