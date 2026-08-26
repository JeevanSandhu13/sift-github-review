#!/usr/bin/env python3
"""Atomically refresh external Sift web assets in a frozen release archive.

This is intentionally limited to the two browser-rendered files that remain
external in PyInstaller's one-directory bundle.  Executables, libraries,
metadata, permissions, and all other archive members are copied unchanged.
The original artifact is replaced only after the new archive has been opened
again and both embedded files match the source tree byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
from pathlib import Path
import shutil
import tarfile
import zipfile


ASSET_NAMES = ("app.js", "desktop-shell.css")


def _source_assets(repo_root: Path) -> dict[str, bytes]:
    web_root = repo_root / "src" / "sift" / "web"
    return {name: (web_root / name).read_bytes() for name in ASSET_NAMES}


def _asset_name(member_name: str) -> str | None:
    normalized = member_name.replace("\\", "/")
    for name in ASSET_NAMES:
        if normalized.endswith(f"/_internal/sift/web/{name}"):
            return name
    return None


def _refresh_tar(archive: Path, temporary: Path, assets: dict[str, bytes]) -> None:
    found = {name: 0 for name in ASSET_NAMES}
    with tarfile.open(str(archive), "r|gz") as source, tarfile.open(
        str(temporary), "w|gz", compresslevel=6
    ) as output:
        for member in source:
            asset = _asset_name(member.name)
            if asset is not None:
                if not member.isfile():
                    raise RuntimeError(f"expected a regular file: {member.name}")
                found[asset] += 1
                payload = assets[asset]
                member.size = len(payload)
                output.addfile(member, io.BytesIO(payload))
                continue
            file_object = source.extractfile(member) if member.isfile() else None
            output.addfile(member, file_object)
    _require_exact_members(found)


def _refresh_zip(archive: Path, temporary: Path, assets: dict[str, bytes]) -> None:
    found = {name: 0 for name in ASSET_NAMES}
    with zipfile.ZipFile(archive, "r") as source, zipfile.ZipFile(
        temporary, "w", allowZip64=True
    ) as output:
        for info in source.infolist():
            asset = _asset_name(info.filename)
            if asset is not None:
                found[asset] += 1
                output.writestr(info, assets[asset])
                continue
            if info.is_dir():
                output.writestr(info, b"")
                continue
            with source.open(info, "r") as input_handle, output.open(
                info, "w", force_zip64=True
            ) as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
    _require_exact_members(found)


def _require_exact_members(found: dict[str, int]) -> None:
    invalid = {name: count for name, count in found.items() if count != 1}
    if invalid:
        raise RuntimeError(f"archive asset surface is incomplete or ambiguous: {invalid}")


def _embedded_assets(archive: Path, archive_format: str) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    if archive_format == "tar.gz":
        with tarfile.open(archive, "r:gz") as source:
            for member in source.getmembers():
                asset = _asset_name(member.name)
                if asset is not None:
                    handle = source.extractfile(member)
                    if handle is None:
                        raise RuntimeError(f"could not read {member.name}")
                    if asset in found:
                        raise RuntimeError(f"duplicate embedded asset: {asset}")
                    found[asset] = handle.read()
    elif archive_format == "zip":
        with zipfile.ZipFile(archive, "r") as source:
            for info in source.infolist():
                asset = _asset_name(info.filename)
                if asset is not None:
                    if asset in found:
                        raise RuntimeError(f"duplicate embedded asset: {asset}")
                    found[asset] = source.read(info)
    else:
        raise ValueError("supported artifacts are .tar.gz and .zip archives")
    return found


def refresh(archive: Path, repo_root: Path) -> None:
    archive = archive.resolve()
    assets = _source_assets(repo_root.resolve())
    temporary = archive.with_name(f".{archive.name}.refreshing")
    temporary.unlink(missing_ok=True)
    try:
        if archive.name.endswith(".tar.gz"):
            archive_format = "tar.gz"
            _refresh_tar(archive, temporary, assets)
        elif archive.suffix.lower() == ".zip":
            archive_format = "zip"
            _refresh_zip(archive, temporary, assets)
        else:
            raise ValueError("supported artifacts are .tar.gz and .zip archives")

        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        embedded = _embedded_assets(temporary, archive_format)
        _require_exact_members({name: int(name in embedded) for name in ASSET_NAMES})
        for name, expected in assets.items():
            actual = embedded[name]
            if actual != expected:
                raise RuntimeError(f"refreshed asset does not match source: {name}")
        os.replace(temporary, archive)
    finally:
        temporary.unlink(missing_ok=True)

    digest_builder = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    print(f"Refreshed {archive.name}: sha256={digest} size={archive.stat().st_size}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    refresh(args.archive, args.repo_root)


if __name__ == "__main__":
    main()
