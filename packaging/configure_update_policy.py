#!/usr/bin/env python3
"""Create the build-embedded update policy from a reviewed trust store."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

from sift.release_manifest import canonical_json, load_trusted_json
from sift.update_service import _origin


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".sift-update-policy-", dir=path.parent)
    try:
        remaining = memoryview(value)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while creating update policy")
            remaining = remaining[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, path)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-url", required=True)
    parser.add_argument("--channel", choices=("stable", "beta"), default="stable")
    parser.add_argument("--trust-store", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("packaging/generated/update"),
    )
    args = parser.parse_args()
    _origin(args.manifest_url)
    raw = args.trust_store.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    load_trusted_json(args.trust_store, digest)
    policy = {
        "format": "sift-update-policy",
        "schema_version": 1,
        "enabled": True,
        "manifest_url": args.manifest_url,
        "channel": args.channel,
        "trust_store_filename": "release-trust-store.json",
        "trust_store_sha256": digest,
    }
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(output / "update-policy.json", canonical_json(policy) + b"\n")
    _atomic_bytes(output / "release-trust-store.json", raw)
    print(f"Configured signed {args.channel} updates in {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
