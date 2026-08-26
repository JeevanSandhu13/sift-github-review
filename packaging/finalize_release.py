"""Assemble and self-verify the signed all-platform release manifest."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

from sift.release_manifest import (
    _load_json_file,
    load_trusted_json,
    main as release_manifest_main,
    verify_release,
)


LINUX_ARTIFACT_FILENAMES = {
    "x86_64": "Sift-Linux-x86_64.tar.gz",
    "aarch64": "Sift-Linux-aarch64.tar.gz",
}


def linux_artifact_arguments(
    artifact_dir: Path, architectures: list[str] | None,
) -> list[str]:
    """Return manifest arguments for every selected Linux release target.

    A production release defaults to both supported Linux architectures.
    ``--linux-architecture`` remains repeatable so a deliberately partial
    development release can still be assembled without weakening the normal
    all-platform path.
    """
    selected = architectures or list(LINUX_ARTIFACT_FILENAMES)
    # Preserve command-line order while rejecting duplicate target entries
    # before the signed-manifest validator has to report them.
    selected = list(dict.fromkeys(selected))
    arguments: list[str] = []
    for architecture in selected:
        filename = LINUX_ARTIFACT_FILENAMES[architecture]
        arguments.extend((
            "--artifact",
            f"linux,{architecture},{artifact_dir / filename},application/gzip",
        ))
    return arguments


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("dist"))
    parser.add_argument("--minimum-supported-version", required=True)
    parser.add_argument("--channel", choices=("stable", "beta"), default="stable")
    parser.add_argument(
        "--linux-architecture",
        dest="linux_architectures",
        action="append",
        choices=tuple(LINUX_ARTIFACT_FILENAMES),
        help=(
            "Linux target to publish; repeat to select more than one. "
            "Defaults to both x86_64 and aarch64."
        ),
    )
    parser.add_argument("--published-at")
    parser.add_argument("--trust-store", type=Path, required=True)
    parser.add_argument("--trust-store-sha256", required=True)
    parser.add_argument("--rollback-from", action="append", default=[])
    parser.add_argument("--rollback-expires-at")
    parser.add_argument("--rollback-reason", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not os.environ.get("SIFT_RELEASE_PRIVATE_KEY_B64"):
        raise SystemExit("SIFT_RELEASE_PRIVATE_KEY_B64 is required")
    key_id = os.environ.get("SIFT_RELEASE_KEY_ID", "")
    if not key_id:
        raise SystemExit("SIFT_RELEASE_KEY_ID is required")

    with Path("pyproject.toml").open("rb") as handle:
        version = tomllib.load(handle)["project"]["version"]
    published_at = args.published_at or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output = args.output or args.artifact_dir / f"release-manifest-{args.channel}.json"
    command = [
        "create",
        "--version", version,
        "--channel", args.channel,
        "--minimum-supported-version", args.minimum_supported_version,
        "--published-at", published_at,
        "--release-id", f"sift-{version.replace('.', '-')}-{args.channel}",
        "--key-id", key_id,
        "--artifact", (
            f"macos,arm64,{args.artifact_dir / 'Sift.dmg'},"
            "application/x-apple-diskimage"
        ),
        "--artifact", (
            f"windows,x86_64,{args.artifact_dir / 'Sift-Windows-x64-Setup.exe'},"
            "application/vnd.microsoft.portable-executable"
        ),
        "--output", str(output),
    ]
    # Insert artifact options before --output so the command remains easy to
    # inspect and mirrors the release-manifest CLI's documented shape.
    output_arguments = command[-2:]
    del command[-2:]
    command.extend(linux_artifact_arguments(
        args.artifact_dir, args.linux_architectures,
    ))
    command.extend(output_arguments)
    for rollback_version in args.rollback_from:
        command.extend(("--rollback-from", rollback_version))
    if args.rollback_expires_at:
        command.extend(("--rollback-expires-at", args.rollback_expires_at))
    if args.rollback_reason:
        command.extend(("--rollback-reason", args.rollback_reason))
    release_manifest_main(command)

    manifest = _load_json_file(output, "manifest", require_canonical=True)
    trust = load_trusted_json(args.trust_store, args.trust_store_sha256)
    verify_release(
        manifest,
        trust,
        args.artifact_dir,
        expected_channel=args.channel,
        installed_version=args.minimum_supported_version,
        highest_seen_version=args.minimum_supported_version,
        now=datetime.now(timezone.utc),
    )
    print(f"Created and offline-verified {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
