"""Canonical, offline-verifiable release manifests for every Sift platform.

This module is the small trust boundary shared by the release pipeline and
the update client: strict schema validation, artifact/SBOM hashing, Ed25519
signatures, channel separation, key validity/revocation, and explicit
rollback authority. Network access is neither used nor required here.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import re
import stat
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_FORMAT = "sift-release-manifest"
TRUST_STORE_FORMAT = "sift-release-trust-store"
FILE_SIGNATURE_FORMAT = "sift-release-file-signature"
SCHEMA_VERSION = 1
PLATFORMS = frozenset({"macos", "windows", "linux"})
CHANNELS = frozenset({"stable", "beta"})
ARCHITECTURES = {
    "macos": frozenset({"arm64", "x86_64", "universal2"}),
    "windows": frozenset({"x86_64", "arm64"}),
    "linux": frozenset({"x86_64", "aarch64"}),
}
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SEMVER = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(?:-([0-9A-Za-z.-]+))?$"
)
_RFC3339_UTC = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")

_PAYLOAD_FIELDS = frozenset({
    "format", "schema_version", "release_id", "version", "channel",
    "published_at", "minimum_supported_version", "rollback", "artifacts",
    "signing_key_id", "signature_algorithm",
})
_MANIFEST_FIELDS = _PAYLOAD_FIELDS | {"signature"}
_ROLLBACK_FIELDS = frozenset({
    "allowed", "from_versions", "expires_at", "reason",
})
_ARTIFACT_FIELDS = frozenset({
    "platform", "architecture", "filename", "media_type", "size",
    "sha256", "sbom",
})
_SBOM_FIELDS = frozenset({"filename", "format", "size", "sha256"})
_TRUST_FIELDS = frozenset({"format", "schema_version", "keys"})
_TRUST_KEY_FIELDS = frozenset({
    "key_id", "algorithm", "public_key", "valid_from", "valid_until",
    "channels", "revoked_at",
})
_FILE_SIGNATURE_PAYLOAD_FIELDS = frozenset({
    "format", "schema_version", "filename", "size", "sha256", "version",
    "channel", "signed_at", "signing_key_id", "signature_algorithm",
})
_FILE_SIGNATURE_FIELDS = _FILE_SIGNATURE_PAYLOAD_FIELDS | {"signature"}


class ReleaseManifestError(ValueError):
    """A release document or local artifact failed a trust check."""


def canonical_json(value: Any) -> bytes:
    """Return the one signed JSON representation; reject NaN/Infinity."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseManifestError(f"document is not canonical JSON: {exc}") from exc


def _strict_json_loads(raw: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseManifestError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseManifestError(f"{label} is invalid JSON") from exc


def _exact_fields(value: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ReleaseManifestError(f"{label} fields do not match schema")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not _RFC3339_UTC.fullmatch(value):
        raise ReleaseManifestError(f"{label} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseManifestError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReleaseManifestError(f"{label} must use UTC")
    return parsed


def _version(value: Any, label: str) -> tuple[int, int, int, str | None]:
    if not isinstance(value, str):
        raise ReleaseManifestError(f"{label} must be a semantic version")
    match = _SEMVER.fullmatch(value)
    if not match:
        raise ReleaseManifestError(f"{label} must be a semantic version")
    return int(match[1]), int(match[2]), int(match[3]), match[4]


def _version_key(value: str) -> tuple[int, int, int, int, tuple[tuple[int, Any], ...]]:
    major, minor, patch, prerelease = _version(value, "version")
    if prerelease is None:
        return major, minor, patch, 1, ()
    identifiers: list[tuple[int, Any]] = []
    for item in prerelease.split("."):
        identifiers.append((0, int(item)) if item.isdigit() else (1, item))
    return major, minor, patch, 0, tuple(identifiers)


def sha256_file(path: Path) -> tuple[str, int]:
    """Hash a regular non-symlink file using a no-follow open where available."""
    target = Path(path)
    # Windows defaults low-level descriptors to text mode unless O_BINARY is
    # explicit. Without it, os.read treats byte 0x1A as EOF and can sign/hash
    # only a tiny prefix of an EXE or ZIP while reporting success.
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        before = target.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise ReleaseManifestError(f"artifact is a symlink: {target.name}")
        fd = os.open(target, flags)
    except OSError as exc:
        raise ReleaseManifestError(f"artifact is unreadable: {target.name}") from exc
    digest = hashlib.sha256()
    size = 0
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ReleaseManifestError(f"artifact is not a regular file: {target.name}")
        if before.st_dev != opened.st_dev or before.st_ino != opened.st_ino:
            raise ReleaseManifestError(f"artifact changed while opening: {target.name}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    finally:
        os.close(fd)
    return digest.hexdigest(), size


def _read_regular_file(path: Path) -> bytes:
    target = Path(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        before = target.lstat()
        if stat.S_ISLNK(before.st_mode):
            raise ReleaseManifestError(f"file is a symlink: {target.name}")
        fd = os.open(target, flags)
    except OSError as exc:
        raise ReleaseManifestError(f"file is unreadable: {target.name}") from exc
    chunks: list[bytes] = []
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ReleaseManifestError(f"file is not regular: {target.name}")
        if before.st_dev != opened.st_dev or before.st_ino != opened.st_ino:
            raise ReleaseManifestError(f"file changed while opening: {target.name}")
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(fd)
    return b"".join(chunks)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".sift-release-", dir=destination.parent)
    try:
        remaining = memoryview(canonical_json(value) + b"\n")
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                raise OSError("short write while creating release JSON")
            remaining = remaining[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp_name, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def _safe_filename(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or Path(value).name != value:
        raise ReleaseManifestError(f"{label} must be a plain filename")
    if "/" in value or "\\" in value or value in {".", ".."}:
        raise ReleaseManifestError(f"{label} contains a path")
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    doc = _exact_fields(manifest, _MANIFEST_FIELDS, "manifest")
    if doc["format"] != MANIFEST_FORMAT or doc["schema_version"] != SCHEMA_VERSION:
        raise ReleaseManifestError("unsupported release manifest format")
    if not isinstance(doc["release_id"], str) or not _KEY_ID.fullmatch(doc["release_id"]):
        raise ReleaseManifestError("release_id is invalid")
    target_version = _version(doc["version"], "version")
    if doc["channel"] not in CHANNELS:
        raise ReleaseManifestError("channel is invalid")
    if doc["channel"] == "stable" and target_version[3] is not None:
        raise ReleaseManifestError("stable channel cannot publish a prerelease version")
    _timestamp(doc["published_at"], "published_at")
    _version(doc["minimum_supported_version"], "minimum_supported_version")
    if _version_key(doc["minimum_supported_version"]) > _version_key(doc["version"]):
        raise ReleaseManifestError("minimum_supported_version exceeds release version")
    if not isinstance(doc["signing_key_id"], str) or not _KEY_ID.fullmatch(doc["signing_key_id"]):
        raise ReleaseManifestError("signing_key_id is invalid")
    if doc["signature_algorithm"] != "Ed25519":
        raise ReleaseManifestError("unsupported signature algorithm")
    try:
        signature = base64.b64decode(doc["signature"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ReleaseManifestError("signature is not valid base64") from exc
    if len(signature) != 64:
        raise ReleaseManifestError("Ed25519 signature must be 64 bytes")

    rollback = _exact_fields(doc["rollback"], _ROLLBACK_FIELDS, "rollback")
    if not isinstance(rollback["allowed"], bool):
        raise ReleaseManifestError("rollback.allowed must be boolean")
    if not isinstance(rollback["from_versions"], list) or not all(
        isinstance(item, str) and _SEMVER.fullmatch(item)
        for item in rollback["from_versions"]
    ):
        raise ReleaseManifestError("rollback.from_versions is invalid")
    if len(set(rollback["from_versions"])) != len(rollback["from_versions"]):
        raise ReleaseManifestError("rollback.from_versions contains duplicates")
    if rollback["allowed"]:
        if (
            not rollback["from_versions"]
            or not isinstance(rollback["reason"], str)
            or not rollback["reason"].strip()
        ):
            raise ReleaseManifestError("authorized rollback requires sources and a reason")
        _timestamp(rollback["expires_at"], "rollback.expires_at")
    elif rollback != {
        "allowed": False, "from_versions": [], "expires_at": None, "reason": "",
    }:
        raise ReleaseManifestError("disabled rollback metadata must be empty")

    artifacts = doc["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ReleaseManifestError("artifacts must be a non-empty list")
    seen_targets: set[tuple[str, str]] = set()
    seen_names: set[str] = set()
    covered: set[str] = set()
    for index, raw in enumerate(artifacts):
        artifact = _exact_fields(raw, _ARTIFACT_FIELDS, f"artifact[{index}]")
        platform_name = artifact["platform"]
        if platform_name not in PLATFORMS:
            raise ReleaseManifestError("artifact platform is invalid")
        architecture = artifact["architecture"]
        if architecture not in ARCHITECTURES[platform_name]:
            raise ReleaseManifestError("artifact architecture is invalid")
        target = (platform_name, architecture)
        if target in seen_targets:
            raise ReleaseManifestError("duplicate platform/architecture artifact")
        seen_targets.add(target)
        filename = _safe_filename(artifact["filename"], "artifact filename")
        if filename in seen_names:
            raise ReleaseManifestError("duplicate artifact filename")
        seen_names.add(filename)
        if not isinstance(artifact["media_type"], str) or "/" not in artifact["media_type"]:
            raise ReleaseManifestError("artifact media_type is invalid")
        if (
            not isinstance(artifact["size"], int)
            or isinstance(artifact["size"], bool)
            or artifact["size"] <= 0
        ):
            raise ReleaseManifestError("artifact size is invalid")
        if not isinstance(artifact["sha256"], str) or not _HEX64.fullmatch(artifact["sha256"]):
            raise ReleaseManifestError("artifact sha256 is invalid")
        sbom = _exact_fields(artifact["sbom"], _SBOM_FIELDS, "artifact sbom")
        sbom_filename = _safe_filename(sbom["filename"], "SBOM filename")
        if sbom_filename in seen_names:
            raise ReleaseManifestError("duplicate artifact/SBOM filename")
        seen_names.add(sbom_filename)
        if sbom["format"] != "cyclonedx-json":
            raise ReleaseManifestError("unsupported SBOM format")
        if (
            not isinstance(sbom["size"], int)
            or isinstance(sbom["size"], bool)
            or sbom["size"] <= 0
        ):
            raise ReleaseManifestError("SBOM size is invalid")
        if not isinstance(sbom["sha256"], str) or not _HEX64.fullmatch(sbom["sha256"]):
            raise ReleaseManifestError("SBOM sha256 is invalid")
        covered.add(platform_name)
    if covered != PLATFORMS:
        raise ReleaseManifestError("manifest must cover macOS, Windows, and Linux")


def sign_manifest(payload: Mapping[str, Any], private_key_b64: str) -> dict[str, Any]:
    if set(payload) != _PAYLOAD_FIELDS:
        raise ReleaseManifestError("unsigned manifest payload fields do not match schema")
    unsigned = dict(payload)
    # Schema validation also checks all non-signature data. A correctly-sized
    # placeholder is used only during validation and is never returned.
    validate_manifest({**unsigned, "signature": base64.b64encode(b"\0" * 64).decode("ascii")})
    try:
        raw = base64.b64decode(private_key_b64, validate=True)
        if len(raw) != 32:
            raise ValueError
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        signature = Ed25519PrivateKey.from_private_bytes(raw).sign(canonical_json(unsigned))
    except Exception as exc:  # noqa: BLE001 — normalize all key failures
        raise ReleaseManifestError("release signing key is invalid") from exc
    return {**unsigned, "signature": base64.b64encode(signature).decode("ascii")}


def validate_trust_store(store: Mapping[str, Any]) -> None:
    doc = _exact_fields(store, _TRUST_FIELDS, "trust store")
    if doc["format"] != TRUST_STORE_FORMAT or doc["schema_version"] != SCHEMA_VERSION:
        raise ReleaseManifestError("unsupported trust store format")
    if not isinstance(doc["keys"], list) or not doc["keys"]:
        raise ReleaseManifestError("trust store has no keys")
    seen: set[str] = set()
    for raw in doc["keys"]:
        key = _exact_fields(raw, _TRUST_KEY_FIELDS, "trust key")
        key_id = key["key_id"]
        if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id) or key_id in seen:
            raise ReleaseManifestError("trust key id is invalid or duplicated")
        seen.add(key_id)
        if key["algorithm"] != "Ed25519":
            raise ReleaseManifestError("trust key algorithm is unsupported")
        try:
            public = base64.b64decode(key["public_key"], validate=True)
        except (TypeError, ValueError) as exc:
            raise ReleaseManifestError("trust public key is invalid") from exc
        if len(public) != 32:
            raise ReleaseManifestError("Ed25519 public key must be 32 bytes")
        valid_from = _timestamp(key["valid_from"], "key.valid_from")
        valid_until = _timestamp(key["valid_until"], "key.valid_until")
        if valid_until <= valid_from:
            raise ReleaseManifestError("trust key validity window is invalid")
        if (
            not isinstance(key["channels"], list)
            or not key["channels"]
            or len(set(key["channels"])) != len(key["channels"])
            or not set(key["channels"]).issubset(CHANNELS)
        ):
            raise ReleaseManifestError("trust key channels are invalid")
        if key["revoked_at"] is not None:
            _timestamp(key["revoked_at"], "key.revoked_at")


def verify_manifest_signature(
    manifest: Mapping[str, Any], trust_store: Mapping[str, Any],
) -> None:
    validate_manifest(manifest)
    validate_trust_store(trust_store)
    candidates = [
        key for key in trust_store["keys"]
        if key["key_id"] == manifest["signing_key_id"]
    ]
    if len(candidates) != 1:
        raise ReleaseManifestError("signing key is not trusted")
    key = candidates[0]
    if key["revoked_at"] is not None:
        raise ReleaseManifestError("signing key is revoked")
    if manifest["channel"] not in key["channels"]:
        raise ReleaseManifestError("signing key is not trusted for this channel")
    published = _timestamp(manifest["published_at"], "published_at")
    if not (
        _timestamp(key["valid_from"], "key.valid_from")
        <= published
        <= _timestamp(key["valid_until"], "key.valid_until")
    ):
        raise ReleaseManifestError("manifest was signed outside the key validity window")
    unsigned = {name: manifest[name] for name in _PAYLOAD_FIELDS}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(
            base64.b64decode(key["public_key"], validate=True)
        ).verify(
            base64.b64decode(manifest["signature"], validate=True),
            canonical_json(unsigned),
        )
    except Exception as exc:  # noqa: BLE001
        raise ReleaseManifestError("release manifest signature is invalid") from exc


def verify_release_policy(
    manifest: Mapping[str, Any],
    trust_store: Mapping[str, Any],
    *,
    expected_channel: str,
    installed_version: str,
    highest_seen_version: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify trust plus update/rollback policy without opening artifacts.

    ``highest_seen_version`` is only a rollback defense when the caller reads
    it from replacement-resistant local state (for example an administrator-
    protected file or OS credential-vault entry) and persists the returned
    version after success. A value recovered from an ordinary user-writable
    preferences file must not be represented as a security boundary.
    """
    verify_manifest_signature(manifest, trust_store)
    current = now or datetime.now(timezone.utc)
    published = _timestamp(manifest["published_at"], "published_at")
    if published > current + timedelta(minutes=5):
        raise ReleaseManifestError("manifest publication time is too far in the future")
    if expected_channel not in CHANNELS or manifest["channel"] != expected_channel:
        raise ReleaseManifestError("release channel does not match configured channel")
    _version(installed_version, "installed_version")
    if _version_key(installed_version) < _version_key(manifest["minimum_supported_version"]):
        raise ReleaseManifestError(
            "installed version is below the manifest's supported update floor"
        )
    comparison_floor = installed_version
    if highest_seen_version is not None:
        _version(highest_seen_version, "highest_seen_version")
        if _version_key(highest_seen_version) > _version_key(comparison_floor):
            comparison_floor = highest_seen_version
    if _version_key(manifest["version"]) < _version_key(comparison_floor):
        rollback = manifest["rollback"]
        if (
            not rollback["allowed"]
            or installed_version not in rollback["from_versions"]
            or current > _timestamp(rollback["expires_at"], "rollback.expires_at")
        ):
            raise ReleaseManifestError("release would be an unauthorized rollback")

    return {
        "ok": True,
        "version": manifest["version"],
        "channel": manifest["channel"],
        "signing_key_id": manifest["signing_key_id"],
        "highest_seen_version_to_persist": max(
            filter(None, (highest_seen_version, manifest["version"])),
            key=_version_key,
        ),
    }


def verify_release(
    manifest: Mapping[str, Any],
    trust_store: Mapping[str, Any],
    artifact_dir: Path,
    *,
    expected_channel: str,
    installed_version: str,
    highest_seen_version: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Verify trust, update/rollback policy, and every local artifact/SBOM."""
    result = verify_release_policy(
        manifest,
        trust_store,
        expected_channel=expected_channel,
        installed_version=installed_version,
        highest_seen_version=highest_seen_version,
        now=now,
    )
    root = Path(artifact_dir)
    verified: list[str] = []
    for artifact in manifest["artifacts"]:
        for descriptor in (artifact, artifact["sbom"]):
            path = root / descriptor["filename"]
            digest, size = sha256_file(path)
            if size != descriptor["size"] or digest != descriptor["sha256"]:
                raise ReleaseManifestError(f"artifact hash/size mismatch: {path.name}")
            verified.append(path.name)
    return {**result, "verified_files": verified}


def load_trusted_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    """Load a local trust store only when its independently pinned hash matches."""
    if not _HEX64.fullmatch(expected_sha256 or ""):
        raise ReleaseManifestError("a valid pinned trust-store SHA-256 is required")
    raw = _read_regular_file(path)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ReleaseManifestError("trust-store SHA-256 does not match the pinned value")
    value = _strict_json_loads(raw, "trust store")
    if not isinstance(value, Mapping):
        raise ReleaseManifestError("trust store must be a JSON object")
    validate_trust_store(value)
    return value


def validate_file_signature(statement: Mapping[str, Any]) -> None:
    doc = _exact_fields(statement, _FILE_SIGNATURE_FIELDS, "file signature")
    if doc["format"] != FILE_SIGNATURE_FORMAT or doc["schema_version"] != SCHEMA_VERSION:
        raise ReleaseManifestError("unsupported file signature format")
    _safe_filename(doc["filename"], "signed filename")
    if not isinstance(doc["size"], int) or isinstance(doc["size"], bool) or doc["size"] <= 0:
        raise ReleaseManifestError("signed file size is invalid")
    if not isinstance(doc["sha256"], str) or not _HEX64.fullmatch(doc["sha256"]):
        raise ReleaseManifestError("signed file sha256 is invalid")
    _version(doc["version"], "version")
    if doc["channel"] not in CHANNELS:
        raise ReleaseManifestError("signed file channel is invalid")
    _timestamp(doc["signed_at"], "signed_at")
    if not isinstance(doc["signing_key_id"], str) or not _KEY_ID.fullmatch(doc["signing_key_id"]):
        raise ReleaseManifestError("signed file key id is invalid")
    if doc["signature_algorithm"] != "Ed25519":
        raise ReleaseManifestError("signed file algorithm is unsupported")
    try:
        signature = base64.b64decode(doc["signature"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ReleaseManifestError("file signature is invalid base64") from exc
    if len(signature) != 64:
        raise ReleaseManifestError("file signature must be 64 bytes")


def sign_file(
    path: Path,
    *,
    version: str,
    channel: str,
    signed_at: str,
    key_id: str,
    private_key_b64: str,
) -> dict[str, Any]:
    digest, size = sha256_file(path)
    payload = {
        "format": FILE_SIGNATURE_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "filename": Path(path).name,
        "size": size,
        "sha256": digest,
        "version": version,
        "channel": channel,
        "signed_at": signed_at,
        "signing_key_id": key_id,
        "signature_algorithm": "Ed25519",
    }
    validate_file_signature({
        **payload, "signature": base64.b64encode(b"\0" * 64).decode("ascii"),
    })
    try:
        raw = base64.b64decode(private_key_b64, validate=True)
        if len(raw) != 32:
            raise ValueError
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        signature = Ed25519PrivateKey.from_private_bytes(raw).sign(canonical_json(payload))
    except Exception as exc:  # noqa: BLE001
        raise ReleaseManifestError("release signing key is invalid") from exc
    return {**payload, "signature": base64.b64encode(signature).decode("ascii")}


def verify_file(
    statement: Mapping[str, Any], path: Path, trust_store: Mapping[str, Any],
) -> dict[str, Any]:
    validate_file_signature(statement)
    validate_trust_store(trust_store)
    matches = [
        key for key in trust_store["keys"]
        if key["key_id"] == statement["signing_key_id"]
    ]
    if len(matches) != 1:
        raise ReleaseManifestError("file signing key is not trusted")
    key = matches[0]
    if key["revoked_at"] is not None:
        raise ReleaseManifestError("file signing key is revoked")
    if statement["channel"] not in key["channels"]:
        raise ReleaseManifestError("file signing key is not trusted for channel")
    signed_at = _timestamp(statement["signed_at"], "signed_at")
    if not (
        _timestamp(key["valid_from"], "key.valid_from")
        <= signed_at
        <= _timestamp(key["valid_until"], "key.valid_until")
    ):
        raise ReleaseManifestError("file was signed outside the key validity window")
    payload = {name: statement[name] for name in _FILE_SIGNATURE_PAYLOAD_FIELDS}
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        Ed25519PublicKey.from_public_bytes(
            base64.b64decode(key["public_key"], validate=True)
        ).verify(
            base64.b64decode(statement["signature"], validate=True),
            canonical_json(payload),
        )
    except Exception as exc:  # noqa: BLE001
        raise ReleaseManifestError("file signature is invalid") from exc
    digest, size = sha256_file(path)
    if (
        Path(path).name != statement["filename"]
        or digest != statement["sha256"]
        or size != statement["size"]
    ):
        raise ReleaseManifestError("signed file hash/size mismatch")
    return {"ok": True, "filename": statement["filename"], "sha256": digest}


def generate_cyclonedx_sbom(artifact: Path, version: str) -> dict[str, Any]:
    """Generate a deterministic dependency inventory for a built artifact."""
    digest, size = sha256_file(artifact)
    components = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name or not distribution.version:
            continue
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        components.append({
            "type": "library",
            "name": name,
            "version": distribution.version,
            "purl": f"pkg:pypi/{normalized}@{distribution.version}",
        })
    components.sort(key=lambda item: (item["name"].casefold(), item["version"]))
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:" + str(uuid.UUID(hex=hashlib.sha256(
            (artifact.name + digest).encode("utf-8")
        ).hexdigest()[:32])),
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": "Sift",
                "version": version,
                "properties": [
                    {"name": "sift:artifact:filename", "value": artifact.name},
                    {"name": "sift:artifact:size", "value": str(size)},
                ],
                "hashes": [{"alg": "SHA-256", "content": digest}],
            },
        },
        "components": components,
    }


def write_sbom(artifact: Path, output: Path, version: str) -> None:
    _write_json_atomic(output, generate_cyclonedx_sbom(artifact, version))


def verify_sbom_binding(artifact: Path, sbom_path: Path) -> dict[str, Any]:
    """Fail closed unless a CycloneDX SBOM binds the current artifact bytes."""
    sbom = _load_json_file(sbom_path, "SBOM", require_canonical=True)
    if sbom.get("bomFormat") != "CycloneDX" or sbom.get("specVersion") != "1.5":
        raise ReleaseManifestError("SBOM is not supported CycloneDX 1.5 JSON")
    metadata = sbom.get("metadata")
    component = metadata.get("component") if isinstance(metadata, Mapping) else None
    if not isinstance(component, Mapping):
        raise ReleaseManifestError("SBOM metadata component is missing")
    hashes = component.get("hashes")
    sha256 = None
    if isinstance(hashes, list):
        for entry in hashes:
            if isinstance(entry, Mapping) and entry.get("alg") == "SHA-256":
                sha256 = entry.get("content")
                break
    properties = component.get("properties")
    property_values: dict[str, Any] = {}
    if isinstance(properties, list):
        for entry in properties:
            if isinstance(entry, Mapping) and isinstance(entry.get("name"), str):
                property_values[str(entry["name"])] = entry.get("value")
    digest, size = sha256_file(artifact)
    if (
        sha256 != digest
        or property_values.get("sift:artifact:filename") != artifact.name
        or property_values.get("sift:artifact:size") != str(size)
    ):
        raise ReleaseManifestError("SBOM artifact hash/size/filename mismatch")
    return {"ok": True, "filename": artifact.name, "sha256": digest, "size": size}


def _load_json_file(
    path: Path, label: str, *, require_canonical: bool = False,
) -> Mapping[str, Any]:
    raw = _read_regular_file(path)
    value = _strict_json_loads(raw, label)
    if not isinstance(value, Mapping):
        raise ReleaseManifestError(f"{label} must be a JSON object")
    if require_canonical and raw not in {
        canonical_json(value), canonical_json(value) + b"\n",
    }:
        raise ReleaseManifestError(f"{label} is not canonical JSON")
    return value


def _artifact_descriptor(spec: str) -> dict[str, Any]:
    parts = spec.split(",", 3)
    if len(parts) != 4:
        raise ReleaseManifestError(
            "--artifact must be platform,architecture,path,media-type"
        )
    platform_name, architecture, raw_path, media_type = parts
    artifact = Path(raw_path)
    sbom_path = artifact.with_name(artifact.name + ".sbom.cdx.json")
    digest, size = sha256_file(artifact)
    sbom_digest, sbom_size = sha256_file(sbom_path)
    return {
        "platform": platform_name,
        "architecture": architecture,
        "filename": artifact.name,
        "media_type": media_type,
        "size": size,
        "sha256": digest,
        "sbom": {
            "filename": sbom_path.name,
            "format": "cyclonedx-json",
            "size": sbom_size,
            "sha256": sbom_digest,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sift-release-manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    sbom = sub.add_parser("sbom", help="write a CycloneDX dependency inventory")
    sbom.add_argument("artifact", type=Path)
    sbom.add_argument("output", type=Path)
    sbom.add_argument("--version", required=True)
    verify_sbom_parser = sub.add_parser(
        "verify-sbom", help="verify that a CycloneDX SBOM binds an artifact",
    )
    verify_sbom_parser.add_argument("artifact", type=Path)
    verify_sbom_parser.add_argument("sbom", type=Path)
    sign_file_parser = sub.add_parser(
        "sign-file", help="write a canonical detached Ed25519 signature",
    )
    sign_file_parser.add_argument("artifact", type=Path)
    sign_file_parser.add_argument("output", type=Path)
    sign_file_parser.add_argument("--version", required=True)
    sign_file_parser.add_argument("--channel", choices=sorted(CHANNELS), required=True)
    sign_file_parser.add_argument("--signed-at", required=True)
    sign_file_parser.add_argument("--key-id", required=True)
    create = sub.add_parser(
        "create", help="create the signed all-platform release manifest",
    )
    create.add_argument("--version", required=True)
    create.add_argument("--channel", choices=sorted(CHANNELS), required=True)
    create.add_argument("--minimum-supported-version", required=True)
    create.add_argument("--published-at", required=True)
    create.add_argument("--release-id", required=True)
    create.add_argument("--key-id", required=True)
    create.add_argument("--artifact", action="append", required=True)
    create.add_argument("--rollback-from", action="append", default=[])
    create.add_argument("--rollback-expires-at")
    create.add_argument("--rollback-reason", default="")
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="verify a release completely offline")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("trust_store", type=Path)
    verify.add_argument("artifact_dir", type=Path)
    verify.add_argument("--trust-store-sha256", required=True)
    verify.add_argument("--channel", choices=sorted(CHANNELS), required=True)
    verify.add_argument("--installed-version", required=True)
    verify.add_argument(
        "--highest-seen-version",
        help=(
            "highest version from replacement-resistant local/admin state; "
            "do not pass a value from ordinary user-writable preferences"
        ),
    )
    args = parser.parse_args(argv)
    if args.command == "sbom":
        _version(args.version, "version")
        write_sbom(args.artifact, args.output, args.version)
        return 0
    if args.command == "verify-sbom":
        print(json.dumps(verify_sbom_binding(args.artifact, args.sbom), sort_keys=True))
        return 0
    if args.command == "sign-file":
        private_key = os.environ.get("SIFT_RELEASE_PRIVATE_KEY_B64", "")
        if not private_key:
            raise ReleaseManifestError("SIFT_RELEASE_PRIVATE_KEY_B64 is required")
        statement = sign_file(
            args.artifact,
            version=args.version,
            channel=args.channel,
            signed_at=args.signed_at,
            key_id=args.key_id,
            private_key_b64=private_key,
        )
        _write_json_atomic(args.output, statement)
        return 0
    if args.command == "create":
        private_key = os.environ.get("SIFT_RELEASE_PRIVATE_KEY_B64", "")
        if not private_key:
            raise ReleaseManifestError("SIFT_RELEASE_PRIVATE_KEY_B64 is required")
        rollback_allowed = bool(args.rollback_from)
        if rollback_allowed and (
            not args.rollback_expires_at or not args.rollback_reason.strip()
        ):
            raise ReleaseManifestError(
                "rollback requires --rollback-expires-at and --rollback-reason"
            )
        payload = {
            "format": MANIFEST_FORMAT,
            "schema_version": SCHEMA_VERSION,
            "release_id": args.release_id,
            "version": args.version,
            "channel": args.channel,
            "published_at": args.published_at,
            "minimum_supported_version": args.minimum_supported_version,
            "rollback": {
                "allowed": rollback_allowed,
                "from_versions": args.rollback_from,
                "expires_at": args.rollback_expires_at if rollback_allowed else None,
                "reason": args.rollback_reason.strip() if rollback_allowed else "",
            },
            "artifacts": [_artifact_descriptor(spec) for spec in args.artifact],
            "signing_key_id": args.key_id,
            "signature_algorithm": "Ed25519",
        }
        _write_json_atomic(args.output, sign_manifest(payload, private_key))
        return 0
    if args.command == "verify":
        manifest = _load_json_file(
            args.manifest, "manifest", require_canonical=True,
        )
        trust_store = load_trusted_json(
            args.trust_store, args.trust_store_sha256,
        )
        result = verify_release(
            manifest,
            trust_store,
            args.artifact_dir,
            expected_channel=args.channel,
            installed_version=args.installed_version,
            highest_seen_version=args.highest_seen_version,
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    raise ReleaseManifestError("unknown command")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
