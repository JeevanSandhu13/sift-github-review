#!/usr/bin/env python3
"""Manage Sift's publisher-owned release signature without exposing its key.

Platform signatures (Apple Developer ID and Windows Authenticode) establish the
publisher identity that Gatekeeper and SmartScreen display.  This separate
Ed25519 identity binds every Sift artifact, SBOM, and update manifest to one
offline-verifiable release authority on all three platforms.

The private key lives in the operating system credential vault.  Only the
public trust store is written into the source/build tree.  In particular, this
tool never prints the private key and never places it in a command line,
temporary file, or environment variable.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 support
    import tomli as tomllib  # type: ignore[no-redef]

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sift.release_manifest import (
    MANIFEST_FORMAT,
    TRUST_STORE_FORMAT,
    _artifact_descriptor,
    _load_json_file,
    _write_json_atomic,
    canonical_json,
    sign_file,
    sign_manifest,
    validate_trust_store,
    verify_file,
    verify_release,
    verify_sbom_binding,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRUST_STORE = REPOSITORY_ROOT / "packaging" / "release-trust-store.json"
KEYCHAIN_SERVICE = "org.sapieninstitute.sift.release-signing"
DEFAULT_KEY_ID = "sift-release-2026-01"
DEFAULT_VALID_FROM = "2026-08-26T00:00:00Z"
DEFAULT_VALID_UNTIL = "2031-08-26T00:00:00Z"

RELEASE_ARTIFACTS = (
    "Sift.dmg",
    "Sift-Windows-x64-Setup.exe",
    "Sift-Windows-x64.zip",
    "Sift-Linux-x86_64.tar.gz",
    "Sift-Linux-aarch64.tar.gz",
)


class CredentialStore(Protocol):
    def get_password(self, service_name: str, username: str) -> str | None: ...

    def set_password(self, service_name: str, username: str, password: str) -> None: ...


class MacOSSecurityCredentialStore:
    """Minimal Keychain adapter that keeps secrets out of argv and logs.

    The Security framework calls are deprecated for new GUI applications but
    remain the stable system API beneath the ``security`` command.  They are a
    better fit for this narrow release tool because ``security -w <value>``
    exposes the secret in the process table, while its prompt mode requires a
    real TTY and silently stored an empty password when driven by automation.
    """

    _SECURITY_FRAMEWORK = (
        "/System/Library/Frameworks/Security.framework/Security"
    )
    _CORE_FOUNDATION = (
        "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
    )
    _ITEM_NOT_FOUND = -25300

    def __init__(self) -> None:
        self._security = ctypes.CDLL(self._SECURITY_FRAMEWORK)
        self._core_foundation = ctypes.CDLL(self._CORE_FOUNDATION)
        self._security.SecKeychainFindGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecKeychainFindGenericPassword.restype = ctypes.c_int32
        self._security.SecKeychainAddGenericPassword.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p,
            ctypes.c_uint32, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._security.SecKeychainAddGenericPassword.restype = ctypes.c_int32
        self._security.SecKeychainItemFreeContent.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p,
        ]
        self._security.SecKeychainItemFreeContent.restype = ctypes.c_int32
        self._core_foundation.CFRelease.argtypes = [ctypes.c_void_p]
        self._core_foundation.CFRelease.restype = None

    @staticmethod
    def _buffer(value: str) -> tuple[bytes, ctypes.Array[ctypes.c_char]]:
        encoded = value.encode("utf-8")
        return encoded, ctypes.create_string_buffer(encoded)

    def get_password(self, service_name: str, username: str) -> str | None:
        service, service_buffer = self._buffer(service_name)
        account, account_buffer = self._buffer(username)
        password_length = ctypes.c_uint32()
        password_data = ctypes.c_void_p()
        item = ctypes.c_void_p()
        status = self._security.SecKeychainFindGenericPassword(
            None,
            len(service), service_buffer,
            len(account), account_buffer,
            ctypes.byref(password_length),
            ctypes.byref(password_data),
            ctypes.byref(item),
        )
        if status == self._ITEM_NOT_FOUND:
            return None
        if status != 0:
            raise RuntimeError("macOS Keychain could not read the release signing key")
        try:
            raw = ctypes.string_at(password_data, password_length.value)
            return raw.decode("utf-8")
        except UnicodeError as exc:
            raise RuntimeError("the macOS Keychain release key is malformed") from exc
        finally:
            if password_data:
                self._security.SecKeychainItemFreeContent(None, password_data)
            if item:
                self._core_foundation.CFRelease(item)

    def set_password(self, service_name: str, username: str, password: str) -> None:
        if self.get_password(service_name, username) is not None:
            raise RuntimeError("refusing to replace an existing release signing key")
        service, service_buffer = self._buffer(service_name)
        account, account_buffer = self._buffer(username)
        secret, secret_buffer = self._buffer(password)
        item = ctypes.c_void_p()
        status = self._security.SecKeychainAddGenericPassword(
            None,
            len(service), service_buffer,
            len(account), account_buffer,
            len(secret), secret_buffer,
            ctypes.byref(item),
        )
        if item:
            self._core_foundation.CFRelease(item)
        if status != 0:
            raise RuntimeError("macOS Keychain could not protect the release signing key")


def _credential_store() -> CredentialStore:
    try:
        import keyring
    except ImportError:
        keyring = None
    if keyring is not None:
        backend = keyring.get_keyring()
        if getattr(backend, "priority", 0) > 0:
            return keyring
    if sys.platform == "darwin":
        # On current macOS releases system framework executables live in the
        # dyld shared cache.  Their compatibility symlink can therefore report
        # ``exists() == False`` even though dlopen resolves it correctly.
        try:
            return MacOSSecurityCredentialStore()
        except OSError:
            pass
    raise RuntimeError("no protected operating-system credential store is available")


def _validate_key_id(key_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", key_id):
        raise RuntimeError("the release signing key id is invalid")


def _private_key_bytes(encoded: str) -> bytes:
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("the stored release signing key is invalid") from exc
    if len(raw) != 32:
        raise RuntimeError("the stored release signing key is invalid")
    return raw


def _public_key_b64(private_key_b64: str) -> str:
    private = Ed25519PrivateKey.from_private_bytes(_private_key_bytes(private_key_b64))
    public = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return base64.b64encode(public).decode("ascii")


def _new_private_key_b64() -> str:
    raw = Ed25519PrivateKey.generate().private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    return base64.b64encode(raw).decode("ascii")


def _load_private_key(key_id: str, store: CredentialStore | None = None) -> str:
    _validate_key_id(key_id)
    value = (store or _credential_store()).get_password(KEYCHAIN_SERVICE, key_id)
    if not value:
        raise RuntimeError(
            f"release key {key_id!r} is not present in the OS credential store; "
            "run the init command first"
        )
    _private_key_bytes(value)
    return value


def _trust_document(
    *, key_id: str, public_key: str, valid_from: str, valid_until: str,
) -> dict[str, Any]:
    document = {
        "format": TRUST_STORE_FORMAT,
        "schema_version": 1,
        "keys": [{
            "key_id": key_id,
            "algorithm": "Ed25519",
            "public_key": public_key,
            "valid_from": valid_from,
            "valid_until": valid_until,
            "channels": ["stable", "beta"],
            "revoked_at": None,
        }],
    }
    validate_trust_store(document)
    return document


def initialize_release_key(
    *,
    key_id: str,
    trust_store_path: Path,
    valid_from: str,
    valid_until: str,
    store: CredentialStore | None = None,
) -> dict[str, Any]:
    """Create or validate the release identity, never replacing an existing key."""
    _validate_key_id(key_id)
    credential_store = store or _credential_store()
    private_key = credential_store.get_password(KEYCHAIN_SERVICE, key_id)
    created = private_key is None
    if created:
        private_key = _new_private_key_b64()
        credential_store.set_password(KEYCHAIN_SERVICE, key_id, private_key)
        # A successful write is not enough: prove the selected backend can read
        # the exact value before producing a public trust record for it.
        persisted = credential_store.get_password(KEYCHAIN_SERVICE, key_id)
        if persisted != private_key:
            raise RuntimeError("the operating-system credential store did not retain the key")
    assert private_key is not None
    public_key = _public_key_b64(private_key)
    document = _trust_document(
        key_id=key_id,
        public_key=public_key,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    if trust_store_path.exists():
        existing = _load_json_file(
            trust_store_path, "release trust store", require_canonical=True,
        )
        validate_trust_store(existing)
        if existing != document:
            raise RuntimeError(
                "the existing public trust store does not match the protected key; "
                "refusing to replace release trust"
            )
    else:
        _write_json_atomic(trust_store_path, document)
    os.chmod(trust_store_path, 0o644)
    digest = hashlib.sha256(canonical_json(document) + b"\n").hexdigest()
    return {
        "created": created,
        "key_id": key_id,
        "public_key_fingerprint_sha256": hashlib.sha256(
            base64.b64decode(public_key, validate=True)
        ).hexdigest(),
        "trust_store": str(trust_store_path),
        "trust_store_sha256": digest,
    }


def _project_version() -> str:
    with (REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def sign_release_artifacts(
    *,
    key_id: str,
    trust_store_path: Path,
    artifact_dir: Path,
    channel: str,
    signed_at: str,
    store: CredentialStore | None = None,
) -> dict[str, Any]:
    """Sign and immediately verify every complete release artifact."""
    private_key = _load_private_key(key_id, store)
    trust = _load_json_file(
        trust_store_path, "release trust store", require_canonical=True,
    )
    validate_trust_store(trust)
    matching = [item for item in trust["keys"] if item["key_id"] == key_id]
    if len(matching) != 1 or matching[0]["public_key"] != _public_key_b64(private_key):
        raise RuntimeError("the protected private key does not match the public trust store")

    artifacts = [artifact_dir / name for name in RELEASE_ARTIFACTS]
    # Complete the full preflight before writing any signature.  A partial
    # platform set or stale SBOM must never look like a signed release.
    for artifact in artifacts:
        if artifact.is_symlink() or not artifact.is_file():
            raise RuntimeError(f"release artifact is missing or unsafe: {artifact.name}")
        verify_sbom_binding(
            artifact, artifact.with_name(artifact.name + ".sbom.cdx.json")
        )

    version = _project_version()
    statements = {
        artifact: sign_file(
            artifact,
            version=version,
            channel=channel,
            signed_at=signed_at,
            key_id=key_id,
            private_key_b64=private_key,
        )
        for artifact in artifacts
    }
    for artifact, statement in statements.items():
        verify_file(statement, artifact, trust)
    for artifact, statement in statements.items():
        signature_path = artifact.with_name(artifact.name + ".sig.json")
        _write_json_atomic(signature_path, statement)
        os.chmod(signature_path, 0o644)
    return {
        "key_id": key_id,
        "channel": channel,
        "signed_at": signed_at,
        "signed_artifacts": [artifact.name for artifact in artifacts],
    }


def create_release_manifest(
    *,
    key_id: str,
    trust_store_path: Path,
    artifact_dir: Path,
    output_path: Path,
    channel: str,
    published_at: str,
    minimum_supported_version: str,
    store: CredentialStore | None = None,
) -> dict[str, Any]:
    """Create and offline-verify the canonical all-platform release record."""
    private_key = _load_private_key(key_id, store)
    trust = _load_json_file(
        trust_store_path, "release trust store", require_canonical=True,
    )
    validate_trust_store(trust)
    matching = [item for item in trust["keys"] if item["key_id"] == key_id]
    if len(matching) != 1 or matching[0]["public_key"] != _public_key_b64(private_key):
        raise RuntimeError("the protected private key does not match the public trust store")

    version = _project_version()
    specifications = (
        ("macos", "arm64", "Sift.dmg", "application/x-apple-diskimage"),
        (
            "windows", "x86_64", "Sift-Windows-x64-Setup.exe",
            "application/vnd.microsoft.portable-executable",
        ),
        ("linux", "x86_64", "Sift-Linux-x86_64.tar.gz", "application/gzip"),
        ("linux", "aarch64", "Sift-Linux-aarch64.tar.gz", "application/gzip"),
    )
    artifacts = [
        _artifact_descriptor(
            f"{platform_name},{architecture},{artifact_dir / filename},{media_type}"
        )
        for platform_name, architecture, filename, media_type in specifications
    ]
    manifest = sign_manifest({
        "format": MANIFEST_FORMAT,
        "schema_version": 1,
        "release_id": f"sift-{version.replace('.', '-')}-{channel}",
        "version": version,
        "channel": channel,
        "published_at": published_at,
        "minimum_supported_version": minimum_supported_version,
        "rollback": {
            "allowed": False,
            "from_versions": [],
            "expires_at": None,
            "reason": "",
        },
        "artifacts": artifacts,
        "signing_key_id": key_id,
        "signature_algorithm": "Ed25519",
    }, private_key)
    verify_release(
        manifest,
        trust,
        artifact_dir,
        expected_channel=channel,
        installed_version=minimum_supported_version,
        highest_seen_version=minimum_supported_version,
        now=datetime.now(timezone.utc),
    )
    _write_json_atomic(output_path, manifest)
    os.chmod(output_path, 0o644)
    return {
        "key_id": key_id,
        "channel": channel,
        "manifest": str(output_path),
        "release_id": manifest["release_id"],
        "artifacts": len(artifacts),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Protect and use Sift's cross-platform release signing key.",
    )
    parser.add_argument("--key-id", default=DEFAULT_KEY_ID)
    parser.add_argument("--trust-store", type=Path, default=DEFAULT_TRUST_STORE)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser("init", help="create or validate release trust")
    initialize.add_argument("--valid-from", default=DEFAULT_VALID_FROM)
    initialize.add_argument("--valid-until", default=DEFAULT_VALID_UNTIL)
    sign = commands.add_parser("sign", help="sign and verify current artifacts")
    sign.add_argument("--artifact-dir", type=Path, default=REPOSITORY_ROOT / "dist")
    sign.add_argument("--channel", choices=("stable", "beta"), default="stable")
    sign.add_argument("--signed-at", default=None)
    manifest = commands.add_parser(
        "manifest", help="create and verify the all-platform release manifest",
    )
    manifest.add_argument(
        "--artifact-dir", type=Path, default=REPOSITORY_ROOT / "dist",
    )
    manifest.add_argument("--channel", choices=("stable", "beta"), default="stable")
    manifest.add_argument("--published-at", default=None)
    manifest.add_argument("--minimum-supported-version", default=None)
    manifest.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        if args.command == "init":
            result = initialize_release_key(
                key_id=args.key_id,
                trust_store_path=args.trust_store,
                valid_from=args.valid_from,
                valid_until=args.valid_until,
            )
        elif args.command == "sign":
            result = sign_release_artifacts(
                key_id=args.key_id,
                trust_store_path=args.trust_store,
                artifact_dir=args.artifact_dir,
                channel=args.channel,
                signed_at=args.signed_at or _utc_now(),
            )
        else:
            channel = args.channel
            result = create_release_manifest(
                key_id=args.key_id,
                trust_store_path=args.trust_store,
                artifact_dir=args.artifact_dir,
                output_path=args.output or (
                    args.artifact_dir / f"release-manifest-{channel}.json"
                ),
                channel=channel,
                published_at=args.published_at or _utc_now(),
                minimum_supported_version=(
                    args.minimum_supported_version or _project_version()
                ),
            )
    except Exception as exc:  # noqa: BLE001 - concise release-tool boundary
        print(f"release signing failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
