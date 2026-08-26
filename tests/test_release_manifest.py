"""Trust-boundary tests for cross-platform release manifests."""

from __future__ import annotations

import base64
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sift.release_manifest import (
    MANIFEST_FORMAT,
    ReleaseManifestError,
    TRUST_STORE_FORMAT,
    canonical_json,
    generate_cyclonedx_sbom,
    load_trusted_json,
    main,
    sha256_file,
    sign_file,
    sign_manifest,
    verify_file,
    verify_release,
    verify_sbom_binding,
)
from sift import release_manifest as release_manifest_module


def _keys() -> tuple[str, str]:
    private = Ed25519PrivateKey.generate()
    private_raw = private.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_raw = private.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return (
        base64.b64encode(private_raw).decode("ascii"),
        base64.b64encode(public_raw).decode("ascii"),
    )


def _documents(tmp_path: Path, *, version: str = "1.2.3"):
    private, public = _keys()
    artifacts = []
    specs = [
        ("macos", "arm64", "Sift.dmg", "application/x-apple-diskimage"),
        ("windows", "x86_64", "Sift-Windows-x64-Setup.exe", "application/vnd.microsoft.portable-executable"),
        ("linux", "x86_64", "Sift-Linux-x86_64.tar.gz", "application/gzip"),
    ]
    for platform_name, architecture, filename, media_type in specs:
        artifact = tmp_path / filename
        artifact.write_bytes((platform_name + version).encode("utf-8") * 20)
        sbom = tmp_path / f"{filename}.sbom.cdx.json"
        sbom.write_text('{"bomFormat":"CycloneDX"}\n', encoding="utf-8")
        digest, size = sha256_file(artifact)
        sbom_digest, sbom_size = sha256_file(sbom)
        artifacts.append({
            "platform": platform_name,
            "architecture": architecture,
            "filename": filename,
            "media_type": media_type,
            "size": size,
            "sha256": digest,
            "sbom": {
                "filename": sbom.name,
                "format": "cyclonedx-json",
                "size": sbom_size,
                "sha256": sbom_digest,
            },
        })
    payload = {
        "format": MANIFEST_FORMAT,
        "schema_version": 1,
        "release_id": "sift-" + version.replace(".", "-") + "-stable",
        "version": version,
        "channel": "stable",
        "published_at": "2026-01-22T20:00:00Z",
        "minimum_supported_version": "1.0.0",
        "rollback": {
            "allowed": False,
            "from_versions": [],
            "expires_at": None,
            "reason": "",
        },
        "artifacts": artifacts,
        "signing_key_id": "release-2026",
        "signature_algorithm": "Ed25519",
    }
    trust = {
        "format": TRUST_STORE_FORMAT,
        "schema_version": 1,
        "keys": [{
            "key_id": "release-2026",
            "algorithm": "Ed25519",
            "public_key": public,
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_until": "2027-01-01T00:00:00Z",
            "channels": ["stable", "beta"],
            "revoked_at": None,
        }],
    }
    return sign_manifest(payload, private), trust


def test_offline_verification_covers_all_artifacts_and_sboms(tmp_path) -> None:
    manifest, trust = _documents(tmp_path)
    result = verify_release(
        manifest,
        trust,
        tmp_path,
        expected_channel="stable",
        installed_version="1.1.0",
        highest_seen_version="1.1.0",
        now=datetime(2026, 8, 22, tzinfo=timezone.utc),
    )
    assert result["ok"] is True
    assert len(result["verified_files"]) == 6


def test_tampered_manifest_signature_fails_closed(tmp_path) -> None:
    manifest, trust = _documents(tmp_path)
    manifest["version"] = "1.2.4"
    with pytest.raises(ReleaseManifestError, match="signature"):
        verify_release(
            manifest, trust, tmp_path,
            expected_channel="stable", installed_version="1.1.0",
        )


def test_manifest_too_far_in_future_is_rejected_even_when_validly_signed(
    tmp_path,
) -> None:
    manifest, trust = _documents(tmp_path)
    # The fixture is genuinely signed and the key is valid at publication;
    # only the verifier's trusted clock makes it ineligible.
    with pytest.raises(ReleaseManifestError, match="future"):
        verify_release(
            manifest, trust, tmp_path,
            expected_channel="stable",
            installed_version="1.1.0",
            now=datetime(2026, 1, 22, 19, 54, tzinfo=timezone.utc),
        )


def test_tampered_artifact_fails_strict_hash_check(tmp_path) -> None:
    manifest, trust = _documents(tmp_path)
    (tmp_path / "Sift.dmg").write_bytes(b"replacement")
    with pytest.raises(ReleaseManifestError, match="hash/size mismatch"):
        verify_release(
            manifest, trust, tmp_path,
            expected_channel="stable", installed_version="1.1.0",
        )


def test_revoked_and_wrong_channel_keys_are_rejected(tmp_path) -> None:
    manifest, trust = _documents(tmp_path)
    revoked = copy.deepcopy(trust)
    revoked["keys"][0]["revoked_at"] = "2026-08-23T00:00:00Z"
    with pytest.raises(ReleaseManifestError, match="revoked"):
        verify_release(
            manifest, revoked, tmp_path,
            expected_channel="stable", installed_version="1.1.0",
        )
    beta_only = copy.deepcopy(trust)
    beta_only["keys"][0]["channels"] = ["beta"]
    with pytest.raises(ReleaseManifestError, match="channel"):
        verify_release(
            manifest, beta_only, tmp_path,
            expected_channel="stable", installed_version="1.1.0",
        )


def test_channel_pinning_rejects_beta_on_stable_client(tmp_path) -> None:
    manifest, trust = _documents(tmp_path)
    with pytest.raises(ReleaseManifestError, match="configured channel"):
        verify_release(
            manifest, trust, tmp_path,
            expected_channel="beta", installed_version="1.1.0",
        )


def test_downgrade_requires_signed_unexpired_rollback_authority(tmp_path) -> None:
    unsigned, trust = _documents(tmp_path, version="1.1.0")
    # Recover a fresh key/trust pair because rollback metadata is signed.
    private, public = _keys()
    payload = {key: value for key, value in unsigned.items() if key != "signature"}
    payload["rollback"] = {
        "allowed": True,
        "from_versions": ["1.2.0"],
        "expires_at": "2026-09-01T00:00:00Z",
        "reason": "Emergency rollback after a verified regression.",
    }
    trust["keys"][0]["public_key"] = public
    manifest = sign_manifest(payload, private)
    result = verify_release(
        manifest, trust, tmp_path,
        expected_channel="stable",
        installed_version="1.2.0",
        highest_seen_version="1.2.0",
        now=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )
    assert result["ok"] is True
    with pytest.raises(ReleaseManifestError, match="unauthorized rollback"):
        verify_release(
            manifest, trust, tmp_path,
            expected_channel="stable",
            installed_version="1.2.0",
            highest_seen_version="1.2.0",
            now=datetime(2026, 9, 2, tzinfo=timezone.utc),
        )


def test_pinned_trust_store_hash_and_symlink_fail_closed(tmp_path) -> None:
    _, trust = _documents(tmp_path)
    trust_path = tmp_path / "release-trust.json"
    trust_path.write_bytes(canonical_json(trust) + b"\n")
    digest, _ = sha256_file(trust_path)
    assert load_trusted_json(trust_path, digest)["keys"][0]["key_id"] == "release-2026"
    with pytest.raises(ReleaseManifestError, match="pinned"):
        load_trusted_json(trust_path, "0" * 64)

    link = tmp_path / "linked-trust.json"
    try:
        link.symlink_to(trust_path)
    except (OSError, NotImplementedError):
        return
    with pytest.raises(ReleaseManifestError, match="symlink"):
        load_trusted_json(link, digest)


def test_schema_rejects_unknown_fields_and_missing_platform(tmp_path) -> None:
    manifest, _ = _documents(tmp_path)
    payload = {key: value for key, value in manifest.items() if key != "signature"}
    payload["unexpected"] = True
    private, _ = _keys()
    with pytest.raises(ReleaseManifestError, match="fields"):
        sign_manifest(payload, private)

    payload.pop("unexpected")
    payload["artifacts"] = payload["artifacts"][:-1]
    with pytest.raises(ReleaseManifestError, match="cover"):
        sign_manifest(payload, private)


def test_generated_sbom_is_deterministic_and_binds_artifact(tmp_path) -> None:
    artifact = tmp_path / "Sift.dmg"
    artifact.write_bytes(b"release bytes")
    first = generate_cyclonedx_sbom(artifact, "1.2.3")
    second = generate_cyclonedx_sbom(artifact, "1.2.3")
    assert canonical_json(first) == canonical_json(second)
    assert first["bomFormat"] == "CycloneDX"
    assert first["metadata"]["component"]["hashes"][0]["content"] == sha256_file(artifact)[0]


def test_windows_low_level_release_reads_force_binary_mode(
    tmp_path, monkeypatch,
) -> None:
    artifact = tmp_path / "release.bin"
    artifact.write_bytes(b"prefix\x1asuffix")
    binary_flag = 1 << 29
    real_open = release_manifest_module.os.open
    seen_flags: list[int] = []

    def capturing_open(path, flags, *args):
        seen_flags.append(flags)
        return real_open(path, flags & ~binary_flag, *args)

    monkeypatch.setattr(release_manifest_module.os, "O_BINARY", binary_flag, raising=False)
    monkeypatch.setattr(release_manifest_module.os, "open", capturing_open)

    assert sha256_file(artifact)[1] == len(b"prefix\x1asuffix")
    assert release_manifest_module._read_regular_file(artifact) == b"prefix\x1asuffix"
    assert len(seen_flags) == 2
    assert all(flags & binary_flag for flags in seen_flags)


def test_sbom_binding_verification_rejects_changed_artifact(tmp_path) -> None:
    artifact = tmp_path / "Sift.dmg"
    sbom_path = tmp_path / "Sift.dmg.sbom.cdx.json"
    artifact.write_bytes(b"release bytes")
    sbom_path.write_bytes(canonical_json(generate_cyclonedx_sbom(artifact, "1.2.3")))
    assert verify_sbom_binding(artifact, sbom_path)["ok"] is True

    artifact.write_bytes(b"changed after SBOM generation")
    with pytest.raises(ReleaseManifestError, match="SBOM artifact"):
        verify_sbom_binding(artifact, sbom_path)


def test_detached_archive_signature_is_canonical_and_hash_bound(tmp_path) -> None:
    private, public = _keys()
    archive = tmp_path / "Sift-Linux-x86_64.tar.gz"
    archive.write_bytes(b"signed release archive")
    statement = sign_file(
        archive,
        version="1.2.3",
        channel="stable",
        signed_at="2026-08-22T20:00:00Z",
        key_id="release-2026",
        private_key_b64=private,
    )
    trust = {
        "format": TRUST_STORE_FORMAT,
        "schema_version": 1,
        "keys": [{
            "key_id": "release-2026",
            "algorithm": "Ed25519",
            "public_key": public,
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_until": "2027-01-01T00:00:00Z",
            "channels": ["stable"],
            "revoked_at": None,
        }],
    }
    assert verify_file(statement, archive, trust)["ok"] is True
    archive.write_bytes(b"tampered")
    with pytest.raises(ReleaseManifestError, match="hash/size"):
        verify_file(statement, archive, trust)


def test_create_cli_assembles_signed_all_platform_manifest(
    tmp_path, monkeypatch,
) -> None:
    _, trust = _documents(tmp_path)
    private, public = _keys()
    trust["keys"][0]["public_key"] = public
    output = tmp_path / "release-manifest.json"
    monkeypatch.setenv("SIFT_RELEASE_PRIVATE_KEY_B64", private)
    result = main([
        "create",
        "--version", "1.2.3",
        "--channel", "stable",
        "--minimum-supported-version", "1.0.0",
        "--published-at", "2026-01-22T20:00:00Z",
        "--release-id", "sift-1-2-3-stable",
        "--key-id", "release-2026",
        "--artifact", f"macos,arm64,{tmp_path / 'Sift.dmg'},application/x-apple-diskimage",
        "--artifact", f"windows,x86_64,{tmp_path / 'Sift-Windows-x64-Setup.exe'},application/vnd.microsoft.portable-executable",
        "--artifact", f"linux,x86_64,{tmp_path / 'Sift-Linux-x86_64.tar.gz'},application/gzip",
        "--output", str(output),
    ])
    assert result == 0
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert verify_release(
        manifest, trust, tmp_path,
        expected_channel="stable", installed_version="1.1.0",
    )["ok"] is True


def test_offline_manifest_loader_rejects_noncanonical_and_duplicate_json(
    tmp_path,
) -> None:
    pretty = tmp_path / "pretty.json"
    pretty.write_text('{\n  "a": 1\n}\n', encoding="utf-8")
    with pytest.raises(ReleaseManifestError, match="canonical"):
        release_manifest_module._load_json_file(
            pretty, "manifest", require_canonical=True,
        )
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(ReleaseManifestError, match="duplicate"):
        release_manifest_module._load_json_file(duplicate, "manifest")


def test_atomic_json_writer_retries_partial_os_writes(tmp_path, monkeypatch) -> None:
    output = tmp_path / "statement.json"
    real_write = release_manifest_module.os.write

    def partial_write(fd, data):
        return real_write(fd, bytes(data[:3]))

    monkeypatch.setattr(release_manifest_module.os, "write", partial_write)
    release_manifest_module._write_json_atomic(output, {"b": 2, "a": 1})
    assert output.read_bytes() == b'{"a":1,"b":2}\n'
