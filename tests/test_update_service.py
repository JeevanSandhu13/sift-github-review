from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from sift.release_manifest import canonical_json, sign_manifest
from sift.update_service import (
    UpdateError,
    check_for_update,
    load_highest_seen_version,
    native_target,
    persist_highest_seen_version,
    prepare_update,
)
from sift.update_config import load_update_policy


class _Response:
    def __init__(self, body: bytes, *, declared: int | None = None) -> None:
        self.body = body
        self.offset = 0
        self.headers = (
            {} if declared is None else {"Content-Length": str(declared)}
        )

    def read(self, amount: int = -1) -> bytes:
        if amount < 0:
            amount = len(self.body) - self.offset
        result = self.body[self.offset:self.offset + min(amount, 7)]
        self.offset += len(result)
        return result

    def __enter__(self):
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _Transport:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.opened: list[str] = []

    def open(self, url: str, timeout: float) -> _Response:
        self.opened.append(url)
        if url not in self.bodies:
            raise AssertionError(f"unexpected update URL: {url}")
        return _Response(self.bodies[url], declared=len(self.bodies[url]))


class _Keyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, account: str) -> str | None:
        return self.values.get((service, account))

    def set_password(self, service: str, account: str, value: str) -> None:
        self.values[(service, account)] = value


def _release(tmp_path: Path, *, version: str = "1.2.0"):
    key = Ed25519PrivateKey.generate()
    private = key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    trust = {
        "format": "sift-release-trust-store",
        "schema_version": 1,
        "keys": [{
            "key_id": "release-2026",
            "algorithm": "Ed25519",
            "public_key": base64.b64encode(public).decode("ascii"),
            "valid_from": "2026-01-01T00:00:00Z",
            "valid_until": "2028-01-01T00:00:00Z",
            "channels": ["stable"],
            "revoked_at": None,
        }],
    }
    trust_raw = canonical_json(trust) + b"\n"
    trust_path = tmp_path / "trust.json"
    trust_path.write_bytes(trust_raw)
    bodies: dict[str, bytes] = {}
    artifacts = []
    for platform_name, architecture, filename in (
        ("macos", "arm64", "Sift.dmg"),
        ("windows", "x86_64", "Sift-Windows-x64-Setup.exe"),
        ("linux", "x86_64", "Sift-Linux-x86_64.tar.gz"),
        ("linux", "aarch64", "Sift-Linux-aarch64.tar.gz"),
    ):
        artifact = (filename + " verified payload").encode()
        sbom_name = filename + ".sbom.cdx.json"
        sbom = b'{"bomFormat":"CycloneDX"}\n'
        bodies[f"https://updates.example.test/{filename}"] = artifact
        bodies[f"https://updates.example.test/{sbom_name}"] = sbom
        artifacts.append({
            "platform": platform_name,
            "architecture": architecture,
            "filename": filename,
            "media_type": "application/octet-stream",
            "size": len(artifact),
            "sha256": hashlib.sha256(artifact).hexdigest(),
            "sbom": {
                "filename": sbom_name,
                "format": "cyclonedx-json",
                "size": len(sbom),
                "sha256": hashlib.sha256(sbom).hexdigest(),
            },
        })
    payload = {
        "format": "sift-release-manifest",
        "schema_version": 1,
        "release_id": "sift-1-2-0-stable",
        "version": version,
        "channel": "stable",
        "published_at": "2026-08-24T12:00:00Z",
        "minimum_supported_version": "1.0.0",
        "rollback": {
            "allowed": False, "from_versions": [],
            "expires_at": None, "reason": "",
        },
        "artifacts": artifacts,
        "signing_key_id": "release-2026",
        "signature_algorithm": "Ed25519",
    }
    manifest = sign_manifest(payload, base64.b64encode(private).decode("ascii"))
    manifest_url = "https://updates.example.test/release-manifest-stable.json"
    bodies[manifest_url] = canonical_json(manifest) + b"\n"
    return (
        manifest_url,
        trust_path,
        hashlib.sha256(trust_raw).hexdigest(),
        bodies,
    )


def test_native_target_is_explicit_and_rejects_unknown() -> None:
    assert native_target("Darwin", "arm64") == ("macos", "arm64")
    assert native_target("Windows", "AMD64") == ("windows", "x86_64")
    assert native_target("Linux", "aarch64") == ("linux", "aarch64")
    with pytest.raises(UpdateError, match="not supported"):
        native_target("Plan9", "mips")


def test_verified_native_update_is_streamed_and_staged_atomically(tmp_path: Path) -> None:
    url, trust, digest, bodies = _release(tmp_path)
    transport = _Transport(bodies)
    candidate = check_for_update(
        url,
        trust_store_path=trust,
        trust_store_sha256=digest,
        channel="stable",
        installed_version="1.1.0",
        transport=transport,
        system="Darwin",
        machine="arm64",
    )
    assert candidate.available is True
    result = prepare_update(candidate, tmp_path / "updates", transport=transport)
    assert result["status"] == "ready"
    assert Path(result["installer"]).read_bytes() == bodies[
        "https://updates.example.test/Sift.dmg"
    ]
    assert not list((tmp_path / "updates").glob(".sift-update-*"))
    assert all("Windows" not in item and "Linux" not in item for item in transport.opened)


def test_signed_manifest_selects_the_arm64_linux_artifact(tmp_path: Path) -> None:
    url, trust, digest, bodies = _release(tmp_path)
    candidate = check_for_update(
        url,
        trust_store_path=trust,
        trust_store_sha256=digest,
        channel="stable",
        installed_version="1.1.0",
        transport=_Transport(bodies),
        system="Linux",
        machine="aarch64",
    )
    assert candidate.artifact["filename"] == "Sift-Linux-aarch64.tar.gz"


def test_tampered_download_fails_closed_and_cleans_staging(tmp_path: Path) -> None:
    url, trust, digest, bodies = _release(tmp_path)
    candidate = check_for_update(
        url,
        trust_store_path=trust,
        trust_store_sha256=digest,
        channel="stable",
        installed_version="1.1.0",
        transport=_Transport(bodies),
        system="Darwin",
        machine="arm64",
    )
    bodies["https://updates.example.test/Sift.dmg"] = b"x" * candidate.artifact["size"]
    root = tmp_path / "updates"
    with pytest.raises(UpdateError, match="SHA-256"):
        prepare_update(candidate, root, transport=_Transport(bodies))
    assert not (root / "1.2.0").exists()
    assert not list(root.glob(".sift-update-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes do not apply")
def test_update_staging_repairs_permissive_directory_mode(tmp_path: Path) -> None:
    url, trust, digest, bodies = _release(tmp_path)
    candidate = check_for_update(
        url,
        trust_store_path=trust,
        trust_store_sha256=digest,
        channel="stable",
        installed_version="1.1.0",
        transport=_Transport(bodies),
        system="Darwin",
        machine="arm64",
    )
    root = tmp_path / "updates"
    root.mkdir(mode=0o755)
    prepare_update(candidate, root, transport=_Transport(bodies))
    assert root.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize(
    "url",
    [
        "http://updates.example.test/release.json",
        "https://user:secret@updates.example.test/release.json",
        "https://updates.example.test/release.json?machine=id",
    ],
)
def test_update_location_rejects_insecure_or_identifying_urls(
    tmp_path: Path, url: str,
) -> None:
    _, trust, digest, _ = _release(tmp_path)
    with pytest.raises(UpdateError):
        check_for_update(
            url,
            trust_store_path=trust,
            trust_store_sha256=digest,
            channel="stable",
            installed_version="1.1.0",
            transport=_Transport({}),
        )


def test_current_version_does_not_download_artifact(tmp_path: Path) -> None:
    url, trust, digest, bodies = _release(tmp_path)
    transport = _Transport(bodies)
    candidate = check_for_update(
        url,
        trust_store_path=trust,
        trust_store_sha256=digest,
        channel="stable",
        installed_version="1.2.0",
        transport=transport,
        system="Darwin",
        machine="arm64",
    )
    assert prepare_update(candidate, tmp_path / "updates", transport=transport) == {
        "ok": True, "status": "current", "version": "1.2.0",
    }
    assert transport.opened == [url]


def test_manifest_replay_below_highest_seen_version_is_rejected(tmp_path: Path) -> None:
    url, trust, digest, bodies = _release(tmp_path)
    with pytest.raises(UpdateError, match="unauthorized rollback"):
        check_for_update(
            url,
            trust_store_path=trust,
            trust_store_sha256=digest,
            channel="stable",
            installed_version="1.1.0",
            highest_seen_version="1.3.0",
            transport=_Transport(bodies),
            system="Darwin",
            machine="arm64",
        )


def test_build_policy_embeds_only_public_trust_material(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, trust, _, _ = _release(tmp_path)
    output = tmp_path / "embedded"
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    subprocess.run(
        [
            sys.executable,
            "packaging/configure_update_policy.py",
            "--manifest-url", "https://updates.example.test/release.json",
            "--trust-store", str(trust),
            "--output-dir", str(output),
        ],
        check=True,
        cwd=Path.cwd(),
        env=environment,
        capture_output=True,
        text=True,
    )
    monkeypatch.setenv("SIFT_UPDATE_POLICY_DIR", str(output))
    policy = load_update_policy()
    assert policy["configured"] is True
    assert policy["channel"] == "stable"
    joined = b"".join(path.read_bytes() for path in output.iterdir())
    assert b"private" not in joined.lower()
    assert set(path.name for path in output.iterdir()) == {
        "update-policy.json", "release-trust-store.json",
    }


def test_development_build_without_policy_is_explicitly_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIFT_UPDATE_POLICY_DIR", raising=False)
    policy = load_update_policy()
    assert policy["ok"] is True
    assert policy["configured"] is False


def test_frozen_development_build_without_policy_is_not_reported_as_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SIFT_UPDATE_POLICY_DIR", raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    policy = load_update_policy()
    assert policy["ok"] is True
    assert policy["configured"] is False
    assert "development build" in policy["reason"]


def test_highest_seen_state_is_monotonic_and_vault_backed() -> None:
    ring = _Keyring()
    assert load_highest_seen_version("stable", ring) is None
    persist_highest_seen_version("stable", "1.2.0", ring)
    persist_highest_seen_version("stable", "1.1.0", ring)
    assert load_highest_seen_version("stable", ring) == "1.2.0"
    assert ring.values == {("org.sapieninstitute.sift.update", "stable"): "1.2.0"}


def test_invalid_or_unavailable_rollback_state_fails_closed() -> None:
    ring = _Keyring()
    ring.values[("org.sapieninstitute.sift.update", "stable")] = "not-a-version"
    with pytest.raises(UpdateError, match="rollback state is invalid"):
        load_highest_seen_version("stable", ring)

    class Broken:
        def get_password(self, service: str, account: str) -> str | None:
            raise RuntimeError("vault locked")

    with pytest.raises(UpdateError, match="credential store is unavailable"):
        load_highest_seen_version("stable", Broken())


def test_development_cli_reports_update_channel_as_unavailable() -> None:
    environment = {**os.environ, "PYTHONPATH": str(Path.cwd() / "src")}
    environment.pop("SIFT_UPDATE_POLICY_DIR", None)
    completed = subprocess.run(
        [sys.executable, "-m", "sift", "--check-update"],
        cwd=Path.cwd(),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert '"status": "unavailable"' in completed.stdout
    assert "development build" in completed.stdout


def test_desktop_bridge_describes_and_checks_updates_without_a_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift.ui import SiftBridge

    monkeypatch.delenv("SIFT_UPDATE_POLICY_DIR", raising=False)
    bridge = SiftBridge()
    configuration = bridge.update_configuration()
    assert configuration["ok"] is True
    assert configuration["configured"] is False
    assert configuration["installed_version"]
    result = bridge.check_for_updates(False)
    assert result["ok"] is False
    assert result["status"] == "unavailable"
