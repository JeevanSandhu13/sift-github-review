"""Release-key custody and complete-artifact signing tests."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest

from sift.release_manifest import (
    _load_json_file,
    canonical_json,
    generate_cyclonedx_sbom,
    verify_file,
)


_SCRIPT = Path(__file__).resolve().parents[1] / "packaging" / "manage_release_signing.py"
_SPEC = importlib.util.spec_from_file_location("sift_manage_release_signing", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
signing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(signing)


class MemoryCredentialStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service_name: str, username: str) -> str | None:
        return self.values.get((service_name, username))

    def set_password(self, service_name: str, username: str, password: str) -> None:
        self.values[(service_name, username)] = password


def _artifacts(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(signing, "RELEASE_ARTIFACTS", (
        "Sift.dmg",
        "Sift-Windows-x64-Setup.exe",
        "Sift-Linux-x86_64.tar.gz",
    ))
    for name in signing.RELEASE_ARTIFACTS:
        artifact = root / name
        artifact.write_bytes((name + " release bytes").encode("utf-8"))
        sbom = generate_cyclonedx_sbom(artifact, "0.1.0")
        (root / f"{name}.sbom.cdx.json").write_bytes(canonical_json(sbom) + b"\n")


def test_key_is_kept_in_credential_store_and_only_public_trust_is_written(
    tmp_path: Path,
) -> None:
    store = MemoryCredentialStore()
    trust_path = tmp_path / "release-trust-store.json"
    result = signing.initialize_release_key(
        key_id="release-test",
        trust_store_path=trust_path,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2031-01-01T00:00:00Z",
        store=store,
    )
    private = store.get_password(signing.KEYCHAIN_SERVICE, "release-test")
    assert private is not None
    assert private not in trust_path.read_text(encoding="utf-8")
    assert result["created"] is True
    assert result["trust_store_sha256"]

    repeated = signing.initialize_release_key(
        key_id="release-test",
        trust_store_path=trust_path,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2031-01-01T00:00:00Z",
        store=store,
    )
    assert repeated["created"] is False
    assert repeated["public_key_fingerprint_sha256"] == result[
        "public_key_fingerprint_sha256"
    ]


def test_existing_mismatched_trust_store_is_never_replaced(tmp_path: Path) -> None:
    first = MemoryCredentialStore()
    second = MemoryCredentialStore()
    trust_path = tmp_path / "release-trust-store.json"
    signing.initialize_release_key(
        key_id="release-test",
        trust_store_path=trust_path,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2031-01-01T00:00:00Z",
        store=first,
    )
    original = trust_path.read_bytes()
    with pytest.raises(RuntimeError, match="does not match"):
        signing.initialize_release_key(
            key_id="release-test",
            trust_store_path=trust_path,
            valid_from="2026-01-01T00:00:00Z",
            valid_until="2031-01-01T00:00:00Z",
            store=second,
        )
    assert trust_path.read_bytes() == original


def test_all_artifacts_are_signed_and_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryCredentialStore()
    trust_path = tmp_path / "release-trust-store.json"
    signing.initialize_release_key(
        key_id="release-test",
        trust_store_path=trust_path,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2031-01-01T00:00:00Z",
        store=store,
    )
    _artifacts(tmp_path, monkeypatch)
    result = signing.sign_release_artifacts(
        key_id="release-test",
        trust_store_path=trust_path,
        artifact_dir=tmp_path,
        channel="stable",
        signed_at="2026-08-26T12:00:00Z",
        store=store,
    )
    trust = _load_json_file(trust_path, "trust", require_canonical=True)
    assert result["signed_artifacts"] == list(signing.RELEASE_ARTIFACTS)
    for name in signing.RELEASE_ARTIFACTS:
        statement_path = tmp_path / f"{name}.sig.json"
        statement = json.loads(statement_path.read_text(encoding="utf-8"))
        assert verify_file(statement, tmp_path / name, trust)["ok"] is True


def test_missing_artifact_preflight_writes_no_partial_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryCredentialStore()
    trust_path = tmp_path / "release-trust-store.json"
    signing.initialize_release_key(
        key_id="release-test",
        trust_store_path=trust_path,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2031-01-01T00:00:00Z",
        store=store,
    )
    _artifacts(tmp_path, monkeypatch)
    (tmp_path / signing.RELEASE_ARTIFACTS[-1]).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        signing.sign_release_artifacts(
            key_id="release-test",
            trust_store_path=trust_path,
            artifact_dir=tmp_path,
            channel="stable",
            signed_at="2026-08-26T12:00:00Z",
            store=store,
        )
    assert not list(tmp_path.glob("*.sig.json"))


def test_canonical_manifest_binds_all_platforms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryCredentialStore()
    trust_path = tmp_path / "release-trust-store.json"
    signing.initialize_release_key(
        key_id="release-test",
        trust_store_path=trust_path,
        valid_from="2026-01-01T00:00:00Z",
        valid_until="2031-01-01T00:00:00Z",
        store=store,
    )
    monkeypatch.setattr(signing, "RELEASE_ARTIFACTS", (
        "Sift.dmg",
        "Sift-Windows-x64-Setup.exe",
        "Sift-Linux-x86_64.tar.gz",
        "Sift-Linux-aarch64.tar.gz",
    ))
    for name in signing.RELEASE_ARTIFACTS:
        artifact = tmp_path / name
        artifact.write_bytes((name + " release bytes").encode("utf-8"))
        sbom = generate_cyclonedx_sbom(artifact, "0.1.0")
        (tmp_path / f"{name}.sbom.cdx.json").write_bytes(canonical_json(sbom) + b"\n")
    output = tmp_path / "release-manifest-stable.json"
    result = signing.create_release_manifest(
        key_id="release-test",
        trust_store_path=trust_path,
        artifact_dir=tmp_path,
        output_path=output,
        channel="stable",
        published_at="2026-08-26T12:00:00Z",
        minimum_supported_version="0.1.0",
        store=store,
    )
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert result["artifacts"] == 4
    assert {row["platform"] for row in manifest["artifacts"]} == {
        "macos", "windows", "linux",
    }
    assert [
        row["architecture"] for row in manifest["artifacts"]
        if row["platform"] == "linux"
    ] == ["x86_64", "aarch64"]
