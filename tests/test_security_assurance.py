from __future__ import annotations

import base64
import io
import json
import os
import stat
import struct
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import sift.security_assurance as security_assurance
from sift.security_assurance import (
    _safe_tar_target,
    apply_run_retention,
    encrypt_session_bundle,
    generate_cyclonedx_sbom,
    generate_ed25519_private_key,
    retention_candidates,
    review_pre_provider_disclosure,
    restore_encrypted_session,
    run_dependency_vulnerability_scan,
    run_bandit_static_scan,
    scan_python_static_security,
    scan_source_secrets,
    security_qualification_report,
    security_release_binding,
    sign_provenance_export,
    verify_provenance_signature,
)


ROOT = Path(__file__).resolve().parents[1]


def _release_fixture(tmp_path: Path, *, requires_dist: str | None = None) -> Path:
    root = tmp_path / "release-project"
    files = {
        "src/sift/__init__.py": b'__version__ = "0.1.0"\n',
        "scripts/check.py": b"print('ok')\n",
        "packaging/build.sh": b"#!/bin/sh\nexit 0\n",
        "siftbench/__init__.py": b"",
        "docs/security.md": b"documented\n",
        "tests/test_fixture.py": b"def test_ok():\n    assert True\n",
        "pyproject.toml": b'[project]\nname="sift"\nversion="0.1.0"\n',
        "CHANGELOG.md": b"initial\n",
        "SECURITY.md": b"report privately\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "uv.lock").write_text(
        """
version = 1
[[package]]
name = "alpha"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "alpha"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "beta"
version = "3.0.0"
source = { registry = "https://pypi.org/simple" }
[[package]]
name = "sift"
version = "0.1.0"
source = { editable = "." }
""".strip() + "\n",
        encoding="utf-8",
    )
    dist = root / "dist"
    dist.mkdir()
    metadata = "Name: sift\nVersion: 0.1.0\n"
    if requires_dist is not None:
        metadata += f"Requires-Dist: {requires_dist}\n"
    with zipfile.ZipFile(dist / "sift-0.1.0-py3-none-any.whl", "w") as archive:
        archive.writestr("sift/__init__.py", files["src/sift/__init__.py"])
        archive.writestr("sift-0.1.0.dist-info/METADATA", metadata)
    with tarfile.open(dist / "sift-0.1.0.tar.gz", "w:gz") as archive:
        for relative, content in files.items():
            member = tarfile.TarInfo(f"sift-0.1.0/{relative}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        metadata_bytes = metadata.encode("utf-8")
        member = tarfile.TarInfo("sift-0.1.0/PKG-INFO")
        member.size = len(metadata_bytes)
        archive.addfile(member, io.BytesIO(metadata_bytes))
    return root


def _encrypted_record_offsets(payload: bytes) -> list[tuple[bytes, int, int]]:
    """Return (type, start, end) for every well-formed v2 record."""
    position = len(security_assurance._ENC_MAGIC_V2)
    position = payload.index(b"\n", position) + 1
    records: list[tuple[bytes, int, int]] = []
    while position < len(payload):
        start = position
        record_type = payload[position : position + 1]
        assert len(payload[position + 1 : position + 5]) == 4
        length = struct.unpack(">I", payload[position + 1 : position + 5])[0]
        position += 5 + length
        assert position <= len(payload)
        records.append((record_type, start, position))
    assert position == len(payload)
    return records


def _write_legacy_v1_bundle(path: Path, *, passphrase: str) -> None:
    """Construct the historical v1 format for compatibility tests."""
    tar_bytes = io.BytesIO()
    content = b"legacy content\n"
    with tarfile.open(fileobj=tar_bytes, mode="w") as archive:
        member = tarfile.TarInfo("legacy.txt")
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))

    salt = os.urandom(16)
    nonce_prefix = os.urandom(8)
    header = {
        "format": "sift-encrypted-session",
        "version": 1,
        "cipher": "AES-256-GCM",
        "kdf": "scrypt-n32768-r8-p1",
        "chunk_size": security_assurance._ENC_CHUNK_SIZE,
        "salt": base64.b64encode(salt).decode("ascii"),
        "nonce_prefix": base64.b64encode(nonce_prefix).decode("ascii"),
        "created_at": "2026-08-21T00:00:00+00:00",
    }
    header_bytes = security_assurance._canonical_json(header)
    AESGCM, _ = security_assurance._crypto()
    aes = AESGCM(security_assurance._derive_key(passphrase, salt))
    plaintext = tar_bytes.getvalue()
    output = bytearray(security_assurance._ENC_MAGIC_V1 + header_bytes + b"\n")
    for counter, offset in enumerate(
        range(0, len(plaintext), security_assurance._ENC_CHUNK_SIZE)
    ):
        chunk = plaintext[offset : offset + security_assurance._ENC_CHUNK_SIZE]
        ciphertext = aes.encrypt(
            nonce_prefix + counter.to_bytes(4, "big"),
            chunk,
            header_bytes + counter.to_bytes(4, "big"),
        )
        output.extend(struct.pack(">I", len(ciphertext)))
        output.extend(ciphertext)
    path.write_bytes(output)


def test_threat_model_maps_every_boundary_to_real_controls_and_tests() -> None:
    model = json.loads((ROOT / "docs/security_threat_model.json").read_text(encoding="utf-8"))
    assert model["version"] == 1
    assert len(model["boundaries"]) >= 12
    ids = {row["id"] for row in model["boundaries"]}
    assert len(ids) == len(model["boundaries"])
    for boundary in model["boundaries"]:
        assert boundary["source"] and boundary["destination"]
        assert boundary["data"] and boundary["controls"] and boundary["tests"]
        for evidence in boundary["tests"]:
            assert (ROOT / evidence).is_file(), evidence
    for threat in model["threats"]:
        assert set(threat["boundaries"]).issubset(ids)
        assert threat["mitigation"]
    assert any("does not claim" in value for value in model["non_claims"])


def test_disclosure_review_is_content_free_and_optional() -> None:
    secret = "sk-" + "A" * 32
    review = review_pre_provider_disclosure(
        f"Contact jane@example.org. api_key={secret}",
        attachment_names=["patient name 123-45-6789.csv"],
    )
    assert review["warn"] is True
    assert {row["category"] for row in review["findings"]} >= {
        "credential",
        "email_address",
        "us_social_security_number",
    }
    encoded = json.dumps(review)
    assert secret not in encoded
    assert "jane@example.org" not in encoded
    assert "123-45-6789" not in encoded
    assert review_pre_provider_disclosure(secret, enabled=False) == {
        "enabled": False,
        "warn": False,
        "findings": [],
    }


def test_organization_sensitive_field_warning() -> None:
    review = review_pre_provider_disclosure(
        "summarize",
        field_names=["Participant ID", "age"],
        organization_sensitive_fields=["participant_id"],
    )
    assert any(
        row["category"] == "organization_sensitive_field"
        for row in review["findings"]
    )


def test_retention_requires_preview_and_confirmation(tmp_path: Path) -> None:
    old = tmp_path / ".sift" / "runs" / "old"
    fresh = tmp_path / ".sift" / "runs" / "fresh"
    old.mkdir(parents=True)
    fresh.mkdir()
    secret = old / "stderr.log"
    secret.write_text("sensitive output")
    now = datetime(2026, 8, 21, tzinfo=timezone.utc)
    old_stamp = (now - timedelta(days=31)).timestamp()
    os.utime(old, (old_stamp, old_stamp))
    fresh_stamp = (now - timedelta(days=2)).timestamp()
    os.utime(fresh, (fresh_stamp, fresh_stamp))

    rows = retention_candidates(tmp_path, run_retention_days=30, now=now)
    assert [row.path for row in rows] == [".sift/runs/old"]
    preview = apply_run_retention(
        tmp_path, run_retention_days=30, now=now, confirmed=False
    )
    assert preview["requires_confirmation"] is True
    assert secret.exists()
    result = apply_run_retention(
        tmp_path, run_retention_days=30, now=now, confirmed=True
    )
    assert result["removed"] == [".sift/runs/old"]
    assert not old.exists()
    assert fresh.exists()
    assert "cannot guarantee" in result["secure_delete_limitations"]


def test_retention_ignores_symlinked_run(tmp_path: Path) -> None:
    runs = tmp_path / ".sift" / "runs"
    runs.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep")
    try:
        (runs / "old").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    assert retention_candidates(tmp_path, run_retention_days=1) == []
    assert (outside / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_encrypted_session_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    (session / ".sift").mkdir(parents=True)
    (session / "data.csv").write_bytes(b"x,y\n1,2\n")
    (session / ".sift" / "workflow.json").write_text('{"approved":true}')
    bundle = tmp_path / "session.siftenc"
    result = encrypt_session_bundle(
        session, bundle, passphrase="correct horse battery staple"
    )
    assert result["cipher"] == "AES-256-GCM"
    assert result["format_version"] == 2
    assert result["authenticated_completion"] is True
    assert b"approved" not in bundle.read_bytes()
    restored = tmp_path / "restored"
    status = restore_encrypted_session(
        bundle, restored, passphrase="correct horse battery staple"
    )
    assert status["authenticated"] is True
    assert status["format_version"] == 2
    assert status["authentication_status"] == "authenticated_complete"
    assert (restored / "data.csv").read_bytes() == b"x,y\n1,2\n"
    assert (restored / ".sift" / "workflow.json").read_text(encoding="utf-8") == '{"approved":true}'
    if os.name != "nt":
        assert stat.S_IMODE(restored.stat().st_mode) == 0o700
        assert stat.S_IMODE((restored / ".sift").stat().st_mode) == 0o700

    tampered = tmp_path / "tampered.siftenc"
    raw = bytearray(bundle.read_bytes())
    raw[-1] ^= 1
    tampered.write_bytes(raw)
    with pytest.raises(Exception):
        restore_encrypted_session(
            tampered,
            tmp_path / "must_not_restore",
            passphrase="correct horse battery staple",
        )
    assert not (tmp_path / "must_not_restore").exists()


def test_failed_restore_preserves_preexisting_empty_destination(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "data.txt").write_text("sensitive")
    bundle = tmp_path / "session.siftenc"
    encrypt_session_bundle(session, bundle, passphrase="correct password")
    destination = tmp_path / "existing-empty"
    destination.mkdir()

    with pytest.raises(Exception):
        restore_encrypted_session(
            bundle, destination, passphrase="wrong password",
        )

    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_encrypted_session_rejects_complete_record_truncation(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    # Ensure the archive spans multiple data records.  Truncating exactly after
    # a valid first record used to be accepted because every remaining byte was
    # individually authenticated, but stream completion was not.
    (session / "large.bin").write_bytes(
        b"x" * (security_assurance._ENC_CHUNK_SIZE + 1024)
    )
    bundle = tmp_path / "session.siftenc"
    encrypt_session_bundle(session, bundle, passphrase="correct horse battery staple")
    payload = bundle.read_bytes()
    records = _encrypted_record_offsets(payload)
    assert [row[0] for row in records[:2]] == [
        security_assurance._ENC_RECORD_DATA,
        security_assurance._ENC_RECORD_DATA,
    ]
    truncated = tmp_path / "complete-record-truncation.siftenc"
    truncated.write_bytes(payload[: records[0][2]])

    destination = tmp_path / "must_not_restore"
    with pytest.raises(ValueError, match="authenticated final record"):
        restore_encrypted_session(
            truncated,
            destination,
            passphrase="correct horse battery staple",
        )
    assert not destination.exists()


def test_encrypted_session_rejects_removed_terminal_record(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "data.txt").write_text("complete data")
    bundle = tmp_path / "session.siftenc"
    encrypt_session_bundle(session, bundle, passphrase="correct horse battery staple")
    payload = bundle.read_bytes()
    records = _encrypted_record_offsets(payload)
    assert records[-1][0] == security_assurance._ENC_RECORD_FINAL
    without_terminal = tmp_path / "without-terminal.siftenc"
    without_terminal.write_bytes(payload[: records[-1][1]])

    destination = tmp_path / "must_not_restore"
    with pytest.raises(ValueError, match="authenticated final record"):
        restore_encrypted_session(
            without_terminal,
            destination,
            passphrase="correct horse battery staple",
        )
    assert not destination.exists()


def test_encrypted_session_rejects_bytes_after_terminal_record(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "data.txt").write_text("complete data")
    bundle = tmp_path / "session.siftenc"
    encrypt_session_bundle(session, bundle, passphrase="correct horse battery staple")
    trailing = tmp_path / "trailing.siftenc"
    trailing.write_bytes(bundle.read_bytes() + b"unexpected")

    destination = tmp_path / "must_not_restore"
    with pytest.raises(ValueError, match="trailing data"):
        restore_encrypted_session(
            trailing,
            destination,
            passphrase="correct horse battery staple",
        )
    assert not destination.exists()


def test_legacy_v1_restore_requires_opt_in_and_never_claims_full_authentication(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "legacy.siftenc"
    passphrase = "correct horse battery staple"
    _write_legacy_v1_bundle(bundle, passphrase=passphrase)

    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="allow_legacy_v1"):
        restore_encrypted_session(bundle, rejected, passphrase=passphrase)
    assert not rejected.exists()

    restored = tmp_path / "restored"
    status = restore_encrypted_session(
        bundle,
        restored,
        passphrase=passphrase,
        allow_legacy_v1=True,
    )
    assert (restored / "legacy.txt").read_bytes() == b"legacy content\n"
    assert status["format_version"] == 1
    assert status["authenticated"] is False
    assert (
        status["authentication_status"]
        == "legacy_chunks_authenticated_completion_unverified"
    )


def test_archive_extraction_enforces_member_limit_while_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "members.tar"
    with tarfile.open(archive_path, "w") as archive:
        for index in range(3):
            content = str(index).encode("ascii")
            member = tarfile.TarInfo(f"{index}.txt")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    monkeypatch.setattr(security_assurance, "_MAX_ARCHIVE_MEMBERS", 2)

    with pytest.raises(ValueError, match="member-count safety limit"):
        security_assurance._extract_session_tar(
            archive_path,
            tmp_path / "restored",
        )


def test_archive_creation_enforces_member_limit_during_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    for index in range(3):
        (session / f"{index}.txt").write_text(str(index))
    monkeypatch.setattr(security_assurance, "_MAX_ARCHIVE_MEMBERS", 2)
    output = tmp_path / "must_not_exist.siftenc"

    with pytest.raises(ValueError, match="member-count safety limit"):
        encrypt_session_bundle(
            session,
            output,
            passphrase="correct horse battery staple",
        )
    assert not output.exists()


def test_encrypted_session_rejects_wrong_password_and_unsafe_paths(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    (session / "a.txt").write_text("a")
    bundle = tmp_path / "session.siftenc"
    encrypt_session_bundle(session, bundle, passphrase="a very long password")
    with pytest.raises(Exception):
        restore_encrypted_session(
            bundle, tmp_path / "wrong", passphrase="a different password"
        )
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(ValueError):
        _safe_tar_target(tmp_path, "../escape")
    with pytest.raises(ValueError):
        _safe_tar_target(tmp_path, "/absolute")


def test_signed_provenance_detects_any_file_change(tmp_path: Path) -> None:
    bundle = tmp_path / "export"
    bundle.mkdir()
    (bundle / "results.json").write_text('{"estimate":1.25}')
    (bundle / "METHODS.md").write_text("Fixed seed: 7")
    key = generate_ed25519_private_key()
    signed = sign_provenance_export(bundle, private_key_b64=key, key_id="test")
    assert Path(signed["path"]).name == "provenance.signature.json"
    assert verify_provenance_signature(bundle) == {
        "valid": True,
        "files": 2,
        "key_id": "test",
    }
    (bundle / "results.json").write_text('{"estimate":9.99}')
    invalid = verify_provenance_signature(bundle)
    assert invalid["valid"] is False
    assert "manifest" in invalid["reason"]


def test_sbom_is_cyclonedx_and_bound_to_lockfile(tmp_path: Path) -> None:
    output = tmp_path / "sift.cdx.json"
    result = generate_cyclonedx_sbom(ROOT, output)
    document = json.loads(output.read_text(encoding="utf-8"))
    assert result["components"] > 100
    assert document["bomFormat"] == "CycloneDX"
    assert document["specVersion"] == "1.6"
    assert document["metadata"]["component"]["name"] == "sift"
    assert len(document["metadata"]["properties"][1]["value"]) == 64
    assert any(row["name"] == "cryptography" for row in document["components"])
    assert document["dependencies"][0]["dependsOn"]


def test_local_secret_and_static_scans_report_locations_without_values(tmp_path: Path) -> None:
    source = tmp_path / "src" / "pkg"
    source.mkdir(parents=True)
    secret = "sk-" + "x" * 32
    (source / "bad.py").write_text(
        f'api_key = "{secret}"\n'
        'subprocess.run("x", shell=True)\n'
        'requests.get("https://example.test", verify=False)\n'
        'ssl._create_unverified_context()\n'
        'context.check_hostname = False\n'
        'context.verify_mode = ssl.CERT_NONE\n'
        'hashlib.sha1(b"security-sensitive")\n'
        'yaml.unsafe_load("x")\n'
    )
    secrets = scan_source_secrets(tmp_path)
    static = scan_python_static_security(tmp_path)
    assert secrets[0]["path"] == "src/pkg/bad.py"
    assert secret not in json.dumps(secrets)
    categories = {row["category"] for row in static}
    assert {
        "subprocess_shell_true", "tls_verification_disabled",
        "tls_hostname_verification_disabled",
        "tls_certificate_verification_disabled",
        "weak_hash_without_integrity_marker", "unsafe_yaml_load",
    } <= categories


def test_project_local_high_signal_scans_are_clean() -> None:
    assert scan_source_secrets(ROOT) == []
    assert scan_python_static_security(ROOT) == []


def test_release_binding_requires_exact_current_wheel_sdist_source_and_lock(
    tmp_path: Path,
) -> None:
    root = _release_fixture(tmp_path)
    binding = security_release_binding(root)
    assert binding["status"] == "ready"
    assert binding["release_packages_match_source"] is True
    assert binding["sdist_matches_source_tree"] is True
    assert binding["artifact_metadata_match"] is True
    assert binding["declared_dependencies_covered_by_lock"] is True
    assert len(binding["source_tree_sha256"]) == 64
    assert len(binding["uv_lock_sha256"]) == 64
    assert len(binding["wheel"]["sha256"]) == 64
    assert len(binding["sdist"]["sha256"]) == 64
    assert len(binding["binding_sha256"]) == 64

    (root / "src/sift/__init__.py").write_text('__version__ = "changed"\n')
    stale = security_release_binding(root)
    assert stale["status"] == "stale"
    assert stale["release_packages_match_source"] is False
    assert stale["sdist_matches_source_tree"] is False


def test_release_binding_ignores_post_build_derived_database_evidence(
    tmp_path: Path,
) -> None:
    root = _release_fixture(tmp_path)
    report = root / "docs" / "live_database_certification.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"status":"blocked","wheel":"first"}\n')

    # The fixture artifacts predate this generated report.  Updating the
    # report again must not create an impossible source/artifact cycle.
    assert security_release_binding(root)["status"] == "ready"
    report.write_text('{"status":"blocked","wheel":"second"}\n')
    assert security_release_binding(root)["status"] == "ready"


def test_release_binding_ignores_generated_bundled_python_runtime(
    tmp_path: Path,
) -> None:
    root = _release_fixture(tmp_path)
    initial = security_release_binding(root)
    assert initial["status"] == "ready"

    generated = root / "packaging" / "vendor" / "python"
    generated.mkdir(parents=True)
    (generated / "third-party.py").write_text("print('dependency')\n")
    try:
        (generated / "python").symlink_to("bin/python3")
    except OSError:
        pass

    refreshed = security_release_binding(root)
    assert refreshed["status"] == "ready"
    assert refreshed["binding_sha256"] == initial["binding_sha256"]


@pytest.mark.parametrize("requirement", ["alpha>=9", "!!!", "alpha @ https://example.test/a.whl"])
def test_release_binding_rejects_unmet_malformed_or_direct_requirements(
    tmp_path: Path, requirement: str,
) -> None:
    root = _release_fixture(tmp_path, requires_dist=requirement)
    binding = security_release_binding(root)
    assert binding["status"] == "stale"
    assert binding["declared_dependencies_covered_by_lock"] is False


def test_sdist_scan_rejects_traversal_and_duplicate_archive_paths(
    tmp_path: Path,
) -> None:
    root = _release_fixture(tmp_path)
    sdist = root / "dist/sift-0.1.0.tar.gz"

    for names in (
        ("../escape.py",),
        ("sift-0.1.0/src/sift/a.py", "sift-0.1.0/src/sift/a.py"),
    ):
        with tarfile.open(sdist, "w:gz") as archive:
            for name in names:
                content = b"pass\n"
                member = tarfile.TarInfo(name)
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        findings = scan_source_secrets(root, sdist_path=sdist)
        assert findings == [{
            "severity": "high",
            "category": "unsafe_or_unreadable_sdist",
            "path": sdist.name,
            "line": 0,
        }]


def test_exact_sdist_scans_docs_scripts_and_never_weakens_test_findings(
    tmp_path: Path,
) -> None:
    root = _release_fixture(tmp_path)
    sdist = root / "dist/sift-0.1.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        surfaces = {
            "src/sift/__init__.py": b'__version__ = "0.1.0"\n',
            "scripts/bad.py": b'import subprocess\nsubprocess.run("x", shell=True)\n',
            "docs/config.md": b'api_key = "abcdefghijklmnop123456"\n',
            "tests/test_fake.py": b'secret = "sk-this-is-a-test-fixture-1234567890"\nexec("pass")\n',
            "pyproject.toml": (root / "pyproject.toml").read_bytes(),
            "CHANGELOG.md": (root / "CHANGELOG.md").read_bytes(),
            "SECURITY.md": (root / "SECURITY.md").read_bytes(),
        }
        for relative, content in surfaces.items():
            member = tarfile.TarInfo(f"sift-0.1.0/{relative}")
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    secrets = scan_source_secrets(root, sdist_path=sdist)
    static = scan_python_static_security(root, sdist_path=sdist)
    assert any(
        row["path"] == "docs/config.md" and row["severity"] == "high"
        for row in secrets
    )
    assert any(
        row["path"] == "tests/test_fake.py" and row["severity"] == "high"
        for row in secrets
    )
    assert any(
        row["path"] == "scripts/bad.py" and row["category"] == "subprocess_shell_true"
        for row in static
    )
    assert any(
        row["path"] == "tests/test_fake.py" and row["severity"] == "high"
        for row in static
    )


def test_dependency_scan_uses_every_exact_lock_pin_not_launcher_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release_fixture(tmp_path)
    monkeypatch.setattr(security_assurance, "_find_tool", lambda name: "/tools/pip-audit")
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "pip-audit 2.10.0\n", "")
        requirement = Path(command[command.index("--requirement") + 1])
        dependencies = []
        for line in requirement.read_text(encoding="utf-8").splitlines():
            name, version = line.split("==", 1)
            dependencies.append({"name": name, "version": version, "vulns": []})
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"dependencies": dependencies}), "",
        )

    monkeypatch.setattr(security_assurance, "run_bounded_capture", fake_run)
    report = run_dependency_vulnerability_scan(root)
    assert report["status"] == "pass"
    assert report["packages_expected"] == report["packages_audited"] == 3
    assert report["manifests_audited"] == 2
    assert report["scanner_version"] == "pip-audit 2.10.0"
    assert len(report["wheel_sha256"]) == len(report["sdist_sha256"]) == 64
    scan_commands = [command for command in commands if "--requirement" in command]
    assert scan_commands
    assert all("--local" not in command for command in scan_commands)
    assert all("--no-deps" in command and "--disable-pip" in command for command in scan_commands)


def test_dependency_scan_distinguishes_findings_errors_and_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release_fixture(tmp_path)
    monkeypatch.setattr(security_assurance, "_find_tool", lambda name: "/tools/pip-audit")

    def finding_run(command, **_kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "pip-audit 2.10.0", "")
        requirement = Path(command[command.index("--requirement") + 1])
        dependencies = []
        for index, line in enumerate(requirement.read_text(encoding="utf-8").splitlines()):
            name, version = line.split("==", 1)
            vulnerabilities = [{"id": "CVE-TEST", "aliases": []}] if index == 0 else []
            dependencies.append({"name": name, "version": version, "vulns": vulnerabilities})
        return subprocess.CompletedProcess(command, 1, json.dumps({"dependencies": dependencies}), "")

    monkeypatch.setattr(security_assurance, "run_bounded_capture", finding_run)
    assert run_dependency_vulnerability_scan(root)["status"] == "findings"

    def network_error(command, **_kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "pip-audit 2.10.0", "")
        return subprocess.CompletedProcess(command, 1, "", "network unavailable")

    monkeypatch.setattr(security_assurance, "run_bounded_capture", network_error)
    failed = run_dependency_vulnerability_scan(root)
    assert failed["status"] == "scanner_error"
    assert "valid JSON" in failed["reason"]

    def malformed_inventory(command, **_kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "pip-audit 2.10.0", "")
        requirement = Path(command[command.index("--requirement") + 1])
        dependencies = []
        for line in requirement.read_text(encoding="utf-8").splitlines():
            name, version = line.split("==", 1)
            dependencies.append({"name": name, "version": version, "vulns": []})
        dependencies[0]["vulns"] = "not-a-list"
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"dependencies": dependencies}), "",
        )

    monkeypatch.setattr(security_assurance, "run_bounded_capture", malformed_inventory)
    malformed = run_dependency_vulnerability_scan(root)
    assert malformed["status"] == "scanner_error"
    assert "malformed vulnerability inventory" in malformed["reason"]

    monkeypatch.setattr(security_assurance, "_find_tool", lambda name: None)
    assert run_dependency_vulnerability_scan(root)["status"] == "unavailable"


def test_bandit_medium_findings_and_malformed_output_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release_fixture(tmp_path)
    monkeypatch.setattr(security_assurance, "_find_tool", lambda name: "/tools/bandit")

    def medium_finding(command, **_kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "bandit 1.8.6", "")
        payload = {"results": [{
            "issue_severity": "MEDIUM", "test_id": "B999",
            "filename": str(root / "src/sift/__init__.py"),
            "line_number": 1, "issue_text": "review required",
        }]}
        return subprocess.CompletedProcess(command, 1, json.dumps(payload), "")

    monkeypatch.setattr(security_assurance, "run_bounded_capture", medium_finding)
    assert run_bandit_static_scan(root)["status"] == "findings"

    def malformed(command, **_kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "bandit 1.8.6", "")
        return subprocess.CompletedProcess(
            command, 0, json.dumps({"results": ["invalid"]}), "",
        )

    monkeypatch.setattr(security_assurance, "run_bounded_capture", malformed)
    assert run_bandit_static_scan(root)["status"] == "scanner_error"


def test_bandit_excludes_generated_bundled_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _release_fixture(tmp_path)
    generated = root / "packaging" / "vendor" / "python"
    generated.mkdir(parents=True)
    (generated / "dependency.py").write_text("eval('third-party fixture')\n")
    observed: list[list[str]] = []
    monkeypatch.setattr(security_assurance, "_find_tool", lambda name: "/tools/bandit")

    def clean(command, **_kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "bandit 1.8.6", "")
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, json.dumps({"results": []}), "")

    monkeypatch.setattr(security_assurance, "run_bounded_capture", clean)

    assert run_bandit_static_scan(root)["status"] == "pass"
    assert len(observed) == 1
    command = observed[0]
    assert "-x" in command
    exclusions = command[command.index("-x") + 1].split(",")
    assert str(generated) in exclusions


def test_security_qualification_never_claims_independent_pentest() -> None:
    report = security_qualification_report(ROOT)
    assert report["secret_scan"]["status"] == "pass"
    assert report["static_analysis"]["status"] == "pass"
    assert report["independent_penetration_test"]["status"] == "external_required"
    assert report["confidential_production_ready"] is False
    assert "independent_third_party_penetration_test_not_supplied" in report["blockers"]
    assert len(report["qualification_binding"]["binding_sha256"]) == 64


def test_security_qualification_can_run_local_static_without_public_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        security_assurance, "run_bandit_static_scan",
        lambda _root: {"status": "pass", "tool": "bandit", "scanner_version": "test"},
    )
    monkeypatch.setattr(
        security_assurance, "run_dependency_vulnerability_scan",
        lambda _root: pytest.fail("public dependency lookup must not run"),
    )
    report = security_qualification_report(ROOT, run_static_scan=True)
    assert report["bandit_static_analysis"]["status"] == "pass"
    assert report["dependency_scan"]["status"] == "not_run"
    assert "independent_static_scan_not_clear" not in report["blockers"]


def test_security_cli_requires_explicit_public_dependency_disclosure_approval() -> None:
    completed = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts/security_qualification.py"),
            "--run-external",
        ],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 2
    assert "--authorize-public-dependency-disclosure" in completed.stderr
