"""Focused tests for bounded, redacted, policy-controlled diagnostics."""

from __future__ import annotations

import os
import sys

from sift.diagnostics import (
    RedactingLogStream,
    configure_diagnostic_logging,
    prune_diagnostic_logs,
    redact_diagnostic_text,
)
from sift.enterprise_policy import EnterprisePolicy


def test_diagnostic_redaction_covers_credentials_and_local_paths() -> None:
    secret = "sk-" + "ant-abcdefghijklmnopqrstuvwxyz123456"
    raw = (
        f"api_key={secret} password=hunter2 "
        "Authorization: Bearer abc.def.ghi "
        "at /Users/researcher/confidential/study.csv"
    )
    safe = redact_diagnostic_text(raw)
    assert secret not in safe
    assert "hunter2" not in safe
    assert "abc.def.ghi" not in safe
    assert "/Users/researcher" not in safe
    assert "[redacted-credential]" in safe


def test_diagnostic_redaction_covers_signed_auth_and_cookie_headers() -> None:
    safe = redact_diagnostic_text(
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE, "
        "Signature=highly-sensitive-signature\n"
        "Set-Cookie: session=browser-secret; Secure\n"
        "ordinary diagnostic"
    )
    assert "AKIAEXAMPLE" not in safe
    assert "highly-sensitive-signature" not in safe
    assert "browser-secret" not in safe
    assert "ordinary diagnostic" in safe


def test_private_key_block_is_never_logged() -> None:
    raw = (
        "-----BEGIN " + "PRIVATE KEY-----\n"
        "highly-sensitive-material\n"
        "-----END PRIVATE KEY-----"
    )
    safe = redact_diagnostic_text(raw)
    assert "highly-sensitive-material" not in safe
    assert "[redacted-private-key]" in safe


def test_one_line_private_key_does_not_hide_later_diagnostics(tmp_path) -> None:
    path = tmp_path / "sift-test.log"
    path.touch()
    stream = RedactingLogStream(path, max_bytes=4096)
    stream.write(
        "-----BEGIN " + "PRIVATE KEY-----secret-----END PRIVATE KEY-----\n"
    )
    stream.write("ordinary diagnostic remains visible\n")

    content = path.read_text(encoding="utf-8")
    assert "secret" not in content
    assert "ordinary diagnostic remains visible" in content


def test_prune_removes_expired_and_bounds_total_bytes(tmp_path) -> None:
    now = 2_000_000_000.0
    expired = tmp_path / "sift-2020-01-01.log"
    older = tmp_path / "sift-2033-01-01.log"
    newest = tmp_path / "sift-2033-01-02.log"
    unrelated = tmp_path / "keep.txt"
    expired.write_bytes(b"expired")
    older.write_bytes(b"a" * 80)
    newest.write_bytes(b"b" * 80)
    unrelated.write_text("not a Sift log", encoding="utf-8")
    os.utime(expired, (now - 10 * 86_400, now - 10 * 86_400))
    os.utime(older, (now - 100, now - 100))
    os.utime(newest, (now - 10, now - 10))

    prune_diagnostic_logs(
        tmp_path, retention_days=7, total_bytes=100, now=now,
    )

    assert not expired.exists()
    assert not older.exists()
    assert newest.exists()
    assert newest.stat().st_size <= 100
    assert unrelated.exists()


def test_prune_ignores_symlinks(tmp_path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("must remain", encoding="utf-8")
    link = tmp_path / "sift-linked.log"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        return

    prune_diagnostic_logs(tmp_path, retention_days=1, total_bytes=1)

    assert link.is_symlink()
    assert outside.read_text(encoding="utf-8") == "must remain"


def test_redacting_stream_keeps_newest_output_within_file_ceiling(tmp_path) -> None:
    path = tmp_path / "sift-test.log"
    path.touch()
    stream = RedactingLogStream(path, max_bytes=96)
    for index in range(20):
        stream.write(f"line-{index:02d}-" + ("x" * 20) + "\n")

    content = path.read_text(encoding="utf-8")
    assert path.stat().st_size <= 96
    assert "line-19" in content
    assert "line-00" not in content


def test_redacting_stream_catches_secrets_split_across_writes(tmp_path) -> None:
    path = tmp_path / "sift-test.log"
    path.touch()
    stream = RedactingLogStream(path, max_bytes=4096)
    stream.write("api_key=")
    stream.write("sk-" + "ant-abcdefghijklmnopqrstuvwxyz123456")
    stream.write("\n")

    content = path.read_text(encoding="utf-8")
    assert "abcdefghijklmnopqrstuvwxyz123456" not in content
    assert "[redacted-credential]" in content


def test_enterprise_policy_can_disable_diagnostics_without_side_effects(
    tmp_path,
) -> None:
    before_stdout = sys.stdout
    before_stderr = sys.stderr
    policy = EnterprisePolicy(allow_local_diagnostics=False)

    result = configure_diagnostic_logging(log_dir=tmp_path, enterprise=policy)

    assert result is None
    assert sys.stdout is before_stdout
    assert sys.stderr is before_stderr
    assert list(tmp_path.iterdir()) == []


def test_configure_refuses_precreated_daily_log_symlink(
    tmp_path, monkeypatch,
) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("do not modify", encoding="utf-8")
    link = tmp_path / "sift-2099-01-02.log"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        return
    monkeypatch.setattr(
        "sift.diagnostics.time.strftime",
        lambda _format: "sift-2099-01-02.log",
    )
    before_stdout = sys.stdout
    before_stderr = sys.stderr

    result = configure_diagnostic_logging(
        log_dir=tmp_path,
        enterprise=EnterprisePolicy(allow_local_diagnostics=True),
    )

    assert result is None
    assert sys.stdout is before_stdout
    assert sys.stderr is before_stderr
    assert link.is_symlink()
    assert target.read_text(encoding="utf-8") == "do not modify"


def test_configure_applies_enterprise_byte_ceiling_and_redacts(tmp_path) -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    policy = EnterprisePolicy(
        allow_local_diagnostics=True,
        diagnostic_retention_days_ceiling=1,
        diagnostic_log_bytes_ceiling=256,
    )
    try:
        path = configure_diagnostic_logging(log_dir=tmp_path, enterprise=policy)
        assert path is not None
        sys.stdout.write(
            "password=do-not-store-this-value " + ("x" * 400) + "\n"
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr

    assert path is not None
    content = path.read_text(encoding="utf-8")
    assert "do-not-store-this-value" not in content
    assert path.stat().st_size <= 256


def test_repeated_configuration_is_idempotent(tmp_path) -> None:
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    try:
        first_path = configure_diagnostic_logging(
            log_dir=tmp_path,
            enterprise=EnterprisePolicy(allow_local_diagnostics=True),
        )
        first_stdout = sys.stdout
        first_stderr = sys.stderr
        second_path = configure_diagnostic_logging(
            log_dir=tmp_path,
            enterprise=EnterprisePolicy(allow_local_diagnostics=True),
        )
        assert second_path == first_path
        assert sys.stdout is first_stdout
        assert sys.stderr is first_stderr
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
