from __future__ import annotations

from pathlib import Path

from sift.provider.usage_log import append_usage_line


def test_usage_log_refuses_script_planted_symlink(tmp_path: Path) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    (tmp_path / ".sift-usage.log").symlink_to(target)

    append_usage_line(tmp_path, "diagnostic")

    assert target.read_text(encoding="utf-8") == "preserve"


def test_usage_log_refuses_script_planted_lock_symlink(tmp_path: Path) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    (tmp_path / ".sift-usage.lock").symlink_to(target)

    append_usage_line(tmp_path, "diagnostic")

    assert target.read_text(encoding="utf-8") == "preserve"
    assert not (tmp_path / ".sift-usage.log").exists()
