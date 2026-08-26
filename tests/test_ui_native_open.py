"""Cross-platform contracts for opening Sift-managed files and folders."""

from __future__ import annotations

import subprocess
from pathlib import Path

import sift.ui as ui
from sift.ui import SiftBridge


def test_windows_python_uses_edit_verb(
    tmp_path: Path, monkeypatch,
) -> None:
    script = tmp_path / "review.py"
    script.write_text("print('must not execute')\n", encoding="utf-8")
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(
        ui.os, "startfile", lambda path, operation: calls.append((path, operation)),
        raising=False,
    )

    result = SiftBridge(cwd=tmp_path).open_path(str(script), "run_python")

    assert result == {"ok": True}
    assert calls == [(str(script.resolve()), "edit")]


def test_windows_folder_uses_open_verb(
    tmp_path: Path, monkeypatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(ui.sys, "platform", "win32")
    monkeypatch.setattr(
        ui.os, "startfile", lambda path, operation: calls.append((path, operation)),
        raising=False,
    )

    result = SiftBridge(cwd=tmp_path).open_path(str(tmp_path))

    assert result == {"ok": True}
    assert calls == [(str(tmp_path.resolve()), "open")]


def test_linux_uses_xdg_open_and_propagates_failure(
    tmp_path: Path, monkeypatch,
) -> None:
    graph = tmp_path / "plot.png"
    graph.write_bytes(b"png")
    monkeypatch.setattr(ui.sys, "platform", "linux")
    monkeypatch.setattr(
        ui.shutil, "which",
        lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None,
    )

    class Failed:
        returncode = 3

    class Process:
        @staticmethod
        def wait(timeout):
            assert timeout == 1
            return Failed.returncode

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: Process())

    result = SiftBridge(cwd=tmp_path).open_path(str(graph))

    assert result == {
        "ok": False,
        "reason": "native opener exited with status 3",
    }


def test_linux_python_requires_explicit_editor(
    tmp_path: Path, monkeypatch,
) -> None:
    script = tmp_path / "review.py"
    script.write_text("print('must not execute')\n", encoding="utf-8")
    monkeypatch.setattr(ui.sys, "platform", "linux")
    monkeypatch.setattr(ui.shutil, "which", lambda _name: None)

    result = SiftBridge(cwd=tmp_path).open_path(str(script), "run_python")

    assert result == {
        "ok": False,
        "reason": "no supported Linux text editor was found",
    }


def test_open_path_rejects_unknown_mode(tmp_path: Path) -> None:
    target = tmp_path / "plot.png"
    target.write_bytes(b"png")

    assert SiftBridge(cwd=tmp_path).open_path(str(target), "execute") == {
        "ok": False,
        "reason": "unsupported open mode",
    }
