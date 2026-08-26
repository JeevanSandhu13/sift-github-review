from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from sift.subprocess_safety import (
    SubprocessOutputLimitExceeded,
    _terminate_windows_process_tree,
    run_bounded_capture,
)


def test_bounded_capture_returns_normal_completed_process() -> None:
    result = run_bounded_capture(
        [sys.executable, "-c", "import sys;print('ok');sys.stderr.write('note')"],
        timeout=5,
    )
    assert result.returncode == 0
    assert result.stdout == "ok\n"
    assert result.stderr == "note"


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_bounded_capture_stops_a_flood(stream: str) -> None:
    target = "sys.stdout.buffer" if stream == "stdout" else "sys.stderr.buffer"
    with pytest.raises(SubprocessOutputLimitExceeded) as caught:
        run_bounded_capture(
            [sys.executable, "-c", f"import sys\nwhile True:{target}.write(b'x'*65536);{target}.flush()"],
            timeout=10,
            stdout_limit=128 * 1024,
            stderr_limit=128 * 1024,
        )
    assert len(caught.value.stdout) <= 128 * 1024
    assert len(caught.value.stderr) <= 128 * 1024


def test_bounded_capture_preserves_timeout_shape() -> None:
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_bounded_capture(
            [sys.executable, "-c", "import time;print('early',flush=True);time.sleep(30)"],
            timeout=0.1,
        )
    assert "early" in str(caught.value.output)


def test_bounded_capture_check_raises_with_bounded_diagnostics() -> None:
    with pytest.raises(subprocess.CalledProcessError) as caught:
        run_bounded_capture(
            [sys.executable, "-c", "import sys;sys.stderr.write('bad');raise SystemExit(7)"],
            timeout=5,
            check=True,
        )
    assert caught.value.returncode == 7
    assert caught.value.stderr == "bad"


def test_windows_teardown_kills_descendants_before_root(monkeypatch) -> None:
    events: list[str] = []

    class FakeProcess:
        def __init__(self, pid: int, name: str = "root") -> None:
            self.pid = pid
            self.name = name

        def children(self, *, recursive: bool) -> list[FakeProcess]:
            assert recursive is True
            return [FakeProcess(11, "child"), FakeProcess(12, "grandchild")]

        def kill(self) -> None:
            events.append(self.name)

    class FakePsutilError(Exception):
        pass

    fake_psutil = SimpleNamespace(
        Process=lambda pid: FakeProcess(pid),
        wait_procs=lambda processes, timeout: events.append(
            f"wait:{timeout}:{len(processes)}"
        ),
        Error=FakePsutilError,
        NoSuchProcess=FakePsutilError,
        AccessDenied=FakePsutilError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    popen = SimpleNamespace(pid=10, kill=lambda: events.append("popen"))

    _terminate_windows_process_tree(popen)  # type: ignore[arg-type]

    assert events == ["grandchild", "child", "root", "wait:1:3", "popen"]


def test_windows_teardown_falls_back_for_an_invalid_pid(monkeypatch) -> None:
    """A stale/substituted handle must still reach the direct kill fallback."""
    events: list[str] = []

    fake_psutil = SimpleNamespace(
        Process=lambda _pid: (_ for _ in ()).throw(ValueError("invalid pid")),
        Error=RuntimeError,
        NoSuchProcess=RuntimeError,
        AccessDenied=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    popen = SimpleNamespace(pid=-1, kill=lambda: events.append("popen"))

    _terminate_windows_process_tree(popen)  # type: ignore[arg-type]

    assert events == ["popen"]
