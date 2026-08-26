"""Portable cancellation tests for the runner's subprocess teardown."""

from __future__ import annotations

import signal
from types import SimpleNamespace

from sift import runner


class _JobProcess:
    """AppContainerProcess-shaped fake: deliberately no poll or wait."""

    pid = 4242

    def __init__(self) -> None:
        self.kill_calls = 0

    def kill(self) -> None:
        self.kill_calls += 1


class _PopenProcess(_JobProcess):
    def __init__(self, returncode=None) -> None:
        super().__init__()
        self.returncode = returncode
        self.wait_calls: list[int] = []

    def poll(self):
        return self.returncode

    def wait(self, timeout: int) -> None:
        self.wait_calls.append(timeout)


def test_kill_proc_uses_job_kill_without_posix_process_apis(monkeypatch) -> None:
    """Windows AppContainer wrappers must reach their Job-object kill."""
    proc = _JobProcess()
    # A private namespace avoids mutating Python's process-wide ``os`` module
    # while faithfully representing Windows, where these APIs do not exist.
    fake_os = SimpleNamespace()
    monkeypatch.setattr(runner, "os", fake_os)

    runner._kill_proc_quietly(proc)

    assert proc.kill_calls == 1


def test_kill_proc_uses_windows_tree_teardown_for_ordinary_popen(monkeypatch) -> None:
    proc = _PopenProcess()
    calls: list[object] = []
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="nt"))
    import sift.subprocess_safety as subprocess_safety

    monkeypatch.setattr(
        subprocess_safety,
        "_terminate_windows_process_tree",
        lambda candidate: calls.append(candidate),
    )

    runner._kill_proc_quietly(proc)

    assert calls == [proc]
    assert proc.kill_calls == 0
    assert proc.wait_calls == [2]


def test_kill_proc_prefers_posix_process_group(monkeypatch) -> None:
    proc = _PopenProcess()
    calls: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    fake_os = SimpleNamespace(
        getpgid=lambda pid: pid + 10,
        killpg=lambda pgid, sig: calls.append((pgid, sig)),
    )
    monkeypatch.setattr(runner, "os", fake_os)

    runner._kill_proc_quietly(proc)

    assert calls == [(proc.pid + 10, signal.SIGKILL)]
    assert proc.kill_calls == 0
    assert proc.wait_calls == [2]


def test_kill_proc_falls_back_when_group_kill_fails(monkeypatch) -> None:
    proc = _PopenProcess()

    def _denied(_pgid, _sig) -> None:
        raise PermissionError("denied")

    fake_os = SimpleNamespace(getpgid=lambda pid: pid, killpg=_denied)
    monkeypatch.setattr(runner, "os", fake_os)

    runner._kill_proc_quietly(proc)

    assert proc.kill_calls == 1
    assert proc.wait_calls == [2]


def test_kill_proc_leaves_already_exited_process_alone(monkeypatch) -> None:
    proc = _PopenProcess(returncode=0)
    fake_os = SimpleNamespace(
        getpgid=lambda _pid: (_ for _ in ()).throw(AssertionError("unexpected")),
        killpg=lambda _pgid, _sig: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    monkeypatch.setattr(runner, "os", fake_os)

    runner._kill_proc_quietly(proc)

    assert proc.kill_calls == 0
    assert proc.wait_calls == []


def test_kill_proc_cleans_tracker_even_when_direct_process_exited(
    monkeypatch,
) -> None:
    """A successful root status does not imply its descendants exited."""
    proc = _PopenProcess(returncode=0)
    tracked: list[object] = []
    monkeypatch.setattr(
        runner,
        "terminate_tracked_process_tree",
        lambda candidate: tracked.append(candidate) or True,
    )
    monkeypatch.setattr(runner, "os", SimpleNamespace())

    runner._kill_proc_quietly(proc)

    assert tracked == [proc]
    assert proc.kill_calls == 0
    assert proc.wait_calls == [2]


def test_kill_proc_does_not_trust_a_failing_poll(monkeypatch) -> None:
    proc = _PopenProcess()

    def _broken_poll():
        raise RuntimeError("status unavailable")

    proc.poll = _broken_poll  # type: ignore[method-assign]
    monkeypatch.setattr(runner, "os", SimpleNamespace())

    runner._kill_proc_quietly(proc)

    assert proc.kill_calls == 1
    assert proc.wait_calls == [2]
