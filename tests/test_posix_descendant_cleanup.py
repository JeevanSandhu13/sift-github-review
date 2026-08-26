"""Portable POSIX coverage for descendants which escape process groups."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from sift import env_detect, executor, process_tree, runner

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="POSIX process identity and setsid coverage",
)


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # Linux keeps an adopted child visible briefly as a zombie.  It cannot do
    # work and is semantically dead, so don't make CI depend on init's reap
    # latency.  macOS has no procfs and normally reaps this promptly.
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError:
        return True
    tail = raw[raw.rfind(")") + 2 :].split()
    return not tail or tail[0] != "Z"


def _wait_until_stopped(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            return True
        time.sleep(0.05)
    return not _process_is_running(pid)


def _spawn_tree_with_setsid_child(
    tmp_path: Path,
) -> tuple[subprocess.Popen[str], int, tuple[str, str]]:
    pid_path = tmp_path / "escaped.pid"
    marker = (executor._PROCESS_TREE_MARKER_ENV_VAR, uuid.uuid4().hex)
    child_code = (
        "import os,time\n"
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import os,subprocess,sys,time\n"
        # The real runtime consumes this result-authentication token before
        # user code.  Cleanup must therefore rely on its separate marker.
        "os.environ.pop('SIFT_RUN_TOKEN', None)\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True)\n"
        "time.sleep(60)\n"
    )
    env = dict(os.environ)
    env["SIFT_RUN_TOKEN"] = uuid.uuid4().hex
    env[marker[0]] = marker[1]
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        env=env,
        text=True,
        start_new_session=True,
    )
    tracker = process_tree.attach_posix_descendant_tracker(proc, marker=marker)
    assert tracker is not None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if pid_path.exists() and pid_path.read_text(encoding="utf-8").strip():
            return proc, int(pid_path.read_text(encoding="utf-8")), marker
        time.sleep(0.05)
    process_tree.terminate_tracked_process_tree(proc)
    proc.wait(timeout=5)
    pytest.fail("setsid child did not publish its PID")


def test_tracker_kills_child_which_escaped_with_setsid(tmp_path: Path) -> None:
    """Runner cancellation reaches a child outside the root's group."""

    proc, child_pid, _marker = _spawn_tree_with_setsid_child(tmp_path)
    try:
        assert os.getpgid(child_pid) != os.getpgid(proc.pid)
        runner._kill_proc_quietly(proc)
        assert _wait_until_stopped(
            child_pid
        ), f"setsid child {child_pid} survived runner cancellation"
    finally:
        for pid in (proc.pid, child_pid):
            if _process_is_running(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_tracker_cleans_setsid_child_after_normal_parent_exit(
    tmp_path: Path,
) -> None:
    """Normal completion also removes daemonized generated-script work."""

    proc, child_pid, _marker = _spawn_tree_with_setsid_child(tmp_path)
    try:
        # Let the ancestry monitor observe the child, then model a successful
        # interpreter exit while its detached child remains alive.
        time.sleep(0.15)
        os.kill(proc.pid, signal.SIGTERM)
        proc.wait(timeout=5)
        assert _process_is_running(child_pid)

        assert process_tree.terminate_tracked_process_tree(proc)
        assert _wait_until_stopped(
            child_pid
        ), f"setsid child {child_pid} survived normal-completion cleanup"
    finally:
        if _process_is_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_tracker_refuses_to_signal_reused_pid_identity(monkeypatch) -> None:
    """A retained PID with a different birth marker is never killed."""

    original = process_tree.ProcessIdentity(700001, 1, "birth-a")
    reused = process_tree.ProcessIdentity(700001, 1, "birth-b")
    tracker = process_tree.PosixDescendantTracker(original)
    tracker._stop.set()
    tracker.start()

    monkeypatch.setattr(process_tree, "process_snapshot", lambda: {reused.pid: reused})
    killed: list[int] = []
    monkeypatch.setattr(process_tree.os, "kill", lambda pid, _sig: killed.append(pid))
    monkeypatch.setattr(
        process_tree.os,
        "killpg",
        lambda _pgid, _sig: pytest.fail("reused root must not trigger killpg"),
    )

    tracker.terminate()

    assert killed == []


def test_tracker_never_kills_hosts_own_process_group(monkeypatch) -> None:
    root = process_tree.ProcessIdentity(700005, 1, "birth-root")
    tracker = process_tree.PosixDescendantTracker(root)
    tracker._stop.set()
    tracker.start()
    monkeypatch.setattr(process_tree, "process_snapshot", lambda: {root.pid: root})
    monkeypatch.setattr(process_tree.os, "getpgid", lambda _pid: 12345)
    monkeypatch.setattr(process_tree.os, "getpgrp", lambda: 12345)
    monkeypatch.setattr(
        process_tree.os,
        "killpg",
        lambda _pgid, _sig: pytest.fail("must not kill Sift's process group"),
    )
    killed: list[int] = []
    monkeypatch.setattr(process_tree.os, "kill", lambda pid, _sig: killed.append(pid))

    tracker.terminate()

    assert killed == [root.pid, root.pid, root.pid]


def test_live_accounting_fails_closed_when_root_cannot_be_verified(
    monkeypatch,
) -> None:
    root = process_tree.ProcessIdentity(700002, 1, "birth-root")
    tracker = process_tree.PosixDescendantTracker(root)
    monkeypatch.setattr(process_tree, "process_snapshot", lambda: {})

    with pytest.raises(process_tree.ProcessTreeSnapshotUnavailable):
        tracker.live_identities()


def test_unverified_initial_root_promotes_only_via_exact_marker(
    monkeypatch,
) -> None:
    synthetic = process_tree.ProcessIdentity(
        700003,
        -1,
        process_tree._UNVERIFIED_START,
    )
    actual = process_tree.ProcessIdentity(700003, 42, "real-birth")
    child = process_tree.ProcessIdentity(700004, 700003, "child-birth")
    tracker = process_tree.PosixDescendantTracker(
        synthetic,
        marker=("SIFT_PROCESS_TREE_MARKER", "owned"),
    )
    monkeypatch.setattr(
        process_tree,
        "process_snapshot",
        lambda: {700003: actual, 700004: child},
    )
    monkeypatch.setattr(process_tree, "_marker_pids", lambda *_args: {700003})

    live = tracker.live_identities()

    assert tracker.root == actual
    assert {identity.pid for identity in live} == {700003, 700004}


def test_cleanup_marker_recovers_runtime_loaded_reparented_escape(
    tmp_path: Path,
) -> None:
    """Cleanup metadata survives runtime token removal and reparenting."""
    proc, child_pid, marker = _spawn_tree_with_setsid_child(tmp_path)
    tracker = getattr(proc, process_tree._TRACKER_ATTRIBUTE)
    try:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)
        # Deliberately discard historical ancestry: marker discovery alone
        # must rediscover the escaped/reparented process.
        with tracker._known_lock:
            tracker._known = {tracker.root.pid: tracker.root}
        tracker.terminate()
        assert _wait_until_stopped(child_pid)
    finally:
        if _process_is_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        assert marker[0] and marker[1]


def test_marker_cleanup_survives_root_exit_before_tracker_attach(
    tmp_path: Path,
) -> None:
    """A very short daemonizing script cannot outrun initial tracking."""

    pid_path = tmp_path / "fast-escaped.pid"
    marker = (executor._PROCESS_TREE_MARKER_ENV_VAR, uuid.uuid4().hex)
    child_code = (
        "import os,time\n"
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import subprocess,sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True)\n"
    )
    env = dict(os.environ)
    env[marker[0]] = marker[1]
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        env=env,
        start_new_session=True,
    )
    proc.wait(timeout=5)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not pid_path.exists():
        time.sleep(0.05)
    assert pid_path.exists()
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    try:
        tracker = process_tree.attach_posix_descendant_tracker(proc, marker=marker)
        assert tracker is not None
        assert process_tree.terminate_tracked_process_tree(proc)
        assert _wait_until_stopped(child_pid)
    finally:
        if _process_is_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_normal_root_exit_closes_pipes_held_by_setsid_child(
    tmp_path: Path,
) -> None:
    """A background child cannot turn clean completion into a timeout."""

    pid_path = tmp_path / "pipe-holder.pid"
    marker = (executor._PROCESS_TREE_MARKER_ENV_VAR, uuid.uuid4().hex)
    child_code = (
        "import os,time\n"
        f"open({str(pid_path)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import subprocess,sys\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True)\n"
    )
    env = dict(os.environ)
    env[marker[0]] = marker[1]
    proc = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    tracker = process_tree.attach_posix_descendant_tracker(proc, marker=marker)
    assert tracker is not None
    started = time.monotonic()
    child_pid: int | None = None
    try:
        executor._communicate_with_memory_guard(
            proc,
            timeout_seconds=5,
            memory_limit_bytes=0,
            process_limit=0,
            cpu_limit_seconds=0,
        )
        assert time.monotonic() - started < 3
        if pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
            assert _wait_until_stopped(child_pid)
    finally:
        process_tree.terminate_tracked_process_tree(proc)
        if child_pid is not None and _process_is_running(child_pid):
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=5)


@pytest.mark.parametrize("mode", ["normal", "timeout", "error"])
def test_executor_invokes_tracked_cleanup_on_every_completion_path(
    monkeypatch,
    tmp_path: Path,
    mode: str,
) -> None:
    """Clean exit, timeout, and unexpected communicate errors all clean."""

    class _Boom(OSError):
        pass

    class _FakeProc:
        pid = 2_000_000_001
        args = ["fake-interpreter"]
        stdout = None
        stderr = None
        returncode = 0

        def __init__(self) -> None:
            self.communications = 0

        def communicate(self, timeout=None):
            self.communications += 1
            if mode == "timeout" and self.communications == 1:
                raise subprocess.TimeoutExpired(self.args, timeout)
            if mode == "error":
                raise _Boom("pipe failed")
            return "", ""

        def poll(self):
            return self.returncode

        def kill(self) -> None:
            self.returncode = -signal.SIGKILL

    proc = _FakeProc()
    attached: list[object] = []
    cleaned: list[object] = []
    monkeypatch.setattr(executor.sys, "platform", "darwin")
    monkeypatch.setattr(
        env_detect,
        "sandbox_baseline_result",
        lambda: (True, ""),
    )
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *a, **kw: proc)
    monkeypatch.setattr(
        executor,
        "attach_posix_descendant_tracker",
        lambda p, marker=None: attached.append(p),
    )
    monkeypatch.setattr(
        executor,
        "terminate_tracked_process_tree",
        lambda p: cleaned.append(p) or True,
    )
    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )

    if mode == "error":
        with pytest.raises(_Boom):
            executor.run_script("R", "cat('ok')", tmp_path, env=fake_env)
    else:
        result = executor.run_script(
            "R",
            "cat('ok')",
            tmp_path,
            env=fake_env,
            timeout_seconds=1,
        )
        if mode == "timeout":
            assert "timed out" in (result.error or "")

    assert attached == [proc]
    assert cleaned == [proc]
