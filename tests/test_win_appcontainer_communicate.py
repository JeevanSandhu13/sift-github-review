"""``AppContainerProcess.communicate()`` reader-thread reuse.

Scope note, stated up front the same way this module's own test file
does for the rest of the ctypes application layer: real
``kernel32.dll`` (``WaitForSingleObject``, ``GetExitCodeProcess``,
``ReadFile``) is entirely FAKED here via monkeypatched module
attributes — this is not, and cannot be, a proof that the real Win32
calls behave as this code assumes (that is only verifiable on real
Windows, per ``win_appcontainer``'s own module docstring). What IS
being verified, precisely: the PYTHON-LEVEL threading logic inside
``communicate()`` — specifically, that a second call to
``communicate()`` (the "kill, then drain again" pattern
``executor.run_script`` uses after a timeout, mirroring
``subprocess.run``'s own recovery shape) reuses the SAME reader
threads rather than starting a second pair against the same pipe
handles the first pair is still draining. That reuse-vs-duplicate
distinction is pure Python control flow, independent of whatever the
real OS calls underneath do, so faking them here doesn't manufacture
false confidence about Windows behavior — it isolates and proves the
one thing that actually changed.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as real_wintypes
import subprocess
from types import SimpleNamespace

import pytest

from sift import win_appcontainer as wa


class _FakeKernel32:
    """Stands in for ``ctypes.WinDLL("kernel32")``. Only implements
    the three entry points ``AppContainerProcess.communicate`` /
    ``_read_pipe_all`` actually call."""

    def __init__(
        self,
        wait_sequence: list[int],
        content_by_handle: dict[int, bytes],
    ) -> None:
        self._wait_sequence = list(wait_sequence)
        self._content_by_handle = {
            h: bytearray(c) for h, c in content_by_handle.items()
        }
        self.read_file_call_count: dict[int, int] = {}
        self.terminate_job_call_count = 0

    def WaitForSingleObject(self, handle, wait_ms):
        return self._wait_sequence.pop(0)

    def GetExitCodeProcess(self, handle, byref_exit_code) -> bool:
        ptr = ctypes.cast(byref_exit_code, ctypes.POINTER(real_wintypes.DWORD))
        ptr.contents.value = 0
        return True

    def TerminateJobObject(self, handle, exit_code) -> bool:
        self.terminate_job_call_count += 1
        return True

    def ReadFile(self, handle, buf, buf_len, byref_bytes_read, overlapped) -> bool:
        self.read_file_call_count[handle] = self.read_file_call_count.get(handle, 0) + 1
        remaining = self._content_by_handle.get(handle, bytearray())
        n = min(len(remaining), buf_len)
        chunk = bytes(remaining[:n])
        del remaining[:n]
        ctypes.memmove(buf, chunk, len(chunk))
        ptr = ctypes.cast(byref_bytes_read, ctypes.POINTER(real_wintypes.DWORD))
        ptr.contents.value = len(chunk)
        return True


@pytest.fixture
def fake_win32(monkeypatch: pytest.MonkeyPatch):
    """Patch the module globals ``communicate()``/``_read_pipe_all``
    resolve at call time. ``wintypes`` itself needs no faking — it's
    pure ctypes struct/type definitions and imports fine off Windows;
    only the DLL entry points (which would otherwise fail to even
    load via ``ctypes.WinDLL`` on a non-Windows host) are replaced.
    """
    monkeypatch.setattr(wa, "wintypes", real_wintypes, raising=False)
    monkeypatch.setattr(wa, "WAIT_TIMEOUT", 0x00000102, raising=False)
    monkeypatch.setattr(wa, "WAIT_OBJECT_0", 0, raising=False)
    monkeypatch.setattr(wa, "WAIT_FAILED", 0xFFFFFFFF, raising=False)
    monkeypatch.setattr(wa, "INFINITE", 0xFFFFFFFF, raising=False)

    def _install(wait_sequence, content_by_handle):
        fake = _FakeKernel32(wait_sequence, content_by_handle)
        monkeypatch.setattr(wa, "_kernel32", fake, raising=False)
        return fake

    return _install


def _make_process(
    stdout_handle: int,
    stderr_handle: int,
    file_size_monitor=None,
) -> wa.AppContainerProcess:
    proc_info = SimpleNamespace(dwProcessId=999, hProcess=1)
    return wa.AppContainerProcess(
        proc_info=proc_info,  # type: ignore[arg-type]
        job_handle=2,  # type: ignore[arg-type]
        stdout_read=stdout_handle,  # type: ignore[arg-type]
        stderr_read=stderr_handle,  # type: ignore[arg-type]
        cleanup=lambda: None,
        file_size_monitor=file_size_monitor,
    )


def test_reader_threads_started_exactly_once(fake_win32) -> None:
    """The core invariant: after two ``communicate()`` calls (one
    timing out, one succeeding — the exact shape
    ``executor.run_script``'s timeout-recovery path drives), the
    SAME thread objects handled both, not a fresh pair on the second
    call. Before the fix, ``_t_out``/``_t_err`` would be brand-new
    objects after the second call — this assertion is exactly what
    would have failed against the old implementation."""
    fake_win32(
        wait_sequence=[0x00000102, 0],  # 1st call: timeout: 2nd: exited
        content_by_handle={10: b"stdout content", 20: b"stderr content"},
    )
    proc = _make_process(stdout_handle=10, stderr_handle=20)

    with pytest.raises(subprocess.TimeoutExpired):
        proc.communicate(timeout=0.01)

    assert proc._reader_threads_started is True
    t_out_first, t_err_first = proc._t_out, proc._t_err
    assert t_out_first is not None and t_err_first is not None

    stdout, stderr = proc.communicate(timeout=2)

    assert proc._t_out is t_out_first, (
        "a second communicate() call started a NEW stdout reader "
        "thread instead of reusing the first — this is the race "
        "the fix closes: two threads would then call ReadFile on "
        "the same handle concurrently"
    )
    assert proc._t_err is t_err_first, "same as above, for the stderr reader thread"
    assert stdout == "stdout content"
    assert stderr == "stderr content"


def test_output_is_captured_correctly_across_a_timeout_retry(fake_win32) -> None:
    """End result correctness, not just thread identity: content
    written to the pipes must come back whole and uncorrupted after
    the timeout-then-retry sequence, exactly matching what a
    real R/Stata/Python script's stdout/stderr would contain."""
    long_stdout = (f"line {i:d}\n" for i in range(500))
    payload = "".join(long_stdout).encode()
    fake_win32(
        wait_sequence=[0x00000102, 0],
        content_by_handle={10: payload, 20: b""},
    )
    proc = _make_process(stdout_handle=10, stderr_handle=20)

    with pytest.raises(subprocess.TimeoutExpired):
        proc.communicate(timeout=0.01)

    stdout, stderr = proc.communicate(timeout=2)
    assert stdout == payload.decode()
    assert stderr == ""


def test_single_successful_call_needs_no_retry(fake_win32) -> None:
    """Negative control: the common case (process finishes within
    the timeout, communicate() called exactly once) must still work
    -- the reuse guard must not require a prior timeout to function."""
    fake_win32(
        wait_sequence=[0],  # exits immediately, no timeout
        content_by_handle={10: b"ok", 20: b""},
    )
    proc = _make_process(stdout_handle=10, stderr_handle=20)

    stdout, stderr = proc.communicate(timeout=5)
    assert stdout == "ok"
    assert stderr == ""
    assert proc.returncode == 0


def test_windows_pipe_capture_is_bounded_and_continues_draining(
    fake_win32, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wa, "_MAX_CAPTURED_STREAM_BYTES", 16)
    fake = fake_win32(
        wait_sequence=[0],
        content_by_handle={10: b"x" * 100, 20: b""},
    )
    proc = _make_process(stdout_handle=10, stderr_handle=20)
    stdout, stderr = proc.communicate(timeout=5)
    assert stdout.startswith("x" * 16)
    assert "SIFT OUTPUT TRUNCATED" in stdout
    assert stderr == ""
    # One content read plus EOF proves bytes beyond the retention cap were
    # still drained instead of filling the pipe and deadlocking the child.
    assert fake.read_file_call_count[10] == 2


def test_each_handle_is_only_ever_drained_by_one_threads_worth_of_reads(
    fake_win32,
) -> None:
    """Indirect confirmation of the race being closed: ReadFile calls
    against each handle must form ONE coherent drain-to-EOF sequence
    (final call returns 0 bytes) rather than two interleaved
    sequences racing to read the same handle. With the fix, exactly
    one thread ever touches each handle across both communicate()
    calls, so the call count is deterministic — bug-shaped code
    (two threads reading concurrently) would make this flaky or
    over-count depending on scheduling, not reliably reproduce this
    exact figure."""
    fake = fake_win32(
        wait_sequence=[0x00000102, 0],
        content_by_handle={10: b"x" * 100, 20: b"y" * 50},
    )
    proc = _make_process(stdout_handle=10, stderr_handle=20)
    with pytest.raises(subprocess.TimeoutExpired):
        proc.communicate(timeout=0.01)
    proc.communicate(timeout=2)

    # One content-bearing read + one EOF read per handle.
    assert fake.read_file_call_count[10] == 2
    assert fake.read_file_call_count[20] == 2


def test_file_size_violation_terminates_job_and_forces_failure(fake_win32) -> None:
    target = wa.Path("C:/workspace/runaway.bin")

    class _ViolatingMonitor:
        def check(self):
            return wa.FileSizeViolation(target, 65, 64)

    fake = fake_win32(
        # First timed slice lets the monitor run; the second wait confirms
        # asynchronous TerminateJobObject completion.
        wait_sequence=[0x00000102, 0],
        content_by_handle={10: b"partial output", 20: b""},
    )
    proc = _make_process(10, 20, file_size_monitor=_ViolatingMonitor())

    stdout, stderr = proc.communicate(timeout=5)

    assert stdout == "partial output"
    assert "single-file limit" in stderr
    assert "65 bytes" in stderr
    assert proc.returncode == 1
    assert proc.file_size_violation == wa.FileSizeViolation(target, 65, 64)
    # One termination for the violation and the existing descendant-cleanup
    # backstop after the direct process status is collected.
    assert fake.terminate_job_call_count == 2


def test_disk_reserve_violation_terminates_complete_job_and_forces_failure(
    fake_win32,
) -> None:
    target = wa.Path("C:/workspace")

    class _ViolatingMonitor:
        def check(self):
            return wa.DiskReserveViolation(target, 511, 512)

    fake = fake_win32(
        wait_sequence=[0x00000102, 0],
        content_by_handle={10: b"partial output", 20: b""},
    )
    proc = _make_process(10, 20, file_size_monitor=_ViolatingMonitor())

    stdout, stderr = proc.communicate(timeout=5)

    assert stdout == "partial output"
    assert "safety reserve" in stderr
    assert "511 free bytes" in stderr
    assert proc.returncode == 1
    assert proc.disk_reserve_violation == wa.DiskReserveViolation(target, 511, 512)
    assert fake.terminate_job_call_count == 2


def test_file_size_monitor_failure_terminates_job_fail_closed(fake_win32) -> None:
    class _BrokenMonitor:
        def check(self):
            raise PermissionError("cannot enumerate writable root")

    fake = fake_win32(
        wait_sequence=[0x00000102, 0],
        content_by_handle={10: b"", 20: b""},
    )
    proc = _make_process(10, 20, file_size_monitor=_BrokenMonitor())

    _stdout, stderr = proc.communicate(timeout=5)

    assert "failed closed" in stderr
    assert isinstance(proc.file_size_monitor_error, PermissionError)
    assert proc.returncode == 1
    assert fake.terminate_job_call_count == 2


def test_final_scan_runs_after_descendants_are_terminated(fake_win32) -> None:
    target = wa.Path("C:/workspace/late-worker-output.bin")

    class _LateViolationMonitor:
        def __init__(self) -> None:
            self.calls = 0

        def check(self):
            self.calls += 1
            if self.calls == 1:
                return None
            return wa.FileSizeViolation(target, 129, 128)

    monitor = _LateViolationMonitor()
    fake = fake_win32(
        wait_sequence=[0],
        content_by_handle={10: b"", 20: b""},
    )
    proc = _make_process(10, 20, file_size_monitor=monitor)

    _stdout, stderr = proc.communicate(timeout=5)

    assert monitor.calls == 2
    assert fake.terminate_job_call_count == 1
    assert "late-worker-output.bin" in stderr
    assert proc.returncode == 1
