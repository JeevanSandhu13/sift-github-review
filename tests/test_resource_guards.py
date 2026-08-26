"""Memory / size guards: the machine must survive a bad dataset or script.

Two independent protections, both added in Sift:

1. ``schema.load_data`` refuses a full in-memory load above a size
   ceiling. Uploads are capped at 1 GB and pandas' in-memory form runs
   several times the on-disk size, so an unguarded load can request
   ~10 GB on a laptop.
2. The executor applies an RLIMIT_AS ceiling to every script.
   ``sandbox-exec`` bounds what a script can reach, not what it can
   allocate; without the rlimit a runaway script OOM-kills the app.

Both are fail-safe: an unparseable override falls back to the default
rather than disabling the guard.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sift import process_tree, schema
from sift.executor import (
    MAX_CAPTURED_STREAM_BYTES,
    _CpuLimitExceeded,
    _communicate_with_memory_guard,
    _disk_reserve_preflight_error,
    _DiskReserveExceeded,
    _memory_limit_preexec,
    _MemoryLimitExceeded,
    _merge_bounded_capture,
    _ProcessLimitExceeded,
    _ResourceMonitorUnavailable,
    _process_tree_process_count,
    _resource_limited_argv,
    _rlimit_preexec,
    resource_limits_preexec,
    script_cpu_limit_seconds,
    script_file_size_limit_bytes,
    script_memory_limit_bytes,
    script_min_free_disk_bytes,
    script_process_limit,
)
from sift.schema import DatasetTooLargeError, SchemaExtractError

try:
    import resource as _resource
except ImportError:  # Windows
    _resource = None


def _has_rlimit(name: str) -> bool:
    return _resource is not None and hasattr(_resource, name)


_HAS_BASH = any(Path(path).is_file() for path in ("/bin/bash", "/usr/bin/bash"))


def _terminate_test_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    getpgid = getattr(os, "getpgid", None)
    killpg = getattr(os, "killpg", None)
    if callable(getpgid) and callable(killpg):
        import signal

        try:
            killpg(getpgid(proc.pid), signal.SIGKILL)
            return
        except (OSError, ProcessLookupError):
            pass
    proc.kill()


# --------------------------------------------------------------------
# Dataset full-load ceiling
# --------------------------------------------------------------------

def test_oversized_dataset_refused_with_actionable_message(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(1024))  # 1 KB ceiling
    big = tmp_path / "huge.csv"
    big.write_text("a,b\n" + "1,2\n" * 2000)
    with pytest.raises(DatasetTooLargeError) as excinfo:
        schema.load_data(big)
    msg = str(excinfo.value)
    # Must name the file, both sizes, and the way forward.
    assert "huge.csv" in msg
    assert "script" in msg.lower()
    assert "SIFT_MAX_LOAD_BYTES" in msg


def test_under_ceiling_still_loads(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(10 * 1024 * 1024))
    small = tmp_path / "small.csv"
    small.write_text("a,b\n1,2\n3,4\n")
    df = schema.load_data(small)
    assert len(df) == 2


def test_too_large_is_catchable_as_schema_error() -> None:
    """Existing ``except SchemaExtractError`` sites must keep working."""
    assert issubclass(DatasetTooLargeError, SchemaExtractError)


@pytest.mark.parametrize("bad", ["", "bogus", "-1", "0"])
def test_invalid_ceiling_falls_back_to_default_not_off(
        bad, monkeypatch) -> None:
    """A typo must never silently disable the memory guard."""
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", bad)
    assert schema.full_load_max_bytes() == schema._DEFAULT_FULL_LOAD_MAX_BYTES


def test_ceiling_is_raisable(monkeypatch) -> None:
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(4 * 1024 ** 3))
    assert schema.full_load_max_bytes() == 4 * 1024 ** 3


def test_unstattable_path_does_not_raise_from_the_guard(tmp_path) -> None:
    """A missing file must produce the normal load error, not a
    confusing 'too large' error from the guard."""
    with pytest.raises(Exception) as excinfo:
        schema.load_data(tmp_path / "does_not_exist.csv")
    assert not isinstance(excinfo.value, DatasetTooLargeError)


# --------------------------------------------------------------------
# Script address-space ceiling
# --------------------------------------------------------------------

def test_memory_limit_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", raising=False)
    assert script_memory_limit_bytes() == 8 * 1024 ** 3
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "junk")
    assert script_memory_limit_bytes() == 8 * 1024 ** 3   # fail-safe
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "-9")
    assert script_memory_limit_bytes() == 8 * 1024 ** 3   # fail-safe
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    assert script_memory_limit_bytes() == 0               # explicit off
    assert _memory_limit_preexec() is None


@pytest.mark.skipif(
    sys.platform == "darwin"
    or not _has_rlimit("RLIMIT_AS"),
    reason="RLIMIT_AS is unavailable or non-binding on macOS",
)
def test_limit_actually_binds_a_child_process(monkeypatch) -> None:
    """The load-bearing assertion: an over-limit allocation fails as a
    catchable MemoryError inside the script, rather than the OS
    OOM-killing the whole app."""
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", str(256 * 1024 * 1024))
    code = "a = bytearray(1024*1024*1024); print('ALLOCATED')"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=_memory_limit_preexec(),
    )
    assert proc.returncode != 0
    assert "ALLOCATED" not in proc.stdout
    assert "MemoryError" in proc.stderr

    # Control: with the guard off the same allocation succeeds, proving
    # the failure above came from our limit and not the environment.
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    ok = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=_memory_limit_preexec(),
    )
    assert ok.returncode == 0 and "ALLOCATED" in ok.stdout


# --------------------------------------------------------------------
# CPU time, process count, single-file size ceilings
# --------------------------------------------------------------------
#
# Same empirical-proof standard as the memory ceiling above: each
# test actually spawns a child that would otherwise misbehave and
# confirms the guard is what stops it, with an "off" control run
# proving the failure isn't an artifact of the test environment.

def test_cpu_limit_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("SIFT_SCRIPT_MAX_CPU_SECONDS", raising=False)
    assert script_cpu_limit_seconds() == 600
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "junk")
    assert script_cpu_limit_seconds() == 600  # fail-safe
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "-5")
    assert script_cpu_limit_seconds() == 600  # fail-safe
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "0")
    assert script_cpu_limit_seconds() == 0    # explicit off


def test_process_limit_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("SIFT_SCRIPT_MAX_PROCESSES", raising=False)
    assert script_process_limit() == 64
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "junk")
    assert script_process_limit() == 64
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "0")
    assert script_process_limit() == 0


def test_file_size_limit_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", raising=False)
    assert script_file_size_limit_bytes() == 2 * 1024 ** 3
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "bogus")
    assert script_file_size_limit_bytes() == 2 * 1024 ** 3
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "0")
    assert script_file_size_limit_bytes() == 0


def test_disk_reserve_defaults_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", raising=False)
    assert script_min_free_disk_bytes() == 512 * 1024 ** 2
    monkeypatch.setenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", "bogus")
    assert script_min_free_disk_bytes() == 512 * 1024 ** 2
    monkeypatch.setenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", "-5")
    assert script_min_free_disk_bytes() == 512 * 1024 ** 2
    monkeypatch.setenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", "0")
    assert script_min_free_disk_bytes() == 0


def test_disk_reserve_preflight_uses_one_metadata_query(
    monkeypatch, tmp_path,
) -> None:
    calls: list[Path] = []

    class _Usage:
        free = 99

    def _disk_usage(path):
        calls.append(Path(path))
        return _Usage()

    monkeypatch.setattr("sift.executor.shutil.disk_usage", _disk_usage)
    error = _disk_reserve_preflight_error(tmp_path, 100)
    assert calls == [tmp_path]
    assert error is not None
    assert "99 free bytes" in error
    assert "SIFT_SCRIPT_MIN_FREE_DISK_BYTES" in error


def test_disk_reserve_preflight_fails_closed_when_query_is_unavailable(
    monkeypatch, tmp_path,
) -> None:
    def _unavailable(_path):
        raise OSError("unmounted")

    monkeypatch.setattr("sift.executor.shutil.disk_usage", _unavailable)
    error = _disk_reserve_preflight_error(tmp_path, 100)
    assert error is not None
    assert "could not inspect free disk space" in error
    assert _disk_reserve_preflight_error(tmp_path, 0) is None


@pytest.mark.skipif(
    not _has_rlimit("RLIMIT_CPU"),
    reason="platform has no RLIMIT_CPU",
)
def test_cpu_limit_actually_kills_a_busy_loop(monkeypatch) -> None:
    """A CPU-bound infinite loop must be terminated once it exceeds
    the configured CPU-time ceiling - proves RLIMIT_CPU is wired,
    not just parsed."""
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "1")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "0")
    code = "i = 0\nwhile True:\n    i += 1\n"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=resource_limits_preexec(), timeout=15,
    )
    # SIGXCPU - Python installs no handler for it, so the default
    # disposition terminates the process either way once the soft
    # ceiling is crossed. Assert it did NOT run to completion.
    assert proc.returncode != 0


@pytest.mark.skipif(
    not _has_rlimit("RLIMIT_NPROC"),
    reason="platform has no RLIMIT_NPROC",
)
def test_process_limit_blocks_a_fork_bomb(monkeypatch) -> None:
    """A script that tries to fork far past the configured process
    ceiling must have those forks fail (OSError/BlockingIOError from
    os.fork()) rather than succeed and take the machine down. Bounded
    attempt count so a failure of 'the guard didn't work' can't
    itself fork-bomb the test runner."""
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "5")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "0")
    code = (
        "import os, sys, time\n"
        "created = 0\n"
        "try:\n"
        "    for _ in range(2000):\n"
        "        pid = os.fork()\n"
        "        if pid == 0:\n"
        "            time.sleep(2)\n"
        "            os._exit(0)\n"
        "        created += 1\n"
        "except (OSError, BlockingIOError):\n"
        "    pass\n"
        "print('CREATED', created)\n"
        "sys.exit(0)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=resource_limits_preexec(), timeout=15,
    )
    assert "CREATED" in proc.stdout
    created = int(proc.stdout.strip().split()[-1])
    # Nowhere near the 2000 attempted - the rlimit must have cut this
    # off early. Generous ceiling (not exactly 5) because the parent
    # process's own thread/process count shares the same UID quota.
    assert created < 100, (
        f"created {created} processes - RLIMIT_NPROC does not appear "
        f"to be constraining forks"
    )


@pytest.mark.skipif(
    not _has_rlimit("RLIMIT_FSIZE"),
    reason="platform has no RLIMIT_FSIZE",
)
def test_file_size_limit_stops_a_runaway_write(monkeypatch, tmp_path) -> None:
    """A script writing well past the configured single-file ceiling
    must be killed with SIGXFSZ / get a write failure, not be allowed
    to fill the disk."""
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", str(1024 * 1024))  # 1 MB
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "0")
    target = tmp_path / "runaway.bin"
    code = (
        f"path = {str(target)!r}\n"
        "chunk = b'x' * (1024 * 1024)\n"
        "with open(path, 'wb') as f:\n"
        "    for _ in range(200):\n"
        "        f.write(chunk)\n"
        "        f.flush()\n"
        "print('WROTE_ALL')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=resource_limits_preexec(), timeout=15,
    )
    assert "WROTE_ALL" not in proc.stdout
    assert target.exists()
    assert target.stat().st_size < 50 * 1024 * 1024


def test_combined_preexec_is_none_when_everything_disabled(monkeypatch) -> None:
    """All four limits off must yield None - Popen treats that as
    'no child hook', correct for a researcher who explicitly disabled
    every guard (e.g. known-heavy hardware)."""
    for var in (
        "SIFT_SCRIPT_MAX_MEMORY_BYTES", "SIFT_SCRIPT_MAX_CPU_SECONDS",
        "SIFT_SCRIPT_MAX_PROCESSES", "SIFT_SCRIPT_MAX_FILE_SIZE_BYTES",
    ):
        monkeypatch.setenv(var, "0")
    assert resource_limits_preexec() is None


@pytest.mark.skipif(not _HAS_BASH, reason="Bash launcher is unavailable")
def test_resource_launcher_keeps_untrusted_command_in_argv(monkeypatch) -> None:
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    original = ["/usr/bin/python3", "script with spaces.py", "$(touch nope)"]
    wrapped = _resource_limited_argv(original, "darwin")
    assert wrapped[-len(original):] == original
    assert "$(touch nope)" not in wrapped[2]
    assert 'exec "$@"' in wrapped[2]
    assert "sift_soft_limit -t" in wrapped[2]
    assert "ulimit -u" not in wrapped[2]


@pytest.mark.skipif(not _HAS_BASH, reason="Bash launcher is unavailable")
def test_linux_resource_launcher_avoids_per_uid_process_guard(monkeypatch) -> None:
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "17")
    wrapped = _resource_limited_argv(["/usr/bin/python3"], "linux")
    assert "ulimit -u" not in wrapped[2]


@pytest.mark.skipif(not _HAS_BASH, reason="Bash launcher is unavailable")
def test_file_size_launcher_uses_bash_kib_units(monkeypatch) -> None:
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", str(1024 * 1024))
    wrapped = _resource_limited_argv(["/usr/bin/python3"], "darwin")
    assert "sift_soft_limit -f 1024" in wrapped[2]


def test_resource_launcher_fails_closed_when_bash_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "60")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "1024")
    monkeypatch.setattr(Path, "is_file", lambda _path: False)

    with pytest.raises(RuntimeError, match="Bash is unavailable"):
        _resource_limited_argv(["python", "analysis.py"], "linux")


def test_resource_launcher_needs_no_shell_when_all_posix_rlimits_are_off(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "0")
    monkeypatch.setattr(Path, "is_file", lambda _path: False)
    command = ["python", "analysis.py"]

    assert _resource_limited_argv(command, "linux") == command


@pytest.mark.skipif(
    not _HAS_BASH or not _has_rlimit("RLIMIT_CPU"),
    reason="Bash CPU limits are unavailable",
)
def test_resource_launcher_retains_tighter_inherited_hard_limit(monkeypatch) -> None:
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "600")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "0")
    wrapped = _resource_limited_argv(
        [sys.executable, "-c", "print('EXECUTED')"],
        "darwin",
    )
    completed = subprocess.run(
        [
            "/bin/bash", "-c", 'ulimit -t 1; exec "$@"',
            "outer-limit", *wrapped,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "EXECUTED"


def test_parent_memory_guard_stops_large_allocation() -> None:
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import time; x=bytearray(256*1024*1024); time.sleep(5)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    tracker = process_tree.attach_posix_descendant_tracker(proc)
    if tracker is None:
        proc.kill()
        proc.wait()
        pytest.skip("process birth identity is unavailable")
    try:
        try:
            with pytest.raises(_MemoryLimitExceeded):
                _communicate_with_memory_guard(
                    proc, timeout_seconds=5,
                    memory_limit_bytes=96 * 1024 * 1024,
                )
        except _ResourceMonitorUnavailable:
            pytest.skip("outer test sandbox blocks process-tree memory discovery")
    finally:
        process_tree.terminate_tracked_process_tree(proc)
        proc.communicate()


def test_parent_process_guard_counts_only_the_script_tree() -> None:
    code = (
        "import subprocess, sys, time\n"
        "children = [subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(10)']) for _ in range(8)]\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    tracker = process_tree.attach_posix_descendant_tracker(proc)
    if tracker is None:
        proc.kill()
        proc.wait()
        pytest.skip("process birth identity is unavailable")
    try:
        time.sleep(0.2)
        identities = process_tree.tracked_process_identities(proc)
        observed = _process_tree_process_count(proc.pid, identities)
        if observed is None or observed <= 1:
            pytest.skip("outer test sandbox blocks descendant-process discovery")
        with pytest.raises(_ProcessLimitExceeded) as excinfo:
            _communicate_with_memory_guard(
                proc, timeout_seconds=5, memory_limit_bytes=0,
                process_limit=3,
            )
        assert excinfo.value.limit_processes == 3
        assert excinfo.value.observed_processes > 3
    finally:
        process_tree.terminate_tracked_process_tree(proc)
        proc.communicate()


def test_parent_cpu_guard_aggregates_forked_workers() -> None:
    """Workers cannot each spend the full per-process RLIMIT_CPU budget."""

    worker = "while True: pass"
    code = (
        "import subprocess, sys, time\n"
        f"children = [subprocess.Popen([sys.executable, '-c', {worker!r}]) "
        "for _ in range(3)]\n"
        "time.sleep(20)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    tracker = process_tree.attach_posix_descendant_tracker(proc)
    if tracker is None:
        proc.kill()
        proc.wait()
        pytest.skip("process birth identity is unavailable")
    try:
        try:
            with pytest.raises(_CpuLimitExceeded) as excinfo:
                _communicate_with_memory_guard(
                    proc,
                    timeout_seconds=8,
                    memory_limit_bytes=0,
                    process_limit=0,
                    cpu_limit_seconds=0.05,
                )
        except _ResourceMonitorUnavailable:
            pytest.skip("outer test sandbox blocks process-tree CPU discovery")
        assert excinfo.value.limit_seconds == 0.05
        assert excinfo.value.observed_seconds > 0.05
    finally:
        process_tree.terminate_tracked_process_tree(proc)
        proc.communicate()


def test_parent_cpu_guard_counts_reaped_short_workers() -> None:
    """Serial workers cannot disappear from accounting between samples."""
    code = (
        "import subprocess, sys, time\n"
        "for _ in range(5):\n"
        "    subprocess.run([sys.executable, '-c', "
        "'for i in range(5000000): pass'], check=True)\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    tracker = process_tree.attach_posix_descendant_tracker(proc)
    if tracker is None:
        proc.kill()
        proc.wait()
        pytest.skip("process birth identity is unavailable")
    try:
        try:
            with pytest.raises(_CpuLimitExceeded):
                _communicate_with_memory_guard(
                    proc,
                    timeout_seconds=5,
                    memory_limit_bytes=0,
                    process_limit=0,
                    cpu_limit_seconds=0.05,
                )
        except _ResourceMonitorUnavailable:
            pytest.skip("outer test sandbox blocks process-tree CPU discovery")
    finally:
        process_tree.terminate_tracked_process_tree(proc)
        proc.communicate()


def test_enabled_process_guard_fails_closed_when_monitor_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sift.executor._process_tree_process_count", lambda *_args: None,
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        with pytest.raises(_ResourceMonitorUnavailable):
            _communicate_with_memory_guard(
                proc, timeout_seconds=2, memory_limit_bytes=0,
                process_limit=3,
            )
    finally:
        _terminate_test_tree(proc)
        proc.communicate()


def test_enabled_guard_fails_closed_when_identity_snapshot_is_unavailable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sift.executor.tracked_process_identities",
        lambda _proc: (_ for _ in ()).throw(
            process_tree.ProcessTreeSnapshotUnavailable("unavailable")
        ),
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(2)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    # Make the function take the tracker-owned branch without creating a
    # monitor thread; the monkeypatch above models its failed live snapshot.
    setattr(proc, process_tree._TRACKER_ATTRIBUTE, object())
    try:
        with pytest.raises(_ResourceMonitorUnavailable) as excinfo:
            _communicate_with_memory_guard(
                proc, timeout_seconds=2, memory_limit_bytes=1,
            )
        assert excinfo.value.resource_name == "process-identity"
    finally:
        _terminate_test_tree(proc)
        proc.communicate()


def test_identity_snapshot_race_accepts_a_root_that_exits_during_bounded_retry(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sift.executor.tracked_process_identities",
        lambda _proc: (_ for _ in ()).throw(
            process_tree.ProcessTreeSnapshotUnavailable("already reaped from ps")
        ),
    )

    class ExitingProcess:
        args = ["synthetic-short-worker"]
        pid = 424242
        returncode: int | None = None

        def __init__(self) -> None:
            stdout_read, stdout_write = os.pipe()
            stderr_read, stderr_write = os.pipe()
            os.close(stdout_write)
            os.close(stderr_write)
            self.stdout = os.fdopen(stdout_read, "rb")
            self.stderr = os.fdopen(stderr_read, "rb")
            self.polls = 0

        def poll(self) -> int | None:
            self.polls += 1
            if self.polls == 1:
                return None
            self.returncode = 0
            return 0

    proc = ExitingProcess()
    # Make the function take the tracker-owned branch. The first poll reports
    # live, the identity snapshot observes the clean-exit race, and the first
    # bounded retry sees the exit status. This must be normal completion, not
    # a false monitor-unavailable failure.
    setattr(proc, process_tree._TRACKER_ATTRIBUTE, object())
    try:
        stdout, stderr = _communicate_with_memory_guard(
            proc, timeout_seconds=2, memory_limit_bytes=1,
        )
    finally:
        proc.stdout.close()
        proc.stderr.close()
    assert stdout == "" and stderr == ""
    assert proc.returncode == 0
    assert proc.polls == 2


def test_parent_capture_bounds_untrusted_stdout_without_deadlock() -> None:
    requested = MAX_CAPTURED_STREAM_BYTES + 2 * 1024 * 1024
    proc = subprocess.Popen(
        [
            sys.executable, "-c",
            f"import sys; sys.stdout.write('x' * {requested})",
        ],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = _communicate_with_memory_guard(
        proc, timeout_seconds=10, memory_limit_bytes=0, process_limit=0,
    )
    assert proc.returncode == 0
    assert stderr == ""
    assert len(stdout.encode("utf-8")) < requested
    assert stdout.startswith("x" * 100)
    assert "SIFT OUTPUT TRUNCATED" in stdout
    assert proc.stdout is not None and proc.stdout.closed
    assert proc.stderr is not None and proc.stderr.closed


def test_parent_disk_guard_stops_many_small_file_exhaustion_shape(
    monkeypatch, tmp_path,
) -> None:
    """Free-space depletion is bounded without walking individual files."""
    calls: list[Path] = []

    def _below_reserve(path):
        calls.append(Path(path))
        return 99

    monkeypatch.setattr("sift.executor._free_disk_bytes", _below_reserve)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with pytest.raises(_DiskReserveExceeded) as excinfo:
            _communicate_with_memory_guard(
                proc,
                timeout_seconds=5,
                memory_limit_bytes=0,
                disk_directory=tmp_path,
                disk_reserve_bytes=100,
            )
        assert excinfo.value.observed_free_bytes == 99
        assert excinfo.value.reserve_bytes == 100
        assert calls == [tmp_path]
    finally:
        _terminate_test_tree(proc)
        proc.communicate()


def test_parent_disk_guard_fails_closed_when_monitor_is_unavailable(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.setattr("sift.executor._free_disk_bytes", lambda _path: None)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with pytest.raises(_ResourceMonitorUnavailable) as excinfo:
            _communicate_with_memory_guard(
                proc,
                timeout_seconds=5,
                memory_limit_bytes=0,
                disk_directory=tmp_path,
                disk_reserve_bytes=100,
            )
        assert excinfo.value.resource_name == "free-disk"
    finally:
        _terminate_test_tree(proc)
        proc.communicate()


def test_parent_disk_guard_samples_a_fast_process_at_exit(
    monkeypatch, tmp_path,
) -> None:
    """A writer that exits inside one monitor interval cannot evade the guard."""
    monkeypatch.setattr("sift.executor._free_disk_bytes", lambda _path: 1)
    proc = subprocess.Popen(
        [sys.executable, "-c", "print('done')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with pytest.raises(_DiskReserveExceeded):
            _communicate_with_memory_guard(
                proc,
                timeout_seconds=5,
                memory_limit_bytes=0,
                disk_directory=tmp_path,
                disk_reserve_bytes=100,
            )
    finally:
        _terminate_test_tree(proc)
        proc.communicate()


def test_timeout_exception_retains_bounded_early_output() -> None:
    distinctive = b"EARLY-DIAGNOSTIC\n"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,time; "
                f"os.write(1, {distinctive!r} + "
                f"b'x' * {MAX_CAPTURED_STREAM_BYTES + 1024}); "
                "time.sleep(5)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _communicate_with_memory_guard(
                proc,
                timeout_seconds=0.25,
                memory_limit_bytes=0,
            )
        prefix = excinfo.value._sift_captured_stdout
        assert prefix.startswith(distinctive)
        # A short deadline need not let a slower Windows/virtualized pipe
        # deliver the entire producer write before timeout.  Whatever arrived
        # must be retained and must never exceed the parent capture bound.
        assert len(distinctive) <= len(prefix) <= MAX_CAPTURED_STREAM_BYTES
        assert excinfo.value._sift_stdout_truncated is (
            len(prefix) == MAX_CAPTURED_STREAM_BYTES
        )
    finally:
        _terminate_test_tree(proc)
        proc.communicate()


def test_timeout_capture_merge_keeps_early_output_and_overall_data_cap() -> None:
    distinctive = b"EARLY-DIAGNOSTIC\n"
    merged = _merge_bounded_capture(
        distinctive,
        False,
        b"x" * (MAX_CAPTURED_STREAM_BYTES + 1024),
    )
    marker = (
        f"\n[SIFT OUTPUT TRUNCATED AT {MAX_CAPTURED_STREAM_BYTES} BYTES]\n"
    )
    assert merged.startswith(distinctive.decode())
    assert merged.endswith(marker)
    assert len(merged.removesuffix(marker).encode("utf-8")) == (
        MAX_CAPTURED_STREAM_BYTES
    )


def test_run_script_timeout_preserves_guard_prefix_and_cap(
    monkeypatch, tmp_path,
) -> None:
    """The outer timeout recovery must merge, not replace, drained output."""
    from sift import env_detect, executor

    distinctive = b"EARLY-DIAGNOSTIC\n"

    class _FakeProc:
        pid = 2_000_000_000
        returncode = None
        args = ["fake"]

        def communicate(self, timeout=None):
            return "x" * (MAX_CAPTURED_STREAM_BYTES + 1024), "late stderr"

        def kill(self):
            self.returncode = -9

    def _stop_with_prefix(_proc, **_kwargs):
        exc = subprocess.TimeoutExpired(["fake"], 1)
        exc._sift_captured_stdout = distinctive
        exc._sift_stdout_truncated = False
        exc._sift_captured_stderr = b"early stderr\n"
        exc._sift_stderr_truncated = False
        raise exc

    monkeypatch.setenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", "0")
    # This test isolates timeout-output merging; native resource-limit
    # launchers and their platform prerequisites are covered separately.
    monkeypatch.setenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_CPU_SECONDS", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_PROCESSES", "0")
    monkeypatch.setenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", "0")
    # Establish the real Windows private-state ACL before replacing the
    # simulated platform and process constructor.
    from sift.config import ensure_private_sift_dir
    ensure_private_sift_dir(tmp_path)
    monkeypatch.setattr(executor.sys, "platform", "darwin")
    monkeypatch.setattr(
        env_detect, "sandbox_baseline_result", lambda: (True, ""),
    )
    monkeypatch.setattr(executor.subprocess, "Popen", lambda *_a, **_k: _FakeProc())
    monkeypatch.setattr(executor, "attach_posix_descendant_tracker", lambda *_a, **_k: None)
    monkeypatch.setattr(executor, "terminate_tracked_process_tree", lambda *_a, **_k: False)
    monkeypatch.setattr(executor.os, "getpgid", lambda _pid: 123, raising=False)
    monkeypatch.setattr(executor.os, "killpg", lambda *_a: None, raising=False)
    import signal
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(executor, "_communicate_with_memory_guard", _stop_with_prefix)

    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    result = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    marker = (
        f"\n[SIFT OUTPUT TRUNCATED AT {MAX_CAPTURED_STREAM_BYTES} BYTES]\n"
    )
    assert result.ok is False
    assert "timed out" in (result.error or "")
    assert result.raw_stdout.startswith(distinctive.decode())
    assert result.raw_stdout.endswith(marker)
    assert len(result.raw_stdout.removesuffix(marker).encode("utf-8")) == (
        MAX_CAPTURED_STREAM_BYTES
    )
    assert result.raw_stderr == "early stderr\nlate stderr"


def test_run_script_refuses_low_disk_before_staging(
    monkeypatch, tmp_path,
) -> None:
    from sift import env_detect, executor

    monkeypatch.setattr(executor.sys, "platform", "darwin")
    monkeypatch.setattr(executor, "_free_disk_bytes", lambda _path: 99)
    monkeypatch.setenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", "100")
    monkeypatch.setattr(
        executor,
        "_stage_runtime",
        lambda *_a, **_k: pytest.fail("low-disk preflight must precede staging"),
    )
    monkeypatch.setattr(
        executor,
        "_make_run_dir",
        lambda *_a, **_k: pytest.fail("low-disk preflight must precede mkdir"),
    )
    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    result = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert result.ok is False
    assert result.script_path is None
    assert "99 free bytes" in (result.error or "")
    assert "SIFT_SCRIPT_MIN_FREE_DISK_BYTES" in (result.error or "")


def test_run_script_reports_run_directory_creation_race(
    monkeypatch, tmp_path,
) -> None:
    from sift import env_detect, executor

    monkeypatch.setenv("SIFT_SCRIPT_MIN_FREE_DISK_BYTES", "100")
    monkeypatch.setattr(executor, "_free_disk_bytes", lambda _path: 101)

    def _disk_filled_after_probe(_cwd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(executor, "_make_run_dir", _disk_filled_after_probe)
    fake_env = env_detect.Environment(
        r=env_detect.Tool(name="R", binary="/bin/true"),
        stata=None,
        python=None,
        sandbox_exec="/usr/bin/sandbox-exec",
    )
    result = executor.run_script("R", "cat('hi')", tmp_path, env=fake_env)
    assert result.ok is False
    assert result.script_path is None
    assert "insufficient free disk space" in (result.error or "")
    assert result.run_dir == tmp_path / executor.RUNS_SUBDIR


def test_chatty_output_does_not_drive_resource_snapshot_rate(monkeypatch) -> None:
    """Monitor cost is time-bounded, not proportional to bytes printed."""
    calls = 0

    def _count(_pid, _identities=None):
        nonlocal calls
        calls += 1
        return 1

    monkeypatch.setattr("sift.executor._process_tree_process_count", _count)
    requested = 4 * 1024 * 1024
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import os,time; "
                f"os.write(1, b'x' * {requested}); "
                "time.sleep(0.35)"
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    stdout, stderr = _communicate_with_memory_guard(
        proc,
        timeout_seconds=5,
        memory_limit_bytes=0,
        process_limit=2,
    )

    assert len(stdout) == requested
    assert stderr == ""
    assert 1 <= calls <= 10


@pytest.mark.skipif(
    os.name == "nt",
    reason="preexec_fn and POSIX rlimits are unavailable on Windows",
)
def test_combined_preexec_applies_every_enabled_limit(monkeypatch) -> None:
    """With every limit enabled, the combined preexec must actually
    set all four rlimits inside the child - not silently drop any of
    them when composing."""
    monkeypatch.delenv("SIFT_SCRIPT_MAX_MEMORY_BYTES", raising=False)
    monkeypatch.delenv("SIFT_SCRIPT_MAX_CPU_SECONDS", raising=False)
    monkeypatch.delenv("SIFT_SCRIPT_MAX_PROCESSES", raising=False)
    monkeypatch.delenv("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES", raising=False)
    code = (
        "import resource\n"
        "for name in ('RLIMIT_AS', 'RLIMIT_CPU', 'RLIMIT_NPROC', 'RLIMIT_FSIZE'):\n"
        "    const = getattr(resource, name, None)\n"
        "    if const is not None:\n"
        "        soft, hard = resource.getrlimit(const)\n"
        "        print(name, soft)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        preexec_fn=resource_limits_preexec(), timeout=10,
    )
    out = proc.stdout
    if sys.platform != "darwin":
        assert "RLIMIT_AS " + str(8 * 1024 ** 3) in out
    assert "RLIMIT_CPU " + str(600) in out
    assert "RLIMIT_NPROC " + str(64) in out
    assert "RLIMIT_FSIZE " + str(2 * 1024 ** 3) in out


def test_rlimit_preexec_returns_none_for_unknown_limit_name(monkeypatch) -> None:
    """A typo'd or platform-absent rlimit constant must degrade to
    'not applied', never raise."""
    assert _rlimit_preexec("RLIMIT_DOES_NOT_EXIST", 100) is None
    assert _rlimit_preexec("RLIMIT_CPU", 0) is None  # disabled
