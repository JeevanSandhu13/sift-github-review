"""Windows confinement backend using AppContainer and Job Objects.

Linux and macOS apply an external wrapper to a subprocess command. Windows
instead requires the caller to create a LowBox token, attach security
capabilities through ``STARTUPINFOEX``, grant per-run ACLs, and assign the new
process to a Job Object. This module keeps policy planning in platform-neutral
functions and isolates the Windows API calls behind fail-closed entry points.

The backend enforces four properties:

1. No networking capability SIDs are granted.
2. System and interpreter paths are read-only; the workspace and current run
   receive narrow write grants while private ``.sift`` state remains hidden.
3. AppContainer integrity boundaries and Job Object accounting isolate the
   process tree.
4. Job limits cap CPU, memory, and process count. A parent monitor enforces
   file-size and free-space reserves because Windows has no ``RLIMIT_FSIZE``
   equivalent.

ACLs and inheritance state are restored in ``AppContainerRun`` cleanup.
Before any researcher script runs, ``probe_appcontainer_health`` must confirm
an outside-file denial and a network denial on the current machine. An absent,
failed, or inconclusive probe causes the executor to refuse the run. Native
Windows qualification remains required for releases; see
``docs/windows_appcontainer.md``.
"""

from __future__ import annotations

import ctypes
import errno
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IS_WINDOWS = sys.platform == "win32"


def _get_last_error() -> int:
    """Read Win32's thread-local last error without platform-stub assumptions."""
    getter = getattr(ctypes, "get_last_error", None)
    return int(getter()) if callable(getter) else 0

if _IS_WINDOWS:  # pragma: no cover — exercised only on real Windows.
    from ctypes import wintypes


# ---------------------------------------------------------------------------
# Pure planning functions — no OS calls, exercised by the test suite on
# every platform (mirrors how ``_bwrap_argv`` is unit-tested on macOS/CI
# without bwrap ever running). These are the single source of truth for
# *what* gets granted; the ctypes application code below just executes
# the plan.
# ---------------------------------------------------------------------------

# Access masks used in the ACL plan below. These are the standard
# Win32 FILE_* access-right bits (winnt.h) — deliberately narrow:
# GENERIC_READ/GENERIC_EXECUTE for read-only system trees,
# GENERIC_READ/GENERIC_WRITE/GENERIC_EXECUTE for the two writable directory
# paths. GENERIC_EXECUTE supplies directory traversal (the Windows analogue
# of POSIX directory ``x``); without it a confined parser can open an existing
# file but cannot reliably create and enter a scratch subdirectory. No DELETE,
# no WRITE_DAC, no WRITE_OWNER anywhere in this plan — a script cannot
# rewrite its own confinement by changing permissions on a path it can
# write to.
GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
GENERIC_EXECUTE = 0x20000000

_READ_EXECUTE_MASK = GENERIC_READ | GENERIC_EXECUTE
_READ_WRITE_MASK = GENERIC_READ | GENERIC_WRITE | GENERIC_EXECUTE
_MAX_CAPTURED_STREAM_BYTES = 8 * 1024 * 1024
_CAPTURE_TRUNCATION_MARKER = "\n[SIFT OUTPUT TRUNCATED AT {limit} BYTES]\n"
_FILE_SIZE_MIN_POLL_SECONDS = 0.05
_FILE_SIZE_MAX_POLL_SECONDS = 0.5
_FILE_SIZE_SCAN_DUTY_MULTIPLIER = 4.0
_FILE_SIZE_MAX_SCAN_SECONDS = 1.0
# DeleteAppContainerProfile can remain temporarily busy after every process
# and handle owned by Sift has closed. Microsoft explicitly requires callers
# to invoke it again after a failure because the profile state is then
# undetermined. Keep the recovery bounded, but give Windows Security and
# filesystem filter drivers a realistic 1.55-second drain window instead of
# the former 150 ms window.
_PROFILE_DELETE_RETRY_DELAYS_SECONDS = (0.05, 0.1, 0.2, 0.4, 0.8)
_FILE_SIZE_TERMINATION_MARKER = (
    "\n[SIFT SCRIPT STOPPED: file {path!r} reached {observed} bytes, "
    "exceeding the {limit}-byte single-file limit]\n"
)
_DISK_RESERVE_TERMINATION_MARKER = (
    "\n[SIFT SCRIPT STOPPED: writable filesystem {path!r} has {free} free "
    "bytes, below the {reserve}-byte safety reserve]\n"
)


@dataclass(frozen=True)
class WritableScope:
    """One recursively writable tree and any denied subtrees within it.

    AppContainer grants the workspace recursively, protects ``.sift``
    from inheriting that grant, and then allows only the current run
    directory. Representing scopes
    explicitly keeps the file-size monitor bound to the same effective
    write policy instead of accidentally scanning too little (missing the
    run directory) or too much (treating unrelated Sift state as script
    output).
    """

    root: Path
    excluded_subtrees: tuple[Path, ...] = ()


@dataclass(frozen=True)
class FileSizeViolation:
    """A script-created or script-modified file exceeded its ceiling."""

    path: Path
    observed_bytes: int
    limit_bytes: int


@dataclass(frozen=True)
class DiskReserveViolation:
    """A writable filesystem fell below its configured safety reserve."""

    path: Path
    free_bytes: int
    reserve_bytes: int


class FileSizeMonitorUnavailable(OSError):
    """The writable tree cannot be scanned quickly enough to police safely."""


class DiskReserveExceeded(OSError):
    """A run would start, or has continued, below its free-space reserve."""

    def __init__(self, violation: DiskReserveViolation) -> None:
        self.violation = violation
        super().__init__(
            errno.ENOSPC,
            "writable filesystem "
            f"{str(violation.path)!r} has {violation.free_bytes} free bytes, "
            f"below the {violation.reserve_bytes}-byte safety reserve",
        )


@dataclass(frozen=True)
class _FileSnapshot:
    size: int
    mtime_ns: int
    ctime_ns: int
    device: int
    inode: int


def _normalized_path(path: Path) -> str:
    """Absolute, case-normalized spelling without following reparse points."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _is_within(path: str, root: str) -> bool:
    try:
        return os.path.commonpath((path, root)) == root
    except ValueError:
        # Different Windows drive letters cannot contain one another.
        return False


def _is_reparse_point(file_stat: os.stat_result) -> bool:
    attributes = int(getattr(file_stat, "st_file_attributes", 0))
    return bool(attributes & int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)))


def _scan_writable_scopes(
    scopes: tuple[WritableScope, ...],
    *,
    minimum_size_exclusive: int = -1,
) -> dict[str, tuple[Path, _FileSnapshot]]:
    """Snapshot regular files in every effective writable tree.

    Symlinks and Windows reparse points are never followed.  AppContainer
    does not grant their targets merely because the link itself is below a
    writable root, and following them in the unsandboxed parent would both
    overstate the monitor's authority and enable cycles.  Disappearing
    entries are normal during a live scan; other traversal failures are
    propagated so the caller can fail closed rather than silently stop
    enforcing an enabled guard.
    """
    files: dict[str, tuple[Path, _FileSnapshot]] = {}
    visited_dirs: set[tuple[int, int]] = set()
    visited_paths: set[str] = set()

    for scope in scopes:
        root = Path(scope.root)
        root_key = _normalized_path(root)
        excluded = tuple(_normalized_path(path) for path in scope.excluded_subtrees)
        pending = [root]
        while pending:
            directory = pending.pop()
            directory_key = _normalized_path(directory)
            if any(_is_within(directory_key, denied) for denied in excluded):
                continue
            # Multiple writable scopes can overlap (and run_dir may be
            # passed separately from a broader caller-defined scope).  Skip
            # a directory already walked by path before paying for another
            # scandir.  The inode/device set below remains the hard-link /
            # alternate-spelling backstop.
            if directory_key in visited_paths:
                continue
            visited_paths.add(directory_key)
            try:
                directory_stat = directory.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if _is_reparse_point(directory_stat) or not stat.S_ISDIR(directory_stat.st_mode):
                continue
            identity = (int(directory_stat.st_dev), int(directory_stat.st_ino))
            if identity != (0, 0):
                if identity in visited_dirs:
                    continue
                visited_dirs.add(identity)
            try:
                # Stream directory entries.  Materializing ``list(scandir)``
                # first let an extremely wide researcher directory allocate
                # unbounded memory in Sift's unsandboxed parent before the
                # scan-duration fail-closed check could run.
                with os.scandir(directory) as entries:
                    for entry in entries:
                        entry_path = Path(entry.path)
                        entry_key = _normalized_path(entry_path)
                        if not _is_within(entry_key, root_key):
                            continue
                        if any(_is_within(entry_key, denied) for denied in excluded):
                            continue
                        try:
                            entry_stat = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        if _is_reparse_point(entry_stat) or stat.S_ISLNK(entry_stat.st_mode):
                            continue
                        if stat.S_ISDIR(entry_stat.st_mode):
                            pending.append(entry_path)
                            continue
                        if not stat.S_ISREG(entry_stat.st_mode):
                            continue
                        if int(entry_stat.st_size) <= minimum_size_exclusive:
                            continue
                        files.setdefault(
                            entry_key,
                            (
                                entry_path,
                                _FileSnapshot(
                                    size=int(entry_stat.st_size),
                                    mtime_ns=int(entry_stat.st_mtime_ns),
                                    ctime_ns=int(entry_stat.st_ctime_ns),
                                    device=int(entry_stat.st_dev),
                                    inode=int(entry_stat.st_ino),
                                ),
                            ),
                        )
            except FileNotFoundError:
                continue
    return files


class WritableFileSizeMonitor:
    """Parent-side output-capacity guard for Windows AppContainer runs.

    Windows Job Objects have CPU, memory, and process-count limits but no
    analogue of POSIX ``RLIMIT_FSIZE``.  This monitor snapshots every
    effective writable scope before the suspended child is resumed, then
    repeatedly checks those same scopes while the job runs.  A new file
    above the ceiling, a file that grows across it, or any change to a
    grandfathered pre-existing oversized file is a violation.

    This is deliberately parent-side: no interpreter cooperation is
    required and descendants cannot disable it. In addition to the per-file
    ceiling, the guard checks free space on every distinct filesystem that
    backs an effective writable scope. This closes the many-small-files gap:
    no individual file needs to cross the ceiling before the complete Job
    Object is stopped at the configured safety reserve. The capacity probe is
    a cheap filesystem-stat call integrated into the same polling loop; it
    does not add a second directory walk. Traversal and capacity-probe errors
    propagate and terminate the job, so an enabled guard never silently
    becomes a no-op. Unlike kernel-synchronous ``RLIMIT_FSIZE``, detection occurs at
    scan boundaries: a write can overshoot briefly, and a file created and
    removed wholly between scans leaves no artifact for a metadata scan to
    observe.  Adaptive pacing bounds routine scan load, while a tree that
    cannot be scanned inside the safety budget is refused rather than run
    with a misleadingly weak guard.
    """

    def __init__(
        self,
        scopes: tuple[WritableScope, ...],
        limit_bytes: int,
        min_free_disk_bytes: int = 0,
    ) -> None:
        self.scopes = scopes
        self.limit_bytes = max(0, int(limit_bytes))
        self.min_free_disk_bytes = max(0, int(min_free_disk_bytes))
        self.poll_seconds = _FILE_SIZE_MIN_POLL_SECONDS
        self._disk_usage_probes = self._distinct_filesystem_probes(scopes)
        initial_capacity_violation = self._check_disk_reserve()
        if initial_capacity_violation is not None:
            # Refuse before the suspended child is created. Starting below
            # the reserve would guarantee the guard has no safety margin in
            # which to observe and terminate a runaway writer.
            raise DiskReserveExceeded(initial_capacity_violation)
        if self.limit_bytes > 0:
            started = time.monotonic()
            # Only oversized pre-existing files need grandfather metadata.
            # Files at/below the limit are irrelevant until a later scan sees
            # them above it, at which point absence from this baseline is
            # exactly the evidence that they grew.  This keeps memory bounded
            # by exceptional large files rather than every file in the user's
            # workspace.
            self._baseline = _scan_writable_scopes(
                scopes,
                minimum_size_exclusive=self.limit_bytes,
            )
            self._record_scan_duration(time.monotonic() - started)
        else:
            self._baseline = {}

    def _distinct_filesystem_probes(
        self,
        scopes: tuple[WritableScope, ...],
    ) -> tuple[Path, ...]:
        """Return one existing probe path per writable filesystem.

        Workspace and run-directory scopes normally overlap on one volume.
        De-duplicating by ``st_dev`` avoids issuing the same capacity call
        twice per polling interval while still covering configurations whose
        run directory is on another mounted volume. A missing/unstatable root
        is an enforcement failure when the reserve is enabled, not a reason
        to silently omit that writable scope.
        """
        if self.min_free_disk_bytes <= 0:
            return ()
        probes: list[Path] = []
        seen_devices: set[int | str] = set()
        for scope in scopes:
            root = Path(scope.root)
            root_stat = root.stat(follow_symlinks=False)
            device = int(root_stat.st_dev)
            # CPython normally provides a stable non-zero drive identifier on
            # Windows. Retain a normalized path fallback for exotic filesystems
            # that report zero instead of incorrectly merging unrelated roots.
            key: int | str = device if device else _normalized_path(root)
            if key in seen_devices:
                continue
            seen_devices.add(key)
            probes.append(root)
        return tuple(probes)

    def _check_disk_reserve(self) -> DiskReserveViolation | None:
        if self.min_free_disk_bytes <= 0:
            return None
        for probe in self._disk_usage_probes:
            free_bytes = int(shutil.disk_usage(probe).free)
            if free_bytes < self.min_free_disk_bytes:
                return DiskReserveViolation(
                    path=probe,
                    free_bytes=free_bytes,
                    reserve_bytes=self.min_free_disk_bytes,
                )
        return None

    def _record_scan_duration(self, elapsed_seconds: float) -> None:
        """Bound scanner duty cycle without accepting unusably slow scans.

        For normal local trees the 50 ms floor keeps detection responsive.
        As metadata walks get more expensive, waiting roughly four scan
        durations between walks keeps the monitor near a 20% duty cycle.
        A one-second scan is too slow to provide a meaningful runaway-write
        guard and would monopolize filesystem metadata bandwidth, so the
        enabled guard fails closed instead of pretending it is effective.
        """
        elapsed = max(0.0, float(elapsed_seconds))
        if elapsed > _FILE_SIZE_MAX_SCAN_SECONDS:
            raise FileSizeMonitorUnavailable(
                "writable-root scan took "
                f"{elapsed:.3f}s (maximum {_FILE_SIZE_MAX_SCAN_SECONDS:.3f}s)"
            )
        self.poll_seconds = min(
            _FILE_SIZE_MAX_POLL_SECONDS,
            max(
                _FILE_SIZE_MIN_POLL_SECONDS,
                elapsed * _FILE_SIZE_SCAN_DUTY_MULTIPLIER,
            ),
        )

    def check(self) -> FileSizeViolation | DiskReserveViolation | None:
        capacity_violation = self._check_disk_reserve()
        if capacity_violation is not None:
            return capacity_violation
        if self.limit_bytes <= 0:
            return None
        started = time.monotonic()
        current = _scan_writable_scopes(
            self.scopes,
            minimum_size_exclusive=self.limit_bytes,
        )
        self._record_scan_duration(time.monotonic() - started)
        for key, (path, observed) in current.items():
            if observed.size <= self.limit_bytes:
                continue
            baseline_entry = self._baseline.get(key)
            if baseline_entry is None or baseline_entry[1] != observed:
                return FileSizeViolation(
                    path=path,
                    observed_bytes=observed.size,
                    limit_bytes=self.limit_bytes,
                )
        return None


@dataclass(frozen=True)
class AclGrant:
    """One planned ACL operation against a single path.

    ``allow=True`` grants ``mask`` to the run's AppContainer SID.
    ``inherit=False`` makes that ACE apply only to the named directory.
    ``protect=True`` prevents later inheritable ACEs on a parent from
    entering this DACL. This is the safe way to carve ``.sift`` out of
    the writable workspace: the private tree must never inherit the broad
    workspace ALLOW in the first place.
    """

    path: str
    mask: int
    allow: bool
    inherit: bool = True
    protect: bool = False


def plan_acl_grants(
    run_dir: Path,
    cwd: Path,
    extra_read_paths: tuple[str, ...] = (),
) -> tuple[AclGrant, ...]:
    """Build the ordered list of ACL grants for one script run.

    Pure function — touches the filesystem only to skip a
    ``extra_read_paths`` entry that doesn't exist on this machine
    (same existence-check-before-bind pattern ``_bwrap_argv`` uses),
    never to actually apply anything. Order matters and is preserved
    exactly as returned: ``AppContainerRun`` applies grants in this
    order and reverts in reverse order. The ``.sift`` and
    ``.sift/runs`` DACLs are protected *before* the inheritable cwd ACE
    is added, so that broad ACE can never propagate into Sift's private
    state. The package receives traversal-only access on those two
    directories and read/write access on the exact current run.

    Interpreter-owning system trees are intentionally NOT hardcoded
    here the way ``_bwrap_argv`` hardcodes ``/usr``, ``/bin``, etc. —
    Windows has no single, stable, cross-install-method equivalent
    (Python/R can live under ``C:\\Python312``, ``%LOCALAPPDATA%``,
    a venv, a conda env, or a Windows Store alias, and the Windows
    system trees ``C:\\Windows`` /
    ``C:\\Windows\\System32`` are always implicitly readable by every
    process regardless of AppContainer — AppContainer's model is
    "deny extra things," not "deny everything," so standard OS DLLs
    resolve without an explicit grant). Every *interpreter-specific*
    path the executor already collects via ``extra_read_paths``
    (mirroring the macOS/Linux call sites in ``run_script``) is
    granted read+execute here instead.
    """
    grants: list[AclGrant] = []
    for p in extra_read_paths:
        # Windows and Program Files are normally already granted
        # read/execute to ALL APPLICATION PACKAGES. A standard user cannot
        # rewrite their protected DACLs, so attempting to add a redundant
        # per-run ACE would make normally installed R/Stata/Python fail.
        protected = False
        if _IS_WINDOWS and p:
            candidate = os.path.normcase(os.path.abspath(p))
            for variable in (
                "SystemRoot",
                "WINDIR",
                "ProgramFiles",
                "ProgramFiles(x86)",
            ):
                root = os.environ.get(variable)
                if root:
                    try:
                        protected = os.path.commonpath(
                            (candidate, os.path.normcase(os.path.abspath(root)))
                        ) == os.path.normcase(os.path.abspath(root))
                    except ValueError:
                        protected = False
                    if protected:
                        break
        if p and Path(p).exists() and not protected:
            grants.append(AclGrant(path=str(p), mask=_READ_EXECUTE_MASK, allow=True))

    # Protect the private control tree BEFORE placing an inheritable
    # workspace ALLOW on its parent. Existing user/system ACEs remain;
    # only this AppContainer package receives non-inheriting directory
    # traversal, which lets it reach the exact run grant below without
    # listing or reading any other Sift state. Protecting ``runs`` as a
    # second boundary prevents a future parent-DACL change from widening
    # sibling runs while this process is alive.
    sift_dir = cwd / ".sift"
    runs_dir = sift_dir / "runs"
    for private_boundary in (sift_dir, runs_dir):
        if private_boundary.is_dir():
            grants.append(
                AclGrant(
                    path=str(private_boundary),
                    mask=GENERIC_EXECUTE,
                    allow=True,
                    inherit=False,
                    protect=True,
                )
            )

    # Writable: researcher's analysis workspace + this run's scratch
    # dir. Same two-path write scope as the bwrap/sandbox-exec backends.
    grants.append(AclGrant(path=str(cwd), mask=_READ_WRITE_MASK, allow=True))
    if _normalized_path(run_dir) != _normalized_path(cwd):
        grants.append(AclGrant(path=str(run_dir), mask=_READ_WRITE_MASK, allow=True))

    return tuple(grants)


def plan_capability_sids() -> tuple[str, ...]:
    """The list of Windows capability SID names to grant the
    AppContainer token — deliberately EMPTY.

    An AppContainer with zero capabilities cannot access the network
    (no ``internetClient``/``internetClientServer``), cannot access
    the microphone/webcam/location/contacts/etc., and cannot access
    removable storage — every one of those requires an explicit
    capability SID that this function never includes. This is the
    central design decision behind guarantee (1) in the
    module docstring: rather than trying to enumerate and deny every
    capability individually (an allowlist-shaped mistake that's one
    missed entry away from a hole), grant literally none and let
    AppContainer's own default-deny model do the work.

    Returned as a tuple (not applied here) so ``plan_acl_grants``'s
    sibling pure-function shape holds for this too, and so a test can
    assert the emptiness directly without needing ``ctypes`` at all.
    """
    return ()


@dataclass(frozen=True)
class JobLimits:
    """Resource-limit plan for the run's Job Object — the Windows
    analogue of ``resource_limits_preexec``'s RLIMIT_* values.
    Pure data; ``create_job_object`` below is what actually calls
    ``SetInformationJobObject``.
    """

    cpu_seconds: int
    memory_bytes: int
    max_processes: int
    kill_on_job_close: bool = True


def plan_job_limits(
    cpu_seconds: int,
    memory_bytes: int,
    max_processes: int,
) -> JobLimits:
    """Clamp inputs to sane non-negative values and return a
    ``JobLimits`` plan. ``0`` disables that specific limit — same
    "explicit 0 means off, anything else falls back rather than
    silently disabling" posture as
    ``executor.script_cpu_limit_seconds`` etc. (the actual env-var
    parsing / fallback logic is reused from ``executor`` by the
    caller; this function just shapes whatever ints it's given into
    the struct-ready form).
    """
    return JobLimits(
        cpu_seconds=max(0, cpu_seconds),
        memory_bytes=max(0, memory_bytes),
        max_processes=max(0, max_processes),
    )


# ---------------------------------------------------------------------------
# ctypes structures. Field order/types below are transcribed directly
# from the Microsoft Learn pages cited in the module docstring for the
# three structs fetched live while writing this file; the remainder
# follow the stable, decades-unchanged Win32 headers
# (processthreadsapi.h / winbase.h / accctrl.h / aclapi.h).
# ---------------------------------------------------------------------------

if _IS_WINDOWS:  # pragma: no cover — Windows-only definitions.
    ULONG_PTR = ctypes.c_size_t
    SIZE_T = ctypes.c_size_t

    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", ctypes.c_void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    class SID_AND_ATTRIBUTES(ctypes.Structure):
        _fields_ = [
            ("Sid", ctypes.c_void_p),
            ("Attributes", wintypes.DWORD),
        ]

    class SECURITY_CAPABILITIES(ctypes.Structure):
        # Matches learn.microsoft.com/.../ns-winnt-security_capabilities
        # exactly: AppContainerSid, then Capabilities (pointer to an
        # array of SID_AND_ATTRIBUTES), CapabilityCount, Reserved.
        _fields_ = [
            ("AppContainerSid", ctypes.c_void_p),
            ("Capabilities", ctypes.POINTER(SID_AND_ATTRIBUTES)),
            ("CapabilityCount", wintypes.DWORD),
            ("Reserved", wintypes.DWORD),
        ]

    class STARTUPINFOW(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("lpReserved", wintypes.LPWSTR),
            ("lpDesktop", wintypes.LPWSTR),
            ("lpTitle", wintypes.LPWSTR),
            ("dwX", wintypes.DWORD),
            ("dwY", wintypes.DWORD),
            ("dwXSize", wintypes.DWORD),
            ("dwYSize", wintypes.DWORD),
            ("dwXCountChars", wintypes.DWORD),
            ("dwYCountChars", wintypes.DWORD),
            ("dwFillAttribute", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("wShowWindow", wintypes.WORD),
            ("cbReserved2", wintypes.WORD),
            ("lpReserved2", ctypes.c_void_p),
            ("hStdInput", wintypes.HANDLE),
            ("hStdOutput", wintypes.HANDLE),
            ("hStdError", wintypes.HANDLE),
        ]

    class STARTUPINFOEXW(ctypes.Structure):
        _fields_ = [
            ("StartupInfo", STARTUPINFOW),
            ("lpAttributeList", ctypes.c_void_p),
        ]

    class PROCESS_INFORMATION(ctypes.Structure):
        _fields_ = [
            ("hProcess", wintypes.HANDLE),
            ("hThread", wintypes.HANDLE),
            ("dwProcessId", wintypes.DWORD),
            ("dwThreadId", wintypes.DWORD),
        ]

    class IO_COUNTERS(ctypes.Structure):
        # "Reserved" per JOBOBJECT_EXTENDED_LIMIT_INFORMATION's own
        # docs, but the struct's six ULARGE_INTEGER fields still have
        # to be laid out correctly for the surrounding struct's total
        # size/offsets to be right.
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        # Matches learn.microsoft.com/.../ns-winnt-jobobject_basic_limit_information
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),  # LARGE_INTEGER
            ("PerJobUserTimeLimit", ctypes.c_int64),  # LARGE_INTEGER
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", SIZE_T),
            ("MaximumWorkingSetSize", SIZE_T),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ULONG_PTR),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        # Matches learn.microsoft.com/.../ns-winnt-jobobject_extended_limit_information
        _fields_ = [
            ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", SIZE_T),
            ("JobMemoryLimit", SIZE_T),
            ("PeakProcessMemoryUsed", SIZE_T),
            ("PeakJobMemoryUsed", SIZE_T),
        ]

    class JOBOBJECT_BASIC_UI_RESTRICTIONS(ctypes.Structure):
        _fields_ = [("UIRestrictionsClass", wintypes.DWORD)]

    class TRUSTEE_W(ctypes.Structure):
        _fields_ = [
            ("pMultipleTrustee", ctypes.c_void_p),
            ("MultipleTrusteeOperation", ctypes.c_int),
            ("TrusteeForm", ctypes.c_int),
            ("TrusteeType", ctypes.c_int),
            ("ptstrName", ctypes.c_void_p),  # PSID when TrusteeForm=TRUSTEE_IS_SID
        ]

    class EXPLICIT_ACCESS_W(ctypes.Structure):
        _fields_ = [
            ("grfAccessPermissions", wintypes.DWORD),
            ("grfAccessMode", ctypes.c_int),
            ("grfInheritance", wintypes.DWORD),
            ("Trustee", TRUSTEE_W),
        ]

    # accctrl.h enum values used above.
    TRUSTEE_IS_SID = 0
    TRUSTEE_IS_UNKNOWN = 0
    GRANT_ACCESS = 1
    DENY_ACCESS = 3
    NO_MULTIPLE_TRUSTEE = 0
    SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
    NO_INHERITANCE = 0x0
    SE_FILE_OBJECT = 1
    DACL_SECURITY_INFORMATION = 0x00000004
    UNPROTECTED_DACL_SECURITY_INFORMATION = 0x20000000
    PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
    SE_DACL_PROTECTED = 0x1000

    JobObjectBasicUIRestrictions = 4
    JobObjectExtendedLimitInformation = 9

    JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004
    JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
    JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
    JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

    # Headless research scripts have no reason to reach the desktop,
    # clipboard, display settings, or global atom table.  Blocking these
    # surfaces also removes several routes to broker another desktop
    # process into doing work outside the AppContainer on the script's
    # behalf.
    JOB_OBJECT_UILIMIT_HANDLES = 0x00000001
    JOB_OBJECT_UILIMIT_READCLIPBOARD = 0x00000002
    JOB_OBJECT_UILIMIT_WRITECLIPBOARD = 0x00000004
    JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS = 0x00000008
    JOB_OBJECT_UILIMIT_DISPLAYSETTINGS = 0x00000010
    JOB_OBJECT_UILIMIT_GLOBALATOMS = 0x00000020
    JOB_OBJECT_UILIMIT_DESKTOP = 0x00000040
    JOB_OBJECT_UILIMIT_EXITWINDOWS = 0x00000080

    EXTENDED_STARTUPINFO_PRESENT = 0x00080000
    CREATE_SUSPENDED = 0x00000004
    CREATE_NO_WINDOW = 0x08000000
    CREATE_UNICODE_ENVIRONMENT = 0x00000400

    PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
    PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009

    INFINITE = 0xFFFFFFFF
    WAIT_TIMEOUT = 0x00000102
    WAIT_OBJECT_0 = 0x00000000
    WAIT_ABANDONED = 0x00000080
    WAIT_FAILED = 0xFFFFFFFF
    STILL_ACTIVE = 259
    ERROR_BROKEN_PIPE = 109

    _win_dll_factory = getattr(ctypes, "WinDLL", None)
    if _win_dll_factory is None:
        raise RuntimeError("Windows DLL loading is unavailable on this platform")
    _kernel32 = _win_dll_factory("kernel32", use_last_error=True)
    _advapi32 = _win_dll_factory("advapi32", use_last_error=True)
    _userenv = _win_dll_factory("userenv", use_last_error=True)

    _kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.LPCWSTR,
        ctypes.POINTER(STARTUPINFOEXW),
        ctypes.POINTER(PROCESS_INFORMATION),
    ]
    _kernel32.CreateProcessW.restype = wintypes.BOOL

    _kernel32.InitializeProcThreadAttributeList.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    _kernel32.InitializeProcThreadAttributeList.restype = wintypes.BOOL

    _kernel32.UpdateProcThreadAttribute.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _kernel32.UpdateProcThreadAttribute.restype = wintypes.BOOL

    _kernel32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
    _kernel32.DeleteProcThreadAttributeList.restype = None

    _kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    _kernel32.CreateJobObjectW.restype = wintypes.HANDLE

    _kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    _kernel32.SetInformationJobObject.restype = wintypes.BOOL

    _kernel32.CreateMutexW.argtypes = [
        ctypes.POINTER(SECURITY_ATTRIBUTES),
        wintypes.BOOL,
        wintypes.LPCWSTR,
    ]
    _kernel32.CreateMutexW.restype = wintypes.HANDLE

    _kernel32.ReleaseMutex.argtypes = [wintypes.HANDLE]
    _kernel32.ReleaseMutex.restype = wintypes.BOOL

    _kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    _kernel32.AssignProcessToJobObject.restype = wintypes.BOOL

    _kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateJobObject.restype = wintypes.BOOL

    _kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
    _kernel32.ResumeThread.restype = wintypes.DWORD

    _kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _kernel32.WaitForSingleObject.restype = wintypes.DWORD

    _kernel32.GetExitCodeProcess.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.DWORD),
    ]
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL

    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel32.CloseHandle.restype = wintypes.BOOL

    _kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    _kernel32.TerminateProcess.restype = wintypes.BOOL

    _kernel32.CreatePipe.argtypes = [
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.POINTER(wintypes.HANDLE),
        ctypes.POINTER(SECURITY_ATTRIBUTES),
        wintypes.DWORD,
    ]
    _kernel32.CreatePipe.restype = wintypes.BOOL

    _kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    _kernel32.ReadFile.restype = wintypes.BOOL

    _kernel32.SetHandleInformation.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    _kernel32.SetHandleInformation.restype = wintypes.BOOL

    # LocalFree is exported by Kernel32.dll (winbase.h), not Advapi32.
    # SetEntriesInAclW and GetNamedSecurityInfoW return buffers that the
    # caller must release with this function.
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p

    HANDLE_FLAG_INHERIT = 0x00000001

    _advapi32.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL

    _advapi32.SetEntriesInAclW.argtypes = [
        wintypes.ULONG,
        ctypes.POINTER(EXPLICIT_ACCESS_W),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _advapi32.SetEntriesInAclW.restype = wintypes.DWORD

    _advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD

    _advapi32.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.WORD),
        ctypes.POINTER(wintypes.DWORD),
    ]
    _advapi32.GetSecurityDescriptorControl.restype = wintypes.BOOL

    _advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    _advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD

    # ``CreateAppContainerProfile``'s ``ppSidAppContainerSid`` out-SID
    # is explicitly documented (learn.microsoft.com/.../
    # nf-userenv-createappcontainerprofile) as: "This buffer must be
    # freed using the FreeSid function" -- NOT LocalFree. FreeSid is
    # exported by Advapi32.dll (securitybaseapi.h) despite freeing a
    # buffer userenv.dll allocated; that's Microsoft's documented
    # contract for this specific API, not a mismatch in this binding.
    # Calling LocalFree on a SID buffer FreeSid owns is undefined
    # behaviour per that contract.
    _advapi32.FreeSid.argtypes = [ctypes.c_void_p]
    _advapi32.FreeSid.restype = ctypes.c_void_p

    _userenv.CreateAppContainerProfile.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        ctypes.POINTER(SID_AND_ATTRIBUTES),
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _userenv.CreateAppContainerProfile.restype = ctypes.c_long  # HRESULT

    _userenv.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
    _userenv.DeleteAppContainerProfile.restype = ctypes.c_long  # HRESULT


# ---------------------------------------------------------------------------
# Windows-only application layer. Every function below raises
# ``RuntimeError`` immediately if called off-Windows — this module is
# imported unconditionally (its pure planning functions are exercised
# in tests on every platform), but nothing past this point should ever
# execute anywhere except a real Windows machine.
# ---------------------------------------------------------------------------


def _require_windows() -> None:
    if not _IS_WINDOWS:
        raise RuntimeError(
            "sift.win_appcontainer's OS-calling functions only run on "
            "win32; this platform is "
            f"{sys.platform!r}. This should be unreachable — callers "
            "must check sys.platform before reaching here, exactly like "
            "executor.run_script's platform dispatch does for the "
            "macOS/Linux backends."
        )


def _hresult_failed(hr: int) -> bool:
    """HRESULT failure test: the sign bit (bit 31) is set on failure —
    the standard ``FAILED(hr)`` macro."""
    return (hr & 0x80000000) != 0


# Win32 error codes relevant to distinguishing "the interpreter
# binary itself is missing" from "Sift's sandbox plumbing broke" --
# both surface through ``CreateProcessW`` failing inside
# ``spawn_in_appcontainer`` and both raise ``AppContainerError``, but
# they need different researcher-facing messages (see
# ``executor.run_script``'s ``except _AppContainerErrorCls`` handler).
# Defined unconditionally (not gated behind ``_IS_WINDOWS``) since
# they're plain ints a test can reference on any platform to
# construct an ``AppContainerError`` matching this exact shape.
ERROR_FILE_NOT_FOUND = 2  # CreateProcessW's application path doesn't exist.
ERROR_PATH_NOT_FOUND = 3  # A directory component of the path doesn't exist.
ERROR_ACCESS_DENIED = 5


class AppContainerError(RuntimeError):
    """Raised when a Windows AppContainer/Job Object API call fails.
    Carries the raw Win32/HRESULT error code so callers (the health
    probe, the doctor report) can surface something more actionable
    than a bare exception message.

    ``operation`` is the specific Win32/HRESULT call that failed
    (``"CreateProcessW"``, ``"AssignProcessToJobObject"``, etc.) --
    callers use it to tell "the interpreter binary vanished"
    (``operation == "CreateProcessW"`` with a file-not-found code)
    apart from an actual Sift-side sandbox-plumbing bug (any other
    operation, or any other code), which need different advice.
    """

    def __init__(self, operation: str, code: int) -> None:
        self.operation = operation
        self.code = code
        super().__init__(f"{operation} failed (error {code:#010x})")

    def is_missing_interpreter(self) -> bool:
        """True iff this failure is ``CreateProcessW`` unable to find
        the target executable -- the Windows-side equivalent of the
        POSIX ``FileNotFoundError`` the macOS/Linux spawn path raises
        for the exact same underlying condition (interpreter binary
        doesn't exist, e.g. uninstalled or removed after the startup
        doctor check ran). Every OTHER ``AppContainerError`` (ACL
        setup, job object, profile creation, any other CreateProcessW
        failure code) is a genuine sandbox-plumbing failure that
        SHOULD be reported as a bug.
        """
        return self.operation == "CreateProcessW" and self.code in (
            ERROR_FILE_NOT_FOUND,
            ERROR_PATH_NOT_FOUND,
        )


def _windows_environment_block(env: dict[str, str]) -> str:
    """Return a valid, deterministically ordered CreateProcess block.

    Windows environment names are case-insensitive and the documented
    block representation cannot contain embedded NULs.  Reject malformed
    input rather than truncating it at the native boundary, where the
    child could receive a materially different policy environment from
    the one the caller inspected.
    """
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw_key, raw_value in env.items():
        key, value = str(raw_key), str(raw_value)
        if not key or "=" in key or "\0" in key or "\0" in value:
            raise ValueError(f"invalid Windows environment entry: {key!r}")
        folded = key.casefold()
        if folded in seen:
            raise ValueError(f"duplicate case-insensitive environment key: {key!r}")
        seen.add(folded)
        entries.append((key, value))
    entries.sort(key=lambda item: item[0].casefold())
    return "\0".join(f"{key}={value}" for key, value in entries) + "\0\0"


def _acquire_acl_mutex() -> Callable[[], None]:
    """Serialize Windows ACL snapshots for the complete sandbox run.

    Interpreter and workspace DACLs are shared persistent state.  Without
    this cross-process mutex, two Sift windows can snapshot, modify, and
    restore the same DACL out of order, either revoking a live run's ACE or
    leaving a stale one behind.  The mutex is held until all children are
    dead and every original DACL has been restored.
    """
    _require_windows()
    handle = _kernel32.CreateMutexW(None, False, "Local\\Sift.AppContainer.ACL.v1")
    if not handle:
        raise AppContainerError("CreateMutexW(ACL)", _get_last_error())
    result = _kernel32.WaitForSingleObject(handle, INFINITE)
    if result not in (WAIT_OBJECT_0, WAIT_ABANDONED):
        err = _get_last_error() if result == WAIT_FAILED else int(result)
        _kernel32.CloseHandle(handle)
        raise AppContainerError("WaitForSingleObject(ACL mutex)", err)

    released = False

    def _release() -> None:
        nonlocal released
        if released:
            return
        released = True
        try:
            if not _kernel32.ReleaseMutex(handle):
                raise AppContainerError("ReleaseMutex(ACL)", _get_last_error())
        finally:
            _kernel32.CloseHandle(handle)

    return _release


def create_appcontainer_profile(
    name: str,
) -> tuple[ctypes.c_void_p, Callable[[], None]]:
    """Create a per-run AppContainer profile with zero capabilities.

    Returns ``(sid, cleanup)``. ``sid`` is the ``PSID`` (as an opaque
    ``c_void_p``) to embed in the ``SECURITY_CAPABILITIES`` struct and
    to grant/deny ACEs against. ``cleanup()`` frees the SID and
    deletes the profile — callers MUST call it, and
    ``AppContainerRun`` does so in a ``finally`` block so a crashed or
    timed-out script can never leak a profile.

    ``name`` must be unique per run (a stale profile from an earlier
    crashed run left behind would be a genuine, if low-severity,
    persistence leak — a fresh UUID-suffixed name per run, which every
    caller in this module uses, avoids that entirely rather than
    relying on cleanup always succeeding).
    """
    _require_windows()
    out_sid = ctypes.c_void_p()
    hr = _userenv.CreateAppContainerProfile(
        name,
        name,
        "Sift per-run sandbox — no persistent capabilities.",
        None,
        0,
        ctypes.byref(out_sid),
    )
    if _hresult_failed(hr):
        raise AppContainerError("CreateAppContainerProfile", hr)

    def _cleanup() -> None:
        # Microsoft documents that deletion can transiently fail while
        # profile resources finish closing, and explicitly recommends a
        # retry.  A leaked profile is security-relevant persistent state,
        # so do not silently convert failure into success.
        last_hr = 0
        try:
            attempts = len(_PROFILE_DELETE_RETRY_DELAYS_SECONDS) + 1
            for attempt in range(attempts):
                last_hr = int(_userenv.DeleteAppContainerProfile(name))
                if not _hresult_failed(last_hr):
                    break
                if attempt < len(_PROFILE_DELETE_RETRY_DELAYS_SECONDS):
                    time.sleep(_PROFILE_DELETE_RETRY_DELAYS_SECONDS[attempt])
            else:
                raise AppContainerError("DeleteAppContainerProfile", last_hr)
        finally:
            # FreeSid, not LocalFree -- see the binding comment above.
            if _advapi32.FreeSid(out_sid):
                raise AppContainerError("FreeSid", _get_last_error())

    return out_sid, _cleanup


def grant_acl(grant: AclGrant, sid: ctypes.c_void_p) -> Callable[[], None]:
    """Apply one planned ACL grant/deny to the AppContainer SID on
    ``grant.path``, returning a ``revert()`` callable that restores
    the path's original DACL.

    Reads the CURRENT DACL first (via ``GetNamedSecurityInfoW``) and
    snapshots it before calling ``SetEntriesInAclW`` to merge in the
    new ACE and ``SetNamedSecurityInfoW`` to apply the merged result —
    the snapshot is what ``revert()`` writes back, rather than trying
    to compute "the DACL minus just our ACE" (which would be wrong if
    anything else touched the DACL mid-run). This mirrors the module
    docstring's point about ACL grants being persistent filesystem
    state, unlike bwrap's mount namespace which vanishes on its own —
    the caller (``AppContainerRun``) is responsible for calling every
    ``revert()`` even when the run crashes or times out.
    """
    _require_windows()
    path = grant.path

    old_dacl = ctypes.c_void_p()
    old_sd = ctypes.c_void_p()
    err = _advapi32.GetNamedSecurityInfoW(
        path,
        SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(old_dacl),
        None,
        ctypes.byref(old_sd),
    )
    if err != 0:
        raise AppContainerError(f"GetNamedSecurityInfoW({path!r})", err)

    old_control = wintypes.WORD()
    old_revision = wintypes.DWORD()
    if not _advapi32.GetSecurityDescriptorControl(
        old_sd,
        ctypes.byref(old_control),
        ctypes.byref(old_revision),
    ):
        code = _get_last_error()
        _kernel32.LocalFree(old_sd)
        raise AppContainerError(
            f"GetSecurityDescriptorControl({path!r})",
            code,
        )
    was_protected = bool(old_control.value & SE_DACL_PROTECTED)

    trustee = TRUSTEE_W()
    trustee.pMultipleTrustee = None
    trustee.MultipleTrusteeOperation = NO_MULTIPLE_TRUSTEE
    trustee.TrusteeForm = TRUSTEE_IS_SID
    trustee.TrusteeType = TRUSTEE_IS_UNKNOWN
    trustee.ptstrName = ctypes.cast(sid, ctypes.c_void_p)

    ea = EXPLICIT_ACCESS_W()
    ea.grfAccessPermissions = grant.mask
    ea.grfAccessMode = GRANT_ACCESS if grant.allow else DENY_ACCESS
    ea.grfInheritance = (
        SUB_CONTAINERS_AND_OBJECTS_INHERIT if grant.inherit else NO_INHERITANCE
    )
    ea.Trustee = trustee

    new_dacl = ctypes.c_void_p()
    err = _advapi32.SetEntriesInAclW(
        1, ctypes.byref(ea), old_dacl, ctypes.byref(new_dacl)
    )
    if err != 0:
        _kernel32.LocalFree(old_sd)
        raise AppContainerError(f"SetEntriesInAclW({path!r})", err)

    apply_security_information = DACL_SECURITY_INFORMATION
    if grant.protect:
        apply_security_information |= PROTECTED_DACL_SECURITY_INFORMATION
    err = _advapi32.SetNamedSecurityInfoW(
        path,
        SE_FILE_OBJECT,
        apply_security_information,
        None,
        None,
        new_dacl,
        None,
    )
    _kernel32.LocalFree(new_dacl)
    if err != 0:
        _kernel32.LocalFree(old_sd)
        raise AppContainerError(f"SetNamedSecurityInfoW({path!r}, apply)", err)

    def _revert() -> None:
        try:
            restore_security_information = DACL_SECURITY_INFORMATION | (
                PROTECTED_DACL_SECURITY_INFORMATION
                if was_protected
                else UNPROTECTED_DACL_SECURITY_INFORMATION
            )
            revert_err = _advapi32.SetNamedSecurityInfoW(
                path,
                SE_FILE_OBJECT,
                restore_security_information,
                None,
                None,
                old_dacl,
                None,
            )
            if revert_err != 0:
                raise AppContainerError(
                    f"SetNamedSecurityInfoW({path!r}, revert)",
                    revert_err,
                )
        finally:
            _kernel32.LocalFree(old_sd)

    return _revert


def create_job_object(limits: JobLimits) -> ctypes.c_void_p:
    """Create an unnamed Job Object configured with ``limits``.
    Caller owns the returned handle and must ``CloseHandle`` it (which
    — combined with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` — is what
    guarantees every process ever assigned to this job, including any
    the researcher's script itself spawned, is torn down when the run
    ends, whether cleanly, by timeout, or by crash).
    """
    _require_windows()
    handle = _kernel32.CreateJobObjectW(None, None)
    if not handle:
        raise AppContainerError("CreateJobObjectW", _get_last_error())

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    flags = 0
    if limits.kill_on_job_close:
        flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    # Also force SEM_NOGPFAULTERRORBOX-equivalent behaviour for every
    # process in the job — an interpreter crash shouldn't be able to
    # pop a blocking WER dialog on the researcher's desktop and hang
    # the run past the wall-clock timeout.
    flags |= JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
    if limits.cpu_seconds > 0:
        flags |= JOB_OBJECT_LIMIT_JOB_TIME
        # 100-nanosecond ticks, per JOBOBJECT_BASIC_LIMIT_INFORMATION's docs.
        info.BasicLimitInformation.PerJobUserTimeLimit = limits.cpu_seconds * 10_000_000
    if limits.max_processes > 0:
        flags |= JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        info.BasicLimitInformation.ActiveProcessLimit = limits.max_processes
    info.BasicLimitInformation.LimitFlags = flags
    if limits.memory_bytes > 0:
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY
        info.JobMemoryLimit = limits.memory_bytes

    ok = _kernel32.SetInformationJobObject(
        handle,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        err = _get_last_error()
        _kernel32.CloseHandle(handle)
        raise AppContainerError("SetInformationJobObject", err)

    ui = JOBOBJECT_BASIC_UI_RESTRICTIONS()
    ui.UIRestrictionsClass = (
        JOB_OBJECT_UILIMIT_HANDLES
        | JOB_OBJECT_UILIMIT_READCLIPBOARD
        | JOB_OBJECT_UILIMIT_WRITECLIPBOARD
        | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS
        | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS
        | JOB_OBJECT_UILIMIT_GLOBALATOMS
        | JOB_OBJECT_UILIMIT_DESKTOP
        | JOB_OBJECT_UILIMIT_EXITWINDOWS
    )
    if not _kernel32.SetInformationJobObject(
        handle,
        JobObjectBasicUIRestrictions,
        ctypes.byref(ui),
        ctypes.sizeof(ui),
    ):
        err = _get_last_error()
        _kernel32.CloseHandle(handle)
        raise AppContainerError("SetInformationJobObject(UI)", err)
    return handle


def _build_security_capabilities(sid: ctypes.c_void_p) -> SECURITY_CAPABILITIES:
    """Build a ``SECURITY_CAPABILITIES`` struct with the given
    AppContainer SID and ZERO capabilities — see
    ``plan_capability_sids``'s docstring for why an empty capability
    set is the correct, deliberate choice here rather than an
    oversight."""
    sec_cap = SECURITY_CAPABILITIES()
    sec_cap.AppContainerSid = sid
    sec_cap.Capabilities = None
    sec_cap.CapabilityCount = 0
    sec_cap.Reserved = 0
    return sec_cap


class AppContainerProcess:
    """``subprocess.Popen``-shaped wrapper around a process spawned
    inside an AppContainer + Job Object. Exposes exactly the surface
    ``executor.run_script`` needs (``pid``, ``communicate(timeout=)``,
    ``kill()``, ``returncode``) so the shared post-spawn logic in
    ``run_script`` (result-file parsing, stderr splitting, etc.) does
    not need a second, Windows-specific copy.

    ``kill()`` calls ``TerminateJobObject`` rather than
    ``TerminateProcess`` on the direct child — this is the Windows
    replacement for the Linux/macOS ``os.killpg(os.getpgid(...))``
    call in ``run_script``'s timeout handler, and is actually a
    STRONGER guarantee: it reaches every process ever assigned to the
    job (R ``parallel::makeCluster`` workers, Python
    ``multiprocessing.Pool`` children, anything the script itself
    spawned), not just processes that stayed in the same POSIX
    process group.
    """

    def __init__(
        self,
        proc_info: PROCESS_INFORMATION,
        job_handle: ctypes.c_void_p,
        stdout_read: ctypes.c_void_p,
        stderr_read: ctypes.c_void_p,
        cleanup: Callable[[], None],
        file_size_monitor: WritableFileSizeMonitor | None = None,
    ) -> None:
        self._proc_info = proc_info
        self._job = job_handle
        self._stdout_read = stdout_read
        self._stderr_read = stderr_read
        self._cleanup = cleanup
        self._file_size_monitor = file_size_monitor
        self.file_size_violation: FileSizeViolation | None = None
        self.disk_reserve_violation: DiskReserveViolation | None = None
        self.file_size_monitor_error: Exception | None = None
        self._cleaned_up = False
        self.pid = proc_info.dwProcessId
        self.returncode: int | None = None
        # Reader-thread state lives on the instance, not as
        # ``communicate()`` locals — see that method's docstring for
        # why a naive per-call implementation is a real threading bug
        # here (unlike ``subprocess.Popen.communicate()``, which is
        # documented as safe to call again after a ``TimeoutExpired``).
        self._out_holder: dict[str, Any] = {}
        self._err_holder: dict[str, Any] = {}
        self._reader_threads_started = False
        self._t_out: Any = None
        self._t_err: Any = None

    def _read_pipe_all(self, handle: ctypes.c_void_p) -> str:
        captured = bytearray()
        truncated = False
        buf = ctypes.create_string_buffer(65536)
        bytes_read = wintypes.DWORD(0)
        while True:
            ok = _kernel32.ReadFile(
                handle,
                buf,
                len(buf),
                ctypes.byref(bytes_read),
                None,
            )
            if not ok:
                err = _get_last_error()
                if err == ERROR_BROKEN_PIPE:
                    break
                raise AppContainerError("ReadFile(pipe)", err)
            if bytes_read.value == 0:
                break
            chunk = buf.raw[: bytes_read.value]
            remaining = _MAX_CAPTURED_STREAM_BYTES - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                truncated = True
        text = bytes(captured).decode("utf-8", errors="replace")
        if truncated:
            text += _CAPTURE_TRUNCATION_MARKER.format(
                limit=_MAX_CAPTURED_STREAM_BYTES,
            )
        return text

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        """Wait for the process to exit (or ``timeout`` seconds,
        whichever comes first), draining stdout/stderr pipes on
        background threads so a chatty script can't deadlock against
        an unread, full pipe buffer the way a naive
        wait-then-read implementation would.

        Raises ``subprocess.TimeoutExpired`` on timeout — the SAME
        exception type ``run_script``'s existing
        ``except subprocess.TimeoutExpired:`` handler already catches,
        so no Windows-specific branch is needed at that call site
        beyond the ``killpg`` platform check documented in
        ``executor.run_script``.

        ``executor.run_script``'s own timeout-recovery path calls
        ``communicate()`` a SECOND time (with a short timeout) after
        killing a timed-out process — the same "kill, then drain"
        pattern ``subprocess.run`` itself uses, and which
        ``subprocess.Popen.communicate()`` is explicitly documented
        as safe to call again for. A first implementation of this
        method started BRAND NEW reader threads on every call,
        unconditionally — the second call would spin up a second
        pair of threads reading from the SAME pipe handles the
        first call's (still-running, never-joined, never-stopped)
        threads were already draining. Two threads calling
        ``ReadFile`` concurrently on one handle is a genuine data
        race: bytes could be split unpredictably between the
        abandoned first call's ``out_holder`` (which the caller can
        never see again, since that call already raised
        ``TimeoutExpired`` and returned control) and the second
        call's fresh one, silently losing part of the script's
        output. The reader threads and their holders are now
        instance state, started exactly once (guarded by
        ``_reader_threads_started``); every subsequent call reuses
        the SAME threads and joins them again rather than racing a
        new pair against the old.
        """
        import threading

        if not self._reader_threads_started:
            self._reader_threads_started = True

            def _drain(handle, holder, key) -> None:
                try:
                    holder[key] = self._read_pipe_all(handle)
                except Exception as exc:  # noqa: BLE001 - preserve native failures
                    holder["error"] = exc

            self._t_out = threading.Thread(
                target=_drain,
                args=(self._stdout_read, self._out_holder, "v"),
                daemon=True,
            )
            self._t_err = threading.Thread(
                target=_drain,
                args=(self._stderr_read, self._err_holder, "v"),
                daemon=True,
            )
            self._t_out.start()
            self._t_err.start()

        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._file_size_monitor is None:
                wait_ms = (
                    INFINITE
                    if deadline is None
                    else max(0, int((deadline - time.monotonic()) * 1000))
                )
            else:
                poll_seconds = min(
                    _FILE_SIZE_MAX_POLL_SECONDS,
                    max(
                        _FILE_SIZE_MIN_POLL_SECONDS,
                        float(
                            getattr(
                                self._file_size_monitor,
                                "poll_seconds",
                                _FILE_SIZE_MIN_POLL_SECONDS,
                            )
                        ),
                    ),
                )
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(
                            cmd="<appcontainer>", timeout=timeout or 0,
                        )
                    poll_seconds = min(poll_seconds, remaining)
                wait_ms = max(1, int(poll_seconds * 1000))

            rc = _kernel32.WaitForSingleObject(self._proc_info.hProcess, wait_ms)
            if rc not in (WAIT_OBJECT_0, WAIT_TIMEOUT):
                err = _get_last_error() if rc == WAIT_FAILED else int(rc)
                raise AppContainerError("WaitForSingleObject(process)", err)
            if rc == WAIT_TIMEOUT and self._file_size_monitor is None:
                # With no polling monitor, the single native wait covered
                # the caller's complete timeout (the pre-monitor behavior).
                raise subprocess.TimeoutExpired(
                    cmd="<appcontainer>", timeout=timeout or 0,
                )

            if self._file_size_monitor is not None:
                try:
                    guard_violation = self._file_size_monitor.check()
                    if isinstance(guard_violation, FileSizeViolation):
                        self.file_size_violation = guard_violation
                    elif isinstance(guard_violation, DiskReserveViolation):
                        self.disk_reserve_violation = guard_violation
                    elif guard_violation is not None:
                        raise TypeError(
                            "Windows writable-output monitor returned an "
                            "unknown violation type"
                        )
                except Exception as exc:  # noqa: BLE001 - fail closed on scan loss
                    self.file_size_monitor_error = exc
                if (
                    self.file_size_violation is not None
                    or self.disk_reserve_violation is not None
                    or self.file_size_monitor_error
                ):
                    if not _kernel32.TerminateJobObject(self._job, 1):
                        raise AppContainerError(
                            "TerminateJobObject(writable-output guard)",
                            _get_last_error(),
                        )
                    # Job termination is asynchronous.  Wait for the direct
                    # process to become signalled before reading its exit code
                    # and closing inherited output handles below.
                    stopped = _kernel32.WaitForSingleObject(
                        self._proc_info.hProcess,
                        5000,
                    )
                    if stopped != WAIT_OBJECT_0:
                        err = (
                            _get_last_error()
                            if stopped == WAIT_FAILED
                            else int(stopped)
                        )
                        raise AppContainerError(
                            "WaitForSingleObject(writable-output termination)", err,
                        )
                    break

            if rc == WAIT_OBJECT_0:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(
                    cmd="<appcontainer>", timeout=timeout or 0,
                )

        exit_code = wintypes.DWORD(0)
        if not _kernel32.GetExitCodeProcess(
            self._proc_info.hProcess,
            ctypes.byref(exit_code),
        ):
            raise AppContainerError("GetExitCodeProcess", _get_last_error())
        self.returncode = exit_code.value

        # The direct interpreter can exit while descendants retain copies
        # of its pipe handles.  End the job after recording the interpreter's
        # exit status so those handles close and output draining cannot hang
        # or silently return partial data.
        if not _kernel32.TerminateJobObject(self._job, 1):
            raise AppContainerError(
                "TerminateJobObject(descendants)", _get_last_error()
            )

        # Scan once more only after every descendant has been stopped.  This
        # closes the direct-child-exit race: a worker could otherwise create
        # or extend a file between the last timed scan and the job teardown.
        if (
            self._file_size_monitor is not None
            and self.file_size_violation is None
            and self.disk_reserve_violation is None
            and self.file_size_monitor_error is None
        ):
            try:
                guard_violation = self._file_size_monitor.check()
                if isinstance(guard_violation, FileSizeViolation):
                    self.file_size_violation = guard_violation
                elif isinstance(guard_violation, DiskReserveViolation):
                    self.disk_reserve_violation = guard_violation
                elif guard_violation is not None:
                    raise TypeError(
                        "Windows writable-output monitor returned an unknown "
                        "violation type"
                    )
            except Exception as exc:  # noqa: BLE001 - fail closed on scan loss
                self.file_size_monitor_error = exc
        if (
            self.file_size_violation is not None
            or self.disk_reserve_violation is not None
            or self.file_size_monitor_error
        ):
            # A process may finish milliseconds before the final scan notices
            # its oversized output.  Override a possibly-successful native
            # exit status so result handling can never accept that run.
            self.returncode = 1

        self._t_out.join(timeout=5)
        self._t_err.join(timeout=5)
        if self._t_out.is_alive() or self._t_err.is_alive():
            raise AppContainerError("drain inherited output handles", WAIT_TIMEOUT)
        for holder in (self._out_holder, self._err_holder):
            if "error" in holder:
                raise holder["error"]
        stdout = str(self._out_holder.get("v", ""))
        stderr = str(self._err_holder.get("v", ""))
        if self.file_size_violation is not None:
            file_violation = self.file_size_violation
            stderr += _FILE_SIZE_TERMINATION_MARKER.format(
                path=str(file_violation.path),
                observed=file_violation.observed_bytes,
                limit=file_violation.limit_bytes,
            )
        elif self.disk_reserve_violation is not None:
            disk_violation = self.disk_reserve_violation
            stderr += _DISK_RESERVE_TERMINATION_MARKER.format(
                path=str(disk_violation.path),
                free=disk_violation.free_bytes,
                reserve=disk_violation.reserve_bytes,
            )
        elif self.file_size_monitor_error is not None:
            stderr += (
                "\n[SIFT SCRIPT STOPPED: the enabled Windows writable-output "
                "monitor could not verify every writable root; enforcement "
                "failed closed]\n"
            )
        return stdout, stderr

    def kill(self) -> None:
        """Terminate every process in this run's Job Object. Safe to
        call multiple times or after the process has already exited —
        ``TerminateJobObject`` on an already-empty/exited job is a
        harmless no-op, matching ``proc.kill()``'s idempotent
        contract on the Linux/macOS side."""
        _kernel32.TerminateJobObject(self._job, 1)
        _kernel32.TerminateProcess(self._proc_info.hProcess, 1)

    def close(self) -> None:
        """Release every OS handle this process holds, and run the
        caller-supplied cleanup (ACL reverts + AppContainer profile
        deletion + job handle close). Idempotent — safe to call from
        both the normal completion path and an exception handler."""
        if self._cleaned_up:
            return
        self._cleaned_up = True
        # Kill the complete tree before any ACL is restored or the profile
        # is deleted. CloseHandle(job) is a second kill-on-close backstop.
        _kernel32.TerminateJobObject(self._job, 1)
        for h in (
            self._proc_info.hProcess,
            self._proc_info.hThread,
            self._stdout_read,
            self._stderr_read,
        ):
            if h:
                _kernel32.CloseHandle(h)
        self._cleanup()


def spawn_in_appcontainer(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    sid: ctypes.c_void_p,
    job_handle: ctypes.c_void_p,
    file_size_monitor: WritableFileSizeMonitor | None = None,
) -> AppContainerProcess:
    """Spawn ``cmd`` inside the AppContainer identified by ``sid``,
    assign it to ``job_handle`` before it runs any code, and return an
    ``AppContainerProcess`` wrapping it.

    Sequencing (each step's rationale noted inline):

      1. Create inheritable-by-child stdin/stdout/stderr pipes, then
         explicitly allowlist only their child ends for inheritance.
      2. Build a ``PROC_THREAD_ATTRIBUTE_LIST`` carrying the
         ``SECURITY_CAPABILITIES`` — this is what actually places the
         new process's primary token inside the AppContainer; there is
         no separate "assign token" call the way there is for the Job
         Object.
      3. ``CreateProcessW`` with ``CREATE_SUSPENDED`` — the process
         exists (so it CAN be assigned to a job) but its main thread
         has not run a single instruction of the target program yet.
      4. ``AssignProcessToJobObject`` while still suspended — assigning
         AFTER resume would leave a window where the process runs
         briefly outside any resource limit at all.
      5. ``ResumeThread`` — only now does the interpreter actually
         start executing.

    Steps 3-4-5 must remain in this order:
    reversing steps 4 and 5 (resume before assign) would mean the
    confinement's resource-limit half doesn't apply for however long
    it takes the parent process to make the ``AssignProcessToJobObject``
    call — a real, if narrow, TOCTOU-shaped gap. Suspended-create
    closes it entirely.
    """
    _require_windows()

    sec_attrs = SECURITY_ATTRIBUTES()
    sec_attrs.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
    sec_attrs.bInheritHandle = True
    sec_attrs.lpSecurityDescriptor = None

    stdin_read, stdin_write = wintypes.HANDLE(), wintypes.HANDLE()
    stdout_read, stdout_write = wintypes.HANDLE(), wintypes.HANDLE()
    stderr_read, stderr_write = wintypes.HANDLE(), wintypes.HANDLE()
    if not _kernel32.CreatePipe(
        ctypes.byref(stdin_read),
        ctypes.byref(stdin_write),
        ctypes.byref(sec_attrs),
        0,
    ):
        raise AppContainerError("CreatePipe(stdin)", _get_last_error())
    if not _kernel32.CreatePipe(
        ctypes.byref(stdout_read),
        ctypes.byref(stdout_write),
        ctypes.byref(sec_attrs),
        0,
    ):
        _kernel32.CloseHandle(stdin_read)
        _kernel32.CloseHandle(stdin_write)
        raise AppContainerError("CreatePipe(stdout)", _get_last_error())
    if not _kernel32.CreatePipe(
        ctypes.byref(stderr_read),
        ctypes.byref(stderr_write),
        ctypes.byref(sec_attrs),
        0,
    ):
        # The stdout pipe pair from the call above already succeeded
        # and would otherwise leak two HANDLEs on every stderr-pipe
        # failure -- nothing past this point ever gets a reference to
        # close them (``AppContainerProcess.close()`` is only reached
        # on the full-success return at the bottom of this function).
        for _h in (stdin_read, stdin_write, stdout_read, stdout_write):
            _kernel32.CloseHandle(_h)
        raise AppContainerError("CreatePipe(stderr)", _get_last_error())
    # The PARENT's read ends must not be inherited by the child, or
    # the pipe never reports EOF (the child would hold its own copy
    # of the read handle open even after exiting).
    for parent_handle, label in (
        (stdin_write, "stdin write"),
        (stdout_read, "stdout read"),
        (stderr_read, "stderr read"),
    ):
        if not _kernel32.SetHandleInformation(parent_handle, HANDLE_FLAG_INHERIT, 0):
            err = _get_last_error()
            for _h in (
                stdin_read,
                stdin_write,
                stdout_read,
                stdout_write,
                stderr_read,
                stderr_write,
            ):
                _kernel32.CloseHandle(_h)
            raise AppContainerError(f"SetHandleInformation({label})", err)

    sec_cap = _build_security_capabilities(sid)

    try:
        attr_size = ctypes.c_size_t(0)
        _kernel32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(attr_size))
        attr_buf = ctypes.create_string_buffer(attr_size.value)
        attr_list = ctypes.cast(attr_buf, ctypes.c_void_p)
        if not _kernel32.InitializeProcThreadAttributeList(
            attr_list,
            2,
            0,
            ctypes.byref(attr_size),
        ):
            raise AppContainerError(
                "InitializeProcThreadAttributeList",
                _get_last_error(),
            )
        try:
            if not _kernel32.UpdateProcThreadAttribute(
                attr_list,
                0,
                PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
                ctypes.byref(sec_cap),
                ctypes.sizeof(sec_cap),
                None,
                None,
            ):
                raise AppContainerError(
                    "UpdateProcThreadAttribute",
                    _get_last_error(),
                )

            # bInheritHandles=True is required for redirected standard
            # streams, but without HANDLE_LIST it would also duplicate every
            # unrelated inheritable handle owned by Sift.  Such a handle can
            # carry access the AppContainer token could never open itself.
            inherited_handles = (wintypes.HANDLE * 3)(
                stdin_read,
                stdout_write,
                stderr_write,
            )
            if not _kernel32.UpdateProcThreadAttribute(
                attr_list,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                ctypes.cast(inherited_handles, ctypes.c_void_p),
                ctypes.sizeof(inherited_handles),
                None,
                None,
            ):
                raise AppContainerError(
                    "UpdateProcThreadAttribute(handle list)",
                    _get_last_error(),
                )

            startup_info = STARTUPINFOEXW()
            startup_info.StartupInfo.cb = ctypes.sizeof(STARTUPINFOEXW)
            startup_info.StartupInfo.dwFlags = 0x00000100  # STARTF_USESTDHANDLES
            startup_info.StartupInfo.hStdOutput = stdout_write
            startup_info.StartupInfo.hStdError = stderr_write
            startup_info.StartupInfo.hStdInput = stdin_read
            startup_info.lpAttributeList = attr_list

            proc_info = PROCESS_INFORMATION()

            env_block = _windows_environment_block(env)
            # ``lpCommandLine`` is documented by Microsoft as a parameter
            # CreateProcessW may WRITE THROUGH (it normalizes/rewrites the
            # string in place) -- "this parameter cannot be a pointer to
            # read-only memory (such as a const variable or a literal
            # string); if it is, the function may cause an access
            # violation." ``_kernel32.CreateProcessW.argtypes`` declares
            # this parameter ``wintypes.LPWSTR`` (mutable) precisely
            # because of this. Passing a plain Python ``str`` here lets
            # ctypes auto-marshal it into a ``c_wchar_p``, which is NOT
            # guaranteed to be a fresh, writable allocation -- CPython is
            # free to reuse/intern the underlying buffer for other strings
            # in the process, so the actual failure mode ranges from a
            # hard crash to silent corruption of an unrelated string
            # elsewhere in the interpreter, and either could occur only
            # intermittently depending on what CPython's string interning
            # happens to do with this particular computed command line on
            # a given run. ``ctypes.create_unicode_buffer`` (already used
            # correctly for ``env_block`` right below) allocates a real,
            # private, writable buffer -- the same fix, applied here too.
            cmdline_buf = ctypes.create_unicode_buffer(subprocess.list2cmdline(cmd))

            flags = (
                EXTENDED_STARTUPINFO_PRESENT
                | CREATE_SUSPENDED
                | CREATE_NO_WINDOW
                | CREATE_UNICODE_ENVIRONMENT
            )

            if not cmd or not cmd[0]:
                raise ValueError("AppContainer command must name an executable")
            resolved_application = (
                cmd[0]
                if Path(cmd[0]).is_absolute()
                else shutil.which(cmd[0], path=env.get("PATH"))
            )
            if not resolved_application:
                raise AppContainerError("CreateProcessW", ERROR_FILE_NOT_FOUND)
            application = str(Path(resolved_application).resolve())
            ok = _kernel32.CreateProcessW(
                application,
                cmdline_buf,
                None,
                None,
                True,
                flags,
                ctypes.create_unicode_buffer(env_block),
                str(cwd),
                ctypes.byref(startup_info),
                ctypes.byref(proc_info),
            )
            if not ok:
                raise AppContainerError("CreateProcessW", _get_last_error())
        finally:
            _kernel32.DeleteProcThreadAttributeList(attr_list)
            # The parent's write-end handles must be closed after
            # CreateProcess duplicates them into the child — otherwise the
            # parent's own copy keeps the pipe alive forever and
            # ``communicate()``'s ReadFile loop never sees EOF, even after
            # the child exits.
            _kernel32.CloseHandle(stdout_write)
            _kernel32.CloseHandle(stderr_write)
            # No stdin is supplied by the product. Closing both parent
            # copies leaves the child's read handle at immediate EOF.
            _kernel32.CloseHandle(stdin_read)
            _kernel32.CloseHandle(stdin_write)

        try:
            if not _kernel32.AssignProcessToJobObject(job_handle, proc_info.hProcess):
                err = _get_last_error()
                _kernel32.TerminateProcess(proc_info.hProcess, 1)
                raise AppContainerError("AssignProcessToJobObject", err)
            if _kernel32.ResumeThread(proc_info.hThread) == 0xFFFFFFFF:
                err = _get_last_error()
                _kernel32.TerminateJobObject(job_handle, 1)
                raise AppContainerError("ResumeThread", err)
        except Exception:
            _kernel32.CloseHandle(proc_info.hProcess)
            _kernel32.CloseHandle(proc_info.hThread)
            raise
    except Exception:
        # Every failure point in the block above used to leak
        # ``stdout_read``/``stderr_read`` -- see the comment above
        # the ``CreatePipe(stderr)`` check earlier in this function
        # for the full audit finding. ``stdout_write``/
        # ``stderr_write`` are already reliably closed by the
        # ``finally:`` above once CreateProcessW has been
        # attempted; closing an already-closed HANDLE here is a
        # harmless no-op (CloseHandle just fails with
        # ERROR_INVALID_HANDLE), so closing all four
        # unconditionally is simpler and just as safe as tracking
        # exactly which ones are still open at each raise site.
        for _h in (
            stdin_read,
            stdin_write,
            stdout_read,
            stderr_read,
            stdout_write,
            stderr_write,
        ):
            _kernel32.CloseHandle(_h)
        raise

    return AppContainerProcess(
        proc_info,
        job_handle,
        stdout_read,
        stderr_read,
        cleanup=lambda: None,
        file_size_monitor=file_size_monitor,
    )


class AppContainerRun:
    """Context manager tying every piece above together for one
    script run: create profile -> grant ACLs -> create job -> spawn
    suspended -> assign to job -> resume, and on exit — success,
    exception, or timeout, unconditionally — revert every ACL grant,
    delete the AppContainer profile, and close the job handle.

    This is the single highest-risk piece of the whole design (see
    ``docs/windows_appcontainer.md``'s native qualification checklist
    item 3): an ACL grant that fails to revert after a crashed script
    leaks filesystem access to whatever runs under the same
    AppContainer SID next. Because every run uses a UUID-suffixed SID
    name (see ``create_appcontainer_profile``), there IS no "next run
    under the same SID" even if a revert somehow failed — a failed
    revert leaks access to a profile nothing will ever reuse, rather
    than to a name a future run's grants would compound onto. Belt and
    suspenders: the ``finally`` block still always attempts every
    revert, in reverse grant order, on top of that.

    Usage mirrors a ``with`` block around a ``subprocess.Popen``:

        with AppContainerRun(cmd, cwd, run_dir, env, extra_read_paths,
                              cpu_seconds, memory_bytes, max_processes,
                              max_file_size_bytes, min_free_disk_bytes) as proc:
            stdout, stderr = proc.communicate(timeout=...)
    """

    def __init__(
        self,
        cmd: list[str],
        cwd: Path,
        run_dir: Path,
        env: dict[str, str],
        extra_read_paths: tuple[str, ...],
        cpu_seconds: int,
        memory_bytes: int,
        max_processes: int,
        max_file_size_bytes: int = 0,
        min_free_disk_bytes: int = 0,
    ) -> None:
        self._cmd = cmd
        self._cwd = cwd
        self._run_dir = run_dir
        self._env = env
        self._extra_read_paths = extra_read_paths
        self._limits = plan_job_limits(cpu_seconds, memory_bytes, max_processes)
        self._max_file_size_bytes = max(0, int(max_file_size_bytes))
        self._min_free_disk_bytes = max(0, int(min_free_disk_bytes))
        self._reverts: list[Callable[[], None]] = []
        self._profile_cleanup: Callable[[], None] | None = None
        self._mutex_release: Callable[[], None] | None = None
        self._job_handle: ctypes.c_void_p | None = None
        self._proc: AppContainerProcess | None = None

    def __enter__(self) -> AppContainerProcess:
        _require_windows()
        # Hold this across profile creation, execution, descendant death,
        # and DACL restoration.  See ``_acquire_acl_mutex`` for the race it
        # closes across concurrent Sift processes.
        self._mutex_release = _acquire_acl_mutex()
        profile_name = f"Sift.Run.{uuid.uuid4().hex}"
        try:
            # Capture the baseline before the child exists.  The workspace
            # grant excludes all of .sift while the second scope re-includes
            # exactly this run directory, matching plan_acl_grants()'s
            # effective writable surface rather than merely its two broad
            # ALLOW entries.
            try:
                file_size_monitor = WritableFileSizeMonitor(
                    scopes=(
                        WritableScope(
                            root=self._cwd,
                            excluded_subtrees=(self._cwd / ".sift",),
                        ),
                        WritableScope(root=self._run_dir),
                    ),
                    limit_bytes=self._max_file_size_bytes,
                    min_free_disk_bytes=self._min_free_disk_bytes,
                )
            except OSError as exc:
                code = (
                    getattr(exc, "winerror", None)
                    or getattr(exc, "errno", None)
                    or ERROR_ACCESS_DENIED
                )
                raise AppContainerError(
                    "initialize writable-output monitor",
                    int(code),
                ) from exc
            sid, profile_cleanup = create_appcontainer_profile(profile_name)
            self._profile_cleanup = profile_cleanup
            for grant in plan_acl_grants(
                self._run_dir,
                self._cwd,
                self._extra_read_paths,
            ):
                self._reverts.append(grant_acl(grant, sid))

            self._job_handle = create_job_object(self._limits)
            self._proc = spawn_in_appcontainer(
                self._cmd,
                self._cwd,
                self._env,
                sid,
                self._job_handle,
                file_size_monitor=file_size_monitor,
            )
            # Fold the profile/ACL/job cleanup into the process
            # object's own ``close()`` so ``run_script``'s existing
            # call sites (which already ``proc.kill()`` /
            # rely on the process object going out of scope) don't
            # need Windows-specific teardown code of their own.
            self._proc._cleanup = self._cleanup
            return self._proc
        except Exception:
            self._cleanup()
            raise

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._proc is not None:
            self._proc.close()
        else:
            self._cleanup()

    def _cleanup(self) -> None:
        errors: list[Exception] = []
        # Closing a kill-on-close Job comes first: no untrusted descendant
        # may remain alive while its ACLs are being removed or restored.
        if self._job_handle:
            try:
                if not _kernel32.CloseHandle(self._job_handle):
                    raise AppContainerError("CloseHandle(job)", _get_last_error())
            except Exception as exc:  # noqa: BLE001 - attempt every cleanup stage
                errors.append(exc)
            self._job_handle = None
        for revert in reversed(self._reverts):
            try:
                revert()
            except Exception as exc:  # noqa: BLE001 - attempt every cleanup stage
                errors.append(exc)
        self._reverts.clear()
        if self._profile_cleanup is not None:
            try:
                self._profile_cleanup()
            except Exception as exc:  # noqa: BLE001 - attempt every cleanup stage
                errors.append(exc)
            self._profile_cleanup = None
        if self._mutex_release is not None:
            try:
                self._mutex_release()
            except Exception as exc:  # noqa: BLE001 - attempt every cleanup stage
                errors.append(exc)
            self._mutex_release = None
        if errors:
            first = errors[0]
            if len(errors) > 1:
                first.add_note(
                    "Additional AppContainer cleanup failures: "
                    + "; ".join(str(error) for error in errors[1:])
                )
            raise first


# ---------------------------------------------------------------------------
# Mandatory live health probe. See the module docstring's closing
# paragraph — this is what stands between "code that was written
# carefully but never run" and "code Sift will actually trust to
# confine a researcher's script."
# ---------------------------------------------------------------------------


def probe_appcontainer_health() -> tuple[bool, str]:
    """Prove the production-shaped Windows sandbox, fail closed.

    Passing requires positive evidence for allowed workspace/run-dir writes,
    enforcement of both the parent-side single-file ceiling and aggregate
    free-space reserve, denial of both Sift's private state and an unrelated
    file, and a raw socket denial with the Windows ``WSAEACCES`` code. A tool
    launch error, timeout, no route, ambiguous output, or any ctypes failure
    is a failed probe; none are treated as evidence of isolation.
    """
    if not _IS_WINDOWS:
        return False, "not running on Windows"

    import tempfile

    def _probe_interpreter() -> tuple[Path, tuple[str, ...]]:
        """Return the exact Python runtime Sift executes and its read roots."""
        roots: tuple[str, ...]
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if not meipass:
                raise RuntimeError("frozen Sift has no PyInstaller data root")
            binary = Path(meipass) / "sift" / "vendor_python" / "python.exe"
            roots = (str(binary.parent),)
        else:
            binary = Path(sys.executable)
            roots = tuple(dict.fromkeys((str(Path(sys.prefix)), str(Path(sys.base_prefix)))))
        if not binary.is_file():
            raise RuntimeError(f"AppContainer probe interpreter missing at {binary}")
        return binary, roots

    def _run_probe(
        interpreter: Path,
        interpreter_roots: tuple[str, ...],
        script: str,
        scratch: Path,
        run_dir: Path,
        max_file_size_bytes: int = 0,
        min_free_disk_bytes: int = 0,
    ) -> tuple[int, str, str]:
        probe_env = {
            key: os.environ[key]
            for key in (
                "PATH",
                "SystemRoot",
                "WINDIR",
                "SystemDrive",
                "COMSPEC",
                "PATHEXT",
                "PROCESSOR_ARCHITECTURE",
                "NUMBER_OF_PROCESSORS",
                "LOCALAPPDATA",
            )
            if key in os.environ
        }
        with AppContainerRun(
            [
                str(interpreter),
                "-I",
                "-B",
                "-c",
                script,
            ],
            scratch,
            run_dir,
            probe_env,
            extra_read_paths=interpreter_roots,
            cpu_seconds=10,
            memory_bytes=256 * 1024**2,
            max_processes=4,
            max_file_size_bytes=max_file_size_bytes,
            min_free_disk_bytes=min_free_disk_bytes,
        ) as process:
            stdout, stderr = process.communicate(timeout=10)
            if process.returncode is None:
                raise RuntimeError("AppContainer probe ended without an exit status")
            return process.returncode, stdout.strip(), stderr.strip()

    try:
        with tempfile.TemporaryDirectory(
            prefix="sift-appcontainer-probe-"
        ) as scratch_str:
            scratch = Path(scratch_str)
            run_dir = scratch / ".sift" / "runs" / "health-probe"
            run_dir.mkdir(parents=True)
            hidden = scratch / ".sift" / "private-canary.txt"
            hidden.write_text("SIFT_PRIVATE_CANARY", encoding="utf-8")
            interpreter, interpreter_roots = _probe_interpreter()

            basic_script = (
                "from pathlib import Path\n"
                f"Path({str(scratch / 'allowed-workspace.txt')!r}).write_text('CWD_OK')\n"
                f"Path({str(run_dir / 'allowed-run.txt')!r}).write_text('RUN_OK')\n"
                "print('BASIC_OK')\n"
            )
            rc, out, err = _run_probe(
                interpreter, interpreter_roots, basic_script, scratch, run_dir
            )
            if rc != 0 or out != "BASIC_OK":
                return False, (
                    "allowed-path probe failed or was ambiguous: "
                    f"exit={rc}, stdout={out!r}, stderr={err!r}"
                )
            if (
                not (scratch / "allowed-workspace.txt").is_file()
                or not (run_dir / "allowed-run.txt").is_file()
            ):
                return False, "allowed-path probe did not create both canary files"

            oversized = scratch / "file-size-canary.bin"
            file_size_script = (
                "import time\n"
                "from pathlib import Path\n"
                f"Path({str(oversized)!r}).write_bytes(b'x' * (1024 * 1024))\n"
                "time.sleep(2)\n"
                "print('FILE_LIMIT_BYPASSED')\n"
            )
            rc, out, err = _run_probe(
                interpreter,
                interpreter_roots,
                file_size_script,
                scratch,
                run_dir,
                max_file_size_bytes=64 * 1024,
            )
            if rc == 0 or "FILE_LIMIT_BYPASSED" in out or "single-file limit" not in err:
                return False, (
                    "single-file ceiling was not positively proven: "
                    f"exit={rc}, stdout={out!r}, stderr={err!r}"
                )

            # Prove the aggregate guard deterministically at the launch
            # boundary. A threshold one byte above the volume's total
            # capacity can never be satisfied, regardless of background disk
            # activity, delayed NTFS allocation accounting, compression, or a
            # dynamically growing VM image. The former live-write probe used
            # a reserve only 4 MiB below current free space and was flaky for
            # exactly those reasons. Unit tests separately prove that a
            # reserve crossed after launch terminates the process; this native
            # probe proves AppContainerRun wires the guard fail-closed before
            # any untrusted child starts.
            impossible_reserve = int(shutil.disk_usage(scratch).total) + 1
            try:
                _run_probe(
                    interpreter,
                    interpreter_roots,
                    "print('DISK_RESERVE_BYPASSED')\n",
                    scratch,
                    run_dir,
                    min_free_disk_bytes=impossible_reserve,
                )
            except AppContainerError as exc:
                if (
                    exc.operation != "initialize writable-output monitor"
                    or exc.code != errno.ENOSPC
                ):
                    return False, (
                        "aggregate disk-reserve guard failed ambiguously: "
                        f"{exc}"
                    )
            else:
                return False, (
                    "aggregate disk-reserve guard did not refuse an "
                    "impossible safety reserve"
                )

            hidden_script = (
                "from pathlib import Path\n"
                "try:\n"
                f"    Path({str(hidden)!r}).read_text()\n"
                "except OSError:\n"
                "    print('HIDDEN_DENIED')\n"
                "    raise SystemExit(17)\n"
                "print('HIDDEN_READ')\n"
                "raise SystemExit(41)\n"
            )
            rc, out, err = _run_probe(
                interpreter,
                interpreter_roots,
                hidden_script,
                scratch,
                run_dir,
            )
            if rc != 17 or out != "HIDDEN_DENIED":
                return False, (
                    "Sift private-state denial was not positively proven: "
                    f"exit={rc}, stdout={out!r}, stderr={err!r}"
                )

            outside_dir = Path(tempfile.mkdtemp(prefix="sift-appcontainer-canary-"))
            canary = outside_dir / "outside-canary.txt"
            canary.write_text("OUTSIDE_PRIVATE_CANARY", encoding="utf-8")
            try:
                outside_script = (
                    "from pathlib import Path\n"
                    "try:\n"
                    f"    Path({str(canary)!r}).read_text()\n"
                    "except OSError:\n"
                    "    print('OUTSIDE_DENIED')\n"
                    "    raise SystemExit(18)\n"
                    "print('OUTSIDE_READ')\n"
                    "raise SystemExit(42)\n"
                )
                rc, out, err = _run_probe(
                    interpreter,
                    interpreter_roots,
                    outside_script,
                    scratch,
                    run_dir,
                )
                if rc != 18 or out != "OUTSIDE_DENIED":
                    return False, (
                        "outside-file denial was not positively proven: "
                        f"exit={rc}, stdout={out!r}, stderr={err!r}"
                    )
            finally:
                try:
                    canary.unlink()
                    outside_dir.rmdir()
                except OSError:
                    pass

            network_script = (
                "import socket\n"
                "try:\n"
                "    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
                "    sock.connect(('198.51.100.1', 9))\n"
                "except OSError as exc:\n"
                "    code = getattr(exc, 'winerror', None) or exc.errno\n"
                "    print(f'SOCKET_ERROR={code}')\n"
                "    raise SystemExit(19)\n"
                "print('CONNECTED')\n"
                "raise SystemExit(43)\n"
            )
            rc, out, err = _run_probe(
                interpreter,
                interpreter_roots,
                network_script,
                scratch,
                run_dir,
            )
            if rc != 19 or out != "SOCKET_ERROR=10013":
                return False, (
                    "network isolation was not positively proven as WSAEACCES: "
                    f"exit={rc}, stdout={out!r}, stderr={err!r}"
                )
    except Exception as e:  # noqa: BLE001 — the probe itself must never crash the caller
        return False, f"health probe raised an unexpected error: {e}"

    return True, ""
