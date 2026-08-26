"""Identity-safe POSIX subprocess-tree tracking and termination.

Process groups are useful, but they are not a process-tree boundary: code in a
generated script can call ``setsid()`` or ``setpgid()`` and escape the group
created by ``start_new_session=True``.  This module records descendants while
the launched interpreter is alive and verifies each process' birth identity
again immediately before signalling it.  The identity check is important -- a
bare PID retained after exit can refer to an unrelated process after PID reuse.

Linux exposes a kernel boot-tick start value in ``/proc/<pid>/stat``.  macOS
does not expose procfs, so the portable fallback uses ``ps``'s full start time.
The run's private environment marker is also consulted at cleanup time.  That
recovers a long-lived child which detached and was reparented between ancestry
snapshots without broad, unsafe name- or command-based killing.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_TRACKER_ATTRIBUTE = "_sift_posix_descendant_tracker"
_POLL_SECONDS = 0.05
_UNVERIFIED_START = "unverified-initial-snapshot"


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    """A PID paired with an immutable process-birth marker."""

    pid: int
    ppid: int
    started: str


class ProcessTreeSnapshotUnavailable(RuntimeError):
    """The live root identity could not be verified for accounting."""


def _linux_snapshot() -> dict[int, ProcessIdentity]:
    result: dict[int, ProcessIdentity] = {}
    proc_root = Path("/proc")
    try:
        entries = proc_root.iterdir()
    except OSError:
        return result
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            raw = (entry / "stat").read_text(encoding="utf-8")
            # The comm field is parenthesized and may itself contain spaces or
            # parentheses.  Everything after the final ')' has fixed fields:
            # state (3), ppid (4), ... starttime (22).
            tail = raw[raw.rfind(")") + 2 :].split()
            pid = int(entry.name)
            ppid = int(tail[1])
            started = tail[19]
        except (OSError, ValueError, IndexError):
            # A process commonly disappears midway through a procfs read.
            continue
        result[pid] = ProcessIdentity(pid, ppid, started)
    return result


def _ps_snapshot(
    *, include_environment: bool = False
) -> tuple[dict[int, ProcessIdentity], str]:
    # ``lstart`` is available in both BSD/macOS ps and procps.  It is kept as
    # text rather than parsed through locale-sensitive datetime machinery.
    args = ["ps"]
    if include_environment:
        args.append("eww")
    args.extend(["-axo", "pid=,ppid=,lstart=,command="])
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=3,
        )
    except Exception:  # noqa: BLE001 - ps is only a portability fallback
        return {}, ""
    if completed.returncode != 0:
        return {}, ""

    result: dict[int, ProcessIdentity] = {}
    for line in completed.stdout.splitlines():
        # pid, ppid, and lstart's five whitespace-delimited fields; the
        # command/environment remainder is deliberately ignored for identity.
        fields = line.split(None, 7)
        if len(fields) < 7:
            continue
        try:
            pid = int(fields[0])
            ppid = int(fields[1])
        except ValueError:
            continue
        started = " ".join(fields[2:7])
        result[pid] = ProcessIdentity(pid, ppid, started)
    return result, completed.stdout


def process_snapshot() -> dict[int, ProcessIdentity]:
    """Return a best-effort snapshot keyed by PID."""

    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        return _linux_snapshot()
    if sys.platform == "darwin":
        snapshot = _darwin_snapshot()
        if snapshot:
            return snapshot
    return _ps_snapshot()[0]


def _darwin_snapshot() -> dict[int, ProcessIdentity]:
    """Read PID, parent, and microsecond birth time through libproc.

    Unlike spawning ``ps``, libproc works in Sift's restricted host context
    and supplies microseconds rather than a locale-formatted second.  The
    latter matters for the PID-reuse guarantee.
    """

    try:
        import ctypes

        class _ProcBSDInfo(ctypes.Structure):
            _fields_ = [
                ("pbi_flags", ctypes.c_uint32),
                ("pbi_status", ctypes.c_uint32),
                ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32),
                ("pbi_ppid", ctypes.c_uint32),
                ("pbi_uid", ctypes.c_uint32),
                ("pbi_gid", ctypes.c_uint32),
                ("pbi_ruid", ctypes.c_uint32),
                ("pbi_rgid", ctypes.c_uint32),
                ("pbi_svuid", ctypes.c_uint32),
                ("pbi_svgid", ctypes.c_uint32),
                ("rfu_1", ctypes.c_uint32),
                ("pbi_comm", ctypes.c_char * 16),
                ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32),
                ("pbi_pgid", ctypes.c_uint32),
                ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32),
                ("e_tpgid", ctypes.c_uint32),
                ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64),
                ("pbi_start_tvusec", ctypes.c_uint64),
            ]

        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        libproc.proc_listallpids.argtypes = [ctypes.c_void_p, ctypes.c_int]
        libproc.proc_listallpids.restype = ctypes.c_int
        libproc.proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        libproc.proc_pidinfo.restype = ctypes.c_int
        count = libproc.proc_listallpids(None, 0)
        if count <= 0:
            return {}
        # Process count can grow between sizing and fill calls.
        pids = (ctypes.c_int * (count + 128))()
        filled = libproc.proc_listallpids(pids, ctypes.sizeof(pids))
        result: dict[int, ProcessIdentity] = {}
        for pid in pids[: max(0, filled)]:
            if pid <= 0:
                continue
            info = _ProcBSDInfo()
            size = libproc.proc_pidinfo(
                pid,
                3,
                0,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if size != ctypes.sizeof(info) or info.pbi_pid != pid:
                continue
            started = f"{info.pbi_start_tvsec}:{info.pbi_start_tvusec}"
            result[pid] = ProcessIdentity(pid, int(info.pbi_ppid), started)
        return result
    except Exception:  # noqa: BLE001 - fall through to ps portability path
        return {}


def _darwin_process_environment(pid: int) -> set[bytes]:
    """Return one same-user process environment through KERN_PROCARGS2."""

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2
        size = ctypes.c_size_t()
        if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
            return set()
        if not 4 < size.value <= 16 * 1024 * 1024:
            return set()
        buffer = ctypes.create_string_buffer(size.value)
        if (
            libc.sysctl(
                mib,
                3,
                buffer,
                ctypes.byref(size),
                None,
                0,
            )
            != 0
        ):
            return set()
        raw = buffer.raw[: size.value]
        argc = int.from_bytes(raw[:4], byteorder=sys.byteorder, signed=True)
        if argc < 0 or argc > 1_000_000:
            return set()
        pos = raw.find(b"\0", 4)
        if pos < 0:
            return set()
        while pos < len(raw) and raw[pos] == 0:
            pos += 1
        for _ in range(argc):
            end = raw.find(b"\0", pos)
            if end < 0:
                return set()
            pos = end + 1
            while pos < len(raw) and raw[pos] == 0:
                pos += 1
        return {part for part in raw[pos:].split(b"\0") if b"=" in part}
    except Exception:  # noqa: BLE001 - inaccessible/exited processes are normal
        return set()


def _linux_marker_pids(name: str, value: str) -> set[int]:
    needle = f"{name}={value}".encode("utf-8")
    found: set[int] = set()
    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return found
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        try:
            environ = (entry / "environ").read_bytes()
        except OSError:
            continue
        if needle in environ.split(b"\0"):
            found.add(int(entry.name))
    return found


def _marker_pids(name: str, value: str) -> set[int]:
    """Find processes inheriting an exact private run marker."""

    if not name or not value:
        return set()
    if sys.platform.startswith("linux") and Path("/proc").is_dir():
        return _linux_marker_pids(name, value)
    if sys.platform == "darwin":
        needle = f"{name}={value}".encode("utf-8")
        return {
            pid
            for pid in process_snapshot()
            if needle in _darwin_process_environment(pid)
        }
    _snapshot, output = _ps_snapshot(include_environment=True)
    needle = f"{name}={value}"
    found: set[int] = set()
    for line in output.splitlines():
        if needle not in line:
            continue
        fields = line.split(None, 1)
        try:
            found.add(int(fields[0]))
        except (ValueError, IndexError):
            continue
    return found


class PosixDescendantTracker:
    """Track descendants of one process until an idempotent cleanup."""

    def __init__(
        self,
        root: ProcessIdentity,
        *,
        marker: tuple[str, str] | None = None,
    ) -> None:
        self.root = root
        self.marker = marker
        self._known: dict[int, ProcessIdentity] = {root.pid: root}
        self._known_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._stop = threading.Event()
        self._cleaned = False
        self._thread = threading.Thread(
            target=self._monitor,
            name=f"sift-process-tree-{root.pid}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _collect(
        self,
        snapshot: Mapping[int, ProcessIdentity],
        *,
        include_marker: bool = False,
    ) -> None:
        with self._known_lock:
            known = dict(self._known)

        # Only identities still matching the snapshot are ancestry seeds.  A
        # reused PID must never make its new children look like ours.
        owned = {
            pid for pid, identity in known.items() if snapshot.get(pid) == identity
        }
        if include_marker and self.marker is not None:
            name, value = self.marker
            for pid in _marker_pids(name, value):
                if pid in snapshot:
                    owned.add(pid)

        changed = True
        while changed:
            changed = False
            for identity in snapshot.values():
                if identity.pid not in owned and identity.ppid in owned:
                    owned.add(identity.pid)
                    changed = True

        with self._known_lock:
            for pid in owned:
                self._known[pid] = snapshot[pid]

    def _monitor(self) -> None:
        while not self._stop.wait(_POLL_SECONDS):
            self._collect(process_snapshot())

    def live_identities(
        self,
        *,
        include_marker: bool = True,
    ) -> tuple[ProcessIdentity, ...]:
        """Return only owned identities still matching the live table.

        Resource monitors use this same ownership set as cleanup.  In
        particular, marker recovery means an already-reparented setsid child
        cannot disappear from memory/process/CPU accounting merely because it
        escaped the root's current ancestry.
        """

        snapshot = process_snapshot()
        if snapshot.get(self.root.pid) != self.root:
            # Initial host snapshot failure is represented by a synthetic
            # identity.  Promote it only when the exact private marker proves
            # the currently-live PID is this run's process, never merely
            # because the numeric PID happens to exist.
            current_root = snapshot.get(self.root.pid)
            if (
                self.root.started == _UNVERIFIED_START
                and current_root is not None
                and self.marker is not None
                and self.root.pid in _marker_pids(*self.marker)
            ):
                self.root = current_root
                with self._known_lock:
                    self._known[current_root.pid] = current_root
            else:
                # Resource accounting is invoked only after proc.poll()
                # reported the root live.  An empty/partial snapshot must
                # fail closed, not masquerade as zero resource use.
                raise ProcessTreeSnapshotUnavailable(
                    f"cannot verify process-tree root {self.root.pid}"
                )
        self._collect(snapshot, include_marker=include_marker)
        with self._known_lock:
            known = tuple(self._known.values())
        return tuple(
            identity for identity in known if snapshot.get(identity.pid) == identity
        )

    def terminate(self) -> None:
        """SIGKILL every still-matching recorded process, once.

        The operation is serialized because executor timeout recovery and the
        runner's Stop path may arrive concurrently for the same process.
        """

        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
            self._stop.set()
            if self._thread is not threading.current_thread():
                self._thread.join(timeout=1)

            # Multiple short passes close the window where an observed child
            # forks while teardown begins.  Marker discovery also recovers a
            # detached/reparented child which ancestry polling missed.
            for pass_number in range(3):
                snapshot = process_snapshot()
                self._collect(snapshot, include_marker=True)
                with self._known_lock:
                    targets = list(self._known.values())

                # Retain process-group efficiency for the ordinary case, but
                # only while the original root identity still matches.  This
                # prevents a reused root PID from directing SIGKILL at an
                # unrelated process group.
                if snapshot.get(self.root.pid) == self.root:
                    try:
                        pgid = os.getpgid(self.root.pid)
                        # Defensive against utility misuse: only executor
                        # launches are guaranteed start_new_session=True.  A
                        # tracker attached to a same-group Popen must never
                        # SIGKILL Sift's own process group.
                        if pgid > 0 and pgid != os.getpgrp():
                            os.killpg(pgid, signal.SIGKILL)
                    except (AttributeError, OSError):
                        pass

                for identity in targets:
                    if identity.pid == os.getpid():
                        continue
                    if snapshot.get(identity.pid) != identity:
                        continue
                    try:
                        os.kill(identity.pid, signal.SIGKILL)
                    except OSError:
                        pass
                if pass_number < 2:
                    time.sleep(_POLL_SECONDS)


def attach_posix_descendant_tracker(
    proc: Any,
    *,
    marker: tuple[str, str] | None = None,
) -> PosixDescendantTracker | None:
    """Attach a tracker to a newly spawned POSIX process.

    If the process vanished before its birth identity could be captured, no
    tracker is attached; callers retain their direct-process fallback.
    """

    if sys.platform.startswith("win"):
        return None
    try:
        pid = int(proc.pid)
    except (AttributeError, TypeError, ValueError):
        return None
    root = process_snapshot().get(pid)
    if root is None:
        # A very short script can daemonize a marker-inheriting child and exit
        # before the host's first identity snapshot; a transient snapshot
        # failure can also occur while it is live.  Preserve marker cleanup in
        # both cases.  Popen-shaped fakes without a lifecycle probe retain the
        # caller's direct-kill fallback.
        poll = getattr(proc, "poll", None)
        if marker is None or not callable(poll):
            return None
        try:
            poll()
        except Exception:  # noqa: BLE001
            return None
        root = ProcessIdentity(pid, -1, _UNVERIFIED_START)
    tracker = PosixDescendantTracker(root, marker=marker)
    try:
        setattr(proc, _TRACKER_ATTRIBUTE, tracker)
    except Exception:  # noqa: BLE001 - third-party Popen-shaped wrappers
        return None
    tracker.start()
    return tracker


def terminate_tracked_process_tree(proc: Any) -> bool:
    """Terminate an attached tree; return whether a tracker was present."""

    tracker = getattr(proc, _TRACKER_ATTRIBUTE, None)
    if not isinstance(tracker, PosixDescendantTracker):
        return False
    tracker.terminate()
    return True


def tracked_process_identities(
    proc: Any,
) -> tuple[ProcessIdentity, ...] | None:
    """Return identity-verified live members, or None without a tracker."""

    tracker = getattr(proc, _TRACKER_ATTRIBUTE, None)
    if not isinstance(tracker, PosixDescendantTracker):
        return None
    return tracker.live_identities(include_marker=True)
