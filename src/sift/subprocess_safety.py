"""Small, bounded subprocess capture for trusted host-side probes.

This is intentionally narrower than the analysis executor.  It is for short
version, capability, conversion, and qualification commands that need their
stdout/stderr but must not let a broken external executable retain unbounded
bytes in the Sift parent process.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_PROBE_STREAM_LIMIT = 2 * 1024 * 1024


class SubprocessOutputLimitExceeded(subprocess.SubprocessError):
    """A host-side probe wrote more diagnostic output than Sift permits."""

    def __init__(
        self,
        cmd: Sequence[str],
        *,
        stdout: str | bytes,
        stderr: str | bytes,
    ) -> None:
        super().__init__("subprocess diagnostic output exceeded Sift's bounded limit")
        self.cmd = tuple(cmd)
        self.stdout = stdout
        self.stderr = stderr


def _terminate_windows_process_tree(proc: subprocess.Popen[bytes]) -> None:
    """Best-effort identity-aware stop of a Windows probe and its children.

    ``Popen.kill()`` only terminates the direct process on Windows.  Several
    otherwise trusted host-side probes (package managers, converters, and
    scientific runtimes) can start helper processes, so a timeout or output
    flood must not leave those helpers running after Sift has returned.

    psutil is a required Sift dependency and protects destructive operations
    against PID reuse.  Kill descendants before the root so the root cannot
    disappear and make its child list unrecoverable.  The direct ``Popen``
    handle remains the final fallback when enumeration is unavailable.
    """
    try:
        import psutil
    except ImportError:
        # Defensive: psutil is a required dependency, but a partially damaged
        # install must still terminate the direct process below.
        pass
    else:
        try:
            root = psutil.Process(proc.pid)
            try:
                descendants = root.children(recursive=True)
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                descendants = []
            for child in reversed(descendants):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                    pass
            try:
                root.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                pass
            try:
                psutil.wait_procs([*descendants, root], timeout=1)
            except (psutil.Error, OSError):
                pass
        except (psutil.Error, OSError, TypeError, ValueError):
            pass
    try:
        proc.kill()
    except (OSError, ProcessLookupError):
        pass


def _terminate_probe(proc: subprocess.Popen[bytes]) -> None:
    """Best-effort whole-tree stop on every supported platform."""
    if proc.poll() is not None:
        return
    if os.name == "nt":
        _terminate_windows_process_tree(proc)
        return
    else:
        try:
            group = os.getpgid(proc.pid)
            if group != os.getpgrp():
                os.killpg(group, signal.SIGKILL)
                return
        except (AttributeError, OSError, ProcessLookupError):
            pass
    try:
        proc.kill()
    except (OSError, ProcessLookupError):
        pass


def run_bounded_capture(
    cmd: Sequence[str],
    *,
    timeout: float,
    cwd: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    check: bool = False,
    text: bool = True,
    encoding: str = "utf-8",
    errors: str = "replace",
    stdout_limit: int = DEFAULT_PROBE_STREAM_LIMIT,
    stderr_limit: int = DEFAULT_PROBE_STREAM_LIMIT,
) -> subprocess.CompletedProcess[Any]:
    """Run one short probe while continuously capping both output streams.

    Readers drain concurrently so a chatty stderr cannot deadlock a command
    whose stdout is quiet.  Crossing either independent byte ceiling stops the
    process immediately; only the bounded prefix is retained.
    """
    argv = [str(value) for value in cmd]
    if not argv:
        raise ValueError("cmd must contain at least one argument")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if stdout_limit < 1 or stderr_limit < 1:
        raise ValueError("stream limits must be positive")

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd) if cwd is not None else None,
        env=dict(env) if env is not None else None,
        start_new_session=(os.name != "nt"),
    )
    if proc.stdout is None or proc.stderr is None:
        # PIPE was requested for both streams, so this indicates a broken or
        # substituted process implementation. Do not leave it running when
        # the bounded drain contract cannot be enforced.
        try:
            _terminate_probe(proc)
        finally:
            proc.wait()
        raise RuntimeError("subprocess did not provide the requested output pipes")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {"stdout": stdout_limit, "stderr": stderr_limit}
    overflow = threading.Event()
    lock = threading.Lock()

    def drain(label: str, stream: Any) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    return
                with lock:
                    remaining = limits[label] - len(buffers[label])
                    if remaining > 0:
                        buffers[label].extend(chunk[:remaining])
                    if len(chunk) > remaining:
                        overflow.set()
                if overflow.is_set():
                    _terminate_probe(proc)
                    return
        except (OSError, ValueError):
            return

    readers = [
        threading.Thread(target=drain, args=("stdout", proc.stdout), daemon=True),
        threading.Thread(target=drain, args=("stderr", proc.stderr), daemon=True),
    ]
    for reader in readers:
        reader.start()

    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_probe(proc)
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
    finally:
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except OSError:
                pass
        for reader in readers:
            reader.join(timeout=1)

    raw_stdout = bytes(buffers["stdout"])
    raw_stderr = bytes(buffers["stderr"])
    def decode_text(raw: bytes) -> str:
        # We drain binary streams to enforce byte ceilings. Restore the
        # universal-newline semantics callers receive from subprocess text
        # mode so Windows CRLF does not leak into parsing or exact probes.
        decoded = raw.decode(encoding, errors)
        return decoded.replace("\r\n", "\n").replace("\r", "\n")

    stdout: str | bytes = decode_text(raw_stdout) if text else raw_stdout
    stderr: str | bytes = decode_text(raw_stderr) if text else raw_stderr
    if timed_out:
        raise subprocess.TimeoutExpired(
            argv, timeout, output=stdout, stderr=stderr,
        )
    if overflow.is_set():
        raise SubprocessOutputLimitExceeded(
            argv, stdout=stdout, stderr=stderr,
        )
    completed = subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
    if check:
        completed.check_returncode()
    return completed


__all__ = [
    "DEFAULT_PROBE_STREAM_LIMIT",
    "SubprocessOutputLimitExceeded",
    "run_bounded_capture",
]
