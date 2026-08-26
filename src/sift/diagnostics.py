"""Private-by-default, bounded local diagnostics for the desktop process.

Diagnostic output is useful when a local runtime or provider client fails,
but an unbounded plaintext transcript is a poor default for confidential-data
work.  This module centralizes the invariant the platform launchers cannot
reliably enforce themselves:

* credential-shaped values and absolute paths are redacted before writing;
* files are user-private and retained for a short, bounded period;
* both the lifetime and byte ceiling can only be tightened by enterprise
  policy; and
* diagnostics can be disabled completely without an external service or IdP.
"""

from __future__ import annotations

import io
import os
import platform
import re
import stat
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TextIO

from sift.enterprise_policy import (
    EnterprisePolicy,
    apply_diagnostic_bytes_ceiling,
    apply_diagnostic_retention_ceiling,
    load_enterprise_policy,
    local_diagnostics_allowed,
)
from sift.error_summary import scrub_raw_output
from sift.secure_file import open_regular_no_follow


DEFAULT_RETENTION_DAYS = 7
DEFAULT_TOTAL_BYTES = 8 * 1024 * 1024
DEFAULT_FILE_BYTES = 2 * 1024 * 1024
MAX_WRITE_CHARS = 64 * 1024

_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|"
    r"client[_-]?secret|password|passwd|pwd|authorization|cookie)\b"
    r"\s*[:=]\s*[\"']?)[^\"'\s,;&}]+"
)
_BEARER_RE = re.compile(r"(?i)(\bbearer\s+)[^\s,;]+")
_AUTHORIZATION_HEADER_RE = re.compile(
    r"(?i)(\b(?:proxy-authorization|authorization)\s*[:=]\s*)[^\r\n]+"
)
_COOKIE_HEADER_RE = re.compile(
    r"(?i)(\b(?:set-cookie|cookie)\s*[:=]\s*)[^\r\n]+"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?"
    r"-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)
_LOG_WRITE_LOCK = threading.RLock()


def _open_log_fd(path: Path, flags: int, mode: int = 0o600) -> int:
    """Open a diagnostic file without following a pre-created symlink."""
    return open_regular_no_follow(Path(path), flags, mode)


def _ensure_private_log_file(path: Path) -> None:
    fd = _open_log_fd(
        path,
        os.O_WRONLY | os.O_APPEND | os.O_CREAT,
        0o600,
    )
    os.close(fd)
    try:
        path.chmod(0o600, follow_symlinks=False)
    except (NotImplementedError, OSError, TypeError):
        pass


def diagnostic_log_directory() -> Path:
    """Return the current user's conventional per-platform log directory."""
    system = platform.system()
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA")
        return (Path(base) if base else Path.home() / "AppData" / "Local") / "Sift" / "Logs"
    if system == "Darwin":
        return Path.home() / "Library" / "Logs" / "Sift"
    state_home = os.environ.get("XDG_STATE_HOME")
    state_base = (
        Path(state_home) if state_home else Path.home() / ".local" / "state"
    )
    return state_base / "sift" / "log"


def redact_diagnostic_text(text: str) -> str:
    """Redact credential patterns and machine-local paths from log text."""
    value = str(text)
    if len(value) > MAX_WRITE_CHARS:
        value = value[:MAX_WRITE_CHARS] + "\n…[diagnostic write truncated]"
    value = _PRIVATE_KEY_RE.sub("[redacted-private-key]", value)
    value = _AUTHORIZATION_HEADER_RE.sub(r"\1[redacted-credential]", value)
    value = _COOKIE_HEADER_RE.sub(r"\1[redacted-credential]", value)
    value = _BEARER_RE.sub(r"\1[redacted-credential]", value)
    value = _SECRET_ASSIGNMENT_RE.sub(r"\1[redacted-credential]", value)
    return scrub_raw_output(value, cap_bytes=MAX_WRITE_CHARS + 64)


def _log_files(log_dir: Path) -> list[Path]:
    try:
        candidates = list(log_dir.glob("sift-*.log"))
    except OSError:
        return []
    files: list[Path] = []
    for path in candidates:
        try:
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path)
        except OSError:
            continue
    return files


def _trim_to_newest_bytes(path: Path, limit: int) -> None:
    """Retain at most ``limit`` newest bytes without following symlinks."""
    if limit < 1:
        return
    try:
        read_fd = _open_log_fd(path, os.O_RDONLY)
        try:
            opened = os.fstat(read_fd)
            size = opened.st_size
            if size <= limit:
                return
            os.lseek(read_fd, -limit, os.SEEK_END)
            tail = os.read(read_fd, limit)
        finally:
            os.close(read_fd)
        # Starting at the next line avoids a misleading partial first record.
        newline = tail.find(b"\n")
        if newline != -1 and newline + 1 < len(tail):
            tail = tail[newline + 1:]
        temp_fd, temp_name = tempfile.mkstemp(
            prefix=".sift-log-trim-", dir=path.parent,
        )
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(temp_fd, 0o600)
            os.write(temp_fd, tail)
            os.fsync(temp_fd)
            os.close(temp_fd)
            temp_fd = -1
            # Replacing the directory entry replaces a symlink itself rather
            # than following it if an external process races this operation.
            os.replace(temp_name, path)
        finally:
            if temp_fd >= 0:
                os.close(temp_fd)
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
    except OSError:
        return


def prune_diagnostic_logs(
    log_dir: Path,
    *,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    total_bytes: int = DEFAULT_TOTAL_BYTES,
    now: float | None = None,
) -> None:
    """Delete expired/oldest Sift logs until both configured bounds hold."""
    retention_days = max(1, int(retention_days))
    total_bytes = max(1, int(total_bytes))
    current_time = time.time() if now is None else float(now)
    cutoff = current_time - retention_days * 86_400
    entries: list[tuple[float, int, Path]] = []
    for path in _log_files(log_dir):
        try:
            stat = path.stat()
            if stat.st_mtime < cutoff:
                path.unlink()
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
        except OSError:
            continue

    entries.sort(key=lambda item: (item[0], item[2].name))
    size = sum(item[1] for item in entries)
    while len(entries) > 1 and size > total_bytes:
        _, entry_size, path = entries.pop(0)
        try:
            path.unlink()
            size -= entry_size
        except OSError:
            pass
    if entries and size > total_bytes:
        _trim_to_newest_bytes(entries[-1][2], total_bytes)


class RedactingLogStream(io.TextIOBase):
    """A small text stream that redacts every write and caps its file."""

    def __init__(
        self, path: Path, *, max_bytes: int, mirror: TextIO | None = None,
    ) -> None:
        self.path = Path(path)
        self.max_bytes = max(1, int(max_bytes))
        self.mirror = mirror
        self._pending = ""
        self._private_key_mode = False

    encoding = "utf-8"  # pragma: no cover - interpreter integration

    def writable(self) -> bool:
        return True

    def _emit(self, value: str) -> None:
        if "-----BEGIN " in value and "PRIVATE KEY-----" in value:
            self._private_key_mode = not (
                "-----END " in value and "PRIVATE KEY-----" in value
            )
            safe = "[redacted-private-key]\n"
        elif self._private_key_mode:
            if "-----END " in value and "PRIVATE KEY-----" in value:
                self._private_key_mode = False
            safe = ""
        else:
            safe = redact_diagnostic_text(value)
        if not safe:
            return
        if self.mirror is not None:
            try:
                self.mirror.write(safe)
            except (OSError, ValueError):
                pass
        try:
            with _LOG_WRITE_LOCK:
                fd = _open_log_fd(
                    self.path,
                    os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                    0o600,
                )
                try:
                    os.write(fd, safe.encode("utf-8"))
                finally:
                    os.close(fd)
                _trim_to_newest_bytes(self.path, self.max_bytes)
        except OSError:
            pass

    def write(self, value: str) -> int:
        raw = str(value)
        self._pending += raw
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._emit(line + "\n")
        if len(self._pending) > MAX_WRITE_CHARS:
            self._emit("[oversized diagnostic line redacted]\n")
            self._pending = ""
        return len(raw)

    def flush(self) -> None:
        if self.mirror is not None:
            try:
                self.mirror.flush()
            except (OSError, ValueError):
                pass

    def close(self) -> None:
        if not self.closed and self._pending:
            self._emit(self._pending)
            self._pending = ""
        self.flush()
        super().close()


def configure_diagnostic_logging(
    *,
    log_dir: Path | None = None,
    enterprise: EnterprisePolicy | None = None,
) -> Path | None:
    """Install bounded redacting stdout/stderr streams; never raises.

    Passing ``enterprise`` is useful for callers/tests that already loaded the
    policy. Otherwise it is loaded here. A malformed deployed policy resolves
    to the fail-closed policy and therefore disables local diagnostics.
    """
    try:
        if (
            isinstance(sys.stdout, RedactingLogStream)
            and isinstance(sys.stderr, RedactingLogStream)
            and sys.stdout.path == sys.stderr.path
        ):
            return sys.stdout.path
        policy = enterprise if enterprise is not None else load_enterprise_policy()
        if not local_diagnostics_allowed(policy):
            return None
        retention = apply_diagnostic_retention_ceiling(
            DEFAULT_RETENTION_DAYS, policy,
        )
        total_bytes = apply_diagnostic_bytes_ceiling(DEFAULT_TOTAL_BYTES, policy)
        file_bytes = min(DEFAULT_FILE_BYTES, total_bytes)
        directory = Path(log_dir) if log_dir is not None else diagnostic_log_directory()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            directory.chmod(0o700)
        except OSError:
            pass
        prune_diagnostic_logs(
            directory,
            retention_days=retention,
            total_bytes=total_bytes,
        )
        path = directory / time.strftime("sift-%Y-%m-%d.log")
        _ensure_private_log_file(path)
        _trim_to_newest_bytes(path, file_bytes)
        current_stdout = sys.stdout
        current_stderr = sys.stderr
        sys.stdout = RedactingLogStream(
            path, max_bytes=file_bytes, mirror=current_stdout,
        )
        sys.stderr = RedactingLogStream(
            path, max_bytes=file_bytes, mirror=current_stderr,
        )
        return path
    except Exception:  # noqa: BLE001 — diagnostics must never stop the app
        return None


__all__ = [
    "DEFAULT_FILE_BYTES",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_TOTAL_BYTES",
    "RedactingLogStream",
    "configure_diagnostic_logging",
    "diagnostic_log_directory",
    "prune_diagnostic_logs",
    "redact_diagnostic_text",
]
