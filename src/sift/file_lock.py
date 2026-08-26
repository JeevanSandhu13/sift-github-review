"""Small cross-platform advisory file lock used by local metadata stores."""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from sift.secure_file import open_regular_no_follow

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.RLock] = {}


def _thread_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.RLock())


def _acquire_os_lock(handle: BinaryIO) -> None:
    if os.name == "nt":  # pragma: win32 cover - exercised on Windows CI
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        locking = getattr(msvcrt, "locking", None)
        lock_mode = getattr(msvcrt, "LK_LOCK", None)
        if not callable(locking) or not isinstance(lock_mode, int):
            raise RuntimeError("Windows advisory locking is unavailable")
        locking(handle.fileno(), lock_mode, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release_os_lock(handle: BinaryIO) -> None:
    if os.name == "nt":  # pragma: win32 cover - exercised on Windows CI
        import msvcrt

        handle.seek(0)
        locking = getattr(msvcrt, "locking", None)
        unlock_mode = getattr(msvcrt, "LK_UNLCK", None)
        if not callable(locking) or not isinstance(unlock_mode, int):
            raise RuntimeError("Windows advisory unlocking is unavailable")
        locking(handle.fileno(), unlock_mode, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_file_lock(path: Path) -> Iterator[None]:
    """Serialize threads and processes that mutate one metadata resource."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mutex = _thread_lock(path)
    with mutex:
        descriptor = open_regular_no_follow(
            path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600,
        )
        with os.fdopen(descriptor, "a+b") as handle:
            _acquire_os_lock(handle)
            try:
                yield
            finally:
                _release_os_lock(handle)


__all__ = ["exclusive_file_lock"]
