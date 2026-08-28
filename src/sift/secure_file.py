"""Race-resistant no-follow file opens on POSIX and Windows."""

from __future__ import annotations

import ctypes
import hashlib
import os
import stat
from pathlib import Path


def _windows_creation_disposition(flags: int) -> int:
    """Map Python creation flags to Win32 without truncating before validation."""
    if flags & os.O_CREAT and flags & os.O_EXCL:
        return 1  # CREATE_NEW
    if flags & os.O_CREAT:
        return 4  # OPEN_ALWAYS; truncate only after reparse validation
    return 3  # OPEN_EXISTING


def _open_windows_no_follow(path: Path, flags: int) -> int:
    import msvcrt
    from ctypes import wintypes

    class FILE_ATTRIBUTE_TAG_INFO(ctypes.Structure):
        _fields_ = [
            ("FileAttributes", wintypes.DWORD),
            ("ReparseTag", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL(  # type: ignore[attr-defined]
        "kernel32", use_last_error=True,
    )
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    get_info = kernel32.GetFileInformationByHandleEx
    get_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    get_info.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    access_mode = flags & 0x3
    desired_access = 0x80000000 if access_mode == os.O_RDONLY else 0x40000000
    if access_mode == os.O_RDWR:
        desired_access = 0x80000000 | 0x40000000
    share_mode = 0 if access_mode != os.O_RDONLY else 0x1 | 0x2 | 0x4
    handle = create_file(
        str(path),
        desired_access,
        share_mode,
        None,
        _windows_creation_disposition(flags),
        0x00000080 | 0x00200000,  # NORMAL | OPEN_REPARSE_POINT
        None,
    )
    invalid = ctypes.c_void_p(-1).value
    if handle == invalid:
        error = ctypes.get_last_error()  # type: ignore[attr-defined]
        if error in (2, 3):
            raise FileNotFoundError(error, "file not found", str(path))
        if error in (80, 183):
            raise FileExistsError(error, "file exists", str(path))
        raise OSError(error, "CreateFileW failed", str(path))
    transferred = False
    try:
        info = FILE_ATTRIBUTE_TAG_INFO()
        if not get_info(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise OSError(
                ctypes.get_last_error(),  # type: ignore[attr-defined]
                "file attribute query failed",
                str(path),
            )
        if info.FileAttributes & 0x00000400:  # FILE_ATTRIBUTE_REPARSE_POINT
            raise OSError("refusing to follow a Windows reparse point")
        if info.FileAttributes & 0x00000010:  # FILE_ATTRIBUTE_DIRECTORY
            raise IsADirectoryError(str(path))
        crt_flags = access_mode | getattr(os, "O_BINARY", 0)
        if flags & os.O_APPEND:
            crt_flags |= os.O_APPEND
        descriptor = msvcrt.open_osfhandle(  # type: ignore[attr-defined]
            int(handle), crt_flags,
        )
        transferred = True
        try:
            if flags & os.O_APPEND:
                os.lseek(descriptor, 0, os.SEEK_END)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor
    finally:
        if not transferred:
            kernel32.CloseHandle(handle)


def open_regular_no_follow(path: Path, flags: int, mode: int = 0o600) -> int:
    """Open a regular file without following a final symlink/reparse point."""
    target = Path(path)
    # O_TRUNC acts during open on POSIX. Defer it until after the descriptor is
    # proven to be a regular file, matching the Windows reparse-safe contract.
    # This prevents an unexpected special file from being modified before the
    # type guard rejects it.
    truncate = bool(flags & os.O_TRUNC)
    open_flags = flags & ~os.O_TRUNC
    if os.name == "nt":
        descriptor = _open_windows_no_follow(target, open_flags)
    else:
        descriptor = os.open(
            target, open_flags | getattr(os, "O_NOFOLLOW", 0), mode,
        )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise OSError("secure file target is not a regular file")
        if truncate:
            os.ftruncate(descriptor, 0)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def append_bytes_no_follow(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    sync: bool = False,
) -> None:
    """Append all ``payload`` bytes without following the final path entry."""
    descriptor = open_regular_no_follow(
        Path(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, mode,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while appending secure file")
            written += count
        if sync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes_no_follow(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o600,
    sync: bool = False,
) -> None:
    """Replace a regular file's bytes without following a final link."""
    descriptor = open_regular_no_follow(
        Path(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while writing secure file")
            written += count
        if sync:
            os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_bytes_no_follow(path: Path, *, max_bytes: int | None = None) -> bytes:
    """Read a regular file through a no-follow descriptor with an optional cap."""
    descriptor = open_regular_no_follow(Path(path), os.O_RDONLY)
    try:
        size = os.fstat(descriptor).st_size
        if max_bytes is not None and size > max_bytes:
            raise OSError("secure file exceeds the configured size limit")
        chunks: list[bytes] = []
        remaining = size if max_bytes is None else min(size, max_bytes)
        while remaining > 0:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def copy_regular_no_follow(
    source: Path,
    destination: Path,
    *,
    max_bytes: int | None = None,
) -> tuple[int, str]:
    """Copy one stable regular file without following either final path.

    The destination must not already exist.  The source is held by descriptor
    for the entire copy, bounded while streaming, and checked again afterward
    so a concurrent rewrite cannot be silently accepted as a stable import.
    Returns the byte count and SHA-256 of the bytes actually copied.
    """
    source_fd = open_regular_no_follow(Path(source), os.O_RDONLY)
    destination_path = Path(destination)
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if max_bytes is not None and before.st_size > max_bytes:
            raise OSError("secure file exceeds the configured size limit")
        destination_fd = open_regular_no_follow(
            destination_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if max_bytes is not None and copied > max_bytes:
                raise OSError("secure file exceeds the configured size limit")
            digest.update(chunk)
            view = memoryview(chunk)
            written = 0
            while written < len(view):
                count = os.write(destination_fd, view[written:])
                if count <= 0:
                    raise OSError("short write while copying secure file")
                written += count
        after = os.fstat(source_fd)
        content_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if copied != before.st_size or any(
            getattr(before, field) != getattr(after, field)
            for field in content_fields
        ):
            raise OSError("source file changed while copying")

        # ctime also changes for metadata-only operations. In particular,
        # macOS cloud-file hydration and provenance/xattr updates can happen
        # on the first read without changing the file's contents. Treating
        # that as an unconditional failure made valid iCloud, OneDrive, and
        # other provider-backed files impossible to import. Keep the stronger
        # race check by re-reading and hashing only when ctime moved: this
        # accepts harmless metadata updates while still rejecting a same-size
        # rewrite whose mtime was restored.
        if before.st_ctime_ns != after.st_ctime_ns:
            os.lseek(source_fd, 0, os.SEEK_SET)
            verification_digest = hashlib.sha256()
            verified = 0
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                verified += len(chunk)
                if max_bytes is not None and verified > max_bytes:
                    raise OSError(
                        "secure file exceeds the configured size limit"
                    )
                verification_digest.update(chunk)
            confirmed = os.fstat(source_fd)
            if (
                verified != copied
                or verification_digest.digest() != digest.digest()
                or any(
                    getattr(before, field) != getattr(confirmed, field)
                    for field in content_fields
                )
            ):
                raise OSError("source file changed while copying")
        os.fsync(destination_fd)
        return copied, digest.hexdigest()
    except BaseException:
        # Windows cannot unlink an open file. Close our private destination
        # handle before rolling back a failed or unstable copy.
        if destination_fd is not None:
            os.close(destination_fd)
            destination_fd = None
        try:
            destination_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


__all__ = [
    "append_bytes_no_follow",
    "copy_regular_no_follow",
    "open_regular_no_follow",
    "read_bytes_no_follow",
    "write_bytes_no_follow",
]
