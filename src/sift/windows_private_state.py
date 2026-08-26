"""Windows ACL hardening for confidential Sift state directories."""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

from sift.subprocess_safety import run_bounded_capture


class WindowsAclError(RuntimeError):
    pass


_SID_RE = re.compile(r"S-1-(?:\d+-)+\d+", re.IGNORECASE)


def _system_executable(name: str) -> str | None:
    root = os.environ.get("SystemRoot") or os.environ.get("WINDIR")
    if root:
        candidate = Path(root) / "System32" / name
        if candidate.is_file():
            return str(candidate)
    return shutil.which(name)


@lru_cache(maxsize=1)
def _current_user_sid() -> str:
    whoami = _system_executable("whoami.exe")
    if whoami is None:
        raise WindowsAclError("Windows whoami.exe is unavailable")
    try:
        result = run_bounded_capture(
            [whoami, "/user", "/fo", "csv", "/nh"],
            timeout=10,
            check=False,
            stdout_limit=64 * 1024,
            stderr_limit=64 * 1024,
        )
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise WindowsAclError("Windows user SID lookup failed") from exc
    match = _SID_RE.search(result.stdout or "")
    if result.returncode != 0 or match is None:
        raise WindowsAclError("Windows user SID lookup returned no valid SID")
    return match.group(0).upper()


def _private_dacl_sddl(user_sid: str) -> str:
    if _SID_RE.fullmatch(user_sid) is None:
        raise WindowsAclError("refusing malformed Windows user SID")
    # D:P protects the DACL from inheritance.  Only LocalSystem and the
    # current researcher receive inheritable full control; pre-existing ACEs
    # are replaced, not merged, so an explicitly broad old ACL cannot survive.
    return f"D:P(A;OICI;FA;;;SY)(A;OICI;FA;;;{user_sid.upper()})"


def secure_private_directory(path: Path) -> None:
    """Replace ``path``'s DACL with a private, inheritance-protected DACL.

    The file identity cache avoids repeating native ACL work on every state
    access while still re-securing a directory that is deleted and recreated.
    Network filesystems which expose no stable file ID deliberately bypass the
    cache. Each process performs its own first-use check.
    """
    if os.name != "nt":
        raise WindowsAclError("Windows private-directory ACL called off Windows")
    resolved = Path(path).resolve()
    try:
        metadata = resolved.stat()
    except OSError as exc:
        raise WindowsAclError("Windows private directory is unavailable") from exc
    if metadata.st_ino:
        _secure_private_directory_once(str(resolved), metadata.st_dev, metadata.st_ino)
    else:
        _apply_private_directory_acl(resolved)


@lru_cache(maxsize=512)
def _secure_private_directory_once(path: str, device: int, inode: int) -> None:
    del device, inode  # identity is intentionally part of the cache key
    _apply_private_directory_acl(Path(path))


def _apply_private_directory_acl(path: Path) -> None:
    sddl = _private_dacl_sddl(_current_user_sid())
    try:
        from ctypes import wintypes

        advapi32 = ctypes.WinDLL(  # type: ignore[attr-defined]
            "advapi32", use_last_error=True,
        )
        kernel32 = ctypes.WinDLL(  # type: ignore[attr-defined]
            "kernel32", use_last_error=True,
        )
    except (AttributeError, OSError) as exc:
        raise WindowsAclError("Windows ACL APIs are unavailable") from exc

    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.ULONG),
    ]
    convert.restype = wintypes.BOOL
    get_dacl = advapi32.GetSecurityDescriptorDacl
    get_dacl.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(wintypes.BOOL),
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(wintypes.BOOL),
    ]
    get_dacl.restype = wintypes.BOOL
    set_named = advapi32.SetNamedSecurityInfoW
    set_named.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    set_named.restype = wintypes.DWORD
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    # Serialize with AppContainer's per-run ACL snapshots. Without the shared
    # named mutex, hardening .sift while a Windows script is active could make
    # its later snapshot restore overwrite this DACL (or remove the run's
    # temporary re-allow mid-execution).
    try:
        from sift.win_appcontainer import _acquire_acl_mutex

        release_mutex = _acquire_acl_mutex()
    except Exception as exc:  # noqa: BLE001 -- normalize native API failures
        raise WindowsAclError("Windows ACL mutation lock is unavailable") from exc
    try:
        descriptor = ctypes.c_void_p()
        if not convert(sddl, 1, ctypes.byref(descriptor), None):
            raise WindowsAclError(
                "Windows could not build the private DACL "
                f"(error {ctypes.get_last_error()})"  # type: ignore[attr-defined]
            )
        try:
            present = wintypes.BOOL()
            defaulted = wintypes.BOOL()
            dacl = ctypes.c_void_p()
            if not get_dacl(
                descriptor,
                ctypes.byref(present),
                ctypes.byref(dacl),
                ctypes.byref(defaulted),
            ) or not present.value or not dacl.value:
                raise WindowsAclError(
                    "Windows could not read the private DACL "
                    f"(error {ctypes.get_last_error()})"  # type: ignore[attr-defined]
                )
            # SE_FILE_OBJECT | DACL_SECURITY_INFORMATION |
            # PROTECTED_DACL_SECURITY_INFORMATION.
            path_buffer = ctypes.create_unicode_buffer(str(path))
            error = int(
                set_named(
                    path_buffer, 1, 0x00000004 | 0x80000000,
                    None, None, dacl, None,
                )
            )
            if error != 0:
                raise WindowsAclError(
                    f"Windows could not apply the private DACL (error {error})"
                )
        finally:
            kernel32.LocalFree(descriptor)
    finally:
        try:
            release_mutex()
        except Exception as exc:  # noqa: BLE001 -- normalize native API failures
            raise WindowsAclError("Windows ACL mutation lock release failed") from exc


__all__ = ["WindowsAclError", "secure_private_directory"]
