"""Sift — working-directory configuration and path sandboxing.

Every Sift MCP tool is gated by ``resolve_in_cwd()``: the
researcher/Claude-supplied path must resolve inside the active working
directory or the call is refused.

The active cwd is sourced in two layers:

1. **Per-task override** (``_cwd_var``) — a :class:`contextvars.ContextVar`
   bound by :func:`use_cwd` around any code that runs inside a specific
   session. This is what makes concurrent sessions safe: each
   :class:`~sift.runner.SessionRunner` runs its turn inside
   ``use_cwd(runner.cwd)`` so tool handlers (and any sub-tasks the SDK
   spawns) see the runner's cwd, regardless of which other runners are
   simultaneously executing turns. ContextVar binding is asyncio-task
   local — sister tasks see their own bindings.
2. **Process default** (``_cwd_default``) — the fallback used when no
   per-task override is in scope. Set by :func:`set_cwd` at startup
   and from tests. Web UI runners DO NOT update this; they bind via
   ``use_cwd`` only, so a focus switch in the UI doesn't trample tool
   execution in another session.

Anything that reads ``get_cwd()`` outside of a runner-bound context
(e.g., startup, test fixtures) gets the default. Anything inside a
runner-bound context gets the override.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator


class PathEscapeError(ValueError):
    """Raised when a tool input resolves to a path outside the working directory."""


class WorkspaceScopeError(ValueError):
    """Raised when a proposed workspace would expose an unreasonably broad tree."""


class PrivateStateError(RuntimeError):
    """Raised when Sift cannot prove its confidential state directory is private."""


_DANGEROUS_WORKSPACE_LITERALS: frozenset[Path] = frozenset({
    Path("/Users"),
    Path("/home"),
    # Denylisted literal only; it is never opened here.
    Path("/tmp"),  # nosec B108
    Path("/private"),
    Path("/private/tmp"),
    Path("/var"),
    Path("/private/var"),
    Path("/etc"),
    Path("/private/etc"),
    Path("/usr"),
    Path("/System"),
    Path("/Library"),
    Path("/Applications"),
    Path("/Volumes"),
    Path("/mnt"),
    Path("/media"),
    Path("/opt"),
    Path("/bin"),
    Path("/sbin"),
    Path("/dev"),
})

_DANGEROUS_TOP_LEVEL_NAMES: frozenset[str] = frozenset({
    "applications",
    "bin",
    "dev",
    "etc",
    "home",
    "library",
    "media",
    "mnt",
    "opt",
    "private",
    "program files",
    "program files (x86)",
    "programdata",
    "sbin",
    "system",
    "users",
    "usr",
    "var",
    "volumes",
    "windows",
})


def dangerous_workspace_reason(path: Path) -> str | None:
    """Return an explanation when ``path`` is too broad for a workspace.

    The generic anchor check covers POSIX ``/``, every Windows drive root,
    and UNC share roots.  This belongs in backend configuration—not only the
    folder picker—so CLI, automation, and future GUI clients cannot bypass the
    privacy boundary before tools and script sandboxes receive their cwd.
    """
    resolved = Path(path).expanduser().resolve()
    anchor = resolved.anchor
    if anchor and resolved == Path(anchor):
        return (
            f"{resolved} is a filesystem root and is too broad to use as a "
            "project folder; pick a specific project directory instead"
        )
    if resolved in _DANGEROUS_WORKSPACE_LITERALS:
        return (
            f"{resolved} is too broad to use as a project folder—the sandbox "
            "would grant scripts access to this entire subtree; pick a specific "
            "project directory instead"
        )
    # ``Path('/Users')`` resolves to ``C:\Users`` when a cross-platform
    # workflow or imported project supplies POSIX spelling on Windows. Match
    # unambiguous top-level system/profile directory names relative to *each*
    # platform's anchor instead of relying only on POSIX literal equality.
    if (
        anchor
        and resolved.parent == Path(anchor)
        and resolved.name.casefold() in _DANGEROUS_TOP_LEVEL_NAMES
    ):
        return (
            f"{resolved} is a system or profile root and is too broad to use "
            "as a project folder; pick a specific project directory instead"
        )
    if resolved == Path.home().resolve():
        return (
            "your home directory is too broad to use as a project folder—the "
            "sandbox would grant scripts access to credentials, documents, and "
            "other personal files; pick a specific project subdirectory instead"
        )
    return None


def validate_workspace(path: Path) -> Path:
    """Resolve and validate one workspace root for every product entry path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(f"not a directory: {resolved}")
    reason = dangerous_workspace_reason(resolved)
    if reason is not None:
        raise WorkspaceScopeError(reason)
    return resolved


# Process-wide default. Used when no per-task ``use_cwd`` is in
# effect (startup, tests). Web UI runners override this via the
# ContextVar below; they do NOT mutate it, so concurrent runners
# don't trample each other.
_cwd_default: Path = Path.cwd().resolve()

# Per-asyncio-task override. ``ContextVar.set`` returns a Token that
# scopes the binding to the current task; sister tasks see the
# default. ``None`` means "no override — use the process default."
_cwd_var: ContextVar[Path | None] = ContextVar("sift_cwd", default=None)


def set_cwd(path: Path) -> Path:
    """Set the process-wide default working directory.

    Called from terminal startup (``ui.py``) and tests. Web UI runners
    bind their cwd via :func:`use_cwd` instead — see the module
    docstring for why.
    """
    global _cwd_default
    resolved = validate_workspace(path)
    _cwd_default = resolved
    return _cwd_default


def get_cwd() -> Path:
    """Return the active working directory.

    Returns the per-task override if one is in effect (set by
    :func:`use_cwd`), otherwise the process default.
    """
    val = _cwd_var.get()
    return val if val is not None else _cwd_default


@contextmanager
def use_cwd(path: Path) -> Iterator[Path]:
    """Bind the active cwd for the current asyncio task / context.

    Usage::

        with use_cwd(runner.cwd):
            # get_cwd() and resolve_in_cwd() see runner.cwd here,
            # regardless of any other concurrent runners.
            ...

    The binding is scoped to the calling task's context — sister
    tasks running concurrently see their own bindings (or the
    process default). Restores the previous value on exit.
    """
    resolved = validate_workspace(path)
    token = _cwd_var.set(resolved)
    try:
        yield resolved
    finally:
        _cwd_var.reset(token)


def ensure_private_sift_dir(cwd: Path) -> Path:
    """Create ``<cwd>/.sift`` and prove it is private to this account.

    Every Sift-owned file lives under ``.sift`` — chat history with
    verbatim user messages (including any credentials a researcher
    pasted), the SQLite results store with sanitized payloads and
    script source, the pre-SDC ``result.json`` for each run, the
    raw ``stdout.log`` / ``stderr.log`` for each script, the
    researcher-authored script source itself, and per-run scratch
    via ``TMPDIR=<run_dir>/tmp``. None of that should be readable
    by other users on the same machine.

    POSIX uses a no-follow directory descriptor, owner verification, and mode
    0700. Windows replaces the DACL with an inheritance-protected ACL granting
    only the current account and LocalSystem. A symlink/junction or filesystem
    that cannot enforce the relevant privacy control fails closed: Sift must
    not quietly store raw logs, chat history, and pre-SDC results in a merely
    "less hardened" directory while advertising confidential-data support.
    """
    sift_dir = Path(cwd) / ".sift"
    try:
        sift_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise PrivateStateError("Sift could not create its private state directory") from exc
    is_junction = getattr(sift_dir, "is_junction", lambda: False)
    if sift_dir.is_symlink() or is_junction():
        raise PrivateStateError(
            "Sift's private state directory cannot be a symlink or junction"
        )

    if os.name == "nt":
        from sift.windows_private_state import (
            WindowsAclError,
            secure_private_directory,
        )

        try:
            secure_private_directory(sift_dir)
        except WindowsAclError as exc:
            raise PrivateStateError(
                "Sift could not apply a current-user-only Windows ACL to its "
                "private state directory; refusing to store confidential state"
            ) from exc
        return sift_dir

    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(sift_dir, flags)
    except OSError as exc:
        raise PrivateStateError(
            "Sift could not safely open its private state directory"
        ) from exc
    try:
        os.fchmod(descriptor, 0o700)
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise PrivateStateError("Sift's private state path is not a directory")
        get_euid = getattr(os, "geteuid", None)
        if callable(get_euid) and metadata.st_uid != get_euid():
            raise PrivateStateError(
                "Sift's private state directory is not owned by the current user"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise PrivateStateError(
                "Sift's private state directory permits group or other access"
            )
    except OSError as exc:
        raise PrivateStateError(
            "Sift could not enforce owner-only permissions on its private state "
            "directory; refusing to store confidential state"
        ) from exc
    finally:
        os.close(descriptor)
    return sift_dir


def resolve_in_cwd(user_path: str) -> Path:
    """Resolve a researcher/Claude-supplied path within the working directory.

    Accepts either a relative path (resolved against the cwd) or an absolute
    path (must be within the cwd). Follows symlinks via .resolve(). Rejects
    anything that ends up outside the cwd tree.

    Security note: this is the one authoritative place paths get validated.
    Every MCP tool that takes a path argument goes through this function.
    If a new tool touches paths without calling this, that's a regression.

    Reads the cwd via :func:`get_cwd`, which honors the per-task
    override — so concurrent runners get sandboxed against THEIR cwd,
    not whichever session the UI happens to be focused on.
    """
    if not user_path:
        raise PathEscapeError("empty path")
    p = Path(user_path).expanduser()
    cwd = get_cwd()
    if not p.is_absolute():
        p = cwd / p
    resolved = p.resolve()
    # `is_relative_to` was added in Python 3.9; we require 3.10+.
    if not (resolved == cwd or resolved.is_relative_to(cwd)):
        raise PathEscapeError(
            f"path {user_path!r} resolves to {resolved}, which is outside "
            f"the working directory {cwd}. Sift tools can only access "
            f"files inside the working directory."
        )
    return resolved
