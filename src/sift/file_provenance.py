"""Researcher-staged file manifest for the per-session cwd.

The script sandbox at ``executor.py`` intentionally allows writes to
the session cwd — ``saveRDS`` / ``df.to_csv`` / ``save "panel.dta"``
are part of the normal R/Stata/Python workflow. That same surface,
however, lets a model-authored script write raw row values into a
file with a script extension (``data_dump.R``, ``smuggled.py``) and
then later recall the bytes through ``read_attached_file`` /
``submit_script_file``, bypassing the SDC sanitizer that gates every
``submit_script`` result payload.

This module is the closure for that gap. The manifest at
``<cwd>/.sift/staged_files.json`` records every cwd top-level file
Sift considers researcher-known, ALONG WITH a content fingerprint
(sha256 + size) captured at stage time. The fingerprint is the key
difference from a name-only manifest: a sandbox script that
overwrites a previously-staged filename (say ``analysis.py``) with
raw row bytes can't round-trip the data through
``read_attached_file`` / ``submit_script_file`` because the
current file's sha256 no longer matches any recorded fingerprint
for that name.

The manifest tracks two kinds of entries:

  * Files present in cwd at session-open time (the initial snapshot
    — the researcher staged them before opening Sift).
  * Files added through the bridge's file-staging endpoints
    (``add_files`` native picker, ``add_files_from_blobs`` paste/
    drop, ``upload_files`` landing-page drop). These also originate
    from explicit researcher action.

In both cases the fingerprint is captured at the moment of
staging. A later re-stage of the same name with NEW content (e.g.
researcher updates ``analysis.py`` outside Sift and drops it back
in) appends a new fingerprint; the old one stays in the manifest
so the audit trail is preserved, but only the current on-disk
sha256 needs to match SOMETHING in the recorded set for
``is_known`` to return True.

Recallable-only fingerprinting: only files whose extension matches
a recall surface (scripts, logs, graphs — the set
``read_attached_file`` / ``submit_script_file`` /
``search_in_session_files`` would actually return bytes for) are
fingerprinted. Data files (``.csv``, ``.dta``, ``.parquet``, ...)
cannot have their bytes returned by any recall path regardless of
provenance state, so spending I/O to hash a multi-GB dataset at
session-open is wasteful and never gains security. A 3 GB ``.dta``
in cwd would otherwise block folder-open for seconds on SSD and
much longer on network mounts.

The manifest lives under ``<cwd>/.sift/`` which the analysis sandbox
already deny-reads/writes, so a model script can't read or tamper
with it. Writes are atomic via tempfile + ``os.replace`` so a crash
mid-write leaves either the prior snapshot intact or the new one
fully written, never a half-truncated JSON.

Schema versions:

  * v1 (legacy): ``{"version": 1, "names": [list of strings]}`` —
    name-only entries. Sessions opened before the fingerprint
    upgrade carry these. v1 entries are read for the ``known_names``
    diagnostic but ``is_known`` returns False for them (no
    fingerprint to verify) — the researcher re-stages via the
    bridge to upgrade those entries to v2. We deliberately do NOT
    silently re-fingerprint v1 entries on read: doing so would
    promote whatever content is currently on disk to "the trusted
    fingerprint", which is exactly what an attacker would want if
    they'd already overwritten a staged file.
  * v2 (current): ``{"version": 2, "entries": [{"name": ...,
    "sha256": ..., "size_bytes": ...}, ...]}`` — multiple
    fingerprints per name are allowed (legitimate re-stage of an
    updated file). Entries without ``sha256`` / ``size_bytes`` are
    legacy / file-missing-at-stage-time and ``is_known`` rejects
    them too.

Fail-closed by design: a manifest read that yields a corrupt JSON,
an unreadable file at fingerprint check time, or any other I/O
hiccup returns False from ``is_known`` rather than letting the
call through with no verification.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Iterable
from pathlib import Path

from sift.file_lock import exclusive_file_lock
from sift.secure_file import open_regular_no_follow

MANIFEST_FILENAME = "staged_files.json"
MANIFEST_VERSION = 2

# Streaming chunk size for sha256. 64 KiB is large enough that the
# Python loop overhead is negligible vs. the hash work, small
# enough that we never hold a researcher's multi-GB CSV in memory.
_FINGERPRINT_CHUNK_BYTES = 64 * 1024


def _is_recallable_basename(name: str) -> bool:
    """Whether a file with this basename could have its bytes
    returned to the model through any recall surface.

    Only such files need a content fingerprint. The provenance
    manifest exists to defend against "script overwrites a staged
    file with raw rows, then recalls it as bytes." Data files
    (``.csv``, ``.dta``, ``.rds``, ``.parquet``, ``.jsonl``, ...)
    are rejected by ``read_attached_file``, ``submit_script_file``,
    and ``search_in_session_files`` regardless of provenance state
    — they cannot be recalled as bytes at all. Hashing them at
    session-open does no security work and reads multi-GB datasets
    end-to-end before the UI is usable.

    The recallable set is the union of script / log / graph
    extensions from ``session_files`` — that's the single source
    of truth for what the recall tools surface. If a new
    extension becomes recallable there, it picks up provenance
    binding automatically through this gate.
    """
    # Import lazily so this module stays light-weight and avoids
    # circulars with ``session_files`` if it ever grows imports
    # from here.
    from sift.session_files import GRAPH_EXTS, LOG_EXTS, SCRIPT_EXTS
    ext = Path(name).suffix.lower()
    return ext in SCRIPT_EXTS or ext in LOG_EXTS or ext in GRAPH_EXTS


def _manifest_path(cwd: Path) -> Path:
    return cwd / ".sift" / MANIFEST_FILENAME


def _manifest_lock_path(cwd: Path) -> Path:
    from sift.config import ensure_private_sift_dir

    ensure_private_sift_dir(cwd)
    return cwd / ".sift" / f"{MANIFEST_FILENAME}.lock"


def _fingerprint(path: Path) -> dict[str, int | str] | None:
    """Compute sha256 + size of a regular file, returning ``None``
    on any failure (missing, unreadable, OSError mid-stream).

    Callers MUST treat ``None`` as a hard "cannot verify" — never
    fall through to a name-only match. Streams in 64 KiB chunks so
    arbitrarily-large data files don't balloon memory.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return None
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_FINGERPRINT_CHUNK_BYTES), b""):
                h.update(chunk)
    except OSError:
        return None
    return {"sha256": h.hexdigest(), "size_bytes": int(size)}


def _read_entries(path: Path) -> list[dict[str, object]]:
    """Read the manifest as a normalised list of entry dicts.

    Each entry has at least a ``"name"`` (string). v2 entries
    additionally have ``"sha256"`` (hex string) and ``"size_bytes"``
    (int). Reads handle three schemas defensively:

      * v2 (current): top-level ``{"version": 2, "entries": [...]}``.
        Entries are returned as-is (filtered to those with a valid
        string ``"name"``).
      * v1 (legacy): top-level ``{"version": 1, "names": [...]}``.
        Each name is upgraded to ``{"name": n}`` — no fingerprint,
        ``is_known`` will reject it until re-staged.
      * Anything else (missing file, malformed JSON, unrecognised
        schema): empty list — callers treat that as "nothing is
        staged yet" rather than crashing the read.

    The atomic-write path means a half-written manifest cannot be
    observed; corruption only surfaces if the file was edited
    externally.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    version = data.get("version", 1)
    if isinstance(version, int) and version >= 2:
        entries_raw = data.get("entries")
        if not isinstance(entries_raw, list):
            return []
        out: list[dict[str, object]] = []
        for e in entries_raw:
            if not isinstance(e, dict):
                continue
            name = e.get("name")
            if not isinstance(name, str) or not name:
                continue
            kept: dict[str, object] = {"name": name}
            sha = e.get("sha256")
            size = e.get("size_bytes")
            if isinstance(sha, str) and sha:
                kept["sha256"] = sha
            if isinstance(size, int) and size >= 0:
                kept["size_bytes"] = size
            out.append(kept)
        return out
    # v1 fallback (or version field missing).
    names = data.get("names")
    if not isinstance(names, list):
        return []
    return [{"name": n} for n in names if isinstance(n, str) and n]


def _write_entries(path: Path, entries: list[dict[str, object]]) -> None:
    """Atomic write of the v2 manifest. See ``_read_entries`` for
    schema.

    Same posture as ``policy.save_policy`` — direct ``write_text``
    on a manifest the bridge re-writes on every staging event would
    leave half-written JSON observable to a concurrent read after a
    crash, and the read would silently start over with an empty
    list. ``os.replace`` is a true atomic rename within one
    filesystem.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sort entries deterministically: by name first, then by sha256
    # (lex order over hex digest), then by size. Stable order keeps
    # diffs across rewrites legible for researcher audit.
    def sort_size(entry: dict[str, object]) -> int:
        value = entry.get("size_bytes")
        return value if isinstance(value, int) and not isinstance(value, bool) else -1

    sorted_entries = sorted(
        entries,
        key=lambda e: (
            str(e.get("name") or ""),
            str(e.get("sha256") or ""),
            sort_size(e),
        ),
    )
    payload = json.dumps(
        {"version": MANIFEST_VERSION, "entries": sorted_entries},
        indent=2, sort_keys=True,
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=".staged_files.json.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _enumerate_cwd_top_level(cwd: Path) -> set[str]:
    """Return basenames of all regular files at cwd top level. Skips
    directories, symlinks (they could point outside cwd; the bridge
    stages by basename only, so a symlink target couldn't go through
    a legitimate add path), and dotfiles (Sift's own state lives
    under ``.sift/`` and any other dotfiles are researcher tooling
    that isn't part of the analysis surface).
    """
    out: set[str] = set()
    try:
        for child in cwd.iterdir():
            if not child.is_file() or child.is_symlink():
                continue
            if child.name.startswith("."):
                continue
            out.add(child.name)
    except OSError:
        pass
    return out


def initialize(cwd: Path) -> set[str]:
    """Snapshot cwd top-level files (with content fingerprints) into
    the manifest at FIRST session-open only.

    On the very first open the cwd snapshot is presumed researcher-
    staged: the researcher dropped those files there before opening
    Sift. Each file's sha256 + size is captured at this moment so a
    later sandbox-script overwrite shows up as a fingerprint
    mismatch in ``is_known``.

    Recallable-only: only files whose extension can have their
    bytes returned by a recall surface (scripts, logs, graphs) are
    fingerprinted. Data files don't need fingerprints because no
    recall path returns their bytes anyway, and a 3 GB ``.dta``
    would otherwise block folder-open while we hashed it.

    Once a manifest exists, subsequent re-opens MUST NOT re-snapshot
    — between sessions, the analysis sandbox may have written its
    own files into cwd (``df.to_csv("out.csv")`` is legitimate;
    ``open("smuggled.py", "w").write(...)`` from a model-authored
    script is the gap). Merging those in on reopen would silently
    promote sandbox output to "researcher-staged" and let
    ``read_attached_file`` / ``submit_script_file`` /
    ``search_in_session_files`` return their bytes — the same SDC
    bypass the manifest exists to prevent. The provenance guard
    must be effective across app restarts, not just within one
    live session.

    Backwards compatibility for sessions that pre-date this manifest:
    when the manifest file does not yet exist, we snapshot once and
    write it (the upgrade path). After that the manifest is the sole
    authority; new files added via the bridge's staging endpoints
    (``add_files`` / ``add_files_from_blobs`` / ``upload_files``)
    extend it through ``mark_known``. v1 manifests (pre-fingerprint)
    are NOT silently re-fingerprinted on read: doing so would
    promote whatever content is currently on disk to "the trusted
    fingerprint", which is exactly what an attacker would want if
    they'd already overwritten a staged file. v1 entries fail
    closed in ``is_known``; researcher re-stages via the bridge to
    upgrade them.
    """
    path = _manifest_path(cwd)
    with exclusive_file_lock(_manifest_lock_path(cwd)):
        # ``_read_entries`` returns ``[]`` for missing OR corrupt
        # manifests. We need to distinguish those: missing -> seed,
        # corrupt -> leave alone (don't silently seed an empty
        # manifest on top of a corrupt one and resnapshot whatever
        # is in cwd right now). ``path.exists()`` is the gate.
        if path.exists():
            return {e["name"] for e in _read_entries(path)}  # type: ignore[misc]
        # First open: fingerprint each top-level file whose extension
        # is recallable. Non-recallable extensions (data files) are
        # skipped — see the module docstring. ``_fingerprint`` returns
        # ``None`` on read failure; in that case we still record the
        # name (so ``known_names`` matches the directory listing) but
        # without a fingerprint — those entries fail closed in
        # ``is_known``.
        entries: list[dict[str, object]] = []
        for name in _enumerate_cwd_top_level(cwd):
            if not _is_recallable_basename(name):
                continue
            entry: dict[str, object] = {"name": name}
            fp = _fingerprint(cwd / name)
            if fp is not None:
                entry.update(fp)
            entries.append(entry)
        _write_entries(path, entries)
        return {e["name"] for e in entries}  # type: ignore[misc]


def mark_known(cwd: Path, names: Iterable[str]) -> set[str]:
    """Add the given basenames to the manifest with their current
    content fingerprints. Returns the resulting full set of names
    so the caller can log the new entries if it wants.

    ``Path(name).name`` is used to defensively basename the input —
    callers should already be passing basenames, but a stray
    ``/foo/bar.csv`` wouldn't smuggle a path-shaped key in.

    Recallable-only: non-recallable extensions (data files) are
    skipped silently. The bridge endpoints call ``mark_known`` after
    every staging event; a researcher dropping a folder of mixed
    files should not pay the cost of hashing each dataset, and the
    provenance gate doesn't apply to data files anyway.

    For each remaining input:

      * If the file exists at ``cwd/name`` and is readable, its
        sha256+size are computed and added as a new entry. A
        re-stage of the same name with IDENTICAL content (same
        sha256) is a no-op. A re-stage with NEW content (researcher
        updated the file outside Sift and re-dropped) appends a
        new entry alongside the existing one; both fingerprints
        remain valid in ``is_known`` until something explicitly
        retires them. This is the legitimate "I updated the script
        and re-attached it" flow.

      * If the file doesn't exist at stage time (test paths,
        traversal-shaped inputs that basename to a non-existent
        file), the name is recorded WITHOUT a fingerprint. Such
        entries fail closed in ``is_known`` — they exist for
        diagnostic visibility but don't authorise reads.
    """
    cleaned = [Path(n).name for n in names if n]
    cleaned = [n for n in cleaned if n and _is_recallable_basename(n)]
    if not cleaned:
        return {e["name"] for e in _read_entries(_manifest_path(cwd))}  # type: ignore[misc]
    path = _manifest_path(cwd)
    with exclusive_file_lock(_manifest_lock_path(cwd)):
        existing = _read_entries(path)
        # Dedup key: (name, sha256-or-None). Two entries with the
        # same name and same fingerprint are the same researcher
        # action; one fingerprintless entry per name is also kept
        # only once.
        seen: set[tuple[str, object]] = {
            (str(e["name"]), e.get("sha256")) for e in existing
        }
        merged: list[dict[str, object]] = list(existing)
        for name in cleaned:
            entry: dict[str, object] = {"name": name}
            fp = _fingerprint(cwd / name)
            if fp is not None:
                entry.update(fp)
            key = (name, entry.get("sha256"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(entry)
        _write_entries(path, merged)
        return {str(e["name"]) for e in merged}


def _allowed_fingerprints(
    entries: list[dict[str, object]], safe_name: str,
) -> set[tuple[str, int]]:
    """Return only fully typed manifest fingerprints for one basename."""
    allowed: set[tuple[str, int]] = set()
    for entry in entries:
        digest = entry.get("sha256")
        size = entry.get("size_bytes")
        if (entry.get("name") == safe_name and isinstance(digest, str)
                and isinstance(size, int) and not isinstance(size, bool)):
            allowed.add((digest, size))
    return allowed


def is_known(cwd: Path, name: str) -> bool:
    """Whether ``name`` (basename) is in the manifest AND its current
    on-disk content matches a recorded fingerprint.

    Fail-closed semantics:

      * Name not in manifest -> False
      * Name in manifest but no fingerprint recorded for it (legacy
        v1 entry, or file was missing at stage time) -> False
      * Name in manifest with fingerprints, file currently missing
        or unreadable -> False
      * Name in manifest with fingerprints, current sha256 not in
        the recorded set for that name -> False (this is the
        attack-detection path: a sandbox script that overwrote
        ``analysis.py`` with raw row bytes shows up here)
      * Name in manifest with fingerprints, current sha256 matches
        any recorded fingerprint for that name -> True

    Refuses path-traversal-shaped inputs by basenaming first; a
    caller that passed ``foo/../bar.csv`` gets the same answer as if
    they passed ``bar.csv``.

    Symlink rejection: if the on-disk file at the basename is a
    symlink, ``is_known`` returns False. The manifest fingerprint
    is meaningless against a symlink target the attacker controls,
    and the original staging path only records regular files.

    No lock needed for reads — the writer's atomic ``os.replace``
    means readers see either the old manifest or the new one,
    never a partial state.
    """
    safe = Path(name).name
    if not safe:
        return False
    entries = _read_entries(_manifest_path(cwd))
    allowed = _allowed_fingerprints(entries, safe)
    if not allowed:
        return False
    target = cwd / safe
    if target.is_symlink() or not target.is_file():
        return False
    fp = _fingerprint(target)
    if fp is None:
        return False
    return (str(fp["sha256"]), int(fp["size_bytes"])) in allowed


def read_verified_bytes(
    cwd: Path, name: str, *, max_bytes: int | None = None,
) -> bytes | None:
    """Read and authenticate the exact bytes returned to a caller.

    A separate ``is_known()`` followed by ``Path.read_bytes()`` has a
    check/use race: the path can be replaced after its fingerprint passes but
    before the second open.  This function opens once with no-follow semantics,
    hashes the bytes from that descriptor, and returns those same bytes only if
    they match a staged fingerprint.  Callers that disclose or execute file
    content must use this function, not a boolean pre-check followed by a new
    path read.
    """
    safe = Path(name).name
    if not safe:
        return None
    entries = _read_entries(_manifest_path(cwd))
    allowed = _allowed_fingerprints(entries, safe)
    if not allowed:
        return None
    target = cwd / safe
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        # The platform helper performs one atomic no-follow open. On Windows
        # it uses CreateFileW(FILE_FLAG_OPEN_REPARSE_POINT) and validates the
        # returned handle; on POSIX it adds O_NOFOLLOW. Holding that verified
        # descriptor is the identity boundary, so a later directory-entry
        # replacement cannot change the bytes we authenticate and return.
        fd = open_regular_no_follow(target, flags)
    except OSError:
        return None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return None
        if max_bytes is not None and info.st_size > max_bytes:
            return None
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, _FINGERPRINT_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
            total += len(chunk)
        if total != info.st_size:
            return None
        if (digest.hexdigest(), total) not in allowed:
            return None
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(fd)


def known_names(cwd: Path) -> set[str]:
    """Return the set of names in the manifest (test/diagnostic helper).

    Note: this does NOT verify fingerprints — it just lists what's
    been staged at some point. Use ``is_known`` for the actual
    "may the model read this file" gate check.
    """
    return {str(e["name"]) for e in _read_entries(_manifest_path(cwd))}
