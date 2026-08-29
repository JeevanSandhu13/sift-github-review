"""Sift — analysis checkpoints.

A checkpoint is a *non-destructive* bookmark: "at turn N, these
result_ids were the model-visible analysis so far." Creating one
touches nothing else — no chat history truncation, no result-store
mutation. That distinction separates it from ``rewind_to``
(``ui.py``), which is destructive: it truncates ``chat_history.jsonl``
and hides every result row the truncated prefix no longer references.

Checkpoints exist so a researcher can mark "this is a state worth
coming back to" *before* trying something risky (a different model
spec, a different subset, a different transformation) without having
to remember a raw turn index. Two operations build on the bookmark:

- **Restore**: rewind to the checkpoint's turn_index. This reuses
  ``ui.SiftBridge.rewind_to`` verbatim — restoring a checkpoint IS a
  rewind, so it inherits every safety property that method already
  has (busy-runner refusal, crash-safe SQL-hide-before-truncate
  ordering, script_code purge on hidden rows, session_state refresh).
  A checkpoint is just a friendlier way to name the target turn_index.
- **Compare**: given two checkpoints, diff the sets of result_ids
  each one's prefix referenced, plus a per-analysis-type tally. This
  never opens ``sanitized_payload`` — every field the compare view
  reads (``label``, ``analysis_type``, ``source_dataset``,
  ``created_at``) is already metadata the model itself was shown
  when the result was created, so exposing it to the compare view
  crosses no new boundary. "Minimal" is deliberate: diffing arbitrary
  payload shapes across 13 analysis types is a much larger, more
  fragile feature than a researcher asking "what did each branch
  actually run."

Storage: ``<cwd>/.sift/checkpoints.json``, one small JSON document,
following the exact durability pattern already established by
``session_state.py`` — tempfile + ``os.replace`` for atomic writes,
a per-cwd lock for read-modify-write safety, best-effort persistence
(a checkpoint write failure must never break a chat turn).

Not a boundary file: nothing here reads raw data, computes a
statistic, or decides what crosses the privacy boundary. It only
bookmarks turn indices and result ids that already exist elsewhere.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sift.file_lock import exclusive_file_lock

CHECKPOINTS_FILENAME = "checkpoints.json"
CHECKPOINTS_VERSION = 1

# Bounded like every other researcher-controlled list in Sift (recent
# results, session names) — a session that runs for months shouldn't
# grow this file without limit. 50 named checkpoints is generous for
# a single analysis session; the researcher can delete stale ones.
MAX_CHECKPOINTS = 50
# Matches session_state.py's custom-name cap — a checkpoint label is
# the same kind of short researcher-authored string.
MAX_LABEL_LEN = 120

_LOCKS: dict[Path, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def _lock_for(cwd: Path) -> threading.Lock:
    key = cwd.resolve() if cwd.is_dir() else cwd
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def evict_lock(cwd: Path) -> None:
    """Drop the cached lock for ``cwd``. Mirrors
    ``session_state.evict_state_lock`` — call from ``delete_session``
    so a long-running daemon that opens/discards many sessions doesn't
    accumulate one lock per cwd forever."""
    try:
        key = cwd.resolve() if cwd.is_dir() else cwd
    except OSError:
        key = cwd
    with _LOCKS_GUARD:
        _LOCKS.pop(key, None)


@dataclass
class Checkpoint:
    id: str
    label: str
    turn_index: int
    created_at: str
    result_ids: list[str] = field(default_factory=list)


def _path(cwd: Path) -> Path:
    return cwd / ".sift" / CHECKPOINTS_FILENAME


def _file_lock_path(cwd: Path) -> Path:
    from sift.config import ensure_private_sift_dir

    ensure_private_sift_dir(cwd)
    return cwd / ".sift" / f"{CHECKPOINTS_FILENAME}.lock"


def _read(cwd: Path) -> list[Checkpoint]:
    path = _path(cwd)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, dict):
        return []
    items = raw.get("checkpoints")
    if not isinstance(items, list):
        return []
    out: list[Checkpoint] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            out.append(Checkpoint(
                id=str(it["id"]),
                label=str(it.get("label", "")),
                turn_index=int(it["turn_index"]),
                created_at=str(it.get("created_at", "")),
                result_ids=[str(x) for x in it.get("result_ids", [])
                            if isinstance(x, (str, int))],
            ))
        except (KeyError, TypeError, ValueError):
            # A corrupted single entry shouldn't make the whole file
            # unreadable — skip it and keep the rest.
            continue
    return out


def _atomic_write(cwd: Path, checkpoints: list[Checkpoint]) -> bool:
    path = _path(cwd)
    try:
        from sift.config import ensure_private_sift_dir
        ensure_private_sift_dir(path.parent.parent)
        payload = {
            "version": CHECKPOINTS_VERSION,
            "checkpoints": [asdict(c) for c in checkpoints],
        }
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=".checkpoints.",
            suffix=".tmp",
            delete=False,
        ) as tf:
            tf.write(encoded)
            tf.flush()
            os.fsync(tf.fileno())
            tmp_name = tf.name
        os.replace(tmp_name, path)
        return True
    except OSError:
        try:
            if "tmp_name" in locals():
                os.unlink(tmp_name)  # type: ignore[name-defined]
        except OSError:
            pass
        return False


def list_checkpoints(cwd: Path) -> list[Checkpoint]:
    """Return checkpoints in creation order (oldest first)."""
    return _read(cwd)


def add_checkpoint(
    cwd: Path, *, label: str, turn_index: int, result_ids: list[str],
) -> tuple[Checkpoint | None, str | None]:
    """Create and persist a new checkpoint.

    Returns ``(checkpoint, None)`` on success or ``(None, reason)``
    on refusal — refused when the label is empty after trimming/
    capping, or the session already holds ``MAX_CHECKPOINTS``.
    """
    with _lock_for(cwd), exclusive_file_lock(_file_lock_path(cwd)):
        label = (label or "").strip()[:MAX_LABEL_LEN]
        if not label:
            return None, "checkpoint label cannot be empty"
        existing = _read(cwd)
        if len(existing) >= MAX_CHECKPOINTS:
            return None, (
                f"this session already has {MAX_CHECKPOINTS} checkpoints — "
                "delete one before creating another"
            )
        next_n = 1
        seen_ids = {c.id for c in existing}
        while f"cp{next_n}" in seen_ids:
            next_n += 1
        cp = Checkpoint(
            id=f"cp{next_n}",
            label=label,
            turn_index=turn_index,
            created_at=datetime.now(timezone.utc).isoformat(),
            result_ids=list(result_ids),
        )
        existing.append(cp)
        if not _atomic_write(cwd, existing):
            return None, "could not persist checkpoint"
        return cp, None


def delete_checkpoint(cwd: Path, checkpoint_id: str) -> bool:
    """Remove a checkpoint by id. Returns whether one was removed."""
    with _lock_for(cwd), exclusive_file_lock(_file_lock_path(cwd)):
        existing = _read(cwd)
        kept = [c for c in existing if c.id != checkpoint_id]
        if len(kept) == len(existing):
            return False
        return _atomic_write(cwd, kept)


def get_checkpoint(cwd: Path, checkpoint_id: str) -> Checkpoint | None:
    for c in _read(cwd):
        if c.id == checkpoint_id:
            return c
    return None


def prune_checkpoints_after(cwd: Path, cut: int) -> list[str]:
    """Drop every checkpoint whose ``turn_index > cut`` and return
    the ids removed.

    Called from ``ui.rewind_to`` right after a truncate to ``cut``
    commits. ``rewind_to(cut)`` keeps user messages ``0..cut-1`` and
    removes everything from message ``cut`` onward, so a checkpoint
    is still valid exactly when its own ``turn_index <= cut`` — its
    bookmarked messages are a subset of (or, at equality, exactly)
    what survives. Only ``turn_index > cut`` references bytes that
    got truncated away (and, per ``hide_results_not_in``, result rows
    that are now hidden).

    The boundary case matters: restoring checkpoint X calls
    ``rewind_to(X.turn_index)``, i.e. ``cut == X.turn_index``. Using
    ``>=`` here would prune X the instant it's restored, even though
    the surviving history is EXACTLY what X bookmarked — the
    researcher would lose their own bookmark by using it. ``>``
    keeps X (and anything earlier) alive; only checkpoints strictly
    past the cut are stale.

    Runs for BOTH the explicit "restore this checkpoint" path and the
    ordinary edit-a-past-message rewind path — both call the same
    ``rewind_to``, so both get pruning for free.

    Best-effort like every other post-commit step in ``rewind_to``:
    a failure here leaves stale checkpoint entries (annoying, not
    unsafe — restoring one just fails cleanly) rather than blocking
    the rewind that already committed.
    """
    with _lock_for(cwd), exclusive_file_lock(_file_lock_path(cwd)):
        existing = _read(cwd)
        kept = [c for c in existing if c.turn_index <= cut]
        removed = [c.id for c in existing if c.turn_index > cut]
        if removed and not _atomic_write(cwd, kept):
            return []
        return removed
