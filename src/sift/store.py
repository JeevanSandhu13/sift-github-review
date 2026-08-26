"""Sift — result store.

Every sanitized analysis payload is persisted to a SQLite database in the
researcher's working directory (`<cwd>/.sift/results.db`). Claude
carries only the ID + one-liner label in context; when it needs the full
payload, it calls `expand_result(id)` which hits this store.

Why SQLite, here:
- Single-file, no server, part of the Python stdlib. Zero ops burden for
  the researcher.
- Survives across invocations. A session's results are available the next
  time Sift is launched in that directory.
- Queryable. Future session-state features (step 7) extend the same
  schema.
- Atomic writes. A crash mid-insert doesn't corrupt the store.

Things this deliberately does NOT handle at v0:
- Multi-session separation. All results for a cwd share one table.
  Sessions are step 7.
- Full-text search. Not needed until long analyses appear.
- Compression. Sanitized payloads are small JSON; no need yet.
- Raw-log archival. Step 4's executor will persist raw logs (in the
  researcher's TUI-visible channel only — NEVER to Claude). This store
  holds the sanitized payload and a pointer to the raw log on disk.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


# Where results land, relative to the cwd. The `.sift` prefix keeps
# the directory out of most project listings (and matches conventions
# like `.git`, `.venv`).
STORE_SUBDIR = ".sift"
DB_FILENAME = "results.db"


def _decode_source_datasets(value: Any) -> tuple[str, ...]:
    """Decode a migrated lineage column without making row reads fragile."""
    if not value:
        return ()
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return ()
    if not isinstance(decoded, (list, tuple)):
        return ()
    result: list[str] = []
    for item in decoded:
        if isinstance(item, str) and item and item not in result:
            result.append(item)
    return tuple(result)


def _decode_mapping(value: Any) -> dict[str, Any]:
    """Decode an optional JSON object without making legacy reads fragile."""
    if not value:
        return {}
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


@dataclass
class StoredResult:
    """One row of the results table, hydrated."""
    id: str
    label: str
    analysis_type: str
    sanitized_payload: dict[str, Any]
    language: str                        # "R" or "Stata"
    script_code: str                     # original source for audit
    transformations: list[str]           # what the sanitizer did
    raw_log_path: str | None             # filesystem pointer, not content
    created_at: str                      # ISO 8601 UTC
    # Groups results that came from the same submit_script call. NULL on
    # rows from before the multi-result wire format (one row per call).
    # Today's submit_script generates one script_run_id per invocation
    # and tags every row produced by that invocation with it.
    script_run_id: str | None = None
    # Visibility — populated when a rewind hides this row. NULL means
    # the row is visible to the model (default for every freshly-
    # inserted result). Audit code that opts into ``include_hidden=True``
    # sees these populated.
    hidden_at: str | None = None
    hidden_reason: str | None = None
    # The dataset the analysis ran against, as passed to
    # ``submit_script``'s ``source_dataset`` argument. NULL on rows
    # from before this column existed, and on any result produced
    # without a source dataset (e.g. a script that only manipulates
    # in-memory data). Local-only provenance: never sent to the
    # model, surfaced solely through the Evidence panel bridge
    # methods for the researcher's own reading.
    source_dataset: str | None = None
    # Complete declared lineage for analyses that read more than one
    # dataset. ``source_dataset`` remains the first item for backwards
    # compatibility with older callers and databases; new code should use
    # ``all_source_datasets`` so joins do not silently lose provenance.
    source_datasets: tuple[str, ...] = ()
    # Immutable reproducibility metadata captured at execution: exact dataset
    # hashes, runtime/package versions, random seed, workflow revision, and
    # primary/sensitivity role.  Kept separate from the sanitized numerical
    # payload so provenance cannot be mistaken for an estimated quantity.
    provenance: dict[str, Any] = field(default_factory=dict)
    # Correction lifecycle is metadata-only. Original payloads/code remain
    # immutable and auditable; a replacement points back to the row it fixes.
    lifecycle_status: str = "active"
    supersedes_result_id: str | None = None
    superseded_by: str | None = None
    correction_reason: str | None = None

    @property
    def all_source_datasets(self) -> tuple[str, ...]:
        """Return complete, de-duplicated dataset lineage.

        Legacy rows only have ``source_dataset``.  Folding it in here keeps
        every reader migration-safe and avoids dozens of subtly different
        fallback implementations across export, verification, and UI code.
        """
        values: list[str] = []
        for value in (*self.source_datasets, self.source_dataset):
            if isinstance(value, str) and value and value not in values:
                values.append(value)
        return tuple(values)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ResultStore:
    """Thin wrapper over a SQLite file.

    Methods are synchronous. Every public method acquires
    ``self._lock`` so the underlying SQLite connection is only ever
    touched from one Python thread at a time — required because the
    runner thread (submit_script → ``insert``, expand_result →
    ``get``) and the UI/bridge thread (rewind → ``hide_results_not_in``,
    ``unhide_results``, ``purge_script_code``; sidebar render →
    ``list_all``) share one cached ``ResultStore`` per cwd. The MCP
    tool serializes tool calls within one session, but it does NOT
    serialize against bridge-driven operations from the UI thread.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS results (
        id                TEXT PRIMARY KEY,
        label             TEXT NOT NULL,
        analysis_type     TEXT NOT NULL,
        sanitized_payload TEXT NOT NULL,  -- JSON
        language          TEXT NOT NULL,
        script_code       TEXT NOT NULL,
        transformations   TEXT NOT NULL,  -- JSON array
        raw_log_path      TEXT,
        created_at        TEXT NOT NULL,
        script_run_id     TEXT,            -- groups multi-result submit_script calls; NULL on legacy rows
        hidden_at         TEXT,            -- ISO 8601; NULL = visible to model. Set by hide_results_not_in
        hidden_reason     TEXT,            -- short tag, e.g. "rewind". NULL while hidden_at is NULL
        source_dataset    TEXT,             -- first source; backwards-compatible singular view
        source_datasets   TEXT,             -- JSON array containing complete declared lineage
        provenance        TEXT,             -- JSON execution/workflow metadata
        lifecycle_status  TEXT NOT NULL DEFAULT 'active',
        supersedes_result_id TEXT,
        superseded_by     TEXT,
        correction_reason TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_results_created_at ON results (created_at);
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        # ``db_path`` is ``<cwd>/.sift/results.db``; route directory
        # creation through the central helper so .sift gets the 0o700
        # mode that gates every Sift-owned file on shared filesystems
        # (HPC, NFS) where the user's home isn't 0o700.
        from sift.config import ensure_private_sift_dir
        ensure_private_sift_dir(self.db_path.parent.parent)
        # isolation_level=None → autocommit; we manage transactions with
        # explicit BEGIN/COMMIT blocks.
        #
        # check_same_thread=False: Sift's bridge runs in pywebview's
        # webview thread while tool calls (submit_script, list_results,
        # expand_result) run on the asyncio runner thread. Whichever
        # thread first calls ``get_store`` opens the connection; the
        # other thread reusing the cached store would otherwise get
        # ``ProgrammingError: SQLite objects created in a thread can
        # only be used in that same thread``.
        self._conn = sqlite3.connect(
            str(db_path),
            isolation_level=None,
            check_same_thread=False,
            timeout=10.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 10000")
        # ``PRAGMA secure_delete = ON`` zeroes freed pages on
        # DELETE / UPDATE that shrinks a value. The rewind path's
        # ``purge_script_code`` blanks ``script_code`` to drop any
        # credentials / PII the researcher pasted into a script;
        # without secure_delete, the previous bytes survive on the
        # SQLite freelist until the page is overwritten by an
        # unrelated insert, and a forensic tool reading raw pages
        # (undark / hexdump / sqlite3_analyzer / file carving on a
        # stolen-laptop or backup-acquisition scenario) can recover
        # the supposedly-purged content. ``VACUUM`` after every
        # purge would also work but rewrites the entire DB; the
        # PRAGMA is the lightweight equivalent, applied once and
        # honoured for every subsequent DELETE / UPDATE.
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.executescript(self.SCHEMA)
        # Per-store lock serializing every operation that touches
        # ``self._conn``. ``check_same_thread=False`` allows cross-
        # thread reuse but the sqlite3 module relies on the caller to
        # serialize statements — overlapping calls from the bridge
        # thread (rewind's ``hide_results_not_in`` /
        # ``unhide_results``) and the runner thread (submit_script's
        # ``insert``, expand_result's ``get``) can interleave inside
        # one transaction and trip ``ProgrammingError: recursive use
        # of cursors not allowed`` on the connection's shared cursor
        # state. The "single-writer-single-reader" property in the
        # class docstring is an INVARIANT of well-behaved callers,
        # not something the connection enforces; we make it concrete
        # by acquiring this lock around every public method. The
        # lock also closes the ``_next_id`` race below (count + 1
        # outside the txn could see the same N from two concurrent
        # inserters and collide on the same M#).
        self._lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        """Apply every additive migration as one rollback-capable unit."""
        cols = {row["name"] for row in self._conn.execute(
            "PRAGMA table_info(results)"
        ).fetchall()}
        with self._txn():
            self._apply_migrations(cols)

    def _apply_migrations(self, cols: set[str]) -> None:
        """Apply additive migrations to an existing DB.

        SQLite's ``CREATE TABLE IF NOT EXISTS`` skips the body when the
        table is already there, so a fresh column declared in SCHEMA
        won't reach a pre-existing DB on its own. Each migration step
        is idempotent: check ``PRAGMA table_info`` before issuing the
        ``ALTER``, so re-running on an already-migrated DB is a no-op.
        Indexes that depend on migrated columns are created here too,
        not in SCHEMA: the SCHEMA's CREATE INDEX runs before _migrate
        on legacy DBs and would fail referencing a column that doesn't
        exist yet.
        """
        if "script_run_id" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN script_run_id TEXT"
            )
        # Visibility columns added with the rewind feature. ``hidden_at``
        # NULL means the row is visible to the model (default for every
        # row inserted at submit_script time); a populated timestamp
        # means the row was hidden by a rewind operation. Hidden rows
        # remain in the database for audit but are filtered out of the
        # default ``list_all`` / ``get`` query paths so the model
        # doesn't see them in warm-start prefixes, ``list_results``,
        # ``list_results_global``, or ``expand_result``. Audit callers
        # opt back in via ``include_hidden=True``.
        if "hidden_at" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN hidden_at TEXT"
            )
        if "hidden_reason" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN hidden_reason TEXT"
            )
        if "source_dataset" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN source_dataset TEXT"
            )
        if "source_datasets" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN source_datasets TEXT"
            )
        if "provenance" not in cols:
            self._conn.execute("ALTER TABLE results ADD COLUMN provenance TEXT")
        if "lifecycle_status" not in cols:
            self._conn.execute(
                "ALTER TABLE results ADD COLUMN lifecycle_status TEXT NOT NULL DEFAULT 'active'"
            )
        if "supersedes_result_id" not in cols:
            self._conn.execute("ALTER TABLE results ADD COLUMN supersedes_result_id TEXT")
        if "superseded_by" not in cols:
            self._conn.execute("ALTER TABLE results ADD COLUMN superseded_by TEXT")
        if "correction_reason" not in cols:
            self._conn.execute("ALTER TABLE results ADD COLUMN correction_reason TEXT")
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_script_run_id "
            "ON results (script_run_id)"
        )
        # Partial index over visible rows only — list_all / get with the
        # default visibility filter benefits when the table accumulates
        # many hidden rows from repeated rewinds. Cheap to maintain
        # because the WHERE clause keeps it sparse.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_results_visible "
            "ON results (created_at) WHERE hidden_at IS NULL"
        )

    def integrity_report(self) -> dict[str, Any]:
        """Run SQLite's built-in consistency checks under the store lock."""
        with self._lock:
            try:
                quick = [row[0] for row in self._conn.execute(
                    "PRAGMA quick_check"
                ).fetchall()]
                foreign = self._conn.execute(
                    "PRAGMA foreign_key_check"
                ).fetchall()
                return {
                    "ok": quick == ["ok"] and not foreign,
                    "quick_check": quick[:20],
                    "foreign_key_issues": len(foreign),
                }
            except sqlite3.Error as exc:
                return {"ok": False, "error": type(exc).__name__}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ResultStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        """Last-resort cleanup for uncached, one-shot store handles.

        Cached interactive stores have an explicit session lifecycle via
        ``close_store``. A direct ``ResultStore`` created by an integration or
        extension can otherwise leave SQLite warning about an unclosed
        database at interpreter shutdown. Never let destructor-time module
        teardown raise.
        """
        conn = getattr(self, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    # -- Insert -------------------------------------------------------------

    def insert(
        self,
        *,
        label: str,
        analysis_type: str,
        sanitized_payload: dict[str, Any],
        language: str,
        script_code: str,
        transformations: list[str],
        raw_log_path: Path | None = None,
        script_run_id: str | None = None,
        source_dataset: str | None = None,
        source_datasets: tuple[str, ...] | list[str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> StoredResult:
        """Add a new result; return the hydrated row (including assigned ID)."""
        lineage: list[str] = []
        for value in (*(source_datasets or ()), source_dataset):
            if isinstance(value, str) and value and value not in lineage:
                lineage.append(value)
        primary_source = lineage[0] if lineage else None
        with self._lock:
            with self._txn():
                from sift.reliability import clock_safe_timestamp
                previous = self._conn.execute(
                    "SELECT created_at FROM results ORDER BY rowid DESC LIMIT 1"
                ).fetchone()
                now = clock_safe_timestamp(
                    str(previous["created_at"]) if previous is not None else None
                )
                # ID allocation inside the transaction so two concurrent
                # inserters can't both compute ``count + 1`` and race
                # to the same ``M#``. BEGIN IMMEDIATE makes one of them
                # wait for the other to commit, so the count this
                # reads reflects every committed row. The Python lock
                # above already serializes Python-side, but moving
                # the count inside the txn also closes the door on
                # any future caller that goes around the lock.
                result_id = self._next_id_locked()
                row = StoredResult(
                    id=result_id,
                    label=label,
                    analysis_type=analysis_type,
                    sanitized_payload=sanitized_payload,
                    language=language,
                    script_code=script_code,
                    transformations=list(transformations),
                    raw_log_path=str(raw_log_path) if raw_log_path else None,
                    created_at=now,
                    script_run_id=script_run_id,
                    source_dataset=primary_source,
                    source_datasets=tuple(lineage),
                    provenance=dict(provenance or {}),
                )
                self._conn.execute(
                    "INSERT INTO results (id, label, analysis_type, "
                    "sanitized_payload, language, script_code, transformations, "
                    "raw_log_path, created_at, script_run_id, source_dataset, "
                    "source_datasets, provenance) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row.id,
                        row.label,
                        row.analysis_type,
                        json.dumps(row.sanitized_payload, ensure_ascii=False),
                        row.language,
                        row.script_code,
                        json.dumps(row.transformations, ensure_ascii=False),
                        row.raw_log_path,
                        row.created_at,
                        row.script_run_id,
                        row.source_dataset,
                        json.dumps(row.source_datasets, ensure_ascii=False),
                        json.dumps(row.provenance, ensure_ascii=False),
                    ),
                )
            return row

    # -- Read ---------------------------------------------------------------

    def get(
        self, result_id: str, *, include_hidden: bool = False,
    ) -> StoredResult | None:
        """Fetch a stored row by id.

        Returns ``None`` for unknown ids AND for ids whose row has
        been hidden by a rewind, unless ``include_hidden=True`` is
        passed. Tool handlers (``expand_result``) call with the
        default so the model can't reach into rows that were
        invalidated by a rewind; audit / debug paths can opt in.
        """
        with self._lock:
            if include_hidden:
                cur = self._conn.execute(
                    "SELECT * FROM results WHERE id = ?", (result_id,)
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM results WHERE id = ? AND hidden_at IS NULL",
                    (result_id,),
                )
            row = cur.fetchone()
            if row is None:
                return None
            return self._hydrate(row)

    def list_all(self, *, include_hidden: bool = False) -> list[StoredResult]:
        """List rows in chronological-ascending order.

        Defaults to visible rows only — rows hidden by a rewind are
        filtered out so warm-start prefixes, ``list_results`` /
        ``list_results_global`` tool calls, and any other model-
        visible enumeration don't surface them. Audit code paths
        pass ``include_hidden=True`` to see the whole history.
        """
        with self._lock:
            if include_hidden:
                cur = self._conn.execute(
                    "SELECT * FROM results ORDER BY created_at ASC"
                )
            else:
                cur = self._conn.execute(
                    "SELECT * FROM results WHERE hidden_at IS NULL "
                    "ORDER BY created_at ASC"
                )
            return [self._hydrate(r) for r in cur.fetchall()]

    def hide_results_not_in(
        self, kept_ids: set[str], *, reason: str,
    ) -> list[str]:
        """Mark every currently-visible row whose id is NOT in
        ``kept_ids`` as hidden, with the given reason and the current
        timestamp. Returns the list of ids newly hidden.

        Used by the rewind path: after the chat history is truncated
        to a cut-point, the bridge collects the result_ids still
        referenced in the kept prefix and passes that set here. Every
        other visible row gets hidden in a single transaction so the
        model's view of the store stays consistent with the truncated
        chat.

        Already-hidden rows are left alone — their ``hidden_at`` and
        ``hidden_reason`` reflect the rewind that hid them; a second
        rewind shouldn't overwrite that with a fresh timestamp. A
        rewind that would hide nothing returns ``[]`` cleanly.

        The return value is a list (not just a count) so callers can
        roll back: if the broader rewind operation fails downstream
        (e.g., the chat-history truncate raises ``OSError``), the
        bridge passes this list to ``unhide_results`` to restore the
        rows. Without that, a failed rewind would leave the store
        partially mutated while the JS side reports "rewind failed".

        Implementation: read all currently-visible ids in Python,
        diff against ``kept_ids`` to compute the to-hide set, then
        UPDATE in batches sized below SQLite's 999-parameter limit.
        Doing the diff in Python avoids the temp-table dance a single
        ``id NOT IN (large list)`` would otherwise need; the visible-
        row count for a Sift session is in the low thousands at most,
        so the in-memory diff is cheap.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._txn():
                visible_rows = self._conn.execute(
                    "SELECT id FROM results WHERE hidden_at IS NULL"
                ).fetchall()
                visible_ids = [r["id"] for r in visible_rows]
                to_hide = [vid for vid in visible_ids if vid not in kept_ids]
                if not to_hide:
                    return []
                # SQLite's default parameter limit is 999. Batch at 500
                # to leave headroom for the two leading parameters
                # (``now``, ``reason``) and any future schema growth.
                BATCH = 500
                for i in range(0, len(to_hide), BATCH):
                    chunk = to_hide[i:i + BATCH]
                    placeholders = ",".join("?" * len(chunk))
                    # The only interpolated fragment is a string of ``?``
                    # placeholders derived from chunk length; all identifiers
                    # and values are static/parameterized.
                    self._conn.execute(
                        f"UPDATE results SET hidden_at = ?, "  # nosec B608
                        f"hidden_reason = ? "
                        f"WHERE hidden_at IS NULL AND id IN "
                        f"({placeholders})",
                        (now, reason, *chunk),
                    )
                return list(to_hide)

    def purge_script_code(self, ids: list[str]) -> int:
        """Blank the ``script_code`` column on the given rows.

        Called by the rewind path AFTER ``hide_results_not_in`` and
        the chat-history truncate have both succeeded — i.e. once the
        rewind has reached its commit point and ``unhide_results``
        rollback is no longer in play. Without this, a researcher who
        pasted a credential or PII into a script (``api_key = "sk-…"``
        in an ``.r`` file, an SSN string-literal in an exploratory
        block) leaves that text on disk in ``results.db`` indefinitely
        even though the model can no longer see the row — the row is
        hidden, but ``sqlite3`` queries against the file (or anyone
        who exfiltrates the file) still see the secret.

        Returns the number of rows whose ``script_code`` was actually
        replaced with the empty string (rows that already had empty
        ``script_code`` are no-ops). The audit/debug ``include_hidden``
        path keeps the row's other columns (label, payload,
        transformations, raw_log_path) intact — only the verbatim
        researcher-authored script text is dropped, since that's the
        only column whose contents come straight from a free-form
        researcher input.
        """
        if not ids:
            return 0
        with self._lock:
            with self._txn():
                BATCH = 500
                total = 0
                for i in range(0, len(ids), BATCH):
                    chunk = ids[i:i + BATCH]
                    placeholders = ",".join("?" * len(chunk))
                    # Placeholder count only; every id remains bound data.
                    cur = self._conn.execute(
                        f"UPDATE results SET script_code = '' "  # nosec B608
                        f"WHERE script_code != '' AND id IN "
                        f"({placeholders})",
                        tuple(chunk),
                    )
                    total += cur.rowcount or 0
                return total

    def unhide_results(self, ids: list[str]) -> int:
        """Clear ``hidden_at`` / ``hidden_reason`` on the given rows.

        Rollback path for ``hide_results_not_in``: when a rewind's
        downstream step fails (chat-history truncate raises), the
        bridge calls this with the ids the hide step just returned.
        Restoring exactly those ids — not "every hidden row" — keeps
        any unrelated prior rewinds intact.

        Returns the number of rows actually unhidden (rows already
        visible are no-ops; rows that don't exist also no-op). A
        best-effort rollback caller can ignore the count.
        """
        if not ids:
            return 0
        with self._lock:
            with self._txn():
                BATCH = 500
                total = 0
                for i in range(0, len(ids), BATCH):
                    chunk = ids[i:i + BATCH]
                    placeholders = ",".join("?" * len(chunk))
                    # Placeholder count only; every id remains bound data.
                    cur = self._conn.execute(
                        f"UPDATE results SET hidden_at = NULL, "  # nosec B608
                        f"hidden_reason = NULL "
                        f"WHERE hidden_at IS NOT NULL AND id IN "
                        f"({placeholders})",
                        tuple(chunk),
                    )
                    total += cur.rowcount or 0
                return total

    def list_by_script_run(self, script_run_id: str) -> list[StoredResult]:
        """All rows produced by one ``submit_script`` invocation, in
        emission order. Returns ``[]`` for unknown ids or for legacy
        rows where the field was never set.

        Orders by SQLite's implicit ``rowid``, which is monotone in
        insertion order regardless of clock resolution. Ordering by
        ``created_at`` alone risks ties on tight loops where multiple
        helpers fire within the same microsecond; falling back to
        ``id ASC`` lexically would then produce M1, M10, M11, ..., M2.
        """
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM results WHERE script_run_id = ? "
                "ORDER BY rowid ASC",
                (script_run_id,),
            )
            return [self._hydrate(r) for r in cur.fetchall()]

    def count(self) -> int:
        with self._lock:
            cur = self._conn.execute("SELECT COUNT(*) AS c FROM results")
            return int(cur.fetchone()["c"])

    def supersede_result(
        self, old_result_id: str, new_result_id: str, *, reason: str,
        correction: bool = False,
    ) -> dict[str, Any]:
        """Link an immutable replacement to the result it supersedes.

        Both rows must exist and be visible, the replacement must be newer,
        and a result may only be superseded once.  Payloads and code are never
        rewritten; lifecycle metadata makes corrections explicit without
        destroying the original evidence.
        """
        if old_result_id == new_result_id:
            raise ValueError("a result cannot supersede itself")
        clean_reason = str(reason or "").strip()
        if not clean_reason or len(clean_reason) > 1000:
            raise ValueError("a bounded correction reason is required")
        with self._lock:
            with self._txn():
                old = self._conn.execute(
                    "SELECT id, created_at, lifecycle_status, hidden_at FROM results WHERE id = ?",
                    (old_result_id,),
                ).fetchone()
                new = self._conn.execute(
                    "SELECT id, created_at, supersedes_result_id, hidden_at FROM results WHERE id = ?",
                    (new_result_id,),
                ).fetchone()
                if old is None or new is None or old["hidden_at"] or new["hidden_at"]:
                    raise ValueError("both results must exist and be visible")
                if old["lifecycle_status"] != "active":
                    raise ValueError("the original result is already superseded")
                if new["supersedes_result_id"]:
                    raise ValueError("the replacement already supersedes another result")
                if str(new["created_at"]) < str(old["created_at"]):
                    raise ValueError("the replacement must not predate the original")
                status = "corrected" if correction else "superseded"
                self._conn.execute(
                    "UPDATE results SET lifecycle_status = ?, superseded_by = ?, "
                    "correction_reason = ? WHERE id = ?",
                    (status, new_result_id, clean_reason, old_result_id),
                )
                self._conn.execute(
                    "UPDATE results SET supersedes_result_id = ?, correction_reason = ? "
                    "WHERE id = ?",
                    (old_result_id, clean_reason, new_result_id),
                )
        audit: dict[str, Any]
        try:
            import hashlib
            from sift.reproducibility import append_audit_event

            event = append_audit_event(
                self.db_path.parent.parent,
                "result_superseded",
                {
                    "result_id": old_result_id,
                    "superseded_by": new_result_id,
                    "reason_sha256": hashlib.sha256(
                        clean_reason.encode("utf-8")
                    ).hexdigest(),
                    "status": status,
                },
            )
            audit = {"status": "recorded", "sequence": event["sequence"]}
        except Exception as exc:  # noqa: BLE001 — correction already committed
            audit = {"status": "unavailable", "reason": type(exc).__name__}
        return {
            "old_result_id": old_result_id, "new_result_id": new_result_id,
            "old_status": status, "reason": clean_reason,
            "reproducibility_audit": audit,
        }

    # -- Internals ----------------------------------------------------------

    def _next_id_locked(self) -> str:
        """Assign the next sequential ID of the form `M1`, `M2`, ...

        We compute it from the current row count rather than using SQLite's
        autoincrement because the IDs are human-facing (Claude types them)
        and should be readable / predictable. MUST be called with
        ``self._lock`` held AND inside a ``BEGIN IMMEDIATE`` transaction —
        the count read and the subsequent insert have to be atomic so
        two concurrent inserters don't both compute the same M#.
        """
        cur = self._conn.execute("SELECT COUNT(*) AS c FROM results")
        return f"M{int(cur.fetchone()['c']) + 1}"

    def _hydrate(self, row: sqlite3.Row) -> StoredResult:
        keys = row.keys() if hasattr(row, "keys") else None

        def _opt(name: str) -> Any:
            # Tolerate legacy schemas where a column doesn't exist yet
            # (the migration runs at __init__ time, but tests that
            # construct rows from raw fixtures may skip migration).
            return row[name] if (keys is None or name in keys) else None

        return StoredResult(
            id=row["id"],
            label=row["label"],
            analysis_type=row["analysis_type"],
            sanitized_payload=json.loads(row["sanitized_payload"]),
            language=row["language"],
            script_code=row["script_code"],
            transformations=json.loads(row["transformations"]),
            raw_log_path=row["raw_log_path"],
            created_at=row["created_at"],
            script_run_id=_opt("script_run_id"),
            hidden_at=_opt("hidden_at"),
            hidden_reason=_opt("hidden_reason"),
            source_dataset=_opt("source_dataset"),
            source_datasets=_decode_source_datasets(_opt("source_datasets")),
            provenance=_decode_mapping(_opt("provenance")),
            lifecycle_status=_opt("lifecycle_status") or "active",
            supersedes_result_id=_opt("supersedes_result_id"),
            superseded_by=_opt("superseded_by"),
            correction_reason=_opt("correction_reason"),
        )

    # Minimal transaction helper — we don't have complex write patterns yet.
    def _txn(self) -> "_Txn":
        return _Txn(self._conn)


class _Txn:
    """Tiny context manager for atomic writes.

    sqlite3's default transactions are implicit and surprising; this makes
    it explicit. BEGIN IMMEDIATE avoids the deferred-lock upgrade that
    can cause busy errors under contention.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __enter__(self) -> "_Txn":
        self._conn.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")


# ---------------------------------------------------------------------------
# Convenience: per-cwd store cache
# ---------------------------------------------------------------------------
#
# Earlier versions cached exactly one ``ResultStore`` instance and
# returned it regardless of the requested cwd — so after a session
# switch, tool calls against the new cwd were writing to the OLD
# session's ``results.db``. Catastrophic: Project A could see
# Project B's stored sanitized results by calling ``list_results``
# in the same app process.
#
# We now key the cache by the resolved cwd. Switching sessions gets
# a fresh store that points at the new session's DB; re-opening the
# same session reuses the existing one (SQLite connection open is
# cheap but not free — reuse is a modest win, and the file handle
# limit is finite).

_stores: dict[Path, ResultStore] = {}
_STORES_LOCK = threading.RLock()


def get_store(cwd: Path) -> ResultStore:
    """Return the store pinned to ``cwd`` — NOT a process-wide
    singleton.

    Resolves the cwd before keying the cache so two paths that
    normalize to the same directory (one with symlinks or ``./``,
    one without) share a store rather than racing on the same
    sqlite file through two different handles.
    """
    key = cwd.resolve()
    with _STORES_LOCK:
        existing = _stores.get(key)
        if existing is not None:
            return existing
        store = ResultStore(key / STORE_SUBDIR / DB_FILENAME)
        _stores[key] = store
        return store


def open_store_uncached(cwd: Path) -> ResultStore:
    """Open a fresh store for ``cwd`` without inserting it into the
    process-wide cache.

    Used by one-shot scans across many sessions
    (``list_results_global``, the cross-session recall path) so a
    broad listing doesn't permanently retain a SQLite connection per
    session it touched. Caller MUST ``close()`` the returned store
    when done — otherwise the file descriptor leaks until the
    process exits.

    If a cached store already exists for the same cwd (e.g. the
    active session's own store, opened earlier by a model-facing
    tool), the cached one is returned instead so we don't double-
    open the same DB. The "don't pin a fresh handle" property only
    needs to hold for sessions that aren't already cached.
    """
    key = cwd.resolve()
    with _STORES_LOCK:
        existing = _stores.get(key)
    if existing is not None:
        return existing
    return ResultStore(key / STORE_SUBDIR / DB_FILENAME)


def close_store(cwd: Path) -> None:
    """Close and drop the cached store for ``cwd``.

    Called from the UI bridge when the session switches so the new
    cwd's store isn't shadowed by a stale handle. Safe to call when
    no store exists for this cwd — it's a no-op in that case.
    """
    key = cwd.resolve()
    with _STORES_LOCK:
        existing = _stores.pop(key, None)
    if existing is not None:
        try:
            existing.close()
        except Exception:  # noqa: BLE001 — closing a dead handle shouldn't crash the switch
            pass


def reset_store_for_tests() -> None:
    """Test-only hook: close and drop every cached store. Used to
    clear process-wide state between tests so one test's store
    can't leak into the next."""
    global _stores
    with _STORES_LOCK:
        stores = list(_stores.values())
        _stores = {}
    for store in stores:
        try:
            store.close()
        except Exception:  # noqa: BLE001
            pass
