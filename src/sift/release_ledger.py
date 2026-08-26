"""Sift — release ledger: disclosure accounting for the privacy boundary.

Every tool response Sift hands to the frontier model is, by
definition, a disclosure across the privacy boundary — that is the
one channel through which information about the researcher's data
leaves the machine (plus the researcher's own typed messages, which
are theirs to make). The sanitizer bounds what each *individual*
response can contain; nothing previously bounded what the responses
*add up to*. Many individually-safe releases can compose into more
than any one of them reveals ("twenty questions" inference).

This module is the accounting half of that problem, not the solution
half: an append-only, hash-chained, per-session record of every
model-crossing tool response. It makes the cumulative disclosure
surface *auditable* — a researcher (or a future composition engine)
can enumerate exactly what crossed, when, produced by which
operation, over which variables, at what sample size. It does NOT
implement differential-privacy composition or automatic query
refusal, and no Sift surface claims it does.

Design:

- One JSONL file per session: ``<cwd>/.sift/release_ledger.jsonl``
  (the ``.sift`` metadata dir is shared with the result store).
- Each record carries ``prev`` = the SHA-256 of the previous record's
  serialized line (chain root: 64 zeros) and ``hash`` = the SHA-256
  of the record's own canonical serialization minus the ``hash``
  field. ``verify_chain`` walks the file and reports the first break.
  This is tamper-*evidence*, not tamper-*proofing* — the researcher
  owns the disk; the chain guards against silent corruption and
  casual edits, and gives a future signed/exported ledger a stable
  format to build on.
- Records hold **metadata about the release, never the release
  itself**: tool name, a small allowlisted subset of the
  model-authored arguments, small scalar facts parsed from the
  (already-sanitized) response (status, analysis type, n, dataset),
  and the SHA-256 of the full response text. The full sanitized
  payloads already persist in the result store; duplicating them
  here would add surface without adding accountability.
- Recording NEVER raises into the tool path: a broken ledger must not take
  analysis down with it. A failed append is written to a separate health
  marker when possible and acknowledged by the next successful record.
  ``verify_chain`` distinguishes an unresolved accounting gap from a hash
  mismatch; a missing record cannot, by itself, break a hash chain.

Wired in exactly one place: the ``tool()`` decorator in
``sift.tools`` wraps every registered handler, so every current and
future tool is recorded automatically — a new tool cannot forget to
be accounted. Model-output plot attachments (the other crossing,
which travels outside tool responses) are recorded by the runner via
``record_plot_release``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from sift.file_lock import exclusive_file_lock
from sift.secure_file import open_regular_no_follow

LEDGER_FILENAME = "release_ledger.jsonl"
LEDGER_HEALTH_FILENAME = "release_ledger.health.json"
PRIVACY_TRANSACTION_LOCK_FILENAME = "privacy_accounting.lock"
_CHAIN_ROOT = "0" * 64

# Model-authored argument keys worth keeping in the record. Everything
# the model sends has already crossed the boundary outward-facing, so
# none of this is disclosive; the allowlist just keeps records small
# and stable (no dumping of full script bodies into the ledger —
# scripts persist in the result store / run dirs already).
_ARG_KEYS = (
    "dataset", "depth", "variable", "variable2", "request_type",
    "group_by", "language", "label", "result_id", "view", "query",
    "limit", "filename", "path", "session_path",
)

# Small scalar facts lifted from the (already sanitized) response
# JSON, when present at the top level or per-result. "epsilon" backs
# differential_privacy.py's cumulative-spend accounting for the
# noisy_count request type (request_data's handler puts a top-level
# "epsilon" key in its response only for a granted noisy_count call
# — see tools.py); every other key here predates that feature.
_RESPONSE_KEYS = ("status", "analysis_type", "n", "source_dataset", "epsilon")


def ledger_path(cwd: Path) -> Path:
    return Path(cwd) / ".sift" / LEDGER_FILENAME


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


# Process-lifetime cache of each ledger's chain tip: {path -> (size,
# tip_hash)}. Without this, every single ``record_release`` call read
# the ENTIRE ledger file just to find the hash of its last line — a
# short first-analysis session pays that once or twice, but a long
# research session can rack up hundreds of tool calls, and each write
# was re-reading everything written before it (quadratic total I/O
# in call count). The file size is a cheap staleness check: this
# process is normally the ledger's only writer, so the size we saw
# right after our last write should still match; if it doesn't
# (another process appended, or the ledger was touched between
# sessions), we fall back to a real read rather than trust a tip that
# might not be the true chain end. This is an optimization only —
# it changes no on-disk format and no chain semantics; a cold cache
# (fresh process) behaves exactly as before.
_TIP_CACHE: dict[str, tuple[int, str]] = {}

# ``record_release`` can be reached from more than one UI/runner thread.
# The file lock below covers separate processes on POSIX, while this
# process-local lock also covers threads (``flock`` locks are associated
# with a process and therefore are not, by themselves, a thread mutex).
_APPEND_LOCK = threading.RLock()


@dataclass(frozen=True)
class LedgerSnapshot:
    """One point-in-time view of ledger records and export artifacts."""

    records: tuple[dict[str, Any], ...]
    ledger_bytes: bytes | None
    health_bytes: bytes | None
    integrity_ok: bool
    record_count: int
    detail: str


def _health_path(cwd: Path) -> Path:
    return Path(cwd) / ".sift" / LEDGER_HEALTH_FILENAME


@contextmanager
def privacy_accounting_transaction(cwd: Path) -> Iterator[None]:
    """Serialize a privacy-budget decision through its ledger append.

    Formal composition is a check-and-record transaction: if two workers
    both read the same remaining epsilon before either response is recorded,
    both can otherwise grant and overspend the cap.  The ordinary ledger lock
    only protects individual appends, so callers that make a budget decision
    must hold this distinct lock from the read through the corresponding
    append.  A separate lock avoids coupling ordinary, non-DP ledger writes to
    a potentially slower bounded-data computation.

    Failure to create or acquire the lock intentionally propagates.  A caller
    enforcing a formal privacy budget must deny the release when it cannot
    establish the transaction; silently proceeding would turn an unavailable
    accounting store into a budget reset.
    """
    root = Path(cwd)
    if not root.is_dir():
        raise OSError("privacy accounting workspace is unavailable")
    from sift.config import ensure_private_sift_dir

    ensure_private_sift_dir(root)
    lock_path = root / ".sift" / PRIVACY_TRANSACTION_LOCK_FILENAME
    with exclusive_file_lock(lock_path):
        yield


def _read_health(cwd: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(_health_path(cwd).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if (
        isinstance(value, dict)
        and isinstance(value.get("id"), str)
        and isinstance(value.get("count"), int)
        and value["count"] > 0
    ):
        return value
    return None


def _note_recording_failure(cwd: Path, error: BaseException) -> None:
    """Best-effort durable marker for an append that could not be recorded."""
    if not cwd.is_dir():
        return
    health = _health_path(cwd)
    lock_path = ledger_path(cwd).with_suffix(".jsonl.lock")
    try:
        health.parent.mkdir(parents=True, exist_ok=True)
        with _APPEND_LOCK, exclusive_file_lock(lock_path):
            current = _read_health(cwd)
            now = _now_iso()
            marker = {
                "v": 1,
                "id": current["id"] if current else uuid.uuid4().hex,
                "count": int(current["count"]) + 1 if current else 1,
                "first_failure": (
                    current.get("first_failure", now) if current else now
                ),
                "last_failure": now,
                "last_error_type": type(error).__name__,
            }
            fd, tmp_name = tempfile.mkstemp(
                prefix=".release-ledger-health-", suffix=".tmp",
                dir=health.parent,
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(
                        marker, handle, sort_keys=True, separators=(",", ":"),
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, health)
            except Exception:
                Path(tmp_name).unlink(missing_ok=True)
                raise
    except Exception:  # noqa: BLE001 - health reporting must never break tools
        return


def dataset_key(raw: Any) -> str:
    """Return the stable accounting key for a dataset identifier.

    Ledger records preserve the raw model-authored path for auditability,
    but every comparison must collapse harmless spelling variants to the
    same key. Normalise both slash conventions so a ledger copied between
    Windows and POSIX does not reset privacy accounting. This is lexical
    only: historical entries need not still exist on disk.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    normalized = raw.replace("\\", "/")
    return PurePosixPath(normalized).name or normalized


def dataset_identity(cwd: Path, raw: Any) -> str:
    """Canonical session-relative identity for a live dataset path.

    Basename-only accounting is deliberately compatible with old ledgers but
    cannot distinguish ``cohort_a/data.csv`` from ``cohort_b/data.csv`` and
    can be reset by accessing one file through a differently named symlink.
    New records therefore carry the resolved path relative to the session.
    Resolving follows symlinks while the containment check prevents an
    identity from blessing an out-of-session target.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        root = Path(cwd).resolve()
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved = candidate.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            return ""
        return resolved.relative_to(root).as_posix()
    except (OSError, RuntimeError, ValueError):
        return ""


def same_dataset(
    record_raw: Any,
    query_raw: Any,
    *,
    cwd: Path | None = None,
    record_identity: Any = None,
) -> bool:
    """Match a ledger dataset to a query without alias resets.

    New records use their canonical identity when a workspace is available.
    Legacy records without that field retain basename matching, preserving
    conservative accounting across upgrades and copied historical ledgers.
    """
    if cwd is not None and isinstance(record_identity, str) and record_identity:
        query_identity = dataset_identity(cwd, query_raw)
        if query_identity:
            return record_identity == query_identity
    return dataset_key(record_raw) == dataset_key(query_raw)


def _annotate_fact_dataset_identities(
    cwd: Path, facts: dict[str, Any],
) -> None:
    """Add canonical identities beside source-dataset provenance fields."""
    source = facts.get("source_dataset")
    identity = dataset_identity(cwd, source)
    if identity:
        facts["source_dataset_identity"] = identity
    sources = facts.get("source_datasets")
    if isinstance(sources, list):
        identities = [dataset_identity(cwd, value) for value in sources]
        if any(identities):
            facts["source_dataset_identities"] = identities
    results = facts.get("results")
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict):
                _annotate_fact_dataset_identities(cwd, item)


def _last_line_hash(path: Path) -> str:
    """SHA-256 of the last non-empty line of the ledger (chain tip).

    Cached per path against the file's size — see ``_TIP_CACHE``.
    """
    key = str(path)
    try:
        size = path.stat().st_size
    except OSError:
        _TIP_CACHE.pop(key, None)
        return _CHAIN_ROOT
    cached = _TIP_CACHE.get(key)
    if cached is not None and cached[0] == size:
        return cached[1]
    try:
        raw = path.read_bytes()
    except OSError:
        _TIP_CACHE.pop(key, None)
        return _CHAIN_ROOT
    lines = [ln for ln in raw.split(b"\n") if ln.strip()]
    if not lines:
        _TIP_CACHE.pop(key, None)
        return _CHAIN_ROOT
    tip = hashlib.sha256(lines[-1]).hexdigest()
    _TIP_CACHE[key] = (size, tip)
    return tip


def _last_timestamp(path: Path) -> str | None:
    try:
        lines = [line for line in path.read_bytes().split(b"\n") if line.strip()]
        if not lines:
            return None
        row = json.loads(lines[-1])
        value = row.get("ts") if isinstance(row, dict) else None
        return value if isinstance(value, str) else None
    except (OSError, ValueError, TypeError):
        return None


def _response_text(response: Any) -> str:
    """Best-effort extraction of the MCP text body from a tool response."""
    try:
        blocks = response.get("content") or []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                return str(b.get("text") or "")
    except AttributeError:
        pass
    return ""


def response_status(response: Any) -> str | None:
    """Return a structured tool response's top-level status, if present."""
    try:
        payload = json.loads(_response_text(response))
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    value = payload.get("status")
    return value if isinstance(value, str) else None


def _response_image_bytes(response: Any) -> list[bytes]:
    """Best-effort extraction of raw (base64-decoded) bytes from every
    ``type: "image"`` content block in a tool response.

    ``read_attached_file`` returns image bytes alongside a compact text
    descriptor. Both representations must contribute to the ledger hash and
    byte count. The response shape is ``content: [{"type": "image", "data":
    <base64>, ...}, {"type": "text", "text": <small JSON
    descriptor>}]``. Mirrors ``_response_text``'s
    error-tolerant shape: a malformed response, an unexpected
    content-block shape, or non-base64 ``data`` is skipped rather
    than raised — ledger recording must never crash the tool call it
    accounts for.
    """
    out: list[bytes] = []
    try:
        blocks = response.get("content") or []
    except AttributeError:
        return out
    for b in blocks:
        if not (isinstance(b, dict) and b.get("type") == "image"):
            continue
        data = b.get("data")
        if not isinstance(data, str) or not data:
            continue
        try:
            out.append(base64.b64decode(data, validate=False))
        except (ValueError, TypeError):
            continue
    return out


def _facts_from_response(text: str) -> dict[str, Any]:
    """Lift a few small scalar facts from a sanitized response body.

    Purely best-effort: a non-JSON body (or an unexpected shape)
    yields an empty dict. Values are size-capped so a pathological
    response can't bloat the ledger.
    """
    facts: dict[str, Any] = {}
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return facts
    if not isinstance(payload, dict):
        return facts

    def _lift(src: dict[str, Any], dst: dict[str, Any]) -> None:
        for key in _RESPONSE_KEYS:
            val = src.get(key)
            if isinstance(val, (int, float)) or (
                isinstance(val, str) and len(val) <= 200
            ):
                dst[key] = val
        sources = src.get("source_datasets")
        if isinstance(sources, list):
            clean = [
                value for value in sources[:16]
                if isinstance(value, str) and 0 < len(value) <= 200
            ]
            if clean:
                dst["source_datasets"] = clean

    _lift(payload, facts)
    results = payload.get("results")
    if isinstance(results, list) and results:
        per: list[dict[str, Any]] = []
        for entry in results[:24]:
            if not isinstance(entry, dict):
                continue
            item: dict[str, Any] = {}
            _lift(entry, item)
            inner = entry.get("payload")
            if isinstance(inner, dict):
                _lift(inner, item)
            if item:
                per.append(item)
        if per:
            facts["results"] = per
    return facts


def record_release(
    cwd: Path | None,
    *,
    kind: str,
    tool: str,
    args: dict[str, Any] | None = None,
    response: Any = None,
    extra: dict[str, Any] | None = None,
) -> bool:
    """Append one release record without raising; report whether it landed."""
    resolved_cwd: Path | None = None
    try:
        if cwd is None:
            return False
        cwd = Path(cwd)
        resolved_cwd = cwd
        if not cwd.is_dir():
            return False
        from sift.config import ensure_private_sift_dir

        ensure_private_sift_dir(cwd)
        text = _response_text(response) if response is not None else ""
        image_chunks = (
            _response_image_bytes(response) if response is not None else []
        )
        record: dict[str, Any] = {
            "v": 1,
            "ts": _now_iso(),
            "kind": kind,
            "tool": tool,
        }
        if args:
            kept: dict[str, Any] = {}
            for key in _ARG_KEYS:
                val = args.get(key)
                if isinstance(val, (int, float)):
                    kept[key] = val
                elif isinstance(val, str) and val:
                    kept[key] = val[:300]
            sources = args.get("source_datasets")
            if isinstance(sources, list):
                clean = [
                    value[:300] for value in sources[:16]
                    if isinstance(value, str) and value
                ]
                if clean:
                    kept["source_datasets"] = clean
            if kept:
                identity = dataset_identity(cwd, kept.get("dataset"))
                if identity:
                    kept["dataset_identity"] = identity
                record["args"] = kept
        if text or image_chunks:
            # Hash/size the FULL disclosed payload -- text plus any
            # image bytes -- not just the text block. A response
            # carrying only images (no text descriptor) is possible
            # in principle even though today's only image-emitting
            # tool always pairs one with a text descriptor; this
            # covers that shape too rather than assuming the current
            # callers are the only ones that will ever exist.
            digest = hashlib.sha256()
            text_bytes = text.encode("utf-8")
            digest.update(text_bytes)
            total_bytes = len(text_bytes)
            for chunk in image_chunks:
                digest.update(chunk)
                total_bytes += len(chunk)
            record["response_sha256"] = digest.hexdigest()
            record["response_bytes"] = total_bytes
            if text:
                facts = _facts_from_response(text)
                if facts:
                    _annotate_fact_dataset_identities(cwd, facts)
                    record["facts"] = facts
        if extra:
            record["extra"] = extra
        path = ledger_path(cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Computing ``prev`` and appending must be one critical section.
        # Otherwise two processes can observe the same tip and append
        # sibling records, breaking the chain despite no tampering.
        lock_path = path.with_suffix(path.suffix + ".lock")
        with _APPEND_LOCK, exclusive_file_lock(lock_path):
            # Resolve the current chain state before opening the destination
            # with its no-follow descriptor.  Windows deliberately gives that
            # writable handle no sharing rights; trying to re-open `path`
            # afterward makes a cold/stale tip-cache lookup fail with a
            # sharing violation and incorrectly fall back to the chain root.
            # The separate cooperative lock remains held throughout, so every
            # Sift writer still observes and appends to one serialized tip.
            last_timestamp = _last_timestamp(path)
            previous_hash = _last_line_hash(path)
            descriptor = open_regular_no_follow(
                path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600,
            )
            with os.fdopen(descriptor, "a+b") as fh:
                from sift.reliability import clock_safe_timestamp
                record["ts"] = clock_safe_timestamp(last_timestamp)
                pending_gap = _read_health(cwd)
                if pending_gap is not None:
                    record["accounting_gap_before"] = {
                        "id": pending_gap["id"],
                        "count": pending_gap["count"],
                        "first_failure": pending_gap.get("first_failure"),
                        "last_failure": pending_gap.get("last_failure"),
                    }
                record["prev"] = previous_hash
                record["hash"] = hashlib.sha256(
                    _canonical(record).encode("utf-8")).hexdigest()
                line_no_newline = _canonical(record).encode("utf-8")
                fh.seek(0, 2)
                fh.write(line_no_newline + b"\n")
                fh.flush()
                # fstat while locked prevents another writer changing
                # the size between append and cache update.
                try:
                    tip = hashlib.sha256(line_no_newline).hexdigest()
                    _TIP_CACHE[str(path)] = (
                        os.fstat(fh.fileno()).st_size, tip,
                    )
                except OSError:
                    _TIP_CACHE.pop(str(path), None)
            if pending_gap is not None:
                try:
                    _health_path(cwd).unlink(missing_ok=True)
                except OSError:
                    # The durable ledger entry already acknowledges this gap;
                    # verification de-duplicates a stale marker by id.
                    pass
        return True
    except Exception as error:  # noqa: BLE001 — never break analysis
        if resolved_cwd is not None:
            _note_recording_failure(resolved_cwd, error)
        return False


def record_plot_release(cwd: Path | None, *, filename: str, kind: str,
                        byte_size: int | None = None) -> None:
    """Record a model-output plot crossing to the model as vision input."""
    extra: dict[str, Any] = {"plot_kind": kind, "filename": filename}
    if byte_size is not None:
        extra["bytes"] = byte_size
    record_release(cwd, kind="plot_vision", tool="(vision attachment)",
                   extra=extra)


def read_ledger(cwd: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Return ledger records, oldest first. Bad lines are skipped."""
    path = ledger_path(Path(cwd))
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    if limit is not None and limit >= 0:
        records = records[-limit:]
    return records


def verify_chain(cwd: Path) -> tuple[bool, int, str]:
    """Walk the ledger and verify the hash chain.

    Returns ``(ok, n_records, detail)``. ``ok`` is True for an empty
    or fully-consistent ledger; on a break, ``detail`` names the
    first inconsistent record index (0-based).
    """
    path = ledger_path(Path(cwd))
    try:
        raw = path.read_bytes()
    except OSError:
        metadata_dir = Path(cwd) / ".sift"
        if metadata_dir.exists() and not metadata_dir.is_dir():
            return False, 0, "ledger metadata path is not a directory"
        pending = _read_health(Path(cwd))
        if pending is not None:
            return (
                False, 0,
                f"no ledger file; accounting gap: {pending['count']} release(s) could not be recorded",
            )
        if _health_path(Path(cwd)).exists():
            return False, 0, "no ledger file; invalid ledger health marker"
        return True, 0, "no ledger file"
    lines = [ln for ln in raw.split(b"\n") if ln.strip()]
    prev_hash = _CHAIN_ROOT
    acknowledged_gap_ids: set[str] = set()
    gap_count = 0
    for i, line in enumerate(lines):
        try:
            rec = json.loads(line)
        except ValueError:
            return False, len(lines), f"record {i}: unparseable line"
        claimed = rec.get("hash")
        if rec.get("prev") != prev_hash:
            return False, len(lines), f"record {i}: prev-hash mismatch"
        body = {k: v for k, v in rec.items() if k != "hash"}
        expect = hashlib.sha256(
            _canonical(body).encode("utf-8")).hexdigest()
        if claimed != expect:
            return False, len(lines), f"record {i}: self-hash mismatch"
        gap = rec.get("accounting_gap_before")
        if isinstance(gap, dict):
            gap_id = gap.get("id")
            count = gap.get("count")
            if isinstance(gap_id, str) and isinstance(count, int) and count > 0:
                acknowledged_gap_ids.add(gap_id)
                gap_count += count
        prev_hash = hashlib.sha256(line).hexdigest()
    pending = _read_health(Path(cwd))
    if pending is not None and pending["id"] not in acknowledged_gap_ids:
        gap_count += int(pending["count"])
    elif pending is None and _health_path(Path(cwd)).exists():
        return False, len(lines), "hash chain valid; invalid ledger health marker"
    if gap_count:
        return (
            False,
            len(lines),
            f"hash chain valid; accounting gap: {gap_count} release(s) could not be recorded",
        )
    return True, len(lines), "ok"


def verified_ledger_snapshot(
    cwd: Path,
) -> tuple[list[dict[str, Any]], bool, str]:
    """Read one stable, integrity-checked accounting snapshot.

    The verification and parse happen under the same append lock, preventing
    a concurrent writer from changing the ledger between the two operations.
    Consumers that enforce privacy budgets should use this instead of a
    separate ``verify_chain`` / ``read_ledger`` pair.  It never raises; an
    unavailable lock or store is returned as an untrusted empty snapshot.
    """
    snapshot = ledger_snapshot(cwd)
    records = list(snapshot.records) if snapshot.integrity_ok else []
    return records, snapshot.integrity_ok, snapshot.detail


def ledger_snapshot(cwd: Path) -> LedgerSnapshot:
    """Capture records, exact artifact bytes, and integrity atomically.

    Exporters must not verify one version of the ledger and then copy a newer
    version.  This snapshot holds the append lock while collecting both the
    verdict and exact bytes, so every derived report can describe precisely
    the artifact placed in the replication package.  Invalid ledgers retain
    best-effort parsed records for forensic display, while formal consumers
    use ``verified_ledger_snapshot`` and receive none of those untrusted rows.
    """
    root = Path(cwd)
    if not root.is_dir():
        return LedgerSnapshot(
            (), None, None, False, 0,
            "privacy accounting workspace is unavailable",
        )
    metadata_dir = root / ".sift"
    if metadata_dir.exists() and not metadata_dir.is_dir():
        return LedgerSnapshot(
            (), None, None, False, 0,
            "ledger metadata path is not a directory",
        )
    if not metadata_dir.exists():
        return LedgerSnapshot((), None, None, True, 0, "no ledger file")
    path = ledger_path(root)
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with _APPEND_LOCK, exclusive_file_lock(lock_path):
            ok, count, detail = verify_chain(root)
            try:
                ledger_bytes = path.read_bytes()
            except OSError:
                ledger_bytes = None
            health_path = _health_path(root)
            try:
                health_bytes = health_path.read_bytes()
            except OSError:
                health_bytes = None
            records = tuple(read_ledger(root))
            return LedgerSnapshot(
                records, ledger_bytes, health_bytes, ok, count, detail,
            )
    except Exception as error:  # noqa: BLE001 - status API must not raise
        return LedgerSnapshot(
            (), None, None, False, 0,
            f"privacy accounting snapshot failed ({type(error).__name__})",
        )
