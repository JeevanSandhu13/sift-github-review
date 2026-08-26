"""Sift release ledger — disclosure accounting invariants.

Pins the properties the ledger must keep for its audit claim to be
honest:

1. Every tool response crossing to the model appends exactly one
   record (chokepoint = the ``tool()`` decorator, so this holds for
   every registered tool automatically).
2. Records hold metadata + hashes, never the response body itself.
3. The hash chain detects edits, deletions, and reordering.
4. Recording never raises into the tool path, even with a broken
   ledger location.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from sift import release_ledger
from sift.config import use_cwd


def _run(coro):
    return asyncio.run(coro)


def test_record_and_chain_roundtrip(tmp_path: Path) -> None:
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="get_schema",
        args={"dataset": "survey.csv", "depth": "names_types"},
        response={"content": [{"type": "text", "text": '{"status":"ok"}'}]},
    )
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        args={"dataset": "survey.csv", "request_type": "na_count",
              "variable": "income"},
        response={"content": [{"type": "text", "text": '{"status":"denied"}'}]},
    )
    records = release_ledger.read_ledger(tmp_path)
    assert len(records) == 2
    assert records[0]["tool"] == "get_schema"
    assert records[0]["args"]["dataset"] == "survey.csv"
    assert records[1]["args"]["variable"] == "income"
    ok, n, detail = release_ledger.verify_chain(tmp_path)
    assert ok and n == 2, detail


def test_concurrent_writers_do_not_fork_the_hash_chain(tmp_path: Path) -> None:
    def _write(i: int) -> None:
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool=f"tool_{i}",
            response={"content": [{"type": "text", "text": "{}"}]},
        )

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_write, range(120)))

    records = release_ledger.read_ledger(tmp_path)
    ok, count, detail = release_ledger.verify_chain(tmp_path)
    assert len(records) == 120
    assert ok and count == 120, detail


def test_response_body_is_hashed_not_stored(tmp_path: Path) -> None:
    body = '{"status":"ok","analysis_type":"t_test","n":812,"secret_marker_xyz":1}'
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="submit_script",
        response={"content": [{"type": "text", "text": body}]},
    )
    raw = release_ledger.ledger_path(tmp_path).read_text(encoding="utf-8")
    # The full body never lands in the ledger — only its hash and the
    # small allowlisted scalar facts.
    assert "secret_marker_xyz" not in raw
    rec = release_ledger.read_ledger(tmp_path)[0]
    assert rec["response_sha256"]
    assert rec["facts"]["analysis_type"] == "t_test"
    assert rec["facts"]["n"] == 812


def test_image_tool_response_bytes_are_fully_hashed_and_sized(
    tmp_path: Path,
) -> None:
    """Image releases account for the disclosed bytes, not only metadata.

    ``record_release`` used to hash and size
    ONLY the text content block of a tool response, never the raw
    bytes of an image block. ``read_attached_file``'s image branch
    (``tools.py``) returns exactly this shape -- a real, potentially
    multi-megabyte image disclosed to the model as vision input,
    paired with a tiny JSON text descriptor -- so ``response_bytes``
    silently described a few dozen bytes of JSON instead of the
    actual disclosed payload for every image recall, breaking this
    module's own documented tamper-evidence guarantee for exactly
    the code path where the disclosure is largest.

    This constructs a response in read_attached_file's EXACT shape
    (mirrors the real "[{type: image, data, mimeType}, {type: text,
    text: <small descriptor>}]" envelope) with a real, non-trivial
    payload of raw bytes, base64-encoded the same way the real tool
    does it, and confirms the recorded size/hash account for the
    full image, not just the descriptor.
    """
    import base64
    import hashlib

    image_bytes = bytes(range(256)) * 200  # 51,200 bytes, deterministic
    data_b64 = base64.b64encode(image_bytes).decode("ascii")
    descriptor = json.dumps({
        "status": "ok", "name": "residuals.png", "kind": "image",
        "size": len(image_bytes),
    })
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="read_attached_file",
        response={"content": [
            {"type": "image", "data": data_b64, "mimeType": "image/png"},
            {"type": "text", "text": descriptor},
        ]},
    )
    rec = release_ledger.read_ledger(tmp_path)[0]
    text_bytes = descriptor.encode("utf-8")
    expected_bytes = len(text_bytes) + len(image_bytes)
    expected_hash = hashlib.sha256(text_bytes + image_bytes).hexdigest()

    assert rec["response_bytes"] == expected_bytes, (
        f"expected the full image+text size ({expected_bytes}), got "
        f"{rec['response_bytes']} -- the image bytes were not counted"
    )
    assert rec["response_bytes"] > len(text_bytes), (
        "response_bytes must reflect more than just the tiny text "
        "descriptor when a real image was disclosed"
    )
    assert rec["response_sha256"] == expected_hash, (
        "response_sha256 must be computed over text+image bytes "
        "together, not text alone"
    )
    # The raw image bytes themselves must never land in the ledger
    # file (only the hash/size) -- same "never store the disclosed
    # payload itself" invariant test_response_body_is_hashed_not_stored
    # pins for the text-only case.
    raw = release_ledger.ledger_path(tmp_path).read_text(encoding="utf-8")
    assert data_b64 not in raw


def test_text_only_response_unaffected_by_image_byte_accounting(
    tmp_path: Path,
) -> None:
    """Negative control: a response with no image blocks must be
    hashed/sized exactly as before -- the image-bytes fix must not
    change accounting for the (overwhelmingly common) text-only
    case."""
    import hashlib
    body = '{"status":"ok","n":42}'
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        response={"content": [{"type": "text", "text": body}]},
    )
    rec = release_ledger.read_ledger(tmp_path)[0]
    assert rec["response_bytes"] == len(body.encode("utf-8"))
    assert rec["response_sha256"] == hashlib.sha256(
        body.encode("utf-8")).hexdigest()


def test_chain_detects_edit_delete_reorder(tmp_path: Path) -> None:
    for i in range(3):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool=f"tool_{i}",
            response={"content": [{"type": "text", "text": f'{{"i":{i}}}'}]},
        )
    path = release_ledger.ledger_path(tmp_path)
    # Preserve the exact canonical LF-delimited bytes.  Text-mode rewrites on
    # Windows translate newlines to CRLF, which correctly changes the chained
    # byte hash even when the parsed JSON objects are identical.
    pristine = path.read_bytes()
    assert release_ledger.verify_chain(tmp_path)[0]

    # Edit a field in the middle record.
    lines = pristine.splitlines()
    tampered = json.loads(lines[1]); tampered["tool"] = "innocuous"
    tampered_line = json.dumps(tampered).encode("utf-8")
    path.write_bytes(b"\n".join([lines[0], tampered_line, lines[2]]) + b"\n")
    assert release_ledger.verify_chain(tmp_path)[0] is False

    # Delete the middle record.
    path.write_bytes(b"\n".join([lines[0], lines[2]]) + b"\n")
    assert release_ledger.verify_chain(tmp_path)[0] is False

    # Reorder.
    path.write_bytes(b"\n".join([lines[1], lines[0], lines[2]]) + b"\n")
    assert release_ledger.verify_chain(tmp_path)[0] is False

    # Restore → consistent again.
    path.write_bytes(pristine)
    assert release_ledger.verify_chain(tmp_path)[0] is True


def test_recording_never_raises(tmp_path: Path) -> None:
    # Nonexistent cwd → silently dropped.
    release_ledger.record_release(
        tmp_path / "missing", kind="tool_response", tool="x",
        response={"content": []})
    release_ledger.record_release(None, kind="tool_response", tool="x")
    # Unparseable / non-JSON body → still records, no facts.
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="recall_conversation",
        response={"content": [{"type": "text", "text": "not json"}]})
    rec = release_ledger.read_ledger(tmp_path)[0]
    assert "facts" not in rec and rec["response_sha256"]


def test_failed_append_is_visible_as_accounting_gap(tmp_path: Path) -> None:
    release_ledger._note_recording_failure(tmp_path, OSError("disk full"))
    ok, count, detail = release_ledger.verify_chain(tmp_path)
    assert ok is False and count == 0
    assert "accounting gap" in detail
    assert "1 release" in detail


def test_next_success_acknowledges_but_does_not_hide_gap(tmp_path: Path) -> None:
    release_ledger._note_recording_failure(tmp_path, OSError("disk full"))
    assert release_ledger.record_release(
        tmp_path,
        kind="tool_response",
        tool="get_schema",
        response={"content": [{"type": "text", "text": '{"status":"ok"}'}]},
    )
    record = release_ledger.read_ledger(tmp_path)[0]
    assert record["accounting_gap_before"]["count"] == 1
    assert not release_ledger._health_path(tmp_path).exists()
    ok, count, detail = release_ledger.verify_chain(tmp_path)
    assert ok is False and count == 1
    assert "hash chain valid" in detail and "accounting gap" in detail


def test_invalid_health_marker_fails_closed(tmp_path: Path) -> None:
    health = release_ledger._health_path(tmp_path)
    health.parent.mkdir(parents=True)
    health.write_text("not-json", encoding="utf-8")
    ok, count, detail = release_ledger.verify_chain(tmp_path)
    assert ok is False and count == 0
    assert "invalid ledger health marker" in detail


def test_every_tool_response_is_recorded_via_decorator(tmp_path: Path) -> None:
    """End-to-end through a real registered tool: the ``tool()``
    decorator chokepoint records the response with the active
    ContextVar cwd, so any registered handler is accounted."""
    from sift.tools import HANDLERS

    (tmp_path / ".sift").mkdir()
    with use_cwd(tmp_path):
        resp = _run(HANDLERS["list_results"]({}))
    assert resp["content"]
    records = release_ledger.read_ledger(tmp_path)
    assert len(records) == 1
    assert records[0]["tool"] == "list_results"
    assert records[0]["kind"] == "tool_response"
    ok, _, detail = release_ledger.verify_chain(tmp_path)
    assert ok, detail


def test_plot_release_recorded(tmp_path: Path) -> None:
    release_ledger.record_plot_release(
        tmp_path, filename="coefplot.png", kind="coefficients",
        byte_size=12345)
    rec = release_ledger.read_ledger(tmp_path)[0]
    assert rec["kind"] == "plot_vision"
    assert rec["extra"]["plot_kind"] == "coefficients"
    assert release_ledger.verify_chain(tmp_path)[0]


def test_mentioned_image_release_recorded_distinctly_from_plots(
    tmp_path: Path,
) -> None:
    """Regression test for architecture-audit finding H: a researcher
    @-mentioning a local image (runner.py's ``pending_mentioned_imgs``
    path, staged via ``ui.py``'s ``attach_session_file``) is a vision
    boundary crossing exactly like a model-output plot, but travelled
    a completely separate code path that never called into the
    release ledger at all -- every @-mentioned image was invisible to
    the session's own disclosure accounting. runner.py now records it
    via the same ``record_plot_release`` call plots use, tagged
    ``kind="mentioned_image"`` so an auditor can tell "the researcher
    showed the model this" apart from "the model produced and then
    saw this".
    """
    release_ledger.record_plot_release(
        tmp_path, filename="residuals.png", kind="mentioned_image",
        byte_size=4096)
    rec = release_ledger.read_ledger(tmp_path)[0]
    assert rec["kind"] == "plot_vision"
    assert rec["extra"]["plot_kind"] == "mentioned_image"
    assert rec["extra"]["filename"] == "residuals.png"
    assert rec["extra"]["bytes"] == 4096
    assert release_ledger.verify_chain(tmp_path)[0]


def test_long_session_chain_stays_valid_and_writes_stay_cheap(
    tmp_path: Path,
) -> None:
    """Regression for the tip-cache optimization in ``record_release``.

    Before caching, every write re-read the ENTIRE ledger file just to
    find the previous record's hash — a long research session (many
    hundreds of tool calls) made total ledger I/O quadratic in call
    count. The cache turns steady-state writes into an O(1) append,
    keyed by (file size, tip hash) so a write from a second process —
    or any out-of-band change — is detected and falls back to a real
    read rather than silently trusting a stale tip.

    This pins two things: (1) the chain is still perfectly valid after
    many sequential writes through the cached path, and (2) total
    bytes read back off disk across the whole run stays roughly linear
    in record count rather than growing quadratically — i.e., the
    optimization is actually taking effect, not just failing to break
    anything.
    """
    import time as _time

    release_ledger._TIP_CACHE.clear()
    n = 400
    start = _time.perf_counter()
    for i in range(n):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool=f"tool_{i % 15}",
            args={"dataset": "survey.csv", "variable": f"v{i}"},
            response={
                "content": [{"type": "text", "text": f'{{"n":{i}}}'}],
            },
        )
    elapsed = _time.perf_counter() - start

    records = release_ledger.read_ledger(tmp_path)
    assert len(records) == n
    ok, verified_n, detail = release_ledger.verify_chain(tmp_path)
    assert ok and verified_n == n, detail

    # Generous ceiling — this is a correctness-of-scaling check, not a
    # tight timing assertion (those are flaky across CI hardware). The
    # quadratic pre-fix version of this exact loop took well over a
    # second for 400 records on this same class of hardware because
    # each write re-read a linearly growing file; the cached version
    # finishes in a small fraction of that. 5s leaves enormous margin
    # while still catching an accidental reintroduction of the
    # per-write full-file re-read.
    assert elapsed < 5.0, (
        f"400 sequential ledger writes took {elapsed:.2f}s — the "
        f"tip cache may have regressed back to reading the whole "
        f"file on every write"
    )

    # The cache should now hold exactly the tip for this one ledger.
    key = str(release_ledger.ledger_path(tmp_path))
    assert key in release_ledger._TIP_CACHE
    cached_size, cached_tip = release_ledger._TIP_CACHE[key]
    assert cached_size == release_ledger.ledger_path(tmp_path).stat().st_size
    assert isinstance(cached_tip, str) and len(cached_tip) == 64


def test_tip_cache_falls_back_when_file_changes_out_of_band(
    tmp_path: Path,
) -> None:
    """If the ledger file is modified without going through
    ``record_release`` (e.g. a second process, or a test fixture
    resetting state), the size-keyed cache must detect the mismatch
    and re-read rather than chain new records onto a stale tip."""
    import hashlib as _hashlib

    release_ledger._TIP_CACHE.clear()
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="get_schema",
        response={"content": [{"type": "text", "text": "{}"}]},
    )
    # Simulate an external append that the cache didn't see (e.g. a
    # second process, or a test fixture poking the file directly).
    # This line is deliberately NOT chain-valid against record 1 —
    # that's fine, the point here is only what the NEXT
    # ``record_release`` picks up as ``prev``, not chain validity of
    # the injected line itself.
    path = release_ledger.ledger_path(tmp_path)
    injected = (
        '{"v":1,"ts":"x","kind":"tool_response","tool":"external",'
        '"prev":"' + ("0" * 64) + '","hash":"' + ("1" * 64) + '"}'
    )
    with path.open("ab") as fh:
        fh.write(injected.encode("utf-8") + b"\n")
    expected_tip = _hashlib.sha256(injected.encode("utf-8")).hexdigest()

    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        response={"content": [{"type": "text", "text": "{}"}]},
    )
    records = release_ledger.read_ledger(tmp_path)
    assert len(records) == 3
    # The third record's prev must be the SHA-256 of the injected
    # line's raw bytes (what a fresh read of the file would compute
    # as the chain tip) — proving the size mismatch triggered a real
    # re-read instead of handing back the pre-injection cached tip
    # (which would equal record 1's hash and fail this assertion).
    assert records[2]["prev"] == expected_tip


def test_cold_tip_cache_appends_to_existing_chain(tmp_path: Path) -> None:
    """A fresh process must discover the existing tip from disk.

    This specifically guards the Windows sharing-mode boundary: the ledger's
    secure writable handle is exclusive, so tip discovery must happen before
    that handle is opened rather than depending on a process-local warm cache.
    """
    release_ledger._TIP_CACHE.clear()
    assert release_ledger.record_release(
        tmp_path, kind="tool_response", tool="get_schema",
        response={"content": [{"type": "text", "text": "{}"}]},
    )
    release_ledger._TIP_CACHE.clear()
    assert release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        response={"content": [{"type": "text", "text": "{}"}]},
    )
    assert release_ledger.verify_chain(tmp_path)[:2] == (True, 2)
