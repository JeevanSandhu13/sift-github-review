"""Tests for analysis checkpoints.

Three layers, matching the pattern established in ``test_rewind.py``:

1. ``sift.checkpoints`` module — pure file-backed CRUD, no bridge.
2. ``SiftBridge`` integration — ``create_checkpoint`` / ``list_checkpoints``
   / ``delete_checkpoint`` / ``restore_checkpoint`` / ``compare_checkpoints``
   wired to a real cwd, real chat_history.jsonl, real result store.
3. Cross-cutting regression: any rewind (whether via ``restore_checkpoint``
   or the ordinary edit-a-past-message path) prunes checkpoints that now
   point past the truncated history — the exact "does editing X silently
   leave Y in a stale state" bug class this project has hit twice before
   (dp_epsilon, excel_sheet on policy setters).
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest

from sift import checkpoints as ckpt


def _write_history(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 1. Module-level CRUD
# ---------------------------------------------------------------------------

def test_add_and_list_roundtrip(tmp_path: Path) -> None:
    cp, reason = ckpt.add_checkpoint(
        tmp_path, label="before reweighting", turn_index=3,
        result_ids=["M1", "M2"],
    )
    assert reason is None
    assert cp is not None
    assert cp.id == "cp1"
    assert cp.label == "before reweighting"
    assert cp.turn_index == 3
    assert cp.result_ids == ["M1", "M2"]

    listed = ckpt.list_checkpoints(tmp_path)
    assert len(listed) == 1
    assert listed[0].id == "cp1"


def test_ids_increment_sequentially(tmp_path: Path) -> None:
    a, _ = ckpt.add_checkpoint(tmp_path, label="a", turn_index=1, result_ids=[])
    b, _ = ckpt.add_checkpoint(tmp_path, label="b", turn_index=2, result_ids=[])
    c, _ = ckpt.add_checkpoint(tmp_path, label="c", turn_index=3, result_ids=[])
    assert [a.id, b.id, c.id] == ["cp1", "cp2", "cp3"]


def test_id_sequencing_skips_existing_ids_after_delete(tmp_path: Path) -> None:
    a, _ = ckpt.add_checkpoint(tmp_path, label="a", turn_index=1, result_ids=[])
    b, _ = ckpt.add_checkpoint(tmp_path, label="b", turn_index=2, result_ids=[])
    ckpt.delete_checkpoint(tmp_path, a.id)
    # "cp1" is free again, but the implementation shouldn't reuse it
    # while "cp2" is still live and "cp1" would collide with nothing
    # — just confirm the new one gets a fresh, non-colliding id.
    c, _ = ckpt.add_checkpoint(tmp_path, label="c", turn_index=3, result_ids=[])
    ids = {x.id for x in ckpt.list_checkpoints(tmp_path)}
    assert ids == {b.id, c.id}
    assert len(ids) == 2


def test_empty_label_refused(tmp_path: Path) -> None:
    cp, reason = ckpt.add_checkpoint(
        tmp_path, label="   ", turn_index=1, result_ids=[],
    )
    assert cp is None
    assert "empty" in reason


def test_label_trimmed_and_capped(tmp_path: Path) -> None:
    long_label = "x" * 500
    cp, reason = ckpt.add_checkpoint(
        tmp_path, label=f"  {long_label}  ", turn_index=1, result_ids=[],
    )
    assert reason is None
    assert cp.label == long_label[: ckpt.MAX_LABEL_LEN]


def test_max_checkpoints_cap_enforced(tmp_path: Path) -> None:
    for i in range(ckpt.MAX_CHECKPOINTS):
        cp, reason = ckpt.add_checkpoint(
            tmp_path, label=f"cp {i}", turn_index=i, result_ids=[],
        )
        assert reason is None, reason
    cp, reason = ckpt.add_checkpoint(
        tmp_path, label="one too many", turn_index=999, result_ids=[],
    )
    assert cp is None
    assert "50" in reason or str(ckpt.MAX_CHECKPOINTS) in reason
    assert len(ckpt.list_checkpoints(tmp_path)) == ckpt.MAX_CHECKPOINTS


def test_checkpoint_does_not_report_success_when_persistence_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(ckpt, "_atomic_write", lambda *args, **kwargs: False)
    checkpoint, reason = ckpt.add_checkpoint(
        tmp_path, label="important", turn_index=1, result_ids=[],
    )
    assert checkpoint is None
    assert reason == "could not persist checkpoint"


def test_concurrent_checkpoint_creation_preserves_every_unique_entry(
    tmp_path: Path,
) -> None:
    def _add(index: int):
        return ckpt.add_checkpoint(
            tmp_path,
            label=f"checkpoint {index}",
            turn_index=index,
            result_ids=[f"r{index}"],
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(_add, range(30)))
    assert all(cp is not None and reason is None for cp, reason in outcomes)
    saved = ckpt.list_checkpoints(tmp_path)
    assert len(saved) == 30
    assert len({cp.id for cp in saved}) == 30


def test_delete_checkpoint(tmp_path: Path) -> None:
    cp, _ = ckpt.add_checkpoint(tmp_path, label="a", turn_index=1, result_ids=[])
    assert ckpt.delete_checkpoint(tmp_path, cp.id) is True
    assert ckpt.list_checkpoints(tmp_path) == []
    assert ckpt.delete_checkpoint(tmp_path, cp.id) is False


def test_get_checkpoint_unknown_id_returns_none(tmp_path: Path) -> None:
    assert ckpt.get_checkpoint(tmp_path, "does-not-exist") is None


def test_persists_across_reload(tmp_path: Path) -> None:
    ckpt.add_checkpoint(tmp_path, label="a", turn_index=1, result_ids=["M1"])
    # Simulate a fresh process: nothing cached, just re-read the file.
    listed = ckpt.list_checkpoints(tmp_path)
    assert len(listed) == 1
    assert listed[0].label == "a"
    assert (tmp_path / ".sift" / "checkpoints.json").exists()


def test_corrupted_file_does_not_crash_list(tmp_path: Path) -> None:
    path = tmp_path / ".sift"
    path.mkdir(parents=True)
    (path / "checkpoints.json").write_text("{ not valid json", encoding="utf-8")
    assert ckpt.list_checkpoints(tmp_path) == []


def test_corrupted_single_entry_skipped_others_kept(tmp_path: Path) -> None:
    path = tmp_path / ".sift"
    path.mkdir(parents=True)
    (path / "checkpoints.json").write_text(json.dumps({
        "version": 1,
        "checkpoints": [
            {"id": "cp1", "label": "good", "turn_index": 1,
             "created_at": "x", "result_ids": []},
            {"id": "cp2", "label": "bad — missing turn_index"},
        ],
    }), encoding="utf-8")
    listed = ckpt.list_checkpoints(tmp_path)
    assert len(listed) == 1
    assert listed[0].id == "cp1"


# ---------------------------------------------------------------------------
# 2. prune_checkpoints_at_or_after
# ---------------------------------------------------------------------------

def test_prune_keeps_checkpoint_exactly_at_cut(tmp_path: Path) -> None:
    """The boundary case: a checkpoint whose turn_index equals the
    cut bookmarks exactly what survives a rewind to that cut — it
    must NOT be pruned, or restoring a checkpoint would delete itself
    the instant it's used."""
    ckpt.add_checkpoint(tmp_path, label="early", turn_index=1, result_ids=[])
    ckpt.add_checkpoint(tmp_path, label="at cut", turn_index=3, result_ids=[])
    ckpt.add_checkpoint(tmp_path, label="late", turn_index=5, result_ids=[])

    removed = ckpt.prune_checkpoints_after(tmp_path, 3)

    remaining = {c.label for c in ckpt.list_checkpoints(tmp_path)}
    # Only "late" (turn_index 5 > cut 3) must go.
    assert remaining == {"early", "at cut"}
    assert len(removed) == 1


def test_prune_no_op_when_nothing_past_cut(tmp_path: Path) -> None:
    ckpt.add_checkpoint(tmp_path, label="early", turn_index=1, result_ids=[])
    removed = ckpt.prune_checkpoints_after(tmp_path, 10)
    assert removed == []
    assert len(ckpt.list_checkpoints(tmp_path)) == 1


# ---------------------------------------------------------------------------
# 3. SiftBridge integration
# ---------------------------------------------------------------------------

def test_create_checkpoint_no_session_refused(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    bridge = SiftBridge(cwd=None)
    res = bridge.create_checkpoint("a label")
    assert res["ok"] is False


def test_create_checkpoint_no_history_refused(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    bridge = SiftBridge(cwd=tmp_path)
    res = bridge.create_checkpoint("a label")
    assert res["ok"] is False
    assert "no chat history" in res["reason"]


def test_create_checkpoint_snapshots_current_state(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    from sift.store import get_store

    bridge = SiftBridge(cwd=tmp_path)
    assert bridge._active_runner() is not None

    store = get_store(tmp_path)
    store.insert(
        label="r1", analysis_type="t", sanitized_payload={},
        language="Python", script_code="x=1", transformations=[],
    )

    history_path = tmp_path / ".sift" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "first"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
        {"type": "assistant_text", "text": "ok"},
    ])

    res = bridge.create_checkpoint("first pass")
    assert res["ok"] is True, res
    cp = res["checkpoint"]
    assert cp["label"] == "first pass"
    assert cp["turn_index"] == 1
    assert cp["result_count"] == 1

    listed = bridge.list_checkpoints()
    assert listed["ok"] is True
    assert len(listed["checkpoints"]) == 1
    assert listed["checkpoints"][0]["id"] == cp["id"]


def test_delete_checkpoint_via_bridge(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    bridge = SiftBridge(cwd=tmp_path)

    history_path = tmp_path / ".sift" / "chat_history.jsonl"
    _write_history(history_path, [{"type": "user_message", "text": "first"}])
    res = bridge.create_checkpoint("cp")
    cp_id = res["checkpoint"]["id"]

    del_res = bridge.delete_checkpoint(cp_id)
    assert del_res["ok"] is True
    assert bridge.list_checkpoints()["checkpoints"] == []

    del_res2 = bridge.delete_checkpoint(cp_id)
    assert del_res2["ok"] is False


def test_restore_checkpoint_delegates_to_rewind(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    from sift.store import get_store

    bridge = SiftBridge(cwd=tmp_path)
    store = get_store(tmp_path)
    store.insert(
        label="kept", analysis_type="t", sanitized_payload={"i": 1},
        language="Python", script_code="x=1", transformations=[],
    )
    store.insert(
        label="dropped", analysis_type="t", sanitized_payload={"i": 2},
        language="Python", script_code="x=2", transformations=[],
    )

    history_path = tmp_path / ".sift" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "first"},
        {"type": "tool_result", "text": json.dumps({"result_id": "M1"})},
        {"type": "assistant_text", "text": "ok"},
    ])
    # Checkpoint at turn_index 1 (bookmarks the whole file above).
    cp_res = bridge.create_checkpoint("clean baseline")
    assert cp_res["ok"] is True
    cp_id = cp_res["checkpoint"]["id"]

    # Conversation continues past the checkpoint.
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "user_message", "text": "second"}) + "\n")
        f.write(json.dumps(
            {"type": "tool_result", "text": json.dumps({"result_id": "M2"})},
        ) + "\n")

    res = bridge.restore_checkpoint(cp_id)
    assert res["ok"] is True, res
    assert res["truncated_from_index"] == 1
    assert res["restored_checkpoint"]["id"] == cp_id

    assert {r.id for r in store.list_all()} == {"M1"}


def test_restore_unknown_checkpoint_refused(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    bridge = SiftBridge(cwd=tmp_path)
    res = bridge.restore_checkpoint("nope")
    assert res["ok"] is False


def test_rewind_prunes_stale_checkpoints_via_restore_path(tmp_path: Path) -> None:
    """A checkpoint taken further along than the restore target must
    disappear after the restore — it points at chat-history bytes
    (and possibly now-hidden result rows) that no longer exist."""
    from sift.ui import SiftBridge

    bridge = SiftBridge(cwd=tmp_path)
    history_path = tmp_path / ".sift" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "t0"},
    ])
    early = bridge.create_checkpoint("early")["checkpoint"]

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "user_message", "text": "t1"}) + "\n")
    late = bridge.create_checkpoint("late")["checkpoint"]

    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "user_message", "text": "t2"}) + "\n")

    # Restore back to the early checkpoint (turn_index 1) — this
    # truncates everything from turn 1 onward, which is exactly
    # where "late" (turn_index 2) points.
    res = bridge.restore_checkpoint(early["id"])
    assert res["ok"] is True, res

    remaining_ids = {c["id"] for c in bridge.list_checkpoints()["checkpoints"]}
    # "early" bookmarks exactly what survives (turn_index 1 == cut 1)
    # — restoring it must not delete it. "late" (turn_index 2 > cut 1)
    # references a message that's now gone and must be pruned.
    assert early["id"] in remaining_ids
    assert late["id"] not in remaining_ids


def test_ordinary_rewind_also_prunes_checkpoints(tmp_path: Path) -> None:
    """Pruning must fire from the plain edit-a-past-message rewind
    path too, not just ``restore_checkpoint`` — both call the same
    ``rewind_to``, and a researcher can edit an old message without
    ever touching the checkpoints UI."""
    from sift.ui import SiftBridge

    bridge = SiftBridge(cwd=tmp_path)
    history_path = tmp_path / ".sift" / "chat_history.jsonl"
    _write_history(history_path, [
        {"type": "user_message", "text": "t0"},
        {"type": "user_message", "text": "t1"},
    ])
    stale = bridge.create_checkpoint("will go stale")["checkpoint"]
    assert stale["turn_index"] == 2

    res = bridge.rewind_to(1)
    assert res["ok"] is True, res

    remaining = bridge.list_checkpoints()["checkpoints"]
    assert remaining == []


# ---------------------------------------------------------------------------
# 4. compare_checkpoints
# ---------------------------------------------------------------------------

def _checkpoint_with_results(bridge, store, label, events, result_specs):
    """Insert result rows, write history through ``events``, create a
    checkpoint, and return its dict."""
    for spec in result_specs:
        store.insert(**spec)
    history_path = bridge.cwd / ".sift" / "chat_history.jsonl"
    with history_path.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    res = bridge.create_checkpoint(label)
    assert res["ok"] is True, res
    return res["checkpoint"]


def test_compare_checkpoints_same_branch_progress(tmp_path: Path) -> None:
    """Two checkpoints on the SAME (never-rewound) branch: B is a
    strict superset of A. This is the "how far did I get since my
    last bookmark" comparison — no divergence, just progress."""
    from sift.ui import SiftBridge
    from sift.store import get_store

    bridge = SiftBridge(cwd=tmp_path)
    store = get_store(tmp_path)

    row_a = store.insert(
        label="corr income~age", analysis_type="correlation",
        sanitized_payload={}, language="Python", script_code="x=1",
        transformations=[], source_dataset="survey.csv",
    )
    a = _checkpoint_with_results(
        bridge, store, "checkpoint a",
        events=[
            {"type": "user_message", "text": "t0"},
            {"type": "tool_result",
             "text": json.dumps({"result_id": row_a.id})},
        ],
        result_specs=[],
    )

    row_b = store.insert(
        label="regression income~age+educ", analysis_type="regression",
        sanitized_payload={}, language="Python", script_code="x=2",
        transformations=[], source_dataset="survey.csv",
    )
    b = _checkpoint_with_results(
        bridge, store, "checkpoint b",
        events=[
            {"type": "user_message", "text": "t1"},
            {"type": "tool_result",
             "text": json.dumps({"result_id": row_b.id})},
        ],
        result_specs=[],
    )

    res = bridge.compare_checkpoints(a["id"], b["id"])
    assert res["ok"] is True, res
    assert res["only_in_a"] == []
    assert {r["id"] for r in res["only_in_b"]} == {row_b.id}
    assert {r["id"] for r in res["common"]} == {row_a.id}
    assert res["tally_a"] == {"correlation": 1}
    assert res["tally_b"] == {"correlation": 1, "regression": 1}


def test_compare_checkpoints_diverged_branches(tmp_path: Path) -> None:
    """A true branch compare: checkpoint the shared base, continue
    down branch A, restore back to the base (branch A's later
    checkpoint gets pruned — expected, per the pruning tests above),
    then continue down branch B. The base checkpoint survives the
    restore (turn_index == cut) so it's still available to compare
    against, and a fresh checkpoint on branch B captures where that
    branch ended up."""
    from sift.ui import SiftBridge
    from sift.store import get_store

    bridge = SiftBridge(cwd=tmp_path)
    store = get_store(tmp_path)

    base = _checkpoint_with_results(
        bridge, store, "shared base",
        events=[{"type": "user_message", "text": "t0"}],
        result_specs=[],
    )

    row_a = store.insert(
        label="branch a result", analysis_type="correlation",
        sanitized_payload={}, language="Python", script_code="x=1",
        transformations=[], source_dataset="survey.csv",
    )
    branch_a = _checkpoint_with_results(
        bridge, store, "branch a",
        events=[
            {"type": "user_message", "text": "t1-a"},
            {"type": "tool_result",
             "text": json.dumps({"result_id": row_a.id})},
        ],
        result_specs=[],
    )

    restore_res = bridge.restore_checkpoint(base["id"])
    assert restore_res["ok"] is True, restore_res
    # branch_a is now stale (turn_index > cut) and gone; base survives.
    remaining_ids = {c["id"] for c in bridge.list_checkpoints()["checkpoints"]}
    assert base["id"] in remaining_ids
    assert branch_a["id"] not in remaining_ids

    row_b = store.insert(
        label="branch b result", analysis_type="regression",
        sanitized_payload={}, language="Python", script_code="x=2",
        transformations=[], source_dataset="survey.csv",
    )
    branch_b = _checkpoint_with_results(
        bridge, store, "branch b",
        events=[
            {"type": "user_message", "text": "t1-b"},
            {"type": "tool_result",
             "text": json.dumps({"result_id": row_b.id})},
        ],
        result_specs=[],
    )

    res = bridge.compare_checkpoints(base["id"], branch_b["id"])
    assert res["ok"] is True, res
    assert res["only_in_a"] == []
    assert {r["id"] for r in res["only_in_b"]} == {row_b.id}
    assert res["tally_b"] == {"regression": 1}
    # row_a from the abandoned branch is visible to neither checkpoint.
    all_ids = ({r["id"] for r in res["only_in_a"]}
               | {r["id"] for r in res["only_in_b"]}
               | {r["id"] for r in res["common"]})
    assert row_a.id not in all_ids


def test_compare_checkpoints_unknown_id_refused(tmp_path: Path) -> None:
    from sift.ui import SiftBridge
    bridge = SiftBridge(cwd=tmp_path)
    res = bridge.compare_checkpoints("nope-a", "nope-b")
    assert res["ok"] is False
