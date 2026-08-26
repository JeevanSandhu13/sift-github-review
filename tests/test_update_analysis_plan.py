"""update_analysis_plan — validation, persistence, and honesty caps."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from sift.config import use_cwd
from sift.tools import HANDLERS


def _call(args, cwd):
    async def go():
        with use_cwd(cwd):
            return await HANDLERS["update_analysis_plan"](args)
    return asyncio.run(go())


def _body(resp):
    return json.loads(resp["content"][0]["text"])


def test_valid_plan_persists(tmp_path: Path) -> None:
    steps = [
        {"title": "Inspect dataset", "status": "done"},
        {"title": "Investigate churn drivers", "status": "active"},
        {"title": "Robustness checks", "status": "pending"},
    ]
    body = _body(_call({"steps": steps}, tmp_path))
    assert body["status"] == "ok"
    assert body["steps"] == 3 and body["done"] == 1
    saved = json.loads(
        (tmp_path / ".sift" / "analysis_plan.json").read_text(encoding="utf-8"))
    assert [s["status"] for s in saved["steps"]] == \
        ["done", "active", "pending"]


def test_invalid_shapes_rejected(tmp_path: Path) -> None:
    assert _body(_call({}, tmp_path))["status"] == "error"
    assert _body(_call({"steps": []}, tmp_path))["status"] == "error"
    assert _body(_call({"steps": ["x"]}, tmp_path))["status"] == "error"
    bad_status = [{"title": "a", "status": "doing"}]
    assert "invalid status" in _body(
        _call({"steps": bad_status}, tmp_path))["reason"]
    too_many = [{"title": f"s{i}", "status": "pending"} for i in range(21)]
    assert "too many" in _body(
        _call({"steps": too_many}, tmp_path))["reason"]


def test_titles_are_sanitized_and_capped(tmp_path: Path) -> None:
    steps = [{"title": "fit‮ model\x00" + "x" * 400,
              "status": "pending"}]
    body = _body(_call({"steps": steps}, tmp_path))
    assert body["status"] == "ok"
    saved = json.loads(
        (tmp_path / ".sift" / "analysis_plan.json").read_text(encoding="utf-8"))
    title = saved["steps"][0]["title"]
    assert "‮" not in title and "\x00" not in title
    assert len(title) <= 132  # cap + truncation marker


def test_latest_call_replaces_plan(tmp_path: Path) -> None:
    _call({"steps": [{"title": "one", "status": "pending"}]}, tmp_path)
    _call({"steps": [{"title": "two", "status": "done"}]}, tmp_path)
    saved = json.loads(
        (tmp_path / ".sift" / "analysis_plan.json").read_text(encoding="utf-8"))
    assert len(saved["steps"]) == 1
    assert saved["steps"][0]["title"] == "two"


# ---------------------------------------------------------------------------
# Lock-in + deviation tracking
# ---------------------------------------------------------------------------

def test_lock_snapshots_current_titles(tmp_path: Path) -> None:
    steps = [
        {"title": "Inspect dataset", "status": "done"},
        {"title": "Fit primary spec", "status": "pending"},
    ]
    body = _body(_call({"steps": steps, "lock": True}, tmp_path))
    assert body["status"] == "ok"
    assert body["locked"] is True
    assert body["locked_at"]
    assert "plan_deviations" not in body  # locking is not a deviation from itself

    saved = json.loads(
        (tmp_path / ".sift" / "analysis_plan.json").read_text(encoding="utf-8"))
    assert saved["locked"]["steps"] == \
        ["Inspect dataset", "Fit primary spec"]
    assert saved["locked"]["locked_at"] == body["locked_at"]


def test_no_lock_means_no_locked_field_in_response(tmp_path: Path) -> None:
    steps = [{"title": "one", "status": "pending"}]
    body = _body(_call({"steps": steps}, tmp_path))
    assert "locked" not in body
    assert "plan_deviations" not in body
    saved = json.loads(
        (tmp_path / ".sift" / "analysis_plan.json").read_text(encoding="utf-8"))
    assert "locked" not in saved


def test_dropped_step_after_lock_is_flagged(tmp_path: Path) -> None:
    """Silently removing a locked step (not marking it skipped) must
    surface as a deviation -- this is the concrete case a
    pre-registered plan exists to catch."""
    locked_steps = [
        {"title": "Fit spec A", "status": "pending"},
        {"title": "Fit spec B", "status": "pending"},
    ]
    _call({"steps": locked_steps, "lock": True}, tmp_path)

    later = [{"title": "Fit spec A", "status": "done"}]  # B silently dropped
    body = _body(_call({"steps": later}, tmp_path))
    assert body["locked"] is True
    assert body["plan_deviations"] == {
        "dropped": ["Fit spec B"], "added": [],
    }


def test_added_step_after_lock_is_flagged(tmp_path: Path) -> None:
    """A step that appears after locking but was never in the
    original snapshot is the mirror-image deviation -- new analysis
    added without amending the pre-registered plan."""
    locked_steps = [{"title": "Fit spec A", "status": "pending"}]
    _call({"steps": locked_steps, "lock": True}, tmp_path)

    later = [
        {"title": "Fit spec A", "status": "done"},
        {"title": "Fit spec A, robust SEs", "status": "pending"},
    ]
    body = _body(_call({"steps": later}, tmp_path))
    assert body["plan_deviations"] == {
        "dropped": [], "added": ["Fit spec A, robust SEs"],
    }


def test_status_only_change_is_not_a_deviation(tmp_path: Path) -> None:
    """Marking a locked step's status pending -> active -> done, or
    to 'skipped', must never be reported as a deviation -- that is
    exactly the ordinary, expected use of the plan. Only a silently
    vanished or newly appearing TITLE counts."""
    locked_steps = [{"title": "Fit spec A", "status": "pending"}]
    _call({"steps": locked_steps, "lock": True}, tmp_path)

    for status in ("active", "done", "skipped"):
        body = _body(_call(
            {"steps": [{"title": "Fit spec A", "status": status}]},
            tmp_path,
        ))
        assert "plan_deviations" not in body, (status, body)


def test_lock_persists_across_ordinary_updates(tmp_path: Path) -> None:
    """A plain steps-update call (no lock arg) must round-trip the
    existing lock rather than silently dropping it -- the snapshot
    has to survive every ordinary status-transition call between
    locking and whenever a deviation eventually happens."""
    locked_steps = [{"title": "Fit spec A", "status": "pending"}]
    _call({"steps": locked_steps, "lock": True}, tmp_path)

    _call({"steps": [{"title": "Fit spec A", "status": "active"}]}, tmp_path)
    body = _body(_call(
        {"steps": [{"title": "Fit spec A", "status": "done"}]}, tmp_path))
    assert body["locked"] is True

    saved = json.loads(
        (tmp_path / ".sift" / "analysis_plan.json").read_text(encoding="utf-8"))
    assert saved["locked"]["steps"] == ["Fit spec A"]


def test_relock_replaces_the_snapshot(tmp_path: Path) -> None:
    """Re-locking is a deliberate, visible re-baseline -- deviations
    are measured against the MOST RECENT lock, not the first one."""
    _call({"steps": [{"title": "Fit spec A", "status": "pending"}],
           "lock": True}, tmp_path)
    _call({"steps": [{"title": "Fit spec A", "status": "done"},
                     {"title": "Fit spec B", "status": "pending"}],
           "lock": True}, tmp_path)

    # Now spec A alone (present at the very first lock) is not a
    # "drop" relative to the SECOND, most recent lock unless it's
    # actually missing from the current steps too.
    body = _body(_call(
        {"steps": [{"title": "Fit spec A", "status": "done"},
                   {"title": "Fit spec B", "status": "done"}]},
        tmp_path,
    ))
    assert "plan_deviations" not in body


def test_lock_true_call_itself_never_reports_deviations(
    tmp_path: Path,
) -> None:
    """A lock call redefines the baseline to exactly the steps it was
    just given -- it can never itself be a deviation from what it is
    now defining, even if the previous lock's steps differ wildly."""
    _call({"steps": [{"title": "Old plan", "status": "done"}],
           "lock": True}, tmp_path)
    body = _body(_call(
        {"steps": [{"title": "Completely different plan",
                   "status": "pending"}], "lock": True},
        tmp_path,
    ))
    assert "plan_deviations" not in body


def test_lock_must_be_boolean(tmp_path: Path) -> None:
    body = _body(_call(
        {"steps": [{"title": "a", "status": "pending"}], "lock": "yes"},
        tmp_path,
    ))
    assert body["status"] == "error"
    assert "lock" in body["reason"]


def test_corrupt_persisted_plan_does_not_break_lock_read(
    tmp_path: Path,
) -> None:
    """A malformed analysis_plan.json on disk (hand-edited, or from
    an older schema) must not crash the tool -- treated as no
    pre-existing lock, same as a missing file."""
    plan_dir = tmp_path / ".sift"
    plan_dir.mkdir(parents=True)
    (plan_dir / "analysis_plan.json").write_text("{not valid json",
                                                   encoding="utf-8")
    body = _body(_call(
        {"steps": [{"title": "a", "status": "pending"}]}, tmp_path))
    assert body["status"] == "ok"
    assert "locked" not in body
