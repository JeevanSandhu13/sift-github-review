from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from sift.beta_study import (
    BetaStudyError,
    PROTOCOL_VERSION,
    export_study_summary,
    participant_token,
    record_event,
    summarize_study,
)


def _consent(path: Path, index: int, audience: str = "researcher") -> str:
    token = participant_token("study-secret-material", f"participant-{index}")
    record_event(path, {
        "participant_token": token,
        "audience": audience,
        "event_type": "consent",
        "consent_version": PROTOCOL_VERSION,
    })
    return token


def test_events_require_consent_and_reject_free_text(tmp_path: Path) -> None:
    path = tmp_path / "study.jsonl"
    token = participant_token("study-secret-material", "p1")
    with pytest.raises(BetaStudyError, match="consent"):
        record_event(path, {
            "participant_token": token, "audience": "researcher",
            "event_type": "task_started", "task_id": "public_reproducible_analysis",
        })
    with pytest.raises(BetaStudyError, match="unsupported event fields"):
        record_event(path, {
            "participant_token": token, "audience": "researcher",
            "event_type": "consent", "consent_version": PROTOCOL_VERSION,
            "comment": "a potentially identifying note",
        })


def test_hash_chain_detects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "study.jsonl"
    token = _consent(path, 1)
    record_event(path, {
        "participant_token": token, "audience": "researcher",
        "event_type": "task_started", "task_id": "public_reproducible_analysis",
    })
    rows = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[0]); payload["audience"] = "administrator"
    rows[0] = json.dumps(payload)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(BetaStudyError, match="digest"):
        summarize_study(path)


def test_study_log_refuses_symlink_without_modifying_target(tmp_path: Path) -> None:
    target = tmp_path / "sensitive.txt"
    target.write_text("preserve", encoding="utf-8")
    path = tmp_path / "study.jsonl"
    path.symlink_to(target)
    token = participant_token("study-secret-material", "p1")
    with pytest.raises((BetaStudyError, OSError)):
        record_event(path, {
            "participant_token": token,
            "audience": "researcher",
            "event_type": "consent",
            "consent_version": PROTOCOL_VERSION,
        })
    assert target.read_text(encoding="utf-8") == "preserve"


def test_small_cohort_is_suppressed_and_export_has_no_tokens(tmp_path: Path) -> None:
    path = tmp_path / "study.jsonl"
    for index in range(4):
        _consent(path, index)
    summary = summarize_study(path)
    assert summary["status"] == "suppressed_small_cohort"
    assert summary["task_metrics"] == {}
    assert "participant-" not in json.dumps(summary)


def test_five_person_study_exports_only_aggregate_metrics(tmp_path: Path) -> None:
    path = tmp_path / "study.jsonl"
    for index in range(5):
        token = _consent(path, index)
        base = {
            "participant_token": token, "audience": "researcher",
            "task_id": "public_reproducible_analysis",
        }
        record_event(path, {**base, "event_type": "task_started"})
        record_event(path, {
            **base, "event_type": "task_completed", "success": True,
            "duration_seconds": 60 + index,
        })
    output = tmp_path / "summary.json"
    summary = export_study_summary(path, output)
    assert summary["status"] == "ready"
    assert summary["overall_completion_rate"] == 1.0
    task = summary["task_metrics"]["public_reproducible_analysis"]
    assert task["attempts"] == task["completions"] == 5
    serialized = output.read_text(encoding="utf-8")
    assert all(participant_token("study-secret-material", f"participant-{i}") not in serialized for i in range(5))


def test_audience_task_boundary_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "study.jsonl"
    token = _consent(path, 1, audience="administrator")
    with pytest.raises(BetaStudyError, match="audience"):
        record_event(path, {
            "participant_token": token, "audience": "administrator",
            "event_type": "task_started", "task_id": "clinical_method_review",
        })


def test_aggregate_tracks_issue_resolution_without_exporting_issue_id(tmp_path: Path) -> None:
    path = tmp_path / "study.jsonl"
    tokens = [_consent(path, index) for index in range(5)]
    issue = hashlib.sha256(b"external-issue-27").hexdigest()
    record_event(path, {
        "participant_token": tokens[0], "audience": "researcher",
        "task_id": "clinical_method_review", "event_type": "task_started",
    })
    event = {
        "participant_token": tokens[0], "audience": "researcher",
        "task_id": "clinical_method_review", "issue_token": issue,
        "issue_severity": "high",
    }
    record_event(path, {**event, "event_type": "issue_opened"})
    summary = summarize_study(path)
    assert summary["unresolved_issue_counts"]["high"] == 1
    assert issue not in json.dumps(summary)
    record_event(path, {**event, "event_type": "issue_resolved"})
    assert summarize_study(path)["unresolved_issue_counts"]["high"] == 0
