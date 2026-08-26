"""Privacy-preserving, local-only instrumentation for Sift beta studies."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from sift.file_lock import exclusive_file_lock
from sift.reliability import atomic_write_json
from sift.secure_file import append_bytes_no_follow, read_bytes_no_follow

SCHEMA_VERSION = 1
PROTOCOL_VERSION = "2026.1"
MIN_EXPORT_COHORT = 5
MAX_EVENT_LOG_BYTES = 10 * 1024 * 1024
ZERO_HASH = "0" * 64

TASKS: dict[str, dict[str, Any]] = {
    "public_reproducible_analysis": {"audience": "researcher", "target_seconds": 900},
    "confidential_local_analysis": {"audience": "researcher", "target_seconds": 1200},
    "clinical_method_review": {"audience": "researcher", "target_seconds": 1200},
    "survey_weighted_analysis": {"audience": "researcher", "target_seconds": 1200},
    "longitudinal_analysis": {"audience": "researcher", "target_seconds": 1500},
    "geospatial_analysis": {"audience": "researcher", "target_seconds": 1500},
    "database_private_extract": {"audience": "researcher", "target_seconds": 1200},
    "institutional_policy_review": {"audience": "administrator", "target_seconds": 900},
    "offline_update_verification": {"audience": "administrator", "target_seconds": 600},
}

EVENT_TYPES = frozenset({
    "consent", "task_started", "task_completed", "task_abandoned",
    "task_error", "trust_rating", "method_correction", "accessibility_observation",
    "privacy_incident", "issue_opened", "issue_resolved",
})
AUDIENCES = frozenset({"researcher", "administrator"})
ACCESSIBILITY_MODES = frozenset({
    "none", "keyboard", "screen_reader", "magnification", "voice_control", "other",
})
ERROR_CATEGORIES = frozenset({
    "none", "navigation", "understanding", "methodology", "integration",
    "performance", "security_policy", "accessibility", "unexpected_failure",
})
ISSUE_SEVERITIES = frozenset({"critical", "high", "medium", "low"})


class BetaStudyError(ValueError):
    """A study event or log failed the privacy/integrity contract."""


def participant_token(study_secret: str, participant_code: str) -> str:
    """Pseudonymize an external study code without retaining the source code."""
    if len(study_secret) < 16:
        raise BetaStudyError("study_secret must contain at least 16 characters")
    code = participant_code.strip()
    if not 1 <= len(code) <= 128:
        raise BetaStudyError("participant_code must contain 1 to 128 characters")
    return hashlib.sha256(f"{study_secret}\0{code}".encode("utf-8")).hexdigest()


def _canonical(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _validate_event(event: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "participant_token", "audience", "event_type", "task_id", "success",
        "duration_seconds", "trust_rating", "method_correction_required",
        "accessibility_mode", "error_category", "consent_version",
        "issue_token", "issue_severity",
    }
    unknown = set(event) - allowed
    if unknown:
        raise BetaStudyError(f"unsupported event fields: {', '.join(sorted(unknown))}")
    token = event.get("participant_token")
    if not isinstance(token, str) or len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise BetaStudyError("participant_token must be a lowercase SHA-256 digest")
    audience = event.get("audience")
    if audience not in AUDIENCES:
        raise BetaStudyError("audience must be researcher or administrator")
    event_type = event.get("event_type")
    if event_type not in EVENT_TYPES:
        raise BetaStudyError("unsupported event_type")
    task_id = event.get("task_id")
    if event_type != "consent":
        if task_id not in TASKS:
            raise BetaStudyError("a recognized task_id is required")
        if TASKS[str(task_id)]["audience"] != audience:
            raise BetaStudyError("task_id is not valid for this audience")
    elif task_id is not None:
        raise BetaStudyError("consent events cannot identify a task")
    consent_version = event.get("consent_version")
    if event_type == "consent":
        if consent_version != PROTOCOL_VERSION:
            raise BetaStudyError("consent_version must match the active protocol")
    elif consent_version is not None:
        raise BetaStudyError("consent_version is valid only on consent events")
    duration = event.get("duration_seconds")
    if duration is not None and (
        not isinstance(duration, (int, float)) or isinstance(duration, bool)
        or not math.isfinite(float(duration)) or not 0 <= float(duration) <= 86400
    ):
        raise BetaStudyError("duration_seconds must be finite and between 0 and 86400")
    rating = event.get("trust_rating")
    if rating is not None and (
        not isinstance(rating, int) or isinstance(rating, bool) or not 1 <= rating <= 7
    ):
        raise BetaStudyError("trust_rating must be an integer from 1 to 7")
    if event.get("accessibility_mode", "none") not in ACCESSIBILITY_MODES:
        raise BetaStudyError("unsupported accessibility_mode")
    if event.get("error_category", "none") not in ERROR_CATEGORIES:
        raise BetaStudyError("unsupported error_category")
    for key in ("success", "method_correction_required"):
        if key in event and not isinstance(event[key], bool):
            raise BetaStudyError(f"{key} must be boolean")
    event_fields = {
        "consent": {"consent_version"},
        "task_started": {"task_id", "accessibility_mode"},
        "task_completed": {"task_id", "success", "duration_seconds", "accessibility_mode"},
        "task_abandoned": {"task_id", "duration_seconds", "error_category"},
        "task_error": {"task_id", "error_category"},
        "trust_rating": {"task_id", "trust_rating"},
        "method_correction": {"task_id", "method_correction_required"},
        "accessibility_observation": {"task_id", "accessibility_mode", "success"},
        "privacy_incident": {"task_id", "error_category"},
        "issue_opened": {"task_id", "issue_token", "issue_severity"},
        "issue_resolved": {"task_id", "issue_token", "issue_severity"},
    }
    common = {"participant_token", "audience", "event_type"}
    allowed_for_type = common | event_fields[str(event_type)]
    misplaced = set(event) - allowed_for_type
    if misplaced:
        raise BetaStudyError(
            f"fields are not valid for {event_type}: {', '.join(sorted(misplaced))}"
        )
    required_for_type = {
        "consent": {"consent_version"},
        "task_started": {"task_id"},
        "task_completed": {"task_id", "success", "duration_seconds"},
        "task_abandoned": {"task_id", "duration_seconds", "error_category"},
        "task_error": {"task_id", "error_category"},
        "trust_rating": {"task_id", "trust_rating"},
        "method_correction": {"task_id", "method_correction_required"},
        "accessibility_observation": {"task_id", "accessibility_mode"},
        "privacy_incident": {"task_id", "error_category"},
        "issue_opened": {"task_id", "issue_token", "issue_severity"},
        "issue_resolved": {"task_id", "issue_token", "issue_severity"},
    }[str(event_type)]
    missing = required_for_type - set(event)
    if missing:
        raise BetaStudyError(
            f"missing fields for {event_type}: {', '.join(sorted(missing))}"
        )
    if event_type in {"task_error", "task_abandoned", "privacy_incident"} and event.get(
        "error_category",
    ) == "none":
        raise BetaStudyError("an explicit non-none error_category is required")
    if event_type == "accessibility_observation" and event.get("accessibility_mode") == "none":
        raise BetaStudyError("an accessibility mode is required")
    if event_type in {"issue_opened", "issue_resolved"}:
        issue_token = event.get("issue_token")
        if not isinstance(issue_token, str) or len(issue_token) != 64 or any(
            character not in "0123456789abcdef" for character in issue_token
        ):
            raise BetaStudyError("issue_token must be a lowercase SHA-256 digest")
        if event.get("issue_severity") not in ISSUE_SEVERITIES:
            raise BetaStudyError("unsupported issue_severity")
    return dict(event)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw = read_bytes_no_follow(path, max_bytes=MAX_EVENT_LOG_BYTES)
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise BetaStudyError("study log is unsafe or exceeds its size limit") from exc
    rows: list[dict[str, Any]] = []
    previous = ZERO_HASH
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise BetaStudyError("study log is not valid UTF-8") from exc
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BetaStudyError("study log contains malformed JSON") from exc
        if not isinstance(row, dict) or row.get("previous_sha256") != previous:
            raise BetaStudyError("study log hash chain is broken")
        digest = row.get("event_sha256")
        unsigned = {key: value for key, value in row.items() if key != "event_sha256"}
        expected = hashlib.sha256(_canonical(unsigned)).hexdigest()
        if digest != expected:
            raise BetaStudyError("study log event digest is invalid")
        previous = digest
        rows.append(row)
    return rows


def record_event(path: Path, event: Mapping[str, Any]) -> dict[str, Any]:
    """Append one consented, fixed-schema event to a local tamper-evident log."""
    path = Path(path)
    validated = _validate_event(event)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with exclusive_file_lock(lock_path):
        rows = _read_rows(path)
        token = validated["participant_token"]
        if validated["event_type"] != "consent" and not any(
            row.get("participant_token") == token and row.get("event_type") == "consent"
            and row.get("consent_version") == PROTOCOL_VERSION for row in rows
        ):
            raise BetaStudyError("active-protocol consent must be recorded first")
        event_type = validated["event_type"]
        task_id = validated.get("task_id")
        if task_id is not None:
            task_history = [
                row for row in rows
                if row.get("participant_token") == token and row.get("task_id") == task_id
            ]
            started = sum(row["event_type"] == "task_started" for row in task_history)
            closed = sum(
                row["event_type"] in {"task_completed", "task_abandoned"}
                for row in task_history
            )
            if event_type == "task_started" and started > closed:
                raise BetaStudyError("the participant already has an active task attempt")
            if event_type in {"task_completed", "task_abandoned"} and started <= closed:
                raise BetaStudyError("task completion or abandonment requires an active attempt")
            if event_type not in {
                "task_started", "task_completed", "task_abandoned",
            } and started == 0:
                raise BetaStudyError("task observations require a recorded task start")
        previous = rows[-1]["event_sha256"] if rows else ZERO_HASH
        row = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "recorded_at": datetime.now(UTC).isoformat(),
            **validated,
            "previous_sha256": previous,
        }
        row["event_sha256"] = hashlib.sha256(_canonical(row)).hexdigest()
        encoded = _canonical(row) + b"\n"
        current_size = path.stat().st_size if path.exists() else 0
        if current_size + len(encoded) > MAX_EVENT_LOG_BYTES:
            raise BetaStudyError("study log size limit reached")
        append_bytes_no_follow(path, encoded, mode=0o600, sync=True)
        if os.name != "nt":
            os.chmod(path, 0o600)
        return row


def summarize_study(path: Path) -> dict[str, Any]:
    """Create an aggregate-only summary, suppressing cohorts below five."""
    rows = _read_rows(Path(path))
    participants = {row["participant_token"] for row in rows if row["event_type"] == "consent"}
    base = {
        "format": "sift-beta-study-summary",
        "schema_version": SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "participant_count": len(participants),
        "minimum_export_cohort": MIN_EXPORT_COHORT,
        "source_chain_tip": rows[-1]["event_sha256"] if rows else ZERO_HASH,
        "contains_participant_tokens": False,
        "contains_free_text": False,
    }
    if len(participants) < MIN_EXPORT_COHORT:
        return {**base, "status": "suppressed_small_cohort", "task_metrics": {}}
    metrics: dict[str, Any] = {}
    for task_id in TASKS:
        task_rows = [row for row in rows if row.get("task_id") == task_id]
        completions = [row for row in task_rows if row["event_type"] == "task_completed"]
        attempts = [row for row in task_rows if row["event_type"] == "task_started"]
        durations = sorted(float(row["duration_seconds"]) for row in completions if "duration_seconds" in row)
        metrics[task_id] = {
            "attempts": len(attempts),
            "completions": len(completions),
            "completion_rate": len(completions) / len(attempts) if attempts else None,
            "median_duration_seconds": durations[len(durations) // 2] if durations else None,
            "method_corrections": sum(
                row["event_type"] == "method_correction" for row in task_rows
            ),
            "errors": sum(row["event_type"] == "task_error" for row in task_rows),
        }
    privacy_incidents = sum(row["event_type"] == "privacy_incident" for row in rows)
    open_issues: dict[str, str] = {}
    for row in rows:
        if row["event_type"] == "issue_opened":
            open_issues[row["issue_token"]] = row["issue_severity"]
        elif row["event_type"] == "issue_resolved":
            if open_issues.get(row["issue_token"]) == row["issue_severity"]:
                open_issues.pop(row["issue_token"], None)
    unresolved_counts = {
        severity: sum(value == severity for value in open_issues.values())
        for severity in sorted(ISSUE_SEVERITIES)
    }
    completed = sum(value["completions"] for value in metrics.values())
    attempted = sum(value["attempts"] for value in metrics.values())
    completion_rate = completed / attempted if attempted else None
    release_ready = bool(
        completion_rate is not None and completion_rate >= 0.90
        and privacy_incidents == 0
        and unresolved_counts["critical"] == 0
        and unresolved_counts["high"] == 0
        and all(value["attempts"] >= 5 for value in metrics.values())
    )
    return {
        **base,
        "status": "ready",
        "task_metrics": metrics,
        "release_thresholds": {
            "overall_completion_rate_minimum": 0.90,
            "privacy_incidents_maximum": 0,
            "critical_unresolved_issues_maximum": 0,
        },
        "overall_completion_rate": completion_rate,
        "privacy_incidents": privacy_incidents,
        "unresolved_issue_counts": unresolved_counts,
        "release_ready": release_ready,
    }


def export_study_summary(path: Path, output: Path) -> dict[str, Any]:
    summary = summarize_study(path)
    atomic_write_json(Path(output), summary)
    return summary


__all__ = [
    "ACCESSIBILITY_MODES", "AUDIENCES", "BetaStudyError", "ERROR_CATEGORIES",
    "EVENT_TYPES", "ISSUE_SEVERITIES", "MIN_EXPORT_COHORT", "PROTOCOL_VERSION", "TASKS",
    "export_study_summary", "participant_token", "record_event", "summarize_study",
]
