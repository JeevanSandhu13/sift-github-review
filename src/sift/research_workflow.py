"""Local, approval-bound research workflow state.

This module turns Sift's existing methodology, data-quality, execution, and
verification components into one durable research contract.  The contract is
metadata only: it contains questions, variable *names*, design choices, and
result identifiers, never observations or excerpts from a dataset.

The model may propose or revise a workflow.  It cannot approve one.  Approval
is a separate researcher-side operation bound to the exact consequential
content hash; changing the estimand, method, primary analysis, sensitivity
analyses, missing-data strategy, or unresolved quality decisions invalidates
the approval automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from sift.config import ensure_private_sift_dir
from sift.file_lock import exclusive_file_lock
from sift.secure_file import append_bytes_no_follow
from sift.methodology import evaluate_method, validate_research_specification
from sift.text_safety import safe_text

WORKFLOW_VERSION = 1
WORKFLOW_FILENAME = "research_workflow.json"
PROJECT_MEMORY_FILENAME = "project_memory.json"
CLAIMS_FILENAME = "evidence_claims.jsonl"

_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,79}$")
_ISSUE_STATUSES = frozenset({"open", "accepted", "resolved"})
_ANALYSIS_ROLES = frozenset({"primary", "sensitivity"})
_MAX_TEXT = 1000
_MAX_LIST = 50


class WorkflowError(ValueError):
    """A workflow cannot be persisted or used safely."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _clean_text(value: Any, *, field: str, required: bool = True,
                cap: int = _MAX_TEXT) -> str:
    if not isinstance(value, str):
        if required:
            raise WorkflowError(f"{field} must be text")
        return ""
    cleaned = safe_text(value, max_len=cap).strip()
    if required and not cleaned:
        raise WorkflowError(f"{field} must not be empty")
    return cleaned


def _clean_id(value: Any, *, field: str) -> str:
    cleaned = _clean_text(value, field=field, cap=80)
    if not _ID_RE.fullmatch(cleaned):
        raise WorkflowError(
            f"{field} must start with a letter and contain only letters, "
            "numbers, '.', '_', ':', or '-'"
        )
    return cleaned


def _text_list(value: Any, *, field: str, required: bool = False,
               cap: int = _MAX_LIST) -> list[str]:
    if value is None and not required:
        return []
    if not isinstance(value, list) or (required and not value):
        raise WorkflowError(f"{field} must be a{' non-empty' if required else ''} list")
    if len(value) > cap:
        raise WorkflowError(f"{field} has more than {cap} entries")
    result: list[str] = []
    for index, item in enumerate(value):
        cleaned = _clean_text(item, field=f"{field}[{index}]", cap=500)
        if cleaned not in result:
            result.append(cleaned)
    return result


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _clean_specification(raw: Any) -> dict[str, Any]:
    """Keep only methodology fields and scalar/identifier-list metadata."""
    validation = validate_research_specification(raw)
    specification = validation.get("specification")
    if not isinstance(specification, Mapping):
        raise WorkflowError("research_specification must be an object")
    clean: dict[str, Any] = {}
    for key, value in specification.items():
        if value is None or isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            clean[key] = value
        elif isinstance(value, str):
            clean[key] = _clean_text(
                value, field=f"research_specification.{key}", required=False,
                cap=1000,
            )
        elif isinstance(value, (list, tuple)):
            if len(value) > 100 or any(not isinstance(item, str) for item in value):
                raise WorkflowError(
                    f"research_specification.{key} must be a bounded list of names"
                )
            clean[key] = [
                _clean_text(item, field=f"research_specification.{key}", cap=200)
                for item in value
            ]
        else:
            raise WorkflowError(
                f"research_specification.{key} has an unsupported metadata type"
            )
    return clean


def _workflow_path(cwd: Path) -> Path:
    return Path(cwd) / ".sift" / WORKFLOW_FILENAME


def _lock_path(cwd: Path) -> Path:
    return Path(cwd) / ".sift" / f"{WORKFLOW_FILENAME}.lock"


def _atomic_json(path: Path, value: Any) -> None:
    ensure_private_sift_dir(path.parent.parent)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb", dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(json.dumps(
                value, ensure_ascii=False, indent=2, sort_keys=True,
            ).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
            temp_name = handle.name
        os.replace(temp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except BaseException:
        if temp_name:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        raise


def read_workflow(cwd: Path) -> dict[str, Any] | None:
    """Read the active workflow, returning ``None`` for absent/corrupt state."""
    try:
        value = json.loads(_workflow_path(cwd).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("version") != WORKFLOW_VERSION:
        return None
    return value


def _clean_issue(raw: Any, index: int) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise WorkflowError(f"unresolved_quality_issues[{index}] must be an object")
    status = str(raw.get("status") or "open")
    if status not in _ISSUE_STATUSES:
        raise WorkflowError(f"quality issue {index} has invalid status")
    severity = str(raw.get("severity") or "warning")
    if severity not in {"info", "warning", "critical"}:
        raise WorkflowError(f"quality issue {index} has invalid severity")
    return {
        "id": _clean_id(raw.get("id") or f"quality_{index + 1}",
                        field=f"quality issue {index} id"),
        "summary": _clean_text(raw.get("summary"),
                               field=f"quality issue {index} summary", cap=500),
        "severity": severity,
        "status": status,
    }


def _clean_analysis(raw: Any, index: int, method_id: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkflowError(f"analyses[{index}] must be an object")
    role = str(raw.get("role") or "")
    if role not in _ANALYSIS_ROLES:
        raise WorkflowError(f"analysis {index} role must be primary or sensitivity")
    seed = raw.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= 2**32 - 1):
        raise WorkflowError(f"analysis {index} needs an integer seed from 0 to 2^32-1")
    changes = _text_list(raw.get("changes"), field=f"analyses[{index}].changes")
    if role == "sensitivity" and not changes:
        raise WorkflowError(f"sensitivity analysis {index} must state what changes")
    return {
        "id": _clean_id(raw.get("id") or f"analysis_{index + 1}",
                        field=f"analysis {index} id"),
        "title": _clean_text(raw.get("title"), field=f"analysis {index} title", cap=240),
        "role": role,
        "method_id": _clean_id(raw.get("method_id") or method_id,
                               field=f"analysis {index} method_id"),
        "rationale": _clean_text(raw.get("rationale"),
                                 field=f"analysis {index} rationale", cap=500),
        "changes": changes,
        "seed": seed,
    }


def _clean_choice(raw: Any, index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise WorkflowError(f"additional_choices[{index}] must be an object")
    alternatives = _text_list(
        raw.get("alternatives"), field=f"additional_choices[{index}].alternatives",
        required=True, cap=12,
    )
    return {
        "id": _clean_id(raw.get("id") or f"choice_{index + 1}",
                        field=f"choice {index} id"),
        "question": _clean_text(raw.get("question"),
                                field=f"choice {index} question", cap=300),
        "proposed": _clean_text(raw.get("proposed"),
                                field=f"choice {index} proposed", cap=500),
        "alternatives": alternatives,
        "consequence": _clean_text(raw.get("consequence"),
                                   field=f"choice {index} consequence", cap=500),
        "consequential": bool(raw.get("consequential", True)),
    }


def _required_choices(method_id: str, specification: Mapping[str, Any],
                      analyses: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    primary = next(row for row in analyses if row["role"] == "primary")
    choices = [
        {
            "id": "estimand_definition",
            "question": "Is this the intended estimand?",
            "proposed": str(specification.get("estimand")),
            "alternatives": ["Revise the population, outcome, contrast, or time horizon"],
            "consequence": "The estimand determines what quantity the analysis can answer.",
            "consequential": True,
        },
        {
            "id": "method_selection",
            "question": "Approve the proposed primary method?",
            "proposed": method_id,
            "alternatives": ["Choose another compatible registry method"],
            "consequence": "The method fixes identifying assumptions, diagnostics, and claim limits.",
            "consequential": True,
        },
        {
            "id": "primary_analysis",
            "question": "Approve this primary analysis before examining its result?",
            "proposed": str(primary["title"]),
            "alternatives": ["Revise the primary specification or designate another analysis"],
            "consequence": "Pre-designating the primary analysis limits outcome-driven specification search.",
            "consequential": True,
        },
        {
            "id": "missing_data_strategy",
            "question": "Approve the missing-data assumption and strategy?",
            "proposed": str(specification.get("missing_data_assumption")),
            "alternatives": ["Revise the assumption or add a missing-data sensitivity analysis"],
            "consequence": "Missing-data handling can materially change the target population and estimates.",
            "consequential": True,
        },
    ]
    return choices


def _consequential_view(document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "intent": document["intent"],
        "method_id": document["method_id"],
        "research_specification": document["research_specification"],
        "assumptions": document["assumptions"],
        "quality_issues": document["quality_issues"],
        "analyses": document["analyses"],
        "choices": [row for row in document["choices"] if row["consequential"]],
    }


def consequential_hash(document: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_consequential_view(document))).hexdigest()


def _readiness(document: Mapping[str, Any]) -> tuple[bool, list[str]]:
    blockers: list[str] = []
    analyses = document.get("analyses") or []
    primary = [row for row in analyses if row.get("role") == "primary"]
    sensitivity = [row for row in analyses if row.get("role") == "sensitivity"]
    if len(primary) != 1:
        blockers.append("Exactly one primary analysis must be designated.")
    goal = str(document.get("research_specification", {}).get("goal") or "")
    if goal in {"inferential", "associational", "predictive", "causal"} and not sensitivity:
        blockers.append("At least one reasonable sensitivity analysis must be planned.")
    open_critical = [
        row["id"] for row in document.get("quality_issues", [])
        if row.get("severity") == "critical" and row.get("status") == "open"
    ]
    if open_critical:
        blockers.append(
            "Critical data-quality issues remain open: " + ", ".join(open_critical)
        )
    approval = document.get("approval")
    expected = consequential_hash(document)
    if not isinstance(approval, Mapping) or approval.get("content_sha256") != expected:
        blockers.append("Researcher approval is required for the current consequential choices.")
    return not blockers, blockers


def propose_workflow(cwd: Path, proposal: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and atomically persist a model-proposed workflow revision."""
    if not isinstance(proposal, Mapping):
        raise WorkflowError("workflow must be an object")
    method_id = _clean_id(proposal.get("method_id"), field="method_id")
    specification = _clean_specification(proposal.get("research_specification"))
    evaluated = evaluate_method(method_id, specification)
    if not evaluated.get("valid"):
        questions = evaluated.get("clarifications", [])
        raise WorkflowError("methodology is incomplete: " + "; ".join(questions[:8]))
    question = _clean_text(specification.get("research_question"),
                           field="research question")
    intent = {
        "research_question": question,
        "goal": _clean_text(specification.get("goal"), field="goal", cap=80),
        "unit_of_analysis": _clean_text(
            specification.get("unit_of_analysis"), field="unit of analysis", cap=200,
        ),
        "target_population": _clean_text(
            specification.get("target_population"), field="target population", cap=300,
        ),
        "estimand": _clean_text(specification.get("estimand"), field="estimand", cap=500),
    }
    assumptions = list(dict.fromkeys([
        *_text_list(proposal.get("assumptions"), field="assumptions"),
        *[str(value) for value in evaluated["contract"]["assumptions"]],
    ]))
    issues_raw = proposal.get("unresolved_quality_issues") or []
    if not isinstance(issues_raw, list) or len(issues_raw) > _MAX_LIST:
        raise WorkflowError("unresolved_quality_issues must be a bounded list")
    issues = [_clean_issue(row, index) for index, row in enumerate(issues_raw)]
    analyses_raw = proposal.get("analyses")
    if not isinstance(analyses_raw, list) or not analyses_raw or len(analyses_raw) > 20:
        raise WorkflowError("analyses must be a non-empty list with at most 20 entries")
    analyses = [_clean_analysis(row, index, method_id)
                for index, row in enumerate(analyses_raw)]
    if len({row["id"] for row in analyses}) != len(analyses):
        raise WorkflowError("analysis ids must be unique")
    if len([row for row in analyses if row["role"] == "primary"]) != 1:
        raise WorkflowError("exactly one analysis must have role='primary'")
    choices = _required_choices(method_id, specification, analyses)
    extra_raw = proposal.get("additional_choices") or []
    if not isinstance(extra_raw, list) or len(extra_raw) > 20:
        raise WorkflowError("additional_choices must be a bounded list")
    choices.extend(_clean_choice(row, index) for index, row in enumerate(extra_raw))
    if len({row["id"] for row in choices}) != len(choices):
        raise WorkflowError("choice ids must be unique")

    with exclusive_file_lock(_lock_path(cwd)):
        old = read_workflow(cwd)
        revision = int(old.get("revision", 0)) + 1 if old else 1
        workflow_id = str(old.get("workflow_id")) if old else (
            "wf-" + hashlib.sha256(
                f"{question}\0{_now()}".encode("utf-8")
            ).hexdigest()[:12]
        )
        document: dict[str, Any] = {
            "version": WORKFLOW_VERSION,
            "workflow_id": workflow_id,
            "revision": revision,
            "state": "awaiting_approval",
            "created_at": old.get("created_at", _now()) if old else _now(),
            "updated_at": _now(),
            "intent": intent,
            "method_id": method_id,
            "research_specification": dict(specification),
            "assumptions": assumptions,
            "quality_issues": issues,
            "analyses": analyses,
            "choices": choices,
            "method_contract": evaluated["contract"],
        }
        # Exact no-op proposals retain approval; any material change loses it.
        if old and old.get("approval") and consequential_hash(old) == consequential_hash(document):
            document["approval"] = old["approval"]
        ready, blockers = _readiness(document)
        document["state"] = "ready" if ready else "awaiting_approval"
        _atomic_json(_workflow_path(cwd), document)
        _write_project_memory(cwd, document)
    return workflow_summary(document, blockers=blockers)


def approve_workflow(cwd: Path, workflow_id: str, revision: int,
                     *, approved_by: str = "researcher") -> dict[str, Any]:
    """Researcher-side approval bound to one exact workflow revision."""
    with exclusive_file_lock(_lock_path(cwd)):
        document = read_workflow(cwd)
        if document is None:
            raise WorkflowError("no research workflow exists")
        if document.get("workflow_id") != workflow_id:
            raise WorkflowError("workflow id does not match the active workflow")
        if document.get("revision") != revision:
            raise WorkflowError("workflow changed; review and approve the latest revision")
        document["approval"] = {
            "approved_at": _now(),
            "approved_by": _clean_text(approved_by, field="approved_by", cap=120),
            "revision": revision,
            "content_sha256": consequential_hash(document),
        }
        ready, blockers = _readiness(document)
        document["state"] = "ready" if ready else "blocked"
        document["updated_at"] = _now()
        _atomic_json(_workflow_path(cwd), document)
        _write_project_memory(cwd, document)
    return workflow_summary(document, blockers=blockers)


def workflow_summary(document: Mapping[str, Any], *,
                     blockers: Sequence[str] | None = None) -> dict[str, Any]:
    """Return bounded, model-safe methodological state (no raw observations)."""
    if blockers is None:
        _ready, blockers = _readiness(document)
    return {
        "workflow_id": document.get("workflow_id"),
        "revision": document.get("revision"),
        "state": document.get("state"),
        "intent": document.get("intent"),
        "method_id": document.get("method_id"),
        "assumptions": document.get("assumptions", []),
        "quality_issues": document.get("quality_issues", []),
        "analyses": document.get("analyses", []),
        "choices": document.get("choices", []),
        "approval": document.get("approval"),
        "blockers": list(blockers),
        "raw_data_included": False,
    }


def execution_context(cwd: Path, workflow_id: str, method_id: str,
                      research_specification: Mapping[str, Any],
                      analysis_ids: Sequence[str] | None = None) -> dict[str, Any]:
    """Fail closed unless execution matches an approved research contract."""
    document = read_workflow(cwd)
    if document is None or document.get("workflow_id") != workflow_id:
        raise WorkflowError("no matching approved research workflow")
    ready, blockers = _readiness(document)
    if not ready:
        raise WorkflowError("workflow is not execution-ready: " + "; ".join(blockers))
    if document.get("method_id") != method_id:
        raise WorkflowError("method_id differs from the approved workflow")
    normalized_specification = _clean_specification(research_specification)
    if _canonical_json(document.get("research_specification")) != _canonical_json(normalized_specification):
        raise WorkflowError("research specification differs from the approved workflow")
    selected = list(analysis_ids or [])
    known = {row["id"]: row for row in document.get("analyses", [])}
    if not selected:
        raise WorkflowError("analysis_ids must identify the approved analyses being run")
    if len(selected) != len(set(selected)) or any(value not in known for value in selected):
        raise WorkflowError("analysis_ids contain duplicates or unapproved analyses")
    return {
        "workflow_id": workflow_id,
        "workflow_revision": document["revision"],
        "approval_sha256": document["approval"]["content_sha256"],
        "analyses": [known[value] for value in selected],
    }


def _write_project_memory(cwd: Path, document: Mapping[str, Any]) -> None:
    """Persist resumable methodological state assembled only from metadata."""
    memory = {
        "version": 1,
        "updated_at": _now(),
        "workflow": workflow_summary(document),
        "privacy": {
            "contains_raw_data": False,
            "included": "research design metadata, variable names, decisions, and result ids",
            "excluded": "observation values, row excerpts, dataset samples, credentials, and local paths",
        },
    }
    _atomic_json(Path(cwd) / ".sift" / PROJECT_MEMORY_FILENAME, memory)


def read_project_memory(cwd: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(
            (Path(cwd) / ".sift" / PROJECT_MEMORY_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def record_evidence_claim(cwd: Path, *, statement: str,
                          result_ids: Sequence[str], uncertainty: str,
                          limitations: Sequence[str], claim_type: str) -> dict[str, Any]:
    """Record a bounded narrative claim only when every citation exists.

    Semantic truth cannot be proven from syntax.  This function therefore
    makes the enforceable promise: a reportable claim must cite extant stored
    evidence, carry explicit uncertainty/limitations, and stay within the
    deterministic causality label of each cited result.
    """
    from sift.store import get_store
    from sift.verification import verify_payload

    if claim_type not in {"descriptive", "associational", "predictive", "causal"}:
        raise WorkflowError("invalid claim_type")
    ids = [_clean_id(value, field="result id") for value in result_ids]
    if not ids or len(ids) > 20 or len(ids) != len(set(ids)):
        raise WorkflowError("a claim needs 1-20 unique evidence result ids")
    store = get_store(cwd)
    rows = [store.get(value) for value in ids]
    missing = [value for value, row in zip(ids, rows) if row is None]
    if missing:
        raise WorkflowError("claim references missing or hidden evidence: " + ", ".join(missing))
    inactive = [row.id for row in rows if row and row.lifecycle_status != "active"]
    if inactive:
        raise WorkflowError(
            "claim references superseded or corrected evidence: " + ", ".join(inactive)
        )
    verifications = [verify_payload(row.sanitized_payload or {}) for row in rows if row]
    labels = {
        (verification or {}).get("causality", {}).get("label")
        for verification in verifications
    }
    if claim_type == "causal" and (
        not labels or any(label != "quasi_experimental" for label in labels)
    ):
        raise WorkflowError("causal wording is not supported by the cited evidence")
    if claim_type == "predictive" and any(
        (row.sanitized_payload or {}).get("method_family") != "predictive"
        for row in rows if row
    ):
        raise WorkflowError("predictive wording is not supported by every cited result")
    limits = _text_list(list(limitations), field="limitations", required=True, cap=20)
    claim = {
        "id": "claim-" + hashlib.sha256(
            f"{_now()}\0{statement}\0{','.join(ids)}".encode("utf-8")
        ).hexdigest()[:12],
        "created_at": _now(),
        "statement": _clean_text(statement, field="statement", cap=1000),
        "claim_type": claim_type,
        "result_ids": ids,
        "uncertainty": _clean_text(uncertainty, field="uncertainty", cap=500),
        "limitations": limits,
        "verification_levels": sorted(value for value in labels if value),
        "status": "supported",
    }
    path = Path(cwd) / ".sift" / CLAIMS_FILENAME
    ensure_private_sift_dir(Path(cwd))
    with exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        append_bytes_no_follow(
            path,
            (json.dumps(claim, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
            sync=True,
        )
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return claim


def list_evidence_claims(cwd: Path) -> list[dict[str, Any]]:
    """Read claims and dynamically invalidate those using replaced evidence."""
    from sift.store import get_store

    path = Path(cwd) / ".sift" / CLAIMS_FILENAME
    claims: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return claims
    store = get_store(cwd)
    for line in lines:
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if not isinstance(value, dict):
            continue
        stale: list[str] = []
        for result_id in value.get("result_ids", []):
            row = store.get(str(result_id), include_hidden=True)
            if row is None or row.hidden_at or row.lifecycle_status != "active":
                stale.append(str(result_id))
        value = dict(value)
        if stale:
            value["status"] = "superseded_evidence"
            value["stale_result_ids"] = stale
        claims.append(value)
    return claims


__all__ = [
    "WorkflowError", "approve_workflow", "consequential_hash", "execution_context",
    "list_evidence_claims", "propose_workflow", "read_project_memory", "read_workflow",
    "record_evidence_claim", "workflow_summary",
]
