from __future__ import annotations

import json
import asyncio
import os
from pathlib import Path

import pytest

from sift.research_workflow import (
    WorkflowError,
    approve_workflow,
    execution_context,
    list_evidence_claims,
    propose_workflow,
    read_project_memory,
    read_workflow,
    record_evidence_claim,
)
from sift.store import get_store
from sift.config import use_cwd


def _spec(**updates):
    value = {
        "research_question": "How is x associated with y?",
        "unit_of_analysis": "person",
        "outcome": "y",
        "exposures": ["x"],
        "treatment": None,
        "predictors": ["x"],
        "controls": [],
        "target_population": "eligible adults",
        "estimand": "adjusted mean difference in y per unit x",
        "study_design": "cross-sectional observational",
        "goal": "associational",
        "repeated_measures": False,
        "clusters": None,
        "weights": None,
        "strata": None,
        "psu": None,
        "fpc": None,
        "replicate_weights": None,
        "time_ordering": "x measured before or with y; causal order unresolved",
        "missing_data_assumption": "MAR for adjusted analysis",
    }
    value.update(updates)
    return value


def _proposal(**updates):
    value = {
        "method_id": "linear_regression",
        "research_specification": _spec(),
        "assumptions": ["Measurements are valid for their intended constructs."],
        "unresolved_quality_issues": [],
        "analyses": [
            {
                "id": "primary",
                "title": "Primary adjusted linear model",
                "role": "primary",
                "rationale": "Directly estimates the declared adjusted association.",
                "changes": [],
                "seed": 92831,
            },
            {
                "id": "robustness",
                "title": "Robust standard-error sensitivity",
                "role": "sensitivity",
                "rationale": "Checks sensitivity to the variance estimator.",
                "changes": ["Use HC3 standard errors"],
                "seed": 92831,
            },
        ],
    }
    value.update(updates)
    return value


def test_proposal_is_structured_persistent_and_requires_researcher_approval(
    tmp_path: Path,
) -> None:
    summary = propose_workflow(tmp_path, _proposal())
    assert summary["state"] == "awaiting_approval"
    assert summary["raw_data_included"] is False
    assert {row["id"] for row in summary["choices"]} >= {
        "estimand_definition", "method_selection", "primary_analysis",
        "missing_data_strategy",
    }
    document = read_workflow(tmp_path)
    assert document and document["intent"]["estimand"]
    assert document["analyses"][0]["role"] == "primary"
    assert document["analyses"][1]["role"] == "sensitivity"
    assert document["analyses"][0]["seed"] == 92831
    if os.name != "nt":
        assert (
            (tmp_path / ".sift" / "research_workflow.json").stat().st_mode
            & 0o077
            == 0
        )


def test_workflow_drops_unknown_fields_and_rejects_observation_like_lists(tmp_path: Path) -> None:
    proposal = _proposal()
    proposal["research_specification"]["raw_rows"] = [{"x": 123456789}]
    propose_workflow(tmp_path, proposal)
    persisted = read_workflow(tmp_path)
    assert "raw_rows" not in persisted["research_specification"]
    bad = _proposal()
    bad["research_specification"]["predictors"] = list(range(20))
    with pytest.raises(WorkflowError, match="bounded list of names"):
        propose_workflow(tmp_path, bad)


def test_approval_is_bound_to_exact_revision_and_execution_contract(tmp_path: Path) -> None:
    proposed = propose_workflow(tmp_path, _proposal())
    approved = approve_workflow(
        tmp_path, proposed["workflow_id"], proposed["revision"],
    )
    assert approved["state"] == "ready"
    context = execution_context(
        tmp_path, proposed["workflow_id"], "linear_regression", _spec(),
        ["primary", "robustness"],
    )
    assert context["analyses"][0]["role"] == "primary"
    assert context["analyses"][1]["role"] == "sensitivity"
    with pytest.raises(WorkflowError, match="differs"):
        execution_context(
            tmp_path, proposed["workflow_id"], "linear_regression",
            _spec(estimand="a changed estimand"), ["primary"],
        )


def test_model_tool_can_propose_and_read_but_cannot_approve(tmp_path: Path) -> None:
    from sift.tools import HANDLERS

    async def call(args):
        with use_cwd(tmp_path):
            return await HANDLERS["update_research_workflow"](args)

    proposed = json.loads(asyncio.run(call({
        "operation": "propose", "workflow": _proposal(),
    }))["content"][0]["text"])
    assert proposed["status"] == "awaiting_researcher_approval"
    assert proposed["approval"] is None
    resumed = json.loads(asyncio.run(call({
        "operation": "read",
    }))["content"][0]["text"])
    assert resumed["workflow_id"] == proposed["workflow_id"]
    assert resumed["state"] == "awaiting_approval"


def test_registry_execution_is_blocked_before_subprocess_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift.tools import HANDLERS
    import sift.tools as tool_module

    called = False

    async def should_not_execute(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("executor must not run")

    monkeypatch.setattr(tool_module, "_execute_script_for_submit", should_not_execute)

    async def call():
        with use_cwd(tmp_path):
            return await HANDLERS["submit_script"]({
                "language": "Python", "code": "print('must not run')",
                "label": "blocked", "method_id": "linear_regression",
                "research_specification": _spec(),
            })

    response = json.loads(asyncio.run(call())["content"][0]["text"])
    assert response["status"] == "needs_researcher_approval"
    assert called is False


def test_material_revision_invalidates_prior_approval(tmp_path: Path) -> None:
    first = propose_workflow(tmp_path, _proposal())
    approve_workflow(tmp_path, first["workflow_id"], first["revision"])
    changed = _proposal()
    changed["analyses"][1]["changes"] = ["Exclude influential clusters"]
    second = propose_workflow(tmp_path, changed)
    assert second["revision"] == first["revision"] + 1
    assert second["state"] == "awaiting_approval"
    assert second["approval"] is None
    with pytest.raises(WorkflowError, match="not execution-ready"):
        execution_context(
            tmp_path, second["workflow_id"], "linear_regression", _spec(),
            ["primary"],
        )


def test_exact_noop_revision_retains_approval(tmp_path: Path) -> None:
    first = propose_workflow(tmp_path, _proposal())
    approve_workflow(tmp_path, first["workflow_id"], first["revision"])
    second = propose_workflow(tmp_path, _proposal())
    assert second["state"] == "ready"
    assert second["approval"]["content_sha256"]


def test_open_critical_quality_issue_blocks_even_after_approval(tmp_path: Path) -> None:
    proposal = _proposal(unresolved_quality_issues=[{
        "id": "duplicate_panel_time",
        "summary": "Panel-time keys are not unique.",
        "severity": "critical",
        "status": "open",
    }])
    summary = propose_workflow(tmp_path, proposal)
    approved = approve_workflow(tmp_path, summary["workflow_id"], summary["revision"])
    assert approved["state"] == "blocked"
    assert any("Critical data-quality" in item for item in approved["blockers"])


def test_non_descriptive_workflow_requires_sensitivity_plan(tmp_path: Path) -> None:
    proposal = _proposal(analyses=[_proposal()["analyses"][0]])
    summary = propose_workflow(tmp_path, proposal)
    approved = approve_workflow(tmp_path, summary["workflow_id"], summary["revision"])
    assert approved["state"] == "blocked"
    assert any("sensitivity" in item for item in approved["blockers"])


def test_project_memory_contains_only_method_state(tmp_path: Path) -> None:
    propose_workflow(tmp_path, _proposal())
    memory = read_project_memory(tmp_path)
    assert memory["privacy"]["contains_raw_data"] is False
    assert memory["workflow"]["method_id"] == "linear_regression"
    from sift.ui import _build_context_prefix
    resumed = _build_context_prefix(tmp_path)
    assert "Methodological state at resume" in resumed
    assert "linear_regression" in resumed


def test_claims_require_existing_evidence_uncertainty_and_limitations(tmp_path: Path) -> None:
    store = get_store(tmp_path)
    row = store.insert(
        label="Association", analysis_type="linear_regression",
        sanitized_payload={
            "type": "linear_regression", "n": 100,
            "coefficients": {"x": 1.2}, "standard_errors": {"x": .2},
            "p_values": {"x": .01},
        },
        language="Python", script_code="# fit", transformations=[],
    )
    claim = record_evidence_claim(
        tmp_path, statement="X is positively associated with Y.",
        result_ids=[row.id], uncertainty="Sampling uncertainty remains.",
        limitations=["Observational design."], claim_type="associational",
    )
    assert claim["status"] == "supported"
    assert claim["result_ids"] == [row.id]
    with pytest.raises(WorkflowError, match="missing or hidden"):
        record_evidence_claim(
            tmp_path, statement="Unsupported", result_ids=["M999"],
            uncertainty="Unknown", limitations=["No evidence"],
            claim_type="associational",
        )
    with pytest.raises(WorkflowError, match="causal wording"):
        record_evidence_claim(
            tmp_path, statement="X causes Y.", result_ids=[row.id],
            uncertainty="Sampling uncertainty remains.",
            limitations=["Observational design."], claim_type="causal",
        )


def test_result_provenance_and_correction_chain_are_durable(tmp_path: Path) -> None:
    store = get_store(tmp_path)
    provenance = {
        "workflow_id": "wf-test", "workflow_revision": 2,
        "analysis_id": "primary", "analysis_role": "primary", "random_seed": 42,
        "dataset_hashes": {"data.csv": "a" * 64},
        "environment": {"python": "3.14.0", "packages": {"pandas": "3.0"}},
        "schema_verified": True, "local_verifier": "sift.verify_payload",
    }
    old = store.insert(
        label="Original", analysis_type="descriptive",
        sanitized_payload={"type": "descriptive", "n": 100, "mean": 2.0},
        language="Python", script_code="# original", transformations=[],
        provenance=provenance,
    )
    new = store.insert(
        label="Corrected", analysis_type="descriptive",
        sanitized_payload={"type": "descriptive", "n": 100, "mean": 2.1},
        language="Python", script_code="# corrected", transformations=[],
        provenance={**provenance, "analysis_id": "correction"},
    )
    record_evidence_claim(
        tmp_path, statement="The original mean is 2.0.", result_ids=[old.id],
        uncertainty="Sampling uncertainty remains.", limitations=["Descriptive only."],
        claim_type="descriptive",
    )
    link = store.supersede_result(
        old.id, new.id, reason="Corrected the declared unit conversion.",
        correction=True,
    )
    assert link["old_status"] == "corrected"
    loaded_old = store.get(old.id)
    loaded_new = store.get(new.id)
    assert loaded_old.lifecycle_status == "corrected"
    assert loaded_old.superseded_by == new.id
    assert loaded_new.supersedes_result_id == old.id
    assert loaded_new.provenance["dataset_hashes"]["data.csv"] == "a" * 64
    assert loaded_old.script_code == "# original"
    assert list_evidence_claims(tmp_path)[0]["status"] == "superseded_evidence"
    with pytest.raises(WorkflowError, match="superseded or corrected"):
        record_evidence_claim(
            tmp_path, statement="Stale", result_ids=[old.id],
            uncertainty="Unknown", limitations=["Corrected"],
            claim_type="descriptive",
        )
    with pytest.raises(ValueError, match="already superseded"):
        store.supersede_result(old.id, new.id, reason="again")
