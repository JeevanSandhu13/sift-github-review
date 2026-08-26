from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


INVENTORY = Path(__file__).resolve().parents[1] / "docs" / "qualification_inventory.json"
PROJECT_ROOT = INVENTORY.parents[1]
LIVE_DATABASE_REPORT = PROJECT_ROOT / "docs" / "live_database_certification.json"


def _load() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


def test_qualification_inventory_preserves_full_scope() -> None:
    ledger = _load()
    assert ledger["schema_version"] == 2
    assert ledger["accounting"] == {
        "stages": 18,
        "action_lines": 747,
        "completion_gates": 17,
        "note": (
            "Counts preserve every addressable qualification requirement, "
            "including section-level requirements."
        ),
    }
    assert [stage["id"] for stage in ledger["stages"]] == list(range(18))


def test_ledger_ids_are_unique_and_statuses_are_closed_vocab() -> None:
    ledger = _load()
    allowed = set(ledger["status_values"])
    ids: list[str] = []
    for stage in ledger["stages"]:
        assert stage["status"] in allowed
        for item in stage["items"]:
            ids.append(item["id"])
            assert item["status"] in allowed
            assert item["kind"] in {"action", "gate"}
            assert item["text"].strip()
            assert isinstance(item["evidence"], list)
            if item["status"] == "in_progress":
                assert isinstance(item.get("gap"), str) and item["gap"].strip()
            if item["status"] == "external_validation":
                assert item["evidence"]
    assert len(ids) == len(set(ids))


def test_completed_ledger_evidence_points_to_present_sources() -> None:
    """Completion evidence must stay auditable as tests and files evolve.

    Historical test-count prose and deleted filenames can make a checklist look
    better supported than it is. Every completed record therefore needs at
    least one current project file or directory; qualified pytest node IDs are
    accepted only when their containing file remains present.
    """
    ledger = _load()
    missing: list[tuple[str, str]] = []
    for stage in ledger["stages"]:
        for item in stage["items"]:
            if item["status"] != "complete":
                continue
            for evidence in item["evidence"]:
                relative_path, separator, node = evidence.partition("::")
                path = PROJECT_ROOT / relative_path
                if not path.exists():
                    missing.append((item["id"], evidence))
                    continue
                if separator:
                    node_name = node.split("[", 1)[0]
                    if f"def {node_name}(" not in path.read_text(encoding="utf-8"):
                        missing.append((item["id"], evidence))
    assert not missing, f"completed ledger evidence is stale: {missing}"


def test_remote_database_program_is_external_and_passes_remain_auditable() -> None:
    """BYO support is external; any recorded pass remains evidence-bound.

    A source checkout may contain no generated wheel, or it may contain a newer
    wheel than the last optional vendor report while a release is being built.
    Neither state is evidence of a vendor pass.  Exact context equality is
    required only when the report actually claims a pass; the qualification
    runner separately refuses to preserve or record passes against stale
    artifacts.
    """
    ledger = _load()
    report = json.loads(LIVE_DATABASE_REPORT.read_text(encoding="utf-8"))
    preflight = json.loads(subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "database_qualification.py"),
            "--preflight",
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout)
    assert report["schema_version"] == 5
    assert report["program"] == "optional_external_compatibility"
    assert report["product_release_blocking"] is False
    assert preflight["program"] == "optional_external_compatibility"
    assert preflight["product_release_blocking"] is False
    assert isinstance(preflight["qualification_context_ready"], bool)
    recorded_passes = [
        row
        for row in report["remote_scenarios"]
        if row.get("status") == "passed"
    ]
    recorded_passes.extend(
        result
        for row in report["remote_scenarios"]
        for result in row.get("authentication_results", [])
        if isinstance(result, dict) and result.get("status") == "passed"
    )
    if recorded_passes:
        assert preflight["qualification_context_ready"] is True
        assert report["qualification_context"] == preflight["qualification_context"]
    assert isinstance(report["qualification_context"], dict)
    evidence_digest = report["qualification_context"][
        "qualification_context_sha256"
    ]
    report_rows = {
        row["step_id"]: row for row in report["remote_scenarios"]
    }
    database_items = {
        item["id"]: item for item in ledger["stages"][4]["items"]
    }
    remote_ids = [f"S04-{number:03d}" for number in range(58, 105)]
    assert set(remote_ids) <= set(report_rows)
    for step_id in remote_ids:
        assert database_items[step_id]["status"] == "external_validation"
        if report_rows[step_id]["status"] == "passed":
            row = report_rows[step_id]
            assert row["configured"] is True, step_id
            assert row["missing_environment"] == [], step_id
            assert row["checked_at"], step_id
            assert row["evidence_context_sha256"] == evidence_digest, step_id
            if row.get("mode") == "auth" and str(row.get("test_node", "")).startswith(
                "aggregate:authentication_results"
            ):
                expected_variants = set(row["authentication_variants"])
                results = {
                    result["variant"]: result
                    for result in row.get("authentication_results", [])
                }
                assert set(results) == expected_variants, step_id
                assert all(
                    result["status"] == "passed"
                    and result["configured"] is True
                    and result["missing_environment"] == []
                    and result["evidence_context_sha256"] == evidence_digest
                    and result["test_node"].endswith(
                        f"[{step_id}-{variant}]"
                    )
                    for variant, result in results.items()
                ), step_id
            else:
                assert row["test_node"].endswith(f"[{step_id}]"), step_id

    assert database_items["S04-048"]["status"] == "external_validation"
    assert database_items["S04-GATE"]["status"] == "external_validation"
    stage = ledger["stages"][4]
    assert stage["database_product_contract"] == {
        "ownership": "bring_your_own_database",
        "database_account": "researcher_owned_or_researcher_authorized",
        "credentials": "researcher_supplied",
        "billing": "researcher_to_database_operator",
        "sift_operated_database_or_proxy": False,
        "live_vendor_certification_release_blocking": False,
    }
    assert stage["external_validation_program"]["release_blocking"] is False


def test_live_agent_evaluation_claims_follow_the_qualification_artifact() -> None:
    """A tested harness is not evidence that provider/model runs occurred."""
    ledger = _load()
    evaluation_items = {
        item["id"]: item for item in ledger["stages"][16]["items"]
    }
    live_agent_ids = {"S16-024", "S16-025"}
    assert all(
        evaluation_items[item_id]["status"] == "blocked"
        for item_id in live_agent_ids
    )
    assert ledger["stages"][16]["status"] == "in_progress"
    assert evaluation_items["S16-GATE"]["status"] == "blocked"
    assert all(
        item_id in evaluation_items["S16-GATE"]["blocker"]
        for item_id in ("S16-015", "S16-024", "S16-025")
    )


def test_ledger_matches_audited_stage_progress() -> None:
    ledger = _load()
    completed = ledger["stages"][:4]
    assert [stage["id"] for stage in completed] == [0, 1, 2, 3]
    for stage in completed:
        assert stage["status"] == "complete"
        assert all(item["status"] == "complete" for item in stage["items"])
        assert all(item["evidence"] for item in stage["items"])

    database_stage = ledger["stages"][4]
    assert database_stage["id"] == 4
    assert database_stage["status"] == "complete"
    assert all(
        item["evidence"]
        for item in database_stage["items"]
        if item["status"] != "not_started"
    )
    external_ids = {
        "S04-048", "S04-GATE",
        *(f"S04-{number:03d}" for number in range(58, 105)),
    }
    assert {
        item["id"] for item in database_stage["items"]
        if item["status"] == "external_validation"
    } == external_ids
    assert all(
        item["status"] in {"complete", "external_validation"}
        for item in database_stage["items"]
    )
    assert not any(item["status"] == "blocked" for item in database_stage["items"])

    cloud_stage = ledger["stages"][5]
    assert cloud_stage["id"] == 5
    assert cloud_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in cloud_stage["items"])
    assert all(item["evidence"] for item in cloud_stage["items"])

    research_stage = ledger["stages"][6]
    assert research_stage["id"] == 6
    assert research_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in research_stage["items"])
    assert all(item["evidence"] for item in research_stage["items"])

    format_stage = ledger["stages"][7]
    assert format_stage["id"] == 7
    assert format_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in format_stage["items"])
    assert all(item["evidence"] for item in format_stage["items"])

    canonical_stage = ledger["stages"][8]
    assert canonical_stage["id"] == 8
    assert canonical_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in canonical_stage["items"])
    assert all(item["evidence"] for item in canonical_stage["items"])

    quality_stage = ledger["stages"][9]
    assert quality_stage["id"] == 9
    assert quality_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in quality_stage["items"])
    assert all(item["evidence"] for item in quality_stage["items"])

    methodology_stage = ledger["stages"][10]
    assert methodology_stage["id"] == 10
    assert methodology_stage["status"] == "complete"
    methodology_in_progress: set[str] = set()
    assert {
        item["id"] for item in methodology_stage["items"]
        if item["status"] == "in_progress"
    } == methodology_in_progress
    assert all(
        isinstance(item.get("gap"), str) and item["gap"].strip()
        for item in methodology_stage["items"]
        if item["status"] == "in_progress"
    )
    assert all(
        item["status"] in {"complete", "in_progress"}
        for item in methodology_stage["items"]
    )
    assert all(item["evidence"] for item in methodology_stage["items"])

    workflow_stage = ledger["stages"][11]
    assert workflow_stage["id"] == 11
    assert workflow_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in workflow_stage["items"])
    assert all(item["evidence"] for item in workflow_stage["items"])

    security_stage = ledger["stages"][12]
    assert security_stage["id"] == 12
    assert security_stage["status"] == "in_progress"
    assert all(item["evidence"] for item in security_stage["items"])
    assert [
        item["id"] for item in security_stage["items"]
        if item["status"] == "blocked"
    ] == ["S12-040", "S12-041"]
    assert security_stage["items"][-1]["id"] == "S12-GATE"
    assert security_stage["items"][-1]["status"] == "complete"

    reproducibility_stage = ledger["stages"][13]
    assert reproducibility_stage["id"] == 13
    assert reproducibility_stage["status"] == "complete"
    assert all(
        item["status"] == "complete" for item in reproducibility_stage["items"]
    )
    assert all(item["evidence"] for item in reproducibility_stage["items"])

    reliability_stage = ledger["stages"][14]
    assert reliability_stage["id"] == 14
    assert reliability_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in reliability_stage["items"])
    assert all(item["evidence"] for item in reliability_stage["items"])

    performance_stage = ledger["stages"][15]
    assert performance_stage["id"] == 15
    assert performance_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in performance_stage["items"])
    assert all(item["evidence"] for item in performance_stage["items"])

    evaluation_stage = ledger["stages"][16]
    assert evaluation_stage["id"] == 16
    assert evaluation_stage["status"] == "in_progress"
    assert {
        item["id"] for item in evaluation_stage["items"]
        if item["status"] == "in_progress"
    } == set()
    assert {
        item["id"] for item in evaluation_stage["items"]
        if item["status"] == "blocked"
    } == {"S16-015", "S16-024", "S16-025", "S16-GATE"}
    assert all(
        isinstance(item.get("gap"), str) and item["gap"].strip()
        for item in evaluation_stage["items"]
        if item["status"] == "in_progress"
    )
    assert all(
        isinstance(item.get("blocker"), str) and item["blocker"].strip()
        for item in evaluation_stage["items"]
        if item["status"] == "blocked"
    )
    assert all(
        item["status"] in {"complete", "in_progress", "blocked"}
        for item in evaluation_stage["items"]
    )
    assert all(item["evidence"] for item in evaluation_stage["items"])

    backend_stage = ledger["stages"][17]
    assert backend_stage["id"] == 17
    assert backend_stage["status"] == "complete"
    assert all(item["status"] == "complete" for item in backend_stage["items"])
    assert all(item["evidence"] for item in backend_stage["items"])
