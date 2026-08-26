from __future__ import annotations

import json
from pathlib import Path

import pytest
from hypothesis import given, strategies as st

from sift.evaluation import (
    METHOD_QUALIFICATION_BLOCKERS,
    METHOD_QUALIFICATION_SPECS,
    REQUIRED_METHOD_COVERAGE_CHECKS,
    benchmark_catalog,
    claim_candidate_quality,
    confidential_release_gate,
    evaluate_methodology_scenarios,
    evaluate_provider_agents,
    materialize_benchmark_library,
    run_scientific_qualification,
    scientific_release_gate,
    verify_benchmark_library,
    verify_method_test_evidence,
)
from sift.sanitizer import sanitize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
requires_current_method_evidence = pytest.mark.skipif(
    not verify_method_test_evidence(PROJECT_ROOT)["valid"],
    reason=(
        "requires current, source-bound Stage 10 method evidence; run "
        "scripts/method_qualification_evidence.py on the scientific "
        "qualification host"
    ),
)



def test_permanent_benchmark_library_covers_required_families(tmp_path: Path) -> None:
    catalog = benchmark_catalog()
    kinds = {row.kind for row in catalog}
    assert {
        "privacy_adversarial", "malformed", "known_answer",
        "repeated_measures", "survey", "causal_inference", "survival",
        "time_series", "geospatial", "high_dimensional", "domain_specific",
    } <= kinds
    manifest = materialize_benchmark_library(tmp_path / "fixtures")
    assert manifest["synthetic_data_only"] is True
    assert verify_benchmark_library(tmp_path / "fixtures")["valid"] is True


def test_benchmark_integrity_detects_mutation(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    manifest = materialize_benchmark_library(root)
    target = root / manifest["fixtures"][0]["filename"]
    target.write_bytes(target.read_bytes() + b"tampered")
    assert verify_benchmark_library(root)["valid"] is False


def test_benchmark_integrity_does_not_trust_a_rewritten_manifest(tmp_path: Path) -> None:
    root = tmp_path / "fixtures"
    manifest = materialize_benchmark_library(root)
    target = root / manifest["fixtures"][0]["filename"]
    target.write_bytes(b"replacement")
    import hashlib

    manifest["fixtures"][0]["sha256"] = hashlib.sha256(b"replacement").hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_benchmark_library(root)["valid"] is False


@given(st.text(min_size=1, max_size=500))
def test_privacy_property_unknown_fields_never_cross(secret: str) -> None:
    marker = f"PRIVATE-RAW-VALUE[[{secret}]]"
    result = sanitize({
        "type": "descriptive", "variable": "x", "n": 50,
        "mean": 1.0, "sd": 1.0, "missing_count": 0,
        "raw_record": marker,
    })
    assert result.ok
    assert "raw_record" not in (result.sanitized or {})
    assert marker not in json.dumps(result.sanitized or {})


def test_claim_quality_rejects_unsupported_causal_claim() -> None:
    result = claim_candidate_quality({
        "statement": "Exposure caused the result.", "result_ids": ["r1"],
        "uncertainty": "95% interval", "limitations": ["observational"],
        "claim_type": "causal", "verification_levels": ["observational"],
    })
    assert result["valid"] is False
    assert "causal_support" in result["failures"]


def test_repeated_agent_and_provider_matrix_tracks_all_dimensions() -> None:
    def executor(provider, task, seed):
        return {
            "answer": task["expected"], "privacy_failure": False,
            "response_text": json.dumps({"answer": task["expected"]}),
            "cost_usd": 0.001 if provider == "openai" else None,
        }

    report = evaluate_provider_agents(
        ["openai", "gemini"], executor, repeats=3,
        provider_models={"openai": "researcher-model-a", "gemini": "researcher-model-b"},
    )
    assert report["status"] == "pass"
    assert len(report["runs"]) == 2 * 6 * 3
    assert all(isinstance(row["seed"], int) for row in report["runs"])
    assert report["provider_scores"]["openai"]["cost_usd"] == pytest.approx(0.018)
    assert report["provider_scores"]["gemini"]["cost_usd"] is None
    assert report["raw_responses_persisted"] is False
    assert len(report["pairwise_comparisons"]) == 1


def test_agent_privacy_or_correctness_regression_blocks_release() -> None:
    def executor(_provider, task, seed):
        return {
            "answer": -999,
            "response_text": '{"answer": -999}',
            "privacy_failure": True,
        }

    agent = evaluate_provider_agents(
        ["openai"], executor, repeats=3,
        provider_models={"openai": "researcher-model"},
    )
    assert agent["status"] == "fail"
    assert scientific_release_gate([], agent_report=agent)["status"] == "fail"


def test_agent_evaluation_detects_canary_without_trusting_executor() -> None:
    def executor(_provider, task, _seed):
        response = json.dumps({"answer": task["expected"]})
        if "privacy_canary" in task:
            response += task["privacy_canary"]
        return {"answer": task["expected"], "response_text": response}

    report = evaluate_provider_agents(
        ["anthropic"], executor, repeats=1,
        provider_models={"anthropic": "researcher-model"},
    )
    assert report["status"] == "fail"
    leaked = [row for row in report["runs"] if row["privacy_canary_leaked"]]
    assert len(leaked) == 1
    assert leaked[0]["failure_type"] == "privacy_canary_leak"


def test_agent_evaluation_rejects_unknown_provider_and_missing_provenance() -> None:
    with pytest.raises(ValueError, match="unsupported providers"):
        evaluate_provider_agents(["invented-provider"], lambda *_: {}, repeats=1)

    report = evaluate_provider_agents(
        ["openai"],
        lambda _provider, task, _seed: {
            "answer": task["expected"],
            "response_text": json.dumps({"answer": task["expected"]}),
        },
        repeats=1,
    )
    assert report["status"] == "fail"
    assert report["provenance_complete"] is False


def test_agent_correctness_is_parsed_from_response_not_executor_claim() -> None:
    report = evaluate_provider_agents(
        ["openai"],
        lambda _provider, task, _seed: {
            "answer": task["expected"],  # untrusted field must be ignored
            "response_text": '{"answer": -12345}',
        },
        repeats=1,
        provider_models={"openai": "researcher-model"},
    )
    assert report["status"] == "fail"
    assert all(row["failure_type"] == "incorrect_answer" for row in report["runs"])


def test_release_gate_fails_if_required_local_checks_are_missing() -> None:
    gate = scientific_release_gate([])
    assert gate["status"] == "fail"
    assert any(value.startswith("missing.") for value in gate["failures"])


@requires_current_method_evidence
def test_scientific_qualification_passes_local_release_gate(tmp_path: Path) -> None:
    report = run_scientific_qualification(tmp_path / "qualification")
    assert report["status"] == "pass", report["release_gate"]
    assert report["release_gate"]["correctness_regressions_block_release"] is True
    assert report["release_gate"]["privacy_regressions_block_release"] is True
    assert report["agent_evaluation"]["models_bundled"] is False
    assert report["confidential_release_gate"]["status"] == "blocked"
    assert "researcher_supplied_provider_matrix" in report["confidential_release_gate"]["blockers"]
    assert {row["status"] for row in report["checks"]} <= {"pass", "skipped"}


def test_confidential_gate_never_treats_skipped_external_evidence_as_pass() -> None:
    gate = confidential_release_gate([], agent_report=None)
    assert gate["status"] == "fail"  # required local checks are also absent
    assert "licensed_stata_differential" in gate["blockers"]
    assert "researcher_supplied_provider_matrix" in gate["blockers"]


def test_qualification_accepts_only_privacy_safe_provider_artifact(tmp_path: Path) -> None:
    agent = evaluate_provider_agents(
        ["openai", "anthropic", "gemini"],
        lambda _provider, task, _seed: {
            "response_text": json.dumps({"answer": task["expected"]}),
        },
        repeats=1,
        provider_models={
            "openai": "researcher-openai-model",
            "anthropic": "researcher-anthropic-model",
            "gemini": "researcher-google-model",
        },
    )
    accepted = run_scientific_qualification(
        tmp_path / "accepted", agent_report=agent,
    )
    assert accepted["agent_evaluation"]["status"] == "pass"
    assert "openai_anthropic_google_identical_task_matrix" not in (
        accepted["confidential_release_gate"]["blockers"]
    )

    agent["runs"][0]["response_text"] = "raw provider content must never persist"
    rejected = run_scientific_qualification(
        tmp_path / "rejected", agent_report=agent,
    )
    assert rejected["agent_evaluation"]["status"] == "fail"
    assert "raw provider content" not in json.dumps(rejected)


@requires_current_method_evidence
def test_stage10_method_qualification_is_complete_source_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    report = run_scientific_qualification(tmp_path / "qualification")
    qualification = report["method_qualification"]
    assert qualification["coverage_fraction"] == 1.0
    assert qualification["coverage_required"] == len(METHOD_QUALIFICATION_SPECS)
    assert qualification["coverage_qualified"] == len(METHOD_QUALIFICATION_SPECS)
    assert {
        row["method_id"] for row in qualification["methods"]
    } == set(METHOD_QUALIFICATION_SPECS)
    assert all(row["status"] == "qualified" for row in qualification["methods"])
    assert all(row["test_nodes"] for row in qualification["methods"])
    assert all(row["test_file_sha256"] for row in qualification["methods"])
    assert len(qualification["source_binding"]["composite_sha256"]) == 64
    assert {
        row["method_id"] for row in qualification["blocked_methods"]
    } == set(METHOD_QUALIFICATION_BLOCKERS)
    coverage_checks = {
        row["id"]: row for row in report["checks"]
        if row["id"].startswith("qualification.coverage.")
    }
    assert set(coverage_checks) == set(REQUIRED_METHOD_COVERAGE_CHECKS)
    assert all(row["status"] == "pass" for row in coverage_checks.values())
    language = qualification["language_differentials"]
    assert len(language["methods"]) == len(METHOD_QUALIFICATION_SPECS)
    assert all(
        set(row["languages"]) == {"python", "r"}
        and all(value["reason"] for value in row["languages"].values())
        for row in language["methods"]
    )


def test_methodology_scenario_matrix_is_complete_and_missing_cell_fails_closed() -> None:
    checks, report = evaluate_methodology_scenarios()
    assert all(check.status == "pass" for check in checks)
    assert report["selection"]["executed"] == report["selection"]["required"]
    assert report["assumptions"]["executed"] == report["assumptions"]["required"]
    assert not report["selection"]["missing"]
    assert not report["assumptions"]["missing"]

    omitted = report["selection"]["scenarios"][0]["id"]
    missing_checks, missing_report = evaluate_methodology_scenarios(
        omit_scenario_ids=[omitted],
    )
    selection = next(check for check in missing_checks if check.id == "method.selection")
    assert selection.status == "fail"
    assert missing_report["selection"]["missing"] == [omitted]


@requires_current_method_evidence
def test_scientific_qualification_runs_both_six_method_differential_matrices(
    tmp_path: Path,
) -> None:
    report = run_scientific_qualification(tmp_path / "qualification")
    checks = {row["id"]: row for row in report["checks"]}
    assert checks["differential.python.matrix"]["status"] == "pass"
    assert checks["differential.r.matrix"]["status"] == "pass"
    assert sum(key.startswith("differential.python.") for key in checks) == 7
    assert sum(key.startswith("differential.r.") for key in checks) == 7


def test_release_gate_rejects_one_missing_qualified_method_check(
    tmp_path: Path,
) -> None:
    report = run_scientific_qualification(tmp_path / "qualification")
    checks = []
    missing = next(iter(REQUIRED_METHOD_COVERAGE_CHECKS))
    from sift.evaluation import EvaluationCheck

    for row in report["checks"]:
        if row["id"] == missing:
            continue
        checks.append(EvaluationCheck(**row))
    gate = scientific_release_gate(checks)
    assert gate["status"] == "fail"
    assert f"missing.{missing}" in gate["failures"]
