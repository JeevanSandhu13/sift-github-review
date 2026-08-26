from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sift.reproducibility import (
    append_audit_event,
    build_bundle_manifest,
    compare_payloads,
    environment_drift,
    rerun_bundle,
    verify_audit_chain,
    verify_bundle,
)
from sift.research_export import build_replication_package
from sift.store import get_store


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptive(mean: float = 2.0) -> dict[str, object]:
    return {
        "type": "descriptive",
        "variable": "x",
        "n": 20,
        "mean": mean,
        "sd": 1.0,
        "missing_count": 0,
    }


def test_append_only_audit_chain_detects_tampering_and_refuses_append(
    tmp_path: Path,
) -> None:
    first = append_audit_event(tmp_path, "script_execution", {
        "script_run_id": "run-1", "status": "ok", "raw_value": "excluded",
    })
    second = append_audit_event(tmp_path, "independent_challenge", {
        "script_run_id": "run-1", "challenge_status": "pass",
    })
    assert first["sequence"] == 1 and second["previous_sha256"] == first["event_sha256"]
    assert "raw_value" not in first["metadata"]
    assert verify_audit_chain(tmp_path) == {
        "valid": True, "events": 2, "last_sha256": second["event_sha256"],
    }

    audit = tmp_path / ".sift" / "reproducibility_audit.jsonl"
    audit.write_text(audit.read_text(encoding="utf-8").replace("independent_challenge", "changed"))
    assert verify_audit_chain(tmp_path)["valid"] is False
    with pytest.raises(RuntimeError, match="chain is corrupt"):
        append_audit_event(tmp_path, "later", {"status": "ok"})


def test_bundle_manifest_detects_add_remove_and_change_but_ignores_report(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.txt").write_text("b")
    manifest = build_bundle_manifest(tmp_path)
    assert len(manifest["files"]) == 2
    assert verify_bundle(tmp_path)["valid"] is True

    (tmp_path / "reproduction_report.json").write_text("{}")
    assert verify_bundle(tmp_path)["valid"] is True
    (tmp_path / "nested" / "reproduction_report.json").write_text("{}")
    assert verify_bundle(tmp_path)["added"] == ["nested/reproduction_report.json"]
    (tmp_path / "nested" / "reproduction_report.json").unlink()
    (tmp_path / "a.txt").write_text("changed")
    check = verify_bundle(tmp_path)
    assert check["valid"] is False and check["changed"] == ["a.txt"]
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "new.txt").write_text("new")
    assert verify_bundle(tmp_path)["added"] == ["new.txt"]
    (tmp_path / "new.txt").unlink()
    (tmp_path / "nested" / "b.txt").unlink()
    assert verify_bundle(tmp_path)["missing"] == ["nested/b.txt"]


def test_bundle_rejects_symbolic_links(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("safe")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic links"):
        build_bundle_manifest(tmp_path)
    link.unlink()
    build_bundle_manifest(tmp_path)
    link.symlink_to(target)
    check = verify_bundle(tmp_path)
    assert check["valid"] is False and check["unsafe"] == ["link.txt"]


def test_numerical_comparison_is_tolerant_but_categorical_comparison_is_exact() -> None:
    close = compare_payloads(
        {"estimate": 1.0, "label": "primary"},
        {"estimate": 1.0 + 1e-9, "label": "primary"},
    )
    assert close["match"] is True
    changed = compare_payloads(
        {"estimate": 1.0, "label": "primary"},
        {"estimate": 1.1, "label": "sensitivity"},
    )
    assert changed["match"] is False
    assert {row["path"] for row in changed["differences"]} == {"estimate", "label"}


def _make_rerunnable_bundle(root: Path, source: Path) -> None:
    (root / "scripts").mkdir(parents=True)
    (root / "results").mkdir()
    (root / "scripts" / "analysis.py").write_text("# exact script\n")
    (root / "results" / "result.json").write_text(json.dumps({"payload": _descriptive()}))
    (root / "reproduce.json").write_text(json.dumps({
        "version": 1,
        "datasets": [{"path": source.name, "source_sha256": _sha(source)}],
        "environment": {"platform": "ImpossibleOS", "packages": {"python": "0"}},
        "comparison": {"rtol": 1e-7, "atol": 1e-10},
        "runs": [{
            "script_run_id": "run-1",
            "language": "Python",
            "script": "scripts/analysis.py",
            "script_sha256": _sha(root / "scripts" / "analysis.py"),
            "runnable": True,
            "privacy_configuration": {"disclosure_settings": {
                "min_n_regression": 10, "min_n_descriptive": 10,
                "min_n_ttest_group": 10, "cell_suppression_threshold": 10,
                "min_n_did_cohort": 10, "dominance_threshold": 0.85,
            }},
            "expected_results": [{"result_id": "M1", "result": "results/result.json"}],
        }],
    }))
    build_bundle_manifest(root)


def test_model_free_rerun_verifies_sources_compares_and_reports_drift(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = data_root / "source.csv"
    source.write_text("x\n1\n")
    bundle = tmp_path / "bundle"
    _make_rerunnable_bundle(bundle, source)
    calls: list[str] = []

    def fake_executor(language, code, cwd, *, timeout_seconds):
        calls.append(f"{language}:{code.strip()}:{cwd.name}:{timeout_seconds}")
        return SimpleNamespace(ok=True, result_payloads=[_descriptive(2.0)])

    report = rerun_bundle(bundle, data_root=data_root, executor_fn=fake_executor)
    assert report["status"] == "match"
    assert report["model_contacted"] is False
    assert report["environment_drift"]["drift"] is True
    assert len(calls) == 1
    assert verify_bundle(bundle)["valid"] is True  # derived report is excluded


def test_model_free_rerun_blocks_changed_source_before_execution(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    source = data_root / "source.csv"
    source.write_text("x\n1\n")
    bundle = tmp_path / "bundle"
    _make_rerunnable_bundle(bundle, source)
    source.write_text("x\n2\n")
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True

    report = rerun_bundle(bundle, data_root=data_root, executor_fn=forbidden)
    assert report["status"] == "blocked"
    assert report["reason"] == "source_dataset_mismatch"
    assert called is False


def test_replication_export_contains_complete_reconstruction_contract(
    tmp_path: Path,
) -> None:
    source = tmp_path / "data.csv"
    source.write_text("x\n1\n")
    script = "# analysis\n"
    provenance = {
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "source_file_hashes": {"data.csv": _sha(source)},
        "dataset_hashes": {"data.csv": "c" * 64},
        "canonical_fingerprints": {"data.csv": "fp"},
        "parser_versions": {"data.csv": {"name": "pandas", "version": "2"}},
        "model_configuration": {
            "provider": "openai", "model": "gpt-test", "reasoning_effort": "high",
        },
        "privacy_configuration": {"policy_sha256": "p" * 64},
        "verification_outcome": {"status": "pass", "checks": []},
        "executed_at": "2026-01-01T00:00:00+00:00",
        "result_schema_version": 1,
        "verification_schema_version": 1,
    }
    store = get_store(tmp_path)
    store.insert(
        label="Summary", analysis_type="descriptive",
        sanitized_payload=_descriptive(), language="Python", script_code=script,
        transformations=[], script_run_id="run-1", source_dataset="data.csv",
        provenance=provenance,
    )
    append_audit_event(tmp_path, "script_execution", {
        "script_run_id": "run-1", "result_ids": ["M1"], "status": "ok",
    })
    destination = tmp_path / "export"
    summary = build_replication_package(tmp_path, destination)
    assert summary["audit_chain_ok"] is True
    assert verify_bundle(destination)["valid"] is True
    reproduce = json.loads((destination / "reproduce.json").read_text(encoding="utf-8"))
    assert reproduce["model_required"] is False
    assert reproduce["datasets"][0]["source_sha256"] == _sha(source)
    assert reproduce["datasets"][0]["parser"]["name"] == "pandas"
    run = reproduce["runs"][0]
    assert run["model_configuration"]["provider"] == "openai"
    assert run["script_hash_matches_record"] is True
    exported_result = json.loads(next((destination / "results").glob("*.json")).read_text(encoding="utf-8"))
    assert exported_result["script_run_id"] == "run-1"
    assert exported_result["provenance"]["verification_outcome"]["status"] == "pass"
    assert (destination / "provenance" / "reproducibility_audit.jsonl").is_file()


def test_environment_drift_reports_missing_and_changed_packages(tmp_path: Path) -> None:
    (tmp_path / "reproduce.json").write_text(json.dumps({
        "environment": {"platform": "Other", "packages": {"made-up-package": "1"}},
    }))
    report = environment_drift(tmp_path)
    assert report["drift"] is True
    assert any(row["package"] == "made-up-package" for row in report["differences"])


def test_privacy_provenance_records_effective_settings_without_field_names(
    tmp_path: Path,
) -> None:
    from sift.sanitizer import SDCConfig
    from sift.tools import _privacy_provenance

    secret_field = "highly_sensitive_diagnosis"
    cfg = SDCConfig(
        min_n_regression=25,
        banned_variables=frozenset({secret_field}),
        non_disclosive_variables=frozenset({"age"}),
    )
    recorded = _privacy_provenance(tmp_path, cfg)
    assert recorded["policy_status"] == "default"
    assert recorded["disclosure_settings"]["min_n_regression"] == 25
    assert recorded["disclosure_settings"]["banned_variables"]["count"] == 1
    assert secret_field not in json.dumps(recorded)
