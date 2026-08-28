from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

from sift.evaluation import (
    METHOD_EXECUTION_NODES,
    _method_qualification_checks,
    _qualification_source_binding,
    verify_method_test_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_method_evidence_failure_excerpt_is_bounded_and_keeps_the_tail() -> None:
    script = ROOT / "scripts" / "method_qualification_evidence.py"
    spec = importlib.util.spec_from_file_location("method_evidence_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._failure_excerpt("short", limit=8) == "short"
    excerpt = module._failure_excerpt("0123456789", limit=4)
    assert excerpt == "[earlier output omitted]\n6789"


def _passing_artifact() -> dict:
    _, _, digest = _qualification_source_binding(ROOT)
    nodes = sorted({node for values in METHOD_EXECUTION_NODES.values() for node in values})
    return {
        "format": "sift-method-test-evidence", "schema_version": 1,
        "status": "pass", "source_binding_sha256": digest,
        "pytest_exit_code": 0, "unmatched_cases": 0,
        "runner": {
            "python": "synthetic-test-runtime", "platform": "synthetic-platform",
            "pytest": "synthetic-version", "python_packages": {"pytest": "synthetic"},
            "r": {"version": "synthetic-R", "platform": "synthetic-R-platform", "packages": {"stats": "synthetic"}},
        },
        "nodes": {
            node: {"status": "pass", "cases": [{"node_id": node, "status": "pass"}]}
            for node in nodes
        },
    }


def test_execution_evidence_rejects_skipped_failed_missing_and_stale_nodes(
    tmp_path: Path,
) -> None:
    baseline = _passing_artifact()
    valid_path = tmp_path / "valid.json"
    valid_path.write_text(json.dumps(baseline), encoding="utf-8")
    assert verify_method_test_evidence(ROOT, valid_path)["valid"] is True

    first_node = next(iter(baseline["nodes"]))
    mutations: dict[str, dict] = {}
    for status in ("skipped", "failed"):
        artifact = copy.deepcopy(baseline)
        artifact["nodes"][first_node] = {
            "status": status, "cases": [{"node_id": first_node, "status": status}],
        }
        mutations[status] = artifact
    missing = copy.deepcopy(baseline)
    missing["nodes"].pop(first_node)
    mutations["missing"] = missing
    stale = copy.deepcopy(baseline)
    stale["source_binding_sha256"] = "0" * 64
    mutations["stale"] = stale

    for label, artifact in mutations.items():
        path = tmp_path / f"{label}.json"
        path.write_text(json.dumps(artifact), encoding="utf-8")
        assert verify_method_test_evidence(ROOT, path)["valid"] is False, label
        checks, report = _method_qualification_checks(ROOT, path)
        execution = next(row for row in checks if row.id == "qualification.execution_evidence")
        assert execution.status == "fail", label
        assert report["coverage_qualified"] < report["coverage_required"], label


def test_missing_execution_evidence_cannot_qualify(tmp_path: Path) -> None:
    checks, report = _method_qualification_checks(ROOT, tmp_path / "absent.json")
    execution = next(row for row in checks if row.id == "qualification.execution_evidence")
    assert execution.status == "fail"
    assert report["coverage_qualified"] == 0


def test_runner_manifest_is_required(tmp_path: Path) -> None:
    artifact = _passing_artifact()
    artifact.pop("runner")
    path = tmp_path / "no-runner.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    verified = verify_method_test_evidence(ROOT, path)
    assert verified["valid"] is False
    assert "runner_manifest" in verified["failures"]
