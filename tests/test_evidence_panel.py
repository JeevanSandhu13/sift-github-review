"""Tests for the Evidence panel bridge method
(``SiftBridge.get_result_evidence``) and the ``get_session_verification``
fix it exposed along the way.

The Evidence panel is the "click a number, see where it came from"
surface: dataset, sample size, the canonical table, deterministic
verification, a Challenge Finding verdict when applicable, the
generated code, and a privacy note. Every field here traces to data
already stored for the result — nothing new crosses the privacy
boundary; this is a researcher-facing re-read of what the model
already saw once.
"""

from __future__ import annotations

from pathlib import Path

from sift.store import ResultStore, get_store, reset_store_for_tests
from sift.ui import SiftBridge

import pytest


@pytest.fixture(autouse=True)
def _reset_store_cache():
    reset_store_for_tests()
    yield
    reset_store_for_tests()


def _reg_payload(coefs: dict) -> dict:
    return {
        "type": "linear_regression", "n": 500,
        "coefficients": coefs,
        "standard_errors": {k: 0.1 for k in coefs},
        "r_squared": 0.3,
    }


def test_evidence_for_unknown_result_id_is_a_clean_miss(tmp_path: Path) -> None:
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_result_evidence("M999")
    assert out["ok"] is False
    assert "M999" in out["reason"]


def test_evidence_with_no_active_session(tmp_path: Path) -> None:
    bridge = SiftBridge(cwd=None)
    out = bridge.get_result_evidence("M1")
    assert out["ok"] is False


def test_evidence_carries_dataset_and_code(tmp_path: Path) -> None:
    store = get_store(tmp_path)
    store.insert(
        label="churn effect", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"treatment": 4.2}),
        language="Python", script_code="model.fit(...)",
        transformations=[], source_dataset="customers.parquet",
    )
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_result_evidence("M1")
    assert out["ok"] is True
    assert out["source_dataset"] == "customers.parquet"
    assert out["script_code"] == "model.fit(...)"
    assert out["n"] == 500
    assert out["markdown"]
    assert out["privacy_note"]
    assert "sanitized" in out["privacy_note"].lower()


def test_evidence_includes_recomputed_verification(tmp_path: Path) -> None:
    """A small-n result should carry the same sample-size warning the
    model saw at result time — the panel recomputes, doesn't quote a
    frozen copy, so it reflects the same deterministic function."""
    store = get_store(tmp_path)
    payload = _reg_payload({"treatment": 4.2})
    payload["n"] = 12  # below _MIN_COMFORTABLE_N
    store.insert(
        label="small sample", analysis_type="linear_regression",
        sanitized_payload=payload, language="R", script_code="lm(...)",
        transformations=[],
    )
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_result_evidence("M1")
    assert out["verification"] is not None
    assert any(c["status"] == "warn" for c in out["verification"]["checks"])


def test_evidence_carries_challenge_summary_when_batched(tmp_path: Path) -> None:
    store = get_store(tmp_path)
    store.insert(
        label="baseline", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"treatment": 4.2}),
        language="Python", script_code="spec_a()", transformations=[],
        script_run_id="run-1",
    )
    store.insert(
        label="drop outliers", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"treatment": 3.9}),
        language="Python", script_code="spec_b()", transformations=[],
        script_run_id="run-1",
    )
    bridge = SiftBridge(cwd=tmp_path)
    baseline_evidence = bridge.get_result_evidence("M1")
    assert baseline_evidence["challenge_summary"] is not None
    assert baseline_evidence["challenge_summary"]["verdict"] == "ROBUST"
    assert baseline_evidence["is_challenge_baseline"] is True

    alt_evidence = bridge.get_result_evidence("M2")
    assert alt_evidence["challenge_summary"] is not None
    assert alt_evidence["is_challenge_baseline"] is False


def test_evidence_omits_challenge_summary_for_solo_result(tmp_path: Path) -> None:
    store = get_store(tmp_path)
    store.insert(
        label="only one", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"treatment": 4.2}),
        language="Python", script_code="spec_a()", transformations=[],
        script_run_id="run-1",
    )
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_result_evidence("M1")
    assert out["challenge_summary"] is None


def test_evidence_for_hidden_result_is_a_clean_miss(tmp_path: Path) -> None:
    """A result hidden by a rewind must not be readable through the
    Evidence panel either — ``store.get`` already filters hidden rows
    by default, and the bridge must not bypass that."""
    store = get_store(tmp_path)
    store.insert(
        label="will be hidden", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"treatment": 4.2}),
        language="Python", script_code="x()", transformations=[],
    )
    store.hide_results_not_in(set(), reason="rewind")
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_result_evidence("M1")
    assert out["ok"] is False


# ---------------------------------------------------------------------------
# get_session_verification — source_dataset now reads the real column
# ---------------------------------------------------------------------------

def test_session_verification_drift_check_now_actually_fires(
    tmp_path: Path,
) -> None:
    """Regression for the bug ``get_result_evidence`` surfaced while
    being built: ``get_session_verification`` sourced ``source_dataset``
    from inside the sanitized payload, where the sanitizer never puts
    it, so the per-dataset sample-size-drift check was silently dead
    (``n_by_dataset`` was always empty). Now reads ``row.source_dataset``
    (the actual store column) and the check fires on real drift."""
    store = get_store(tmp_path)
    store.insert(
        label="full sample", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"x": 1.0}) | {"n": 1000},
        language="R", script_code="m1()", transformations=[],
        source_dataset="customers.csv",
    )
    store.insert(
        label="filtered sample", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"x": 1.0}) | {"n": 400},
        language="R", script_code="m2()", transformations=[],
        source_dataset="customers.csv",
    )
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_session_verification()
    drift_checks = [c for c in out["checks"]
                    if c["id"].startswith("sample_drift::")]
    assert drift_checks, "drift check never fired — regression reintroduced"
    assert drift_checks[0]["status"] == "warn"
    assert "customers.csv" in drift_checks[0]["id"]


def test_session_verification_no_drift_when_dataset_absent(
    tmp_path: Path,
) -> None:
    """Results with no recorded source dataset (e.g. in-memory-only
    scripts) must not be forced into a bogus drift comparison."""
    store = get_store(tmp_path)
    store.insert(
        label="a", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"x": 1.0}) | {"n": 1000},
        language="R", script_code="m1()", transformations=[],
    )
    store.insert(
        label="b", analysis_type="linear_regression",
        sanitized_payload=_reg_payload({"x": 1.0}) | {"n": 400},
        language="R", script_code="m2()", transformations=[],
    )
    bridge = SiftBridge(cwd=tmp_path)
    out = bridge.get_session_verification()
    drift_checks = [c for c in out["checks"]
                    if c["id"].startswith("sample_drift::")]
    assert drift_checks == []


# ---------------------------------------------------------------------------
# Structural guard — bridge-only, never model-reachable
# ---------------------------------------------------------------------------

def test_get_result_evidence_is_not_a_tool() -> None:
    """Same invariant as the dataset profile and linkage report: this
    reads researcher-local material (full script code, unrounded
    verification detail keyed to one result) and must only be
    reachable from the UI bridge, never from the model's tool
    surface."""
    from sift.tools import ALLOWED_TOOL_NAMES, HANDLERS

    assert "get_result_evidence" not in ALLOWED_TOOL_NAMES
    assert "get_result_evidence" not in HANDLERS


def test_tool_layer_does_not_reference_get_result_evidence() -> None:
    src = Path("src/sift/tools.py").read_text(encoding="utf-8")
    assert "get_result_evidence" not in src
