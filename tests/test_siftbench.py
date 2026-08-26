"""Wires SiftBench's deterministic cases into the regular test suite.

Each seed case is a genuine regression check of Sift's pipeline
against a known ground-truth answer -- see siftbench/__init__.py for
what "scored benchmark" honestly means here (pipeline correctness
against synthetic ground truth, not live-model judgment).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from sift.env_detect import detect_environment
from siftbench.cases import (
    SEED_CASES,
    BenchCase,
    ScoreResult,
    _close,
    _score_descriptive,
    score_status,
)
from siftbench.runner import run_all, run_case, summarize


def _python_ready() -> bool:
    e = detect_environment()
    if e.python is None or not e.has_sandbox_backend():
        return False
    needed = {"pandas", "numpy", "scipy", "statsmodels"}
    return not (needed & set(e.python.missing_packages))


_skip_no_python = pytest.mark.skipif(
    not _python_ready(),
    reason="needs python3 + pandas/numpy/scipy/statsmodels + a sandbox backend",
)


@_skip_no_python
@pytest.mark.parametrize("case", SEED_CASES, ids=[c.id for c in SEED_CASES])
def test_seed_case_passes(case: BenchCase, tmp_path: Path) -> None:
    run = run_case(case, tmp_path)
    assert run.score.passed, (
        f"SiftBench case {case.id!r} failed: {run.score.message}"
    )


@_skip_no_python
def test_run_all_isolates_cases_into_separate_dirs(tmp_path: Path) -> None:
    """Two cases must not share a result store or chat history --
    each gets its own subdirectory under the base dir."""
    runs = run_all(SEED_CASES, tmp_path)
    assert len(runs) == len(SEED_CASES)
    for case in SEED_CASES:
        assert (tmp_path / case.id).is_dir()


@_skip_no_python
def test_summarize_counts_correctly(tmp_path: Path) -> None:
    runs = run_all(SEED_CASES, tmp_path)
    report = summarize(runs)
    assert report["total"] == len(SEED_CASES)
    assert report["passed"] == len(SEED_CASES)  # all seed cases pass by design
    assert report["score"] == 1.0
    assert len(report["cases"]) == len(SEED_CASES)
    for c in report["cases"]:
        assert set(c.keys()) == {
            "id", "description", "passed", "message",
            "envelope_status", "duration_seconds",
        }


def test_summarize_handles_empty_run_list() -> None:
    report = summarize([])
    assert report == {"passed": 0, "total": 0, "score": 0.0, "cases": []}


def test_run_case_refuses_unsafe_case_id(tmp_path: Path) -> None:
    case = BenchCase(
        id="../escape",
        description="invalid id",
        prompt="x",
        reference_script="x",
        score=lambda _body, _store: ScoreResult(True, "should not run"),
    )
    run = run_case(case, tmp_path / "case")
    assert run.score.passed is False
    assert run.status == "runner_error"


def test_run_case_refuses_contaminated_directory(tmp_path: Path) -> None:
    case_dir = tmp_path / "existing"
    case_dir.mkdir()
    (case_dir / "old-result.txt").write_text("stale", encoding="utf-8")
    case = BenchCase(
        id="safe_case",
        description="contaminated run",
        prompt="x",
        reference_script="x",
        score=lambda _body, _store: ScoreResult(True, "should not run"),
    )
    run = run_case(case, case_dir)
    assert run.score.passed is False
    assert "not empty" in run.score.message


def test_run_all_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    case = BenchCase(
        id="duplicate",
        description="duplicate",
        prompt="x",
        reference_script="x",
        score=lambda _body, _store: ScoreResult(True, "x"),
    )
    with pytest.raises(ValueError, match="unique"):
        run_all([case, case], tmp_path)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True])
def test_numeric_oracle_rejects_non_finite_and_boolean_values(bad: float) -> None:
    assert _close(bad, 1.0, 0.1, "estimate") is not None


def test_descriptive_oracle_scores_the_disclosure_controlled_value() -> None:
    class Store:
        def __init__(self, mean: float) -> None:
            self.mean = mean

        def get(self, _result_id: str):
            return SimpleNamespace(sanitized_payload={
                "n": 50,
                "missing_count": 12,
                "mean": self.mean,
            })

    body = {"results": [{"status": "ok", "result_id": "R1"}]}
    assert _score_descriptive(body, Store(10.0)).passed is True
    assert _score_descriptive(body, Store(10.049725973369908)).passed is False


@_skip_no_python
def test_scorer_actually_catches_a_wrong_answer(tmp_path: Path) -> None:
    """Sanity-checks the sanity check: a case whose reference script
    computes the RIGHT thing but whose scorer expects the WRONG
    ground truth must fail. Without this, a scorer that always
    returns ``passed=True`` regardless of input would go unnoticed --
    this test proves the harness actually discriminates."""
    real_case = next(c for c in SEED_CASES if c.id == "correlation_known_r")

    def _wrong_scorer(body, store) -> ScoreResult:
        from siftbench.cases import _payload_for
        payload = _payload_for(body, store)
        got_r = payload["correlations"]["x"]["y"]
        # The true correlation is ~0.6; asserting it must be near 0.0
        # (with a tiny tolerance) should fail.
        if abs(got_r - 0.0) > 0.05:
            return ScoreResult(False, f"r={got_r} is not ~0 (deliberately wrong check)")
        return ScoreResult(True, "unexpected pass")

    broken_case = BenchCase(
        id=real_case.id,
        description=real_case.description,
        prompt=real_case.prompt,
        reference_script=real_case.reference_script,
        score=_wrong_scorer,
    )
    run = run_case(broken_case, tmp_path)
    assert run.score.passed is False


@_skip_no_python
def test_score_status_helper_matches_and_mismatches(tmp_path: Path) -> None:
    rejected_case = next(
        c for c in SEED_CASES if c.id == "small_n_correlation_rejected"
    )
    ok_scorer_expecting_wrong_status = BenchCase(
        id=rejected_case.id,
        description=rejected_case.description,
        prompt=rejected_case.prompt,
        reference_script=rejected_case.reference_script,
        score=score_status("ok"),  # actually rejected -- must fail
    )
    run = run_case(ok_scorer_expecting_wrong_status, tmp_path)
    assert run.score.passed is False
    assert "rejected_by_sanitizer" in run.score.message
