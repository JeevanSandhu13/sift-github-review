"""Fail-closed workflow binding for pre-method_result typed payloads."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from sift.methodology import METHODS
from sift.sanitizer import DEFAULT_CONFIG, sanitize
from sift.store import ResultStore
from sift.tools import _legacy_registry_method_id, _sanitize_and_store_payloads


@pytest.mark.parametrize(
    ("analysis_type", "payload", "expected"),
    [
        ("t_test", {}, "t_test"),
        ("descriptive", {}, "descriptive_statistics"),
        ("frequency_table", {}, "frequency_table"),
        ("crosstab", {}, "crosstab"),
        ("magnitude_table", {}, "magnitude_table"),
        ("rdd", {}, "regression_discontinuity"),
        ("kaplan_meier", {}, "kaplan_meier"),
        ("cluster_analysis", {}, "clustering"),
        ("marginal_effects", {}, "marginal_effects"),
        ("factor_decomposition", {"method": "pca"}, "pca"),
        (
            "factor_decomposition", {"method": "maximum_likelihood"},
            "exploratory_factor_analysis",
        ),
        (
            "did_event_study", {"estimator": "callaway_santanna"},
            "staggered_adoption",
        ),
        (
            "did_event_study", {"estimator": "sun_abraham"},
            "event_study",
        ),
        (
            "coefficient_table_with_fit_stats",
            {"registry_method_id": "poisson_regression"},
            "poisson_regression",
        ),
    ],
)
def test_legacy_shape_has_exact_registry_binding(
    analysis_type: str, payload: dict, expected: str,
) -> None:
    assert _legacy_registry_method_id(analysis_type, payload) == expected
    assert expected in METHODS


@pytest.mark.parametrize(
    ("analysis_type", "payload"),
    [
        ("did_event_study", {"estimator": "twfe"}),
        ("did_event_study", {}),
        ("factor_decomposition", {"method": "unknown"}),
        ("coefficient_table_with_fit_stats", {}),
        ("unknown_legacy_shape", {}),
    ],
)
def test_ambiguous_legacy_shape_has_no_registry_binding(
    analysis_type: str, payload: dict,
) -> None:
    assert _legacy_registry_method_id(analysis_type, payload) is None


def _workflow(*analysis_ids: str) -> dict:
    return {
        "workflow_id": "wf-legacy", "workflow_revision": 1,
        "approval_sha256": "a" * 64,
        "analyses": [
            {"id": value, "role": "primary" if i == 0 else "sensitivity",
             "seed": 71 + i, "changes": []}
            for i, value in enumerate(analysis_ids)
        ],
    }


def _ttest() -> dict:
    return {
        "type": "t_test", "test_type": "one_sample", "n1": 80,
        "mean1": 2.0, "t_statistic": 1.5, "p_value": 0.14,
    }


def _store(
    tmp_path: Path, raw: dict, *, method: str | None,
    workflow: dict, run_id: str,
) -> tuple[list[dict], bool, ResultStore]:
    store = ResultStore(tmp_path / f"{run_id}.db")
    results, any_ok, *_ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="legacy", language="Python", code="# fit",
        source_dataset=None, source_n=None, sdc_cfg=DEFAULT_CONFIG,
        run_dir=None, script_run_id=run_id, store=store,
        expected_method_id=method, workflow_context=workflow,
        provenance_base={"dataset_hashes": {"data.csv": "b" * 64}},
    )
    return results, any_ok, store


def test_single_analysis_legacy_result_binds_seed_and_role(tmp_path: Path) -> None:
    results, any_ok, store = _store(
        tmp_path, _ttest(), method="t_test",
        workflow=_workflow("headline"), run_id="accept",
    )
    try:
        assert any_ok and results[0]["status"] == "ok"
        row = store.get(results[0]["result_id"])
        assert row is not None
        assert row.provenance["analysis_id"] == "headline"
        assert row.provenance["analysis_role"] == "primary"
        assert row.provenance["random_seed"] == 71
        assert row.provenance["dataset_hashes"]["data.csv"] == "b" * 64
    finally:
        store.close()


def test_legacy_method_mismatch_is_rejected(tmp_path: Path) -> None:
    results, any_ok, store = _store(
        tmp_path, _ttest(), method="descriptive_statistics",
        workflow=_workflow("headline"), run_id="mismatch",
    )
    try:
        assert not any_ok
        assert "does not match" in results[0]["reason"]
    finally:
        store.close()


def test_workflow_legacy_result_requires_prevalidated_method(tmp_path: Path) -> None:
    results, any_ok, store = _store(
        tmp_path, _ttest(), method=None,
        workflow=_workflow("headline"), run_id="no-method",
    )
    try:
        assert not any_ok
        assert "prevalidated method_id" in results[0]["reason"]
    finally:
        store.close()


def test_multi_analysis_legacy_result_is_rejected_as_ambiguous(
    tmp_path: Path,
) -> None:
    results, any_ok, store = _store(
        tmp_path, _ttest(), method="t_test",
        workflow=_workflow("headline", "sensitivity"), run_id="ambiguous",
    )
    try:
        assert not any_ok
        assert "exactly one analysis" in results[0]["reason"]
    finally:
        store.close()


def _runtime_payload(
    tmp_path: Path, *, generic: bool, unsupported_glm: bool = False,
) -> dict:
    token = "legacy-binding-token"
    result_path = tmp_path / ("generic.jsonl" if generic else "lm.jsonl")
    old_token = os.environ.get("SIFT_RUN_TOKEN")
    old_path = os.environ.get("SIFT_RESULT_PATH")
    os.environ["SIFT_RUN_TOKEN"] = token
    os.environ["SIFT_RESULT_PATH"] = str(result_path)
    sys.modules.pop("sift.runtime.sift", None)
    try:
        runtime = importlib.import_module("sift.runtime.sift")
        if generic:
            runtime.result(
                type="coefficient_table_with_fit_stats", n=80,
                response_variable="y", predictor_variables=["x"],
                coefficients={"const": 1.0, "x": 0.5},
                standard_errors={"const": 0.2, "x": 0.1},
                _registry_method_id="linear_regression",
            )
        else:
            np = pytest.importorskip("numpy")
            sm = pytest.importorskip("statsmodels.api")
            x = np.linspace(-2.0, 2.0, 80)
            if unsupported_glm:
                y = np.exp(1.0 + 0.2 * x) * (
                    1.0 + 0.05 * np.sin(np.arange(x.size))
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.GLM(
                        y, sm.add_constant(x), family=sm.families.Gamma(),
                    ).fit()
            else:
                fit = sm.OLS(
                    1.0 + 0.5 * x + 0.05 * np.sin(np.arange(x.size)),
                    sm.add_constant(x),
                ).fit()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                runtime.from_lm(fit)
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        assert payload.pop("_token") == token
        return payload
    finally:
        sys.modules.pop("sift.runtime.sift", None)
        if old_token is None:
            os.environ.pop("SIFT_RUN_TOKEN", None)
        else:
            os.environ["SIFT_RUN_TOKEN"] = old_token
        if old_path is None:
            os.environ.pop("SIFT_RESULT_PATH", None)
        else:
            os.environ["SIFT_RESULT_PATH"] = old_path


def test_typed_python_lm_stamps_and_stores_exact_method(tmp_path: Path) -> None:
    raw = _runtime_payload(tmp_path, generic=False)
    assert raw["_registry_method_id"] == "linear_regression"
    clean = sanitize(raw)
    assert clean.ok
    assert (clean.sanitized or {})["registry_method_id"] == "linear_regression"

    results, any_ok, store = _store(
        tmp_path, raw, method="linear_regression",
        workflow=_workflow("headline"), run_id="typed-lm",
    )
    try:
        assert any_ok and results[0]["status"] == "ok"
    finally:
        store.close()

    mismatch, mismatch_ok, mismatch_store = _store(
        tmp_path, raw, method="logistic_regression",
        workflow=_workflow("headline"), run_id="typed-lm-mismatch",
    )
    try:
        assert not mismatch_ok
        assert "does not match" in mismatch[0]["reason"]
    finally:
        mismatch_store.close()


def test_generic_python_emit_cannot_forge_regression_method(tmp_path: Path) -> None:
    raw = _runtime_payload(tmp_path, generic=True)
    assert "_registry_method_id" not in raw
    clean = sanitize(raw)
    assert clean.ok and "registry_method_id" not in (clean.sanitized or {})
    results, any_ok, store = _store(
        tmp_path, raw, method="linear_regression",
        workflow=_workflow("headline"), run_id="forged-lm",
    )
    try:
        assert not any_ok
        assert "does not match" in results[0]["reason"]
    finally:
        store.close()


def test_unsupported_python_glm_family_is_not_broadly_bound(
    tmp_path: Path,
) -> None:
    raw = _runtime_payload(tmp_path, generic=False, unsupported_glm=True)
    assert "_registry_method_id" not in raw
    clean = sanitize(raw)
    assert clean.ok and "registry_method_id" not in (clean.sanitized or {})
    results, any_ok, store = _store(
        tmp_path, raw, method="linear_regression",
        workflow=_workflow("headline"), run_id="gamma-unbound",
    )
    try:
        assert not any_ok
        assert "does not match" in results[0]["reason"]
    finally:
        store.close()


def test_unknown_internal_regression_marker_fails_closed() -> None:
    raw = {
        "type": "coefficient_table_with_fit_stats", "n": 80,
        "response_variable": "y", "predictor_variables": ["x"],
        "coefficients": {"const": 1.0, "x": 0.5},
        "standard_errors": {"const": 0.2, "x": 0.1},
        "_registry_method_id": "all_regression_methods",
    }
    clean = sanitize(raw)
    assert not clean.ok
    assert "marker is invalid" in (clean.rejection_reason or "")


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript unavailable")
def test_r_lm_stamps_marker_and_generic_emit_strips_forgery(tmp_path: Path) -> None:
    result_path = tmp_path / "r-results.jsonl"
    sift_r = Path(__file__).resolve().parents[1] / "src/sift/runtime/sift.R"
    script = tmp_path / "binding.R"
    script.write_text(
        "Sys.setenv(SIFT_RUN_TOKEN='r-token')\n"
        f"Sys.setenv(SIFT_RESULT_PATH={json.dumps(str(result_path))})\n"
        f"source({json.dumps(str(sift_r))})\n"
        "x <- seq(-2, 2, length.out=80)\n"
        "y <- 1 + .5*x\n"
        "sift$from_lm(lm(y ~ x))\n"
        "sift$result(type='coefficient_table_with_fit_stats', n=80L, "
        "response_variable='y', predictor_variables=list('x'), "
        "coefficients=list('(Intercept)'=1,x=.5), "
        "standard_errors=list('(Intercept)'=.2,x=.1), "
        "'_registry_method_id'='linear_regression')\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [shutil.which("Rscript") or "Rscript", str(script)],
        capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["_registry_method_id"] == "linear_regression"
    assert "_registry_method_id" not in lines[1]
