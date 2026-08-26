"""Executable-reference and fail-closed tests for missing-data methods."""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sift.sanitizer import sanitize
from sift.verification import verify_payload


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "audit_missing_data_methods.py"
RUNTIME_R = ROOT / "src" / "sift" / "runtime" / "sift.R"
METHODS = {
    "missingness_pattern", "single_imputation", "multiple_imputation",
    "mnar_sensitivity",
}


@pytest.fixture(scope="module")
def runtime_module(tmp_path_factory):
    output = tmp_path_factory.mktemp("missing_runtime") / "results.jsonl"
    old_token = os.environ.get("SIFT_RUN_TOKEN")
    old_path = os.environ.get("SIFT_RESULT_PATH")
    os.environ["SIFT_RUN_TOKEN"] = "missing-runtime-token"
    os.environ["SIFT_RESULT_PATH"] = str(output)
    try:
        spec = importlib.util.spec_from_file_location(
            "sift_missing_runtime", ROOT / "src" / "sift" / "runtime" / "sift.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if old_token is None:
            os.environ.pop("SIFT_RUN_TOKEN", None)
        else:
            os.environ["SIFT_RUN_TOKEN"] = old_token
        if old_path is None:
            os.environ.pop("SIFT_RESULT_PATH", None)
        else:
            os.environ["SIFT_RESULT_PATH"] = old_path
    return module


@pytest.fixture(scope="module")
def qualified_payloads(tmp_path_factory) -> dict[str, dict]:
    pytest.importorskip("statsmodels")
    pytest.importorskip("sklearn")
    destination = tmp_path_factory.mktemp("missing_data") / "results.jsonl"
    environment = os.environ.copy()
    environment["SIFT_RUN_TOKEN"] = "missing-data-qualification-token"
    environment["SIFT_RESULT_PATH"] = str(destination)
    completed = subprocess.run(
        [sys.executable, str(AUDIT)], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    payloads: dict[str, dict] = {}
    for line in destination.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload.pop("_token", None)
        payloads[payload["method_id"]] = payload
    assert set(payloads) == METHODS
    return payloads


@pytest.mark.parametrize("method_id", sorted(METHODS))
def test_real_reference_payloads_are_sanitizer_valid_and_aggregate_only(
    qualified_payloads: dict[str, dict], method_id: str,
) -> None:
    clean = sanitize(qualified_payloads[method_id])
    assert clean.ok, clean.rejection_reason
    assert clean.sanitized["method_id"] == method_id
    forbidden = {
        "rows", "observations", "patterns", "imputed_values", "predictions",
        "residuals", "fitted_values", "influence_by_case",
    }
    assert forbidden.isdisjoint(clean.sanitized)


def test_pattern_analysis_quantifies_attrition_without_claiming_a_mechanism(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["missingness_pattern"]).sanitized
    assert result["metrics"]["missingness_pattern_count"] == 4
    assert result["metrics"]["complete_case_rate"] == pytest.approx(0.5333)
    assert result["diagnostics"]["complete_case_warning"] == "warn"
    assert "do not identify MCAR, MAR, or MNAR" in result["claim_rule"]
    checks = verify_payload(result)["checks"]
    assert any(
        row["id"] == "missing_mechanism_identification"
        and row["status"] == "warn" for row in checks
    )


def test_single_imputation_is_confined_to_preprocessing(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["single_imputation"]).sanitized
    assert result["imputation_scope"] == "prediction_preprocessing"
    assert result["imputation_model"] == "simple_deterministic"
    assert result["metrics"]["imputed_cell_count"] == 126
    for field in (
        "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper",
        "uncertainty_type",
    ):
        assert field not in result


def test_statsmodels_mice_recovers_known_synthetic_coefficients_and_rubin_identity(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["multiple_imputation"]).sanitized
    assert result["estimates"]["x1"] == pytest.approx(1.45, abs=0.12)
    assert result["estimates"]["x2"] == pytest.approx(-0.55, abs=0.12)
    assert result["imputations"] == 16 and result["burn_in"] == 12
    assert result["seed"] == 20260822
    for name, standard_error in result["standard_errors"].items():
        within = result["metrics"][f"within#{name}"]
        between = result["metrics"][f"between#{name}"]
        expected = within + (1 + 1 / result["imputations"]) * between
        assert standard_error**2 == pytest.approx(expected, rel=0.02)
        assert result["metrics"][f"fmi#{name}"] >= result["metrics"][f"lambda#{name}"]
    checks = verify_payload(result)["checks"]
    assert any(row["id"] == "rubin_pooling" and row["status"] == "pass" for row in checks)


def test_mnar_delta_grid_is_pooled_and_preserves_scenario_claim_boundary(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["mnar_sensitivity"]).sanitized
    assert [result["metrics"][f"delta#scenario_{i}"] for i in range(1, 4)] == [
        -0.8, 0.0, 0.8,
    ]
    assert result["estimates"]["scenario_1"] < result["estimates"]["scenario_2"]
    assert result["estimates"]["scenario_2"] < result["estimates"]["scenario_3"]
    assert result["diagnostics"]["sensitivity_parameter_justification"] == "warn"
    assert "data do not identify" in result["claim_rule"]


def test_single_imputation_recomputes_values_and_checks_feature_order(runtime_module) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    SimpleImputer = pytest.importorskip("sklearn.impute").SimpleImputer
    original = pd.DataFrame({"x": [1.0, np.nan, 3.0], "z": [2.0, 4.0, 6.0]})
    imputer = SimpleImputer(strategy="median").fit(original)
    expected = imputer.transform(original)
    forged = expected.copy()
    forged[1, 0] = 999.0
    with pytest.raises(ValueError, match="do not match"):
        runtime_module.from_single_imputation(
            imputer, original, forged, scope="prediction_preprocessing",
        )
    reversed_names = pd.DataFrame(expected, columns=["z", "x"])
    with pytest.raises(ValueError, match="feature names/order"):
        runtime_module.from_single_imputation(
            imputer, original, reversed_names, scope="prediction_preprocessing",
        )


def test_prefitted_mi_and_caller_asserted_mnar_fits_are_rejected(runtime_module) -> None:
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    with pytest.raises(TypeError, match="DataFrame"):
        runtime_module.from_multiple_imputation(
            object(), formula="y ~ x", seed=1, burn_in=2, imputations=4,
        )
    with pytest.raises(ValueError, match="data must contain"):
        runtime_module.from_mnar_sensitivity(
            {0.0: [object()]}, incomplete_outcome="y", formula="y ~ x",
            parameter="x", deltas=(-1, 0, 1), seed=1, burn_in=2,
            imputations=4,
        )
    frame = pd.DataFrame({"y": [1.0, np.nan, 3.0, 4.0], "x": [0.0, 1.0, 0.0, 1.0]})
    with pytest.raises(ValueError, match="formula response"):
        runtime_module.from_mnar_sensitivity(
            frame, incomplete_outcome="y", formula="x ~ y", parameter="y",
            deltas=(-1, 0, 1), seed=1, burn_in=2, imputations=4,
        )


def test_mice_trace_uses_only_retained_post_burn_iterations(runtime_module) -> None:
    # The first callback is the post-burn-in snapshot and is deliberately
    # excluded; changing it cannot change the retained-trace diagnostic.
    first = runtime_module._trace_stability(
        [[-999.0], [0.0], [1.0], [2.0], [3.0]], expected_imputations=4,
    )
    second = runtime_module._trace_stability(
        [[999.0], [0.0], [1.0], [2.0], [3.0]], expected_imputations=4,
    )
    assert first == second
    with pytest.raises(ValueError, match="retained imputation count"):
        runtime_module._trace_stability([[0.0], [1.0]], expected_imputations=4)


def test_rubin_pool_rejects_indefinite_covariance(runtime_module) -> None:
    np = pytest.importorskip("numpy")

    class Model:
        exog_names = ["Intercept", "x"]

    class Fit:
        model = Model()
        params = np.array([1.0, 2.0])
        nobs = 20
        df_resid = 18

        def cov_params(self):
            return np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="positive semidefinite"):
        runtime_module._rubin_pool([Fit(), Fit(), Fit(), Fit()])


@pytest.mark.parametrize(
    ("method_id", "mutate", "message"),
    [
        (
            "missingness_pattern",
            lambda row: row["diagnostics"].__setitem__("complete_case_warning", "pass"),
            "complete-case warning",
        ),
        (
            "single_imputation",
            lambda row: row.__setitem__("estimates", {"effect": 1.0}),
            "preprocessing boundary",
        ),
        (
            "multiple_imputation",
            lambda row: row["metrics"].__setitem__("between#x1", 2.0),
            "Rubin-pooled",
        ),
        (
            "multiple_imputation",
            lambda row: row.__setitem__("uncertainty_type", "classical"),
            "Rubin-pooled",
        ),
        (
            "mnar_sensitivity",
            lambda row: row["metrics"].__setitem__("delta#scenario_2", 0.2),
            "delta grid",
        ),
        (
            "mnar_sensitivity",
            lambda row: row["diagnostics"].__setitem__("conclusion_stability", False),
            "delta grid",
        ),
        (
            "mnar_sensitivity",
            lambda row: row["diagnostics"].__setitem__(
                "sensitivity_parameter_justification", "pass"
            ),
            "delta grid",
        ),
    ],
)
def test_missing_data_method_invariants_fail_closed(
    qualified_payloads: dict[str, dict], method_id: str, mutate, message: str,
) -> None:
    payload = copy.deepcopy(qualified_payloads[method_id])
    mutate(payload)
    clean = sanitize(payload)
    assert not clean.ok
    assert message in clean.rejection_reason


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript unavailable")
def test_r_pattern_and_single_imputation_helpers_execute(tmp_path: Path) -> None:
    destination = tmp_path / "results.jsonl"
    script = tmp_path / "missing.R"
    script.write_text(f"""
Sys.setenv(SIFT_RUN_TOKEN='r-missing-token', SIFT_RESULT_PATH={str(destination)!r})
source({str(RUNTIME_R)!r})
d <- data.frame(x=c(1,2,NA,4,5,6,NA,8,9,10,11,12),
                z=c(2,NA,4,5,6,7,8,9,10,11,12,13))
sift$from_missingness_pattern(d)
completed <- sift$from_single_imputation(
  d, scope='deterministic_nuisance_covariate', strategy='median')
stopifnot(!anyNA(completed), completed[3,1] == stats::median(d$x, na.rm=TRUE))
""", encoding="utf-8")
    process = subprocess.run(
        ["Rscript", str(script)], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stderr
    rows = []
    for line in destination.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload.pop("_token", None)
        rows.append(payload)
    assert [row["method_id"] for row in rows] == [
        "missingness_pattern", "single_imputation",
    ]
    for row in rows:
        clean = sanitize(row)
        assert clean.ok, clean.rejection_reason
