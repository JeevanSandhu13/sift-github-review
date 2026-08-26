"""Executable-reference qualification for longitudinal/survival methods."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sift.sanitizer import sanitize
from sift.verification import verify_payload


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "audit_longitudinal_survival_methods.py"
METHODS = {
    "growth_curve", "gee", "panel_random_effects", "competing_risks",
    "recurrent_events", "time_varying_survival",
}


@pytest.fixture(scope="module")
def qualified_payloads(tmp_path_factory) -> dict[str, dict]:
    pytest.importorskip("statsmodels")
    destination = tmp_path_factory.mktemp("longitudinal_survival") / "results.jsonl"
    environment = os.environ.copy()
    environment["SIFT_RUN_TOKEN"] = "method-qualification-token"
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
def test_real_fit_payloads_are_sanitizer_valid_and_aggregate_only(
    qualified_payloads: dict[str, dict], method_id: str,
) -> None:
    clean = sanitize(qualified_payloads[method_id])
    assert clean.ok, clean.rejection_reason
    result = clean.sanitized
    assert result["method_id"] == method_id
    assert result.get("estimates")
    forbidden = {
        "rows", "observations", "predictions", "residuals",
        "fitted_values", "influence_by_case",
    }
    assert forbidden.isdisjoint(result)


def test_growth_curve_recovers_known_quadratic_trajectory(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["growth_curve"]).sanitized
    assert result["estimates"]["time"] == pytest.approx(0.7, abs=0.12)
    assert result["estimates"]["I(time^2)"] == pytest.approx(-0.08, abs=0.035)
    assert result["diagnostics"]["random_effect_structure"] == "pass"
    assert result["clusters"] == 32 and result["records"] == 160


def test_gee_recovers_positive_population_average_time_association(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["gee"]).sanitized
    assert 0.1 < result["estimates"]["time"] < 0.6
    assert result["uncertainty_type"] == "robust"
    assert result["diagnostics"]["convergence"] is True
    sensitivity = result["diagnostics"]["working_correlation_sensitivity"]
    assert 0 <= sensitivity < 0.1
    assert result["metrics"]["working_correlation_max_abs_change"] == pytest.approx(
        sensitivity,
    )


def test_panel_random_effects_recovers_slope_and_runs_hausman_comparison(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["panel_random_effects"]).sanitized
    assert result["estimates"]["x"] == pytest.approx(0.65, abs=0.08)
    hausman = result["diagnostics"]["hausman"]
    assert 0.05 < hausman <= 1.0
    assert result["metrics"]["hausman_p_value"] == pytest.approx(hausman)


def test_competing_risks_returns_valid_cumulative_incidence_mass(
    qualified_payloads: dict[str, dict],
) -> None:
    result = sanitize(qualified_payloads["competing_risks"]).sanitized
    cumulative = result["estimates"]
    assert set(cumulative) == {"cause_1_final", "cause_2_final"}
    assert all(0 < value < 1 for value in cumulative.values())
    assert sum(cumulative.values()) < 1
    checks = verify_payload(result)["checks"]
    assert any(
        row["id"] == "competing_risk_probability_mass"
        and row["status"] == "pass" for row in checks
    )


def test_counting_process_fits_recover_positive_effects_and_counts(
    qualified_payloads: dict[str, dict],
) -> None:
    recurrent = sanitize(qualified_payloads["recurrent_events"]).sanitized
    time_varying = sanitize(
        qualified_payloads["time_varying_survival"]
    ).sanitized
    assert 0.2 < recurrent["estimates"]["x1"] < 1.1
    assert 0.4 < time_varying["estimates"]["x1"] < 2.2
    assert recurrent["records"] > recurrent["subjects"]
    assert recurrent["events"] > recurrent["subjects"]
    assert recurrent["diagnostics"]["within_subject_dependence"] == "pass"
    assert time_varying["records"] > time_varying["subjects"]
    assert time_varying["diagnostics"]["interval_integrity"] == "pass"


@pytest.mark.parametrize(
    ("method_id", "mutation", "message"),
    [
        ("growth_curve", {"clusters": 31}, "cluster/record"),
        ("gee", {"uncertainty_type": "classical"}, "sandwich"),
        (
            "panel_random_effects",
            {"metrics": {"hausman_p_value": 0.99}},
            "Hausman",
        ),
        (
            "competing_risks",
            {
                "estimates": {"cause_1_final": 1.2},
                "standard_errors": None, "ci_lower": None, "ci_upper": None,
            },
            "cumulative-incidence",
        ),
        ("recurrent_events", {"clusters": 1}, "subject-clustered"),
    ],
)
def test_method_specific_sanitizer_invariants_fail_closed(
    qualified_payloads: dict[str, dict],
    method_id: str,
    mutation: dict,
    message: str,
) -> None:
    payload = dict(qualified_payloads[method_id])
    payload.update(mutation)
    result = sanitize(payload)
    assert not result.ok
    assert message in result.rejection_reason


def test_time_varying_interval_failure_is_rejected(
    qualified_payloads: dict[str, dict],
) -> None:
    payload = dict(qualified_payloads["time_varying_survival"])
    payload["diagnostics"] = dict(payload["diagnostics"])
    payload["diagnostics"]["interval_integrity"] = "fail"
    result = sanitize(payload)
    assert not result.ok
    assert "interval" in result.rejection_reason


def test_competing_risk_probability_mass_above_one_is_rejected(
    qualified_payloads: dict[str, dict],
) -> None:
    payload = dict(qualified_payloads["competing_risks"])
    payload.update({
        "estimates": {"cause_1_final": 0.8, "cause_2_final": 0.8},
        "standard_errors": None, "ci_lower": None, "ci_upper": None,
    })
    result = sanitize(payload)
    assert not result.ok
    assert "cumulative-incidence" in result.rejection_reason


def test_typed_helpers_reject_unproven_covariance_and_indefinite_hausman(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "negative.jsonl"
    code = f"""
import json, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.duration.hazard_regression import PHReg
sys.path.insert(0, {str(ROOT / 'src' / 'sift' / 'runtime')!r})
import sift as sr

out = {{}}
class Model:
    exog_names = ['x']
class Fit:
    model = Model()
    fe_params = pd.Series([0.2], index=['x'])
    params = fe_params
    def cov_params(self): return np.asarray([[1.0]])
class Fixed:
    model = Model()
    params = pd.Series([0.3], index=['x'])
    def cov_params(self): return np.asarray([[0.2]])
try:
    sr._hausman_p_value(Fit(), Fixed())
except ValueError as exc:
    out['hausman'] = 'indefinite' in str(exc)

groups = np.repeat(np.arange(12), 3)
x = np.tile(np.arange(3), 12).astype(float)
y = (x + groups % 2 > 1).astype(int)
gee = sm.GEE(y, sm.add_constant(x), groups=groups,
             family=sm.families.Binomial()).fit(cov_type='naive')
try:
    sr.from_gee(gee, time_values=x)
except ValueError as exc:
    out['gee'] = "cov_type='robust'" in str(exc)

start = np.tile(np.arange(3), 12).astype(float)
stop = start + 1
event = np.tile([1, 0, 1], 12)
ph = PHReg(stop, (groups % 2)[:, None], status=event, entry=start).fit()
try:
    sr.from_recurrent_events(ph)
except ValueError as exc:
    out['phreg'] = 'fit(groups=subject_ids)' in str(exc)
print(json.dumps(out, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["SIFT_RUN_TOKEN"] = "negative-qualification-token"
    environment["SIFT_RESULT_PATH"] = str(result_path)
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "gee": True, "hausman": True, "phreg": True,
    }
    assert not result_path.exists()
