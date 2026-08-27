"""Strict aggregate contracts for optional measurement references.

These tests do not qualify the optional methods as executable.  The ledger
remains open until lavaan/poLCA are installed and the conditional real-fit
tests can run.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from sift.sanitizer import sanitize
from sift.verification import verify_payload
from tests.runtime_probes import r_package_loadable


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "sift" / "runtime"
RSCRIPT = shutil.which("Rscript")


def _r_package_available(package: str) -> bool:
    return r_package_loadable(RSCRIPT, package)


def _read_r_results(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line); raw.pop("_token", None)
        result = sanitize(raw)
        assert result.ok, result.rejection_reason
        rows[result.sanitized["method_id"]] = result.sanitized
    return rows


def _diagnostics(method: str) -> dict:
    common = {
        "sampling_adequacy": "pass", "fit_or_stability": "pass",
        "component_or_class_support": "pass",
    }
    if method == "confirmatory_factor_analysis":
        common.update(cfi=.97, tli=.96, rmsea=.04, srmr=.03)
    elif method == "measurement_invariance":
        common.update(configural_fit="pass", metric_change="pass", scalar_change="pass")
    else:
        common.update(class_sizes="pass", entropy="pass", solution_stability="pass")
    return common


def _cfa() -> dict:
    estimates = {f"loading_{i}": 0.7 + i / 100 for i in range(1, 7)}
    errors = {key: .04 for key in estimates}
    return {
        "type": "method_result", "method_id": "confirmatory_factor_analysis", "n": 500,
        "diagnostics": _diagnostics("confirmatory_factor_analysis"),
        "estimates": estimates, "standard_errors": errors,
        "p_values": {key: .001 for key in estimates},
        "ci_lower": {key: value - .08 for key, value in estimates.items()},
        "ci_upper": {key: value + .08 for key, value in estimates.items()},
        "metrics": {"factor_count": 2, "indicator_count": 6, "loading_count": 6,
                    "degrees_of_freedom": 8, "chi_square": 13.2,
                    "cfi": .97, "tli": .96, "rmsea": .04, "srmr": .03},
        "uncertainty_type": "classical",
    }


def _invariance() -> dict:
    metrics = {"group_count": 2, "indicator_count": 6}
    rows = {
        "configural": (.97, .04, 14.0, 8),
        "metric": (.965, .045, 20.0, 12),
        "scalar": (.958, .052, 29.0, 16),
    }
    for model, values in rows.items():
        for field, value in zip(("cfi", "rmsea", "chisq", "df"), values, strict=True):
            metrics[f"{model}_{field}"] = value
    return {
        "type": "method_result", "method_id": "measurement_invariance", "n": 500,
        "diagnostics": _diagnostics("measurement_invariance"),
        "estimates": {"metric_delta_cfi": .005, "metric_delta_rmsea": .005,
                      "scalar_delta_cfi": .007, "scalar_delta_rmsea": .007},
        "p_values": {"metric_nested": .21, "scalar_nested": .08},
        "metrics": metrics, "uncertainty_type": "classical",
    }


def _latent() -> dict:
    return {
        "type": "method_result", "method_id": "latent_class", "n": 500,
        "diagnostics": _diagnostics("latent_class"),
        "estimates": {"class_1": .42, "class_2": .35, "class_3": .23},
        "metrics": {"class_count": 3, "start_count": 20, "stable_start_count": 4,
                    "min_expected_class_n": 115, "normalized_entropy": .78,
                    "best_log_likelihood": -1022.3, "second_best_gap": .00002,
                    "likelihood_tolerance": .0001, "aic": 2100.6, "bic": 2218.5},
        "uncertainty_type": "classical",
    }


@pytest.mark.parametrize(
    ("raw", "check_id"),
    [(_cfa(), "cfa_fit_contract"), (_invariance(), "invariance_nested_models"),
     (_latent(), "latent_class_stability")],
)
def test_optional_measurement_aggregate_contracts(raw: dict, check_id: str) -> None:
    result = sanitize(raw)
    assert result.ok, result.rejection_reason
    checks = verify_payload(result.sanitized)["checks"]
    assert any(check["id"] == check_id and check["status"] == "pass" for check in checks)


def test_cfa_rejects_unidentified_or_invented_fit_indices() -> None:
    raw = _cfa(); raw["metrics"]["degrees_of_freedom"] = 0
    assert not sanitize(raw).ok
    raw = _cfa(); raw["diagnostics"]["srmr"] = .20
    assert not sanitize(raw).ok


def test_invariance_rejects_non_nested_or_miscalculated_changes() -> None:
    raw = _invariance(); raw["metrics"]["scalar_df"] = 10
    assert not sanitize(raw).ok
    raw = _invariance(); raw["estimates"]["metric_delta_cfi"] = .08
    assert not sanitize(raw).ok


def test_latent_class_rejects_single_optimum_or_tiny_class() -> None:
    raw = _latent(); raw["metrics"]["stable_start_count"] = 1
    assert not sanitize(raw).ok
    raw = _latent(); raw["metrics"]["min_expected_class_n"] = 4
    assert not sanitize(raw).ok
    raw = _latent(); raw["estimates"]["class_1"] = .8
    assert not sanitize(raw).ok


def test_optional_measurement_contract_has_no_row_level_fields() -> None:
    for raw in (_cfa(), _invariance(), _latent()):
        leaked = copy.deepcopy(raw)
        leaked["posterior_rows"] = [[.2, .8]]
        result = sanitize(leaked)
        assert result.ok
        assert "posterior_rows" not in result.sanitized


@pytest.mark.skipif(not _r_package_available("lavaan"), reason="R lavaan unavailable")
def test_lavaan_cfa_and_invariance_real_fits_when_available(tmp_path: Path) -> None:
    output = tmp_path / "lavaan.jsonl"
    script = tmp_path / "lavaan.R"
    script.write_text(f"""
Sys.setenv(SIFT_RUN_TOKEN='qualification-token', SIFT_RESULT_PATH={str(output)!r})
source({str(RUNTIME / 'sift.R')!r})
set.seed(20260822); n <- 800; group <- rep(c('a','b'), each=n/2); eta <- rnorm(n)
d <- data.frame(group=group,
 y1=.90*eta+rnorm(n,sd=.35), y2=.82*eta+rnorm(n,sd=.42),
 y3=.78*eta+rnorm(n,sd=.46), y4=.86*eta+rnorm(n,sd=.38))
model <- 'factor =~ y1 + y2 + y3 + y4'
configural <- lavaan::cfa(model, data=d, group='group')
metric <- lavaan::cfa(model, data=d, group='group', group.equal='loadings')
scalar <- lavaan::cfa(model, data=d, group='group', group.equal=c('loadings','intercepts'))
wrong_metric <- lavaan::cfa(model, data=d, group='group', group.equal='residuals')
wrong_scalar <- lavaan::cfa(model, data=d, group='group', group.equal=c('residuals','means'))
wrong <- try(sift$from_lavaan_invariance(configural, wrong_metric, wrong_scalar), silent=TRUE)
if (!inherits(wrong, 'try-error')) stop('wrong increasingly-constrained sequence accepted')
sift$from_lavaan_cfa(configural)
sift$from_lavaan_invariance(configural, metric, scalar)
""", encoding="utf-8")
    process = subprocess.run([RSCRIPT, str(script)], cwd=ROOT, capture_output=True,
                             text=True, timeout=180)
    assert process.returncode == 0, process.stderr
    rows = _read_r_results(output)
    assert rows["confirmatory_factor_analysis"]["metrics"]["cfi"] > .95
    checks = verify_payload(rows["measurement_invariance"])["checks"]
    assert any(row["id"] == "invariance_nested_models" and row["status"] == "pass"
               for row in checks)


@pytest.mark.skipif(not _r_package_available("poLCA"), reason="R poLCA unavailable")
def test_polca_multistart_real_fit_when_available(tmp_path: Path) -> None:
    output = tmp_path / "polca.jsonl"
    script = tmp_path / "polca.R"
    script.write_text(f"""
Sys.setenv(SIFT_RUN_TOKEN='qualification-token', SIFT_RESULT_PATH={str(output)!r})
source({str(RUNTIME / 'sift.R')!r})
set.seed(20260822); n <- 900; cls <- sample(1:3, n, replace=TRUE, prob=c(.4,.35,.25))
prob <- matrix(c(.9,.85,.2,.15, .2,.25,.85,.8, .75,.2,.75,.2), nrow=3, byrow=TRUE)
d <- as.data.frame(sapply(1:4, function(j) 1L + rbinom(n,1,prob[cls,j])))
names(d) <- paste0('i',1:4); f <- cbind(i1,i2,i3,i4) ~ 1
fits <- lapply(1:10, function(seed) {{ set.seed(seed); poLCA::poLCA(f,d,nclass=3,nrep=1,verbose=FALSE) }})
repeated <- try(sift$from_polca(rep(list(fits[[1]]), 5)), silent=TRUE)
if (!inherits(repeated, 'try-error')) stop('repeated identical starts accepted')
wrong_fit <- poLCA::poLCA(cbind(i1,i2,i3)~1,d,nclass=3,nrep=1,verbose=FALSE)
mixed <- fits; mixed[[length(mixed)]] <- wrong_fit
wrong_manifest <- try(sift$from_polca(mixed), silent=TRUE)
if (!inherits(wrong_manifest, 'try-error')) stop('different manifest variables accepted')
sift$from_polca(fits, likelihood_tolerance=1e-3, minimum_class_n=10)
""", encoding="utf-8")
    process = subprocess.run([RSCRIPT, str(script)], cwd=ROOT, capture_output=True,
                             text=True, timeout=180)
    assert process.returncode == 0, process.stderr
    payload = _read_r_results(output)["latent_class"]
    assert payload["metrics"]["stable_start_count"] >= 2
    assert payload["metrics"]["min_expected_class_n"] >= 10
