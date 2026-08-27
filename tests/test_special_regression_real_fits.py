"""Executable reference qualification for Stage 10 special regressions."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


PYTHON_SCRIPT = r"""
import os, sys
sys.path.insert(0, __RUNTIME__)
import sift
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.discrete.count_model import ZeroInflatedPoisson

rng = np.random.default_rng(20260822)
n = 700
x = rng.normal(size=n)
z = rng.normal(size=n)

latent = 1.15 * x + 0.25 * z + rng.logistic(size=n)
y_ord = np.digitize(latent, [-0.8, 0.65])
ordinal = OrderedModel(
    y_ord, pd.DataFrame({"x": x, "z": z}), distr="logit",
).fit(method="bfgs", disp=False)
try:
    sift.from_ordinal_model(ordinal, proportional_odds="pass")
except ValueError:
    print("forged-proportional-odds-rejected")
else:
    raise AssertionError("caller promoted proportional-odds diagnostic")
sift.from_ordinal_model(ordinal)

eta = np.column_stack((np.zeros(n), 0.25 + 0.9*x + 0.2*z, -0.2 - 0.75*x + 0.1*z))
prob = np.exp(eta)
prob /= prob.sum(axis=1)[:, None]
y_multi = np.array([rng.choice(3, p=row) for row in prob])
X = sm.add_constant(pd.DataFrame({"x": x, "z": z}))
multinomial = MNLogit(y_multi, X).fit(method="newton", disp=False)
sift.from_multinomial_model(multinomial)

mu = np.exp(0.25 + 0.45*x - 0.2*z)
y_count = rng.poisson(mu)
y_count[rng.random(n) < 0.32] = 0
zip_fit = ZeroInflatedPoisson(
    y_count, X, exog_infl=np.ones((n, 1)), inflation="logit",
).fit(method="bfgs", maxiter=300, disp=False)
sift.from_zero_inflated_model(zip_fit)

frame = pd.DataFrame({"y": np.sin(x) + 0.4*z + rng.normal(0, 0.2, n), "x": x, "z": z})
spline = smf.ols("y ~ bs(x, df=4, include_intercept=False) + z", data=frame).fit()
sift.from_spline_model(spline, basis_df=4, basis="bspline")

ordinary = smf.ols("y ~ x + z", data=frame).fit()
try:
    sift.from_spline_model(ordinary, basis_df=2, basis="bspline")
except (TypeError, ValueError):
    print("ordinary-linear-rejected")
else:
    raise AssertionError("ordinary linear fit was accepted as a spline")
"""


R_SCRIPT = r"""
Sys.setenv(SIFT_RUN_TOKEN = "qualification-token")
Sys.setenv(SIFT_RESULT_PATH = {result_path!r})
source({sift_r!r})
set.seed(20260822)
n <- 700
x <- rnorm(n)
z <- rnorm(n)
latent <- 1.15*x + 0.25*z + rlogis(n)
y_ord <- ordered(cut(latent, breaks=c(-Inf, -0.8, 0.65, Inf), labels=FALSE))
ordinal <- MASS::polr(y_ord ~ x + z, Hess=TRUE)
stopifnot(inherits(try(sift$from_ordinal_model(ordinal, proportional_odds="pass"), silent=TRUE), "try-error"))
sift$from_ordinal_model(ordinal)

eta <- cbind(0, 0.25 + 0.9*x + 0.2*z, -0.2 - 0.75*x + 0.1*z)
prob <- exp(eta) / rowSums(exp(eta))
y_multi <- factor(apply(prob, 1, function(row) sample(1:3, 1, prob=row)))
multinomial <- nnet::multinom(y_multi ~ x + z, trace=FALSE)
sift$from_multinomial_model(multinomial)

y <- sin(x) + 0.4*z + rnorm(n, sd=0.2)
spline <- lm(y ~ splines::bs(x, df=4) + z)
sift$from_spline_model(spline, basis_df=4, basis="bspline")
ordinary <- lm(y ~ x + z)
rejected <- try(sift$from_spline_model(ordinary, basis_df=2), silent=TRUE)
if (!inherits(rejected, "try-error")) stop("ordinary linear fit accepted as spline")
cat("ordinary-linear-rejected\n")
"""


def _payloads(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        raw.pop("_token", None)
        result = sanitize(raw)
        assert result.ok, result.rejection_reason
        clean = result.sanitized
        rows[clean["method_id"]] = clean
    return rows


@pytest.fixture(scope="module")
def python_special_results(tmp_path_factory) -> tuple[dict[str, dict], str]:
    base = tmp_path_factory.mktemp("special-regression-python")
    output = base / "results.jsonl"
    script = base / "fit.py"
    script.write_text(
        PYTHON_SCRIPT.replace("__RUNTIME__", repr(str(RUNTIME))),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["SIFT_RUN_TOKEN"] = "qualification-token"
    env["SIFT_RESULT_PATH"] = str(output)
    proc = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return _payloads(output), proc.stdout


@pytest.mark.parametrize(
    ("method_id", "check_id"),
    [
        ("ordinal_regression", "ordinal_thresholds"),
        ("multinomial_regression", "multinomial_equations"),
        ("zero_inflated_model", "zero_inflated_components"),
        ("spline_regression", "nonlinear_basis"),
    ],
)
def test_python_special_models_sanitize_and_verify(
    python_special_results, method_id: str, check_id: str,
) -> None:
    rows, _stdout = python_special_results
    payload = rows[method_id]
    verification = verify_payload(payload)
    assert any(
        row["id"] == check_id and row["status"] == "pass"
        for row in verification["checks"]
    )


def test_python_special_models_recover_known_structure(python_special_results) -> None:
    rows, stdout = python_special_results
    ordinal = rows["ordinal_regression"]
    assert ordinal["estimates"]["x"] > 0.7
    assert ordinal["estimates"]["threshold_1"] < ordinal["estimates"]["threshold_2"]
    assert ordinal["diagnostics"]["proportional_odds"] == "warn"
    assert "forged-proportional-odds-rejected" in stdout
    multinomial = rows["multinomial_regression"]
    assert multinomial["estimates"]["class_1#x"] > 0.5
    assert multinomial["estimates"]["class_2#x"] < -0.4
    zero = rows["zero_inflated_model"]
    assert zero["estimates"]["x"] > 0.25
    assert 0.3 < zero["metrics"]["zero_fraction"] < 0.7
    spline = rows["spline_regression"]
    assert spline["metrics"]["basis_parameter_count"] == 4
    assert abs(spline["estimates"]["z"] - 0.4) < 0.05
    assert spline["metrics"]["r_squared"] > 0.85
    assert "ordinary-linear-rejected" in stdout


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript unavailable")
def test_r_supported_special_models_real_fits(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    script = tmp_path / "fit.R"
    script.write_text(R_SCRIPT.format(
        result_path=str(output), sift_r=str(RUNTIME / "sift.R"),
    ), encoding="utf-8")
    proc = subprocess.run(
        [RSCRIPT, str(script)], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    rows = _payloads(output)
    assert rows["ordinal_regression"]["estimates"]["x"] > 0.7
    assert rows["multinomial_regression"]["estimates"]["class_1#x"] > 0.4
    spline = rows["spline_regression"]
    assert spline["metrics"]["basis_parameter_count"] == 4
    assert abs(spline["estimates"]["z"] - 0.4) < 0.06
    assert "ordinary-linear-rejected" in proc.stdout


@pytest.mark.skipif(
    not _r_package_available("pscl"), reason="R pscl package unavailable",
)
def test_r_zero_inflated_reference_fit_when_pscl_is_available(tmp_path: Path) -> None:
    output = tmp_path / "zero.jsonl"
    script = tmp_path / "zero.R"
    script.write_text(f"""
Sys.setenv(SIFT_RUN_TOKEN="qualification-token", SIFT_RESULT_PATH={str(output)!r})
source({str(RUNTIME / 'sift.R')!r})
set.seed(20260822)
n <- 700; x <- rnorm(n); z <- rnorm(n)
y <- rpois(n, exp(0.2 + 0.45*x)); y[runif(n) < plogis(-0.7 + 0.6*z)] <- 0
fit <- pscl::zeroinfl(y ~ x | z, dist="poisson")
sift$from_zero_inflated_model(fit)
""", encoding="utf-8")
    proc = subprocess.run(
        [RSCRIPT, str(script)], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    payload = _payloads(output)["zero_inflated_model"]
    assert payload["estimates"]["count_x"] > 0.25
    assert verify_payload(payload)["checks"][-1]["status"] == "pass"


def test_special_method_sanitizer_rejects_incoherent_claims(
    python_special_results,
) -> None:
    rows, _stdout = python_special_results
    for method_id, mutate in (
        ("ordinal_regression", lambda row: row["metrics"].update(threshold_count=9)),
        ("ordinal_regression", lambda row: row["diagnostics"].update(proportional_odds="pass")),
        ("multinomial_regression", lambda row: row["metrics"].update(min_category_n=2)),
        ("zero_inflated_model", lambda row: row["metrics"].update(zero_fraction=1.2)),
        ("spline_regression", lambda row: row["metrics"].update(basis_parameter_count=3)),
    ):
        raw = json.loads(json.dumps(rows[method_id]))
        mutate(raw)
        assert not sanitize(raw).ok
