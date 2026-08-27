"""Executable qualification for complex-survey point and variance estimators."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sift.sanitizer import sanitize
from sift.verification import verify_payload
from tests.runtime_probes import r_package_loadable


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "sift" / "runtime"
RSCRIPT = shutil.which("Rscript")


def _r_survey_available() -> bool:
    return r_package_loadable(RSCRIPT, "survey")


PYTHON_SCRIPT = r"""
import sys
sys.path.insert(0, __RUNTIME__)
import sift
import numpy as np

rng = np.random.default_rng(20260822)
strata = np.repeat(np.arange(3), 16)
psu = np.tile(np.repeat(np.arange(4), 4), 3)
secondary = np.tile(np.arange(4), 12)
n = len(strata)
x = rng.normal(size=n)
cluster_effect = np.repeat(rng.normal(scale=0.7, size=12), 4)
y = 10.0 + 1.4*x + cluster_effect + rng.normal(scale=0.25, size=n)
weights = 0.8 + (np.arange(n) % 7) / 5.0

sift.from_survey_mean(
    y, weights, strata=strata, psu=psu, analysis_id="mean_base",
)
sift.from_survey_mean(
    y, weights, strata=strata, psu=psu, fpc=np.full(n, 0.25),
    analysis_id="mean_fpc",
)
sift.from_survey_mean(
    y, weights, strata=strata, psu=psu, fpc=np.full(n, 16),
    fpc_mode="population_size", analysis_id="mean_fpc_population",
)
sift.from_survey_mean(
    y, weights, strata=strata, psu=np.column_stack((psu, secondary)),
    fpc=np.column_stack((np.full(n, 0.25), np.full(n, 0.5))),
    analysis_id="mean_multistage",
)
sift.from_survey_mean(
    y, weights, strata=strata, psu=np.column_stack((psu, secondary)),
    stage1_inclusion_probabilities=np.full(n, 0.25),
    analysis_id="mean_multistage_probabilities",
)
sift.from_survey_mean(
    (x > 0).astype(float), weights, proportion=True,
    strata=strata, psu=psu, analysis_id="proportion",
)

# Delete-one-PSU jackknife replicates. The helper applies the documented
# (R-1)/R scaling; the test independently recomputes it from these weights.
first_units = list(dict.fromkeys(zip(strata.tolist(), psu.tolist())))
replicates = []
for stratum_value, psu_value in first_units:
    keep = ~((strata == stratum_value) & (psu == psu_value))
    replicate = weights.copy()
    replicate[~keep] = 0.0
    replicate[keep] *= len(first_units) / (len(first_units) - 1.0)
    replicates.append(replicate)
replicate_weights = np.column_stack(replicates)
sift.from_survey_mean(
    y, weights, replicate_weights=replicate_weights,
    replicate_method="jackknife", analysis_id="mean_jackknife",
)
sift.from_survey_mean(
    y, weights, replicate_weights=replicate_weights,
    replicate_method="brr", analysis_id="mean_brr",
)
sift.from_survey_mean(
    y, weights, replicate_weights=replicate_weights,
    replicate_method="fay", fay_rho=0.3, analysis_id="mean_fay",
)
sift.from_survey_mean(
    y, weights, replicate_weights=replicate_weights,
    replicate_method="bootstrap", analysis_id="mean_bootstrap",
)
sift.from_survey_mean(
    y, weights, replicate_weights=replicate_weights,
    replicate_method="bootstrap", replicate_mse=True,
    replicate_scale=0.25,
    replicate_rscales=np.linspace(0.5, 1.5, replicate_weights.shape[1]),
    analysis_id="mean_bootstrap_custom",
)

X = np.column_stack((np.ones(n), x))
sift.from_survey_regression(
    y, X, weights, predictor_names=["intercept", "x"],
    strata=strata, psu=psu, analysis_id="regression",
)

# A singleton stratum is rejected by default and accepted only under an
# explicit adjustment policy; both paths are qualified.
lonely_strata = np.r_[np.zeros(4), np.ones(8)]
lonely_psu_ids = np.r_[np.zeros(4), np.repeat([0, 1], 4)]
lonely_y = np.arange(12, dtype=float)
lonely_w = np.ones(12)
try:
    sift.from_survey_mean(
        lonely_y, lonely_w, strata=lonely_strata, psu=lonely_psu_ids,
    )
except ValueError:
    print("lonely-default-rejected")
else:
    raise AssertionError("lonely PSU was silently accepted")
sift.from_survey_mean(
    lonely_y, lonely_w, strata=lonely_strata, psu=lonely_psu_ids,
    lonely_psu="adjust", analysis_id="lonely_adjusted",
)
try:
    sift.from_survey_mean(
        lonely_y, lonely_w, strata=lonely_strata, psu=lonely_psu_ids,
        lonely_psu="certainty",
    )
except ValueError:
    print("unproven-certainty-rejected")
else:
    raise AssertionError("certainty policy accepted without a census FPC")
sift.from_survey_mean(
    lonely_y, lonely_w, strata=lonely_strata, psu=lonely_psu_ids,
    fpc=np.r_[np.ones(4), np.zeros(8)], lonely_psu="certainty",
    analysis_id="lonely_certainty",
)

bad_cases = [
    lambda: sift.from_survey_mean(y, np.r_[0.0, weights[1:]]),
    lambda: sift.from_survey_mean(y, weights, strata=strata, psu=psu,
                                  fpc=np.r_[1.2, np.full(n-1, 0.2)]),
    lambda: sift.from_survey_mean(y, weights, replicate_weights=replicate_weights,
                                  replicate_method="brr", fpc=np.zeros(n)),
    lambda: sift.from_survey_mean(y, weights, strata=strata,
                                  psu=np.column_stack((psu, secondary))),
]
for operation in bad_cases:
    try:
        operation()
    except ValueError:
        pass
    else:
        raise AssertionError("bad survey design was accepted")
print("bad-designs-rejected")
"""


def _read_results(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        raw.pop("_token", None)
        clean = sanitize(raw)
        assert clean.ok, clean.rejection_reason
        payload = clean.sanitized
        rows[payload["analysis_id"]] = payload
    return rows


@pytest.fixture(scope="module")
def survey_results(tmp_path_factory) -> tuple[dict[str, dict], str]:
    directory = tmp_path_factory.mktemp("survey-reference")
    output = directory / "results.jsonl"
    script = directory / "fit.py"
    script.write_text(
        PYTHON_SCRIPT.replace("__RUNTIME__", repr(str(RUNTIME))),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["SIFT_RUN_TOKEN"] = "qualification-token"
    env["SIFT_RESULT_PATH"] = str(output)
    process = subprocess.run(
        [sys.executable, str(script)], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stderr
    return _read_results(output), process.stdout


def test_probability_weight_point_estimates_match_maintained_references(
    survey_results,
) -> None:
    rows, _stdout = survey_results
    rng = np.random.default_rng(20260822)
    _strata = np.repeat(np.arange(3), 16)
    _psu = np.tile(np.repeat(np.arange(4), 4), 3)
    n = len(_strata)
    x = rng.normal(size=n)
    cluster_effect = np.repeat(rng.normal(scale=0.7, size=12), 4)
    y = 10.0 + 1.4*x + cluster_effect + rng.normal(scale=0.25, size=n)
    weights = 0.8 + (np.arange(n) % 7) / 5.0
    from statsmodels.stats.weightstats import DescrStatsW
    import statsmodels.api as sm

    assert rows["mean_base"]["estimates"]["mean"] == pytest.approx(
        DescrStatsW(y, weights=weights).mean, rel=2e-4,
    )
    wls = sm.WLS(y, np.column_stack((np.ones(n), x)), weights=weights).fit()
    assert rows["regression"]["estimates"]["x"] == pytest.approx(
        wls.params[1], abs=0.01,
    )


def test_strata_psu_fpc_and_multistage_variances_are_coherent(survey_results) -> None:
    rows, _stdout = survey_results
    base = rows["mean_base"]
    fpc = rows["mean_fpc"]
    multistage = rows["mean_multistage"]
    assert base["metrics"]["strata_count"] == 3
    assert base["metrics"]["psu_count"] == 12
    assert base["metrics"]["design_df"] == 9
    assert fpc["metrics"]["variance"] == pytest.approx(
        0.75 * base["metrics"]["variance"], rel=0.015,
    )
    assert rows["mean_fpc_population"]["metrics"]["variance"] == pytest.approx(
        fpc["metrics"]["variance"], rel=0.015,
    )
    assert multistage["metrics"]["stage_count"] == 2
    assert multistage["metrics"]["secondary_psu_count"] == 48
    assert multistage["metrics"]["variance"] > fpc["metrics"]["variance"]
    assert rows["mean_multistage_probabilities"]["metrics"]["stage_count"] == 2


def test_replicate_weight_formula_and_design_effect_reporting(survey_results) -> None:
    rows, _stdout = survey_results
    result = rows["mean_jackknife"]
    assert result["variance_method"] == "jackknife"
    assert result["metrics"]["replicate_count"] == 12
    # Recreate the replicate estimates and the JK1 variance independently.
    rng = np.random.default_rng(20260822)
    strata = np.repeat(np.arange(3), 16)
    psu = np.tile(np.repeat(np.arange(4), 4), 3)
    n = len(strata)
    x = rng.normal(size=n)
    cluster_effect = np.repeat(rng.normal(scale=0.7, size=12), 4)
    y = 10.0 + 1.4*x + cluster_effect + rng.normal(scale=0.25, size=n)
    weights = 0.8 + (np.arange(n) % 7) / 5.0
    full = np.dot(weights, y) / weights.sum()
    replicate_estimates = []
    units = list(dict.fromkeys(zip(strata.tolist(), psu.tolist())))
    for stratum_value, psu_value in units:
        wr = weights.copy()
        deleted = (strata == stratum_value) & (psu == psu_value)
        wr[deleted] = 0
        wr[~deleted] *= len(units) / (len(units) - 1)
        replicate_estimates.append(np.dot(wr, y) / wr.sum())
    expected = (len(units) - 1) / len(units) * np.sum(
        (np.asarray(replicate_estimates) - full) ** 2
    )
    assert result["metrics"]["variance"] == pytest.approx(expected, rel=0.015)
    assert result["metrics"]["design_effect"] >= 0
    brr = rows["mean_brr"]["metrics"]["variance"]
    assert rows["mean_fay"]["metrics"]["variance"] == pytest.approx(
        brr / 0.7**2, rel=0.015,
    )
    replicate_mean = np.mean(replicate_estimates)
    expected_bootstrap = np.sum(
        (np.asarray(replicate_estimates) - replicate_mean) ** 2
    ) / (len(units) - 1)
    assert rows["mean_bootstrap"]["metrics"]["variance"] == pytest.approx(
        expected_bootstrap, rel=0.015,
    )
    assert rows["mean_bootstrap"]["metrics"]["replicate_mse"] == 0
    custom = rows["mean_bootstrap_custom"]
    rscales = np.linspace(0.5, 1.5, len(units))
    expected_custom = 0.25 * np.sum(
        rscales * (np.asarray(replicate_estimates) - full) ** 2
    )
    assert custom["metrics"]["variance"] == pytest.approx(expected_custom, rel=0.015)
    assert custom["metrics"]["replicate_mse"] == 1
    assert result["metrics"]["variance"] == pytest.approx(brr * 11, rel=0.015)


@pytest.mark.parametrize(
    "analysis_id",
    [
        "mean_base", "mean_fpc", "mean_fpc_population", "mean_multistage",
        "mean_multistage_probabilities", "proportion", "mean_jackknife",
        "mean_brr", "mean_fay", "mean_bootstrap", "mean_bootstrap_custom",
        "regression",
    ],
)
def test_survey_results_pass_independent_verifier(survey_results, analysis_id: str) -> None:
    rows, _stdout = survey_results
    verification = verify_payload(rows[analysis_id])
    required = {"survey_design_structure", "survey_variance_identity", "survey_effective_sample"}
    assert required <= {
        check["id"] for check in verification["checks"] if check["status"] == "pass"
    }


def test_lonely_psu_and_bad_designs_are_explicit(survey_results) -> None:
    rows, stdout = survey_results
    adjusted = rows["lonely_adjusted"]
    assert adjusted["metrics"]["lonely_strata_count"] == 1
    assert adjusted["metrics"]["lonely_adjusted_count"] == 1
    assert adjusted["metrics"]["lonely_certainty_count"] == 0
    # Independent grand-mean adjustment for the singleton PSU.  For a mean,
    # the linearized PSU contributions are sums of (y - mean) / n.
    y = np.arange(12, dtype=float)
    influence = (y - y.mean()) / len(y)
    psu_totals = np.array([
        influence[:4].sum(), influence[4:8].sum(), influence[8:].sum(),
    ])
    grand = psu_totals.mean()
    nonlonely = 2.0 * np.sum((psu_totals[1:] - psu_totals[1:].mean()) ** 2)
    expected = nonlonely + (psu_totals[0] - grand) ** 2
    assert adjusted["metrics"]["variance"] == pytest.approx(expected, rel=0.015)
    certainty = rows["lonely_certainty"]
    assert certainty["metrics"]["lonely_certainty_count"] == 1
    assert certainty["metrics"]["lonely_adjusted_count"] == 0
    assert adjusted["diagnostics"]["lonely_psu"] == "warn"
    assert "lonely-default-rejected" in stdout
    assert "unproven-certainty-rejected" in stdout
    assert "bad-designs-rejected" in stdout


def test_survey_sanitizer_rejects_forged_variance_identity(survey_results) -> None:
    rows, _stdout = survey_results
    forged = json.loads(json.dumps(rows["regression"]))
    forged["metrics"]["variance#x"] *= 9
    assert not sanitize(forged).ok
    forged_replicate = json.loads(json.dumps(rows["mean_bootstrap"]))
    forged_replicate["metrics"].pop("replicate_mse")
    assert not sanitize(forged_replicate).ok
    forged_certainty = json.loads(json.dumps(rows["lonely_certainty"]))
    forged_certainty["metrics"]["lonely_certainty_count"] = 0
    forged_certainty["metrics"]["lonely_adjusted_count"] = 1
    assert not sanitize(forged_certainty).ok


@pytest.mark.skipif(not _r_survey_available(), reason="R survey package unavailable")
def test_r_survey_reference_differential(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    script = tmp_path / "survey.R"
    script.write_text(f"""
Sys.setenv(SIFT_RUN_TOKEN='qualification-token', SIFT_RESULT_PATH={str(output)!r})
source({str(RUNTIME / 'sift.R')!r})
set.seed(20260822)
d <- data.frame(stratum=rep(1:3, each=16), psu=rep(rep(1:4, each=4),3),
                x=rnorm(48), weight=0.8 + (0:47 %% 7)/5)
d$y <- 10 + 1.4*d$x + rep(rnorm(12, sd=.7), each=4) + rnorm(48, sd=.25)
design <- survey::svydesign(ids=~psu, strata=~stratum, weights=~weight, data=d, nest=TRUE)
sift$from_survey_mean(design, ~y, analysis_id='r_mean')
fit <- survey::svyglm(y ~ x, design=design)
sift$from_survey_regression(fit, analysis_id='r_regression')
""", encoding="utf-8")
    process = subprocess.run(
        [RSCRIPT, str(script)], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert process.returncode == 0, process.stderr
    rows = _read_results(output)
    assert rows["r_mean"]["metrics"]["design_df"] == 9
    assert rows["r_regression"]["estimates"]["x"] > 1.0
