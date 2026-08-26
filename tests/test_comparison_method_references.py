"""Executable reference qualification for Stage 10 comparison methods.

Registry presence is not evidence that a method works.  These tests run the
maintained Python and R reference implementations on fixed synthetic data,
pass the real fitted objects through the bundled typed runtime helpers, then
require sanitizer acceptance, deterministic verification, and known answers.
"""

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
from sift.methodology import METHODS


ROOT = Path(__file__).resolve().parents[1]
PY_RUNTIME = ROOT / "src" / "sift" / "runtime"
R_RUNTIME = PY_RUNTIME / "sift.R"
METHOD_IDS = {
    "descriptive_confidence_interval", "nonparametric_test",
    "proportion_test", "anova", "ancova", "repeated_measures_test",
    "multiple_testing_correction",
}


PYTHON_QUALIFICATION = r'''
import json
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.weightstats import DescrStatsW
import sift

def mark(label):
    sift._write_result({"_qualification_label": label})

x = np.arange(1.0, 21.0)
sift.from_descriptive_confidence_interval(
    DescrStatsW(x), missing_count=0, name="mean"
)
mark("descriptive_confidence_interval")

a = np.arange(1.0, 11.0)
b = np.arange(11.0, 21.0)
sift.from_nonparametric_test(
    stats.mannwhitneyu(a, b, alternative="two-sided"), n=20,
    group_sizes=[10, 10], ties_checked="pass", name="mann_whitney_u"
)
mark("nonparametric_test")

sift.from_proportion_test(60, 100, value=0.5, name="proportion")
mark("proportion_test")

group = np.repeat(["a", "b", "c"], 20)
within = np.tile(np.linspace(-1.0, 1.0, 20), 3)
df = pd.DataFrame({"group": group, "x": within})
df["y"] = 3.0 + 2.0 * df["x"] + df["group"].map({"a": 0.0, "b": 1.0, "c": 2.0})
anova_fit = ols("y ~ C(group)", data=df).fit()
sift.from_anova(anova_fit, diagnostics={
    "group_sample_sizes": "pass", "residual_distribution": "pass",
    "homogeneity_of_variance": "pass",
})
mark("anova")

ancova_fit = ols("y ~ x + C(group)", data=df).fit()
sift.from_anova(ancova_fit, method_id="ancova", diagnostics={
    "group_sample_sizes": "pass", "residual_distribution": "pass",
    "homogeneity_of_variance": "pass", "parallel_slopes": "pass",
})
mark("ancova")

rows = []
for subject in range(12):
    for time in range(3):
        rows.append({"subject": subject, "time": str(time),
                     "y": subject * 0.1 + time * 2.0 + (subject % 3) * 0.01 * time})
repeated_df = pd.DataFrame(rows)
rm = AnovaRM(repeated_df, "y", "subject", within=["time"]).fit()
sift.from_repeated_measures(
    rm, n=36, subjects=12, records=36,
    diagnostics={"sphericity_or_correction": "pass"},
)
mark("repeated_measures_test")

sift.from_multiple_testing(
    [0.01, 0.02, 0.03, 0.5], n=100, method="holm",
    labels=["h1", "h2", "h3", "h4"],
)
mark("multiple_testing_correction")
'''


R_QUALIFICATION = r'''
Sys.setenv(SIFT_RUN_TOKEN = "qualification-token")
Sys.setenv(SIFT_RESULT_PATH = "{result_path}")
source("{runtime}")

mark <- function(label) sift$.write_result(list(`_qualification_label` = label))

sift$from_descriptive_confidence_interval(1:20, name = "mean")
mark("descriptive_confidence_interval")

a <- 1:10; b <- 11:20
sift$from_nonparametric_test(
  wilcox.test(a, b, exact = TRUE), n = 20, group_sizes = c(10, 10),
  ties_checked = "pass", name = "wilcoxon_w"
)
mark("nonparametric_test")

sift$from_proportion_test(
  prop.test(60, 100, p = 0.5, correct = FALSE), nobs = 100,
  name = "proportion"
)
mark("proportion_test")

group <- factor(rep(c("a", "b", "c"), each = 20))
x <- rep(seq(-1, 1, length.out = 20), 3)
y <- 3 + 2*x + c(a = 0, b = 1, c = 2)[as.character(group)]
df <- data.frame(y = y, x = x, group = group)
sift$from_anova(
  lm(y ~ group, data = df), diagnostics = list(
    group_sample_sizes = "pass", residual_distribution = "pass",
    homogeneity_of_variance = "pass"
  )
)
mark("anova")

sift$from_anova(
  lm(y ~ x + group, data = df), method_id = "ancova",
  diagnostics = list(
    group_sample_sizes = "pass", residual_distribution = "pass",
    homogeneity_of_variance = "pass", parallel_slopes = "pass"
  )
)
mark("ancova")

subject <- factor(rep(1:12, each = 3))
time <- factor(rep(1:3, times = 12))
response <- as.numeric(subject) * 0.1 + as.numeric(time) * 2 +
  (as.numeric(subject) %% 3) * 0.01 * as.numeric(time)
rm <- friedman.test(response, time, subject)
sift$from_repeated_measures(rm, n = 36, subjects = 12, records = 36)
mark("repeated_measures_test")

sift$from_multiple_testing(
  c(0.01, 0.02, 0.03, 0.5), n = 100, method = "holm",
  labels = c("h1", "h2", "h3", "h4")
)
mark("multiple_testing_correction")
'''


def _read_labelled(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    pending: dict | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload.pop("_token", None)
        label = payload.get("_qualification_label")
        if label is not None:
            assert pending is not None
            result[str(label)] = pending
            pending = None
        else:
            assert pending is None
            pending = payload
    assert pending is None
    return result


def _qualify_all(payloads: dict[str, dict]) -> dict[str, dict]:
    assert set(payloads) == METHOD_IDS
    clean: dict[str, dict] = {}
    for method_id, payload in payloads.items():
        assert payload["type"] == "method_result"
        assert payload["method_id"] == method_id
        result = sanitize(payload)
        assert result.ok, f"{method_id}: {result.rejection_reason}"
        verification = verify_payload(result.sanitized)
        assert verification is not None
        assert not any(
            row["status"] == "warn" and row["id"] in {
                "omnibus_inference", "descriptive_interval",
                "repeated_measure_structure", "multiplicity_family",
                "multiplicity_recalculation",
            }
            for row in verification["checks"]
        ), (method_id, verification)
        clean[method_id] = result.sanitized
    return clean


@pytest.fixture(scope="module")
def python_methods(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    tmp = tmp_path_factory.mktemp("python-comparison-methods")
    path = tmp / "results.jsonl"
    env = os.environ.copy()
    env["SIFT_RUN_TOKEN"] = "qualification-token"
    env["SIFT_RESULT_PATH"] = str(path)
    env["PYTHONPATH"] = str(PY_RUNTIME)
    proc = subprocess.run(
        [sys.executable, "-c", PYTHON_QUALIFICATION], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return _qualify_all(_read_labelled(path))


def test_python_reference_methods_match_synthetic_truth(
    python_methods: dict[str, dict],
) -> None:
    assert python_methods["descriptive_confidence_interval"]["estimates"]["mean"] == pytest.approx(10.5)
    assert python_methods["nonparametric_test"]["estimates"]["mann_whitney_u"] == 0
    assert python_methods["nonparametric_test"]["p_values"]["mann_whitney_u"] < 0.001
    assert python_methods["proportion_test"]["estimates"]["proportion"] == pytest.approx(0.6)
    # statsmodels' z-test uses the sample-proportion variance here:
    # z = (.6 - .5) / sqrt(.6*.4/100) = 2.04124.
    assert python_methods["proportion_test"]["p_values"]["proportion"] == pytest.approx(0.04123, abs=0.00002)
    assert min(python_methods["anova"]["p_values"].values()) < 1e-4
    assert min(python_methods["ancova"]["p_values"].values()) < 1e-10
    assert python_methods["repeated_measures_test"]["p_values"]["time"] < 1e-10
    assert python_methods["multiple_testing_correction"]["p_values"] == pytest.approx(
        {"h1": 0.04, "h2": 0.06, "h3": 0.06, "h4": 0.5}
    )


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
def test_r_reference_methods_match_synthetic_truth(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    script = R_QUALIFICATION.format(
        result_path=str(path).replace("\\", "/"),
        runtime=str(R_RUNTIME).replace("\\", "/"),
    )
    proc = subprocess.run(
        [shutil.which("Rscript"), "-e", script], cwd=ROOT,
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    clean = _qualify_all(_read_labelled(path))
    assert clean["descriptive_confidence_interval"]["estimates"]["mean"] == pytest.approx(10.5)
    assert clean["nonparametric_test"]["p_values"]["wilcoxon_w"] < 0.001
    assert clean["proportion_test"]["estimates"]["proportion"] == pytest.approx(0.6)
    assert min(clean["anova"]["p_values"].values()) < 1e-4
    assert min(clean["ancova"]["p_values"].values()) < 1e-10
    assert clean["repeated_measures_test"]["p_values"]["omnibus"] < 1e-5
    assert clean["multiple_testing_correction"]["p_values"] == pytest.approx(
        {"h1": 0.04, "h2": 0.06, "h3": 0.06, "h4": 0.5}
    )


def test_method_specific_sanitizer_guards_reject_incomplete_claims() -> None:
    base = {
        "type": "method_result", "method_id": "multiple_testing_correction",
        "n": 100,
        "diagnostics": {"hypothesis_family": "pass", "correction_applied": "pass"},
        "estimates": {"h1": 0.01}, "p_values": {"different": 0.01},
        "metrics": {"hypothesis_count": 1, "rejection_count": 1, "alpha": 0.05},
        "multiple_testing": "holm",
    }
    rejected = sanitize(base)
    assert not rejected.ok
    assert "matching raw and adjusted" in rejected.rejection_reason


def test_descriptive_ci_has_descriptive_claim_and_no_exposure_role() -> None:
    method = METHODS["descriptive_confidence_interval"]
    assert method.family == "descriptive"
    assert method.required_roles == ("outcome",)
    assert method.output_schema == "method_result_v1"
    assert "population claims" in method.claim_rule
    assert "group contrast" not in method.claim_rule


def test_python_helpers_reject_bad_controls_before_emission(tmp_path: Path) -> None:
    result_path = tmp_path / "bad-results.jsonl"
    code = r'''
import sift

class TestResult:
    statistic = 1.0
    pvalue = 0.5

class RepeatedResult:
    class Table:
        def iterrows(self): return iter(())
    anova_table = Table()

bad_calls = [
    lambda: sift.from_nonparametric_test(TestResult(), n=0),
    lambda: sift.from_proportion_test(6, 10, alternative="sideways"),
    lambda: sift.from_proportion_test(6, 10, confidence=1.0),
    lambda: sift.from_repeated_measures(RepeatedResult(), n=10, subjects=0),
    lambda: sift.from_multiple_testing([.01, .02], n=100, alpha=0),
    lambda: sift.from_multiple_testing([.01, .02], n=100, labels=["same", "same"]),
    lambda: sift.from_multiple_testing([.01], n=100, labels=["unsafe label"]),
]
for call in bad_calls:
    try:
        call()
    except ValueError:
        continue
    raise AssertionError("invalid helper input did not raise ValueError")
'''
    env = os.environ.copy()
    env["SIFT_RUN_TOKEN"] = "qualification-token"
    env["SIFT_RESULT_PATH"] = str(result_path)
    env["PYTHONPATH"] = str(PY_RUNTIME)
    proc = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert not result_path.exists()


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
def test_r_helpers_reject_nonpositive_counts_and_duplicate_labels(tmp_path: Path) -> None:
    result_path = tmp_path / "bad-r-results.jsonl"
    script = r'''
Sys.setenv(SIFT_RUN_TOKEN = "qualification-token")
Sys.setenv(SIFT_RESULT_PATH = "{result_path}")
source("{runtime}")
fake <- structure(list(statistic = c(Q = 1), p.value = 0.5), class = "htest")
bad <- list(
  function() sift$from_nonparametric_test(fake, n = 0),
  function() sift$from_repeated_measures(fake, n = 10, subjects = 0),
  function() sift$from_repeated_measures(fake, n = 10, subjects = 5, records = 11),
  function() sift$from_multiple_testing(c(.01, .02), n = 100, alpha = 1),
  function() sift$from_multiple_testing(
    c(.01, .02), n = 100, labels = c("same", "same")
  )
)
for (fn in bad) {{
  failed <- FALSE
  tryCatch(fn(), error = function(e) failed <<- TRUE)
  if (!failed) stop("invalid helper input did not fail")
}}
'''.format(
        result_path=str(result_path).replace("\\", "/"),
        runtime=str(R_RUNTIME).replace("\\", "/"),
    )
    proc = subprocess.run(
        [shutil.which("Rscript"), "-e", script], cwd=ROOT,
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert not result_path.exists()
