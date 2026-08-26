"""Executable qualification for aggregate-only causal-design helpers."""

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


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "src" / "sift" / "runtime"
METHODS = {
    "matching", "propensity_weighting", "synthetic_control",
    "treatment_effect_heterogeneity", "causal_sensitivity",
}


PYTHON_SCRIPT = r'''
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import sift

assert abs(sift._sensemakr_robustness_value(4.18445,783,q=1,alpha=1)-0.13877635435985322)<1e-12
assert abs(sift._sensemakr_robustness_value(4.18445,783,q=1,alpha=.05)-0.0762579660459714)<1e-12

def mark(label): sift._write_result({"_qualification_label": label})

rng=np.random.default_rng(20260822);n=500
X=rng.normal(size=(n,2));prob=1/(1+np.exp(-(0.5*X[:,0]-0.3*X[:,1])))
t=(rng.random(n)<prob).astype(int);y=2*t+0.7*X[:,0]-0.2*X[:,1]+rng.normal(scale=.5,size=n)
sift.from_propensity_matching(X,t,y,falsification_status="pass");mark("matching")
sift.from_propensity_weighting(X,t,y,falsification_status="pass");mark("propensity_weighting")

time=np.arange(20.);d1=.2*time+np.sin(time/3);d2=.1*time+np.cos(time/4)
d3=np.sin(time/2)-.05*time;d4=np.cos(time/5)+.03*time
donors=np.column_stack([d1,d2,d3,d4]);treated=.6*d1+.4*d2;treated[12:]+=3
sift.from_synthetic_control(treated,donors,intervention_index=12,falsification_status="pass")
mark("synthetic_control")

n2=700;X2=rng.normal(size=(n2,2));t2=(rng.random(n2)<.5).astype(int)
tau=1+2*X2[:,0];y2=.5*X2[:,0]-.2*X2[:,1]+t2*tau+rng.normal(scale=.25,size=n2)
sift.from_treatment_heterogeneity(X2,t2,y2,falsification_status="pass",seed=19)
mark("treatment_effect_heterogeneity")

df=pd.DataFrame({"y":y,"treatment":t,"x1":X[:,0],"x2":X[:,1]})
fit=smf.ols("y ~ treatment + x1 + x2",data=df).fit()
sift.from_causal_sensitivity(fit,"treatment",falsification_status="not_applicable")
mark("causal_sensitivity")
'''


R_SCRIPT = r'''
Sys.setenv(SIFT_RUN_TOKEN="qualification",SIFT_RESULT_PATH="{path}")
source("{runtime}")
stopifnot(abs(sift$.sensemakr_robustness_value(4.18445,783,q=1,alpha=1)-0.13877635435985322)<1e-12)
stopifnot(abs(sift$.sensemakr_robustness_value(4.18445,783,q=1,alpha=.05)-0.0762579660459714)<1e-12)
mark<-function(label)sift$.write_result(list(`_qualification_label`=label))
set.seed(20260822);n<-500;X<-cbind(rnorm(n),rnorm(n));prob<-plogis(.5*X[,1]-.3*X[,2])
t<-rbinom(n,1,prob);y<-2*t+.7*X[,1]-.2*X[,2]+rnorm(n,sd=.5)
sift$from_propensity_matching(X,t,y,falsification_status="pass");mark("matching")
sift$from_propensity_weighting(X,t,y,falsification_status="pass");mark("propensity_weighting")
time<-0:19;d1<-.2*time+sin(time/3);d2<-.1*time+cos(time/4)
d3<-sin(time/2)-.05*time;d4<-cos(time/5)+.03*time
donors<-cbind(d1,d2,d3,d4);treated<-.6*d1+.4*d2;treated[13:20]<-treated[13:20]+3
sift$from_synthetic_control(treated,donors,12,falsification_status="pass");mark("synthetic_control")
n2<-700;X2<-cbind(rnorm(n2),rnorm(n2));t2<-rbinom(n2,1,.5);tau<-1+2*X2[,1]
y2<-.5*X2[,1]-.2*X2[,2]+t2*tau+rnorm(n2,sd=.25)
sift$from_treatment_heterogeneity(X2,t2,y2,falsification_status="pass",seed=19);mark("treatment_effect_heterogeneity")
fit<-lm(y~t+X[,1]+X[,2]);sift$from_causal_sensitivity(fit,"t",falsification_status="not_applicable")
mark("causal_sensitivity")
'''


def _read(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}; pending = None
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); row.pop("_token", None)
        if "_qualification_label" in row:
            assert pending is not None
            out[row["_qualification_label"]] = pending; pending = None
        else:
            assert pending is None; pending = row
    assert pending is None and set(out) == METHODS
    return out


def _sanitize(payloads: dict[str, dict]) -> dict[str, dict]:
    clean = {}
    for method, payload in payloads.items():
        result = sanitize(payload)
        assert result.ok, f"{method}: {result.rejection_reason}\n{payload}"
        verification = verify_payload(result.sanitized)
        assert verification and any(c["id"] == "causal_design_contract" for c in verification["checks"])
        assert verification["causality"]["label"] in {"design_conditional_causal", "sensitivity_only"}
        clean[method] = result.sanitized
    return clean


@pytest.fixture(scope="module")
def python_results(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    tmp = tmp_path_factory.mktemp("causal-python"); path = tmp / "results.jsonl"
    env = os.environ.copy(); env["SIFT_RUN_TOKEN"] = "qualification"; env["SIFT_RESULT_PATH"] = str(path)
    env["PYTHONPATH"] = str(RUNTIME)
    proc = subprocess.run([sys.executable,"-c",PYTHON_SCRIPT],cwd=ROOT,env=env,
                          capture_output=True,text=True,timeout=180)
    assert proc.returncode == 0, proc.stderr
    return _sanitize(_read(path))


def test_python_causal_references_recover_synthetic_truth(python_results: dict[str, dict]) -> None:
    assert python_results["matching"]["estimates"]["att"] == pytest.approx(2,abs=.35)
    assert python_results["propensity_weighting"]["estimates"]["ate"] == pytest.approx(2,abs=.25)
    assert python_results["synthetic_control"]["estimates"]["unit_time_att"] == pytest.approx(3,abs=.05)
    hetero=python_results["treatment_effect_heterogeneity"]["metrics"]
    assert hetero["average_cate"] == pytest.approx(1,abs=.35)
    assert hetero["q4_q1_contrast"] > 2
    sensitivity=python_results["causal_sensitivity"]["metrics"]
    assert 0 < sensitivity["robustness_value_alpha"] <= sensitivity["robustness_value_zero"] < 1
    for method in ("matching","propensity_weighting","synthetic_control","treatment_effect_heterogeneity","causal_sensitivity"):
        assert "standard_errors" not in python_results[method]
        assert "uncertainty_type" not in python_results[method]


@pytest.mark.skipif(shutil.which("Rscript") is None,reason="Rscript not installed")
def test_r_causal_references_recover_synthetic_truth(tmp_path: Path) -> None:
    path=tmp_path/"results.jsonl"
    script=R_SCRIPT.format(path=str(path).replace("\\","/"),runtime=str(RUNTIME/"sift.R").replace("\\","/"))
    proc=subprocess.run([shutil.which("Rscript"),"-e",script],cwd=ROOT,capture_output=True,text=True,timeout=180)
    assert proc.returncode == 0,proc.stderr
    clean=_sanitize(_read(path))
    assert clean["matching"]["estimates"]["att"] == pytest.approx(2,abs=.4)
    assert clean["propensity_weighting"]["estimates"]["ate"] == pytest.approx(2,abs=.3)
    assert clean["synthetic_control"]["estimates"]["unit_time_att"] == pytest.approx(3,abs=.08)
    assert clean["treatment_effect_heterogeneity"]["metrics"]["q4_q1_contrast"] > 1
    rv=clean["causal_sensitivity"]["metrics"]
    assert 0 < rv["robustness_value_alpha"] <= rv["robustness_value_zero"] < 1


def test_causal_sanitizer_rejects_missing_design_and_diagnostic_metrics() -> None:
    raw={"type":"method_result","method_id":"matching","n":100,"treated":50,"controls":50,
         "diagnostics":{"propensity_overlap":"pass","standardized_mean_differences":"pass",
                        "effective_matched_sample":80,"effect_uncertainty":"not_applicable",
                        "design_specific_falsification":"pass"},
         "estimates":{"att":2},"metrics":{"effect":2}}
    result=sanitize(raw)
    assert not result.ok and "estimand and design" in result.rejection_reason
    raw.update(estimand="att",design="inverse_probability_weighting")
    result=sanitize(raw)
    assert not result.ok and "incompatible design" in result.rejection_reason
    raw["design"]="propensity_nearest_neighbor"
    result=sanitize(raw)
    assert not result.ok and "missing required aggregate design metrics" in result.rejection_reason


def test_sensitivity_result_never_receives_causal_identification_label(python_results: dict[str, dict]) -> None:
    block=verify_payload(python_results["causal_sensitivity"])
    assert block["causality"]["label"] == "sensitivity_only"
    assert "does not identify" in block["causality"]["caveat"]


def test_invalid_causal_uncertainty_domains_and_estimands_are_rejected(
    python_results: dict[str, dict],
) -> None:
    matching=dict(python_results["matching"]);matching["standard_errors"]={"att":.1}
    result=sanitize(matching)
    assert not result.ok and "does not accept analytic uncertainty" in result.rejection_reason

    heterogeneity=dict(python_results["treatment_effect_heterogeneity"])
    heterogeneity["estimand"]="ate"
    result=sanitize(heterogeneity)
    assert not result.ok and "average predicted CATE" in result.rejection_reason

    sensitivity=dict(python_results["causal_sensitivity"])
    sensitivity["uncertainty_type"]="robust"
    result=sanitize(sensitivity)
    assert not result.ok and "not robust sampling uncertainty" in result.rejection_reason

    weighting=dict(python_results["propensity_weighting"])
    weighting["metrics"]=dict(weighting["metrics"],overlap_fraction=1.5)
    result=sanitize(weighting)
    assert not result.ok and "outside valid domains" in result.rejection_reason
