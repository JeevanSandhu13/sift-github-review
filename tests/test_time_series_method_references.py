"""Executable Python/R qualification for ordered time-series methods."""

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


ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/"src"/"sift"/"runtime"
METHODS={"stationarity_diagnostic","seasonal_decomposition","arima",
         "exponential_smoothing","interrupted_time_series","forecast_backtest"}
R_CORE_METHODS=METHODS-{"stationarity_diagnostic"}

PY_SCRIPT=r'''
import numpy as np
import pandas as pd
import sift
def mark(x):sift._write_result({"_qualification_label":x})
rng=np.random.default_rng(20260822)
n=84;stationary=np.zeros(n)
for i in range(1,n):stationary[i]=.55*stationary[i-1]+rng.normal(scale=.4)
time=np.arange(n);seasonal=10+.04*time+2*np.sin(2*np.pi*time/12)+rng.normal(scale=.15,size=n)
irregular=time.copy();irregular[20:]+=1
for bad_index in (np.r_[time[:20],time[19],time[21:]], irregular):
    try:sift.from_stationarity_diagnostic(stationary,frequency=1,time_index=bad_index,cadence=1)
    except ValueError:pass
    else:raise AssertionError("invalid index accepted")
try:sift.from_stationarity_diagnostic(stationary,frequency=1,time_index=time,cadence=1,ordered=True)
except ValueError:pass
else:raise AssertionError("forged temporal flag accepted")
try:sift.from_arima(stationary,order=(1.5,0,0),holdout=12,frequency=1,time_index=time,cadence=1)
except ValueError:pass
else:raise AssertionError("fractional ARIMA order accepted")
modern=np.datetime64("2026-08-22T12:00:00","ns")+np.arange(n)*np.timedelta64(100,"ns")
sift._time_series_values(stationary,time_index=modern,cadence=np.timedelta64(100,"ns"),frequency=1)
modern_bad=modern.copy();modern_bad[20]+=np.timedelta64(1,"ns")
try:sift._time_series_values(stationary,time_index=modern_bad,cadence=np.timedelta64(100,"ns"),frequency=1)
except ValueError:pass
else:raise AssertionError("sub-microsecond datetime irregularity accepted")
monthly=pd.date_range("2020-01-01",periods=n,freq="MS",tz="UTC")
sift._time_series_values(stationary,time_index=monthly,cadence="MS",frequency=12)
sift.from_stationarity_diagnostic(stationary,frequency=1,time_index=time,cadence=1);mark("stationarity_diagnostic")
sift.from_seasonal_decomposition(seasonal,frequency=12,time_index=time,cadence=1);mark("seasonal_decomposition")
sift.from_arima(stationary,order=(1,0,0),holdout=12,frequency=1,time_index=time,cadence=1);mark("arima")
sift.from_exponential_smoothing(seasonal,holdout=12,frequency=12,time_index=time,cadence=1,
                                trend="add",seasonal="add");mark("exponential_smoothing")
cut=50;noise=np.zeros(n)
for i in range(1,n):noise[i]=.35*noise[i-1]+rng.normal(scale=.15)
its=5+.03*time+(time>=cut)*4+np.maximum(0,time-cut)*.18+noise
sift.from_interrupted_time_series(its,intervention_index=cut,frequency=1,time_index=time,
                                  cadence=1,falsification_status="pass");mark("interrupted_time_series")
sift.from_forecast_backtest(stationary,order=(1,0,0),initial=72,frequency=1,
                            time_index=time,cadence=1);mark("forecast_backtest")
'''

R_SCRIPT=r'''
Sys.setenv(SIFT_RUN_TOKEN="qualification",SIFT_RESULT_PATH="{path}");source("{runtime}")
mark<-function(x)sift$.write_result(list(`_qualification_label`=x))
set.seed(20260822);n<-84;stationary<-numeric(n)
for(i in 2:n)stationary[i]<-.55*stationary[i-1]+rnorm(1,sd=.4)
time<-0:(n-1);seasonal<-10+.04*time+2*sin(2*pi*time/12)+rnorm(n,sd=.15)
bad<-time;bad[21]<-bad[20]
stopifnot(inherits(try(sift$from_stationarity_diagnostic(stationary,1,bad,1),silent=TRUE),"try-error"))
stopifnot(inherits(try(sift$from_stationarity_diagnostic(stationary,1,time,1,ordered=TRUE),silent=TRUE),"try-error"))
stopifnot(inherits(try(sift$from_arima(stationary,c(1.5,0,0),12,1,time,1),silent=TRUE),"try-error"))
if(requireNamespace("urca",quietly=TRUE)) {{
  sift$from_stationarity_diagnostic(stationary,1,time,1);mark("stationarity_diagnostic")
}}
sift$from_seasonal_decomposition(seasonal,12,time,1);mark("seasonal_decomposition")
sift$from_arima(stationary,c(1,0,0),12,1,time,1);mark("arima")
sift$from_exponential_smoothing(seasonal,12,12,time,1,seasonal="additive");mark("exponential_smoothing")
cut<-50;noise<-numeric(n);for(i in 2:n)noise[i]<-.35*noise[i-1]+rnorm(1,sd=.15)
its<-5+.03*time+(time>=cut)*4+pmax(0,time-cut)*.18+noise
sift$from_interrupted_time_series(its,cut,1,time,1,falsification_status="pass");mark("interrupted_time_series")
sift$from_forecast_backtest(stationary,c(1,0,0),72,1,time,1);mark("forecast_backtest")
'''

def _read(path:Path,expected=METHODS)->dict[str,dict]:
    out={};pending=None
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line);row.pop("_token",None)
        if "_qualification_label" in row:
            assert pending is not None;out[row["_qualification_label"]]=pending;pending=None
        else:assert pending is None;pending=row
    assert pending is None and set(out)==expected
    return out

def _clean(payloads):
    out={}
    for method,payload in payloads.items():
        result=sanitize(payload);assert result.ok,f"{method}: {result.rejection_reason}\n{payload}"
        block=verify_payload(result.sanitized);assert block
        out[method]=result.sanitized
    return out

@pytest.fixture(scope="module")
def python_results(tmp_path_factory):
    tmp=tmp_path_factory.mktemp("time-series-python");path=tmp/"results.jsonl"
    env=os.environ.copy();env["SIFT_RUN_TOKEN"]="qualification";env["SIFT_RESULT_PATH"]=str(path)
    env["PYTHONPATH"]=str(RUNTIME)
    proc=subprocess.run([sys.executable,"-c",PY_SCRIPT],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stderr
    return _clean(_read(path))

def test_python_time_series_known_answers(python_results):
    assert python_results["stationarity_diagnostic"]["metrics"]["adf_p_value"]<.05
    decomposition=python_results["seasonal_decomposition"]["metrics"]
    assert decomposition["seasonal_strength"]>.9 and decomposition["trend_strength"]>.5
    for method in ("arima","exponential_smoothing","forecast_backtest"):
        metrics=python_results[method]["metrics"]
        assert metrics["rmse"]>=0 and 0<=metrics["prediction_interval_coverage"]<=1
        assert metrics["prediction_interval_mean_width"]>0
    assert python_results["arima"]["interval_method"]=="model_based_gaussian"
    smoothing=python_results["exponential_smoothing"]
    assert smoothing["interval_method"]=="ets_state_space_exact"
    assert smoothing["metrics"]["last_interval_width"] >= smoothing["metrics"]["first_interval_width"] - 1e-8
    for payload in python_results.values():
        assert payload["metrics"]["cadence_min_ratio"]==pytest.approx(1)
        assert payload["metrics"]["cadence_max_ratio"]==pytest.approx(1)
        assert payload["metrics"]["time_span_steps"]==pytest.approx(payload["n"]-1)
    assert {"ar_stationarity","ma_invertibility"} <= python_results["arima"]["diagnostics"].keys()
    its=python_results["interrupted_time_series"]["estimates"]
    assert its["level_change"]==pytest.approx(4,abs=.6)
    assert its["slope_change"]==pytest.approx(.18,abs=.06)

@pytest.mark.skipif(shutil.which("Rscript") is None,reason="Rscript unavailable")
def test_r_time_series_known_answers(tmp_path):
    path=tmp_path/"results.jsonl";script=R_SCRIPT.format(path=str(path).replace("\\","/"),
        runtime=str(RUNTIME/"sift.R").replace("\\","/"))
    proc=subprocess.run([shutil.which("Rscript"),"-e",script],cwd=ROOT,capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stderr
    labels=R_CORE_METHODS|({"stationarity_diagnostic"} if _r_has_urca() else set())
    clean=_clean(_read(path,labels))
    if "stationarity_diagnostic" in clean:
        assert clean["stationarity_diagnostic"]["metrics"]["adf_statistic"]<clean["stationarity_diagnostic"]["metrics"]["adf_critical_05"]
    assert clean["seasonal_decomposition"]["metrics"]["seasonal_strength"]>.9
    its=clean["interrupted_time_series"]["estimates"]
    assert its["level_change"]==pytest.approx(4,abs=.7)
    assert its["slope_change"]==pytest.approx(.18,abs=.07)
    assert clean["forecast_backtest"]["evaluation_split"]=="rolling_origin"
    smoothing=clean["exponential_smoothing"]
    assert smoothing["interval_method"]=="holtwinters_state_space"
    assert smoothing["metrics"]["last_interval_width"]>smoothing["metrics"]["first_interval_width"]

def _r_has_urca():
    proc=subprocess.run([shutil.which("Rscript"),"-e","quit(status=!requireNamespace('urca',quietly=TRUE))"],
                        capture_output=True,text=True)
    return proc.returncode==0

def test_temporal_leakage_and_interval_contracts_reject_forged_results(python_results):
    backtest=dict(python_results["forecast_backtest"]);backtest["evaluation_split"]="cross_validation"
    result=sanitize(backtest);assert not result.ok and "rolling-origin" in result.rejection_reason
    arima=dict(python_results["arima"]);arima["diagnostics"]=dict(arima["diagnostics"],holdout_leakage="fail")
    result=sanitize(arima);assert not result.ok and "leakage-free" in result.rejection_reason
    smoothing=dict(python_results["exponential_smoothing"])
    smoothing["metrics"]=dict(smoothing["metrics"],prediction_interval_coverage=1.5)
    result=sanitize(smoothing);assert not result.ok and "outside valid domains" in result.rejection_reason
    no_method=dict(python_results["arima"]);no_method.pop("interval_method")
    result=sanitize(no_method);assert not result.ok and "interval method" in result.rejection_reason
    forged=dict(python_results["arima"]);forged["metrics"]=dict(forged["metrics"],cadence_max_ratio=1.2)
    result=sanitize(forged);assert not result.ok and "cadence proof" in result.rejection_reason
    merged_roots=dict(python_results["arima"]);merged_roots["diagnostics"]=dict(merged_roots["diagnostics"])
    merged_roots["diagnostics"].pop("ma_invertibility")
    result=sanitize(merged_roots);assert not result.ok
    shrinking=dict(python_results["exponential_smoothing"])
    shrinking["metrics"]=dict(shrinking["metrics"],last_interval_width=0.5*shrinking["metrics"]["first_interval_width"])
    result=sanitize(shrinking);assert not result.ok and "non-shrinking" in result.rejection_reason
    no_pretrend=dict(python_results["interrupted_time_series"]);no_pretrend["metrics"]=dict(no_pretrend["metrics"])
    no_pretrend["metrics"].pop("pretrend_stability_p_value")
    result=sanitize(no_pretrend);assert not result.ok and "pretrend" in result.rejection_reason

def test_interrupted_series_stays_associational_without_identification(python_results):
    block=verify_payload(python_results["interrupted_time_series"])
    assert block["causality"]["label"]=="associational"
    assert any(c["id"]=="interrupted_series_identification" and c["status"]=="warn"
               for c in block["checks"])
