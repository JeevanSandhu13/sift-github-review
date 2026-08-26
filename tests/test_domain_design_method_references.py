"""Executable qualification for privacy-safe domain and prospective-design methods."""

from __future__ import annotations

import json
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sift.sanitizer import sanitize
from sift.verification import verify_payload


ROOT=Path(__file__).resolve().parents[1];RUNTIME=ROOT/"src"/"sift"/"runtime"
PY_METHODS={"geospatial_analysis","network_analysis","text_analysis","power_precision","simulation_design"}
R_METHODS={"power_precision","power_precision_one_sided","simulation_design"}


def _python_package_available(package: str) -> bool:
    """Check the interpreter running qualification, not a stale temp folder."""
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, AttributeError, ValueError):
        return False

PY_SCRIPT=r'''
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import sift
def mark(x):sift._write_result({"_qualification_label":x})

# Projected US-survey-foot CRS: the emitted metre threshold must use CRS axis conversion.
coords=[(1000*(i%5),1000*(i//5)) for i in range(25)];coords[-1]=(50000,50000)
frame=gpd.GeoDataFrame({"signal":[i%5+i//5 for i in range(25)]},
                       geometry=[Point(x,y) for x,y in coords],crs="EPSG:2263")
try:sift.from_geospatial_moran(frame.to_crs(4326),value="signal",distance_threshold=1100,permutations=199)
except ValueError:pass
else:raise AssertionError("geographic angular distances accepted")
sift.from_geospatial_moran(frame,value="signal",distance_threshold=1100,permutations=199,seed=7);mark("geospatial_analysis")

nodes=[f"confidential_person_{i}" for i in range(20)]
edges=[(nodes[i],nodes[(i+1)%18]) for i in range(18)]+[(nodes[0],nodes[9])]
try:sift.from_network_graph(nodes,edges+[(nodes[0],nodes[1])])
except ValueError:pass
else:raise AssertionError("duplicate edge accepted")
sift.from_network_graph(nodes,edges);mark("network_analysis")

docs=[]
for i in range(25):docs.append("privatealpha astronomy telescope orbit galaxy research " + ("nova "*(1+i%3)))
for i in range(25):docs.append("privatebeta biology protein genome cell laboratory research " + ("enzyme "*(1+i%3)))
try:sift.from_text_stability(docs[:10],clusters=2)
except ValueError:pass
else:raise AssertionError("tiny corpus accepted")
sift.from_text_stability(docs,clusters=2,seed=11,max_features=500);mark("text_analysis")

try:sift.from_power_precision([.5,.3],alpha=.05,target_power=.8)
except ValueError:pass
else:raise AssertionError("unordered scenarios accepted")
try:sift.from_power_precision([.5],alternative="smaller")
except ValueError:pass
else:raise AssertionError("positive magnitude accepted for smaller alternative")
sift.from_power_precision([.3,.5,.8],alpha=.05,target_power=.8);mark("power_precision")
try:sift.from_simulation_design(effect_size=.5,group_n=60,replications=100)
except ValueError:pass
else:raise AssertionError("under-replicated simulation accepted")
try:sift.from_simulation_design(effect_size=10,group_n=60,replications=2000)
except ValueError:pass
else:raise AssertionError("numerically extreme simulation effect accepted")
sift.from_simulation_design(effect_size=.5,group_n=60,replications=2000,seed=19);mark("simulation_design")
'''

R_SCRIPT=r'''
Sys.setenv(SIFT_RUN_TOKEN="qualification",SIFT_RESULT_PATH="{path}");source("{runtime}")
mark<-function(x)sift$.write_result(list(`_qualification_label`=x))
stopifnot(inherits(try(sift$from_power_precision(c(.5,.3)),silent=TRUE),"try-error"))
sift$from_power_precision(c(.3,.5,.8),alpha=.05,target_power=.8);mark("power_precision")
sift$from_power_precision(.5,alpha=.05,target_power=.8,allocation_ratio=1.5,
                          alternative="one.sided");mark("power_precision_one_sided")
stopifnot(inherits(try(sift$from_simulation_design(.5,60,100),silent=TRUE),"try-error"))
stopifnot(inherits(try(sift$from_simulation_design(10,60,2000),silent=TRUE),"try-error"))
sift$from_simulation_design(.5,60,2000,.05,19);mark("simulation_design")
'''

def _read(path:Path, expected:set[str])->dict[str,dict]:
    out={};pending=None
    for line in path.read_text(encoding="utf-8").splitlines():
        row=json.loads(line);row.pop("_token",None)
        if "_qualification_label" in row:
            assert pending is not None;out[row["_qualification_label"]]=pending;pending=None
        else:assert pending is None;pending=row
    assert pending is None and set(out)==expected
    return out

def _clean(payloads):
    clean={}
    for method,payload in payloads.items():
        result=sanitize(payload);assert result.ok,f"{method}: {result.rejection_reason}\n{payload}"
        assert verify_payload(result.sanitized)
        clean[method]=result.sanitized
    return clean

@pytest.fixture(scope="module")
def python_results(tmp_path_factory):
    path=tmp_path_factory.mktemp("domain-design-python")/"results.jsonl"
    env=os.environ.copy();env["SIFT_RUN_TOKEN"]="qualification";env["SIFT_RESULT_PATH"]=str(path)
    env["PYTHONPATH"]=str(RUNTIME);env["PYTHONDONTWRITEBYTECODE"]="1"
    proc=subprocess.run([sys.executable,"-c",PY_SCRIPT],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stderr
    return _clean(_read(path,PY_METHODS))

def test_python_domain_known_answers_and_privacy(python_results):
    geo=python_results["geospatial_analysis"]
    assert geo["crs_epsg"]==2263
    assert geo["metrics"]["distance_threshold_metres"]==pytest.approx(335.28,rel=.01)
    assert geo["metrics"]["island_fraction"]==pytest.approx(1/25,abs=.01)
    graph=python_results["network_analysis"]["metrics"]
    assert graph["node_count"]==20 and graph["isolate_fraction"]==pytest.approx(.1)
    text=python_results["text_analysis"]
    assert text["metrics"]["resampling_stability_ari"]>.9
    serialized=json.dumps(text)+json.dumps(python_results["network_analysis"])
    assert "privatealpha" not in serialized and "confidential_person" not in serialized

def test_python_design_known_answers(python_results):
    power=python_results["power_precision"]
    assert power["test_alternative"]=="two_sided"
    assert power["estimates"]["scenario_2"]==pytest.approx(128,abs=4)
    sim=python_results["simulation_design"]
    assert sim["interval_method"]=="clopper_pearson_binomial"
    assert sim["metrics"]["monte_carlo_standard_error"]<.02
    assert sim["metrics"]["absolute_analytic_difference"]<.04

@pytest.mark.skipif(shutil.which("Rscript") is None,reason="Rscript unavailable")
def test_r_design_known_answers(tmp_path):
    path=tmp_path/"results.jsonl";script=R_SCRIPT.format(path=str(path).replace("\\","/"),
        runtime=str(RUNTIME/"sift.R").replace("\\","/"))
    proc=subprocess.run([shutil.which("Rscript"),"-e",script],cwd=ROOT,capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stderr
    clean=_clean(_read(path,R_METHODS))
    assert clean["power_precision"]["estimates"]["scenario_2"]==pytest.approx(128,abs=4)
    assert clean["power_precision"]["test_alternative"]=="two_sided"
    one_sided=clean["power_precision_one_sided"]
    assert one_sided["test_alternative"]=="larger"
    assert one_sided["metrics"]["group1_n#scenario_1"]==42
    assert one_sided["metrics"]["group2_n#scenario_1"]==63
    assert clean["simulation_design"]["metrics"]["absolute_analytic_difference"]<.04
    assert clean["simulation_design"]["interval_method"]=="clopper_pearson_binomial"

def test_domain_design_forgery_rejected(python_results):
    geo=dict(python_results["geospatial_analysis"]);geo["metrics"]=dict(geo["metrics"],crs_linear_unit_to_metre=1)
    assert not sanitize(geo).ok
    graph=dict(python_results["network_analysis"]);graph["estimates"]={"private_node":1}
    assert not sanitize(graph).ok
    text=dict(python_results["text_analysis"]);text["diagnostics"]=dict(text["diagnostics"],held_out_or_stability_check=.123)
    assert not sanitize(text).ok
    power=dict(python_results["power_precision"]);power["metrics"]=dict(power["metrics"],**{"group1_n#scenario_2":1})
    assert not sanitize(power).ok
    sim=dict(python_results["simulation_design"]);sim["seed"]=20
    # Seed is provenance-bound by the typed helper/storage path; arithmetic forging is separately rejected.
    sim["metrics"]=dict(sim["metrics"],rejection_count=0)
    assert not sanitize(sim).ok

def test_bayesian_caller_assertions_are_insufficient():
    payload={"type":"method_result","method_id":"bayesian_model","n":1000,
      "diagnostics":{"rhat":"pass","bulk_ess":"pass","tail_ess":"pass","divergences":"pass",
                     "posterior_predictive_check":"pass"},
      "estimates":{"theta":0},"ci_lower":{"theta":-1},"ci_upper":{"theta":1},
      "uncertainty_type":"posterior"}
    result=sanitize(payload)
    assert not result.ok and "validated R-hat" in result.rejection_reason
    payload["diagnostics"]={"rhat":1.0,"bulk_ess":800,"tail_ess":700,"divergences":-1,
                            "posterior_predictive_check":.5}
    payload["metrics"]={"chains":4,"draws_per_chain":200,"parameter_count":1,
                        "posterior_predictive_replicates":800}
    assert not sanitize(payload).ok

def test_arviz_helper_fails_closed_on_missing_or_misaligned_draw_diagnostics(monkeypatch,tmp_path):
    import importlib.util
    import types
    import numpy as np
    import pandas as pd
    summary=pd.DataFrame({"mean":[0.],"hdi_2.5%":[-1.],"hdi_97.5%":[1.],
                          "ess_bulk":[700.],"ess_tail":[650.],"r_hat":[1.]},index=["theta"])
    monkeypatch.setitem(sys.modules,"arviz",types.SimpleNamespace(summary=lambda *a,**k:summary))
    monkeypatch.setenv("SIFT_RUN_TOKEN","qualification")
    monkeypatch.setenv("SIFT_RESULT_PATH",str(tmp_path/"results.jsonl"))
    spec=importlib.util.spec_from_file_location("sift_domain_runtime",RUNTIME/"sift.py")
    runtime=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(runtime)
    class Group(dict):
        sizes={"chain":4,"draw":200}
    observed=Group(y=np.zeros(10));posterior=Group(theta=np.zeros((4,200)))
    predictive=Group(y=np.zeros((4,200,10)))
    missing=types.SimpleNamespace(posterior=posterior,observed_data=observed,
                                  posterior_predictive=predictive,sample_stats=Group())
    with pytest.raises(ValueError,match="divergence"):
        runtime.from_arviz_posterior(missing,observed_variable="y")
    mismatched=types.SimpleNamespace(posterior=posterior,observed_data=observed,
        posterior_predictive=Group(y=np.zeros((4,200,9))),
        sample_stats=Group(diverging=np.zeros((4,200),dtype=bool)))
    with pytest.raises(ValueError,match="aligned"):
        runtime.from_arviz_posterior(mismatched,observed_variable="y")

@pytest.mark.skipif(
    not _python_package_available("arviz_stats"),
    reason="qualified ArviZ runtime unavailable",
)
def test_real_arviz_adapter_known_answers(tmp_path):
    path=tmp_path/"results.jsonl"
    script=r'''
import numpy as np
from arviz_base import from_dict
import sift
rng=np.random.default_rng(1242026);chains=4;draws=500
theta=rng.normal(2.0,.5,size=(chains,draws));scaled=2*theta+1
observed=np.linspace(1,3,40)
replicated=rng.normal(loc=observed,scale=.4,size=(chains,draws,len(observed)))
idata=from_dict({
 "posterior":{"theta":theta,"scaled_theta":scaled},
 "sample_stats":{"diverging":np.zeros((chains,draws),dtype=bool)},
 "observed_data":{"y":observed},"posterior_predictive":{"y":replicated}},
 sample_dims=["chain","draw"],dims={"y":["observation"]},pred_dims={"y":["observation"]})
sift.from_arviz_posterior(idata,observed_variable="y")
'''
    env=os.environ.copy();env["SIFT_RUN_TOKEN"]="qualification";env["SIFT_RESULT_PATH"]=str(path)
    env["PYTHONDONTWRITEBYTECODE"]="1";env["PYTHONPATH"]=str(RUNTIME)
    proc=subprocess.run([sys.executable,"-c",script],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
    assert proc.returncode==0,proc.stderr
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()];assert len(rows)==1
    rows[0].pop("_token",None);result=sanitize(rows[0]);assert result.ok,result.rejection_reason
    payload=result.sanitized;block=verify_payload(payload);assert block
    assert payload["diagnostics"]["rhat"]<1.01
    assert payload["diagnostics"]["bulk_ess"]>=400 and payload["diagnostics"]["tail_ess"]>=400
    assert payload["diagnostics"]["divergences"]==0
    assert .2<payload["diagnostics"]["posterior_predictive_check"]<.8
    assert payload["estimates"]["parameter_1"]==pytest.approx(2,abs=.05)
    assert payload["estimates"]["parameter_2"]==pytest.approx(2*payload["estimates"]["parameter_1"]+1,abs=.02)
    assert payload["ci_lower"]["parameter_2"]==pytest.approx(2*payload["ci_lower"]["parameter_1"]+1,abs=.03)
    assert payload["ci_upper"]["parameter_2"]==pytest.approx(2*payload["ci_upper"]["parameter_1"]+1,abs=.03)
    assert any(check["id"]=="bayesian_computation" and check["status"]=="pass" for check in block["checks"])

@pytest.mark.skipif(
    not _python_package_available("pymc"),
    reason="qualified PyMC runtime unavailable",
)
def test_real_pymc_fit_qualifies_bayesian_method(tmp_path):
    path=tmp_path/"results.jsonl"
    script=r'''
import numpy as np
import pymc as pm
import sift
rng=np.random.default_rng(8124);observed=rng.normal(2.0,.5,size=80)
with pm.Model():
    mu=pm.Normal("mu",0,5);sigma=pm.HalfNormal("sigma",1)
    pm.Normal("y",mu,sigma,observed=observed)
    idata=pm.sample(draws=500,tune=500,chains=4,cores=1,random_seed=[11,12,13,14],
                    progressbar=False,compute_convergence_checks=False)
    pm.sample_posterior_predictive(idata,extend_inferencedata=True,random_seed=15,progressbar=False)
sift.from_arviz_posterior(idata,observed_variable="y")
'''
    env=os.environ.copy();env["SIFT_RUN_TOKEN"]="qualification";env["SIFT_RESULT_PATH"]=str(path)
    env["PYTHONDONTWRITEBYTECODE"]="1";env["PYTHONPATH"]=str(RUNTIME)
    env["MPLCONFIGDIR"]=str(tmp_path/"mpl");env["XDG_CACHE_HOME"]=str(tmp_path/"cache")
    compiled_dir = os.environ.get("SIFT_BAYESIAN_COMPILEDIR") or str(tmp_path / "pytensor")
    # The qualification must not depend on the optional Numba/LLVM JIT.  The
    # pure-Python linker is slower but deterministic and keeps the real PyMC
    # fit executable in minimal/offline release-verification environments.
    env["PYTENSOR_FLAGS"] = (
        f"base_compiledir={compiled_dir},linker=py,optimizer=fast_compile"
    )
    proc=subprocess.run([sys.executable,"-c",script],cwd=ROOT,env=env,capture_output=True,text=True,timeout=300)
    assert proc.returncode==0,proc.stderr
    rows=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()];assert len(rows)==1
    rows[0].pop("_token",None);result=sanitize(rows[0]);assert result.ok,result.rejection_reason
    payload=result.sanitized;block=verify_payload(payload);assert block
    assert payload["diagnostics"]["rhat"]<1.01
    assert payload["diagnostics"]["bulk_ess"]>=400 and payload["diagnostics"]["tail_ess"]>=400
    assert payload["diagnostics"]["divergences"]==0
    posterior_means=sorted(payload["estimates"].values())
    assert posterior_means[0]==pytest.approx(.5,abs=.1)
    assert posterior_means[1]==pytest.approx(2,abs=.12)
    assert any(check["id"]=="bayesian_computation" and check["status"]=="pass" for check in block["checks"])
