"""Executable reference qualification for Stage 10 reliability analysis."""

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
RSCRIPT = shutil.which("Rscript")


PYTHON_SCRIPT = r"""
import sys
sys.path.insert(0, __RUNTIME__)
import sift
import numpy as np

rng = np.random.default_rng(20260822)
n = 500
latent = rng.normal(size=n)
items = np.column_stack([
    0.90 * latent + rng.normal(scale=0.42, size=n),
    0.82 * latent + rng.normal(scale=0.50, size=n),
    0.76 * latent + rng.normal(scale=0.55, size=n),
    -(0.86 * latent + rng.normal(scale=0.45, size=n)),
])
try:
    sift.from_reliability(items, bootstrap_replicates=200, seed=71)
except ValueError:
    print("direction-rejected")
else:
    raise AssertionError("negative-direction item was accepted")
sift.from_reliability(
    items, reverse_items=[3], bootstrap_replicates=200, seed=71,
    analysis_id="python_reliability",
)
"""


def _payload(path: Path) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = json.loads(line)
        raw.pop("_token", None)
        result = sanitize(raw)
        assert result.ok, result.rejection_reason
        rows.append(result.sanitized)
    assert len(rows) == 1
    return rows[0]


@pytest.fixture(scope="module")
def python_reliability(tmp_path_factory) -> tuple[dict, str]:
    directory = tmp_path_factory.mktemp("measurement-reliability")
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
    return _payload(output), process.stdout


def test_python_reliability_known_structure_and_direction_guard(python_reliability) -> None:
    payload, stdout = python_reliability
    assert payload["method_id"] == "reliability"
    assert payload["estimates"]["alpha"] > 0.85
    assert payload["estimates"]["omega_total"] > 0.85
    assert payload["metrics"]["reversed_item_count"] == 1
    assert payload["metrics"]["min_item_rest_correlation"] > 0.6
    assert payload["ci_lower"]["alpha"] <= payload["estimates"]["alpha"]
    assert payload["estimates"]["alpha"] <= payload["ci_upper"]["alpha"]
    assert "direction-rejected" in stdout


def test_reliability_independent_verifier_and_privacy_shape(python_reliability) -> None:
    payload, _stdout = python_reliability
    checks = verify_payload(payload)["checks"]
    assert {"reliability_interval", "reliability_direction_stability"} <= {
        check["id"] for check in checks if check["status"] == "pass"
    }
    serialized = json.dumps(payload)
    assert "loadings" not in serialized
    assert "scores" not in serialized
    assert "rows" not in serialized


def test_reliability_sanitizer_rejects_forged_contract(python_reliability) -> None:
    payload, _stdout = python_reliability
    bad_direction = json.loads(json.dumps(payload))
    bad_direction["metrics"]["min_item_rest_correlation"] = -0.1
    assert not sanitize(bad_direction).ok
    bad_bootstrap = json.loads(json.dumps(payload))
    bad_bootstrap["metrics"]["bootstrap_success_count"] = 10
    assert not sanitize(bad_bootstrap).ok
    bad_interval = json.loads(json.dumps(payload))
    bad_interval["ci_lower"]["alpha"] = bad_interval["estimates"]["alpha"] + 0.01
    assert not sanitize(bad_interval).ok


@pytest.mark.skipif(RSCRIPT is None, reason="Rscript unavailable")
def test_r_psych_reliability_reference(tmp_path: Path) -> None:
    probe = subprocess.run(
        [RSCRIPT, "-e", "quit(status=!requireNamespace('psych',quietly=TRUE))"],
        capture_output=True, text=True, timeout=20,
    )
    if probe.returncode != 0:
        pytest.skip("R psych package unavailable")
    output = tmp_path / "results.jsonl"
    script = tmp_path / "fit.R"
    script.write_text(f"""
Sys.setenv(SIFT_RUN_TOKEN='qualification-token', SIFT_RESULT_PATH={str(output)!r})
source({str(RUNTIME / 'sift.R')!r})
set.seed(20260822); n <- 420; latent <- rnorm(n)
items <- cbind(.9*latent+rnorm(n,sd=.42), .82*latent+rnorm(n,sd=.5),
               .76*latent+rnorm(n,sd=.55), -(.86*latent+rnorm(n,sd=.45)))
bad <- try(sift$from_reliability(items, bootstrap_replicates=200L, seed=71L), silent=TRUE)
if (!inherits(bad, 'try-error')) stop('negative item direction accepted')
sift$from_reliability(items, reverse_items=4L, bootstrap_replicates=200L,
                      seed=71L, analysis_id='r_reliability')
""", encoding="utf-8")
    process = subprocess.run(
        [RSCRIPT, str(script)], cwd=ROOT,
        capture_output=True, text=True, timeout=180,
    )
    assert process.returncode == 0, process.stderr
    payload = _payload(output)
    assert payload["estimates"]["alpha"] > 0.85
    assert payload["estimates"]["omega_total"] > 0.85

