from __future__ import annotations

import asyncio
import ast
import json
import re
from pathlib import Path

from sift import method_runtime
from sift.methodology import METHODS
from sift.method_runtime import runtime_guidance


ROOT = Path(__file__).resolve().parents[1]


def _spec() -> dict[str, object]:
    return {
        "research_question": "How accurately can y be predicted?",
        "unit_of_analysis": "record", "target_population": "study population",
        "estimand": "held-out prediction error", "study_design": "observational",
        "goal": "predictive", "missing_data_assumption": "MAR",
        "exposures": "none", "predictors": ["x"], "controls": "none",
        "repeated_measures": False, "clusters": "none", "weights": "none",
        "strata": "none", "psu": "none", "fpc": "none",
        "replicate_weights": "none", "time_ordering": "not applicable",
        "outcome": "y", "target": "y", "split_strategy": "60/20/20",
    }


def test_runtime_guidance_prefers_typed_predictive_workflow() -> None:
    guidance = runtime_guidance("predictive_classification")
    assert guidance["typed_helper_available"] is True
    assert guidance["preferred_helpers"]["Python"] == "sift.from_predictive_workflow"
    assert "do not hand-assemble" in guidance["instruction"]


def test_validate_methodology_returns_runtime_guidance() -> None:
    from sift.tools import validate_methodology

    response = asyncio.run(validate_methodology.handler({
        "method_id": "predictive_regression", "research_specification": _spec(),
    }))
    body = json.loads(response["content"][0]["text"])
    assert body["status"] == "ok"
    assert body["runtime_guidance"]["preferred_helpers"]["Python"] == (
        "sift.from_predictive_workflow"
    )


def test_unknown_runtime_guidance_fails_safe() -> None:
    guidance = runtime_guidance("not_a_method")
    assert guidance["typed_helper_available"] is False
    assert guidance["preferred_helpers"] == {}


def test_every_advertised_python_helper_exists() -> None:
    tree = ast.parse(
        (ROOT / "src/sift/runtime/sift.py").read_text(encoding="utf-8")
    )
    functions = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = [
        (method_id, helper)
        for method_id, helper in method_runtime._PYTHON.items()
        if helper.removeprefix("sift.") not in functions
    ]
    assert missing == []


def test_every_advertised_r_helper_exists() -> None:
    source = (ROOT / "src/sift/runtime/sift.R").read_text(encoding="utf-8")
    missing = [
        (method_id, helper)
        for method_id, helper in method_runtime._R.items()
        if re.search(
            rf"sift\${re.escape(helper.removeprefix('sift$'))}\s*<-\s*function",
            source,
        ) is None
    ]
    assert missing == []


def test_runtime_guidance_only_advertises_registry_methods() -> None:
    assert set(method_runtime._PYTHON) <= set(METHODS)
    assert set(method_runtime._R) <= set(METHODS)


def test_every_registry_method_has_an_exact_typed_execution_path() -> None:
    assert set(METHODS) == set(method_runtime._PYTHON) | set(method_runtime._R)


def test_shared_helpers_pin_method_selecting_arguments() -> None:
    assert runtime_guidance("ancova")["required_arguments"] == {
        "Python": {"method_id": "ancova"},
        "R": {"method_id": "ancova"},
    }
    assert runtime_guidance("survey_proportion")["required_arguments"] == {
        "Python": {"proportion": True},
    }
    assert runtime_guidance("predictive_classification")["required_arguments"] == {
        "Python": {"task": "classification"},
    }


def test_exact_panel_did_and_calibration_helpers_are_advertised() -> None:
    assert runtime_guidance("panel_fixed_effects")["preferred_helpers"] == {
        "Python": "sift.from_panel_fixed_effects",
    }
    assert runtime_guidance("difference_in_differences")["preferred_helpers"] == {
        "Python": "sift.from_difference_in_differences",
    }
    assert runtime_guidance("probability_calibration")["preferred_helpers"] == {
        "Python": "sift.from_probability_calibration",
    }
    assert runtime_guidance("generalized_mixed_effects")["preferred_helpers"] == {
        "R": "sift$from_lm",
    }
