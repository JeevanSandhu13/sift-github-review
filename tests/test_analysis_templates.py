from __future__ import annotations

from sift.analysis_templates import TEMPLATES, instantiate_template, validate_templates
from sift.research_workflow import propose_workflow


def _spec():
    return {
        "research_question": "How is x associated with y?",
        "unit_of_analysis": "person", "outcome": "y", "exposures": ["x"],
        "treatment": None, "predictors": ["x"], "controls": [],
        "target_population": "eligible adults",
        "estimand": "adjusted mean difference in y per unit x",
        "study_design": "cross-sectional observational", "goal": "associational",
        "repeated_measures": False, "clusters": None, "weights": None,
        "strata": None, "psu": None, "fpc": None, "replicate_weights": None,
        "time_ordering": "x measured before or with y; causal order unresolved",
        "missing_data_assumption": "MAR for adjusted analysis",
    }


def test_templates_are_complete_and_cover_core_research_goals() -> None:
    assert validate_templates() == ()
    assert {goal for row in TEMPLATES.values() for goal in row.goals} >= {
        "descriptive", "associational", "inferential", "causal", "predictive",
    }
    assert all(row.sensitivities for row in TEMPLATES.values())


def test_template_instantiation_is_directly_accepted_by_workflow(tmp_path) -> None:
    proposal = instantiate_template(
        "observational_association", method_id="linear_regression",
        research_specification=_spec(), seed=20260821,
    )
    assert proposal["analyses"][0]["role"] == "primary"
    assert all(row["seed"] == 20260821 for row in proposal["analyses"])
    result = propose_workflow(tmp_path, proposal)
    assert result["method_id"] == "linear_regression"
    assert len([row for row in result["analyses"] if row["role"] == "sensitivity"]) == 2
