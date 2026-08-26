"""Validated, reproducible research-plan templates.

Templates generate workflow *metadata*, never executable code.  They provide a
repeatable primary/sensitivity structure and deterministic seeds while method
selection remains governed by :mod:`sift.methodology` and researcher approval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from sift.methodology import METHODS, evaluate_method


@dataclass(frozen=True)
class AnalysisTemplate:
    id: str
    title: str
    goals: tuple[str, ...]
    description: str
    primary_title: str
    primary_rationale: str
    sensitivities: tuple[tuple[str, str, tuple[str, ...]], ...]
    required_quality_roles: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "goals": self.goals,
            "description": self.description,
            "required_quality_roles": self.required_quality_roles,
            "sensitivity_count": len(self.sensitivities),
        }


_TEMPLATES = (
    AnalysisTemplate(
        "descriptive_profile", "Reproducible descriptive profile",
        ("descriptive",),
        "A declared sample profile with uncertainty and a missingness sensitivity.",
        "Primary declared-sample profile",
        "Describes the declared unit and population using the fixed analysis set.",
        (("complete_case_sensitivity", "Missingness sensitivity",
          ("Compare available-case and complete-case summaries",)),),
        ("identifiers", "units"),
    ),
    AnalysisTemplate(
        "observational_association", "Observational association with robustness",
        ("associational", "inferential"),
        "One pre-designated adjusted model plus variance and specification checks.",
        "Primary adjusted association",
        "Targets the declared estimand using the pre-specified adjustment set.",
        (
            ("variance_sensitivity", "Variance-estimator sensitivity",
             ("Use a design-appropriate robust or clustered variance estimator",)),
            ("specification_sensitivity", "Adjustment-set sensitivity",
             ("Compare the declared primary controls with one justified alternative",)),
        ),
        ("outcome", "predictors", "controls", "clusters"),
    ),
    AnalysisTemplate(
        "causal_identification", "Causal identification and falsification",
        ("causal",),
        "A primary causal estimand with overlap, falsification, and design sensitivity checks.",
        "Primary identified causal effect",
        "Estimates the declared causal estimand under the approved identification design.",
        (
            ("overlap_sensitivity", "Overlap/positivity sensitivity",
             ("Vary the supported analysis population or trimming rule",)),
            ("falsification", "Design falsification check",
             ("Run the method-specific placebo, pre-trend, or negative-control check",)),
        ),
        ("outcome", "treatment", "time", "identifiers"),
    ),
    AnalysisTemplate(
        "predictive_validation", "Leakage-safe predictive validation",
        ("predictive",),
        "A fixed out-of-sample evaluation with baseline and split sensitivity.",
        "Primary out-of-sample model",
        "Evaluates the approved model on data not used for fitting or tuning.",
        (
            ("baseline", "Simple-baseline comparison",
             ("Compare with a constant, majority-class, or simple linear baseline",)),
            ("split_sensitivity", "Evaluation-split sensitivity",
             ("Use a second valid grouped or temporal split without leakage",)),
        ),
        ("target", "features", "split", "identifiers", "time"),
    ),
)

TEMPLATES = {template.id: template for template in _TEMPLATES}


def validate_templates() -> tuple[str, ...]:
    errors: list[str] = []
    if len(TEMPLATES) != len(_TEMPLATES):
        errors.append("duplicate template id")
    for template in _TEMPLATES:
        if not template.id or not template.title or not template.description:
            errors.append(f"{template.id or '(missing)'} lacks metadata")
        if not template.goals or not template.primary_title or not template.primary_rationale:
            errors.append(f"{template.id} lacks a primary-analysis contract")
        if not template.sensitivities:
            errors.append(f"{template.id} has no sensitivity analysis")
        if any(not sid or not title or not changes for sid, title, changes in template.sensitivities):
            errors.append(f"{template.id} has an incomplete sensitivity analysis")
    return tuple(errors)


def instantiate_template(template_id: str, *, method_id: str,
                         research_specification: Mapping[str, Any],
                         seed: int) -> dict[str, Any]:
    """Return a proposal accepted by ``research_workflow.propose_workflow``."""
    template = TEMPLATES.get(str(template_id))
    if template is None:
        raise ValueError("unknown analysis template")
    goal = str(research_specification.get("goal") or "")
    if goal not in template.goals:
        raise ValueError(f"template {template_id!r} does not support goal {goal!r}")
    if method_id not in METHODS:
        raise ValueError("method is not in the methodology registry")
    evaluated = evaluate_method(method_id, research_specification)
    if not evaluated.get("valid"):
        raise ValueError("research specification is incomplete for the selected method")
    if isinstance(seed, bool) or not isinstance(seed, int) or not (0 <= seed <= 2**32 - 1):
        raise ValueError("seed must be an integer from 0 to 2^32-1")
    analyses: list[dict[str, Any]] = [{
        "id": "primary", "title": template.primary_title,
        "role": "primary", "method_id": method_id,
        "rationale": template.primary_rationale, "changes": [], "seed": seed,
    }]
    for sensitivity_id, title, changes in template.sensitivities:
        analyses.append({
            "id": sensitivity_id, "title": title, "role": "sensitivity",
            "method_id": method_id,
            "rationale": "Challenges the primary result under a reasonable pre-declared alternative.",
            "changes": list(changes), "seed": seed,
        })
    return {
        "template_id": template.id,
        "method_id": method_id,
        "research_specification": dict(research_specification),
        "assumptions": list(evaluated["contract"]["assumptions"]),
        "unresolved_quality_issues": [],
        "analyses": analyses,
        "required_quality_roles": list(template.required_quality_roles),
    }


_TEMPLATE_ERRORS = validate_templates()
if _TEMPLATE_ERRORS:
    raise RuntimeError(f"invalid built-in analysis templates: {_TEMPLATE_ERRORS}")

__all__ = ["AnalysisTemplate", "TEMPLATES", "instantiate_template", "validate_templates"]
