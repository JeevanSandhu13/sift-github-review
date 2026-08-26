"""Verification coverage across every analysis shape the sanitizer knows.

Two failure modes are equally bad and both are tested:

- **Silence on a broken analysis.** A researcher reading no warnings
  concludes there was nothing to warn about. Every check below is
  therefore exercised against a payload that should trip it.
- **Noise on a sound analysis.** A layer that warns about everything
  is one people learn to ignore, which is worse than not having it.
  Every shape is also exercised against a clean payload that must
  produce zero warnings.

Checks may only reference fields the sanitizer actually emits for
that shape; a check on a stripped field would never fire and would be
dead weight masquerading as coverage. ``test_checks_reference_real_
sanitizer_fields`` guards that.
"""

from __future__ import annotations

import pytest

from sift.verification import verify_payload


# --------------------------------------------------------------------
# Fixtures: one clean and one pathological payload per shape
# --------------------------------------------------------------------

CLEAN = {
    "did_event_study": {
        "type": "did_event_study", "groups": ["2019"],
        "event_times": [-3, -2, -1, 0, 1],
        "att": {"2019": {"0": 0.1}}, "n_treated_per_group": {"2019": 420},
        "pre_trends_p_value": 0.62, "aggregate_att": 0.12,
        "aggregate_se": 0.03, "n_pre_treatment_periods": 3,
    },
    "rdd": {
        "type": "rdd", "running_variable": "score", "cutoff": 50,
        "tau_robust": 2.1, "se_robust": 0.4, "effective_n_left": 420,
        "effective_n_right": 480, "polynomial_order": 1,
        "bandwidth_left": 5.0, "bandwidth_right": 5.4,
        "tau_conventional": 2.0,
    },
    "kaplan_meier": {
        "type": "kaplan_meier", "time_variable": "t", "event_variable": "d",
        "n_subjects": 900, "n_failures": 260, "survival_at_5y": 0.62,
        "n_at_risk_5y": 310, "median_survival_time": 48,
        "median_survival_ci_lower": 41, "median_survival_ci_upper": 55,
    },
    "cluster_analysis": {
        "type": "cluster_analysis", "method": "kmeans",
        "n_observations": 5000, "n_clusters": 4, "n_features": 6,
        "variables": ["a"], "cluster_labels": ["1"],
        "cluster_sizes": {"1": 1200, "2": 1500, "3": 1100, "4": 1200},
        "silhouette_score": 0.58,
    },
    "factor_decomposition": {
        "type": "factor_decomposition", "method": "pca",
        "n_observations": 1200, "n_variables": 12, "n_components": 3,
        "variables": ["a"], "loadings": {}, "kmo": 0.86,
        "bartlett_p_value": 1e-20, "rmsea": 0.041, "tli": 0.97,
    },
    "marginal_effects": {
        "type": "marginal_effects", "n": 5000, "method": "AME",
        "variables": ["x"], "effects": {"x": 0.12},
        "standard_errors": {"x": 0.03},
    },
}

BROKEN = {
    "did_event_study": ({
        "type": "did_event_study", "groups": ["2019"], "event_times": [0],
        "att": {"2019": {"0": 0.1}},
        "n_treated_per_group": {"2019": 4}, "pre_trends_p_value": 0.01,
        "aggregate_att": 0.12, "n_pre_treatment_periods": 1,
    }, {"parallel_trends", "cohort_size", "uncertainty_reported",
        "pre_periods"}),
    "rdd": ({
        "type": "rdd", "running_variable": "score", "cutoff": 50,
        "tau_robust": 2.1, "se_robust": 0.9, "effective_n_left": 18,
        "effective_n_right": 140, "polynomial_order": 4,
        "bandwidth_left": 2.0, "bandwidth_right": 9.0,
        "tau_conventional": 0.4,
    }, {"effective_sample", "sample_balance", "polynomial_order",
        "bandwidth_symmetry", "estimator_agreement"}),
    "kaplan_meier": ({
        "type": "kaplan_meier", "time_variable": "t", "event_variable": "d",
        "n_subjects": 300, "n_failures": 6, "survival_at_5y": 0.62,
        "n_at_risk_5y": 3, "median_survival_time": 48,
    }, {"event_count", "censoring", "at_risk_5y", "median_ci"}),
    "cluster_analysis": ({
        "type": "cluster_analysis", "method": "kmeans",
        "n_observations": 500, "n_clusters": 8, "n_features": 4,
        "variables": ["a"], "cluster_labels": ["1"],
        "cluster_sizes": {"1": 4, "2": 200}, "silhouette_score": 0.11,
    }, {"cluster_separation", "cluster_sizes"}),
    "factor_decomposition": ({
        "type": "factor_decomposition", "method": "pca",
        "n_observations": 90, "n_variables": 30, "n_components": 3,
        "variables": ["a"], "loadings": {}, "kmo": 0.41,
        "bartlett_p_value": 0.31, "rmsea": 0.14,
    }, {"sampling_adequacy", "sphericity", "model_fit",
        "obs_per_variable"}),
    "marginal_effects": ({
        "type": "marginal_effects", "n": 5000, "method": "AME",
        "variables": ["x"], "effects": {"x": 0.12},
    }, {"uncertainty_reported"}),
}


@pytest.mark.parametrize("shape", sorted(CLEAN))
def test_clean_payloads_produce_no_warnings(shape) -> None:
    block = verify_payload(CLEAN[shape])
    assert block is not None, f"{shape}: no checks ran at all"
    offenders = [c["id"] for c in block["checks"] if c["status"] == "warn"]
    assert not offenders, f"{shape} false positives: {offenders}"


@pytest.mark.parametrize("shape", sorted(BROKEN))
def test_broken_payloads_trip_the_expected_checks(shape) -> None:
    payload, expected = BROKEN[shape]
    block = verify_payload(payload)
    assert block is not None
    warned = {c["id"] for c in block["checks"] if c["status"] == "warn"}
    missing = expected - warned
    assert not missing, f"{shape} failed to warn about: {missing}"


def test_every_sanitizer_shape_has_verification_coverage() -> None:
    """No analysis type may silently have zero checks. Silence reads
    to a researcher as 'nothing to flag'."""
    shapes = {
        "linear_regression": {
            "type": "linear_regression", "n": 500,
            "coefficients": {"x": 1.0}, "standard_errors": {"x": 0.1},
        },
        "t_test": {"type": "t_test", "n1": 5, "n2": 400,
                   "t_statistic": 2.0, "p_value": 0.04},
        "descriptive": {"type": "descriptive", "variable": "age",
                        "n": 100, "mean": 40.0, "missing_count": 60},
        "frequency_table": {"type": "frequency_table", "variable": "r",
                            "n": 100, "counts": {"a": 90, "b": "<10"}},
        "crosstab": {"type": "crosstab", "row_variable": "a",
                     "column_variable": "b", "n": 100,
                     "counts": {"a|b": "<10"}},
        "magnitude_table": {"type": "magnitude_table", "row_variable": "r",
                            "value_variable": "v", "aggregation": "sum",
                            "cells": {"north": 100, "south": "<10"}},
    }
    shapes.update(CLEAN)
    for name, payload in shapes.items():
        block = verify_payload(payload)
        assert block is not None and block["checks"], \
            f"{name} produced no verification checks"


def test_missing_diagnostics_are_flagged_not_assumed_fine() -> None:
    """A DiD with no pre-trend test and a cluster run with no
    separation metric must say so rather than stay silent — absence of
    evidence is the thing a reader needs told."""
    did = verify_payload({
        "type": "did_event_study", "groups": ["g"], "event_times": [0],
        "att": {"g": {"0": 0.1}}, "n_treated_per_group": {"g": 500},
    })
    assert any(c["id"] == "parallel_trends" and c["status"] == "warn"
               for c in did["checks"])

    clus = verify_payload({
        "type": "cluster_analysis", "method": "kmeans",
        "n_observations": 5000, "n_clusters": 3, "n_features": 4,
        "variables": ["a"], "cluster_labels": ["1"],
        "cluster_sizes": {"1": 2000, "2": 1500, "3": 1500},
    })
    assert any(c["id"] == "cluster_separation" and c["status"] == "warn"
               for c in clus["checks"])


def test_suppression_markers_never_crash_a_check() -> None:
    """Sanitized payloads carry strings like '<10' in numeric slots."""
    for payload in (
        {"type": "cluster_analysis", "method": "k", "n_observations": 100,
         "n_clusters": 2, "n_features": 2, "variables": ["a"],
         "cluster_labels": ["1"], "cluster_sizes": {"1": "<10", "2": 90}},
        {"type": "kaplan_meier", "time_variable": "t",
         "event_variable": "d", "n_subjects": "<10", "n_failures": "<10"},
        {"type": "did_event_study", "groups": ["g"], "event_times": [0],
         "att": {"g": {"0": 0.1}}, "n_treated_per_group": {"g": "<10"}},
    ):
        verify_payload(payload)   # must not raise


def test_verification_never_raises_on_malformed_payloads() -> None:
    for payload in (
        {"type": "rdd"}, {"type": "kaplan_meier", "n_failures": None},
        {"type": "factor_decomposition", "loadings": "not-a-dict"},
        {"type": "cluster_analysis", "cluster_sizes": []},
        {"type": "did_event_study", "n_treated_per_group": "nope"},
        {"type": "marginal_effects", "effects": None},
    ):
        verify_payload(payload)   # must not raise


# --------------------------------------------------------------------
# Session-level accounting (cross-result)
# --------------------------------------------------------------------

def _item(n, p, label="x", dataset="cohort.parquet"):
    return {"label": label, "analysis_type": "linear_regression",
            "payload": {"type": "linear_regression", "n": n,
                        "p_values": {"x": p},
                        "source_dataset": dataset},
            "source_dataset": dataset}


def test_accumulated_multiple_comparisons_flagged() -> None:
    """Forty individually-clean tests are a multiple-comparisons
    problem no per-result check can see."""
    from sift.verification import session_report

    rep = session_report([_item(5000, 0.01 if i % 5 == 0 else 0.4)
                          for i in range(20)])
    assert rep["tests_run"] == 20
    warn = [c for c in rep["checks"]
            if c["id"] == "session_multiple_comparisons"]
    assert warn and warn[0]["status"] == "warn"
    assert "every tested null were true" in warn[0]["detail"]


def test_nested_did_p_values_count_toward_session_multiplicity() -> None:
    from sift.verification import session_report

    payload = {
        "type": "did_event_study",
        "p_values": {
            "2018": {str(t): 0.01 if t == 0 else 0.4 for t in range(-2, 3)},
            "2019": {str(t): 0.03 if t == 1 else 0.5 for t in range(-2, 3)},
        },
    }
    rep = session_report([{
        "analysis_type": "did_event_study", "payload": payload,
        "source_dataset": "panel.parquet",
    }])
    assert rep["tests_run"] == 10
    assert rep["significant"] == 2
    assert any(c["id"] == "session_multiple_comparisons"
               and c["status"] == "warn" for c in rep["checks"])


def test_small_number_of_tests_is_not_nagged() -> None:
    from sift.verification import session_report

    rep = session_report([_item(5000, 0.02) for _ in range(3)])
    check = [c for c in rep["checks"]
             if c["id"] == "session_multiple_comparisons"][0]
    assert check["status"] == "pass"


def test_sample_drift_detected_per_dataset() -> None:
    """N changing between results on one dataset means a filter or
    merge moved the population."""
    from sift.verification import session_report

    rep = session_report([_item(5000, 0.3), _item(3200, 0.3)])
    drift = [c for c in rep["checks"] if c["id"].startswith("sample_drift")]
    assert drift and drift[0]["status"] == "warn"
    assert "3,200" in drift[0]["detail"] and "5,000" in drift[0]["detail"]


def test_stable_samples_pass() -> None:
    from sift.verification import session_report

    rep = session_report([_item(5000, 0.3), _item(5000, 0.4)])
    drift = [c for c in rep["checks"] if c["id"].startswith("sample_drift")]
    assert drift and drift[0]["status"] == "pass"


def test_drift_is_scoped_per_dataset() -> None:
    """Different datasets legitimately have different N — that is not
    drift and must not be reported as such."""
    from sift.verification import session_report

    rep = session_report([_item(5000, 0.3, dataset="a.csv"),
                          _item(200, 0.3, dataset="b.csv")])
    assert all(c["status"] == "pass" for c in rep["checks"]
               if c["id"].startswith("sample_drift"))


def test_session_report_tolerates_junk() -> None:
    from sift.verification import session_report

    for junk in ([], [{}], [{"payload": None}], [{"payload": "x"}],
                 [{"payload": {"n": float("nan"), "p_value": float("inf")}}],
                 [{"payload": {"p_values": {"a": 10 ** 400}}}]):
        rep = session_report(junk)
        assert isinstance(rep["warnings"], int)


# --------------------------------------------------------------------
# Specification search / garden-of-forking-paths detection
# --------------------------------------------------------------------

_UNSET = object()  # sentinel so script_run_id=None is distinguishable
                    # from "caller didn't pass the argument at all"


def _reg_item(predictors, p_values, response="wage", dataset="cohort.csv",
              label="spec", coefficients=None, analysis_type="linear_regression",
              script_run_id=_UNSET):
    coeffs = coefficients if coefficients is not None else {
        k: 1.0 for k in p_values
    }
    # Default script_run_id to the label when the caller doesn't pass
    # one at all: existing tests already give each specification a
    # distinct label ("spec1", "spec2", ...), so this reproduces "N
    # separate submit_script calls" by default without touching every
    # call site. Tests about the single-batch suppression case pass a
    # shared script_run_id explicitly; tests about the "no run-id
    # recorded at all" case pass script_run_id=None explicitly --
    # which must stay None, not silently fall back to the label.
    run_id = label if script_run_id is _UNSET else script_run_id
    return {
        "label": label,
        "analysis_type": analysis_type,
        "payload": {
            "type": analysis_type,
            "response_variable": response,
            "predictor_variables": predictors,
            "coefficients": coeffs,
            "p_values": p_values,
        },
        "source_dataset": dataset,
        "script_run_id": run_id,
    }


def test_spec_search_flags_many_distinct_specifications() -> None:
    """Three-plus different control sets fit against the same outcome
    is exactly the pattern challenge_summary cannot see because these
    calls were never batched together."""
    from sift.verification import session_report

    items = [
        _reg_item(["educ"], {"educ": 0.2}, label="spec1"),
        _reg_item(["educ", "exper"], {"educ": 0.2, "exper": 0.3},
                  label="spec2"),
        _reg_item(["educ", "exper", "tenure"],
                  {"educ": 0.2, "exper": 0.3, "tenure": 0.4},
                  label="spec3"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"]
    assert hits and hits[0]["status"] == "warn"
    assert "3 distinct specifications" in hits[0]["detail"]


def test_spec_search_two_specs_below_count_threshold() -> None:
    """Two specifications is a normal robustness pair, not a forking-
    paths pattern — the count note must not fire yet."""
    from sift.verification import session_report

    items = [
        _reg_item(["educ"], {"educ": 0.2}, label="spec1"),
        _reg_item(["educ", "exper"], {"educ": 0.2, "exper": 0.3},
                  label="spec2"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"]
    assert not hits


def test_spec_search_flags_significance_flip() -> None:
    """A predictor significant in one specification and not another,
    for the same outcome, is the concrete "kept adding controls until
    it turned significant" tell."""
    from sift.verification import session_report

    items = [
        _reg_item(["treat"], {"treat": 0.01}, label="spec1"),
        _reg_item(["treat", "controls"], {"treat": 0.60, "controls": 0.02},
                  label="spec2"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"
            "::significance_stability"]
    assert hits and hits[0]["status"] == "warn"
    assert "treat" in hits[0]["detail"]


def test_spec_search_stable_significance_passes() -> None:
    """A predictor that stays significant (or stays non-significant)
    across every specification tried is exactly what a real robustness
    check should look like — must report pass, not warn."""
    from sift.verification import session_report

    items = [
        _reg_item(["treat"], {"treat": 0.01}, label="spec1"),
        _reg_item(["treat", "controls"], {"treat": 0.02, "controls": 0.4},
                  label="spec2"),
        _reg_item(["treat", "controls", "region"],
                  {"treat": 0.015, "controls": 0.4, "region": 0.5},
                  label="spec3"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"
            "::significance_stability"]
    assert hits and hits[0]["status"] == "pass"


def test_spec_search_scoped_per_response_variable() -> None:
    """Different outcomes on the same dataset must not cross-
    contaminate each other's specification count or stability check."""
    from sift.verification import session_report

    items = [
        _reg_item(["educ"], {"educ": 0.2}, response="wage", label="w1"),
        _reg_item(["educ", "exper"], {"educ": 0.2, "exper": 0.3},
                  response="wage", label="w2"),
        _reg_item(["age"], {"age": 0.03}, response="tenure", label="t1"),
    ]
    rep = session_report(items)
    tenure_hits = [c for c in rep["checks"]
                   if "tenure" in c["id"]]
    assert not tenure_hits


def test_spec_search_ignores_non_regression_types() -> None:
    """A crosstab or t-test sharing a p_value key must not be swept
    into regression-specification grouping — it has no
    response_variable/predictor_variables shape to compare."""
    from sift.verification import session_report

    items = [
        {"label": "ct", "analysis_type": "crosstab",
         "payload": {"type": "crosstab", "p_value": 0.03}},
        {"label": "tt", "analysis_type": "t_test",
         "payload": {"type": "t_test", "p_value": 0.04}},
    ]
    rep = session_report(items)
    assert not [c for c in rep["checks"]
                if c["id"].startswith("specification_search")]


def test_spec_search_tolerates_missing_predictor_variables() -> None:
    """A regression-bucket payload missing predictor_variables (older
    row, malformed helper output) must be skipped, not crash the
    session report."""
    from sift.verification import session_report

    items = [
        {"label": "bad", "analysis_type": "linear_regression",
         "payload": {"type": "linear_regression",
                     "response_variable": "wage", "p_values": {"x": 0.2}}},
        _reg_item(["educ"], {"educ": 0.2}, label="spec1"),
    ]
    rep = session_report(items)  # must not raise
    assert isinstance(rep["warnings"], int)


def test_spec_search_canonical_type_alias_also_grouped() -> None:
    """The canonical regression-bucket type name must group the same
    as the legacy alias — both route through the same sanitizer
    handler and must be treated identically here."""
    from sift.verification import session_report

    items = [
        _reg_item(["educ"], {"educ": 0.2}, label="spec1",
                  analysis_type="coefficient_table_with_fit_stats"),
        _reg_item(["educ", "exper"], {"educ": 0.2, "exper": 0.3},
                  label="spec2", analysis_type="linear_regression"),
        _reg_item(["educ", "exper", "tenure"],
                  {"educ": 0.2, "exper": 0.3, "tenure": 0.4},
                  label="spec3",
                  analysis_type="coefficient_table_with_fit_stats"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"]
    assert hits and hits[0]["status"] == "warn"


def test_spec_search_single_batch_robustness_table_is_not_flagged() -> None:
    """The critical false-positive guard: a script that runs the
    mandatory robustness pass (baseline + several alternative specs,
    ALL in one script_run_id) is exactly the transparent, reported-
    together case challenge_summary already gives a ROBUST/FRAGILE
    verdict for. Without the script_run_id gate, this exact
    well-behaved pattern would trip specification_search on every
    single robustness pass in the product -- the opposite of what
    the detector is for. Same script_run_id for every spec here."""
    from sift.verification import session_report

    items = [
        _reg_item(["treat"], {"treat": 0.01}, label="baseline",
                  script_run_id="run-A"),
        _reg_item(["treat", "controls"], {"treat": 0.60, "controls": 0.02},
                  label="drop outliers", script_run_id="run-A"),
        _reg_item(["treat", "controls", "region"],
                  {"treat": 0.90, "controls": 0.02, "region": 0.4},
                  label="clustered SE", script_run_id="run-A"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"].startswith("specification_search")]
    assert not hits, hits


def test_spec_search_fires_once_a_second_run_adds_a_spec() -> None:
    """The mirror case: the same three specs as the single-batch test
    above, but the third one came from a SEPARATE script_run_id (a
    later, disconnected submit_script call) -- now it's the exact
    cross-call pattern the detector exists to catch, and must fire."""
    from sift.verification import session_report

    items = [
        _reg_item(["treat"], {"treat": 0.01}, label="baseline",
                  script_run_id="run-A"),
        _reg_item(["treat", "controls"], {"treat": 0.60, "controls": 0.02},
                  label="drop outliers", script_run_id="run-A"),
        _reg_item(["treat", "controls", "region"],
                  {"treat": 0.90, "controls": 0.02, "region": 0.4},
                  label="separate later call", script_run_id="run-B"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"]
    assert hits and hits[0]["status"] == "warn"
    assert "2 separate script runs" in hits[0]["detail"]


def test_spec_search_missing_script_run_id_does_not_fire() -> None:
    """Rows with no recorded script_run_id (older data, or a caller
    that didn't thread it through) must not be treated as
    automatically "separate calls" -- that would silently resurrect
    the false-positive-on-single-batch bug for any caller that
    doesn't populate the field. Absence is absence, not evidence of
    two runs."""
    from sift.verification import session_report

    items = [
        _reg_item(["treat"], {"treat": 0.01}, label="a", script_run_id=None),
        _reg_item(["treat", "controls"], {"treat": 0.6, "controls": 0.02},
                  label="b", script_run_id=None),
        _reg_item(["treat", "controls", "region"],
                  {"treat": 0.9, "controls": 0.02, "region": 0.4},
                  label="c", script_run_id=None),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"].startswith("specification_search")]
    assert not hits, hits


def test_spec_search_flip_confined_to_one_batch_is_not_flagged() -> None:
    """Precision guard on the significance-stability check specifically:
    a group can legitimately contain 2+ distinct script_run_ids (the
    group-level gate) while the actual significance disagreement is
    entirely internal to ONE of those runs -- e.g. three specs
    batched together in run A (already challenge_summary's territory,
    its own internal instability already accounted for there) plus a
    totally unrelated spec on the same outcome from run B elsewhere
    in the session. The flip here must not be attributed to "specs
    run separately" when the disagreeing entries never left run A."""
    from sift.verification import session_report

    items = [
        # Run A: an internal robustness batch where "treat" flips
        # significance among ITSELF.
        _reg_item(["treat"], {"treat": 0.01}, label="baseline",
                  script_run_id="run-A"),
        _reg_item(["treat", "controls"], {"treat": 0.60, "controls": 0.02},
                  label="drop outliers", script_run_id="run-A"),
        # Run B: unrelated later spec on the same outcome that does
        # NOT even reference "treat" -- present only to satisfy the
        # group-level "spans 2 runs" gate.
        _reg_item(["region"], {"region": 0.03}, label="later spec",
                  script_run_id="run-B"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"
            "::significance_stability"]
    assert not hits, hits


def test_spec_search_flip_across_runs_is_still_flagged() -> None:
    """Mirror of the guard above: when the disagreeing entries for a
    predictor genuinely DO span two different runs, the check must
    still fire -- confirming the run-diversity requirement narrows
    the check rather than disabling it."""
    from sift.verification import session_report

    items = [
        _reg_item(["treat"], {"treat": 0.01}, label="baseline",
                  script_run_id="run-A"),
        _reg_item(["treat", "controls"], {"treat": 0.60, "controls": 0.02},
                  label="later separate call", script_run_id="run-B"),
    ]
    rep = session_report(items)
    hits = [c for c in rep["checks"]
            if c["id"] == "specification_search::cohort.csv::wage"
            "::significance_stability"]
    assert hits and hits[0]["status"] == "warn"
    assert "treat" in hits[0]["detail"]
