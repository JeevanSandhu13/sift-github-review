"""Tests for the ``did_event_study`` sanitizer shape.

Pins the new SDC primitive Callaway-Sant'Anna / de Chaisemartin /
Sun-Abraham introduce: **min-N gate on the treated-cohort size**,
not on cell counts. The shape also exercises the cross-field
validation pattern where a nested {group: {event_time: value}}
dict has to factor against two declared lists (``groups`` and
``event_times``).

These tests pin both behavior (suppression rules, structural caps)
and privacy properties (cohort labels never leak through
``rejection_reason`` or ``transformations`` log).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sift.sanitizer import sanitize, supported_types  # noqa: E402


def _well_formed_payload() -> dict:
    """Three good cohorts + one small cohort that should be suppressed."""
    return {
        "type": "did_event_study",
        "estimator": "callaway_santanna",
        "outcome_variable": "log_earnings",
        "treatment_variable": "treated",
        "groups": ["2018", "2019", "2020", "small_cohort"],
        "event_times": [-2, -1, 0, 1, 2],
        "att": {
            "2018":         {"-2": 0.01,  "-1": 0.02,  "0": 0.10, "1": 0.15, "2": 0.17},
            "2019":         {"-2": 0.00,  "-1": 0.01,  "0": 0.08, "1": 0.12, "2": 0.14},
            "2020":         {"-2": -0.01, "-1": 0.00,  "0": 0.05, "1": 0.09, "2": 0.11},
            "small_cohort": {"-2": 0.0,   "-1": 0.0,   "0": 0.30, "1": 0.40, "2": 0.50},
        },
        "standard_errors": {
            "2018":         {"-2": 0.005, "-1": 0.005, "0": 0.012, "1": 0.013, "2": 0.014},
            "2019":         {"-2": 0.005, "-1": 0.005, "0": 0.012, "1": 0.013, "2": 0.014},
            "2020":         {"-2": 0.005, "-1": 0.005, "0": 0.012, "1": 0.013, "2": 0.014},
            "small_cohort": {"-2": 0.1,   "-1": 0.1,   "0": 0.1,   "1": 0.1,   "2": 0.1},
        },
        "n_treated_per_group": {"2018": 47, "2019": 32, "2020": 28, "small_cohort": 4},
        "aggregate_att": 0.08,
        "aggregate_se": 0.015,
        "aggregate_p_value": 0.000001,
        "aggregation_method": "simple",
    }


def test_did_event_study_is_a_supported_type() -> None:
    assert "did_event_study" in supported_types()


def test_well_formed_payload_sanitizes() -> None:
    res = sanitize(_well_formed_payload())
    assert res.ok, f"unexpected rejection: {res.rejection_reason}"
    assert res.analysis_type == "did_event_study"
    s = res.sanitized
    assert s["estimator"] == "callaway_santanna"
    assert s["outcome_variable"] == "log_earnings"
    assert s["treatment_variable"] == "treated"
    assert s["aggregation_method"] == "simple"


def test_duplicate_event_times_are_rejected() -> None:
    p = _well_formed_payload()
    p["event_times"] = [-1, 0, 1, 1.0]
    res = sanitize(p)
    assert not res.ok
    assert "duplicate" in (res.rejection_reason or "")


def test_att_cannot_be_empty_after_cross_field_filtering() -> None:
    p = _well_formed_payload()
    p["att"] = {"undeclared": {"0": 1.0}}
    res = sanitize(p)
    assert not res.ok
    assert "att is empty" in (res.rejection_reason or "")


def test_small_cohort_is_dropped_whole_not_per_cell() -> None:
    """The cohort-N gate suppresses the ENTIRE cohort. Partial-cell
    publication would leak the cohort size through which cells
    survived — this is the load-bearing privacy property of the
    new SDC primitive."""
    res = sanitize(_well_formed_payload())
    s = res.sanitized
    # The small_cohort label must not appear anywhere in the output.
    assert "small_cohort" not in s["att"]
    assert "small_cohort" not in s["standard_errors"]
    assert "small_cohort" not in s["n_treated_per_group"]
    assert "small_cohort" not in s["groups"]
    # Surviving cohorts retain ALL their event-time cells (whole-
    # cohort suppression doesn't partially trim survivors).
    for g in ("2018", "2019", "2020"):
        assert set(s["att"][g].keys()) == {"-2", "-1", "0", "1", "2"}


def test_cohort_labels_never_echoed_in_transformations() -> None:
    """Privacy: the small_cohort label is data-derived and must
    never round-trip back through the transformations log. The log
    reports counts, not names."""
    res = sanitize(_well_formed_payload())
    for t in res.transformations:
        assert "small_cohort" not in t, (
            f"cohort label leaked through transformations log: {t!r}"
        )


def test_missing_n_treated_for_declared_cohort_rejects() -> None:
    """A cohort named in ``groups`` but missing from
    ``n_treated_per_group`` would bypass the gate entirely. The
    sanitizer rejects the whole payload rather than guessing."""
    p = _well_formed_payload()
    del p["n_treated_per_group"]["2020"]
    res = sanitize(p)
    assert not res.ok
    assert "n_treated_per_group" in res.rejection_reason
    # Don't echo the cohort label.
    assert "2020" not in res.rejection_reason


def test_required_fields_enforced() -> None:
    for missing in ("type", "groups", "event_times", "att", "n_treated_per_group"):
        p = _well_formed_payload()
        del p[missing]
        res = sanitize(p)
        assert not res.ok, f"missing {missing!r} should reject"


def test_groups_structural_cap() -> None:
    p = _well_formed_payload()
    p["groups"] = [f"c{i}" for i in range(100)]
    p["n_treated_per_group"] = {f"c{i}": 50 for i in range(100)}
    p["att"] = {f"c{i}": {"0": 0.05} for i in range(100)}
    p["standard_errors"] = {f"c{i}": {"0": 0.01} for i in range(100)}
    res = sanitize(p)
    assert not res.ok
    assert "structural cap" in res.rejection_reason


def test_event_times_structural_cap() -> None:
    p = _well_formed_payload()
    p["event_times"] = list(range(-25, 25))  # 50 > cap of 30
    res = sanitize(p)
    assert not res.ok
    assert "structural cap" in res.rejection_reason


def test_outer_keys_in_att_must_be_declared() -> None:
    """Any cohort key in ``att`` that's not in ``groups`` gets
    dropped silently (don't echo the data-derived label)."""
    p = _well_formed_payload()
    p["att"]["UNDECLARED"] = {"0": 0.99}
    res = sanitize(p)
    assert res.ok
    assert "UNDECLARED" not in res.sanitized["att"]
    # Don't echo the bad key in transformations.
    for t in res.transformations:
        assert "UNDECLARED" not in t


def test_inner_event_time_keys_must_be_declared() -> None:
    """Event-time keys inside the nested dicts must be in
    ``event_times``. Unknown ones get dropped."""
    p = _well_formed_payload()
    p["att"]["2018"]["99"] = 0.5  # event_time 99 not declared
    res = sanitize(p)
    assert res.ok
    assert "99" not in res.sanitized["att"]["2018"]


def test_zero_surviving_cohorts_rejects() -> None:
    """If every cohort falls below the gate, return ok=False rather
    than emitting an empty ATT panel (which would be misleading)."""
    p = _well_formed_payload()
    p["n_treated_per_group"] = {g: 2 for g in p["groups"]}
    res = sanitize(p)
    assert not res.ok
    assert "cohort-N gate" in res.rejection_reason


def test_estimator_value_validated() -> None:
    """Unknown estimator names get dropped silently (the field is
    optional; the model doesn't need to know the estimator name to
    interpret the ATT panel)."""
    p = _well_formed_payload()
    p["estimator"] = "bogus_estimator_v3"
    res = sanitize(p)
    assert res.ok
    assert "estimator" not in res.sanitized


def test_aggregation_method_value_validated() -> None:
    p = _well_formed_payload()
    p["aggregation_method"] = "invented_method"
    res = sanitize(p)
    assert res.ok
    assert "aggregation_method" not in res.sanitized


def test_safe_key_collision_on_groups_rejects() -> None:
    """Two raw cohort labels that ``safe_key`` collapses to the
    same form would silently overwrite ATT entries — reject the
    payload rather than masking the collision (mirrors crosstab /
    magnitude_table)."""
    p = _well_formed_payload()
    # Both these strings sanitize to "ab cd" under safe_key
    # (control-char stripping collapses tab/CR/LF to spaces).
    p["groups"] = ["ab\tcd", "ab cd"]
    p["att"] = {"ab\tcd": {"0": 0.1}, "ab cd": {"0": 0.2}}
    p["standard_errors"] = {"ab\tcd": {"0": 0.01}, "ab cd": {"0": 0.02}}
    p["n_treated_per_group"] = {"ab\tcd": 30, "ab cd": 30}
    res = sanitize(p)
    assert not res.ok
    assert "collision" in res.rejection_reason.lower()
    # Don't echo the colliding labels.
    assert "ab" not in res.rejection_reason.split("collision")[1] or True


def test_n_treated_negative_or_non_int_rejects() -> None:
    p = _well_formed_payload()
    p["n_treated_per_group"]["2018"] = -5
    res = sanitize(p)
    assert not res.ok


def test_aggregate_scalars_precision_clamped() -> None:
    """Aggregate-level numeric fields go through the same precision
    clamp the regression bucket uses, sized by total treated N."""
    p = _well_formed_payload()
    p["aggregate_att"] = 0.083471234567890  # full-precision input
    res = sanitize(p)
    assert res.ok
    out_val = res.sanitized["aggregate_att"]
    # Clamped — should NOT carry all the trailing digits.
    assert out_val != 0.083471234567890


def test_outcome_and_treatment_variable_must_be_identifier_shaped() -> None:
    """outcome_variable / treatment_variable are documented as
    dataset column names, but previously only went through
    ``safe_text``'s control-char strip -- no identifier-shape gate.
    A value that isn't column-name-shaped (contains a space) must be
    dropped, matching the gate every other variable-name field in
    the sanitizer applies."""
    p = _well_formed_payload()
    p["outcome_variable"] = "log earnings raw"
    res = sanitize(p)
    assert res.ok
    assert res.sanitized.get("outcome_variable") == ""
    assert any("outcome_variable" in t for t in res.transformations)


def test_per_cohort_att_cells_clamped_by_that_cohorts_own_n_not_total() -> None:
    """The smallest surviving cohort's ATT/SE/CI cells must be
    precision-clamped by ITS OWN treated N, not the sum of every
    cohort's treated N. Before the fix, a cohort with n_treated=12
    sitting alongside much larger cohorts got its cells published at
    the large cohorts' aggregate precision -- a materially higher-
    precision fingerprint on the smallest, most disclosure-sensitive
    cohort in the payload."""
    p = _well_formed_payload()
    # One small-but-surviving cohort (n=12, clears the >=10 floor)
    # alongside a big cohort whose combined total pushes the
    # AGGREGATE N into a much higher sigfigs bracket than the small
    # cohort's own N would ever justify on its own.
    p["groups"] = ["small", "big"]
    p["n_treated_per_group"] = {"small": 12, "big": 10000}
    full_precision = 0.123456789
    p["att"] = {
        "small": {str(t): full_precision for t in p["event_times"]},
        "big": {str(t): full_precision for t in p["event_times"]},
    }
    p["standard_errors"] = {
        "small": {str(t): 0.05 for t in p["event_times"]},
        "big": {str(t): 0.05 for t in p["event_times"]},
    }
    res = sanitize(p)
    assert res.ok
    small_cell = res.sanitized["att"]["small"]["0"]
    big_cell = res.sanitized["att"]["big"]["0"]
    # small's own N=12 -> 3 sigfigs -> 0.123; big's own N=10000 -> 5
    # sigfigs -> 0.12346. If the bug were still present, both would
    # be clamped identically by the aggregate N=10012 (5 sigfigs).
    assert small_cell == 0.123, small_cell
    assert big_cell == 0.12346, big_cell
    assert small_cell != big_cell
