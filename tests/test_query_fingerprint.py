"""Query fingerprinting (sift.query_fingerprint) — repeated /
combined-release / differencing advisory findings.

Pins:

1. ``analyze_records`` finds each pattern only once its threshold is
   met, and not before.
2. ``submit_script`` batch results (``facts.results``) are scanned
   the same as single-result payloads.
3. Records for other tools, or with missing required fields, are
   ignored rather than raising or corrupting a finding.
4. ``analyze_ledger`` never raises — a missing/empty ledger yields an
   empty report.
5. ``note_for_new_request`` fires exactly once, at the call that
   crosses a threshold — not before, not again after.
6. End-to-end: ``tools.request_data`` attaches ``payload["privacy_note"]``
   at the call that completes a pattern, via the real ledger written
   by the ``@tool`` decorator's own recording chokepoint.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sift import release_ledger
from sift.config import use_cwd
from sift.query_fingerprint import (
    analyze_ledger,
    analyze_records,
    note_for_new_request,
)


def _mcp_text(payload: dict) -> dict:
    text_block = next(b for b in payload["content"] if b.get("type") == "text")
    return json.loads(text_block["text"])


def _rd_record(dataset: str, variable: str, request_type: str) -> dict:
    """A synthetic ledger record shaped like one ``request_data`` call."""
    return {
        "tool": "request_data",
        "args": {"dataset": dataset, "variable": variable,
                  "request_type": request_type},
    }


def _rd_record2(
    dataset: str, variable: str, variable2: str, request_type: str,
) -> dict:
    """A synthetic ledger record shaped like one two-variable
    ``request_data`` call (``correlation_pair``, currently the only
    request_type that carries ``variable2``)."""
    return {
        "tool": "request_data",
        "args": {"dataset": dataset, "variable": variable,
                  "variable2": variable2, "request_type": request_type},
    }


def _ss_record(dataset: str, analysis_type: str, n: float) -> dict:
    """A synthetic ledger record shaped like one ``submit_script``
    single-result call."""
    return {
        "tool": "submit_script",
        "facts": {"status": "ok", "source_dataset": dataset,
                  "analysis_type": analysis_type, "n": n},
    }


def _ss_batch_record(entries: list[tuple[str, str, float]]) -> dict:
    """A synthetic ledger record shaped like one ``submit_script``
    batch call, whose facts live in ``facts["results"]``."""
    return {
        "tool": "submit_script",
        "facts": {"results": [
            {"status": "ok", "source_dataset": d,
             "analysis_type": a, "n": n}
            for d, a, n in entries
        ]},
    }


# ---------------------------------------------------------------------------
# analyze_records: empty / no-pattern baseline
# ---------------------------------------------------------------------------


def test_empty_records_yield_empty_report() -> None:
    report = analyze_records([])
    assert report.is_empty()
    assert report.repeated_queries == ()
    assert report.differencing_candidates == ()
    assert report.combined_release_variables == ()


def test_other_tools_are_ignored() -> None:
    records = [
        {"tool": "get_schema", "args": {"dataset": "d.csv"}},
        {"tool": "list_results", "args": {}},
    ] * 5
    report = analyze_records(records)
    assert report.is_empty()


def test_malformed_request_data_records_are_skipped() -> None:
    """Missing dataset/variable/request_type must not raise or be
    silently counted toward a finding."""
    records = [
        {"tool": "request_data", "args": {"variable": "income"}},  # no dataset
        {"tool": "request_data", "args": {"dataset": "d.csv"}},  # no variable
        {"tool": "request_data", "args": {}},
        {"tool": "request_data"},  # no args key at all
    ]
    report = analyze_records(records)
    assert report.is_empty()


def test_explicitly_denied_request_data_records_are_not_release_events() -> None:
    records = [
        {
            **_rd_record("d.csv", "income", kind),
            "facts": {"status": "denied"},
        }
        for kind in ("na_count", "numeric_bounds", "quartiles")
    ]
    assert analyze_records(records).is_empty()


def test_malformed_submit_script_facts_are_skipped() -> None:
    records = [
        {"tool": "submit_script", "facts": {"analysis_type": "t_test", "n": 50}},  # no dataset
        {"tool": "submit_script", "facts": {"source_dataset": "d.csv", "n": 50}},  # no analysis_type
        {"tool": "submit_script", "facts": {"source_dataset": "d.csv", "analysis_type": "t_test"}},  # no n
        {"tool": "submit_script", "facts": {"source_dataset": "d.csv", "analysis_type": "t_test", "n": True}},  # bool, not real n
        {"tool": "submit_script", "facts": {"status": "ok", "source_dataset": "d.csv", "analysis_type": "t_test", "n": float("nan")}},
        {"tool": "submit_script", "facts": {"status": "ok", "source_dataset": "d.csv", "analysis_type": "t_test", "n": float("inf")}},
        {"tool": "submit_script"},  # no facts key
        {"tool": "submit_script", "facts": {"results": "not-a-list"}},
        {"tool": "submit_script", "facts": {"results": ["not-a-dict"]}},
    ]
    report = analyze_records(records)
    assert report.is_empty()


def test_rejected_submit_script_facts_are_not_analysis_events() -> None:
    records = [{"tool": "submit_script", "facts": {"results": [
        {"status": "rejected_by_sanitizer", "source_dataset": "d.csv",
         "analysis_type": "t_test", "n": 50},
        {"status": "ok", "source_dataset": "d.csv",
         "analysis_type": "t_test", "n": 50},
    ]}}]
    from sift.query_fingerprint import _submit_script_analysis_events
    assert _submit_script_analysis_events(records) == [{
        "dataset": "d.csv", "analysis_type": "t_test", "n": 50,
    }]


# ---------------------------------------------------------------------------
# repeated_queries
# ---------------------------------------------------------------------------


def test_repeated_queries_below_threshold_not_flagged() -> None:
    records = [_rd_record("d.csv", "income", "na_count"),
               _rd_record("d.csv", "income", "numeric_bounds")]
    report = analyze_records(records)
    assert report.repeated_queries == ()


def test_repeated_queries_at_threshold_flagged() -> None:
    records = [_rd_record("d.csv", "income", "na_count"),
               _rd_record("d.csv", "income", "numeric_bounds"),
               _rd_record("d.csv", "income", "quartiles")]
    report = analyze_records(records)
    assert len(report.repeated_queries) == 1
    f = report.repeated_queries[0]
    assert f.dataset == "d.csv" and f.variable == "income"
    assert f.count == 3
    assert f.request_types == ("na_count", "numeric_bounds", "quartiles")


def test_repeated_queries_collapse_equivalent_dataset_path_spellings() -> None:
    records = [_rd_record("./d.csv", "income", "na_count"),
               _rd_record("data/d.csv", "income", "numeric_bounds"),
               _rd_record(r"data\d.csv", "income", "quartiles")]
    report = analyze_records(records)
    assert len(report.repeated_queries) == 1
    assert report.repeated_queries[0].dataset == "d.csv"


def test_repeated_queries_scoped_per_dataset_and_variable() -> None:
    """Three calls split across two different variables must not
    trigger a repeated-query finding on either."""
    records = [_rd_record("d.csv", "income", "na_count"),
               _rd_record("d.csv", "age", "na_count"),
               _rd_record("d.csv", "income", "numeric_bounds")]
    report = analyze_records(records)
    assert report.repeated_queries == ()


# ---------------------------------------------------------------------------
# variable2 tracking (correlation_pair carries a SECOND variable that
# must count toward its own fingerprint tally, not disappear)
# ---------------------------------------------------------------------------


def test_variable2_counts_toward_its_own_repeated_query_tally() -> None:
    """The attack shape this closes: always placing the SENSITIVE
    variable in the variable2 slot against a different primary
    variable each time. Before the fix, ``income`` here would never
    accumulate any tally at all (it only ever appears as variable2),
    so three correlation_pair releases that each disclose a joint
    fact about income would sail past the repeated-query detector
    completely undetected."""
    records = [
        _rd_record2("d.csv", "region", "income", "correlation_pair"),
        _rd_record2("d.csv", "age", "income", "correlation_pair"),
        _rd_record2("d.csv", "employment_status", "income", "correlation_pair"),
    ]
    report = analyze_records(records)
    variables_flagged = {f.variable for f in report.repeated_queries}
    assert "income" in variables_flagged, (
        "variable2 must accumulate its own tally — three different "
        "primary variables all correlated against the same variable2 "
        "is three releases ABOUT that variable2, not zero"
    )
    # The primary variables, each queried only once, must NOT be
    # individually flagged — only income (present in all three) has
    # crossed the threshold.
    assert "region" not in variables_flagged
    assert "age" not in variables_flagged
    assert "employment_status" not in variables_flagged


def test_variable2_counts_toward_combined_release_tally() -> None:
    """Same attack shape, combined-release variant: three DIFFERENT
    request_types each pairing a different primary variable against
    the same variable2 must flag variable2 for the combined-release
    (many distinct fact types about one variable) pattern too."""
    records = [
        _rd_record("d.csv", "income", "na_count"),
        _rd_record2("d.csv", "region", "income", "correlation_pair"),
        _rd_record("d.csv", "income", "quartiles"),
    ]
    report = analyze_records(records)
    variables_flagged = {f.variable for f in report.combined_release_variables}
    assert "income" in variables_flagged


def test_variable2_same_as_variable_is_not_double_counted() -> None:
    """A malformed/degenerate record where variable2 happens to equal
    variable (data_request.py itself refuses this at the handler
    level, but the fingerprinting analysis must not crash or
    double-count if it ever sees one) must count as a SINGLE touch,
    not two."""
    records = [_rd_record2("d.csv", "income", "income", "correlation_pair")]
    report = analyze_records(records)
    # One record, one variable name -> at most one list entry of
    # length 1 for the (d.csv, income) key; nowhere near the
    # repeated-query threshold of 3, and no crash.
    assert report.repeated_queries == ()
    assert report.combined_release_variables == ()


# ---------------------------------------------------------------------------
# combined_release_variables
# ---------------------------------------------------------------------------


def test_combined_release_requires_distinct_types_not_just_count() -> None:
    """Three calls with the SAME request_type repeated is a repeated-
    query finding, not a combined-release finding (only 1 distinct
    type)."""
    records = [_rd_record("d.csv", "income", "na_count")] * 3
    report = analyze_records(records)
    assert len(report.repeated_queries) == 1
    assert report.combined_release_variables == ()


def test_combined_release_flagged_at_three_distinct_types() -> None:
    records = [_rd_record("d.csv", "income", "na_count"),
               _rd_record("d.csv", "income", "numeric_bounds"),
               _rd_record("d.csv", "income", "quartiles")]
    report = analyze_records(records)
    assert len(report.combined_release_variables) == 1
    f = report.combined_release_variables[0]
    assert f.dataset == "d.csv" and f.variable == "income"
    assert f.request_types == ("na_count", "numeric_bounds", "quartiles")


# ---------------------------------------------------------------------------
# differencing_candidates
# ---------------------------------------------------------------------------


def test_differencing_needs_both_min_observations_and_distinct_n() -> None:
    # Only 2 observations total -> below DIFFERENCING_MIN_OBSERVATIONS (3)
    records = [_ss_record("d.csv", "t_test", 200.0),
               _ss_record("d.csv", "t_test", 150.0)]
    assert analyze_records(records).differencing_candidates == ()


def test_differencing_needs_distinct_n_not_just_repeat_count() -> None:
    # 3 observations, but all the SAME n -> not a differencing signature
    records = [_ss_record("d.csv", "t_test", 200.0)] * 3
    assert analyze_records(records).differencing_candidates == ()


def test_differencing_flagged_with_varying_n() -> None:
    records = [_ss_record("d.csv", "t_test", 200.0),
               _ss_record("d.csv", "t_test", 150.0),
               _ss_record("d.csv", "t_test", 200.0)]
    report = analyze_records(records)
    assert len(report.differencing_candidates) == 1
    f = report.differencing_candidates[0]
    assert f.dataset == "d.csv" and f.analysis_type == "t_test"
    assert f.distinct_n_values == (150.0, 200.0)
    assert f.observation_count == 3


def test_differencing_scans_batch_results_list() -> None:
    """A single submit_script batch call recording three results in
    ``facts.results`` must be scanned the same as three separate
    single-result calls."""
    records = [_ss_batch_record([
        ("d.csv", "t_test", 200.0),
        ("d.csv", "t_test", 150.0),
        ("d.csv", "t_test", 200.0),
    ])]
    report = analyze_records(records)
    assert len(report.differencing_candidates) == 1
    assert report.differencing_candidates[0].observation_count == 3


def test_differencing_scoped_per_dataset_and_analysis_type() -> None:
    records = [_ss_record("d.csv", "t_test", 200.0),
               _ss_record("d.csv", "regression", 150.0),
               _ss_record("other.csv", "t_test", 90.0)]
    assert analyze_records(records).differencing_candidates == ()


# ---------------------------------------------------------------------------
# analyze_ledger: real disk I/O, never raises
# ---------------------------------------------------------------------------


def test_analyze_ledger_missing_ledger_is_empty(tmp_path: Path) -> None:
    report = analyze_ledger(tmp_path)
    assert report.is_empty()


def test_analyze_ledger_reads_real_recorded_calls(tmp_path: Path) -> None:
    for rt in ("na_count", "numeric_bounds", "quartiles"):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool="request_data",
            args={"dataset": "d.csv", "variable": "income", "request_type": rt},
            response={"content": [{"type": "text",
                                    "text": json.dumps({"status": "granted"})}]},
        )
    report = analyze_ledger(tmp_path)
    assert len(report.repeated_queries) == 1
    assert report.repeated_queries[0].variable == "income"


def test_analyze_ledger_survives_unreadable_ledger_dir(tmp_path: Path) -> None:
    """Point at a path whose ``.sift`` ledger location cannot exist
    as a directory (a file sits where the dir would go) — must yield
    an empty report, not raise."""
    blocker = tmp_path / ".sift"
    blocker.write_text("not a directory")
    report = analyze_ledger(tmp_path)
    assert report.is_empty()


# ---------------------------------------------------------------------------
# note_for_new_request: exact-threshold-only firing
# ---------------------------------------------------------------------------


def _write_rd(tmp_path: Path, dataset: str, variable: str, request_type: str) -> None:
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        args={"dataset": dataset, "variable": variable, "request_type": request_type},
        response={"content": [{"type": "text",
                                "text": json.dumps({"status": "granted"})}]},
    )


def _write_rd2(
    tmp_path: Path, dataset: str, variable: str, variable2: str,
    request_type: str,
) -> None:
    """Same as ``_write_rd`` for a two-variable call (``correlation_pair``)."""
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        args={"dataset": dataset, "variable": variable, "variable2": variable2,
              "request_type": request_type},
        response={"content": [{"type": "text",
                                "text": json.dumps({"status": "granted"})}]},
    )


def test_note_none_with_empty_ledger(tmp_path: Path) -> None:
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="na_count")
    assert note is None


def test_note_none_before_repeated_threshold(tmp_path: Path) -> None:
    # One prior call; this would be the 2nd -> below threshold (3)
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="numeric_bounds")
    assert note is None


def test_note_fires_exactly_at_repeated_threshold(tmp_path: Path) -> None:
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    # This would be the 3rd call -> exactly at threshold
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="numeric_bounds")
    assert note is not None
    assert "3" in note and "income" in note


def test_note_collapses_equivalent_historical_dataset_paths(tmp_path: Path) -> None:
    _write_rd(tmp_path, "./d.csv", "income", "na_count")
    _write_rd(tmp_path, r"data\d.csv", "income", "na_count")
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                request_type="numeric_bounds")
    assert note is not None
    assert "3" in note and "income" in note


def test_note_does_not_refire_past_repeated_threshold(tmp_path: Path) -> None:
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")
    # This would be the 4th call -> past the exact-match threshold,
    # and only 2 distinct types so far even counting this one.
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="na_count")
    assert note is None


def test_note_fires_at_combined_release_threshold_after_repeated_already_fired(
    tmp_path: Path,
) -> None:
    """Repeated fires at call 3 (count==3). A later call that first
    reaches 3 DISTINCT types (call 4+) must still get the combined
    message, even though the repeated-count check no longer applies."""
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")  # 3rd call: repeated fires (not observed here)
    # This would be the 4th call, and the 3rd DISTINCT type.
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="quartiles")
    assert note is not None
    assert "3" in note and "different request_data types" in note


def test_note_does_not_refire_past_combined_release_threshold(
    tmp_path: Path,
) -> None:
    """Audit pass 2 finding: unlike the repeated-query counter (the
    TOTAL call count, which strictly increases by 1 every call so an
    exact-match check against it is naturally one-shot), the DISTINCT
    request_type count can plateau across many calls whenever a later
    call reuses an already-seen type. Before this fix, the
    combined-release check compared the distinct count with ``==``,
    so it re-fired the identical "has now been queried via 3
    different types" note on EVERY subsequent call sitting at that
    same distinct count, not just the one that first reached it."""
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")
    _write_rd(tmp_path, "d.csv", "income", "quartiles")
    # 4th call: 3 distinct types already present BEFORE this call
    # (na_count, numeric_bounds, quartiles) -- this call reuses
    # "na_count" rather than introducing a 4th distinct type, and
    # isn't the 3rd call either, so NEITHER threshold should fire.
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="na_count")
    assert note is None, (
        "the combined-release note re-fired on a call that merely "
        "reused an already-seen request_type, instead of only firing "
        "once at the call that first reached the distinct-type "
        "threshold"
    )


def test_note_fires_again_when_a_new_distinct_type_first_reaches_threshold(
    tmp_path: Path,
) -> None:
    """Companion positive case: the combined-release note must still
    fire on the call that FIRST reaches the distinct-type threshold,
    even when that call is not the 3rd call overall (i.e. even when
    some earlier calls repeated a type rather than introducing a new
    one)."""
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "na_count")  # repeat, not distinct
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")
    # 4th call: only 2 distinct types so far (na_count, numeric_bounds);
    # this call introduces the 3rd distinct type for the first time.
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="quartiles")
    assert note is not None
    assert "3" in note and "different request_data types" in note


def test_note_scoped_per_dataset_and_variable(tmp_path: Path) -> None:
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")
    # Different variable -> its own count starts fresh
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="age",
                                 request_type="na_count")
    assert note is None


def test_note_fires_for_variable2_when_variable_itself_is_fresh(
    tmp_path: Path,
) -> None:
    """Real-time advisory variant of the variable2 fix: three prior
    correlation_pair calls each pairing a DIFFERENT, never-before-seen
    primary variable against the SAME variable2 must still produce a
    note on the third one — via variable2's own tally, even though the
    current call's ``variable`` argument (the current call's primary)
    has no history of its own."""
    _write_rd2(tmp_path, "d.csv", "region", "income", "correlation_pair")
    _write_rd2(tmp_path, "d.csv", "age", "income", "correlation_pair")
    # This would be the 3rd correlation_pair call with income as
    # variable2, but "employment_status" (this call's own variable)
    # has never been queried before.
    note = note_for_new_request(
        tmp_path, dataset="d.csv", variable="employment_status",
        variable2="income", request_type="correlation_pair",
    )
    assert note is not None
    assert "income" in note


def test_note_checks_variable_before_variable2(tmp_path: Path) -> None:
    """When BOTH variable and variable2 have crossed a threshold on
    the same call, the note names ``variable`` (checked first) — this
    pins the documented precedence, not a specific behavior a
    researcher needs, but a stable contract for what the note says."""
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")
    note = note_for_new_request(
        tmp_path, dataset="d.csv", variable="income",
        variable2="age", request_type="quartiles",
    )
    assert note is not None
    assert "income" in note


def test_note_variable2_defaults_to_none_and_is_backward_compatible(
    tmp_path: Path,
) -> None:
    """Callers that never pass variable2 (single-variable request
    types) must see identical behavior to before this fix."""
    _write_rd(tmp_path, "d.csv", "income", "na_count")
    _write_rd(tmp_path, "d.csv", "income", "numeric_bounds")
    note = note_for_new_request(
        tmp_path, dataset="d.csv", variable="income",
        request_type="quartiles",
    )
    assert note is not None
    assert "income" in note


def test_note_never_raises_on_broken_ledger_dir(tmp_path: Path) -> None:
    blocker = tmp_path / ".sift"
    blocker.write_text("not a directory")
    note = note_for_new_request(tmp_path, dataset="d.csv", variable="income",
                                 request_type="na_count")
    assert note is None


# ---------------------------------------------------------------------------
# End-to-end through tools.request_data: real handler, real ledger
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "income": rng.normal(50000, 10000, size=n).round(2),
        "age": rng.integers(18, 80, size=n),
    })
    path = tmp_path / "synthetic.csv"
    df.to_csv(path, index=False)
    return path


def test_request_data_end_to_end_privacy_note_wiring(
    tmp_path: Path, sample_csv: Path,
) -> None:
    """Calling request_data three times for the same dataset+variable
    via the REAL @tool-decorated handler (which records to the REAL
    ledger after each call returns) must attach ``privacy_note`` only
    on the call that completes the pattern."""
    from sift.tools import HANDLERS

    async def _call(request_type: str) -> dict:
        return await HANDLERS["request_data"]({
            "dataset": "synthetic.csv",
            "variable": "income",
            "request_type": request_type,
        })

    with use_cwd(tmp_path):
        r1 = _mcp_text(asyncio.run(_call("na_count")))
        r2 = _mcp_text(asyncio.run(_call("numeric_bounds")))
        r3 = _mcp_text(asyncio.run(_call("quartiles")))

    assert r1["status"] == "granted", r1
    assert r2["status"] == "granted", r2
    assert r3["status"] == "granted", r3

    assert "privacy_note" not in r1
    assert "privacy_note" not in r2
    assert "privacy_note" in r3
    assert "income" in r3["privacy_note"]

    # And the ledger backing it is real and chain-verifies.
    ok, count, detail = release_ledger.verify_chain(tmp_path)
    assert ok and count == 3, detail


def test_request_data_denied_call_never_carries_a_note(
    tmp_path: Path, sample_csv: Path,
) -> None:
    """A denied call didn't disclose anything new, so it must never
    itself carry a privacy_note even if it's the Nth call."""
    from sift.tools import HANDLERS

    async def _call(request_type: str, variable: str = "income") -> dict:
        return await HANDLERS["request_data"]({
            "dataset": "synthetic.csv",
            "variable": variable,
            "request_type": request_type,
        })

    # An unsupported request type -> status "error", not "granted".
    with use_cwd(tmp_path):
        r = _mcp_text(asyncio.run(_call("not_a_real_request_type")))
    assert r["status"] != "granted"
    assert "privacy_note" not in r


@pytest.fixture
def sample_csv_multi(tmp_path: Path) -> Path:
    """Like ``sample_csv`` but with enough numeric columns to vary the
    PRIMARY variable of a correlation_pair call three times while
    holding ``variable2`` constant."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "income": rng.normal(50000, 10000, size=n).round(2),
        "age": rng.integers(18, 80, size=n),
        "height": rng.normal(170, 10, size=n).round(1),
        "weight": rng.normal(70, 15, size=n).round(1),
    })
    path = tmp_path / "synthetic_multi.csv"
    df.to_csv(path, index=False)
    return path


def test_correlation_pair_variable2_end_to_end_privacy_note_wiring(
    tmp_path: Path, sample_csv_multi: Path,
) -> None:
    """Real handler, real ledger: three correlation_pair calls, each
    pairing a DIFFERENT primary variable against the SAME variable2
    ("income"), must attach ``privacy_note`` on the third — via the
    real @tool-decorated handler and the real ledger it writes to,
    not a synthetic record. Before the variable2 fix, this pattern
    was invisible: "region", "age", and "height" (the primary
    variables) are each queried only once, and "income" — always in
    the variable2 slot — never accumulated a tally at all."""
    from sift.tools import HANDLERS

    async def _call(variable: str) -> dict:
        return await HANDLERS["request_data"]({
            "dataset": "synthetic_multi.csv",
            "variable": variable,
            "variable2": "income",
            "request_type": "correlation_pair",
        })

    with use_cwd(tmp_path):
        r1 = _mcp_text(asyncio.run(_call("age")))
        r2 = _mcp_text(asyncio.run(_call("height")))
        r3 = _mcp_text(asyncio.run(_call("weight")))

    assert r1["status"] == "granted", r1
    assert r2["status"] == "granted", r2
    assert r3["status"] == "granted", r3

    assert "privacy_note" not in r1
    assert "privacy_note" not in r2
    assert "privacy_note" in r3
    assert "income" in r3["privacy_note"]

    ok, count, detail = release_ledger.verify_chain(tmp_path)
    assert ok and count == 3, detail
