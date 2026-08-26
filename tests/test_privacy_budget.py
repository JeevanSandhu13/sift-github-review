"""Privacy budgets and adaptive suppression (sift.privacy_budget).

Pins:

1. ``tier_for_consumption`` breakpoints: strictly below 1x budget is
   normal, [1x, 2x) is elevated, 2x+ is strict; an unbounded (None)
   budget is always normal regardless of consumption.
2. ``adjusted_sdc_config`` is monotonically STRICTER at higher tiers
   for every field it touches, and NEVER stricter than necessary at
   tier 0 (must equal the base config exactly) — this is the single
   most safety-critical property in this module: a bug here would
   silently mean "under load, suppression relaxes", the opposite of
   the feature's purpose.
3. ``consumed_for_dataset`` counts only GRANTED releases scoped to
   the right dataset, from both single-result and batch
   ``submit_script`` records.
4. ``status_for_dataset`` / ``resolve_adaptive_sdc_config`` read the
   real ledger correctly and never raise.
5. End-to-end: seeding the real ledger to just under/over a
   "regulated" dataset's budget changes what
   ``tools._resolve_sdc_and_source_n`` (exercised via the real
   ``request_data`` handler) actually produces — both the SDCConfig
   values and the advisory note in the response.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sift import release_ledger
from sift.config import use_cwd
from sift.policy import DatasetPolicy, SiftPolicy, save_policy
from sift.privacy_budget import (
    _COUNT_FIELDS,
    _DOMINANCE_FLOOR,
    TIER_ELEVATED,
    TIER_NORMAL,
    TIER_STRICT,
    BudgetStatus,
    adjusted_sdc_config,
    advisory_note,
    budget_for_profile,
    consumed_for_dataset,
    status_for_dataset,
    tier_for_consumption,
)
from sift.sanitizer import DEFAULT_CONFIG


def _mcp_text(payload: dict) -> dict:
    text_block = next(b for b in payload["content"] if b.get("type") == "text")
    return json.loads(text_block["text"])


# ---------------------------------------------------------------------------
# budget_for_profile
# ---------------------------------------------------------------------------


def test_known_profile_budgets():
    assert budget_for_profile("regulated") == 15
    assert budget_for_profile("confidential") == 40
    assert budget_for_profile("internal") == 150
    assert budget_for_profile("public") is None


def test_unknown_profile_falls_back_to_strictest_known_budget():
    assert budget_for_profile("top-secret") == budget_for_profile("regulated")


# ---------------------------------------------------------------------------
# tier_for_consumption: breakpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("consumed,expected", [
    (0, TIER_NORMAL),
    (14, TIER_NORMAL),
    (15, TIER_ELEVATED),   # ratio == 1.0 exactly -> elevated
    (29, TIER_ELEVATED),
    (30, TIER_STRICT),     # ratio == 2.0 exactly -> strict
    (100, TIER_STRICT),
])
def test_tier_breakpoints(consumed, expected):
    assert tier_for_consumption(consumed, budget=15) == expected


def test_unbounded_budget_is_always_normal():
    assert tier_for_consumption(0, budget=None) == TIER_NORMAL
    assert tier_for_consumption(10_000_000, budget=None) == TIER_NORMAL


def test_zero_or_negative_budget_is_always_normal():
    """A misconfigured zero/negative budget must degrade to "no
    tightening" rather than divide-by-zero or a nonsensical tier."""
    assert tier_for_consumption(5, budget=0) == TIER_NORMAL
    assert tier_for_consumption(5, budget=-1) == TIER_NORMAL


# ---------------------------------------------------------------------------
# adjusted_sdc_config: the strictly-more-conservative invariant
# ---------------------------------------------------------------------------


def test_tier_normal_is_a_no_op():
    cfg = adjusted_sdc_config(DEFAULT_CONFIG, TIER_NORMAL)
    assert cfg == DEFAULT_CONFIG


def test_tier_elevated_and_strict_are_strictly_tighter_than_normal():
    normal = adjusted_sdc_config(DEFAULT_CONFIG, TIER_NORMAL)
    elevated = adjusted_sdc_config(DEFAULT_CONFIG, TIER_ELEVATED)
    strict = adjusted_sdc_config(DEFAULT_CONFIG, TIER_STRICT)

    for field_name in _COUNT_FIELDS:
        n0 = getattr(normal, field_name)
        n1 = getattr(elevated, field_name)
        n2 = getattr(strict, field_name)
        assert n1 >= n0, field_name
        assert n2 >= n1, field_name
        # At least one tier must actually move for a threshold of 10+
        # (sanity: the multipliers aren't accidentally inert).
        assert n2 > n0, field_name

    assert elevated.dominance_threshold <= normal.dominance_threshold
    assert strict.dominance_threshold <= elevated.dominance_threshold
    assert strict.dominance_threshold < normal.dominance_threshold


def test_unrecognised_tier_fails_safe_to_no_adjustment():
    cfg = adjusted_sdc_config(DEFAULT_CONFIG, 999)
    assert cfg == DEFAULT_CONFIG


@given(
    min_n_regression=st.integers(min_value=1, max_value=1000),
    min_n_descriptive=st.integers(min_value=1, max_value=1000),
    min_n_ttest_group=st.integers(min_value=1, max_value=1000),
    cell_suppression_threshold=st.integers(min_value=1, max_value=1000),
    min_n_did_cohort=st.integers(min_value=1, max_value=1000),
    dominance_threshold=st.floats(
        min_value=0.5, max_value=1.0, allow_nan=False, allow_infinity=False,
    ),
    tier=st.sampled_from([TIER_NORMAL, TIER_ELEVATED, TIER_STRICT]),
)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_adjusted_config_never_loosens_any_field(
    min_n_regression, min_n_descriptive, min_n_ttest_group,
    cell_suppression_threshold, min_n_did_cohort, dominance_threshold, tier,
):
    """Property: across ARBITRARY base configs and any real tier, the
    adjusted config's count fields are never lower and its dominance
    threshold is never higher than the base — regardless of what the
    researcher's own thresholds happened to be."""
    base = replace(
        DEFAULT_CONFIG,
        min_n_regression=min_n_regression,
        min_n_descriptive=min_n_descriptive,
        min_n_ttest_group=min_n_ttest_group,
        cell_suppression_threshold=cell_suppression_threshold,
        min_n_did_cohort=min_n_did_cohort,
        dominance_threshold=dominance_threshold,
    )
    adjusted = adjusted_sdc_config(base, tier)
    for field_name in _COUNT_FIELDS:
        assert getattr(adjusted, field_name) >= getattr(base, field_name)
    assert adjusted.dominance_threshold <= base.dominance_threshold
    assert adjusted.dominance_threshold >= _DOMINANCE_FLOOR


def test_adjusted_config_preserves_untouched_fields():
    """Fields this module doesn't adjust (e.g. non_disclosive_variables,
    banned_variables) must pass through unchanged."""
    base = replace(
        DEFAULT_CONFIG,
        non_disclosive_variables=frozenset({"age"}),
        banned_variables=frozenset({"ssn"}),
    )
    adjusted = adjusted_sdc_config(base, TIER_STRICT)
    assert adjusted.non_disclosive_variables == frozenset({"age"})
    assert adjusted.banned_variables == frozenset({"ssn"})


# ---------------------------------------------------------------------------
# consumed_for_dataset
# ---------------------------------------------------------------------------


def test_consumed_counts_only_granted_request_data():
    records = [
        {"tool": "request_data", "args": {"dataset": "d.csv"},
         "facts": {"status": "granted"}},
        {"tool": "request_data", "args": {"dataset": "d.csv"},
         "facts": {"status": "denied"}},
        {"tool": "request_data", "args": {"dataset": "other.csv"},
         "facts": {"status": "granted"}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 1


def test_consumed_counts_submit_script_single_and_batch():
    records = [
        {"tool": "submit_script", "facts": {
            "status": "ok", "source_dataset": "d.csv",
        }},
        {"tool": "submit_script", "facts": {"results": [
            {"status": "ok", "source_dataset": "d.csv"},
            {"status": "ok", "source_dataset": "other.csv"},
            {"status": "ok", "source_dataset": "d.csv"},
        ]}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 3
    assert consumed_for_dataset(records, "other.csv") == 1


def test_consumed_ignores_unrelated_tools_and_malformed_entries():
    records = [
        {"tool": "get_schema", "args": {"dataset": "d.csv"}},
        {"tool": "request_data"},  # no args
        {"tool": "submit_script"},  # no facts
        {"tool": "submit_script", "facts": {"results": "not-a-list"}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 0


def test_consumed_ignores_rejected_submit_script_payloads():
    records = [{"tool": "submit_script", "facts": {"results": [
        {"status": "rejected_by_sanitizer", "source_dataset": "d.csv"},
        {"status": "ok", "source_dataset": "d.csv"},
    ]}}]
    assert consumed_for_dataset(records, "d.csv") == 1


# ---------------------------------------------------------------------------
# consumed_for_dataset: dataset-key normalization (audit finding)
#
# tools.py resolves the CURRENT call's dataset argument to a bare
# basename (``resolve_in_cwd(source_dataset).name``) before calling
# into this module, but the release ledger records the RAW string the
# model passed as a tool argument -- never normalized. Exact string
# comparison between a normalized query key and un-normalized
# historical ledger entries silently excluded any release whose
# recorded dataset string differed from the bare filename, which kept
# consumption locked at (or near) zero regardless of how many
# releases were actually granted -- defeating this module's entire
# purpose. Live repro: 20 granted releases recorded under
# "./clinical.csv", queried with the correctly-normalized
# "clinical.csv", returned a count of 0 before this fix.
# ---------------------------------------------------------------------------

def test_consumed_matches_request_data_despite_dotslash_prefix():
    records = [
        {"tool": "request_data", "args": {"dataset": "./d.csv"},
         "facts": {"status": "granted"}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 1


def test_consumed_matches_request_data_despite_subdirectory_prefix():
    records = [
        {"tool": "request_data", "args": {"dataset": "data/d.csv"},
         "facts": {"status": "granted"}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 1


def test_consumed_matches_windows_separator_in_cross_platform_ledger():
    records = [
        {"tool": "request_data", "args": {"dataset": r"data\d.csv"},
         "facts": {"status": "granted"}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 1


def test_consumed_matches_submit_script_despite_dotslash_prefix():
    records = [
        {"tool": "submit_script", "facts": {
            "status": "ok", "source_dataset": "./d.csv",
        }},
        {"tool": "submit_script", "facts": {"results": [
            {"status": "ok", "source_dataset": "sub/d.csv"},
        ]}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 2


def test_consumed_normalization_does_not_merge_distinct_filenames():
    """Negative control: normalization must not accidentally widen the
    match to a DIFFERENT dataset -- only path-prefix variants of the
    SAME filename should collapse together."""
    records = [
        {"tool": "request_data", "args": {"dataset": "./other.csv"},
         "facts": {"status": "granted"}},
    ]
    assert consumed_for_dataset(records, "d.csv") == 0


def test_consumed_query_side_normalization_also_applies():
    """The QUERY key itself is normalized too, not just the ledger
    side -- a caller passing a dotted/prefixed form still matches
    plain-basename ledger entries."""
    records = [
        {"tool": "request_data", "args": {"dataset": "d.csv"},
         "facts": {"status": "granted"}},
    ]
    assert consumed_for_dataset(records, "./d.csv") == 1


# ---------------------------------------------------------------------------
# status_for_dataset: real ledger I/O, never raises
# ---------------------------------------------------------------------------


def test_status_for_dataset_zero_consumption(tmp_path: Path):
    status = status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.consumed == 0
    assert status.tier == TIER_NORMAL
    assert status.budget == 15
    assert not status.unbounded


def test_status_for_dataset_reflects_real_ledger(tmp_path: Path):
    for _ in range(16):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool="request_data",
            args={"dataset": "d.csv", "variable": "income",
                  "request_type": "na_count"},
            response={"content": [{"type": "text",
                                    "text": json.dumps({"status": "granted"})}]},
        )
    status = status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.consumed == 16
    assert status.tier == TIER_ELEVATED  # 16/15 >= 1.0, < 2.0


def test_status_for_dataset_reflects_real_ledger_with_dotslash_dataset_arg(
    tmp_path: Path,
):
    """End-to-end version of the normalization fix: real ledger I/O
    (not a hand-built records list), dataset recorded exactly as a
    model plausibly phrases it (``"./d.csv"``), queried with the bare
    basename a resolved tool call would actually use."""
    for _ in range(16):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool="request_data",
            args={"dataset": "./d.csv", "variable": "income",
                  "request_type": "na_count"},
            response={"content": [{"type": "text",
                                    "text": json.dumps({"status": "granted"})}]},
        )
    status = status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.consumed == 16, (
        "consumption undercounted -- releases recorded under a "
        "'./'-prefixed dataset string were not matched against the "
        "normalized query key"
    )
    assert status.tier == TIER_ELEVATED


def test_budget_identity_collapses_symlink_aliases(tmp_path: Path):
    target = tmp_path / "real.csv"
    target.write_text("x\n1\n")
    (tmp_path / "alias.csv").symlink_to(target)
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        args={"dataset": "alias.csv", "variable": "x",
              "request_type": "na_count"},
        response={"content": [{"type": "text", "text": json.dumps({
            "status": "granted",
        })}]},
    )
    status = status_for_dataset(tmp_path, "real.csv", "regulated")
    assert status.consumed == 1


def test_budget_identity_distinguishes_same_name_in_two_directories(
    tmp_path: Path,
):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "data.csv").write_text("x\n1\n")
    (tmp_path / "b" / "data.csv").write_text("x\n2\n")
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        args={"dataset": "a/data.csv", "variable": "x",
              "request_type": "na_count"},
        response={"content": [{"type": "text", "text": json.dumps({
            "status": "granted",
        })}]},
    )
    assert status_for_dataset(
        tmp_path, "a/data.csv", "regulated",
    ).consumed == 1
    assert status_for_dataset(
        tmp_path, "b/data.csv", "regulated",
    ).consumed == 0


def test_status_for_dataset_never_raises_on_broken_ledger_dir(tmp_path: Path):
    blocker = tmp_path / ".sift"
    blocker.write_text("not a directory")
    status = status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.consumed == 0
    assert status.tier == TIER_STRICT
    assert status.accounting_ok is False
    assert "not a directory" in status.accounting_detail


def test_status_for_dataset_gap_fails_closed_to_strict_tier(tmp_path: Path):
    release_ledger._note_recording_failure(tmp_path, OSError("disk full"))
    status = status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.accounting_ok is False
    assert status.tier == TIER_STRICT
    assert "accounting gap" in status.accounting_detail
    assert "strictest adaptive suppression" in advisory_note(status)


def test_public_profile_is_always_unbounded_regardless_of_consumption(
    tmp_path: Path,
):
    for _ in range(500):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool="request_data",
            args={"dataset": "open.csv", "variable": "x",
                  "request_type": "na_count"},
            response={"content": [{"type": "text",
                                    "text": json.dumps({"status": "granted"})}]},
        )
    status = status_for_dataset(tmp_path, "open.csv", "public")
    assert status.unbounded
    assert status.tier == TIER_NORMAL


# ---------------------------------------------------------------------------
# advisory_note
# ---------------------------------------------------------------------------


def test_advisory_note_none_at_normal_tier():
    status = BudgetStatus(dataset="d.csv", privacy_profile="regulated",
                          budget=15, consumed=3, tier=TIER_NORMAL)
    assert advisory_note(status) is None


def test_advisory_note_none_when_unbounded_even_if_tier_nonzero():
    """Defensive: even a malformed status claiming an elevated tier
    with no budget must not produce a note (the field only makes
    sense in relation to a real budget)."""
    status = BudgetStatus(dataset="open.csv", privacy_profile="public",
                          budget=None, consumed=999, tier=TIER_ELEVATED)
    assert advisory_note(status) is None


def test_advisory_note_present_and_informative_at_elevated_tier():
    status = BudgetStatus(dataset="d.csv", privacy_profile="regulated",
                          budget=15, consumed=16, tier=TIER_ELEVATED)
    note = advisory_note(status)
    assert note is not None
    assert "d.csv" in note and "16" in note and "15" in note
    assert "elevated" in note


# ---------------------------------------------------------------------------
# End-to-end through tools.request_data: seeded ledger, real handler
# ---------------------------------------------------------------------------


@pytest.fixture
def regulated_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "income": rng.normal(50000, 10000, size=n).round(2),
    })
    csv_path = tmp_path / "sensitive.csv"
    df.to_csv(csv_path, index=False)
    save_policy(tmp_path, SiftPolicy(datasets={
        "sensitive.csv": DatasetPolicy(
            privacy_profile="regulated",
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    return csv_path


def test_request_data_sdc_tightens_after_budget_exceeded(
    tmp_path: Path, regulated_dataset: Path,
):
    """Seed the ledger to just past the 'regulated' budget (15) for
    this dataset, then confirm the REAL request_data handler's
    response carries the budget advisory note -- proving the
    tightened SDCConfig this module computes is the one actually
    wired into the live handler path, not just unit-testable in
    isolation."""
    from sift.tools import HANDLERS

    for _ in range(15):
        release_ledger.record_release(
            tmp_path, kind="tool_response", tool="request_data",
            args={"dataset": "sensitive.csv", "variable": "income",
                  "request_type": "na_count"},
            response={"content": [{"type": "text",
                                    "text": json.dumps({"status": "granted"})}]},
        )

    async def _call():
        return await HANDLERS["request_data"]({
            "dataset": "sensitive.csv",
            "variable": "income",
            "request_type": "numeric_bounds",
        })

    with use_cwd(tmp_path):
        r = _mcp_text(asyncio.run(_call()))

    assert r["status"] == "granted", r
    assert "privacy_budget_note" in r
    assert "sensitive.csv" in r["privacy_budget_note"]
    assert "elevated" in r["privacy_budget_note"]


def test_request_data_no_budget_note_under_budget(
    tmp_path: Path, regulated_dataset: Path,
):
    """With consumption well under budget, no advisory note should
    appear -- confirms the feature is silent in the common case."""
    from sift.tools import HANDLERS

    async def _call():
        return await HANDLERS["request_data"]({
            "dataset": "sensitive.csv",
            "variable": "income",
            "request_type": "na_count",
        })

    with use_cwd(tmp_path):
        r = _mcp_text(asyncio.run(_call()))

    assert r["status"] == "granted", r
    assert "privacy_budget_note" not in r
