"""Differential privacy: scoped, honest opt-in noise for counts.

Covers, layer by layer:

1. ``differential_privacy.py``'s mechanism math (Laplace sampling,
   epsilon validation, the count wrapper) and its session-level
   epsilon-composition accounting (spend tracking, cap lookup, the
   would-exceed-budget check).
2. ``data_request._noisy_count`` (boundary-file wiring): opt-in-only
   denial, invalid-epsilon denial, a granted call's exact answer
   shape, and that the existing banned-variable gate still applies.
3. ``tools.py``'s live request_data path: the opt-in-required
   denial, a granted noisy_count response's fields, cumulative
   epsilon-cap enforcement across repeated calls (and that a denied-
   for-budget call spends nothing further), and that "public"
   (unbounded) never gets capped.
4. The real, pre-existing bug found and fixed while wiring this in:
   ``SiftBridge.set_dataset_policy`` / ``set_dataset_privacy_profile``
   silently wiped ``banned_variables`` / ``exportable`` / (now)
   ``dp_epsilon`` on every depth-chip or profile-chip edit. Pinned
   here as an explicit regression test alongside the new
   ``set_dataset_dp_epsilon`` bridge method's own tests.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sift import release_ledger
from sift.config import set_cwd
from sift.data_request import handle
from sift.differential_privacy import (
    MAX_EPSILON,
    MIN_EPSILON,
    EpsilonStatus,
    epsilon_cap_for_profile,
    epsilon_spent_for_dataset,
    epsilon_status_for_dataset,
    laplace_sample,
    noisy_count,
    validate_epsilon,
    would_exceed_budget,
)
from sift.policy import DatasetPolicy, SiftPolicy, load_policy, save_policy
from sift.sanitizer import DEFAULT_CONFIG
from sift.tools import request_data
from sift.ui import SiftBridge


def _mcp_text(payload: dict) -> dict:
    return json.loads(payload["content"][0]["text"])


# ---------------------------------------------------------------------------
# validate_epsilon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon,should_be_valid", [
    (MIN_EPSILON, True),
    (MAX_EPSILON, True),
    (1.0, True),
    (0.001, False),   # below min
    (100.0, False),   # above max
    (float("nan"), False),
    (float("inf"), False),
    (True, False),    # bool must not pass as a number
    ("0.5", False),   # string, not numeric
])
def test_validate_epsilon_boundaries(epsilon, should_be_valid):
    err = validate_epsilon(epsilon)
    assert (err is None) == should_be_valid


# ---------------------------------------------------------------------------
# laplace_sample / noisy_count: mechanism correctness
# ---------------------------------------------------------------------------


def test_laplace_sample_is_zero_mean_over_many_draws():
    rng = np.random.default_rng(1)
    draws = [laplace_sample(2.0, rng=rng) for _ in range(50_000)]
    mean = sum(draws) / len(draws)
    # Laplace(0, scale) has mean 0; over 50k draws the sample mean
    # should be small relative to the scale.
    assert abs(mean) < 0.1


def test_laplace_sample_rejects_nonpositive_scale():
    with pytest.raises(ValueError):
        laplace_sample(0.0)
    with pytest.raises(ValueError):
        laplace_sample(-1.0)


def test_noisy_count_converges_to_true_value_over_many_draws():
    rng = np.random.default_rng(2)
    reported = [noisy_count(1000, epsilon=1.0, rng=rng)[0] for _ in range(20_000)]
    mean = sum(reported) / len(reported)
    assert abs(mean - 1000) < 5  # well within sampling noise


def test_noisy_count_never_negative_even_for_tiny_true_count():
    rng = np.random.default_rng(3)
    # true_count=0 with a large noise scale (small epsilon) would
    # naively go negative about half the time without clamping.
    reported = [noisy_count(0, epsilon=MIN_EPSILON, rng=rng)[0] for _ in range(500)]
    assert all(r >= 0 for r in reported)


def test_noisy_count_rejects_invalid_epsilon():
    with pytest.raises(ValueError):
        noisy_count(100, epsilon=0.0001)
    with pytest.raises(ValueError):
        noisy_count(100, epsilon=50.0)


def test_noisy_count_smaller_epsilon_means_more_noise():
    """Sanity check on the privacy/utility tradeoff direction: a
    SMALLER epsilon (stronger privacy) must produce noise with a
    LARGER spread than a bigger epsilon, not the reverse — a flipped
    sign here would silently invert the entire privacy guarantee."""
    rng_a = np.random.default_rng(4)
    rng_b = np.random.default_rng(4)  # same seed, different scale
    strong_privacy = [noisy_count(1000, epsilon=0.05, rng=rng_a)[1]
                      for _ in range(2000)]
    weak_privacy = [noisy_count(1000, epsilon=5.0, rng=rng_b)[1]
                    for _ in range(2000)]
    spread_strong = np.std(strong_privacy)
    spread_weak = np.std(weak_privacy)
    assert spread_strong > spread_weak


# ---------------------------------------------------------------------------
# Epsilon budget: cap lookup, spend accounting, exceed check
# ---------------------------------------------------------------------------


def test_epsilon_cap_known_profiles():
    assert epsilon_cap_for_profile("regulated") == 1.0
    assert epsilon_cap_for_profile("confidential") == 3.0
    assert epsilon_cap_for_profile("internal") == 10.0
    assert epsilon_cap_for_profile("public") is None


def test_epsilon_cap_unknown_profile_falls_back_to_strictest():
    assert epsilon_cap_for_profile("bogus") == epsilon_cap_for_profile("regulated")


def test_epsilon_spent_counts_only_granted_noisy_count_for_dataset():
    records = [
        {"tool": "request_data",
         "args": {"dataset": "d.csv", "request_type": "noisy_count"},
         "facts": {"status": "granted", "epsilon": 0.3}},
        {"tool": "request_data",
         "args": {"dataset": "d.csv", "request_type": "noisy_count"},
         "facts": {"status": "denied", "epsilon": 0.9}},  # denied: not counted
        {"tool": "request_data",
         "args": {"dataset": "d.csv", "request_type": "na_count"},
         "facts": {"status": "granted"}},  # wrong request_type
        {"tool": "request_data",
         "args": {"dataset": "other.csv", "request_type": "noisy_count"},
         "facts": {"status": "granted", "epsilon": 0.4}},  # wrong dataset
        {"tool": "request_data",
         "args": {"dataset": "d.csv", "request_type": "noisy_count"},
         "facts": {"status": "granted", "epsilon": 0.2}},
    ]
    assert epsilon_spent_for_dataset(records, "d.csv") == pytest.approx(0.5)


def test_epsilon_spend_collapses_equivalent_dataset_path_spellings():
    records = [
        {"tool": "request_data",
         "args": {"dataset": path, "request_type": "noisy_count"},
         "facts": {"status": "granted", "epsilon": 0.2}}
        for path in ("./d.csv", "data/d.csv", r"data\d.csv")
    ]
    assert epsilon_spent_for_dataset(records, "d.csv") == pytest.approx(0.6)


def test_epsilon_spend_uses_canonical_live_dataset_identity(tmp_path: Path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    real = tmp_path / "a" / "data.csv"
    real.write_text("x\n1\n")
    (tmp_path / "b" / "data.csv").write_text("x\n2\n")
    (tmp_path / "alias.csv").symlink_to(real)
    response = {"content": [{"type": "text", "text": json.dumps({
        "status": "granted", "epsilon": 0.4,
    })}]}
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="request_data",
        args={"dataset": "alias.csv", "variable": "x",
              "request_type": "noisy_count"}, response=response,
    )
    records = release_ledger.read_ledger(tmp_path)
    assert epsilon_spent_for_dataset(
        records, "a/data.csv", cwd=tmp_path,
    ) == pytest.approx(0.4)
    assert epsilon_spent_for_dataset(
        records, "b/data.csv", cwd=tmp_path,
    ) == 0.0


def test_nonfinite_epsilon_cannot_poison_the_session_sum():
    records = [
        {"tool": "request_data",
         "args": {"dataset": "d.csv", "request_type": "noisy_count"},
         "facts": {"status": "granted", "epsilon": value}}
        for value in (float("nan"), float("inf"), -1.0, 0.4)
    ]
    assert epsilon_spent_for_dataset(records, "d.csv") == pytest.approx(0.4)


def test_would_exceed_budget_boundary():
    status = EpsilonStatus(dataset="d.csv", privacy_profile="regulated",
                           cap=1.0, spent=0.6)
    assert would_exceed_budget(status, 0.4) is False   # exactly at cap: OK
    assert would_exceed_budget(status, 0.41) is True   # just over: denied


def test_would_exceed_budget_never_true_when_unbounded():
    status = EpsilonStatus(dataset="d.csv", privacy_profile="public",
                           cap=None, spent=999.0)
    assert would_exceed_budget(status, 1000.0) is False


def test_epsilon_status_for_dataset_never_raises(tmp_path: Path):
    blocker = tmp_path / ".sift"
    blocker.write_text("not a directory")
    status = epsilon_status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.spent == 0.0
    assert status.cap == 1.0
    assert status.accounting_ok is False
    assert would_exceed_budget(status, 0.1) is True


def test_epsilon_status_fails_closed_on_known_recording_gap(tmp_path: Path):
    release_ledger._note_recording_failure(tmp_path, OSError("disk full"))
    status = epsilon_status_for_dataset(tmp_path, "d.csv", "regulated")
    assert status.accounting_ok is False
    assert "accounting gap" in status.accounting_detail
    assert would_exceed_budget(status, 0.1) is True


# ---------------------------------------------------------------------------
# data_request._noisy_count via handle(): boundary-file wiring
# ---------------------------------------------------------------------------


@pytest.fixture
def dp_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "income": rng.normal(50000, 10000, size=n).round(2),
        "ssn": [f"{i:09d}" for i in range(n)],
    })
    path = tmp_path / "dp.csv"
    df.to_csv(path, index=False)
    return path


def test_noisy_count_denied_without_opt_in(dp_csv: Path):
    result = handle(dp_csv, "noisy_count", "income", config=DEFAULT_CONFIG)
    assert result.status == "denied"
    assert "not enabled" in result.reason


def test_noisy_count_denied_for_invalid_epsilon(dp_csv: Path):
    from dataclasses import replace
    config = replace(DEFAULT_CONFIG, dp_epsilon=999.0)  # way above MAX_EPSILON
    result = handle(dp_csv, "noisy_count", "income", config=config)
    assert result.status == "denied"
    assert "invalid" in result.reason


def test_noisy_count_granted_with_valid_opt_in(dp_csv: Path):
    from dataclasses import replace
    config = replace(DEFAULT_CONFIG, dp_epsilon=1.0)
    result = handle(dp_csv, "noisy_count", "income", config=config)
    assert result.status == "granted"
    assert result.answer["mechanism"] == "laplace"
    assert result.answer["epsilon"] == 1.0
    assert result.answer["privacy_unit"] == "row"
    assert "person-level" in result.answer["note"]
    assert isinstance(result.answer["noisy_non_na_count"], int)
    # Never exactly leaks the true count's exact computation path —
    # can't assert inequality (noise CAN coincidentally match), but
    # must be in a sane ballpark for a 300-row column with epsilon=1.
    assert 250 <= result.answer["noisy_non_na_count"] <= 350


def test_noisy_count_still_respects_banned_variables(dp_csv: Path):
    from dataclasses import replace
    config = replace(DEFAULT_CONFIG, dp_epsilon=1.0,
                     banned_variables=frozenset({"ssn"}))
    result = handle(dp_csv, "noisy_count", "ssn", config=config)
    assert result.status == "denied"


def test_noisy_count_is_in_supported_request_types():
    from sift.data_request import SUPPORTED_REQUEST_TYPES
    assert "noisy_count" in SUPPORTED_REQUEST_TYPES


# ---------------------------------------------------------------------------
# tools.py: live request_data path, cap enforcement across calls
# ---------------------------------------------------------------------------


@pytest.fixture
def regulated_dp_dataset(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"income": rng.normal(50000, 10000, size=200).round(2)})
    path = tmp_path / "sensitive.csv"
    df.to_csv(path, index=False)
    save_policy(tmp_path, SiftPolicy(datasets={
        "sensitive.csv": DatasetPolicy(
            privacy_profile="regulated", dp_epsilon=0.8,
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    return path


def test_request_data_noisy_count_denied_without_policy_opt_in(tmp_path: Path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"income": rng.normal(50000, 10000, size=200)})
    (tmp_path / "plain.csv").write_text(df.to_csv(index=False))
    set_cwd(tmp_path)
    r = _mcp_text(asyncio.run(request_data.handler({
        "dataset": "plain.csv", "variable": "income",
        "request_type": "noisy_count",
    })))
    assert r["status"] == "denied"
    assert "epsilon" not in r


def test_request_data_noisy_count_granted_carries_epsilon_and_note(
    tmp_path: Path, regulated_dp_dataset: Path,
):
    set_cwd(tmp_path)
    r = _mcp_text(asyncio.run(request_data.handler({
        "dataset": "sensitive.csv", "variable": "income",
        "request_type": "noisy_count",
    })))
    assert r["status"] == "granted"
    assert r["epsilon"] == 0.8
    assert "epsilon_budget_note" in r
    assert "0.200" in r["epsilon_budget_note"]  # 1.0 cap - 0.8 spent


def test_request_data_noisy_count_cap_enforced_across_calls(
    tmp_path: Path, regulated_dp_dataset: Path,
):
    """First call spends 0.8 of the 1.0 regulated cap. A second call
    at the same epsilon (0.8) would push cumulative spend to 1.6 >
    1.0, so it must be denied BEFORE data_request.handle runs (no
    further epsilon spent, no true-count computation disclosed)."""
    set_cwd(tmp_path)

    async def _call():
        return await request_data.handler({
            "dataset": "sensitive.csv", "variable": "income",
            "request_type": "noisy_count",
        })

    r1 = _mcp_text(asyncio.run(_call()))
    assert r1["status"] == "granted"

    r2 = _mcp_text(asyncio.run(_call()))
    assert r2["status"] == "denied"
    assert "epsilon budget" in r2["reason"]
    assert "epsilon" not in r2  # nothing further was spent

    # Ledger reflects exactly one granted noisy_count spend, not two.
    from sift.differential_privacy import epsilon_spent_for_dataset
    records = release_ledger.read_ledger(tmp_path)
    assert epsilon_spent_for_dataset(records, "sensitive.csv") == pytest.approx(0.8)


def test_concurrent_noisy_counts_cannot_race_past_cap(
    tmp_path: Path, regulated_dp_dataset: Path,
):
    set_cwd(tmp_path)

    def _call() -> dict:
        return _mcp_text(asyncio.run(request_data.handler({
            "dataset": "sensitive.csv", "variable": "income",
            "request_type": "noisy_count",
        })))

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _i: _call(), range(2)))

    assert sorted(r["status"] for r in results) == ["denied", "granted"]
    records = release_ledger.read_ledger(tmp_path)
    assert epsilon_spent_for_dataset(records, "sensitive.csv") == pytest.approx(0.8)


def test_granted_noisy_count_is_withheld_if_spend_append_fails(
    tmp_path: Path, regulated_dp_dataset: Path, monkeypatch: pytest.MonkeyPatch,
):
    set_cwd(tmp_path)
    monkeypatch.setattr(release_ledger, "record_release", lambda *a, **k: False)
    response = _mcp_text(asyncio.run(request_data.handler({
        "dataset": "sensitive.csv", "variable": "income",
        "request_type": "noisy_count",
    })))
    assert response["status"] == "denied"
    assert "could not be durably recorded" in response["reason"]
    assert "answer" not in response


def test_request_data_noisy_count_unbounded_for_public_profile(tmp_path: Path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"income": rng.normal(50000, 10000, size=200)})
    (tmp_path / "open.csv").write_text(df.to_csv(index=False))
    save_policy(tmp_path, SiftPolicy(datasets={
        "open.csv": DatasetPolicy(
            privacy_profile="public", dp_epsilon=2.0,
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    set_cwd(tmp_path)

    async def _call():
        return await request_data.handler({
            "dataset": "open.csv", "variable": "income",
            "request_type": "noisy_count",
        })

    for _ in range(10):  # would blow any finite cap many times over
        r = _mcp_text(asyncio.run(_call()))
        assert r["status"] == "granted"
        assert "epsilon_budget_note" not in r  # unbounded -> no note


# ---------------------------------------------------------------------------
# The pre-existing bug found while wiring dp_epsilon: policy setters
# silently wiped banned_variables / exportable on unrelated edits
# ---------------------------------------------------------------------------


def test_set_dataset_policy_preserves_banned_and_exportable_and_dp_epsilon(
    tmp_path: Path,
):
    save_policy(tmp_path, SiftPolicy(datasets={
        "x.csv": DatasetPolicy(
            banned_variables=("ssn",), exportable=False, dp_epsilon=0.3,
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path

    result = bridge.set_dataset_policy("x.csv", "names_types_labels")
    assert result["ok"]

    reloaded = load_policy(tmp_path)
    entry = reloaded.datasets["x.csv"]
    assert entry.banned_variables == ("ssn",)
    assert entry.exportable is False
    assert entry.dp_epsilon == 0.3


def test_set_dataset_privacy_profile_preserves_banned_and_exportable_and_dp_epsilon(
    tmp_path: Path,
):
    save_policy(tmp_path, SiftPolicy(datasets={
        "x.csv": DatasetPolicy(
            banned_variables=("ssn",), exportable=False, dp_epsilon=0.3,
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path

    result = bridge.set_dataset_privacy_profile("x.csv", "confidential")
    assert result["ok"]

    reloaded = load_policy(tmp_path)
    entry = reloaded.datasets["x.csv"]
    assert entry.banned_variables == ("ssn",)
    assert entry.exportable is False
    assert entry.dp_epsilon == 0.3
    assert entry.privacy_profile == "confidential"


# ---------------------------------------------------------------------------
# set_dataset_dp_epsilon / get_epsilon_budget_status bridge methods
# ---------------------------------------------------------------------------


def test_set_dataset_dp_epsilon_valid(tmp_path: Path):
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_dp_epsilon("x.csv", 0.7)
    assert result["ok"]
    entry = load_policy(tmp_path).datasets["x.csv"]
    assert entry.dp_epsilon == 0.7


def test_set_dataset_dp_epsilon_rejects_out_of_range(tmp_path: Path):
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_dp_epsilon("x.csv", 500.0)
    assert not result["ok"]
    assert "x.csv" not in load_policy(tmp_path).datasets


def test_set_dataset_dp_epsilon_preserves_other_axes(tmp_path: Path):
    save_policy(tmp_path, SiftPolicy(datasets={
        "x.csv": DatasetPolicy(
            banned_variables=("ssn",), privacy_profile="regulated",
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_dp_epsilon("x.csv", 0.5)
    assert result["ok"]
    entry = load_policy(tmp_path).datasets["x.csv"]
    assert entry.banned_variables == ("ssn",)
    assert entry.privacy_profile == "regulated"
    assert entry.dp_epsilon == 0.5


def test_set_dataset_dp_epsilon_none_clears_and_can_collapse_entry(
    tmp_path: Path,
):
    save_policy(tmp_path, SiftPolicy(datasets={
        "x.csv": DatasetPolicy(dp_epsilon=0.5,
                               set_at="2026-01-01T00:00:00+00:00"),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_dp_epsilon("x.csv", None)
    assert result["ok"]
    reloaded = load_policy(tmp_path)
    # Every axis was at default except dp_epsilon; clearing it should
    # collapse the entry entirely (matching set_dataset_policy's own
    # collapse-when-nothing-left-explicit rule).
    assert "x.csv" not in reloaded.datasets


def test_get_epsilon_budget_status_reports_only_opted_in_datasets(
    tmp_path: Path,
):
    rng = np.random.default_rng(0)
    for name in ("opted_in.csv", "not_opted_in.csv"):
        pd.DataFrame({"x": rng.normal(size=50)}).to_csv(
            tmp_path / name, index=False,
        )
    save_policy(tmp_path, SiftPolicy(datasets={
        "opted_in.csv": DatasetPolicy(
            dp_epsilon=0.4, privacy_profile="confidential",
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    status = bridge.get_epsilon_budget_status()
    names = {d["name"] for d in status["datasets"]}
    assert "opted_in.csv" in names
    assert "not_opted_in.csv" not in names
    entry = next(d for d in status["datasets"] if d["name"] == "opted_in.csv")
    assert entry["cap"] == 3.0
    assert entry["spent"] == 0.0
