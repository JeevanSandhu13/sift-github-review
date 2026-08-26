"""Session usage meter — exactness for tokens, honesty for money.

The distinction this module exists to preserve: token counts are
measured and never go stale; costs are derived from published rates
that change. The tests below pin the honesty rules — an unknown rate
must yield "unavailable", never a zero that reads as "free", and a
provider-reported figure must never be conflated with an estimate.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

from sift import usage_meter
from sift.usage_meter import estimate_cost_usd, read_usage, record_turn, summarize


def test_tokens_accumulate_exactly(tmp_path: Path) -> None:
    for _ in range(3):
        record_turn(
            tmp_path,
            model="claude-sonnet-5",
            provider="anthropic",
            input_tokens=1000,
            output_tokens=200,
            cache_read_tokens=50,
            cache_creation_tokens=10,
        )
    s = summarize(tmp_path)
    assert s["turns"] == 3
    assert s["input_tokens"] == 3000
    assert s["output_tokens"] == 600
    assert s["total_tokens"] == 3 * (1000 + 200 + 50 + 10)


def test_cost_estimate_matches_published_rates() -> None:
    # Sonnet 5 promotional pricing through 2026-08-31: $2/M + $10/M.
    cost = estimate_cost_usd(
        "claude-sonnet-5-20260101", input_tokens=1_000_000, output_tokens=100_000
    )
    assert cost == 3.0


def test_current_provider_rates_and_cache_tiers_are_exact() -> None:
    assert estimate_cost_usd(
        "claude-opus-5", input_tokens=1_000_000, output_tokens=1_000_000,
    ) == 30.0
    assert estimate_cost_usd(
        "claude-fable-5", input_tokens=1_000_000, output_tokens=1_000_000,
    ) == 60.0
    assert estimate_cost_usd(
        "gpt-5.6-terra", input_tokens=1_000_000, output_tokens=1_000_000,
    ) == 14.0
    assert estimate_cost_usd(
        "claude-sonnet-5", cache_creation_tokens=1_000_000,
    ) == 4.0


def test_cost_estimate_covers_gemini_models() -> None:
    """Gemini is a native provider and its rate
    table was never updated to match, so every Gemini session
    estimated an unknown ($0/None) cost regardless of real spend --
    the exact "fabricated zero reads as free" failure this file's
    own docstring warns against, just via omission rather than a
    literal zero. Pins both catalog models against the rates
    documented in provider/catalog.py's GEMINI_MODELS comment."""
    flash_cost = estimate_cost_usd(
        "gemini-3.7-flash",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert flash_cost == 0.75 + 3.75

    pro_cost = estimate_cost_usd(
        "gemini-3.1-pro-preview",
        input_tokens=200_000,
        output_tokens=1_000_000,
    )
    assert pro_cost == 0.4 + 12.0

    long_pro_cost = estimate_cost_usd(
        "gemini-3.1-pro-preview",
        input_tokens=200_001,
        output_tokens=1_000_000,
    )
    assert long_pro_cost == 18.800004

    # Substring matching against a real, dated model id (mirrors how
    # Anthropic/OpenAI ids are matched elsewhere in this file) --
    # confirms the rate table's key is actually reachable through the
    # same matching path a live session would use, not just an exact
    # dict lookup.
    dated_cost = estimate_cost_usd(
        "gemini-3.7-flash-001",
        input_tokens=1_000_000,
    )
    assert dated_cost == 0.75


def test_gemini_session_end_to_end_reports_a_real_estimate(
    tmp_path: Path,
) -> None:
    """Full path through record_turn -> summarize, the same as every
    other provider gets exercised: a Gemini session must show an
    actual dollar estimate, not 'unavailable'."""
    record_turn(
        tmp_path,
        model="gemini-3.7-flash",
        provider="gemini",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    s = summarize(tmp_path)
    assert s["estimated_cost_usd"] == 0.75 + 3.75
    assert s["complete"] is True


def test_unknown_model_yields_none_never_zero() -> None:
    """A fabricated $0.00 would read as 'this was free'."""
    assert estimate_cost_usd("some-unreleased-model", input_tokens=5_000_000) is None
    assert estimate_cost_usd(None, input_tokens=1000) is None


def test_summary_reports_unavailable_cost_for_unknown_models(tmp_path: Path) -> None:
    record_turn(
        tmp_path, model="mystery-model-9", input_tokens=10_000, output_tokens=1_000
    )
    s = summarize(tmp_path)
    assert s["total_tokens"] == 11_000  # tokens still exact
    assert s["estimated_cost_usd"] is None  # cost withheld
    assert s["by_model"][0]["estimated_cost_usd"] is None


def test_partial_rate_coverage_is_flagged(tmp_path: Path) -> None:
    record_turn(tmp_path, model="claude-sonnet-5", input_tokens=1_000_000)
    record_turn(tmp_path, model="mystery-model-9", input_tokens=1_000_000)
    s = summarize(tmp_path)
    assert s["estimated_cost_usd"] is not None  # partial estimate exists
    assert s["complete"] is False  # and is marked partial


def test_reported_cost_kept_separate_from_estimate(tmp_path: Path) -> None:
    """A provider-measured figure must not be blended into the
    rate-table estimate — they have different epistemic status."""
    record_turn(
        tmp_path,
        model="claude-sonnet-5",
        input_tokens=1000,
        output_tokens=100,
        reported_cost_usd=0.0125,
    )
    s = summarize(tmp_path)
    assert s["reported_cost_usd"] == 0.0125
    assert s["reported_cost_turns"] == 1
    assert s["estimated_cost_usd"] is not None
    assert s["estimated_cost_usd"] != s["reported_cost_usd"]


def test_per_model_breakdown(tmp_path: Path) -> None:
    record_turn(
        tmp_path,
        model="claude-opus-5",
        provider="anthropic",
        input_tokens=5000,
        output_tokens=500,
    )
    record_turn(
        tmp_path,
        model="claude-sonnet-5",
        provider="anthropic",
        input_tokens=1000,
        output_tokens=100,
    )
    rows = {r["model"]: r for r in summarize(tmp_path)["by_model"]}
    assert set(rows) == {"claude-opus-5", "claude-sonnet-5"}
    assert rows["claude-opus-5"]["tokens"] == 5500
    # Opus is priced above Sonnet, so its estimate must be larger.
    assert (
        rows["claude-opus-5"]["estimated_cost_usd"]
        > rows["claude-sonnet-5"]["estimated_cost_usd"]
    )


def test_rates_carry_an_as_of_date(tmp_path: Path) -> None:
    """Any displayed cost must be attributable to a dated rate table."""
    assert usage_meter.RATES_AS_OF
    assert summarize(tmp_path)["rates_as_of"] == usage_meter.RATES_AS_OF


def test_recording_never_raises_and_ignores_empty_turns(tmp_path: Path) -> None:
    record_turn(None, model="x", input_tokens=10)
    record_turn(tmp_path / "missing", model="x", input_tokens=10)
    record_turn(tmp_path, model="x", input_tokens=0, output_tokens=0)
    assert read_usage(tmp_path) == {}  # nothing written
    assert summarize(tmp_path)["turns"] == 0


def test_corrupt_usage_file_degrades_quietly(tmp_path: Path) -> None:
    (tmp_path / ".sift").mkdir()
    usage_meter.usage_path(tmp_path).write_text("{not json")
    assert read_usage(tmp_path) == {}
    # A later turn must not overwrite corrupt history and silently reset the
    # apparent total; it is withheld and a durable gap marker is emitted.
    record_turn(tmp_path, model="claude-sonnet-5", input_tokens=100)
    summary = summarize(tmp_path)
    assert summary["turns"] == 0
    assert summary["complete"] is False
    assert summary["usage_accounting_complete"] is False
    assert summary["unrecorded_turns"] == 1
    assert usage_meter._usage_health_path(tmp_path).is_file()


def test_repaired_meter_acknowledges_prior_gap_without_hiding_it(
    tmp_path: Path,
) -> None:
    (tmp_path / ".sift").mkdir()
    usage_meter.usage_path(tmp_path).write_text("{not json")
    record_turn(tmp_path, model="claude-sonnet-5", input_tokens=100)
    usage_meter.usage_path(tmp_path).unlink()
    record_turn(tmp_path, model="claude-sonnet-5", input_tokens=200)
    summary = summarize(tmp_path)
    assert summary["turns"] == 1
    assert summary["total_tokens"] == 200
    assert summary["unrecorded_turns"] == 1
    assert summary["usage_accounting_complete"] is False
    assert not usage_meter._usage_health_path(tmp_path).exists()


def test_concurrent_turns_do_not_lose_usage_updates(tmp_path: Path) -> None:
    def _record(_index: int) -> None:
        record_turn(
            tmp_path,
            model="claude-sonnet-5",
            provider="anthropic",
            input_tokens=10,
            output_tokens=2,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(_record, range(120)))
    summary = summarize(tmp_path)
    assert summary["turns"] == 120
    assert summary["input_tokens"] == 1_200
    assert summary["output_tokens"] == 240
    assert summary["usage_accounting_complete"] is True


def test_meter_stores_no_content(tmp_path: Path) -> None:
    """The meter must be counts and model ids only — not a second
    copy of the transcript."""
    record_turn(tmp_path, model="claude-sonnet-5", input_tokens=100, output_tokens=10)
    raw = usage_meter.usage_path(tmp_path).read_text(encoding="utf-8")
    for key in ("prompt", "message", "text", "content", "result"):
        assert key not in raw.lower()
