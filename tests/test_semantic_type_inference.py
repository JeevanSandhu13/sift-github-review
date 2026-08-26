"""Semantic type inference and likely-target detection.

dtype alone (int/float/text/bool/date) says little about what a
column MEANS. These tests cover the best-effort semantic label layered
on top of dtype, and the heuristic "which column looks like the
outcome variable" guess — both computed entirely locally as part of
``dataset_profile.py`` (never sent to the model; see that module's
docstring and the structural tests in ``test_dataset_profile.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sift.dataset_profile import profile_dataset


@pytest.fixture()
def frame():
    pd = pytest.importorskip("pandas")
    return pd


def _profile(tmp_path: Path, frame, df, name: str = "d.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return profile_dataset(path)


def _semantic_types(prof) -> dict[str, str]:
    return {v["name"]: v.get("semantic_type") for v in prof["variables"]}


# --------------------------------------------------------------------
# Semantic type inference
# --------------------------------------------------------------------

def test_binary_numeric_and_text(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "converted": [0, 1] * 30,
        "flag_word": (["yes", "no"] * 30),
    })
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["converted"] == "binary"
    assert types["flag_word"] == "binary"


def test_currency_by_name(tmp_path, frame) -> None:
    df = frame.DataFrame({"annual_income": [x * 1.1 for x in range(1, 101)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["annual_income"] == "currency"


def test_percentage_by_name_and_bounded_range(tmp_path, frame) -> None:
    df = frame.DataFrame({"conversion_rate": [float(x % 101) for x in range(200)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["conversion_rate"] == "percentage"


def test_percentage_name_without_bounded_range_is_not_percentage(tmp_path, frame) -> None:
    """A column named like a rate but whose values run well past 100
    is not a plausible percentage — the bound check keeps a name
    match from overriding an implausible value range."""
    df = frame.DataFrame({"exchange_rate": [x * 137.5 for x in range(1, 101)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["exchange_rate"] != "percentage"


def test_count_by_name_requires_nonnegative_integer(tmp_path, frame) -> None:
    # Repeating values (not all-distinct) so this doesn't also trip
    # the "likely identifier" flag -- a real order-count column
    # repeats values across customers; testing the count-by-name path
    # specifically requires a column that ISN'T also a plausible key.
    df = frame.DataFrame({"total_orders": [x % 50 for x in range(300)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["total_orders"] == "count"


def test_geographic_by_name(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "region": (["north", "south", "east", "west"] * 25),
    })
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["region"] == "geographic"


def test_ordinal_low_cardinality_integer(tmp_path, frame) -> None:
    df = frame.DataFrame({"satisfaction_rating": [(x % 5) + 1 for x in range(300)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["satisfaction_rating"] == "ordinal"


def test_continuous_high_cardinality_float(tmp_path, frame) -> None:
    df = frame.DataFrame({"measurement": [x * 0.37 for x in range(300)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["measurement"] == "continuous"


def test_free_text_by_name_and_length(tmp_path, frame) -> None:
    pool = [
        "Called in about billing and seemed satisfied with the resolution.",
        "Requested a refund for a duplicate charge on last month's bill.",
        "Asked about upgrading their plan for more storage capacity.",
    ]
    df = frame.DataFrame({"notes": [pool[i % 3] for i in range(90)]})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["notes"] == "free_text"


def test_short_text_is_categorical_not_free_text(tmp_path, frame) -> None:
    df = frame.DataFrame({"plan_tier": (["basic", "pro", "enterprise"] * 30)})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["plan_tier"] == "categorical"


def test_datetime_dtype_is_date(tmp_path, frame) -> None:
    pytest.importorskip("pyarrow")
    pd = frame
    # CSV round-trips dates as plain text (pandas never auto-parses on
    # read) -- a real datetime64 dtype needs a format that preserves
    # it, e.g. parquet.
    path = tmp_path / "d.parquet"
    df = pd.DataFrame({"event_ts": pd.date_range("2022-01-01", periods=60, freq="D")})
    df.to_parquet(path)
    prof = profile_dataset(path)
    types = _semantic_types(prof)
    assert types["event_ts"] == "date"


def test_all_distinct_date_text_is_date_not_identifier(tmp_path, frame) -> None:
    """A daily-granularity date column, written as text (the normal
    CSV shape — pandas never auto-parses dates on read), is
    all-distinct at 60 rows and so also trips the pre-existing
    "likely identifier" flag. The date-name + actual-parse check must
    outrank that flag, or a real date column gets mislabelled as a
    meaningless key."""
    pd = frame
    path = tmp_path / "d.csv"
    dates = pd.date_range("2022-01-01", periods=60, freq="D").astype(str)
    df = pd.DataFrame({"signup_date": dates})
    df.to_csv(path, index=False)
    prof = profile_dataset(path)
    by_name = {v["name"]: v for v in prof["variables"]}
    assert by_name["signup_date"]["flag"] == "likely identifier"
    assert by_name["signup_date"]["semantic_type"] == "date"


def test_generic_name_token_without_real_dates_is_not_labelled_date(tmp_path, frame) -> None:
    """``start`` is a date-suggestive token, but a column called
    ``start_balance`` full of dollar-shaped text should not be
    mislabelled "date" just because the token matched — the parse
    gate is what prevents this false positive."""
    df = frame.DataFrame({
        "start_balance": [f"${x}.00" for x in range(1, 101)],
    })
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["start_balance"] != "date"


def test_all_unique_float_is_continuous_not_identifier(tmp_path, frame) -> None:
    """The pre-existing identifier detector fires on ANY all-distinct
    column, including a continuous float measurement that happens not
    to repeat in this sample (routine for real-valued data). Semantic
    typing must not inherit that false positive — floats fall through
    to the numeric branch regardless of the identifier flag."""
    df = frame.DataFrame({"income": [20000.13 + x * 7.91 for x in range(100)]})
    prof = _profile(tmp_path, frame, df)
    by_name = {v["name"]: v for v in prof["variables"]}
    assert by_name["income"]["flag"] == "likely identifier"
    assert by_name["income"]["semantic_type"] != "identifier"


def test_all_unique_integer_is_identifier(tmp_path, frame) -> None:
    df = frame.DataFrame({"customer_id": list(range(100))})
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["customer_id"] == "identifier"


def test_constant_and_all_missing_semantic_types(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "site": ["HQ"] * 50,
        "unused": [None] * 50,
        "filler": list(range(50)),  # keeps the frame non-degenerate
    })
    types = _semantic_types(_profile(tmp_path, frame, df))
    assert types["site"] == "constant"
    assert types["unused"] == "unknown"


# --------------------------------------------------------------------
# Likely-target detection
# --------------------------------------------------------------------

def test_name_matched_target_outranks_positional_guess(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "age": list(range(20, 120)),
        "churned": ([0] * 90 + [1] * 10),
        "region": (["north", "south"] * 50),
    })
    prof = _profile(tmp_path, frame, df)
    candidates = prof["likely_target_candidates"]
    assert candidates, "expected at least one target candidate"
    assert candidates[0]["name"] == "churned"
    assert any("vocabulary" in r for r in candidates[0]["reasons"])


def test_candidates_are_sorted_and_capped(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "target": ([0, 1] * 50),
        "outcome": ([0, 1] * 50),
        "response": ([0, 1] * 50),
        "label": ([0, 1] * 50),
        "irrelevant_id": list(range(100)),
    })
    prof = _profile(tmp_path, frame, df)
    candidates = prof["likely_target_candidates"]
    assert len(candidates) <= 3
    scores = [c["score"] for c in candidates]
    assert scores == sorted(scores, reverse=True)


def test_identifier_and_constant_columns_never_become_candidates(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "customer_id": list(range(100)),
        "site": ["HQ"] * 100,
        "age": list(range(20, 120)),
    })
    prof = _profile(tmp_path, frame, df)
    names = [c["name"] for c in prof["likely_target_candidates"]]
    assert "customer_id" not in names
    assert "site" not in names


def test_pii_column_never_becomes_a_target_candidate_even_with_matching_name(tmp_path, frame) -> None:
    """A column literally named ``label`` that also happens to hold
    email addresses (PII) must still be excluded — the PII flag is a
    hard veto, not a factor to outweigh with a name match."""
    df = frame.DataFrame({
        "label": [f"user{i}@example.com" for i in range(60)],
        "age": list(range(20, 80)),
    })
    prof = _profile(tmp_path, frame, df)
    names = [c["name"] for c in prof["likely_target_candidates"]]
    assert "label" not in names


def test_survey_weight_column_never_becomes_a_target_candidate(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "svywt": [1.0 + (x % 10) * 0.1 for x in range(100)],
        "age": list(range(20, 120)),
    })
    prof = _profile(tmp_path, frame, df)
    names = [c["name"] for c in prof["likely_target_candidates"]]
    assert "svywt" not in names


def test_no_plausible_columns_yields_empty_candidate_list(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "id": list(range(50)),
        "constant_col": ["x"] * 50,
    })
    prof = _profile(tmp_path, frame, df)
    assert prof["likely_target_candidates"] == []


def test_imbalanced_binary_scores_higher_than_balanced_binary(tmp_path, frame) -> None:
    """A heavily lopsided binary column (rare-event shape) should
    score at least as high as an ordinary balanced binary column with
    no other distinguishing signal."""
    df = frame.DataFrame({
        "rare_flag": ([1] * 5 + [0] * 95),
        "balanced_flag": ([0, 1] * 50),
    })
    prof = _profile(tmp_path, frame, df)
    by_name = {c["name"]: c["score"] for c in prof["likely_target_candidates"]}
    assert by_name.get("rare_flag", 0) >= by_name.get("balanced_flag", 0)


def test_last_column_convention_is_a_weak_positive_signal(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "feature_a": list(range(100)),
        "feature_b": [x * 2 for x in range(100)],
        "some_measure": ([0, 1] * 50),
    })
    prof = _profile(tmp_path, frame, df)
    names = [c["name"] for c in prof["likely_target_candidates"]]
    assert "some_measure" in names


def test_semantic_type_present_for_every_profiled_variable(tmp_path, frame) -> None:
    """Structural guarantee: every variable in the profile carries a
    semantic_type string, so frontend code can render it without a
    presence check for every column shape."""
    df = frame.DataFrame({
        "a": list(range(50)),
        "b": ["x"] * 50,
        "c": [None] * 50,
    })
    prof = _profile(tmp_path, frame, df)
    for v in prof["variables"]:
        assert isinstance(v.get("semantic_type"), str) and v["semantic_type"]
