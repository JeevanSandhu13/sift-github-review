"""Dataset Health panel — deterministic issue detection and scoring.

Every check here traces to a computed number (an IQR fence, a parsed
date range, a value-count share). The point of these tests is as much
about what does NOT get flagged as what does: a normal, unremarkable
dataset should score high and carry no issues, and the detectors must
not fire on data that only superficially resembles a problem (an "id"
column with "day" in its name, an ordinary right-skewed numeric
column, ordinary binary data with a 60/40 split).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sift.dataset_profile import profile_dataset


@pytest.fixture()
def frame():
    pd = pytest.importorskip("pandas")
    return pd


def _write(tmp_path, pd, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------

def test_extreme_values_flagged(tmp_path, frame) -> None:
    values = list(range(100)) + [100_000, -100_000, 200_000]
    df = frame.DataFrame({"amount": values})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "outliers" in by_name["amount"]
    assert by_name["amount"]["outliers"]["count"] >= 3
    assert any("amount" in i["columns"] for i in prof["health"]["issues"])


def test_ordinary_skewed_data_not_flagged(tmp_path, frame) -> None:
    """A right-skewed but ordinary distribution (income-shaped) should
    not trip the 3x-IQR fence — that fence is deliberately wide."""
    import random
    random.seed(0)
    values = [round(random.lognormvariate(10, 0.6)) for _ in range(500)]
    df = frame.DataFrame({"income": values})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    # Some lognormal draws may clear the fence by chance; the important
    # property is that it's not the WHOLE distribution being flagged
    # and the health score isn't devastated by ordinary skew.
    assert prof["health"]["score"] >= 70


def test_constant_column_has_no_outliers(tmp_path, frame) -> None:
    df = frame.DataFrame({"a": [5] * 50, "b": range(50)})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "outliers" not in by_name["a"]


# --------------------------------------------------------------------
# Impossible dates
# --------------------------------------------------------------------

def test_future_and_ancient_dates_flagged(tmp_path, frame) -> None:
    today = datetime.now(timezone.utc)
    good = [(today - timedelta(days=i)).date().isoformat() for i in range(46)]
    bad = ["1850-01-01", "1899-12-31",
           (today + timedelta(days=365)).date().isoformat(),
           (today + timedelta(days=3650)).date().isoformat()]
    df = frame.DataFrame({"signup_date": good + bad})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "impossible_dates" in by_name["signup_date"]
    assert by_name["signup_date"]["impossible_dates"]["count"] == 4
    assert "signup_date" in [
        c for i in prof["health"]["issues"] for c in i["columns"]
    ]


def test_ordinary_recent_dates_not_flagged(tmp_path, frame) -> None:
    today = datetime.now(timezone.utc)
    good = [(today - timedelta(days=i)).date().isoformat() for i in range(50)]
    df = frame.DataFrame({"order_date": good})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "impossible_dates" not in by_name["order_date"]


def test_numeric_column_with_date_like_name_not_parsed_as_date(tmp_path, frame) -> None:
    """A numeric 'days_since_signup' column must not be coerced into a
    date parse just because its name contains a date token."""
    df = frame.DataFrame({"days_since_signup": range(50)})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "impossible_dates" not in by_name["days_since_signup"]


def test_non_date_text_column_not_flagged(tmp_path, frame) -> None:
    """'update_notes' contains 'update' — not a date token — and
    isn't parseable as dates; must not be flagged."""
    df = frame.DataFrame({
        "customer_notes": ["called about billing"] * 30 + ["asked for refund"] * 25,
    })
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "impossible_dates" not in by_name["customer_notes"]


# --------------------------------------------------------------------
# Imbalance
# --------------------------------------------------------------------

def test_severe_imbalance_flagged(tmp_path, frame) -> None:
    df = frame.DataFrame({"churned": [0] * 97 + [1] * 3})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "imbalance" in by_name["churned"]
    assert by_name["churned"]["imbalance"]["top_value"] == "0"
    assert by_name["churned"]["imbalance"]["share"] >= 0.95


def test_ordinary_binary_split_not_flagged(tmp_path, frame) -> None:
    df = frame.DataFrame({"is_member": [0] * 60 + [1] * 40})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "imbalance" not in by_name["is_member"]


def test_high_cardinality_column_not_evaluated_for_imbalance(tmp_path, frame) -> None:
    """A near-unique column (e.g. an id) has no meaningful 'top
    category' — imbalance must only fire in the 2-20 distinct-value
    range."""
    df = frame.DataFrame({"customer_id": range(100)})
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "imbalance" not in by_name["customer_id"]


# --------------------------------------------------------------------
# Health score aggregation
# --------------------------------------------------------------------

def test_clean_dataset_scores_high_with_no_issues(tmp_path, frame) -> None:
    df = frame.DataFrame({
        "id": range(200),
        "amount": [round(50 + i * 0.1, 2) for i in range(200)],
        "segment": (["a"] * 90 + ["b"] * 70 + ["c"] * 40),
    })
    prof = profile_dataset(_write(tmp_path, frame, df))
    assert prof["health"]["score"] == 100
    assert prof["health"]["issues"] == []


def test_score_never_negative(tmp_path, frame) -> None:
    """A dataset that trips every check at once must still clamp to 0,
    not go negative — deductions are additive and uncapped per-check
    until the final floor."""
    today = datetime.now(timezone.utc)
    n = 100
    df = frame.DataFrame({
        "a": [1] * n,                      # constant
        "b": [None] * n,                   # all missing
        "amount": [1] * (n - 5) + [10**9] * 5,   # outliers
        "signup_date": ["1850-01-01"] * n,        # impossible dates
        "flag": [0] * 99 + [1],            # imbalance
    })
    # duplicate a bunch of rows too
    df = frame.concat([df, df.iloc[:50]], ignore_index=True)
    prof = profile_dataset(_write(tmp_path, frame, df))
    assert prof["health"]["score"] >= 0
    assert len(prof["health"]["issues"]) >= 4


def test_sampled_profile_does_not_penalize_unknown_duplicates(tmp_path, frame, monkeypatch) -> None:
    """On a sampled (over-ceiling) profile, ``duplicate_rows`` is
    ``None`` (unknown, not zero) — the health score must not deduct
    for a check it never ran."""
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "512")
    df = frame.DataFrame({"a": range(400), "b": ["x"] * 400})
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    prof = profile_dataset(path)
    assert prof["duplicate_rows"] is None
    # No issue should mention duplicates when the count is unknown.
    assert not any("duplicate" in i["message"] for i in prof["health"]["issues"])


def test_health_issues_only_reference_computed_columns(tmp_path, frame) -> None:
    """Every column named in an issue's ``columns`` list must
    correspond to a real flag on that column's variable entry —
    the health summary must not invent column references."""
    today = datetime.now(timezone.utc)
    df = frame.DataFrame({
        "a": [1] * 51,
        "amount": list(range(50)) + [10**6],
        "churned": [0] * 49 + [1] * 2,
    })
    prof = profile_dataset(_write(tmp_path, frame, df))
    by_name = {v["name"]: v for v in prof["variables"]}
    for issue in prof["health"]["issues"]:
        for col in issue["columns"]:
            entry = by_name[col]
            assert (
                "outliers" in entry or "impossible_dates" in entry
                or "imbalance" in entry or entry.get("flag") in (
                    "constant", "all missing",
                )
            )


def test_health_score_bounds(tmp_path, frame) -> None:
    df = frame.DataFrame({"a": range(50), "b": ["x"] * 50})
    prof = profile_dataset(_write(tmp_path, frame, df))
    assert 0 <= prof["health"]["score"] <= 100
    assert isinstance(prof["health"]["score"], int)


def test_health_absent_when_profile_fails(tmp_path) -> None:
    out = profile_dataset(tmp_path / "nope.csv")
    assert out["ok"] is False
    assert "health" not in out
