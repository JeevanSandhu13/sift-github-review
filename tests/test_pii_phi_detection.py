"""PII/PHI local detection — name- and value-pattern-based heuristics
in the dataset profile.

Local-only by construction (same module, same "never sent to the
model" framing as every other detector here — see
``dataset_profile.py``'s module docstring). These tests pin two
things equally hard: real PII/PHI shapes get flagged, and ordinary
data that only superficially resembles PII does NOT — a detector that
cries wolf on "phoneme_rate" or "card_id" trains researchers to
ignore it, which is worse than not having it.
"""

from __future__ import annotations

import pytest

from sift.dataset_profile import (
    _card_value_hit,
    _detect_pii_phi,
    _detect_pii_phi_by_name,
    _detect_pii_phi_by_values,
    _luhn_valid,
    profile_dataset,
)


@pytest.fixture()
def pd():
    return pytest.importorskip("pandas")


def _write(tmp_path, pd, df, name="data.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Name-based detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,expected_substr", [
    ("ssn", "Social Security"),
    ("SSN", "Social Security"),
    ("social_security_number", "Social Security"),
    ("email", "email"),
    ("Email_Address", "email"),
    ("phone_number", "phone"),
    ("mobile", "phone"),
    ("date_of_birth", "birth"),
    ("dob", "birth"),
    ("mrn", "medical record"),
    ("patient_id", "medical record"),
    ("credit_card_number", "card"),
    ("cvv", "card"),
    ("passport_number", "passport"),
    ("street_address", "address"),
    ("icd10", "clinical code"),
])
def test_name_based_detection_fires(name, expected_substr):
    category = _detect_pii_phi_by_name(name)
    assert category is not None, f"{name!r} should have been flagged"
    assert expected_substr.lower() in category.lower()


@pytest.mark.parametrize("name", [
    "phoneme_rate",   # "phone" is a substring, not a whole token
    "card_id",        # "card" alone is not a pattern; unrelated FK
    "cardiac_rate",   # "card" is not a standalone token here
    "id",
    "amount",
    "region",
    "age",
    "score",
    "count",
    "embed",
    "recorded_at",
])
def test_name_based_detection_does_not_fire_on_lookalikes(name):
    assert _detect_pii_phi_by_name(name) is None, name


@pytest.mark.parametrize("name,expected_substr", [
    # These ARE flagged, and reasonably so for a name-based heuristic
    # that favors recall over precision (a privacy detector should
    # err toward "flag it, let the researcher confirm" rather than
    # silently missing real PII): "address" and "birthday" as
    # standalone tokens are strong-enough individual signals that a
    # compound name containing them is worth a second look, even
    # though the specific compound isn't the PII value itself.
    ("address_confirmed", "address"),
    ("birthday_party_id", "birth"),
])
def test_name_based_detection_favors_recall_on_compound_names(
    name, expected_substr,
):
    category = _detect_pii_phi_by_name(name)
    assert category is not None
    assert expected_substr in category.lower()


def test_ordinary_id_and_measurement_columns_are_not_flagged():
    for name in ("user_id", "order_id", "cardiac_rate", "phoneme_rate",
                 "recorded_at", "amount_paid", "region", "score"):
        assert _detect_pii_phi_by_name(name) is None, name


# ---------------------------------------------------------------------------
# Value-pattern detection
# ---------------------------------------------------------------------------

def test_ssn_value_pattern_detection(pd):
    col = pd.Series([f"{100+i:03d}-{i%99:02d}-{1000+i:04d}" for i in range(30)])
    assert _detect_pii_phi_by_values(col, len(col)) == "Social Security number"


def test_email_value_pattern_detection(pd):
    col = pd.Series([f"user{i}@example.com" for i in range(30)])
    assert _detect_pii_phi_by_values(col, len(col)) == "email address"


def test_phone_value_pattern_detection(pd):
    col = pd.Series([f"555-123-{4000+i:04d}" for i in range(30)])
    assert _detect_pii_phi_by_values(col, len(col)) == "phone number"


def test_ip_address_value_pattern_detection(pd):
    col = pd.Series([f"10.0.{i % 256}.{(i * 3) % 256}" for i in range(30)])
    assert _detect_pii_phi_by_values(col, len(col)) == "IP address"


def test_credit_card_value_pattern_detection_uses_luhn(pd):
    # Real Luhn-valid test card numbers (Visa/Mastercard public test
    # numbers), repeated to fill the sample.
    valid_cards = ["4111111111111111", "5500005555555559"]
    col = pd.Series([valid_cards[i % 2] for i in range(30)])
    assert _detect_pii_phi_by_values(col, len(col)) == "credit/debit card number"


def test_arbitrary_long_ids_do_not_trigger_card_detection(pd):
    """The Luhn check is the whole point: a column of arbitrary
    16-digit IDs (not real card numbers) must NOT be flagged as
    credit cards just because it's the right length."""
    col = pd.Series([str(1_000_000_000_000_000 + i) for i in range(30)])
    assert _detect_pii_phi_by_values(col, len(col)) is None


def test_free_text_with_one_email_like_substring_is_not_flagged(pd):
    """A notes column that happens to contain ONE email-shaped value
    among many ordinary notes must stay below the match-rate
    threshold and NOT be flagged — this is the guard against a
    detector that fires on a single coincidental match."""
    values = ["patient reported improvement"] * 29 + ["contact a@b.com"]
    col = pd.Series(values)
    assert _detect_pii_phi_by_values(col, len(col)) is None


def test_luhn_checksum_is_correct(pd):
    assert _luhn_valid("4111111111111111") is True
    assert _luhn_valid("4111111111111112") is False  # last digit tampered
    assert _card_value_hit("4111 1111 1111 1111") is True  # spaced form
    assert _card_value_hit("4111-1111-1111-1111") is True  # dashed form
    assert _card_value_hit("not a card") is False
    assert _card_value_hit("123") is False  # too short


# ---------------------------------------------------------------------------
# Combined detector + basis reporting
# ---------------------------------------------------------------------------

def test_detect_pii_phi_reports_both_bases_when_both_fire(pd):
    col = pd.Series([f"user{i}@example.com" for i in range(30)])
    result = _detect_pii_phi("email", col, len(col))
    assert result is not None
    assert result["category"] == "email address"
    assert set(result["basis"]) == {"name", "value_pattern"}


def test_detect_pii_phi_reports_name_only_basis(pd):
    # Name matches "ssn" but the actual values don't look like SSNs
    # (e.g. a numeric-coded/anonymized SSN column) — name-only basis.
    col = pd.Series(list(range(30)))
    result = _detect_pii_phi("ssn_coded", col, len(col))
    assert result is not None
    assert result["basis"] == ["name"]


def test_detect_pii_phi_reports_value_only_basis(pd):
    # Name gives no hint at all, but the values are unmistakably
    # SSN-shaped.
    col = pd.Series([f"{100+i:03d}-{i%99:02d}-{1000+i:04d}" for i in range(30)])
    result = _detect_pii_phi("field_7", col, len(col))
    assert result is not None
    assert result["basis"] == ["value_pattern"]


def test_detect_pii_phi_returns_none_for_ordinary_column(pd):
    col = pd.Series([20 + i for i in range(30)])
    assert _detect_pii_phi("age", col, len(col)) is None


# ---------------------------------------------------------------------------
# End-to-end via profile_dataset()
# ---------------------------------------------------------------------------

def test_profile_dataset_surfaces_pii_phi_columns(tmp_path, pd):
    df = pd.DataFrame({
        "patient_id": range(1, 31),
        "email": [f"p{i}@clinic.example" for i in range(30)],
        "age": [20 + (i % 50) for i in range(30)],
    })
    path = _write(tmp_path, pd, df)
    prof = profile_dataset(path)
    assert prof["ok"] is True

    flagged_names = {c["name"] for c in prof["pii_phi_columns"]}
    assert "email" in flagged_names
    assert "patient_id" in flagged_names   # name-based: "medical record"
    assert "age" not in flagged_names

    email_var = next(v for v in prof["variables"] if v["name"] == "email")
    assert email_var["pii_phi"]["category"] == "email address"


def test_profile_dataset_health_issue_for_pii_phi(tmp_path, pd):
    df = pd.DataFrame({
        "ssn": [f"{100+i:03d}-{i%99:02d}-{1000+i:04d}" for i in range(30)],
        "amount": list(range(30)),
    })
    path = _write(tmp_path, pd, df)
    prof = profile_dataset(path)
    issues = prof["health"]["issues"]
    pii_issue = next(
        (i for i in issues if "personal or health information" in i["message"]),
        None,
    )
    assert pii_issue is not None
    assert pii_issue["severity"] == "warn"
    assert "ssn" in pii_issue["columns"]


def test_ordinary_dataset_has_no_pii_phi_flags(tmp_path, pd):
    """Negative control at the full profile_dataset() level — a
    perfectly ordinary numeric/categorical dataset must not trip
    anything."""
    df = pd.DataFrame({
        "order_id": range(1, 41),
        "amount": [10.5 + i for i in range(40)],
        "region": (["north", "south"] * 20),
    })
    path = _write(tmp_path, pd, df)
    prof = profile_dataset(path)
    assert prof["pii_phi_columns"] == []
    messages = [i["message"] for i in prof["health"]["issues"]]
    assert not any("personal or health information" in m for m in messages)
