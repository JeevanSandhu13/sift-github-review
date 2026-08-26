"""Linkage diagnostics and complex-survey design detection.

Two structural gaps Sift previously could not express at all:

1. **Relationships between files.** Every tool was dataset-oriented,
   yet research data is relational and the merge is where the
   catastrophic *silent* errors live — unmatched records dropping,
   many-to-many fan-out, keys that aren't unique.
2. **Survey design.** Weights, strata and PSUs were nowhere in the
   system, so a national survey analysed as a simple random sample
   produced biased estimates and wrong standard errors with nothing
   out of the ordinary in the output.

Both layers must be *quiet when they should be*: a linkage warning on
a clean 1:1 join, or a "sampling weight" badge on a clinical body-
weight column, would train researchers to ignore them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from sift.dataset_profile import profile_dataset
from sift.linkage import analyze_pair, analyze_session
from sift.policy import DatasetPolicy, SiftPolicy, save_policy


# --------------------------------------------------------------------
# Privacy posture
# --------------------------------------------------------------------

def test_linkage_is_not_a_model_capability() -> None:
    """Like the profile: researcher-local, unreachable from the tool
    surface. If this fails, key-level match counts became model-
    visible without passing the sanitizer."""
    from sift.tools import ALLOWED_TOOL_NAMES, HANDLERS

    joined = " ".join(ALLOWED_TOOL_NAMES) + " " + " ".join(HANDLERS)
    assert "linkage" not in joined.lower()
    assert "linkage" not in Path("src/sift/tools.py").read_text(encoding="utf-8")


def test_linkage_never_returns_key_values(tmp_path: Path) -> None:
    """Counts and rates only — never the identifiers themselves."""
    pd.DataFrame({"patient_id": ["SECRET-A", "SECRET-B"]}).to_csv(
        tmp_path / "a.csv", index=False)
    pd.DataFrame({"patient_id": ["SECRET-A", "SECRET-C"]}).to_csv(
        tmp_path / "b.csv", index=False)
    import json
    blob = json.dumps(analyze_session(tmp_path))
    assert "SECRET" not in blob


def test_linkage_load_failure_is_not_reported_as_no_candidate_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pd.DataFrame({"patient_id": [1, 2]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"patient_id": [1, 2]}).to_csv(tmp_path / "b.csv", index=False)
    from sift import canonical_dataset

    monkeypatch.setattr(
        canonical_dataset,
        "load_canonical_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("simulated")),
    )
    report = analyze_session(tmp_path)
    assert report["ok"] is False
    assert report["checks_complete"] is False
    assert report["pairs"] == []
    assert report["diagnostics"] == [{
        "dataset": "a.csv + b.csv",
        "stage": "canonical_load",
        "reason": "OSError",
    }]


# --------------------------------------------------------------------
# Linkage: the four scenarios that matter
# --------------------------------------------------------------------

def test_clean_one_to_one_produces_no_warnings(tmp_path: Path) -> None:
    pd.DataFrame({"patient_id": range(1, 101), "age": [30] * 100}).to_csv(
        tmp_path / "roster.csv", index=False)
    pd.DataFrame({"patient_id": range(1, 101), "bmi": [24] * 100}).to_csv(
        tmp_path / "labs.csv", index=False)
    keys = analyze_pair(tmp_path / "roster.csv", tmp_path / "labs.csv")
    assert len(keys) == 1
    assert keys[0]["relationship"] == "one-to-one"
    assert keys[0]["left_match_pct"] == 100.0
    assert keys[0]["warnings"] == []


def test_linkage_honors_selected_spreadsheet_sheet(tmp_path: Path) -> None:
    """Link diagnostics must inspect the same worksheet shown elsewhere."""
    left = tmp_path / "roster.xlsx"
    right = tmp_path / "labs.xlsx"
    with pd.ExcelWriter(left) as writer:
        pd.DataFrame({"not_the_key": [1, 2]}).to_excel(
            writer, sheet_name="Notes", index=False
        )
        pd.DataFrame({"patient_id": [1, 2]}).to_excel(
            writer, sheet_name="Analysis", index=False
        )
    with pd.ExcelWriter(right) as writer:
        pd.DataFrame({"different": [3, 4]}).to_excel(
            writer, sheet_name="Notes", index=False
        )
        pd.DataFrame({"patient_id": [1, 2]}).to_excel(
            writer, sheet_name="Analysis", index=False
        )
    save_policy(
        tmp_path,
        SiftPolicy(
            datasets={
                left.name: DatasetPolicy(excel_sheet="Analysis"),
                right.name: DatasetPolicy(excel_sheet="Analysis"),
            }
        ),
    )

    keys = analyze_pair(left, right)
    assert len(keys) == 1
    assert keys[0]["left_match_pct"] == 100.0
    assert keys[0]["right_match_pct"] == 100.0


def test_unmatched_records_are_quantified(tmp_path: Path) -> None:
    """The error that silently changes the population under study."""
    pd.DataFrame({"patient_id": range(1, 101)}).to_csv(
        tmp_path / "roster.csv", index=False)
    pd.DataFrame({"patient_id": range(50, 151)}).to_csv(
        tmp_path / "claims.csv", index=False)
    keys = analyze_pair(tmp_path / "roster.csv", tmp_path / "claims.csv")
    warn = " ".join(keys[0]["warnings"])
    assert "no match" in warn and "silently drops" in warn
    assert keys[0]["matched_keys"] == 51


def test_many_to_many_fanout_is_flagged(tmp_path: Path) -> None:
    pd.DataFrame({"firm_id": [1, 1, 2, 2]}).to_csv(
        tmp_path / "a.csv", index=False)
    pd.DataFrame({"firm_id": [1, 1, 2, 2]}).to_csv(
        tmp_path / "b.csv", index=False)
    keys = analyze_pair(tmp_path / "a.csv", tmp_path / "b.csv")
    assert keys[0]["relationship"] == "many-to-many"
    assert any("multiplies rows" in w for w in keys[0]["warnings"])


def test_type_mismatch_reported_as_no_match(tmp_path: Path) -> None:
    """String vs numeric keys — the leading-zero case that makes a
    merge return nothing in Stata/R."""
    pd.DataFrame({"case_id": ["001", "002"]}).to_parquet(
        tmp_path / "a.parquet")
    pd.DataFrame({"case_id": [1, 2]}).to_parquet(tmp_path / "b.parquet")
    keys = analyze_pair(tmp_path / "a.parquet", tmp_path / "b.parquet")
    assert keys[0]["matched_keys"] == 0
    assert any("different formats" in w for w in keys[0]["warnings"])


def test_non_key_shared_columns_are_ignored(tmp_path: Path) -> None:
    """`age` and `year` are shared by everything and join nothing."""
    pd.DataFrame({"age": [1, 2], "year": [2020, 2021]}).to_csv(
        tmp_path / "a.csv", index=False)
    pd.DataFrame({"age": [1, 2], "year": [2020, 2021]}).to_csv(
        tmp_path / "b.csv", index=False)
    assert analyze_pair(tmp_path / "a.csv", tmp_path / "b.csv") == []


def test_oversized_files_are_skipped_not_sampled(tmp_path: Path,
                                                 monkeypatch) -> None:
    """A match rate from a partial read is not a match rate."""
    pd.DataFrame({"patient_id": range(500)}).to_csv(
        tmp_path / "a.csv", index=False)
    pd.DataFrame({"patient_id": range(500)}).to_csv(
        tmp_path / "b.csv", index=False)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "50")
    report = analyze_session(tmp_path)
    assert report["pairs"] == []
    assert len(report["skipped"]) == 2
    assert report["checks_complete"] is False
    assert "misleading" in report["skipped"][0]["reason"]


def test_session_report_survives_broken_files(tmp_path: Path) -> None:
    pd.DataFrame({"patient_id": [1, 2]}).to_csv(
        tmp_path / "good.csv", index=False)
    (tmp_path / "broken.parquet").write_bytes(b"not parquet")
    report = analyze_session(tmp_path)
    assert report["ok"] is False
    assert report["checks_complete"] is False
    assert report["diagnostics"][0]["dataset"] == "broken.parquet"


# --------------------------------------------------------------------
# Survey design detection
# --------------------------------------------------------------------

def _profile_with(tmp_path: Path, frame: dict) -> dict:
    pd.DataFrame(frame).to_csv(tmp_path / "s.csv", index=False)
    return profile_dataset(tmp_path / "s.csv")


@pytest.mark.parametrize("column,role", [
    ("wtmec2yr", "sampling weight"),
    ("pweight", "sampling weight"),
    ("finalwt", "sampling weight"),
    ("sdmvstra", "stratum"),
    ("strata", "stratum"),
    ("sdmvpsu", "primary sampling unit"),
    ("psu", "primary sampling unit"),
    ("fpc", "finite population correction"),
    ("repwt17", "replicate weights"),
])
def test_design_variables_detected(tmp_path: Path, column, role) -> None:
    prof = _profile_with(tmp_path, {column: [1.5] * 50, "y": [2] * 50})
    roles = {c["name"]: c["role"]
             for c in prof["survey_design_columns"]}
    assert roles.get(column) == role


@pytest.mark.parametrize("column", [
    "body_weight_kg", "birth_weight", "weight_gain_kg", "height",
])
def test_measured_quantities_are_not_design_variables(
        tmp_path: Path, column) -> None:
    """Telling a clinician their outcome is a sampling weight is
    exactly the confident-but-wrong output to avoid."""
    prof = _profile_with(tmp_path, {column: [70.5] * 50})
    assert prof["survey_design_columns"] == []


@pytest.mark.parametrize("column", [
    "stress_level", "strength_score", "street_number", "stroke_count",
    "streak_length", "strike_rate", "structure_id",
])
def test_ordinary_str_prefixed_columns_are_not_a_stratum(
        tmp_path: Path, column) -> None:
    """Audit pass 2 finding: the "stratum" pattern list includes the
    3-character token "str", previously matched via a boundary-blind
    substring check against the WHOLE column name -- so any numeric
    column merely starting or ending with "str" was misclassified as
    a survey stratification variable, regardless of what the rest of
    the name said. All of these are ordinary numeric columns a real
    research dataset would plausibly have, and none of them are
    survey design variables."""
    prof = _profile_with(tmp_path, {column: [1.5] * 50, "y": [2] * 50})
    roles = {c["name"]: c["role"] for c in prof["survey_design_columns"]}
    assert column not in roles, (
        f"{column!r} was misclassified as a survey-design variable "
        f"(role={roles.get(column)!r}) -- it's an ordinary numeric "
        f"column whose name happens to start/end with the short "
        f"'str' pattern fragment"
    )


def test_string_cluster_column_is_not_a_psu(tmp_path: Path) -> None:
    prof = _profile_with(tmp_path, {"cluster_label": ["a", "b"] * 25})
    assert prof["survey_design_columns"] == []


def test_negative_weights_rejected(tmp_path: Path) -> None:
    """Sampling weights are positive by construction."""
    prof = _profile_with(tmp_path, {"pweight": [-1.0] * 50})
    assert prof["survey_design_columns"] == []


def test_design_columns_flagged_on_the_variable_too(tmp_path: Path) -> None:
    prof = _profile_with(tmp_path, {"wtmec2yr": [1.2] * 50})
    var = prof["variables"][0]
    assert var["survey_design_role"] == "sampling weight"


def test_codebook_surfaces_design_variables(tmp_path: Path) -> None:
    from sift.research_export import build_codebook

    pd.DataFrame({"wtmec2yr": [1.5] * 50, "sdmvpsu": [1, 2] * 25,
                  "bmi": [24.0] * 50}).to_csv(
        tmp_path / "nhanes.csv", index=False)
    md = build_codebook(tmp_path)["markdown"]
    assert "Survey design variables detected" in md
    assert "wtmec2yr" in md and "sampling weight" in md
