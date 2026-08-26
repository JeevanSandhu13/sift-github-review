"""Codebook export — completeness and honesty.

The codebook's claim is a complete data dictionary for the session.
The tests pin: every dataset appears (unreadable ones as explicit
UNREADABLE entries, never silently dropped), SPSS value labels
survive into it, statistics agree with the profile, and the
researcher-local posture is stated in the document itself.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pyreadstat = pytest.importorskip("pyreadstat")

from sift.research_export import build_codebook


def _session(tmp_path: Path) -> Path:
    df = pd.DataFrame({"age": [30, 41, None, 28], "region": [1, 2, 1, 2]})
    pyreadstat.write_sav(
        df, str(tmp_path / "survey.sav"),
        column_labels=["Age in years", "Region code"],
        variable_value_labels={"region": {1: "north", 2: "south"}})
    pd.DataFrame({"dept": ["bio", "chem"], "grant": [1, 2]}).to_csv(
        tmp_path / "budget.csv", index=False)
    return tmp_path


def test_every_dataset_appears_with_metadata(tmp_path: Path) -> None:
    book = build_codebook(_session(tmp_path))
    md = book["markdown"]
    assert "## survey.sav" in md and "## budget.csv" in md
    assert "Age in years" in md
    assert "1.0 = north" in md
    # Missingness from the local profile (1 of 4 ages missing).
    assert "25.0%" in md
    rows = list(csv.DictReader(io.StringIO(book["csv"])))
    by_var = {(r["dataset"], r["variable"]): r for r in rows}
    assert ("survey.sav", "age") in by_var
    assert by_var[("survey.sav", "region")]["value_labels"] == \
        "1.0=north; 2.0=south"
    assert ("budget.csv", "dept") in by_var


def test_unreadable_dataset_is_explicit_not_skipped(tmp_path: Path) -> None:
    _session(tmp_path)
    (tmp_path / "broken.parquet").write_bytes(b"not parquet")
    book = build_codebook(tmp_path)
    assert "## broken.parquet" in book["markdown"]
    assert "Could not read" in book["markdown"]
    assert "UNREADABLE" in book["csv"]


def test_states_researcher_local_posture(tmp_path: Path) -> None:
    book = build_codebook(_session(tmp_path))
    assert "none of it is sent to a model" in book["markdown"].lower()


def test_empty_session(tmp_path: Path) -> None:
    book = build_codebook(tmp_path)
    assert "No datasets in this session" in book["markdown"]


def test_pipe_in_metadata_cannot_break_the_table(tmp_path: Path) -> None:
    pyreadstat.write_sav(
        pd.DataFrame({"x": [1.0]}), str(tmp_path / "h.sav"),
        column_labels=["contains | pipe"])
    md = build_codebook(tmp_path)["markdown"]
    table_rows = [ln for ln in md.splitlines()
                  if ln.startswith("|") and "contains" in ln]
    assert table_rows
    assert "\\|" in table_rows[0]     # escaped, columns intact


# ---------------------------------------------------------------------------
# CSV formula injection (OWASP CSV Injection) — malicious metadata
# ---------------------------------------------------------------------------
#
# The codebook CSV carries variable names, labels, and value labels
# taken verbatim from a data file's own metadata. If a malicious file
# sets one of those to a string starting with =, +, -, @, TAB, or CR,
# and a researcher (or collaborator) later opens the exported
# codebook.csv in Excel, that string would otherwise be interpreted
# as a formula/DDE payload the instant the cell renders — entirely
# independent of Sift's own script sandbox, since Excel is not
# something Sift controls. These tests pin that every file-derived
# field lands in the CSV with a formula-injection guard applied.

def test_malicious_label_is_neutralized_in_csv_export(tmp_path: Path) -> None:
    payload = "=cmd|'/c calc'!A1"
    pyreadstat.write_sav(
        pd.DataFrame({"x": [1.0]}), str(tmp_path / "h.sav"),
        column_labels=[payload])
    csv_text = build_codebook(tmp_path)["csv"]
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    row = next(r for r in rows if r["dataset"] == "h.sav")
    # The formula-trigger character must never be the first character
    # of the cell as Excel would see it.
    assert row["label"].startswith("'"), row["label"]
    assert row["label"] == "'" + payload
    # The original hostile text must still be present (this is a
    # neutralization, not a redaction — a human reading the cell
    # should still see what the source file actually said).
    assert payload in row["label"]


def test_malicious_value_label_is_neutralized_in_csv_export(
    tmp_path: Path,
) -> None:
    """SPSS value labels for a STRING variable key on the string's
    own values (unlike numeric variables, where the code always
    renders digit-first and so can never itself start a cell with a
    formula-trigger character) — pyreadstat round-trips an
    attacker-chosen key like ``=EVIL`` unchanged. If that key sorts
    first in the joined ``k=v; k=v`` cell, the WHOLE cell starts with
    the trigger character."""
    pyreadstat.write_sav(
        pd.DataFrame({"status": ["A", "B"]}), str(tmp_path / "h.sav"),
        variable_value_labels={
            "status": {"=EVIL": "evil label", "B": "ok"},
        })
    csv_text = build_codebook(tmp_path)["csv"]
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    row = next(
        r for r in rows
        if r["dataset"] == "h.sav" and r["variable"] == "status"
    )
    assert row["value_labels"].startswith("'"), row["value_labels"]
    assert "=EVIL=evil label" in row["value_labels"]


def test_malicious_dataset_filename_is_neutralized_in_csv_export(
    tmp_path: Path,
) -> None:
    # A file NAME is just as attacker-controlled as any metadata
    # inside it — a shared/downloaded dataset could be named this.
    # The fixture must itself be legal on every supported filesystem.
    # Quotes and several other characters are forbidden by Windows, while
    # the leading formula trigger is sufficient to exercise the guard.
    evil_name = "=HYPERLINK(evil.example,click).csv"
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / evil_name, index=False)
    csv_text = build_codebook(tmp_path)["csv"]
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    row = next(r for r in rows if evil_name in r["dataset"])
    assert row["dataset"].startswith("'")


def test_benign_metadata_is_left_completely_unmodified(tmp_path: Path) -> None:
    """The guard must be surgical — it only touches cells that
    actually start with a trigger character. Ordinary labels must
    reach the CSV byte-for-byte unchanged."""
    pyreadstat.write_sav(
        pd.DataFrame({"age": [30.0]}), str(tmp_path / "h.sav"),
        column_labels=["Age in years (self-reported)"])
    csv_text = build_codebook(tmp_path)["csv"]
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    row = next(r for r in rows if r["dataset"] == "h.sav")
    assert row["label"] == "Age in years (self-reported)"


def test_csv_formula_safe_unit() -> None:
    from sift.text_safety import csv_formula_safe

    for trigger in ("=1+1", "+1", "-1", "@SUM(A1)", "\t1", "\r1"):
        assert csv_formula_safe(trigger) == "'" + trigger
    for benign in ("normal text", "1 - 2", "a=b", "", "north"):
        assert csv_formula_safe(benign) == benign
