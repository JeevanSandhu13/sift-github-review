"""Real-world academic data: messy files and integrity outputs.

The population Sift targets does not produce clean UTF-8
comma-separated files. European Excel exports semicolons with decimal
commas; legacy SPSS/Access exports arrive Latin-1; Windows tools
write BOMs and UTF-16; survey packages encode missing as -999. And
the journals this population publishes in now require AI-use
disclosure statements. Each behaviour here was verified failing (or
absent) before being built.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from sift.dataset_profile import profile_dataset
from sift.research_export import ai_use_statement
from sift.schema import extract, load_data, row_count, text_table_params


# --------------------------------------------------------------------
# Encodings
# --------------------------------------------------------------------

def test_latin1_csv_reads_with_accents_intact(tmp_path: Path) -> None:
    """Previously a hard UnicodeDecodeError."""
    f = tmp_path / "l.csv"
    f.write_bytes("name,città\nJosé,München\n".encode("latin-1"))
    schema = extract(f, "names_types")
    assert [v["name"] for v in schema["variables"]] == ["name", "città"]
    df = load_data(f)
    assert df.iloc[0, 0] == "José"


def test_utf8_bom_does_not_corrupt_first_column_name(tmp_path: Path) -> None:
    f = tmp_path / "b.csv"
    f.write_bytes(b"\xef\xbb\xbf" + b"age,region\n30,north\n")
    schema = extract(f, "names_types")
    assert schema["variables"][0]["name"] == "age"   # not "﻿age"


def test_utf16_csv_reads(tmp_path: Path) -> None:
    f = tmp_path / "u.csv"
    f.write_bytes("a,b\n1,2\n3,4\n".encode("utf-16"))
    assert len(load_data(f)) == 2
    assert profile_dataset(f)["ok"] is True


# --------------------------------------------------------------------
# European CSV dialect
# --------------------------------------------------------------------

def test_semicolon_csv_splits_into_real_columns(tmp_path: Path) -> None:
    """Previously parsed SILENTLY into one mashed column — the worst
    failure class, because nothing errored anywhere."""
    f = tmp_path / "s.csv"
    f.write_text("age;income;region\n30;45,5;north\n41;52,1;south\n")
    schema = extract(f, "names_types")
    assert [v["name"] for v in schema["variables"]] == [
        "age", "income", "region"]


def test_decimal_commas_become_numbers(tmp_path: Path) -> None:
    f = tmp_path / "s.csv"
    f.write_text("age;income\n30;45,5\n41;52,1\n")
    df = load_data(f)
    assert df["income"].dtype.kind == "f"
    assert list(df["income"]) == [45.5, 52.1]


def test_semicolon_with_dot_decimals_keeps_dots(tmp_path: Path) -> None:
    """A ;-file that uses . decimals must not be 'corrected'."""
    f = tmp_path / "s.csv"
    f.write_text("age;income\n30;45.5\n41;52.1\n")
    df = load_data(f)
    assert list(df["income"]) == [45.5, 52.1]


def test_plain_comma_csv_unchanged(tmp_path: Path) -> None:
    """Regression guard: the sniffer must not disturb the common case,
    including commas inside quoted fields."""
    f = tmp_path / "p.csv"
    f.write_text('name,notes\nA,"hello, world"\nB,"x, y"\n')
    df = load_data(f)
    assert list(df.columns) == ["name", "notes"]
    assert df.iloc[0]["notes"] == "hello, world"


def test_all_views_agree_on_a_semicolon_file(tmp_path: Path) -> None:
    f = tmp_path / "s.csv"
    f.write_text("a;b\n1;2\n3;4\n5;6\n")
    assert row_count(f) == 3
    assert len(load_data(f)) == 3
    assert extract(f, "names_types_labels_summary")[
        "observation_count"] == 3
    prof = profile_dataset(f)
    assert prof["columns"] == 2 and prof["rows"] == 3


def test_sniffer_is_total_on_garbage(tmp_path: Path) -> None:
    f = tmp_path / "g.csv"
    f.write_bytes(bytes(range(256)) * 4)
    enc, sep, dec = text_table_params(f, ".csv")
    assert enc and sep and dec       # never raises, always answers


def test_sniffer_closes_its_bounded_input_handle() -> None:
    """Repeated schema/cloud inspection must not leak file descriptors."""
    handle = io.BytesIO(b"a,b\n1,2\n")

    class ProbePath:
        def open(self, _mode: str):
            return handle

    assert text_table_params(ProbePath(), ".csv") == ("utf-8", ",", ".")  # type: ignore[arg-type]
    assert handle.closed


# --------------------------------------------------------------------
# Sentinel missing values
# --------------------------------------------------------------------

def test_coded_missing_flagged_at_extreme_only(tmp_path: Path) -> None:
    pd.DataFrame({
        "income": [45000, 52000, -999, 61000, -999, 38000] * 20,
        "temp_c": [-5, -12, -9, 3, 8, -2] * 20,     # -9 real, inside range
    }).to_csv(tmp_path / "s.csv", index=False)
    prof = profile_dataset(tmp_path / "s.csv")
    flags = {v["name"]: v.get("possible_missing_code")
             for v in prof["variables"]}
    assert flags["income"] == -999
    assert flags["temp_c"] is None


def test_rare_sentinel_not_flagged(tmp_path: Path) -> None:
    """A single -999 among thousands is below the share threshold —
    flagging every odd value would train researchers to ignore the
    flag."""
    vals = list(range(1000, 4000)) + [-999]
    pd.DataFrame({"x": vals}).to_csv(tmp_path / "s.csv", index=False)
    prof = profile_dataset(tmp_path / "s.csv")
    assert prof["variables"][0].get("possible_missing_code") is None


# --------------------------------------------------------------------
# AI-use disclosure statement
# --------------------------------------------------------------------

def test_statement_is_specific_and_grammatical(tmp_path: Path) -> None:
    from sift import release_ledger, usage_meter

    (tmp_path / ".sift").mkdir()
    usage_meter.record_turn(tmp_path, model="claude-sonnet-5",
                            input_tokens=100, output_tokens=10)
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="submit_script",
        response={"content": [{"type": "text", "text": "{}"}]})
    stmt = ai_use_statement(tmp_path)
    assert "claude-sonnet-5" in stmt
    assert "across 1 interactive turn" in stmt
    assert "turn " in stmt or stmt.endswith("turn.") or "turns" not in \
        stmt.split("interactive")[1][:12]
    assert "1 disclosure-controlled release to the model" in stmt
    # Honesty anchors that must never be edited away:
    assert "raw data remained on the researcher's machine" in stmt
    assert "authors reviewed all analyses and remain responsible" in stmt
    assert "generated from the session's own records" in stmt


def test_statement_omits_rather_than_guesses(tmp_path: Path) -> None:
    """An empty session yields a generic-model, no-turn-count
    statement — not fabricated specifics."""
    (tmp_path / ".sift").mkdir()
    stmt = ai_use_statement(tmp_path)
    assert "a large language model" in stmt
    assert "interactive turn" not in stmt
    assert "0 disclosure-controlled releases" in stmt
