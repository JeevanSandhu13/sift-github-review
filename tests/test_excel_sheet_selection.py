"""Excel sheet selection.

Every ``.xlsx`` reader in this codebase defaulted to the first
worksheet with no way to pick another. These tests cover the three
new pieces: ``schema.list_excel_sheets`` (cheap sheet-name listing),
``schema.load_data`` / ``schema.extract`` honouring an explicit
``sheet`` argument, and — because ``sheet`` defaults to ``None``
everywhere — that every pre-existing call site with no opinion on the
matter still reads the first worksheet exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sift import schema


@pytest.fixture()
def multi_sheet_workbook(tmp_path: Path) -> Path:
    path = tmp_path / "workbook.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}).to_excel(
            writer, sheet_name="Sheet1", index=False,
        )
        pd.DataFrame({"c": [10, 20], "d": ["p", "q"]}).to_excel(
            writer, sheet_name="SecondSheet", index=False,
        )
        pd.DataFrame({"e": [99]}).to_excel(
            writer, sheet_name="ThirdSheet", index=False,
        )
    return path


def test_list_excel_sheets_returns_names_in_order(multi_sheet_workbook) -> None:
    assert schema.list_excel_sheets(multi_sheet_workbook) == [
        "Sheet1", "SecondSheet", "ThirdSheet",
    ]


def test_load_data_defaults_to_first_sheet(multi_sheet_workbook) -> None:
    df = schema.load_data(multi_sheet_workbook)
    assert list(df.columns) == ["a", "b"]


def test_load_data_honors_explicit_sheet_by_name(multi_sheet_workbook) -> None:
    df = schema.load_data(multi_sheet_workbook, sheet="SecondSheet")
    assert list(df.columns) == ["c", "d"]
    assert df["c"].tolist() == [10, 20]


def test_load_data_honors_explicit_sheet_by_index(multi_sheet_workbook) -> None:
    df = schema.load_data(multi_sheet_workbook, sheet=2)
    assert list(df.columns) == ["e"]


def test_extract_reports_sheet_read_and_available_sheets(multi_sheet_workbook) -> None:
    result = schema.extract(multi_sheet_workbook, "names_types")
    assert result["sheet_read"] == 0
    assert result["available_sheets"] == ["Sheet1", "SecondSheet", "ThirdSheet"]


def test_extract_with_explicit_sheet_reads_that_sheet(multi_sheet_workbook) -> None:
    result = schema.extract(
        multi_sheet_workbook, "names_types", sheet="SecondSheet",
    )
    assert result["sheet_read"] == "SecondSheet"
    names = [v["name"] for v in result["variables"]]
    assert names == ["c", "d"]


def test_extract_summary_depth_with_explicit_sheet(multi_sheet_workbook) -> None:
    result = schema.extract(
        multi_sheet_workbook, "names_types_labels_summary", sheet="ThirdSheet",
    )
    assert result["sheet_read"] == "ThirdSheet"
    names = [v["name"] for v in result["variables"]]
    assert names == ["e"]


def test_single_sheet_workbook_unaffected(tmp_path: Path) -> None:
    """The overwhelmingly common case (one worksheet) must produce
    byte-identical schema output to before this feature existed."""
    path = tmp_path / "single.xlsx"
    pd.DataFrame({"x": [1, 2], "y": [3, 4]}).to_excel(path, index=False)
    result = schema.extract(path, "names_types")
    assert result["sheet_read"] == 0
    assert result["available_sheets"] == ["Sheet1"]
    assert [v["name"] for v in result["variables"]] == ["x", "y"]


@pytest.mark.parametrize("suffix", [".xls", ".ods"])
def test_legacy_and_open_spreadsheets_use_same_typed_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, suffix: str,
) -> None:
    """Dispatch symmetry without requiring the optional writer engines in
    the test environment.  pandas is still the single parsing/type path."""
    path = tmp_path / f"study{suffix}"
    path.write_bytes(b"fixture")
    calls: list[dict] = []

    def fake_read_excel(*args, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame({"subject_id": [1, 2], "score": [3.5, 4.5]})

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    monkeypatch.setattr(schema, "_guard_full_load", lambda *a, **k: None)
    loaded = schema.load_data(path, sheet="Data")
    extracted = schema.extract(path, "names_types", sheet="Data")

    assert list(loaded.columns) == ["subject_id", "score"]
    assert extracted["file_type"] == suffix.removeprefix(".")
    assert extracted["sheet_read"] == "Data"
    assert calls[0]["sheet_name"] == "Data"
    assert calls[1]["sheet_name"] == "Data"
    assert calls[1]["nrows"] == 200
