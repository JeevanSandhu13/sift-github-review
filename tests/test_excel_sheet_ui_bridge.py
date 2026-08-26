"""Excel sheet selection in the UI bridge.

Covers the researcher-facing surface: ``SiftBridge.get_excel_sheets``
(list worksheets), ``SiftBridge.set_dataset_excel_sheet`` (save a
choice, preserving every other independent policy axis — the exact
same cross-axis reset bug class as ``dp_epsilon``, guarded here
from the moment the new axis was added), and
``SiftBridge.get_dataset_profile`` picking up the saved choice (or
honouring an explicit preview override without persisting it).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sift.policy import DatasetPolicy, SiftPolicy, load_policy, save_policy
from sift.ui import SiftBridge


@pytest.fixture()
def multi_sheet_workbook(tmp_path: Path) -> Path:
    path = tmp_path / "workbook.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"a": [1, 2, 3]}).to_excel(
            writer, sheet_name="Sheet1", index=False,
        )
        pd.DataFrame({"b": [10, 20]}).to_excel(
            writer, sheet_name="Budget2026", index=False,
        )
    return path


# ---------------------------------------------------------------------------
# get_excel_sheets
# ---------------------------------------------------------------------------

def test_get_excel_sheets_lists_names(tmp_path: Path, multi_sheet_workbook) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_excel_sheets("workbook.xlsx")
    assert result["ok"] is True
    assert result["sheets"] == ["Sheet1", "Budget2026"]


def test_get_excel_sheets_rejects_non_xlsx(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n")
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_excel_sheets("data.csv")
    assert result["ok"] is False


def test_get_excel_sheets_rejects_path_outside_session(tmp_path: Path) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_excel_sheets("../outside.xlsx")
    assert result["ok"] is False


def test_get_excel_sheets_missing_file(tmp_path: Path) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_excel_sheets("nope.xlsx")
    assert result["ok"] is False


# ---------------------------------------------------------------------------
# set_dataset_excel_sheet
# ---------------------------------------------------------------------------

def test_set_dataset_excel_sheet_saves_choice(tmp_path: Path) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_excel_sheet("workbook.xlsx", "Budget2026")
    assert result["ok"] is True
    entry = load_policy(tmp_path).datasets["workbook.xlsx"]
    assert entry.excel_sheet == "Budget2026"


def test_set_dataset_excel_sheet_none_clears_and_can_collapse_entry(
    tmp_path: Path,
) -> None:
    save_policy(tmp_path, SiftPolicy(datasets={
        "workbook.xlsx": DatasetPolicy(
            excel_sheet="Budget2026", set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_excel_sheet("workbook.xlsx", None)
    assert result["ok"] is True
    reloaded = load_policy(tmp_path)
    # Every other axis was at default, so clearing the only non-default
    # axis should collapse the entry entirely — same rule every other
    # set_dataset_* method in ui.py follows.
    assert "workbook.xlsx" not in reloaded.datasets


def test_set_dataset_excel_sheet_preserves_other_axes(tmp_path: Path) -> None:
    save_policy(tmp_path, SiftPolicy(datasets={
        "workbook.xlsx": DatasetPolicy(
            banned_variables=("ssn",), exportable=False, dp_epsilon=0.3,
            privacy_profile="regulated",
            set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_excel_sheet("workbook.xlsx", "Budget2026")
    assert result["ok"] is True
    entry = load_policy(tmp_path).datasets["workbook.xlsx"]
    assert entry.banned_variables == ("ssn",)
    assert entry.exportable is False
    assert entry.dp_epsilon == 0.3
    assert entry.privacy_profile == "regulated"
    assert entry.excel_sheet == "Budget2026"


# ---------------------------------------------------------------------------
# The other three set_dataset_* methods must not wipe excel_sheet —
# same cross-axis reset bug class as dp_epsilon.
# ---------------------------------------------------------------------------

def test_set_dataset_policy_preserves_excel_sheet(tmp_path: Path) -> None:
    save_policy(tmp_path, SiftPolicy(datasets={
        "workbook.xlsx": DatasetPolicy(
            excel_sheet="Budget2026", set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_policy("workbook.xlsx", "names_types_labels")
    assert result["ok"] is True
    entry = load_policy(tmp_path).datasets["workbook.xlsx"]
    assert entry.excel_sheet == "Budget2026"


def test_set_dataset_privacy_profile_preserves_excel_sheet(tmp_path: Path) -> None:
    save_policy(tmp_path, SiftPolicy(datasets={
        "workbook.xlsx": DatasetPolicy(
            excel_sheet="Budget2026", set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_privacy_profile("workbook.xlsx", "confidential")
    assert result["ok"] is True
    entry = load_policy(tmp_path).datasets["workbook.xlsx"]
    assert entry.excel_sheet == "Budget2026"


def test_set_dataset_dp_epsilon_preserves_excel_sheet(tmp_path: Path) -> None:
    save_policy(tmp_path, SiftPolicy(datasets={
        "workbook.xlsx": DatasetPolicy(
            excel_sheet="Budget2026", set_at="2026-01-01T00:00:00+00:00",
        ),
    }))
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.set_dataset_dp_epsilon("workbook.xlsx", 0.5)
    assert result["ok"] is True
    entry = load_policy(tmp_path).datasets["workbook.xlsx"]
    assert entry.excel_sheet == "Budget2026"
    assert entry.dp_epsilon == 0.5


# ---------------------------------------------------------------------------
# get_dataset_profile: saved choice vs. explicit preview override
# ---------------------------------------------------------------------------

def test_get_dataset_profile_uses_saved_sheet(
    tmp_path: Path, multi_sheet_workbook,
) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    bridge.set_dataset_excel_sheet("workbook.xlsx", "Budget2026")
    result = bridge.get_dataset_profile("workbook.xlsx")
    assert result["ok"] is True
    names = [v["name"] for v in result["variables"]]
    assert names == ["b"]
    assert result["sheet_read"] == "Budget2026"


def test_get_dataset_profile_defaults_to_first_sheet_when_unset(
    tmp_path: Path, multi_sheet_workbook,
) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_dataset_profile("workbook.xlsx")
    assert result["ok"] is True
    names = [v["name"] for v in result["variables"]]
    assert names == ["a"]
    assert result["sheet_read"] == 0


def test_get_dataset_profile_explicit_sheet_previews_without_saving(
    tmp_path: Path, multi_sheet_workbook,
) -> None:
    """Passing sheet= directly previews that sheet but must NOT
    persist it — the researcher can page through sheets before
    picking one, and only set_dataset_excel_sheet commits a choice."""
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_dataset_profile("workbook.xlsx", sheet="Budget2026")
    assert result["ok"] is True
    names = [v["name"] for v in result["variables"]]
    assert names == ["b"]

    policy = load_policy(tmp_path)
    assert "workbook.xlsx" not in policy.datasets

    # A follow-up call with no override reverts to the (still unset)
    # default -- confirms the preview really wasn't saved.
    default_result = bridge.get_dataset_profile("workbook.xlsx")
    assert [v["name"] for v in default_result["variables"]] == ["a"]


def test_available_sheets_surfaced_in_profile(
    tmp_path: Path, multi_sheet_workbook,
) -> None:
    bridge = SiftBridge()
    bridge.cwd = tmp_path
    result = bridge.get_dataset_profile("workbook.xlsx")
    assert result["available_sheets"] == ["Sheet1", "Budget2026"]
