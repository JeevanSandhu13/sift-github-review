"""Parquet column projection.

``request_data`` (data_request.py) only ever needs one or two columns
out of a dataset to answer a bounded fact — but used to load every
column of a parquet file to get them. For a wide file that's real
wasted I/O and memory. ``_parquet_projection_columns`` resolves the
needed column(s) against the file's own schema (metadata only, no row
data read) so ``load_data`` can request just those columns from
pyarrow.

The load-bearing property under test isn't the performance win itself
(hard to assert from a unit test) but that projection can NEVER change
what ``handle()`` returns or which requests it denies — every case
here compares the projected path's result against the same computation
without projection, and confirms the denial paths (unknown variable,
sanitized-name collision) come back identically whether or not
projection could apply.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sift.data_request import _parquet_projection_columns, handle
from sift.sanitizer import SDCConfig


def _wide_parquet(tmp_path: Path, n_extra_cols: int = 40) -> Path:
    """A parquet file with two 'real' columns and many filler columns,
    wide enough that loading everything vs. two columns is a
    meaningfully different amount of data."""
    rng = np.random.default_rng(0)
    n = 300
    data = {
        "income": rng.normal(50000, 10000, size=n).round(2),
        "age": rng.integers(18, 80, size=n),
        "category": (["common"] * 270 + ["medium"] * 25 + ["rare"] * 5),
    }
    for i in range(n_extra_cols):
        data[f"filler_{i}"] = rng.normal(0, 1, size=n)
    df = pd.DataFrame(data)
    path = tmp_path / "wide.parquet"
    df.to_parquet(path)
    return path


@pytest.fixture()
def wide_parquet(tmp_path: Path) -> Path:
    pytest.importorskip("pyarrow")
    return _wide_parquet(tmp_path)


# ---------------------------------------------------------------------------
# _parquet_projection_columns — the planning function
# ---------------------------------------------------------------------------

def test_projects_single_variable(wide_parquet) -> None:
    cols = _parquet_projection_columns(wide_parquet, "income", "numeric_bounds", None)
    assert cols == ["income"]


def test_projects_both_variables_for_correlation_pair(wide_parquet) -> None:
    cols = _parquet_projection_columns(
        wide_parquet, "income", "correlation_pair", "age",
    )
    assert cols == ["age", "income"]


def test_non_correlation_request_ignores_variable2(wide_parquet) -> None:
    """variable2 is only relevant to correlation_pair; for any other
    request type it must not be pulled into the projection even if a
    caller happens to pass it."""
    cols = _parquet_projection_columns(wide_parquet, "income", "numeric_bounds", "age")
    assert cols == ["income"]


def test_returns_none_for_non_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(path, index=False)
    assert _parquet_projection_columns(path, "a", "numeric_bounds", None) is None


def test_returns_none_for_unknown_variable(wide_parquet) -> None:
    """No column named this — projection declines rather than guessing
    or raising; handle() falls back to the full-load path and
    produces the ordinary 'not found' denial there."""
    assert _parquet_projection_columns(
        wide_parquet, "does_not_exist", "numeric_bounds", None,
    ) is None


def test_returns_none_for_missing_correlation_partner(wide_parquet) -> None:
    """correlation_pair with variable1 resolvable but variable2 not —
    projection must decline entirely (not project variable1 alone),
    or _correlation_pair's own denial for the missing var2 would run
    against a frame that never had var2's column loaded, which is a
    different (wrong) failure mode than 'column not found'."""
    assert _parquet_projection_columns(
        wide_parquet, "income", "correlation_pair", "does_not_exist",
    ) is None


def test_returns_none_when_schema_cannot_be_read(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.parquet"
    bad.write_bytes(b"not actually parquet")
    assert _parquet_projection_columns(bad, "income", "numeric_bounds", None) is None


# ---------------------------------------------------------------------------
# handle() end-to-end: projection must never change the answer
# ---------------------------------------------------------------------------

def test_numeric_bounds_result_identical_with_and_without_projection(
    wide_parquet, tmp_path: Path,
) -> None:
    projected = handle(wide_parquet, "numeric_bounds", "income")

    # Force the non-projected path by pointing at a copy with a
    # non-.parquet suffix trick is awkward; instead directly disable
    # projection by patching it to return None, isolating "did
    # projection change the answer" from "does load_data(columns=...)
    # work at all" (covered separately below).
    with patch(
        "sift.data_request._parquet_projection_columns", return_value=None,
    ):
        unprojected = handle(wide_parquet, "numeric_bounds", "income")

    assert projected.status == unprojected.status == "granted"
    assert projected.answer == unprojected.answer


def test_correlation_pair_result_identical_with_and_without_projection(
    wide_parquet,
) -> None:
    projected = handle(
        wide_parquet, "correlation_pair", "income", variable2="age",
    )
    with patch(
        "sift.data_request._parquet_projection_columns", return_value=None,
    ):
        unprojected = handle(
            wide_parquet, "correlation_pair", "income", variable2="age",
        )
    assert projected.status == unprojected.status
    assert projected.answer == unprojected.answer


def test_unknown_variable_denial_identical_with_and_without_projection(
    wide_parquet,
) -> None:
    projected = handle(wide_parquet, "numeric_bounds", "nonexistent_column")
    with patch(
        "sift.data_request._parquet_projection_columns", return_value=None,
    ):
        unprojected = handle(wide_parquet, "numeric_bounds", "nonexistent_column")
    assert projected.status == unprojected.status == "denied"
    assert projected.reason == unprojected.reason


def test_load_data_actually_receives_the_narrow_projection(wide_parquet) -> None:
    """Confirms the projection plan is really wired through to the
    parquet reader, not just computed and discarded — spies on
    pandas.read_parquet's columns= argument. ``load_data`` does
    ``import pandas as pd`` locally (inside the function), so patching
    the top-level ``pandas.read_parquet`` name (rather than a
    module-level import inside ``sift.schema``) is what actually
    intercepts the call."""
    calls = []
    real_read_parquet = pd.read_parquet

    def spy(path, columns=None, **kwargs):
        calls.append(columns)
        return real_read_parquet(path, columns=columns, **kwargs)

    with patch("pandas.read_parquet", side_effect=spy):
        result = handle(wide_parquet, "numeric_bounds", "income")
    assert result.status == "granted"
    assert calls == [["income"]]


def test_correlation_pair_projection_includes_both_columns_in_read(wide_parquet) -> None:
    import pandas as real_pd

    calls = []
    real_read_parquet = real_pd.read_parquet

    def spy(path, columns=None, **kwargs):
        calls.append(columns)
        return real_read_parquet(path, columns=columns, **kwargs)

    with patch("pandas.read_parquet", side_effect=spy):
        result = handle(
            wide_parquet, "correlation_pair", "income", variable2="age",
        )
    assert result.status == "granted"
    assert calls == [["age", "income"]]


def test_full_load_path_unaffected_for_non_parquet_formats(tmp_path: Path) -> None:
    """CSV (and every other format) never had column projection and
    still doesn't — this is a parquet-only optimization by design."""
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({
        "income": rng.normal(50000, 10000, size=n).round(2),
        "category": (["common"] * 180 + ["rare"] * 20),
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = handle(path, "numeric_bounds", "income")
    assert result.status == "granted"


def test_self_correlation_denied_identically_with_and_without_projection(
    wide_parquet,
) -> None:
    """``variable == variable2`` is refused by the SDC layer as
    structurally redundant (r=1 always) -- unrelated to projection,
    but must produce the identical denial whether or not projection
    resolves the (degenerate, single-column) plan first."""
    projected = handle(
        wide_parquet, "correlation_pair", "income", variable2="income",
    )
    with patch(
        "sift.data_request._parquet_projection_columns", return_value=None,
    ):
        unprojected = handle(
            wide_parquet, "correlation_pair", "income", variable2="income",
        )
    assert projected.status == unprojected.status == "denied"
    assert projected.reason == unprojected.reason
    assert "must differ" in projected.reason


def test_sanitized_name_collision_denied_identically_with_projection(
    tmp_path: Path,
) -> None:
    """Two raw columns ("A B" and "A\nB") that sanitize to the same
    safe_key -- the ambiguous-collision path _resolve_variable takes
    for a real loaded frame. ``_parquet_projection_columns`` must
    decline (return None) rather than guess which raw column to
    project, so the full-load path re-resolves and produces the
    SAME collision denial -- never a silently-picked wrong column."""
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"A B": [1, 2, 3], "A\nB": [4, 5, 6], "other": [7, 8, 9]})
    path = tmp_path / "collide.parquet"
    df.to_parquet(path)

    assert _parquet_projection_columns(path, "A B", "numeric_bounds", None) is None

    projected = handle(path, "numeric_bounds", "A B")
    with patch(
        "sift.data_request._parquet_projection_columns", return_value=None,
    ):
        unprojected = handle(path, "numeric_bounds", "A B")
    assert projected.status == unprojected.status == "denied"
    assert projected.reason == unprojected.reason
    assert "collide" in projected.reason


def test_duplicate_raw_schema_field_names_decline_projection(
    tmp_path: Path,
) -> None:
    """A parquet file with two physically identical field names (a
    pathological, non-standard-writer case pyarrow nonetheless
    permits at the schema level) makes ``_resolve_variable`` see two
    literal matches -- an ambiguous collision exactly like the
    sanitized-name case above, so projection must decline the same
    way. The subsequent full-load ALSO fails on this file (an
    inherent pyarrow limitation reading a duplicate-field-name
    schema at all, reproduced here even via a vanilla
    ``pd.read_parquet`` with no column selection whatsoever) --
    confirming the failure is not something projection introduced,
    and that it surfaces through handle()'s existing generic
    error path rather than crashing uncaught."""
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema([("income", pa.float64()), ("income", pa.int64())])
    table = pa.table(
        [pa.array([1.0] * 60), pa.array([20] * 60)], schema=schema,
    )
    path = tmp_path / "dup_schema.parquet"
    pq.write_table(table, path)

    assert _parquet_projection_columns(
        path, "income", "numeric_bounds", None,
    ) is None

    result = handle(path, "numeric_bounds", "income")
    assert result.status == "error"
    assert "ArrowInvalid" in result.reason


def test_excel_sheet_config_passed_through_to_load_data(tmp_path: Path) -> None:
    """SDCConfig.excel_sheet rides through handle() -> load_data() the
    same way dp_epsilon does — a request against a non-default sheet
    reads that sheet's data, not the first sheet's."""
    path = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"income": [1.0, 2.0, 3.0]}).to_excel(
            writer, sheet_name="Sheet1", index=False,
        )
        pd.DataFrame({"income": [100.0] * 30 + [200.0] * 30}).to_excel(
            writer, sheet_name="Real", index=False,
        )

    default_sheet = handle(path, "numeric_bounds", "income")
    selected_sheet = handle(
        path, "numeric_bounds", "income",
        config=SDCConfig(excel_sheet="Real"),
    )
    assert default_sheet.status == "denied"  # only 3 rows on Sheet1
    assert selected_sheet.status == "granted"
