"""R workspace ingestion is explicit and reproducible."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pyreadr = pytest.importorskip("pyreadr")

from sift.schema import DATA_EXTENSIONS, extract, load_data, row_count


@pytest.mark.parametrize("extension", [".rda", ".RData"])
def test_single_dataframe_r_workspace_is_first_class(
    tmp_path: Path,
    extension: str,
) -> None:
    path = tmp_path / f"study{extension}"
    expected = pd.DataFrame({"id": [1, 2, 3], "score": [2.5, 3.5, 4.5]})
    pyreadr.write_rdata(str(path), expected, df_name="cohort")

    assert path.suffix.lower() in DATA_EXTENSIONS
    loaded = load_data(path)
    assert list(loaded.columns) == ["id", "score"]
    assert row_count(path) == 3
    schema = extract(path, "names_types")
    assert schema["file_type"] == "r_workspace"
    assert [row["name"] for row in schema["variables"]] == ["id", "score"]

