"""Arrow-family research formats share schema, count, load and profile views."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
feather = pytest.importorskip("pyarrow.feather")
ipc = pytest.importorskip("pyarrow.ipc")
orc = pytest.importorskip("pyarrow.orc")

from sift.data_request import handle
from sift.dataset_profile import profile_dataset
from sift.schema import DATA_EXTENSIONS, extract, load_data, row_count


@pytest.fixture(params=["feather", "arrow", "ipc", "orc"])
def arrow_dataset(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    frame = pd.DataFrame(
        {
            "subject_id": range(40),
            "score": [float(value) for value in range(40)],
            "arm": ["control", "treatment"] * 20,
        }
    )
    table = pa.Table.from_pandas(frame, preserve_index=False)
    path = tmp_path / f"trial.{request.param}"
    if request.param == "feather":
        feather.write_feather(table, path)
    elif request.param in {"arrow", "ipc"}:
        with (
            pa.OSFile(str(path), "wb") as sink,
            ipc.new_file(sink, table.schema) as writer,
        ):
            writer.write_table(table)
    else:
        orc.write_table(table, path)
        try:
            orc.ORCFile(path).read().to_pandas()
        except OSError as exc:
            if "sysctlbyname failed" in str(exc):
                pytest.skip("outer macOS test sandbox blocks pyarrow ORC CPU discovery")
            raise
    return path


def test_arrow_extensions_registered() -> None:
    for extension in (".feather", ".arrow", ".ipc", ".orc"):
        assert extension in DATA_EXTENSIONS


def test_arrow_views_agree(arrow_dataset: Path) -> None:
    assert row_count(arrow_dataset) == 40
    frame = load_data(arrow_dataset)
    assert list(frame.columns) == ["subject_id", "score", "arm"]
    assert len(frame) == 40
    names = extract(arrow_dataset, "names_only")
    assert [variable["name"] for variable in names["variables"]] == [
        "subject_id",
        "score",
        "arm",
    ]
    summary = extract(arrow_dataset, "names_types_labels_summary")
    assert summary["observation_count"] == 40


def test_arrow_projection_and_sanitized_request(arrow_dataset: Path) -> None:
    projected = load_data(arrow_dataset, columns=["score"])
    assert list(projected.columns) == ["score"]
    result = handle(arrow_dataset, "quartiles", "score")
    assert result.status == "granted"


def test_arrow_local_profile(arrow_dataset: Path) -> None:
    profile = profile_dataset(arrow_dataset)
    assert profile["ok"] is True
    assert profile["rows"] == 40
    assert profile["columns"] == 3


@pytest.mark.parametrize("extension", ["arrow", "ipc"])
def test_arrow_stream_format_works_across_every_view(
    tmp_path: Path,
    extension: str,
) -> None:
    table = pa.table({"id": range(25), "score": [float(v) for v in range(25)]})
    path = tmp_path / f"stream.{extension}"
    with (
        pa.OSFile(str(path), "wb") as sink,
        ipc.new_stream(sink, table.schema) as writer,
    ):
        writer.write_batch(table.to_batches(max_chunksize=10)[0])
        for batch in table.to_batches(max_chunksize=10)[1:]:
            writer.write_batch(batch)

    assert row_count(path) == 25
    assert list(load_data(path).columns) == ["id", "score"]
    assert [v["name"] for v in extract(path, "names_only")["variables"]] == [
        "id", "score",
    ]
    profile = profile_dataset(path)
    assert profile["ok"] is True
    assert profile["rows"] == 25
