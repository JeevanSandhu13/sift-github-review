"""Local dataset profile — correctness and the privacy separation.

The profile deliberately contains information the SDC boundary
withholds from the model (exact distinct counts, per-variable min and
max). That is only safe because it is researcher-local. The most
important test here is therefore the structural one: the profile must
be unreachable from the model's tool surface.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sift.dataset_profile import profile_dataset


@pytest.fixture()
def frame():
    pd = pytest.importorskip("pandas")
    return pd


def _write_csv(tmp_path: Path, pd) -> Path:
    df = pd.DataFrame({
        "customer_id": range(100),
        "age": [None] * 10 + list(range(20, 110)),
        "region": ["north", "south"] * 50,
        "site": ["HQ"] * 100,
        "notes": [None] * 100,
    })
    path = tmp_path / "customers.csv"
    df.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------
# Privacy separation — the load-bearing property
# --------------------------------------------------------------------

def test_profile_is_not_reachable_from_any_tool() -> None:
    """The model must have no way to invoke the profile. If this ever
    fails, per-variable min/max and exact distinct counts have become
    model-visible, bypassing the SDC boundary."""
    from sift.tools import ALLOWED_TOOL_NAMES, HANDLERS

    joined = " ".join(ALLOWED_TOOL_NAMES) + " " + " ".join(HANDLERS)
    assert "profile" not in joined.lower()


def test_tool_layer_does_not_import_the_profile_module() -> None:
    """Structural guard: no import means no accidental tool response."""
    src = Path("src/sift/tools.py").read_text(encoding="utf-8")
    assert "dataset_profile" not in src


# --------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------

def test_profile_reports_shape_and_missingness(tmp_path, frame) -> None:
    prof = profile_dataset(_write_csv(tmp_path, frame))
    assert prof["ok"] is True
    assert prof["rows"] == 100
    assert prof["columns"] == 5
    assert prof["rows_exact"] is True
    assert prof["sampled"] is False
    assert prof["missing_pct"] > 0
    by_name = {v["name"]: v for v in prof["variables"]}
    assert by_name["age"]["missing"] == 10
    assert by_name["age"]["missing_pct"] == 10.0


def test_structural_flags(tmp_path, frame) -> None:
    prof = profile_dataset(_write_csv(tmp_path, frame))
    assert "customer_id" in prof["likely_identifiers"]
    assert "site" in prof["constant_columns"]
    assert "notes" in prof["all_missing_columns"]
    assert prof["duplicate_rows"] == 0


def test_duplicate_detection(tmp_path, frame) -> None:
    df = frame.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    path = tmp_path / "dupes.csv"
    df.to_csv(path, index=False)
    assert profile_dataset(path)["duplicate_rows"] == 1


def test_numeric_range_present_text_range_absent(tmp_path, frame) -> None:
    prof = profile_dataset(_write_csv(tmp_path, frame))
    by_name = {v["name"]: v for v in prof["variables"]}
    assert "min" in by_name["age"] and "max" in by_name["age"]
    assert "min" not in by_name["region"]


def test_large_file_is_sampled_not_refused(tmp_path, frame, monkeypatch) -> None:
    """A file over the load ceiling still profiles, from a bounded
    sample, and says so — a partial answer beats no answer."""
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "512")
    df = frame.DataFrame({"a": range(400), "b": ["x"] * 400})
    path = tmp_path / "big.csv"
    df.to_csv(path, index=False)
    prof = profile_dataset(path)
    assert prof["ok"] is True
    assert prof["sampled"] is True
    # Duplicate count and identifier flags are withheld on a sample
    # rather than reported from partial evidence.
    assert prof["duplicate_rows"] is None
    assert prof["likely_identifiers"] == []
    # True N still reported via the cheap row-count path.
    assert prof["rows"] == 400


def test_bad_inputs_return_reason_not_exception(tmp_path) -> None:
    assert profile_dataset(tmp_path / "nope.csv")["ok"] is False
    weird = tmp_path / "notes.xyz"
    weird.write_text("hello")
    out = profile_dataset(weird)
    assert out["ok"] is False and "unsupported" in out["reason"]


def test_malformed_file_reports_cleanly(tmp_path) -> None:
    bad = tmp_path / "broken.parquet"
    bad.write_bytes(b"this is not parquet")
    out = profile_dataset(bad)
    assert out["ok"] is False and "could not read" in out["reason"]


def test_column_names_are_text_safety_sanitized(tmp_path, frame) -> None:
    """Column names reach the DOM, so they get the same treatment as
    every other data-origin string."""
    df = frame.DataFrame({"ok_col": [1, 2], "bad‮col": [3, 4]})
    path = tmp_path / "hostile.csv"
    df.to_csv(path, index=False)
    prof = profile_dataset(path)
    names = [v["name"] for v in prof["variables"]]
    assert all("‮" not in n for n in names)


def test_unchanged_profile_is_cached_but_return_values_are_isolated(
    tmp_path, frame, monkeypatch,
) -> None:
    """Repeated panel opens should not reread an unchanged dataset."""
    from sift import dataset_profile as module

    module._clear_profile_cache()
    path = _write_csv(tmp_path, frame)
    real = module._read_frame
    calls = {"n": 0}

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_read_frame", counted)
    first = profile_dataset(path)
    first["name"] = "mutated by caller"
    second = profile_dataset(path)

    assert calls["n"] == 1
    assert second["name"] == path.name
