from __future__ import annotations

from sift.filename_safety import portable_filename, portable_stem


def test_long_portable_filename_preserves_extension() -> None:
    name = portable_filename(f"{'x' * 400}.parquet")
    assert len(name) == 160
    assert name.endswith(".parquet")


def test_windows_device_names_are_neutralized_with_extensions() -> None:
    assert portable_filename("CON.csv") == "_CON.csv"
    assert portable_filename("lpt9.PARQUET") == "_lpt9.PARQUET"
    assert portable_stem("nul") == "_nul"


def test_windows_illegal_characters_and_trailing_dots_are_removed() -> None:
    assert portable_filename('study<2026>:final?.csv. ') == (
        "study_2026__final_.csv"
    )
