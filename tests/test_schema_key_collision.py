"""schema.py's ``_extract_pyreadstat`` value-label extraction must not
silently drop entries when two distinct raw codes truncate to the same
``safe_key`` (see the collision-handling fix in
``sift.text_safety.safe_keys_sequence``).

Constructing a REAL .dta/.sav file with value-label codes long enough
to collide after the 40-char safe_key cap isn't practical (Stata's
numeric value-label codes are bounded by its storage types; SPSS
string-variable codes could in principle be long enough, but building
one through pyreadstat's writer for this one edge case adds more
moving parts than it's worth). Instead this drives
``_extract_pyreadstat`` directly with a fake reader returning the
exact ``(df, meta)`` shape pyreadstat produces — the same "reader is
just a callable" interface ``_extract_stata``/``_extract_spss``/etc.
already use, so this exercises the REAL extraction code path, not a
reimplementation of it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from sift.schema import _extract_pyreadstat


def _fake_meta(
    *, column_names: list[str], value_labels: dict[str, dict[Any, str]],
    variable_to_label: dict[str, str], number_rows: int | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        column_names=column_names,
        column_labels=[None] * len(column_names),
        variable_to_label=variable_to_label,
        value_labels=value_labels,
        readstat_variable_types={c: "int32" for c in column_names},
        number_rows=number_rows,
    )


def test_value_labels_with_colliding_codes_are_not_silently_dropped(
) -> None:
    """Two distinct raw value-label codes sharing a >40-char prefix
    must BOTH survive extraction, under distinguishable keys — not
    collapse to one via a naive dict-comprehension overwrite."""
    long_prefix = "9" * 45
    code_a = long_prefix + "1"
    code_b = long_prefix + "2"

    df = pd.DataFrame({"status": [1, 2]})
    meta = _fake_meta(
        column_names=["status"],
        value_labels={"status_labels": {code_a: "Alpha", code_b: "Beta"}},
        variable_to_label={"status": "status_labels"},
        number_rows=2,
    )

    def fake_reader(path: str, metadataonly: bool = False):
        return df, meta

    out = _extract_pyreadstat(
        Path("/fake/path.dta"), "names_types_labels_summary", fake_reader,
    )
    status_var = next(v for v in out["variables"] if v["name"] == "status")
    labels = status_var["value_labels"]

    # Both original labels must be present somewhere in the output —
    # the bug this pins would have silently dropped one.
    assert set(labels.values()) == {"Alpha", "Beta"}
    assert len(labels) == 2
    # And every emitted key still respects the safe_key length cap.
    assert all(len(k) <= 40 for k in labels)


def test_value_labels_without_collision_are_unaffected() -> None:
    """Negative control: normal short numeric codes behave exactly as
    before — plain safe_key output, no spurious disambiguation."""
    df = pd.DataFrame({"region": [1, 2]})
    meta = _fake_meta(
        column_names=["region"],
        value_labels={"region_labels": {1: "North", 2: "South"}},
        variable_to_label={"region": "region_labels"},
        number_rows=2,
    )

    def fake_reader(path: str, metadataonly: bool = False):
        return df, meta

    out = _extract_pyreadstat(
        Path("/fake/path.dta"), "names_types_labels_summary", fake_reader,
    )
    region_var = next(v for v in out["variables"] if v["name"] == "region")
    assert region_var["value_labels"] == {"1": "North", "2": "South"}


def test_variable_names_with_colliding_prefix_are_disambiguated() -> None:
    """Same collision protection for the variable NAME list itself
    (not just value labels) — two column names sharing a >40-char
    prefix must render as two distinguishable entries, not two
    identical-looking "name" strings a later request_data call could
    resolve ambiguously."""
    long_prefix = "v" * 45
    name_a = long_prefix + "A"
    name_b = long_prefix + "B"

    df = pd.DataFrame({name_a: [1], name_b: [2]})
    meta = _fake_meta(
        column_names=[name_a, name_b],
        value_labels={},
        variable_to_label={},
        number_rows=1,
    )

    def fake_reader(path: str, metadataonly: bool = False):
        return df, meta

    out = _extract_pyreadstat(
        Path("/fake/path.dta"), "names_types_labels_summary", fake_reader,
    )
    names = [v["name"] for v in out["variables"]]
    assert len(names) == 2
    assert names[0] != names[1]
    assert all(len(n) <= 40 for n in names)


def test_shared_reader_preserves_format_and_unknown_row_count() -> None:
    """The pyreadstat backend serves four product formats. Its payload must
    identify the caller-selected format, and metadata-only readers that do
    not publish N must return JSON null instead of crashing or claiming 0."""
    meta = _fake_meta(
        column_names=["subject_id"],
        value_labels={},
        variable_to_label={},
        number_rows=None,
    )

    def fake_reader(path: str, metadataonly: bool = False):
        assert metadataonly is True
        return pd.DataFrame(), meta

    out = _extract_pyreadstat(
        Path("/fake/trial.xpt"),
        "names_types",
        fake_reader,
        file_type="sas_xport",
    )
    assert out["file_type"] == "sas_xport"
    assert out["observation_count"] is None
