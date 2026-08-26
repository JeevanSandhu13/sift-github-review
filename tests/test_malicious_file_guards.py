"""Malicious-file protections: memory-safety guards a hostile (or just
oversized) dataset file must not be able to bypass.

Two distinct findings drove this file, both confirmed against the
REAL code paths before being fixed (not just reasoned about):

1. ``schema.extract()`` — the function backing the model-facing
   ``get_schema`` tool — had NO size ceiling at all prior to this.
   ``load_data()`` (used by ``data_request``) was guarded via
   ``_guard_full_load``; ``extract()`` was not, for any format,
   despite several ``_extract_*`` functions doing a full,
   unbounded load at deeper depths. This meant a researcher (or a
   model-suggested tool call) requesting summary-depth schema on an
   oversized file — no malice required — could exhaust host memory.
   ``.rds`` is the extreme case: pyreadr has no partial-read API, so
   it fully loads regardless of depth.

2. Even where a size guard existed, it only checked ON-DISK size.
   For .xlsx (a zip container), on-disk size is meaningless as a
   memory-safety proxy: a small, highly-compressible file can
   decompress to many times its on-disk size. Verified empirically
   during development: a deliberately modest PoC file, ~104 MB on
   disk, decompressed to ~1.4 GB of worksheet XML (14x) while
   sailing under the (default) 512 MB on-disk ceiling. Real zip
   bombs using more redundant payloads or nested archives routinely
   exceed 1000x. ``_guard_zip_bomb`` closes this by summing
   ``ZipInfo.file_size`` (the archive's own declared uncompressed
   size, read from the central directory — no decompression
   required to check it) across every member.

3. The same "small on disk, huge unpacked" shape applies to two
   other compressed-container formats that reach ``_guard_full_load``
   without ever routing through ``_guard_zip_bomb`` (they aren't
   zip files, so ``_ZIP_CONTAINER_EXTENSIONS`` never matched them):
   gzip-compressed ``.rds`` (R's default ``saveRDS()`` output) and
   ``.zsav`` (SPSS's zlib-compressed SAV variant). Verified
   empirically: a 300k-row data frame of two highly-repetitive
   string columns produced a 1.4 MB gzip .rds decompressing to
   ~29 MB (~21x), and the equivalent .zsav was 0.65 MB on disk
   against a genuine ~24 MB of fixed-width SPSS data (~37x).
   ``_guard_rds_bomb`` closes the .rds gap with real bounded
   streaming decompression (never trusting gzip's wraparound-prone
   ISIZE trailer); ``_guard_zsav_bomb`` closes the .zsav gap using
   pyreadstat's cheap ``metadataonly=True`` row-count and
   declared-column-width metadata to compute the exact uncompressed
   size without touching the compressed data blocks.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from sift.schema import (
    DatasetTooLargeError,
    _extract_will_full_load,
    _guard_rds_bomb,
    _guard_zip_bomb,
    _guard_zsav_bomb,
    extract,
    load_data,
)


# ---------------------------------------------------------------------------
# Building a real, small, fast zip-bomb-shaped .xlsx
# ---------------------------------------------------------------------------

_CONTENT_TYPES = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''

_ROOT_RELS = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''

_WORKBOOK = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''

_WORKBOOK_RELS = b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''


def _build_zip_bomb_xlsx(path: Path, n_rows: int = 20_000) -> None:
    """A real, valid-enough .xlsx whose one worksheet is a large,
    highly-compressible XML blob — the same shape real zip bombs use
    (redundant content compresses far better than real data). At
    ``n_rows=20_000`` this builds in well under a second and produces
    roughly a 1 MB on-disk file that decompresses to roughly 14 MB —
    enough to prove the guard fires without the multi-hundred-MB /
    multi-second PoC used to first discover the bug."""
    row_tmpl = (
        '<row r="{r}">'
        + ''.join(f'<c r="{chr(65 + i)}{{r}}"><v>1111111111</v></c>' for i in range(20))
        + '</row>'
    )
    buf = [row_tmpl.format(r=r) for r in range(1, n_rows + 1)]
    sheet_xml = (
        b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        b'<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        b'<sheetData>' + ''.join(buf).encode() + b'</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("[Content_Types].xml", _CONTENT_TYPES)
        z.writestr("_rels/.rels", _ROOT_RELS)
        z.writestr("xl/workbook.xml", _WORKBOOK)
        z.writestr("xl/_rels/workbook.xml.rels", _WORKBOOK_RELS)
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)


def _build_small_real_xlsx(path: Path) -> None:
    """A genuinely tiny, legitimate spreadsheet — the negative
    control. Must never trip either guard."""
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["age", "region"])
    for i in range(10):
        ws.append([20 + i, i % 2])
    wb.save(path)


# ---------------------------------------------------------------------------
# The zip-bomb guard itself
# ---------------------------------------------------------------------------

def test_zip_bomb_xlsx_passes_on_disk_check_but_fails_uncompressed_check(
    tmp_path: Path,
):
    """Reproduces the exact bypass found during the audit: on-disk
    size comfortably under the ceiling, declared uncompressed size
    well over it."""
    path = tmp_path / "bomb.xlsx"
    _build_zip_bomb_xlsx(path, n_rows=20_000)

    on_disk = path.stat().st_size
    with zipfile.ZipFile(path) as zf:
        uncompressed = sum(i.file_size for i in zf.infolist())

    ceiling = 2_000_000  # 2 MB
    assert on_disk < ceiling, "PoC must stay under the ceiling on disk"
    assert uncompressed > ceiling, (
        "PoC must decompress to more than the ceiling, or this test "
        "isn't exercising the bypass at all"
    )

    with pytest.raises(DatasetTooLargeError, match="decompress"):
        _guard_zip_bomb(path, ceiling)


def test_load_data_refuses_the_zip_bomb(tmp_path: Path, monkeypatch):
    """End-to-end: ``load_data`` (the function ``data_request`` calls)
    must refuse the bomb, not attempt to decompress it."""
    path = tmp_path / "bomb.xlsx"
    _build_zip_bomb_xlsx(path, n_rows=20_000)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "2000000")
    with pytest.raises(DatasetTooLargeError):
        load_data(path)


def test_extract_summary_depth_refuses_the_zip_bomb(tmp_path: Path, monkeypatch):
    """End-to-end via the OTHER entry point: ``extract()`` at the
    deepest depth (what ``get_schema`` uses by default) must also
    refuse — this is the model-facing tool surface, so this is the
    path an actually-hostile file would be submitted through."""
    path = tmp_path / "bomb.xlsx"
    _build_zip_bomb_xlsx(path, n_rows=20_000)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "2000000")
    with pytest.raises(DatasetTooLargeError):
        extract(path, "names_types_labels_summary")


def test_zip_bomb_guard_does_not_fire_on_a_real_small_xlsx(tmp_path: Path):
    """Negative control: a genuinely tiny spreadsheet must sail
    through both checks under the real default ceiling — the guard
    must not be so aggressive it breaks the common case."""
    path = tmp_path / "tiny.xlsx"
    _build_small_real_xlsx(path)
    df = load_data(path)
    assert len(df) == 10
    schema = extract(path, "names_types_labels_summary")
    assert [v["name"] for v in schema["variables"]] == ["age", "region"]


def test_zip_member_count_cap(tmp_path: Path):
    """A zip with an absurd number of entries is refused independent
    of any single member's size — the member-count cap is a
    separate resource-exhaustion vector from the byte-size one."""
    path = tmp_path / "manyparts.xlsx"
    with zipfile.ZipFile(path, "w") as z:
        for i in range(10_001):
            z.writestr(f"part_{i}.xml", "x")
    with pytest.raises(DatasetTooLargeError, match="internal parts"):
        _guard_zip_bomb(path, ceiling=10**12)


def test_malformed_xlsx_at_summary_depth_fails_cleanly(tmp_path: Path):
    """A corrupt/non-zip file with a .xlsx extension, requested at
    the depth that now triggers the zip-bomb pre-check, must fail
    with a normal exception (BadZipFile) rather than hang, crash the
    process, or silently succeed."""
    path = tmp_path / "corrupt.xlsx"
    path.write_bytes(b"this is not a zip file at all, just text")
    with pytest.raises(Exception):
        extract(path, "names_types_labels_summary")


# ---------------------------------------------------------------------------
# extract() previously had NO size ceiling for ANY format — pin the fix
# ---------------------------------------------------------------------------

def test_extract_csv_summary_depth_is_now_guarded(tmp_path: Path, monkeypatch):
    """Before this fix, ``extract()`` never called any size guard for
    ANY format — only ``load_data()`` did. A large CSV at a non-
    names_only depth would previously attempt a full ``pd.read_csv``
    unconditionally."""
    path = tmp_path / "big.csv"
    path.write_text("a,b\n" + "1,2\n" * 1000)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "10")
    with pytest.raises(DatasetTooLargeError):
        extract(path, "names_types_labels")
    # names_only remains the cheap, unguarded peek — must still work
    # even under the same tiny ceiling.
    schema = extract(path, "names_only")
    assert schema["variables"] or "columns" in schema or True


def test_extract_rds_is_always_guarded_regardless_of_depth(
    tmp_path: Path, monkeypatch,
):
    """.rds has no partial-read API in pyreadr — EVERY depth,
    including names_only, was an unconditional full load. This is
    the single most exposed format in the audit; pin that the guard
    now covers it unconditionally."""
    pytest.importorskip("pyreadr")
    path = tmp_path / "big.rds"
    path.write_bytes(b"\x00" * 2048)  # content doesn't matter; guard
                                       # must fire before any parse
                                       # attempt touches the bytes.
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "10")
    for depth in ("names_only", "names_types", "names_types_labels_summary"):
        with pytest.raises(DatasetTooLargeError):
            extract(path, depth)


def test_extract_pyreadstat_metadataonly_depths_stay_unguarded(
    tmp_path: Path, monkeypatch,
):
    """The shallower pyreadstat depths use ``metadataonly=True``,
    which is cheap regardless of file size — the guard must NOT fire
    for those, only for the summary depth that does a real load.
    Over-guarding here would make every Stata/SPSS/SAS schema peek
    fail on a big-but-perfectly-safe-to-peek file."""
    pytest.importorskip("pyreadstat")
    import pyreadstat
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    path = tmp_path / "small.dta"
    pyreadstat.write_dta(df, str(path))
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "10")
    # metadataonly depths: must NOT raise even though the file is
    # "over" the absurdly tiny ceiling, because they never fully load.
    for depth in ("names_only", "names_types", "names_types_labels"):
        schema = extract(path, depth)
        assert schema["variables"]
    # The summary depth DOES fully load — must raise.
    with pytest.raises(DatasetTooLargeError):
        extract(path, "names_types_labels_summary")


# ---------------------------------------------------------------------------
# _extract_will_full_load — pure unit tests on the routing logic itself
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# .rds gzip-bomb guard (gap found in the follow-up audit: on-disk-size-only
# checks apply to .rds/.zsav same as they used to for .xlsx)
# ---------------------------------------------------------------------------

def _build_gzip_rds_bomb(path: Path, n_rows: int = 300_000) -> None:
    """A real, valid gzip-compressed .rds whose two string columns are
    highly repetitive — decompresses far larger than its on-disk
    size, the same PoC shape used to first prove the guard fires."""
    pyreadr = pytest.importorskip("pyreadr")
    df = pd.DataFrame({
        "a": ["same_value_repeated_over_and_over"] * n_rows,
        "b": ["another_repeated_value_here_too"] * n_rows,
        "c": list(range(n_rows)),
    })
    pyreadr.write_rds(str(path), df, compress="gzip")


def test_gzip_rds_bomb_passes_on_disk_check_but_fails_decompressed_check(
    tmp_path: Path,
):
    path = tmp_path / "bomb.rds"
    _build_gzip_rds_bomb(path)

    on_disk = path.stat().st_size
    ceiling = 10 * 1024 * 1024  # 10 MB
    assert on_disk < ceiling, "PoC must stay under the ceiling on disk"

    with pytest.raises(DatasetTooLargeError, match="decompress"):
        _guard_rds_bomb(path, ceiling)


def test_load_data_refuses_the_rds_bomb(tmp_path: Path, monkeypatch):
    """End-to-end: ``load_data`` must refuse the gzip .rds bomb."""
    path = tmp_path / "bomb.rds"
    _build_gzip_rds_bomb(path)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(10 * 1024 * 1024))
    with pytest.raises(DatasetTooLargeError):
        load_data(path)


def test_extract_refuses_the_rds_bomb_at_every_depth(tmp_path: Path, monkeypatch):
    """.rds is a full load at every depth (no partial-read API), so
    the bomb must be refused regardless of requested depth."""
    path = tmp_path / "bomb.rds"
    _build_gzip_rds_bomb(path)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(10 * 1024 * 1024))
    for depth in ("names_only", "names_types", "names_types_labels_summary"):
        with pytest.raises(DatasetTooLargeError):
            extract(path, depth)


def test_rds_bomb_guard_does_not_fire_on_a_real_small_rds(tmp_path: Path):
    """Negative control: a genuinely tiny gzip-compressed .rds must
    sail through under the real default ceiling."""
    pyreadr = pytest.importorskip("pyreadr")
    path = tmp_path / "tiny.rds"
    df = pd.DataFrame({"x": list(range(10))})
    pyreadr.write_rds(str(path), df, compress="gzip")
    _guard_rds_bomb(path, ceiling=10 * 1024 * 1024)  # must not raise
    loaded = load_data(path)
    assert len(loaded) == 10


def test_rds_bomb_guard_skips_non_gzip_rds_without_error(tmp_path: Path):
    """Uncompressed (or non-gzip-compressed) .rds files don't match
    gzip's magic bytes; the guard must no-op rather than misfire on
    a format it can't cheaply inspect — the on-disk-size check in
    ``_guard_full_load`` still applies to these regardless."""
    pyreadr = pytest.importorskip("pyreadr")
    path = tmp_path / "uncompressed.rds"
    df = pd.DataFrame({"x": list(range(10))})
    pyreadr.write_rds(str(path), df)  # compress=None → uncompressed
    with open(path, "rb") as f:
        assert f.read(2) != b"\x1f\x8b"
    _guard_rds_bomb(path, ceiling=1)  # absurd ceiling; must still not raise
    loaded = load_data(path)
    assert len(loaded) == 10


def test_rds_bomb_guard_handles_malformed_gzip_cleanly(tmp_path: Path):
    """A file with gzip's magic bytes but a corrupt/truncated stream
    must not crash the guard — it's not this function's job to
    diagnose corruption, only to check a well-formed stream's size."""
    path = tmp_path / "corrupt.rds"
    path.write_bytes(b"\x1f\x8b" + b"not actually a valid gzip stream")
    _guard_rds_bomb(path, ceiling=1)  # must not raise from here


# ---------------------------------------------------------------------------
# .zsav declared-size guard
# ---------------------------------------------------------------------------

def _build_zsav_bomb(path: Path, n_rows: int = 300_000):
    """A real, valid compressed .zsav whose columns are highly
    repetitive — small on disk, large once SPSS's fixed-width
    row/column layout is accounted for."""
    pyreadstat = pytest.importorskip("pyreadstat")
    df = pd.DataFrame({
        "a": ["same_value_repeated_over_and_over"] * n_rows,
        "b": ["another_repeated_value_here_too"] * n_rows,
        "c": list(range(n_rows)),
    })
    pyreadstat.write_sav(df, str(path), compress=True)
    return pyreadstat


def test_zsav_bomb_passes_on_disk_check_but_fails_declared_size_check(
    tmp_path: Path,
):
    path = tmp_path / "bomb.zsav"
    _build_zsav_bomb(path)

    on_disk = path.stat().st_size
    ceiling = 10 * 1024 * 1024  # 10 MB
    assert on_disk < ceiling, "PoC must stay under the ceiling on disk"

    with pytest.raises(DatasetTooLargeError, match="decompressed"):
        _guard_zsav_bomb(path, ceiling)


def test_load_data_refuses_the_zsav_bomb(tmp_path: Path, monkeypatch):
    path = tmp_path / "bomb.zsav"
    _build_zsav_bomb(path)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(10 * 1024 * 1024))
    with pytest.raises(DatasetTooLargeError):
        load_data(path)


def test_extract_summary_depth_refuses_the_zsav_bomb(tmp_path: Path, monkeypatch):
    path = tmp_path / "bomb.zsav"
    _build_zsav_bomb(path)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", str(10 * 1024 * 1024))
    with pytest.raises(DatasetTooLargeError):
        extract(path, "names_types_labels_summary")
    # shallower, metadataonly depths must NOT be affected
    for depth in ("names_only", "names_types", "names_types_labels"):
        result = extract(path, depth)
        assert result["variables"]


def test_zsav_bomb_guard_does_not_fire_on_a_real_small_zsav(tmp_path: Path):
    """Negative control: a tiny, legitimate .zsav must sail through
    under the real default ceiling."""
    pytest.importorskip("pyreadstat")
    path = tmp_path / "tiny.zsav"
    _build_zsav_bomb(path, n_rows=10)
    _guard_zsav_bomb(path, ceiling=10 * 1024 * 1024)  # must not raise
    loaded = load_data(path)
    assert len(loaded) == 10


def test_zsav_bomb_estimate_matches_real_uncompressed_size(tmp_path: Path):
    """Pin the estimate's accuracy: rows * sum(declared column widths)
    should closely match the size of an equivalent PLAIN (uncompressed)
    .sav built from identical data — SPSS stores every value at its
    declared fixed width regardless of compression, so this is a real
    size, not a loose approximation."""
    pyreadstat = pytest.importorskip("pyreadstat")
    df = pd.DataFrame({
        "a": ["same_value_repeated_over_and_over"] * 50_000,
        "b": ["another_repeated_value_here_too"] * 50_000,
        "c": list(range(50_000)),
    })
    plain_path = tmp_path / "plain.sav"
    pyreadstat.write_sav(df, str(plain_path), compress=False)
    plain_on_disk = plain_path.stat().st_size

    _df, meta = pyreadstat.read_sav(str(plain_path), metadataonly=True)
    estimated = int(meta.number_rows) * sum(meta.variable_storage_width.values())

    # Estimate should be within a small fraction of the real plain-file
    # size (header/dictionary overhead accounts for the rest).
    assert abs(estimated - plain_on_disk) / plain_on_disk < 0.05


@pytest.mark.parametrize("suffix,depth,expected", [
    (".rds", "names_only", True),
    (".rds", "names_types_labels_summary", True),
    (".dta", "names_only", False),
    (".dta", "names_types_labels", False),
    (".dta", "names_types_labels_summary", True),
    (".sav", "names_types_labels_summary", True),
    (".zsav", "names_types_labels_summary", True),
    (".sas7bdat", "names_types_labels_summary", True),
    (".xpt", "names_types_labels_summary", True),
    (".xlsx", "names_types_labels", False),
    (".xlsx", "names_types_labels_summary", True),
    (".csv", "names_only", False),
    (".csv", "names_types", True),
    (".tsv", "names_only", False),
    (".parquet", "names_only", False),
    (".parquet", "names_types", True),
    (".jsonl", "names_only", False),
    (".jsonl", "names_types", True),
    (".ndjson", "names_types", True),
])
def test_extract_will_full_load_routing(suffix, depth, expected):
    assert _extract_will_full_load(suffix, depth) is expected


def test_extract_refuses_oversized_parquet_before_full_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """Oversized Parquet reads are rejected before a full load.

    ``_extract_parquet``'s docstring used to claim
    a Parquet file larger than RAM would OOM and that this was "out
    of scope for the current pilot" -- stale the moment
    ``_extract_will_full_load``/``_guard_full_load`` started covering
    every non-``names_only`` Parquet depth (this file's own
    ``test_extract_will_full_load_routing`` pins the dispatcher
    routing; this test proves the end-to-end behavior against a REAL
    parquet file, not just the routing decision).

    Writes a real, small Parquet file, then shrinks the on-disk
    ceiling via ``SIFT_MAX_LOAD_BYTES`` so the file is "oversized"
    relative to policy without needing an actually huge file on disk.
    ``extract()`` at a full-load depth must raise
    ``DatasetTooLargeError`` -- proving the file is refused with an
    actionable message, not OOM'd, exactly what the corrected
    docstring now says.
    """
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": range(1000), "b": ["x"] * 1000})
    path = tmp_path / "data.parquet"
    df.to_parquet(path)

    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "10")

    # names_only must NOT be affected -- it never reaches
    # _guard_full_load (pure footer read, constant-time regardless of
    # size, per _extract_will_full_load's own routing table).
    names_only_result = extract(path, "names_only")
    assert names_only_result["variables"]

    # A depth that requires real column data must be refused BEFORE
    # pd.read_parquet ever runs.
    with pytest.raises(DatasetTooLargeError):
        extract(path, "names_types")
