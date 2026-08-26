from __future__ import annotations

import json
import os
import sys
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest

from sift.canonical_dataset import (
    CanonicalDatasetError,
    compare_schemas,
    create_collection_manifest,
    create_manifest,
    current_manifest,
    discard_uncommitted_manifest,
    discover_partition_files,
    ensure_manifest,
    load_dataset_collection,
    load_canonical_data,
    load_partitioned_data,
    manifest_path,
    snapshot_source_artifact,
)


def _csv(root: Path, name: str = "data.csv") -> Path:
    path = root / name
    pd.DataFrame({
        "Subject ID": ["a", "a", "b"],
        "sampling_weight": [1.0, 2.0, 1.5],
        "when": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "latitude": [49.1, 49.2, 49.3],
        "value": [1, 2, 3],
    }).to_csv(path, index=False)
    return path


def test_manifest_is_complete_deterministic_and_content_addressed(tmp_path: Path) -> None:
    path = _csv(tmp_path)
    first = create_manifest(tmp_path, path)
    second = ensure_manifest(tmp_path, path)

    assert first["fingerprint"] == second["fingerprint"]
    assert len(first["source"]["source_sha256"]) == 64
    assert len(first["content_sha256"]) == 64
    assert first["shape"] == {"rows": 3, "columns": 5}
    assert first["parser"]["package"] == "pandas"
    assert first["selection"] == {}
    assert manifest_path(tmp_path, first["fingerprint"]).is_file()
    snapshot = tmp_path / first["source"]["snapshot_paths"][0]
    assert snapshot.read_bytes() == path.read_bytes()
    if os.name != "nt":
        assert snapshot.stat().st_mode & 0o222 == 0

    columns = {row["normalized_name"]: row for row in first["columns"]}
    assert columns["Subject_ID"]["roles"] == [
        "identifier_like", "repeated_measures_identifier",
    ]
    assert columns["sampling_weight"]["roles"] == ["weight"]
    assert columns["when"]["logical_type"] == "datetime"
    assert columns["when"]["type_confidence"] == 0.85
    assert columns["when"]["roles"] == ["time_index"]
    assert columns["latitude"]["roles"] == ["geospatial"]


def test_uncommitted_manifest_rollback_removes_private_snapshot(
    tmp_path: Path,
) -> None:
    path = _csv(tmp_path)
    manifest = create_manifest(tmp_path, path)
    snapshot = tmp_path / manifest["source"]["snapshot_paths"][0]
    assert snapshot.is_file()

    assert discard_uncommitted_manifest(
        tmp_path, path, manifest["fingerprint"],
    )
    assert current_manifest(tmp_path, path) is None
    assert not manifest_path(tmp_path, manifest["fingerprint"]).exists()
    assert not snapshot.exists()


def test_uncommitted_manifest_rollback_refuses_wrong_identity(
    tmp_path: Path,
) -> None:
    path = _csv(tmp_path)
    manifest = create_manifest(tmp_path, path)
    assert not discard_uncommitted_manifest(tmp_path, path, "0" * 64)
    assert current_manifest(tmp_path, path) == manifest


def test_uncommitted_manifest_rollback_keeps_tracking_if_snapshot_is_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _csv(tmp_path)
    manifest = create_manifest(tmp_path, path)
    snapshot = tmp_path / manifest["source"]["snapshot_paths"][0]
    original_unlink = Path.unlink

    def locked_unlink(candidate: Path, missing_ok: bool = False) -> None:
        if candidate == snapshot:
            raise PermissionError("simulated OS file lock")
        original_unlink(candidate, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    assert not discard_uncommitted_manifest(
        tmp_path, path, manifest["fingerprint"],
    )
    assert manifest_path(tmp_path, manifest["fingerprint"]).is_file()
    assert current_manifest(tmp_path, path) == manifest
    assert snapshot.is_file()


def test_content_semantics_agree_across_csv_and_parquet(tmp_path: Path) -> None:
    frame = pd.DataFrame({"id": [1, 2], "group": ["a", "b"]})
    csv = tmp_path / "rows.csv"
    parquet = tmp_path / "rows.parquet"
    frame.to_csv(csv, index=False)
    frame.to_parquet(parquet, index=False)
    left = create_manifest(tmp_path, csv)
    right = create_manifest(tmp_path, parquet)

    assert left["content_sha256"] == right["content_sha256"]
    assert [c["logical_type"] for c in left["columns"]] == [
        c["logical_type"] for c in right["columns"]
    ]
    # Exact source identities remain distinct even when canonical content agrees.
    assert left["fingerprint"] != right["fingerprint"]


def test_duplicate_source_headers_keep_original_and_materialized_names(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.csv"
    path.write_text("id,id\n1,2\n", encoding="utf-8")
    manifest = create_manifest(tmp_path, path)
    assert [row["original_name"] for row in manifest["columns"]] == ["id", "id"]
    assert [row["materialized_name"] for row in manifest["columns"]] == ["id", "id.1"]
    assert [row["normalized_name"] for row in manifest["columns"]] == ["id", "id.1"]


def test_column_metadata_preserves_categories_timezone_and_decimal(tmp_path: Path) -> None:
    path = tmp_path / "typed.parquet"
    frame = pd.DataFrame({
        "ordered": pd.Categorical(["low", "high"], categories=["low", "high"], ordered=True),
        "at": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
        "amount": [Decimal("1.20"), Decimal("3.456")],
    })
    frame.to_parquet(path, index=False)
    manifest = create_manifest(tmp_path, path)
    columns = {row["original_name"]: row for row in manifest["columns"]}
    assert columns["ordered"]["categorical"] == {
        "ordered": True, "levels": ["low", "high"], "levels_truncated": False,
    }
    assert columns["at"]["timezone"] == "UTC"
    assert columns["amount"]["logical_type"] == "decimal"
    assert columns["amount"]["decimal"] == {"precision": 4, "scale": 3}


def test_readstat_labels_value_labels_and_missing_codes(tmp_path: Path) -> None:
    pyreadstat = pytest.importorskip("pyreadstat")
    path = tmp_path / "survey.sav"
    pyreadstat.write_sav(
        pd.DataFrame({"group": [1.0, 2.0, 9.0]}),
        str(path),
        column_labels={"group": "Study arm"},
        variable_value_labels={"group": {1.0: "Control", 2.0: "Treatment"}},
        missing_ranges={"group": [9.0]},
    )
    manifest = create_manifest(tmp_path, path)
    column = manifest["columns"][0]
    assert column["variable_label"] == "Study arm"
    assert column["value_labels"] == {"1.0": "Control", "2.0": "Treatment"}
    assert column["declared_missing_values"]


def test_selection_scope_changes_identity_and_exact_worksheet(tmp_path: Path) -> None:
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1]}).to_excel(writer, sheet_name="A", index=False)
        pd.DataFrame({"b": [2, 3]}).to_excel(writer, sheet_name="B", index=False)
    first = create_manifest(tmp_path, path, selection={"worksheet": "A"})
    second = create_manifest(tmp_path, path, selection={"worksheet": "B"})
    assert first["selection"] == {"worksheet": "A"}
    assert second["selection"] == {"worksheet": "B"}
    assert first["shape"]["rows"] == 1
    assert second["shape"]["rows"] == 2
    assert first["fingerprint"] != second["fingerprint"]


def test_r_workspace_object_selection_is_explicit_and_recorded(tmp_path: Path) -> None:
    pyreadr = pytest.importorskip("pyreadr")
    from sift.executor import cached_environment
    from sift.format_selection import (
        FormatSelectionError,
        _require_parser_backend,
        list_container_objects,
        materialize_selected_format,
    )
    from sift.schema import SchemaExtractError, load_data

    try:
        _require_parser_backend(cached_environment(), sys.platform)
    except FormatSelectionError:
        pytest.skip(
            "the host sandbox backend cannot be applied inside this test environment",
        )

    source = tmp_path / "workspace.RData"
    pyreadr.write_rdata(
        str(source), pd.DataFrame({"id": [1, 2]}), df_name="cohort",
    )
    assert load_data(source, r_object="cohort").shape == (2, 1)
    with pytest.raises(SchemaExtractError, match="was not found"):
        load_data(source, r_object="other")
    assert list_container_objects(source) == [
        {"id": "cohort", "shape": [2, 1]},
    ]
    session = tmp_path / "session"; session.mkdir()
    output = materialize_selected_format(
        session, source=source, selection={"r_object": "cohort"},
    )
    manifest = current_manifest(
        session, output, selection={"r_object": "cohort"},
    )
    assert manifest is not None
    assert manifest["selection"]["r_object"] == "cohort"
    assert manifest["source"]["kind"] == "derived"
    assert manifest["lineage"]["parents"]
    assert manifest["parser"]["source_parser"]["adapter"] == "r_workspace"
    assert manifest["parser"]["source_parser"]["packages"][0]["name"] == "pyreadr"


def test_cache_hit_avoids_reparsing_source_and_tampering_rebuilds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _csv(tmp_path)
    expected = load_canonical_data(tmp_path, path)
    manifest = ensure_manifest(tmp_path, path)
    cache = tmp_path / ".sift" / "datasets" / "cache" / f"{manifest['fingerprint']}.parquet"
    assert cache.is_file()

    import sift.schema as schema
    original = schema.load_data

    def no_source_reparse(candidate: Path, **kwargs):
        if Path(candidate).resolve() == path.resolve():
            raise AssertionError("source was reparsed on a canonical cache hit")
        return original(candidate, **kwargs)

    monkeypatch.setattr(schema, "load_data", no_source_reparse)
    pd.testing.assert_frame_equal(load_canonical_data(tmp_path, path), expected)

    cache.write_bytes(b"tampered")
    with pytest.raises(AssertionError):
        # Cache integrity fails and rebuilding correctly requires the source.
        load_canonical_data(tmp_path, path)


def test_cold_cache_computes_canonical_scalar_hash_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _csv(tmp_path)
    import sift.canonical_dataset as canonical

    original = canonical._content_hash
    calls = 0

    def counted(frame, normalized_names):
        nonlocal calls
        calls += 1
        return original(frame, normalized_names)

    monkeypatch.setattr(canonical, "_content_hash", counted)
    loaded = load_canonical_data(tmp_path, path)

    assert loaded.shape == (3, 5)
    assert calls == 1


def test_file_change_invalidates_cache_and_advances_path_identity(tmp_path: Path) -> None:
    path = _csv(tmp_path)
    load_canonical_data(tmp_path, path)
    old = ensure_manifest(tmp_path, path)
    old_cache = tmp_path / ".sift" / "datasets" / "cache" / f"{old['fingerprint']}.parquet"
    assert old_cache.exists()

    pd.DataFrame({"id": [1, 2, 3, 4]}).to_csv(path, index=False)
    new = create_manifest(tmp_path, path)
    assert new["fingerprint"] != old["fingerprint"]
    assert new["content_sha256"] != old["content_sha256"]
    assert not old_cache.exists()


def test_large_source_gets_exact_metadata_manifest_without_full_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = _csv(tmp_path)
    monkeypatch.setenv("SIFT_MAX_LOAD_BYTES", "1")
    manifest = ensure_manifest(tmp_path, path)
    assert manifest["content_hash_basis"] == "source_bytes"
    assert manifest["shape"] == {
        "rows": 3, "rows_exact": True, "columns": 5, "columns_exact": True,
    }
    assert manifest["profiling_scope"] == "bounded_sample"
    assert manifest["source"]["snapshot_paths"]


def test_cache_and_manifest_never_cross_session_boundaries(tmp_path: Path) -> None:
    one = tmp_path / "one"; one.mkdir()
    two = tmp_path / "two"; two.mkdir()
    left = _csv(one)
    right = _csv(two)
    load_canonical_data(one, left)
    load_canonical_data(two, right)
    left_manifest = ensure_manifest(one, left)
    right_manifest = ensure_manifest(two, right)
    assert left_manifest["content_sha256"] == right_manifest["content_sha256"]
    assert (one / ".sift" / "datasets").is_dir()
    assert (two / ".sift" / "datasets").is_dir()
    assert not any(path.is_relative_to(two) for path in (one / ".sift" / "datasets").rglob("*"))


def test_derived_lineage_tracks_parents_and_transformations(tmp_path: Path) -> None:
    source = _csv(tmp_path)
    parent = create_manifest(tmp_path, source)
    derived = tmp_path / "derived.parquet"
    pd.DataFrame({"mean": [2.0]}).to_parquet(derived, index=False)
    child = create_manifest(
        tmp_path,
        derived,
        dataset_kind="derived",
        parents=[parent["fingerprint"]],
        transformations=[{
            "operation": "aggregate",
            "code_sha256": "a" * 64,
            "ignored_observation": "must not persist",
        }],
    )
    assert child["source"]["kind"] == "derived"
    assert child["lineage"]["parents"] == [parent["fingerprint"]]
    assert child["lineage"]["transformations"] == [{
        "operation": "aggregate", "code_sha256": "a" * 64,
    }]


def test_collections_partitions_and_schema_evolution(tmp_path: Path) -> None:
    first_dir = tmp_path / "year=2024"; first_dir.mkdir()
    second_dir = tmp_path / "year=2025"; second_dir.mkdir()
    first = first_dir / "part.csv"
    second = second_dir / "part.csv"
    pd.DataFrame({"id": [1], "value": [2]}).to_csv(first, index=False)
    pd.DataFrame({"id": [2], "value": [3], "new": [4]}).to_csv(second, index=False)
    found = discover_partition_files(tmp_path)
    assert found == [first, second]
    collection = create_collection_manifest(tmp_path, found, partitioned=True)
    assert collection["kind"] == "partitioned_dataset"
    assert collection["shape"] == {"rows": 2, "tables": 2}
    assert collection["partition_keys"] == ["year"]
    assert len(collection["content_sha256"]) == 64
    assert collection["schema_evolution"][0]["added"] == ["new"]
    assert collection["schema_evolution"][0]["backward_compatible"] is True
    combined, loaded_manifest = load_partitioned_data(tmp_path, tmp_path)
    assert loaded_manifest["fingerprint"] == collection["fingerprint"]
    assert combined["year"].tolist() == ["2024", "2025"]
    assert combined["new"].isna().tolist() == [True, False]

    tables, multi = load_dataset_collection(tmp_path, found)
    assert set(tables) == {"year=2024/part.csv", "year=2025/part.csv"}
    assert multi["kind"] == "dataset_collection"


def test_schema_evolution_flags_breaking_changes(tmp_path: Path) -> None:
    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "new.csv"
    pd.DataFrame({"id": [1], "amount": [2]}).to_csv(old_path, index=False)
    pd.DataFrame({"id": ["x"]}).to_csv(new_path, index=False)
    change = compare_schemas(
        create_manifest(tmp_path, old_path), create_manifest(tmp_path, new_path),
    )
    assert change["removed"] == ["amount"]
    assert change["type_changed"] == ["id"]
    assert change["backward_compatible"] is False


def test_source_specific_sidecar_keeps_structure_not_arbitrary_values(tmp_path: Path) -> None:
    path = tmp_path / "selected.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(path, index=False)
    sidecar = path.with_suffix(".parquet.metadata.json")
    sidecar.write_text(json.dumps({
        "format": "hdf5", "dataset": "group/table", "shape": [1, 1],
        "attributes": {"unit": "kg", "secret_cell": "individual value"},
        "header": {"PATIENT": "Alice"},
    }))
    manifest = create_manifest(tmp_path, path, selection={"dataset": "group/table"})
    assert manifest["structure"]["multi_table_source"] is True
    metadata = manifest["source_specific_metadata"]
    assert metadata["attributes_keys"] == ["secret_cell", "unit"]
    assert metadata["header_keys"] == ["PATIENT"]
    assert "individual value" not in json.dumps(metadata)
    assert "Alice" not in json.dumps(metadata)


def test_external_source_artifact_snapshot_hides_absolute_path(tmp_path: Path) -> None:
    session = tmp_path / "session"; session.mkdir()
    external = tmp_path / "external.bin"; external.write_bytes(b"source")
    artifact = snapshot_source_artifact(session, external)
    assert artifact["session_relative_path"] is None
    assert artifact["origin"] == "external_selected_file"
    assert str(tmp_path) not in json.dumps(artifact)
    assert (session / artifact["snapshot_paths"][0]).read_bytes() == b"source"


def test_snapshot_freezes_target_only_after_temporary_name_is_removed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Windows cannot unlink a file after chmod(0o400) marks it read-only."""
    session = tmp_path / "session"
    session.mkdir()
    source = tmp_path / "source.xml"
    source.write_bytes(b"<records/>")
    events: list[tuple[str, str]] = []
    original_link = os.link
    original_chmod = os.chmod
    original_unlink = Path.unlink

    def tracked_link(source_name: str, target_name: str) -> None:
        events.append(("link", str(target_name)))
        original_link(source_name, target_name)

    def tracked_chmod(path: str | bytes | os.PathLike[str] | os.PathLike[bytes], mode: int) -> None:
        events.append(("chmod", str(path)))
        original_chmod(path, mode)

    def tracked_unlink(path: Path, missing_ok: bool = False) -> None:
        if path.name.startswith(".snapshot-"):
            events.append(("unlink-temp", str(path)))
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(os, "link", tracked_link)
    monkeypatch.setattr(os, "chmod", tracked_chmod)
    monkeypatch.setattr(Path, "unlink", tracked_unlink)

    artifact = snapshot_source_artifact(session, source)
    target = session / artifact["snapshot_paths"][0]
    link_index = events.index(("link", str(target)))
    unlink_index = next(index for index, event in enumerate(events) if event[0] == "unlink-temp")
    chmod_index = events.index(("chmod", str(target)))
    assert link_index < unlink_index < chmod_index


def test_symlink_and_invalid_lineage_fail_closed(tmp_path: Path) -> None:
    path = _csv(tmp_path)
    link = tmp_path / "link.csv"
    try:
        link.symlink_to(path)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(CanonicalDatasetError, match="symbolic"):
        create_manifest(tmp_path, link)
    with pytest.raises(CanonicalDatasetError, match="fingerprints"):
        create_manifest(tmp_path, path, parents=["not-a-hash"])


def test_all_trusted_analysis_surfaces_bind_the_same_semantics(tmp_path: Path) -> None:
    from sift.data_request import handle
    from sift.dataset_profile import profile_dataset
    from sift.linkage import analyze_pair
    from sift.tools import _canonicalize_analysis_sources

    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    pd.DataFrame({"patient_id": range(20), "group": ["a"] * 20}).to_csv(left, index=False)
    pd.DataFrame({"patient_id": range(20), "outcome": range(20)}).to_csv(right, index=False)

    assert handle(
        left, "na_count", "group", session_root=tmp_path,
    ).status == "granted"
    request_identity = current_manifest(tmp_path, left)
    assert request_identity is not None

    profile = profile_dataset(left, session_root=tmp_path)
    assert profile["ok"] is True
    assert profile["canonical_fingerprint"] == request_identity["fingerprint"]

    assert analyze_pair(left, right, session_root=tmp_path)[0]["key"] == "patient_id"
    assert current_manifest(tmp_path, left)["fingerprint"] == request_identity["fingerprint"]

    script_sources, error = _canonicalize_analysis_sources(
        tmp_path, ("left.csv", "right.csv"),
    )
    assert error is None
    assert script_sources[0]["fingerprint"] == request_identity["fingerprint"]
    assert all(row["status"] == "canonical" for row in script_sources)
    assert script_sources[0]["source_sha256"] == request_identity["source"]["source_sha256"]
    assert script_sources[0]["parser"] == request_identity["parser"]
