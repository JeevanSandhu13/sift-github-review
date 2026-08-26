from __future__ import annotations

from pathlib import Path

import pytest

from sift.data_quality import (
    DataQualityError,
    apply_approved_corrections,
    assess_frame,
    assess_relationships,
    safe_preflight,
)


@pytest.fixture()
def pd():
    return pytest.importorskip("pandas")


def _codes(report):
    return {row["code"] for row in report["findings"]}


def test_missingness_structure_mixed_encodings_and_categories(pd) -> None:
    frame = pd.DataFrame({
        "arm": ["A"] * 50 + ["B"] * 50,
        "measure": [None] * 50 + list(range(50)),
        "mixed": [str(i) for i in range(50)] + ["bad"] * 50,
        "category": ["known"] * 99 + ["unexpected"],
    })
    report = assess_frame(frame, context={"expected_categories": {"category": ["known"]}})
    codes = _codes(report)
    assert {"variable_missingness", "structural_missingness", "mixed_numeric_text",
            "unexpected_categories"} <= codes
    unexpected = next(row for row in report["findings"] if row["code"] == "unexpected_categories")
    assert "unexpected" not in str(list(unexpected["evidence"].values()))


def test_duplicates_keys_sentinels_outliers_and_no_individual_outlier_values(pd) -> None:
    frame = pd.DataFrame({
        "record_id": [1, 1] + list(range(2, 100)),
        "value": [-99.0] * 3 + list(map(float, range(96))) + [100000.0],
    })
    frame = pd.concat([frame, frame.iloc[[3]]], ignore_index=True)
    report = assess_frame(frame, context={"keys": ["record_id"]})
    codes = _codes(report)
    assert {"duplicate_rows", "key_uniqueness", "suspicious_sentinel",
            "aggregate_outliers"} <= codes
    outlier = next(row for row in report["findings"] if row["code"] == "aggregate_outliers")
    assert set(outlier["evidence"]) == {"count", "share", "rule"}
    assert report["summary"]["model_selection_blocked"] is True


def test_dates_timezones_units_heaping_truncation_and_encoding(pd) -> None:
    timestamps = ["2024-01-01T12:00:00Z"] * 30 + ["2024-01-01T12:00:00+05:30"] * 30
    frame = pd.DataFrame({
        "event_time": timestamps,
        "age": [5 * i for i in range(60)],
        "note": ["normal-text"] * 50 + ["caf\ufffd"] * 10,
        "code": [f"{i:04d}-longtext"[:8] for i in range(60)],
        "mass_kg": list(range(60)),
        "mass_lb": list(range(60)),
    })
    report = assess_frame(frame, context={"units": {"mass_kg": "kg", "mass_lb": "lb"}})
    codes = _codes(report)
    assert {"inconsistent_timezones", "unit_mismatch", "heaping_rounding",
            "encoding_corruption", "possible_truncation"} <= codes


def test_panel_gaps_imbalance_and_duplicate_panel_time(pd) -> None:
    frame = pd.DataFrame({
        "person": [1, 1, 1, 2, 2, 3],
        "wave": ["2020-01-01", "2020-01-01", "2022-01-01",
                 "2020-01-01", "2021-01-01", "2020-01-01"],
    })
    report = assess_frame(frame, context={"panel_id": "person", "time": "wave"})
    assert {"duplicate_panel_time", "panel_imbalance", "longitudinal_gaps"} <= _codes(report)


def test_treatment_overlap_target_leakage_class_split_contamination(pd) -> None:
    frame = pd.DataFrame({
        "treated": [0] * 95 + [1] * 5,
        "x": list(range(95)) + list(range(1000, 1005)),
        "target": [0] * 97 + [1] * 3,
        "target_copy": [0] * 97 + [1] * 3,
        "split": ["train"] * 50 + ["test"] * 50,
    })
    # Force one feature pattern across splits while keeping split itself apart.
    frame.loc[50, ["treated", "x", "target", "target_copy"]] = frame.loc[
        0, ["treated", "x", "target", "target_copy"]
    ].values
    report = assess_frame(frame, context={
        "treatment": "treated", "target": "target",
        "features": ["x", "target_copy"], "split": "split",
    })
    assert {"treatment_imbalance", "treatment_overlap", "class_imbalance",
            "target_leakage", "train_test_contamination"} <= _codes(report)


def test_survey_weights_and_geographic_bounds(pd) -> None:
    frame = pd.DataFrame({
        "survey_weight": [1.0] * 98 + [0.0, 1000.0],
        "latitude": [49.0] * 99 + [91.0],
        "longitude": [-123.0] * 99 + [-181.0],
    })
    report = assess_frame(frame, context={
        "weights": ["survey_weight"], "latitude": "latitude", "longitude": "longitude",
    })
    assert {"survey_weight_anomaly", "impossible_coordinates"} <= _codes(report)


def test_relationship_orphans_many_to_many_and_parent_uniqueness(pd) -> None:
    frames = {
        "child.csv": pd.DataFrame({"parent_id": [1, 1, 3]}),
        "parent.csv": pd.DataFrame({"id": [1, 1, 2]}),
    }
    report = assess_relationships(frames, context={"foreign_keys": [{
        "child_dataset": "child.csv", "child_column": "parent_id",
        "parent_dataset": "parent.csv", "parent_column": "id",
    }]})
    assert {"orphan_foreign_keys", "many_to_many_merge", "parent_key_uniqueness"} <= _codes(report)
    assert report["summary"]["model_selection_blocked"] is True


def test_safe_preflight_removes_counts_and_correction_values(pd) -> None:
    report = assess_frame(pd.DataFrame({"x": [-99.0] * 10 + list(range(90))}))
    safe = safe_preflight(report)
    assert "rows_checked" not in safe
    assert all("evidence" not in row and "correction" not in row for row in safe["findings"])
    assert "-99" not in str(safe)


def test_unsupported_quality_check_is_explicitly_incomplete(pd) -> None:
    frame = pd.DataFrame({"nested": [[1], [1], [2]]})
    report = assess_frame(frame)
    assert report["summary"]["checks_complete"] is False
    assert any(
        row["check"] == "cardinality" and row["columns"] == ["nested"]
        for row in report["limitations"]
    )
    safe = safe_preflight(report)
    assert safe["summary"]["checks_complete"] is False
    assert safe["limitations"] == report["limitations"]


def test_relationship_check_failure_is_explicitly_incomplete(pd) -> None:
    frames = {
        "child": pd.DataFrame({"id": [[1], [2]]}),
        "parent": pd.DataFrame({"id": [[1], [2]]}),
    }
    report = assess_relationships(frames, context={"foreign_keys": [{
        "child_dataset": "child", "child_column": "id",
        "parent_dataset": "parent", "parent_column": "id",
    }]})
    assert report["summary"]["checks_complete"] is False
    assert report["limitations"][0]["check"] == "relationship_integrity"


def test_approved_correction_creates_derived_copy_with_lineage(tmp_path: Path, pd) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"id": [1, 1, 2], "x": [-99.0, -99.0, 3.0]}).to_csv(source, index=False)
    source_before = source.read_bytes()
    report = assess_frame(pd.read_csv(source), context={"keys": ["id"]})
    approved = [row["id"] for row in report["findings"] if row.get("correction")]
    result = apply_approved_corrections(
        tmp_path, source, approved_finding_ids=approved,
        output_name="corrected.parquet", context={"keys": ["id"]},
    )
    assert result["ok"] is True
    assert result["source_mutated"] is False
    assert source.read_bytes() == source_before
    corrected = pd.read_parquet(tmp_path / "corrected.parquet")
    assert len(corrected) == 2
    assert corrected["x"].isna().sum() == 1
    from sift.canonical_dataset import current_manifest
    manifest = current_manifest(tmp_path, tmp_path / "corrected.parquet")
    assert manifest["source"]["kind"] == "derived"
    assert manifest["lineage"]["parents"] == [result["parent_fingerprint"]]
    assert manifest["lineage"]["transformations"][0]["accepted_finding_ids"].split(",") == approved


def test_corrections_require_real_approval_and_refuse_overwrite(tmp_path: Path, pd) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"x": [1, 1]}).to_csv(source, index=False)
    with pytest.raises(DataQualityError, match="explicitly approved"):
        apply_approved_corrections(tmp_path, source, approved_finding_ids=[], output_name="out.parquet")
    (tmp_path / "out.parquet").write_bytes(b"existing")
    with pytest.raises(DataQualityError, match="already exists"):
        apply_approved_corrections(tmp_path, source, approved_finding_ids=["fake"], output_name="out.parquet")


def test_analysis_preflight_blocks_declared_invalid_key_without_values(tmp_path: Path, pd) -> None:
    source = tmp_path / "data.csv"
    pd.DataFrame({"subject_id": [1, 1, 2], "secret": ["alpha", "beta", "gamma"]}).to_csv(
        source, index=False,
    )
    from sift.tools import _data_quality_preflight
    report, error = _data_quality_preflight(
        tmp_path, ("data.csv",), {"keys": ["subject_id"]},
    )
    assert error is None
    assert report["model_selection_blocked"] is True
    rendered = str(report)
    assert "alpha" not in rendered and "beta" not in rendered and "gamma" not in rendered
    finding = next(
        row for row in report["datasets"][0]["findings"]
        if row["code"] == "key_uniqueness"
    )
    assert "evidence" not in finding and "correction" not in finding
