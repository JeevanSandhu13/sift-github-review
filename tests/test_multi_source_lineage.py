"""First-class lineage for scripts that read or join multiple datasets."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from sift.config import set_cwd
from sift.privacy_budget import consumed_for_dataset
from sift.query_fingerprint import _submit_script_analysis_events
from sift.release_ledger import _facts_from_response
from sift.research_export import build_replication_package
from sift.sanitizer import DEFAULT_CONFIG
from sift.store import ResultStore, get_store
from sift.tools import (
    _declared_source_datasets,
    _merge_sdc_configs,
    _sanitize_and_store_payloads,
)


def test_store_round_trips_complete_lineage_and_legacy_primary(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path / ".sift" / "results.db")
    inserted = store.insert(
        label="join", analysis_type="descriptive",
        sanitized_payload={"type": "descriptive", "n": 20},
        language="Python", script_code="x", transformations=[],
        source_dataset="people.csv",
        source_datasets=["people.csv", "events.parquet", "people.csv"],
    )
    assert inserted.source_dataset == "people.csv"
    assert inserted.all_source_datasets == ("people.csv", "events.parquet")
    store.close()

    reopened = ResultStore(tmp_path / ".sift" / "results.db")
    row = reopened.get("M1")
    assert row is not None
    assert row.source_datasets == ("people.csv", "events.parquet")
    assert row.all_source_datasets == ("people.csv", "events.parquet")
    reopened.close()


def test_declared_sources_are_canonical_deduplicated_and_sandboxed(
    tmp_path: Path,
) -> None:
    set_cwd(tmp_path)
    sources, error = _declared_source_datasets({
        "source_dataset": "./people.csv",
        "source_datasets": ["people.csv", "data/events.parquet"],
    }, tmp_path)
    assert error is None
    assert sources == ("people.csv", "data/events.parquet")

    sources, error = _declared_source_datasets({
        "source_datasets": ["../outside.csv"],
    }, tmp_path)
    assert sources == ()
    assert error and "outside the working directory" in error


def test_join_policy_is_never_looser_than_any_input() -> None:
    first = replace(
        DEFAULT_CONFIG,
        min_n_regression=12,
        dominance_threshold=0.75,
        non_disclosive_variables=frozenset({"age", "year"}),
        banned_variables=frozenset({"ssn"}),
    )
    second = replace(
        DEFAULT_CONFIG,
        min_n_regression=25,
        dominance_threshold=0.60,
        non_disclosive_variables=frozenset({"year", "region"}),
        banned_variables=frozenset({"diagnosis"}),
    )
    merged = _merge_sdc_configs([first, second])
    assert merged.min_n_regression == 25
    assert merged.dominance_threshold == 0.60
    assert merged.non_disclosive_variables == frozenset({"year"})
    assert merged.banned_variables == frozenset({"ssn", "diagnosis"})


def test_sanitize_store_response_and_rejection_keep_all_sources(
    tmp_path: Path,
) -> None:
    store = ResultStore(tmp_path / ".sift" / "results.db")
    raw = {
        "type": "descriptive", "variable": "x", "n": 20,
        "mean": 1.0, "sd": 0.5, "missing_count": 0,
    }
    results, ok, _, _ = _sanitize_and_store_payloads(
        [raw], cwd=tmp_path, label="join", language="Python", code="x",
        source_dataset="people.csv", source_n=None,
        sdc_cfg=DEFAULT_CONFIG, run_dir=None, script_run_id="run-join",
        store=store, source_datasets=("people.csv", "events.parquet"),
    )
    assert ok
    assert results[0]["source_datasets"] == ["people.csv", "events.parquet"]
    assert any("multi-dataset join" in t for t in results[0]["transformations"])
    row = store.get("M1")
    assert row is not None
    assert row.all_source_datasets == ("people.csv", "events.parquet")
    store.close()


def test_ledger_budget_and_fingerprint_attribute_each_join_input() -> None:
    response = json.dumps({
        "status": "ok",
        "results": [{
            "status": "ok", "analysis_type": "linear_regression", "n": 50,
            "source_dataset": "people.csv",
            "source_datasets": ["people.csv", "events.parquet"],
        }],
    })
    facts = _facts_from_response(response)
    record = {"tool": "submit_script", "facts": facts}
    assert consumed_for_dataset([record], "people.csv") == 1
    assert consumed_for_dataset([record], "events.parquet") == 1
    assert _submit_script_analysis_events([record]) == [
        {"dataset": "people.csv", "analysis_type": "linear_regression", "n": 50},
        {"dataset": "events.parquet", "analysis_type": "linear_regression", "n": 50},
    ]


def test_export_blocks_join_if_any_source_is_nonexportable_and_writes_lineage(
    tmp_path: Path,
) -> None:
    from sift.policy import DatasetPolicy, SiftPolicy, save_policy

    store = get_store(tmp_path)
    store.insert(
        label="joined finding", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x", "n": 20,
            "mean": 1.0, "sd": 0.5, "missing_count": 0,
        },
        language="Python", script_code="x", transformations=[],
        script_run_id="run-joined",
        source_datasets=["public.csv", "restricted.csv"],
    )
    store.insert(
        label="safe finding", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "y", "n": 20,
            "mean": 2.0, "sd": 0.5, "missing_count": 0,
        },
        language="Python", script_code="y", transformations=[],
        script_run_id="run-safe", source_datasets=["public.csv"],
    )
    save_policy(tmp_path, SiftPolicy(datasets={
        "restricted.csv": DatasetPolicy(exportable=False),
    }))

    dest = tmp_path / "export"
    summary = build_replication_package(tmp_path, dest)
    assert summary["results"] == 1
    assert summary["excluded_datasets"] == 1
    lineage = json.loads((dest / "provenance" / "lineage.json").read_text(encoding="utf-8"))
    assert lineage["model"] == "W3C PROV entity/activity"
    assert lineage["activities"][0]["used"] == ["public.csv"]

