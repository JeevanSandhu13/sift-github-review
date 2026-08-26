"""Policy engine extensions: banned variables and export rules.

Banned variables are enforced at TWO Sift-owned points — schema
exposure (get_schema/search_schema drop the variable from the
response entirely) and request_data (denies any request naming a
banned variable, checked against the RESOLVED real column). They are
deliberately NOT enforced inside the submit_script sanitizer — see
the long comment on ``SDCConfig.banned_variables`` for why a
name-based block there can't be trusted (the model/script controls
every label a submit_script payload carries, so a same-named check
there would filter what a script CALLS things, not what it actually
touches).

Export rules (``exportable: bool``) gate whether a dataset's
metadata, or results computed from it, may appear in an artifact
built from ``research_export.py`` (codebook, analysis report) —
independent of whether the underlying result is itself
disclosure-safe (it always is, by the time it reaches the store).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sift.config import set_cwd
from sift.policy import (
    DatasetPolicy,
    SiftPolicy,
    banned_for,
    is_exportable,
    load_policy,
    save_policy,
)
from sift.tools import get_schema, request_data, search_schema


def _call(tool, args: dict) -> dict:
    envelope = asyncio.run(tool.handler(args))
    return json.loads(envelope["content"][0]["text"])


@pytest.fixture()
def pd():
    return pytest.importorskip("pandas")


def _write_csv(tmp_path, pd, name="d.csv"):
    path = tmp_path / name
    pd.DataFrame({
        "age": [30, 40, 50, 60, 70],
        "ssn": ["111-11-1111", "222-22-2222", "333-33-3333",
                "444-44-4444", "555-55-5555"],
        "region": ["north", "south", "north", "south", "north"],
    }).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# policy.py: banned_for()
# ---------------------------------------------------------------------------

def test_banned_for_normalizes_via_safe_key(tmp_path: Path):
    """``banned_for`` normalizes via ``banned_key`` (safe_key PLUS
    case-folding), not bare ``safe_key`` — see the case-sensitivity
    regression tests further down for why the case-folding half of
    this matters on its own."""
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("SSN Number",)),
    }))
    from sift.policy import load_policy
    from sift.text_safety import banned_key
    policy = load_policy(tmp_path)
    banned = banned_for(policy, "d.csv")
    assert banned_key("SSN Number") in banned


def test_banned_for_empty_when_no_entry(tmp_path: Path):
    from sift.policy import load_policy
    policy = load_policy(tmp_path)
    assert banned_for(policy, "anything.csv") == frozenset()


# ---------------------------------------------------------------------------
# get_schema / search_schema: banned variables never named
# ---------------------------------------------------------------------------

def test_get_schema_drops_banned_variable_entirely(tmp_path: Path, pd):
    set_cwd(tmp_path)
    path = _write_csv(tmp_path, pd)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("ssn",)),
    }))
    resp = _call(get_schema, {"dataset": "d.csv", "depth": "names_only"})
    assert resp["status"] == "ok"
    names = [v["name"] for v in resp["variables"]]
    assert "ssn" not in names
    assert "age" in names
    assert "region" in names


def test_get_schema_no_bans_is_unaffected(tmp_path: Path, pd):
    set_cwd(tmp_path)
    _write_csv(tmp_path, pd)
    resp = _call(get_schema, {"dataset": "d.csv", "depth": "names_only"})
    assert resp["status"] == "ok"
    names = [v["name"] for v in resp["variables"]]
    assert set(names) == {"age", "ssn", "region"}


def test_search_schema_cannot_surface_a_banned_variable(tmp_path: Path, pd):
    set_cwd(tmp_path)
    _write_csv(tmp_path, pd)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("ssn",)),
    }))
    resp = _call(search_schema, {"dataset": "d.csv", "query": "ssn"})
    assert resp["status"] == "ok"
    assert resp["total_matches"] == 0


# ---------------------------------------------------------------------------
# request_data: banned variable refused regardless of request type
# ---------------------------------------------------------------------------

def test_request_data_refuses_banned_variable(tmp_path: Path, pd):
    set_cwd(tmp_path)
    _write_csv(tmp_path, pd)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("ssn",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "categorical_levels",
        "variable": "ssn",
    })
    assert resp["status"] == "denied"
    assert "banned" in resp["reason"].lower()


def test_request_data_allows_non_banned_variable(tmp_path: Path, pd):
    set_cwd(tmp_path)
    _write_csv(tmp_path, pd)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("ssn",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "categorical_levels",
        "variable": "region",
    })
    assert resp["status"] == "granted"


def test_request_data_refuses_banned_variable_as_correlation_pair_second_var(
    tmp_path: Path, pd,
):
    """variable2 (correlation_pair) is resolved separately from
    variable — must be checked too, not just the first argument."""
    set_cwd(tmp_path)
    path = tmp_path / "d.csv"
    pd.DataFrame({
        "income": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
        "salary_band": [1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3],
    }).to_csv(path, index=False)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("salary_band",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "correlation_pair",
        "variable": "income", "variable2": "salary_band",
    })
    assert resp["status"] == "denied"
    assert "banned" in resp["reason"].lower()


# ---------------------------------------------------------------------------
# Case-insensitivity (audit pass 2 finding): a policy's banned entry and
# the dataset's REAL column name routinely differ only in case. Before
# the fix, comparisons ran the ban list and the resolved column through
# ``safe_key`` (which does NOT casefold) on both sides, so a policy
# banning "ssn" silently let a column literally named "SSN" straight
# through get_schema/search_schema/request_data with no error, warning,
# or trace anywhere -- exactly the shape a real research dataset export
# would use (headers are conventionally upper/mixed case).
# ---------------------------------------------------------------------------

def _write_uppercase_ssn_csv(tmp_path, pd, name="d.csv"):
    path = tmp_path / name
    pd.DataFrame({
        "AGE": [30, 40, 50, 60, 70],
        "SSN": ["111-11-1111", "222-22-2222", "333-33-3333",
                "444-44-4444", "555-55-5555"],
        "REGION": ["north", "south", "north", "south", "north"],
    }).to_csv(path, index=False)
    return path


def test_get_schema_drops_banned_variable_despite_case_mismatch(
    tmp_path: Path, pd,
):
    """Dataset column is literally "SSN"; policy bans lowercase "ssn".
    Must still be dropped."""
    set_cwd(tmp_path)
    _write_uppercase_ssn_csv(tmp_path, pd)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("ssn",)),
    }))
    resp = _call(get_schema, {"dataset": "d.csv", "depth": "names_only"})
    assert resp["status"] == "ok"
    names = [v["name"] for v in resp["variables"]]
    assert "SSN" not in names, (
        "policy banned \"ssn\" but the real column \"SSN\" (different "
        "case) was NOT dropped -- the ban was silently defeated by the "
        "case mismatch"
    )
    assert "AGE" in names
    assert "REGION" in names


def test_request_data_refuses_banned_variable_despite_case_mismatch(
    tmp_path: Path, pd,
):
    """Same case-mismatch shape, but exercised through request_data's
    separate enforcement point (data_request._check_not_banned)."""
    set_cwd(tmp_path)
    _write_uppercase_ssn_csv(tmp_path, pd)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("ssn",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "categorical_levels",
        "variable": "SSN",
    })
    assert resp["status"] == "denied", (
        "policy banned \"ssn\" but a request for the real column "
        "\"SSN\" (different case) was GRANTED -- the ban was silently "
        "defeated by the case mismatch"
    )
    assert "banned" in resp["reason"].lower()


def test_policy_banned_uppercase_still_bans_lowercase_column(
    tmp_path: Path, pd,
):
    """Mirror case: policy bans "SSN" (uppercase, as an admin might
    type it to match a data dictionary), real column is lowercase
    "ssn" -- must also be dropped, confirming the fix is symmetric."""
    set_cwd(tmp_path)
    path = tmp_path / "d.csv"
    pd.DataFrame({
        "age": [30, 40, 50],
        "ssn": ["111-11-1111", "222-22-2222", "333-33-3333"],
    }).to_csv(path, index=False)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(banned_variables=("SSN",)),
    }))
    resp = _call(get_schema, {"dataset": "d.csv", "depth": "names_only"})
    names = [v["name"] for v in resp["variables"]]
    assert "ssn" not in names
    assert "age" in names


# ---------------------------------------------------------------------------
# non_disclosive_variables opt-in -- same case-mismatch shape as
# banned_variables above, exercised through request_data's
# numeric_bounds (the one place the opt-in is actually enforced --
# see sanitizer.py's SDCConfig.non_disclosive_variables comment).
# ---------------------------------------------------------------------------

def _write_numeric_age_csv(tmp_path, pd, *, column: str, name="d.csv"):
    path = tmp_path / name
    pd.DataFrame({column: list(range(18, 18 + 40))}).to_csv(path, index=False)
    return path


def test_non_disclosive_opt_in_survives_researcher_uppercase_typo(
    tmp_path: Path, pd,
):
    """Policy opts in "Age" (as a researcher glancing at a data
    dictionary might type it), real column is lowercase "age" -- the
    opt-in must still activate, not silently do nothing."""
    set_cwd(tmp_path)
    _write_numeric_age_csv(tmp_path, pd, column="age")
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(non_disclosive_variables=("Age",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "numeric_bounds",
        "variable": "age",
    })
    assert resp["status"] == "granted", resp
    answer = resp["answer"]
    assert "exact_min" in answer and "exact_max" in answer, (
        'policy opted in "Age" but the real column "age" (different '
        'case) got no exact bounds -- the opt-in was silently defeated '
        'by the case mismatch'
    )
    assert answer["exact_min"] == 18
    assert answer["exact_max"] == 57


def test_non_disclosive_opt_in_lowercase_still_matches_uppercase_column(
    tmp_path: Path, pd,
):
    """Mirror case: policy opts in "age" (lowercase), real column is
    "AGE" (uppercase) -- must also activate, confirming the fix is
    symmetric."""
    set_cwd(tmp_path)
    _write_numeric_age_csv(tmp_path, pd, column="AGE")
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(non_disclosive_variables=("age",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "numeric_bounds",
        "variable": "AGE",
    })
    assert resp["status"] == "granted", resp
    answer = resp["answer"]
    assert "exact_min" in answer and "exact_max" in answer


def test_non_disclosive_opt_in_does_not_leak_to_unrelated_variable(
    tmp_path: Path, pd,
):
    """The opt-in is per-variable -- a policy entry for "age" must
    not accidentally grant exact bounds to a different numeric column
    in the same dataset."""
    set_cwd(tmp_path)
    path = tmp_path / "d.csv"
    pd.DataFrame({
        "age": list(range(18, 58)),
        "income": [1000.0 + i for i in range(40)],
    }).to_csv(path, index=False)
    save_policy(tmp_path, SiftPolicy(datasets={
        "d.csv": DatasetPolicy(non_disclosive_variables=("age",)),
    }))
    resp = _call(request_data, {
        "dataset": "d.csv", "request_type": "numeric_bounds",
        "variable": "income",
    })
    assert resp["status"] == "granted", resp
    answer = resp["answer"]
    assert "exact_min" not in answer and "exact_max" not in answer


# ---------------------------------------------------------------------------
# banned_variables enforcement does NOT apply inside submit_script
# ---------------------------------------------------------------------------

def test_sdc_config_carries_banned_variables_but_sanitizer_does_not_use_them():
    """Documents the deliberate scope boundary: SDCConfig carries the
    field (so data_request can read it off the same object every
    other policy-derived knob rides on) but the sanitizer module
    itself never inspects it — see the long comment on the field."""
    from sift.sanitizer import DEFAULT_CONFIG, SDCConfig
    cfg = SDCConfig(banned_variables=frozenset({"ssn"}))
    assert cfg.banned_variables == frozenset({"ssn"})
    # No public sanitizer function takes a variable name to check
    # against banned_variables — the field exists purely for
    # data_request.py to read.
    import inspect
    import sift.sanitizer as sanitizer_mod
    src = inspect.getsource(sanitizer_mod)
    assert "config.banned_variables" not in src
    assert "config.banned_variables" not in inspect.getsource(SDCConfig)


# ---------------------------------------------------------------------------
# Export rules: is_exportable()
# ---------------------------------------------------------------------------

def test_is_exportable_defaults_true(tmp_path: Path):
    from sift.policy import load_policy
    policy = load_policy(tmp_path)
    assert is_exportable(policy, "anything.csv") is True


def test_is_exportable_false_when_set(tmp_path: Path):
    save_policy(tmp_path, SiftPolicy(datasets={
        "restricted.csv": DatasetPolicy(exportable=False),
    }))
    from sift.policy import load_policy
    policy = load_policy(tmp_path)
    assert is_exportable(policy, "restricted.csv") is False


def test_exportable_round_trips_through_save_and_load(tmp_path: Path):
    save_policy(tmp_path, SiftPolicy(datasets={
        "a.csv": DatasetPolicy(exportable=False),
        "b.csv": DatasetPolicy(exportable=True),
    }))
    raw = json.loads((tmp_path / ".sift" / "policy.json").read_text(encoding="utf-8"))
    # True is the default and omitted for file tidiness; False is
    # explicit researcher intent and must be persisted.
    assert raw["datasets"]["a.csv"]["exportable"] is False
    assert "exportable" not in raw["datasets"]["b.csv"]


@pytest.mark.parametrize("malformed", ["false", 0, None, [], {}])
def test_present_malformed_exportable_value_fails_closed(
    tmp_path: Path, malformed,
) -> None:
    import json

    policy_dir = tmp_path / ".sift"
    policy_dir.mkdir()
    (policy_dir / "policy.json").write_text(
        json.dumps(
            {
                "version": 1,
                "default_max_depth": "names_types_labels_summary",
                "datasets": {"restricted.csv": {"exportable": malformed}},
            }
        ),
        encoding="utf-8",
    )
    policy = load_policy(tmp_path)
    assert is_exportable(policy, "restricted.csv") is False


# ---------------------------------------------------------------------------
# build_codebook honors exportable
# ---------------------------------------------------------------------------

def test_codebook_excludes_non_exportable_dataset(tmp_path: Path, pd):
    from sift.research_export import build_codebook

    _write_csv(tmp_path, pd, name="public.csv")
    _write_csv(tmp_path, pd, name="restricted.csv")
    save_policy(tmp_path, SiftPolicy(datasets={
        "restricted.csv": DatasetPolicy(exportable=False),
    }))
    book = build_codebook(tmp_path)
    assert "## public.csv" in book["markdown"]
    assert "## restricted.csv" in book["markdown"]
    assert "Excluded from this export by dataset policy" in book["markdown"]
    # The exclusion must not leak the dataset's actual variable
    # names/labels into the CSV row either.
    import csv as _csv
    import io
    rows = list(_csv.DictReader(io.StringIO(book["csv"])))
    restricted_rows = [r for r in rows if r["dataset"] == "restricted.csv"]
    assert len(restricted_rows) == 1
    assert restricted_rows[0]["label"] == "EXCLUDED_BY_POLICY"
    assert restricted_rows[0]["variable"] == ""


# ---------------------------------------------------------------------------
# build_analysis_report honors exportable
# ---------------------------------------------------------------------------

def test_analysis_report_excludes_results_from_non_exportable_dataset(
    tmp_path: Path,
):
    from sift.research_export import build_analysis_report
    from sift.store import get_store

    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    store.insert(
        label="Public finding", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x", "n": 42,
            "mean": 3.14, "sd": 0.5, "missing_count": 0,
            "source_dataset": "public.csv",
        },
        language="Python", script_code="x", transformations=[],
        raw_log_path=None, script_run_id="r1",
        source_dataset="public.csv",
    )
    store.insert(
        label="Restricted finding", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "y", "n": 42,
            "mean": 1.0, "sd": 0.2, "missing_count": 0,
            "source_dataset": "restricted.csv",
        },
        language="Python", script_code="x", transformations=[],
        raw_log_path=None, script_run_id="r2",
        source_dataset="restricted.csv",
    )
    save_policy(tmp_path, SiftPolicy(datasets={
        "restricted.csv": DatasetPolicy(exportable=False),
    }))
    report = build_analysis_report(tmp_path)
    assert "Public finding" in report["markdown"]
    assert "Restricted finding" not in report["markdown"]
    assert "excluded from this report by dataset policy" in \
        report["markdown"].lower()


def test_analysis_report_includes_everything_with_no_restrictions(
    tmp_path: Path,
):
    from sift.research_export import build_analysis_report
    from sift.store import get_store

    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    store.insert(
        label="Finding A", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x", "n": 42,
            "mean": 3.14, "sd": 0.5, "missing_count": 0,
        },
        language="Python", script_code="x", transformations=[],
        raw_log_path=None, script_run_id="r1",
        source_dataset="d.csv",
    )
    report = build_analysis_report(tmp_path)
    assert "Finding A" in report["markdown"]
    assert "excluded from this report" not in report["markdown"].lower()
