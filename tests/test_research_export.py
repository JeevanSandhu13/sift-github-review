"""Research exports — LaTeX tables, replication packages, disclosure reports.

The load-bearing properties:

1. LaTeX escaping is safe by default. Data-origin text is escaped;
   only explicitly-marked Sift-authored markup passes through raw.
   Getting this backwards either breaks compilation or silently
   typesets the wrong thing.
2. Exported numbers are identical to the ones the researcher saw on
   screen (same formatters as ``result_render``).
3. A replication package contains no raw data and is complete enough
   to hand to a journal.
4. The disclosure report states its own limits rather than implying a
   guarantee Sift does not provide.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sift import release_ledger
from sift.research_export import (
    Raw,
    _cells_from_md_row,
    _md_table_to_html,
    _parse_md_table,
    build_disclosure_report,
    build_replication_package,
    capture_environment,
    latex_escape,
    render_latex_table,
)
from sift.store import get_store


def _regression_payload():
    return {
        "type": "linear_regression", "n": 28194,
        "coefficients": {"(Intercept)": 0.41, "tenure": -0.12, "share_%": 0.03},
        "standard_errors": {"(Intercept)": 0.02, "tenure": 0.03, "share_%": 0.01},
        "p_values": {"(Intercept)": 1e-9, "tenure": 0.0004, "share_%": 0.43},
        "r_squared": 0.148, "robust_se_type": "cluster",
        "response_variable": "churn", "predictor_variables": ["tenure"],
    }


def _seed_session(tmp_path: Path):
    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    store.insert(
        label="Churn on tenure", analysis_type="linear_regression",
        sanitized_payload=_regression_payload(), language="Python",
        script_code="import sift\n# fit\n",
        transformations=["clamped SEs at N=28194"],
        raw_log_path=None, script_run_id="run1",
    )
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="submit_script",
        args={"dataset": "customers.parquet"},
        response={"content": [{"type": "text",
                               "text": '{"status":"ok","n":28194}'}]},
    )
    return store


# --------------------------------------------------------------------
# LaTeX
# --------------------------------------------------------------------

def test_data_origin_text_is_escaped() -> None:
    for hostile, expected in [
        ("share_%", "share\\_\\%"),
        ("a&b", "a\\&b"),
        ("cost_$", "cost\\_\\$"),
        ("50#", "50\\#"),
        ("x^2", "x\\textasciicircum{}2"),
        ("{brace}", "\\{brace\\}"),
    ]:
        assert latex_escape(hostile) == expected


def test_raw_markup_passes_through_but_is_opt_in() -> None:
    """Default must be escape; only explicit Raw survives unescaped."""
    assert latex_escape(Raw("$p$-value")) == "$p$-value"
    assert latex_escape("$p$-value") == "\\$p\\$-value"


def test_regression_table_is_valid_booktabs() -> None:
    tex = render_latex_table(_regression_payload(),
                             caption="Churn", label="tab:churn")
    assert tex is not None
    for required in ("\\begin{table}", "\\toprule", "\\midrule",
                     "\\bottomrule", "\\end{tabular}", "\\end{table}"):
        assert required in tex
    # Our own math mode survives; data-origin text is escaped.
    assert "$p$-value" in tex
    assert "share\\_\\%" in tex
    # Relational operators must be in math mode, not bare.
    assert "$<$0.001" in tex
    assert "& <0.001" not in tex
    # Column count is consistent: 4 headers → 4 cells per body row.
    body = [ln for ln in tex.splitlines()
            if ln.endswith("\\\\") and "toprule" not in ln]
    assert all(ln.count("&") == 3 for ln in body), body


def test_notes_carry_n_fit_and_se_type() -> None:
    tex = render_latex_table(_regression_payload())
    assert "N = 28,194" in tex
    assert "$R^2$ = 0.148" in tex
    assert "cluster-robust standard errors" in tex


def test_unknown_type_returns_none_not_broken_latex() -> None:
    assert render_latex_table({"type": "no_such_type"}) is None
    assert render_latex_table({}) is None
    assert render_latex_table(None) is None  # type: ignore[arg-type]


def test_other_supported_types_render() -> None:
    assert render_latex_table(
        {"type": "descriptive", "variable": "age", "n": 500,
         "mean": 41.2, "sd": 12.1, "missing_count": 3}) is not None
    assert render_latex_table(
        {"type": "frequency_table", "variable": "region",
         "counts": {"north": 40, "south": "<10"}}) is not None
    assert render_latex_table(
        {"type": "t_test", "statistic": 2.1, "p_value": 0.03,
         "degrees_of_freedom": 88}) is not None


def test_suppression_markers_survive_into_latex() -> None:
    """A suppressed cell must remain visibly suppressed in the paper,
    never silently become a number or an empty cell."""
    tex = render_latex_table(
        {"type": "frequency_table", "variable": "region",
         "counts": {"north": 400, "islands": "<10"}})
    assert "<10" in tex


# --------------------------------------------------------------------
# Replication package
# --------------------------------------------------------------------

def test_replication_package_is_complete(tmp_path: Path) -> None:
    _seed_session(tmp_path)
    dest = tmp_path / "out"
    summary = build_replication_package(tmp_path, dest)
    assert summary["ok"] and summary["results"] == 1
    assert summary["ledger_chain_ok"] is True
    for rel in ("README.md", "METHODS.md", "environment.json",
                "disclosure/disclosure_report.md",
                "disclosure/release_ledger.jsonl"):
        assert (dest / rel).is_file(), rel
    assert list((dest / "results").glob("*.json"))
    assert list((dest / "tables").glob("*.tex"))
    assert list((dest / "tables").glob("*.md"))
    assert list((dest / "scripts").glob("*.py"))


def test_replication_package_carries_hashes_workflow_and_lifecycle(tmp_path: Path) -> None:
    store = _seed_session(tmp_path)
    first = store.list_all()[0]
    second = store.insert(
        label="Corrected", analysis_type="linear_regression",
        sanitized_payload=_regression_payload(), language="Python",
        script_code="# corrected", transformations=[], script_run_id="run2",
        source_dataset="customers.parquet",
        provenance={
            "dataset_hashes": {"customers.parquet": "c" * 64},
            "canonical_fingerprints": {"customers.parquet": "fp-1"},
            "script_sha256": "d" * 64, "workflow_id": "wf-1",
            "workflow_revision": 2, "analysis_id": "primary",
            "analysis_role": "primary", "random_seed": 17,
        },
    )
    store.supersede_result(first.id, second.id, reason="corrected coding", correction=True)
    (tmp_path / ".sift" / "research_workflow.json").write_text(
        '{"version":1,"workflow_id":"wf-1"}', encoding="utf-8",
    )
    (tmp_path / ".sift" / "project_memory.json").write_text(
        '{"privacy":{"contains_raw_data":false}}', encoding="utf-8",
    )
    dest = tmp_path / "out"
    build_replication_package(tmp_path, dest)
    lineage = json.loads((dest / "provenance" / "lineage.json").read_text(encoding="utf-8"))
    datasets = lineage["entities"]["datasets"]
    assert any(row.get("content_sha256") == "c" * 64 for row in datasets)
    results = lineage["entities"]["results"]
    assert any(row.get("analysis_role") == "primary" for row in results)
    assert (dest / "workflow" / "research_workflow.json").is_file()
    exported = [json.loads(path.read_text(encoding="utf-8")) for path in (dest / "results").glob("*.json")]
    assert any(row["lifecycle"]["status"] == "corrected" for row in exported)


def test_replication_package_never_reuses_existing_destination(
    tmp_path: Path,
) -> None:
    _seed_session(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    sentinel = dest / "stale_sensitive_result.json"
    sentinel.write_text("must not be mixed into a new export")
    with pytest.raises(FileExistsError):
        build_replication_package(tmp_path, dest)
    assert sentinel.read_text(encoding="utf-8") == "must not be mixed into a new export"


def test_replication_package_failure_never_publishes_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sift.research_export as module

    dest = tmp_path / "out"

    def _fail(_cwd, stage, *, include_scripts=True):
        (stage / "partial.txt").write_text("partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(module, "_build_replication_package_into", _fail)
    with pytest.raises(OSError, match="simulated write failure"):
        module.build_replication_package(tmp_path, dest)
    assert not dest.exists()
    assert not list(tmp_path.glob(".out.building-*"))


def test_replication_package_copies_accounting_gap_evidence(
    tmp_path: Path,
) -> None:
    _seed_session(tmp_path)
    release_ledger._note_recording_failure(tmp_path, OSError("disk full"))
    dest = tmp_path / "out"
    summary = build_replication_package(tmp_path, dest)
    assert summary["ledger_chain_ok"] is False
    assert "accounting gap" in summary["ledger_detail"]
    assert (
        dest / "disclosure" / release_ledger.LEDGER_HEALTH_FILENAME
    ).is_file()
    report = (dest / "disclosure" / "disclosure_report.md").read_text(encoding="utf-8")
    assert "INCONSISTENT" in report
    assert "accounting gap" in report


def test_replication_package_excludes_non_exportable_dataset(
    tmp_path: Path,
) -> None:
    """Regression test for architecture-audit finding G:
    ``build_replication_package`` iterated every stored result with no
    ``exportable: false`` policy check at all -- the ONE export
    surface (full result JSON + rendered tables + the exact script
    code) that skipped a check every other result-based export format
    (Markdown/HTML/PDF/PowerPoint) already applied.
    """
    from sift.policy import DatasetPolicy, SiftPolicy, save_policy

    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    store.insert(
        label="Public finding", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x", "n": 42,
            "mean": 3.14, "sd": 0.5, "missing_count": 0,
        },
        language="Python", script_code="print('public')",
        transformations=[], raw_log_path=None, script_run_id="r1",
        source_dataset="public.csv",
    )
    store.insert(
        label="Restricted finding", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "y", "n": 42,
            "mean": 1.0, "sd": 0.2, "missing_count": 0,
        },
        language="Python", script_code="print('restricted')",
        transformations=[], raw_log_path=None, script_run_id="r2",
        source_dataset="restricted.csv",
    )
    save_policy(tmp_path, SiftPolicy(datasets={
        "restricted.csv": DatasetPolicy(exportable=False),
    }))

    dest = tmp_path / "out"
    summary = build_replication_package(tmp_path, dest)

    assert summary["results"] == 1
    assert summary["excluded_datasets"] == 1
    result_files = list((dest / "results").glob("*.json"))
    assert len(result_files) == 1
    assert "Public" in result_files[0].read_text(encoding="utf-8")
    script_files = list((dest / "scripts").glob("*.py"))
    assert len(script_files) == 1
    assert "public" in script_files[0].read_text(encoding="utf-8")
    assert "restricted" not in script_files[0].read_text(encoding="utf-8")
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "exportable: false" in readme
    assert "1 result(s)" in readme


def test_replication_package_unaffected_with_no_restrictions(
    tmp_path: Path,
) -> None:
    _seed_session(tmp_path)
    dest = tmp_path / "out"
    summary = build_replication_package(tmp_path, dest)
    assert summary["excluded_datasets"] == 0
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "exportable: false" not in readme


def test_package_carries_verification_verdicts(tmp_path: Path) -> None:
    _seed_session(tmp_path)
    dest = tmp_path / "out"
    build_replication_package(tmp_path, dest)
    methods = (dest / "METHODS.md").read_text(encoding="utf-8")
    assert "[PASS]" in methods or "[WARN]" in methods
    assert "28,194" in methods
    # Disclosure control applied must be recorded, not hidden.
    assert "clamped SEs" in methods


def test_package_states_no_raw_data(tmp_path: Path) -> None:
    _seed_session(tmp_path)
    dest = tmp_path / "out"
    build_replication_package(tmp_path, dest)
    readme = (dest / "README.md").read_text(encoding="utf-8")
    assert "no raw data" in readme.lower()


def test_pipe_escaped_cell_survives_table_reparse() -> None:
    """Regression test for architecture-audit finding T:
    result_render._escape_table_cell escapes a data-origin ``|``
    inside a cell as a backslash-pipe sequence so a category label like "north | south"
    doesn't derail the markdown table's column count. The PDF/PPTX
    and HTML exporters both re-parse that same markdown table, and
    both used to split every row on a raw ``|`` -- including the
    escaped one -- silently shifting every following cell in the row
    out of alignment. This drives the real renderer (not a synthetic
    string) to confirm the escaped pipe survives a full
    render -> reparse round trip.
    """
    from sift.result_render import render_table

    payload = {
        "type": "frequency_table", "variable": "region",
        "counts": {"north | south": 400, "east": 100},
    }
    md = render_table(payload)
    assert md is not None
    assert "\\|" in md  # the renderer did escape it

    parsed = _parse_md_table(md)
    assert parsed is not None
    header, body, _caption = parsed
    assert len(header) == 3, header
    # Every body row must have exactly as many cells as the header --
    # the naive split produced 4 for the row containing the escaped
    # pipe (splitting the escaped "north...south" cell into two).
    for row in body:
        assert len(row) == len(header), (row, header)
    first_col = [row[0] for row in body]
    assert "north | south" in first_col
    assert "east" in first_col

    html = _md_table_to_html(md)
    assert "north | south" in html
    assert "north \\|" not in html


def test_cells_from_md_row_handles_literal_backslash() -> None:
    """A cell containing a literal backslash (not a pipe-escape) must
    round-trip too -- the reversal order matters here (undo the
    pipe-escape before the backslash-escape, mirroring the forward
    escape's own documented ordering requirement)."""
    from sift.result_render import _escape_table_cell

    raw = "C:\\path\\to|file"
    escaped = _escape_table_cell(raw)
    row = f"| {escaped} | 2 |"
    cells = _cells_from_md_row(row)
    assert cells[0] == raw
    assert cells[1] == "2"


def test_environment_capture_has_no_machine_identifiers() -> None:
    env = capture_environment()
    assert "python_version" in env and "python_packages" in env
    blob = json.dumps(env).lower()
    # Reproducibility needs versions, not who ran it or from where.
    for leaky in ("/users/", "/home/", "username", "hostname"):
        assert leaky not in blob


def test_empty_session_still_exports(tmp_path: Path) -> None:
    """A researcher who exports before running anything gets a valid,
    honest package rather than a crash."""
    (tmp_path / ".sift").mkdir()
    dest = tmp_path / "out"
    summary = build_replication_package(tmp_path, dest)
    assert summary["ok"] and summary["results"] == 0
    assert "No results recorded" in (dest / "METHODS.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------
# Disclosure report
# --------------------------------------------------------------------

def test_disclosure_report_lists_releases_and_states_limits(
        tmp_path: Path) -> None:
    _seed_session(tmp_path)
    report = build_disclosure_report(tmp_path)
    assert "customers.parquet" in report
    assert "submit_script" in report
    assert "verified" in report
    # The honesty requirements: it must not overclaim.
    low = report.lower()
    assert "does not bound what could be inferred" in low
    assert "tamper-evident, not tamper-proof" not in low  # phrased in README
    assert "not a formal privacy guarantee" in low or "does not" in low
    assert "differential-privacy" in low


def test_disclosure_report_surfaces_a_broken_chain(tmp_path: Path) -> None:
    _seed_session(tmp_path)
    path = release_ledger.ledger_path(tmp_path)
    rec = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    rec["tool"] = "tampered"
    path.write_text(json.dumps(rec) + "\n")
    report = build_disclosure_report(tmp_path)
    assert "INCONSISTENT" in report


def test_empty_ledger_report_is_valid(tmp_path: Path) -> None:
    (tmp_path / ".sift").mkdir()
    report = build_disclosure_report(tmp_path)
    assert "Total releases recorded: **0**" in report
    assert "no releases recorded" in report


def test_batch_script_written_once_not_per_result(tmp_path: Path) -> None:
    """A script that emitted several results is one script. Writing it
    once per result would pad the package and obscure the run order."""
    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    for i in range(4):
        store.insert(
            label=f"spec {i}", analysis_type="linear_regression",
            sanitized_payload=_regression_payload(), language="Python",
            script_code="# one batch script emitting four results\n",
            transformations=[], raw_log_path=None,
            script_run_id="shared_run",     # same run for all four
        )
    dest = tmp_path / "out"
    summary = build_replication_package(tmp_path, dest)
    assert summary["results"] == 4
    assert summary["scripts"] == 1
    assert len(list((dest / "scripts").glob("*.py"))) == 1
