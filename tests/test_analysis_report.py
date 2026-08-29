"""Analysis report — assembly rules and the figure allowlist.

The report's promise is threefold: everything in it comes from stored
sanitized material, warnings are surfaced rather than buried, and the
only figures it embeds are ones already cleared for sharing. The
figure allowlist is the critical test: a report that embedded
raw-data plots would be a disclosure channel dressed as a feature.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from sift import release_ledger
from sift import research_export
from sift.research_export import build_analysis_report
from sift.store import get_store

# 1x1 transparent PNG.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")


def _seed(tmp_path: Path, *, robust="classical"):
    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    store.insert(
        label="Churn on tenure", analysis_type="linear_regression",
        sanitized_payload={
            "type": "linear_regression", "n": 28194,
            "coefficients": {"(Intercept)": 0.41, "tenure": -0.12},
            "standard_errors": {"(Intercept)": 0.02, "tenure": 0.03},
            "p_values": {"(Intercept)": 0.001, "tenure": 0.0004},
            "r_squared": 0.148, "robust_se_type": robust,
        },
        language="Python", script_code="x",
        transformations=["clamped SEs"], raw_log_path=None,
        script_run_id="r1")
    release_ledger.record_release(
        tmp_path, kind="tool_response", tool="submit_script",
        response={"content": [{"type": "text", "text": '{"status":"ok"}'}]})


def _plot(tmp_path: Path, kind: str, fname: str, label: str,
          run: str = "run1") -> None:
    d = tmp_path / ".sift" / "runs" / run / "_sift_plots"
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_bytes(_PNG)
    with (d / "manifest.jsonl").open("a") as fh:
        fh.write(json.dumps(
            {"kind": kind, "file": fname, "label": label}) + "\n")


def test_report_contains_findings_verdicts_and_ledger(tmp_path) -> None:
    _seed(tmp_path)
    rep = build_analysis_report(tmp_path)
    for surface in (rep["markdown"], rep["html"]):
        assert "Churn on tenure" in surface
        assert "28,194" in surface
        # The warning must be surfaced, and before the passes.
        assert "classical (non-robust)" in surface
        assert surface.index("classical (non-robust)") < surface.index(
            "adequate for the reported statistics")
        assert "clamped SEs" in surface
    assert "Disclosures to the external model: **1**" in rep["markdown"]


def test_figure_allowlist_is_enforced(tmp_path) -> None:
    """Only manifest-listed helper kinds embed. A raw-data plot with a
    forged kind outside the allowlist, or a file with no manifest
    entry at all, must never appear."""
    _seed(tmp_path)
    _plot(tmp_path, "coefficients", "coef.png", "Tenure effect")
    _plot(tmp_path, "raw_histogram", "hist.png", "FORBIDDEN-KIND")
    # File in the plots dir with no manifest entry at all:
    (tmp_path / ".sift" / "runs" / "run1" / "_sift_plots"
     / "orphan.png").write_bytes(_PNG)

    rep = build_analysis_report(tmp_path)
    assert "Tenure effect" in rep["html"]
    assert "FORBIDDEN-KIND" not in rep["html"]
    assert rep["html"].count("data:image/png") == 1


def test_oversized_figures_are_skipped(tmp_path) -> None:
    _seed(tmp_path)
    d = tmp_path / ".sift" / "runs" / "run1" / "_sift_plots"
    d.mkdir(parents=True)
    (d / "big.png").write_bytes(b"\x89PNG" + b"0" * (4 * 1024 * 1024))
    (d / "manifest.jsonl").write_text(json.dumps(
        {"kind": "coefficients", "file": "big.png", "label": "big"}) + "\n")
    rep = build_analysis_report(tmp_path)
    assert "data:image/png" not in rep["html"]


def test_traversal_in_manifest_filename_is_neutralised(tmp_path) -> None:
    """A manifest entry naming ``../../secret.png`` must resolve to a
    basename inside the plots dir, never outside it."""
    _seed(tmp_path)
    secret = tmp_path / "secret.png"
    secret.write_bytes(_PNG)
    d = tmp_path / ".sift" / "runs" / "run1" / "_sift_plots"
    d.mkdir(parents=True)
    (d / "manifest.jsonl").write_text(json.dumps(
        {"kind": "coefficients", "file": "../../../secret.png",
         "label": "esc"}) + "\n")
    rep = build_analysis_report(tmp_path)
    # basename 'secret.png' doesn't exist inside _sift_plots → skipped.
    assert "data:image/png" not in rep["html"]


def test_empty_session_report_is_honest(tmp_path) -> None:
    (tmp_path / ".sift").mkdir()
    rep = build_analysis_report(tmp_path)
    assert "No analysis results have been recorded" in rep["markdown"]
    assert "Results: **0**" in rep["markdown"]


def test_unreadable_result_store_is_not_reported_as_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sift import store

    class BrokenStore:
        def list_all(self):
            raise OSError("simulated corruption")

    monkeypatch.setattr(store, "get_store", lambda _cwd: BrokenStore())
    with pytest.raises(RuntimeError, match="could not read the result store"):
        build_analysis_report(tmp_path)


def test_verification_failures_are_visible_not_silent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _seed(tmp_path)
    monkeypatch.setattr(
        research_export, "verify_payload",
        lambda _payload: (_ for _ in ()).throw(RuntimeError("simulated")),
    )
    from sift import verification

    monkeypatch.setattr(
        verification, "session_report",
        lambda _rows: (_ for _ in ()).throw(ValueError("simulated")),
    )
    report = build_analysis_report(tmp_path)["markdown"]
    assert "Result verification was unavailable (RuntimeError)" in report
    assert "Across-result verification was unavailable (ValueError)" in report


def test_html_is_self_contained_and_escaped(tmp_path) -> None:
    (tmp_path / ".sift").mkdir()
    store = get_store(tmp_path)
    store.insert(
        label='<script>alert(1)</script>', analysis_type="descriptive",
        sanitized_payload={"type": "descriptive", "variable": "x<y&z",
                           "n": 50, "mean": 1.0},
        language="R", script_code="x", transformations=[],
        raw_log_path=None, script_run_id="r1")
    html = build_analysis_report(tmp_path)["html"]
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
    # Self-contained: no external fetches of any kind.
    for external in ("http://", "https://", "src=\"/", "href="):
        assert external not in html
    assert html.startswith("<!DOCTYPE html>")


def test_report_never_raises_on_corrupt_state(tmp_path) -> None:
    (tmp_path / ".sift").mkdir()
    # Corrupt ledger + corrupt manifest + result whose payload breaks
    # the table renderer.
    release_ledger.ledger_path(tmp_path).write_text("{not json\n")
    d = tmp_path / ".sift" / "runs" / "runX" / "_sift_plots"
    d.mkdir(parents=True)
    (d / "manifest.jsonl").write_text("also not json\n")
    store = get_store(tmp_path)
    store.insert(label="weird", analysis_type="linear_regression",
                 sanitized_payload={"type": "linear_regression",
                                    "n": 10 ** 400,
                                    "coefficients": {"a": 10 ** 400}},
                 language="Python", script_code="x", transformations=[],
                 raw_log_path=None, script_run_id="r1")
    rep = build_analysis_report(tmp_path)
    assert rep["markdown"] and rep["html"]


def test_across_result_checks_surface_session_sample_drift(
    tmp_path: Path,
) -> None:
    """Regression test: ``_gather_report_material``'s session_report
    call must read ``source_dataset`` from each stored row's own
    column, not from the sanitized payload -- no sanitizer analysis
    shape has ever put a "source_dataset" key inside the payload, so
    reading it from there always returned None and made the
    sample-size-drift check permanently inert in this report,
    regardless of how much N actually moved between two results on
    the same dataset. Two results on the same dataset with N far
    enough apart to cross the drift threshold must produce an
    "Across-result checks" section naming both sample sizes.
    """
    (tmp_path / ".sift").mkdir(exist_ok=True)
    store = get_store(tmp_path)
    store.insert(
        label="full sample", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x", "n": 5000,
            "mean": 3.14, "sd": 0.5, "missing_count": 0,
        },
        language="Python", script_code="print('a')",
        transformations=[], raw_log_path=None, script_run_id="r1",
        source_dataset="cohort.csv",
    )
    store.insert(
        label="filtered sample", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "x", "n": 3200,
            "mean": 3.20, "sd": 0.4, "missing_count": 0,
        },
        language="Python", script_code="print('b')",
        transformations=[], raw_log_path=None, script_run_id="r2",
        source_dataset="cohort.csv",
    )

    rep = build_analysis_report(tmp_path)

    assert "Across-result checks" in rep["markdown"]
    assert "3,200" in rep["markdown"] and "5,000" in rep["markdown"]
