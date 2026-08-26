"""Session advisories wiring — session_report's cross-result checks
(accumulated multiple comparisons, sample drift, and specification
search) surfaced to the model on ``submit_script``, not just to a
human who opens the Verification panel.

Prior to this, ``session_report`` was only ever called from ``ui.py``
(the Verification panel) and ``research_export.py`` (replication
package). The model — the one actually choosing what to run next —
never saw any of it. ``session_advisories`` closes that gap: the real
``submit_script`` response, seeded with real prior rows in the real
SQLite store, must carry the warn-level session checks.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sift import env_detect
from sift.config import use_cwd
from sift.store import get_store


def _python_ready() -> bool:
    e = env_detect.detect_environment()
    if e.python is None or not e.has_sandbox_backend():
        return False
    hard = {"pandas", "numpy"} & set(e.python.missing_packages)
    return not hard


_skip_no_python = pytest.mark.skipif(
    not _python_ready(),
    reason=(
        "needs python3 + pandas + numpy + a sandbox backend "
        "(sandbox-exec on macOS, bwrap on Linux)"
    ),
)


def _mcp_text(payload: dict) -> dict:
    import json
    return json.loads(payload["content"][0]["text"])


def _seed_prior_spec(cwd: Path, *, predictors: list[str],
                      p_values: dict[str, float], label: str,
                      script_run_id: str) -> None:
    """Insert a stored row shaped like a real prior ``submit_script``
    call in this session -- a separate, unbatched regression on the
    same response variable and dataset, the exact case
    ``challenge_summary`` cannot see because these were never one
    batch. ``script_run_id`` must be distinct per seeded call -- the
    specification-search detector requires results to span at least
    two distinct script runs before it fires at all (a single script
    emitting several specs together is the mandatory-robustness-pass
    shape challenge_summary already verifies, not this detector's
    target; see verification.py's module comment)."""
    store = get_store(cwd)
    store.insert(
        label=label,
        analysis_type="linear_regression",
        sanitized_payload={
            "type": "linear_regression",
            "response_variable": "wage",
            "predictor_variables": predictors,
            "coefficients": {k: 1.0 for k in p_values},
            "p_values": p_values,
            "n": 500,
        },
        language="Python",
        script_code="# prior spec",
        transformations=[],
        source_dataset="cohort.csv",
        script_run_id=script_run_id,
    )


@_skip_no_python
def test_session_advisories_flags_specification_search(tmp_path: Path) -> None:
    """Two prior, separately-stored specifications plus one real
    submit_script call completing a third distinct specification on
    the same (dataset, response_variable) must trip the count-based
    specification_search check in the REAL submit_script response."""
    from sift.tools import HANDLERS

    with use_cwd(tmp_path):
        _seed_prior_spec(tmp_path, predictors=["educ"],
                          p_values={"educ": 0.2}, label="spec1",
                          script_run_id="run-seed-1")
        _seed_prior_spec(tmp_path, predictors=["educ", "exper"],
                          p_values={"educ": 0.2, "exper": 0.3}, label="spec2",
                          script_run_id="run-seed-2")

        code = (
            "import sift\n"
            "sift.result(type='linear_regression', "
            "response_variable='wage', "
            "predictor_variables=['educ', 'exper', 'tenure'], "
            "coefficients={'educ': 0.2, 'exper': 0.3, 'tenure': 0.4}, "
            "standard_errors={'educ': 0.1, 'exper': 0.1, 'tenure': 0.1}, "
            "p_values={'educ': 0.2, 'exper': 0.3, 'tenure': 0.4}, n=500)\n"
        )

        async def _call():
            return await HANDLERS["submit_script"]({
                "language": "Python",
                "code": code,
                "label": "spec3",
                "source_dataset": "cohort.csv",
            })

        r = _mcp_text(asyncio.run(_call()))

    assert r["status"] == "ok", r
    assert "session_advisories" in r, r
    ids = [c["id"] for c in r["session_advisories"]]
    assert "specification_search::cohort.csv::wage" in ids


@_skip_no_python
def test_session_advisories_absent_on_first_result(tmp_path: Path) -> None:
    """A session's very first result has nothing to compare across —
    ``session_advisories`` must be entirely absent, not an empty list
    (matching this module's established "omit, don't fabricate"
    convention for challenge_summary and verification_note)."""
    from sift.tools import HANDLERS

    with use_cwd(tmp_path):
        code = (
            "import sift\n"
            "sift.from_summarize('outcome', n=42, mean=3.14, sd=0.5, "
            "missing_count=2)\n"
        )

        async def _call():
            return await HANDLERS["submit_script"]({
                "language": "Python",
                "code": code,
                "label": "only one",
            })

        r = _mcp_text(asyncio.run(_call()))

    assert r["status"] == "ok", r
    assert "session_advisories" not in r


@_skip_no_python
def test_session_advisories_absent_when_call_is_fully_rejected(
    tmp_path: Path,
) -> None:
    """A call where every emitted payload is rejected by the
    sanitizer added nothing new to the session -- must not spend a
    session_report pass or surface stale advisories."""
    from sift.tools import HANDLERS

    with use_cwd(tmp_path):
        _seed_prior_spec(tmp_path, predictors=["educ"],
                          p_values={"educ": 0.2}, label="spec1",
                          script_run_id="run-seed-1")

        # Missing required fields (standard_errors, predictor_variables)
        # -> sanitizer rejection, any_ok stays False.
        code = (
            "import sift\n"
            "sift.result(type='linear_regression', "
            "response_variable='wage', n=500)\n"
        )

        async def _call():
            return await HANDLERS["submit_script"]({
                "language": "Python",
                "code": code,
                "label": "malformed",
                "source_dataset": "cohort.csv",
            })

        r = _mcp_text(asyncio.run(_call()))

    assert r["status"] == "rejected_by_sanitizer", r
    assert "session_advisories" not in r
