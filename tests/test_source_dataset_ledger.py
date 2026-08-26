"""Regression test for the submit_script "source_dataset" ledger blind
spot (architecture audit finding E).

Before the fix, a submit_script result's response JSON never carried
a "source_dataset" key anywhere (the store row got it via
``store.insert(source_dataset=...)``, but the outward-facing
``result_entry`` dict fed to the model — and hashed/fact-extracted
into the release ledger — did not). ``privacy_budget.py``'s
per-dataset adaptive-suppression accounting and ``query_fingerprint.
py``'s repeated-query detection both key off exactly this field being
present in the ledger's recorded facts, so both silently saw ZERO
submit_script consumption against any dataset, no matter how many
granted releases actually happened — for the single most disclosure-
heavy tool in the system.

This test exercises the real ``submit_script`` handler end-to-end
(not a synthetic ledger record) and checks the fix holds all the way
through: response shape, ledger facts, and both downstream consumers.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sift.config import set_cwd
from sift.env_detect import detect_environment
from sift.privacy_budget import consumed_for_dataset
from sift.query_fingerprint import analyze_ledger
from sift.release_ledger import read_ledger
from sift.store import reset_store_for_tests
from sift.tools import submit_script


def _python_ready() -> bool:
    e = detect_environment()
    if e.python is None or not e.has_sandbox_backend():
        return False
    return not ({"pandas", "numpy"} & set(e.python.missing_packages))


_skip_no_python = pytest.mark.skipif(
    not _python_ready(),
    reason="needs python3 + pandas + numpy + a sandbox backend (sandbox-exec or bwrap)",
)


def _text_payload(response: dict) -> dict:
    text_block = next(
        b for b in response["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


@_skip_no_python
def test_submit_script_response_carries_source_dataset_per_result(
    tmp_path: Path,
) -> None:
    set_cwd(tmp_path)
    reset_store_for_tests()

    import pandas as pd
    df = pd.DataFrame({"x": list(range(20))})
    src = tmp_path / "tiny.csv"
    df.to_csv(src, index=False)

    code = (
        "import sift\n"
        "for i in range(3):\n"
        "    sift.from_summarize(f'v{i}', n=10, mean=float(i), "
        "sd=0.1, missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "source_dataset ledger regression",
        "source_dataset": "tiny.csv",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    assert len(body["results"]) == 3
    for entry in body["results"]:
        assert entry["source_dataset"] == "tiny.csv", (
            "submit_script's per-result response entry must carry "
            "source_dataset -- without it the release ledger's fact "
            "extraction has nothing to find (see this test's "
            "module docstring)"
        )


@_skip_no_python
def test_submit_script_source_dataset_reaches_privacy_budget_and_fingerprint(
    tmp_path: Path,
) -> None:
    set_cwd(tmp_path)
    reset_store_for_tests()

    import pandas as pd
    df = pd.DataFrame({"x": list(range(20))})
    src = tmp_path / "tiny.csv"
    df.to_csv(src, index=False)

    code = "import sift\nsift.from_summarize('x', n=10, mean=1.0, sd=0.1, missing_count=0)\n"
    for _ in range(3):
        response = asyncio.run(submit_script.handler({
            "language": "Python",
            "code": code,
            "label": "budget accounting regression",
            "source_dataset": "tiny.csv",
        }))
        body = _text_payload(response)
        assert body["status"] == "ok"

    records = read_ledger(tmp_path)
    consumed = consumed_for_dataset(records, "tiny.csv")
    assert consumed == 3, (
        f"expected 3 granted submit_script releases counted against "
        f"tiny.csv, got {consumed} -- the source_dataset field isn't "
        f"reaching privacy_budget.py's per-dataset accounting"
    )

    report = analyze_ledger(tmp_path)
    # query_fingerprint's repeated-analysis detection only has
    # anything to say about submit_script calls once it can see their
    # (dataset, analysis_type, n) tuples at all -- confirm the events
    # extractor actually picked up all three calls against tiny.csv.
    from sift.query_fingerprint import _submit_script_analysis_events
    events = _submit_script_analysis_events(records)
    matching = [e for e in events if e["dataset"] == "tiny.csv"]
    assert len(matching) == 3
    assert report is not None  # never raises; smoke check it ran
