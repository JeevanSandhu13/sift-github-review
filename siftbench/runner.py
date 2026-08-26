"""SiftBench runner: execute a case's reference script through Sift's
real pipeline and score the result.

Deliberately thin. All the interesting logic — dataset construction,
ground truth, scoring — lives on the :class:`~siftbench.cases.BenchCase`
itself; this module's only job is running ``submit_script`` against a
throwaway session directory and handing the parsed response + result
store to the case's scorer.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from siftbench.cases import BenchCase, ScoreResult

_SAFE_CASE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")


@dataclass
class CaseRun:
    case_id: str
    description: str
    score: ScoreResult
    status: str  # the submit_script envelope status, for diagnostics
    duration_seconds: float


def _text_payload(response: dict[str, Any]) -> dict[str, Any]:
    text_block = next(
        b for b in response["content"] if b.get("type") == "text"
    )
    return json.loads(text_block["text"])


def run_case(case: BenchCase, cwd: Path) -> CaseRun:
    """Run one :class:`BenchCase` against a fresh session at ``cwd``.

    ``cwd`` must be an empty, writable directory the caller owns —
    typically a ``tmp_path`` in tests, or a throwaway dir from the
    CLI. Sets up Sift's cwd + result store exactly the way a real
    session would, submits the case's reference script through the
    real ``submit_script`` tool handler (same sandboxed executor,
    same disclosure-control sanitizer), and hands the parsed response
    plus the store to the case's ``score`` function.
    """
    import time as _time

    from sift.config import set_cwd
    from sift.store import close_store, get_store
    from sift.tools import submit_script

    t0 = _time.monotonic()
    if not _SAFE_CASE_ID.fullmatch(case.id):
        return CaseRun(
            case_id=case.id,
            description=case.description,
            score=ScoreResult(False, "case id is not a safe directory name"),
            status="runner_error",
            duration_seconds=_time.monotonic() - t0,
        )

    try:
        cwd.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        if any(cwd.iterdir()):
            return CaseRun(
                case_id=case.id,
                description=case.description,
                score=ScoreResult(
                    False,
                    "benchmark directory is not empty; refusing contaminated run",
                ),
                status="runner_error",
                duration_seconds=_time.monotonic() - t0,
            )

    try:
        set_cwd(cwd)
        response = asyncio.run(submit_script.handler({
            "language": case.language,
            "code": case.reference_script,
            "label": f"siftbench: {case.id}",
            "source_dataset": case.source_dataset,
        }))
        body = _text_payload(response)
        store = get_store(cwd)
        result = case.score(body, store)
        status = body.get("status", "<no status>")
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort suite
        result = ScoreResult(False, f"benchmark runner error: {type(exc).__name__}")
        status = "runner_error"
    finally:
        close_store(cwd)
    duration = _time.monotonic() - t0

    return CaseRun(
        case_id=case.id,
        description=case.description,
        score=result,
        status=status,
        duration_seconds=duration,
    )


def run_all(cases: list[BenchCase], base_dir: Path) -> list[CaseRun]:
    """Run every case in ``cases``, each under its own subdirectory
    of ``base_dir`` (so cases never share a result store or chat
    history — each is a fully independent session)."""
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("benchmark case ids must be unique")

    runs = []
    for case in cases:
        case_dir = base_dir / case.id
        runs.append(run_case(case, case_dir))
    return runs


def summarize(runs: list[CaseRun]) -> dict[str, Any]:
    """Build a small JSON-serializable report: pass/fail counts and
    per-case detail. This is the "score" SiftBench reports — a
    fraction, not a single pass/fail, so a partial regression is
    visible rather than binary."""
    passed = sum(1 for r in runs if r.score.passed)
    total = len(runs)
    return {
        "passed": passed,
        "total": total,
        "score": (passed / total) if total else 0.0,
        "cases": [
            {
                "id": r.case_id,
                "description": r.description,
                "passed": r.score.passed,
                "message": r.score.message,
                "envelope_status": r.status,
                "duration_seconds": round(r.duration_seconds, 4),
            }
            for r in runs
        ],
    }
