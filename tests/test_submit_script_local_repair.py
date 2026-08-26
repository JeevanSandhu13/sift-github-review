"""End-to-end tests for deterministic local repair.

Real subprocess executions through ``submit_script`` — a script whose
only problem is a flagged gremlin character (smart quote / zero-width
/ NBSP) must be silently-but-disclosed repaired and re-run locally,
with no model round trip spent. A script that fails for an unrelated
reason must be left completely alone, including not paying for a
second subprocess run it has no chance of needing.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sift.config import set_cwd
from sift.env_detect import detect_environment
from sift.store import get_store, reset_store_for_tests
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
def test_curly_quotes_are_auto_repaired_and_rerun(tmp_path: Path) -> None:
    """A script broken ONLY by curly quotes around a string argument
    must come back as an overall success, with ``local_repair``
    disclosing exactly what was changed."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    # The curly quotes below are not valid Python string delimiters —
    # this genuinely fails to parse as written.
    code = (
        "import sift\n"
        "sift.from_summarize(‘income’, n=100, mean=50000.0, sd=1200.0, "
        "missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "curly quote canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok", body
    assert len(body["results"]) == 1
    assert "local_repair" in body
    note = body["local_repair"]
    assert "auto-corrected" in note
    assert "quotation mark" in note
    assert "not your original submission" in note


@_skip_no_python
def test_zero_width_space_in_identifier_is_auto_repaired(tmp_path: Path) -> None:
    """A zero-width space spliced into an otherwise-valid identifier
    breaks Python's tokenizer; stripping it must fix the run."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import sift\n"
        "s​ift.from_summarize('x', n=100, mean=1.0, sd=0.5, "
        "missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "zero-width canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok", body
    assert "local_repair" in body
    assert "zero-width" in body["local_repair"]


@_skip_no_python
def test_repair_attempted_but_does_not_fix_unrelated_failure(tmp_path: Path) -> None:
    """Curly quotes are present, but the script is ALSO broken for a
    real reason (calling an undefined function). The repaired
    version must still fail, so the repair is discarded: overall
    status is the original failure, and the note says the fix
    didn't help rather than claiming success."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import sift\n"
        "x = ‘hello’\n"
        "this_function_does_not_exist()\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "unfixable canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] in ("execution_failed", "execution_failed_partial")
    assert "local_repair" in body
    note = body["local_repair"]
    assert "still failed" in note
    assert "no need to retry the same character fix" in note
    # The ORIGINAL exec_result must be what's kept — not the repaired
    # attempt's own (different) failure. The repaired script fails
    # with a NameError on ``this_function_does_not_exist``, but that
    # second run's failure is discarded entirely; the debug_excerpt
    # the model sees still points at the original curly-quote parse
    # failure, proving nothing from the discarded repair attempt
    # leaked into the response.
    excerpt = body.get("debug_excerpt", "")
    assert "SyntaxError" in excerpt
    assert "this_function_does_not_exist" not in excerpt


@_skip_no_python
def test_unrelated_failure_does_not_trigger_a_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A script with NO gremlin characters that simply fails must
    incur exactly one subprocess execution — the whole value
    proposition is that this feature costs nothing when it doesn't
    apply. Spies on ``_execute_script_for_submit`` to prove it."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    calls = {"n": 0}
    from sift import tools as tools_mod
    real = tools_mod._execute_script_for_submit

    async def counting(language, code, cwd):
        calls["n"] += 1
        return await real(language, code, cwd)

    monkeypatch.setattr(tools_mod, "_execute_script_for_submit", counting)

    code = "import sift\nthis_function_does_not_exist()\n"
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "plain failure canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] in ("execution_failed", "execution_failed_partial")
    assert "local_repair" not in body
    assert calls["n"] == 1, (
        f"_execute_script_for_submit called {calls['n']} times for a "
        f"script with no gremlin characters — should be exactly 1"
    )


@_skip_no_python
def test_clean_successful_script_never_triggers_repair(tmp_path: Path) -> None:
    """A script that succeeds on the first try must never even reach
    the repair-detection code path (it's gated on ``not exec_result.ok``)."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import sift\n"
        "sift.from_summarize('clean', n=10, mean=1.0, sd=0.1, "
        "missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "clean canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok"
    assert "local_repair" not in body


@_skip_no_python
def test_repaired_run_is_stored_under_its_own_result(tmp_path: Path) -> None:
    """The stored result row must reflect the REPAIRED script's
    output (script_code column), not silently keep the broken
    original — the researcher's audit trail (expand_result / Evidence
    panel) should show what actually ran."""
    set_cwd(tmp_path)
    reset_store_for_tests()

    code = (
        "import sift\n"
        "sift.from_summarize(‘repaired_var’, n=10, mean=2.0, sd=0.5, "
        "missing_count=0)\n"
    )
    response = asyncio.run(submit_script.handler({
        "language": "Python",
        "code": code,
        "label": "audit trail canary",
        "source_dataset": "",
    }))
    body = _text_payload(response)
    assert body["status"] == "ok", body
    result_id = body["results"][0]["result_id"]
    store = get_store(tmp_path)
    row = store.get(result_id)
    assert row is not None
    # The stored script is the REPAIRED (straight-quote) version.
    assert "‘" not in row.script_code
    assert "'repaired_var'" in row.script_code
