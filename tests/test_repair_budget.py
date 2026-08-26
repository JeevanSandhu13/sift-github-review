"""Bounded repair budget — the circuit breaker on failing scripts.

Without a bound, a model that misreads an error can resubmit a
near-identical script indefinitely: a subprocess run and a full turn
of tokens per attempt, with a spinner and a bill at the researcher's
end. These tests pin when the breaker trips, when it stays out of the
way, and that it never becomes a new failure mode itself.
"""

from __future__ import annotations

import pytest

from sift import repair_budget
from sift.repair_budget import (
    MAX_CONSECUTIVE_FAILURES,
    MAX_IDENTICAL_ATTEMPTS,
    guidance,
    record_failure,
    record_success,
)


@pytest.fixture(autouse=True)
def _clean_state():
    repair_budget.reset_all()
    yield
    repair_budget.reset_all()


def test_stays_quiet_while_genuinely_debugging() -> None:
    """Different scripts failing a couple of times is ordinary work
    and must not be nagged."""
    for i in range(MAX_CONSECUTIVE_FAILURES - 1):
        state = record_failure("/s", f"import pandas; approach_{i}()")
        assert state["exhausted"] is False
        assert guidance(state) is None


def test_trips_after_consecutive_failures() -> None:
    state = {}
    for i in range(MAX_CONSECUTIVE_FAILURES):
        state = record_failure("/s", f"totally_different_script_{i}()")
    assert state["exhausted"] is True
    advice = guidance(state)
    assert advice and "Stop retrying" in advice
    # It must route to the researcher, not to another attempt.
    assert "researcher" in advice


def test_identical_resubmission_trips_sooner() -> None:
    """Repeating the same script cannot inform anything new."""
    code = "df = pd.read_csv('x.csv')\nbroken(\n"
    for _ in range(MAX_IDENTICAL_ATTEMPTS):
        state = record_failure("/s", code)
        assert state["exhausted"] is False
    state = record_failure("/s", code)
    assert state["exhausted"] is True
    assert "same script" in guidance(state)


def test_reindentation_counts_as_identical() -> None:
    """Layout-only edits are the same script. Indentation and blank
    lines are normalised; spacing that changes tokenisation is not
    (see ``_hash_code``), and the consecutive counter backstops that."""
    for variant in ("a=1\nbroken(", "  a=1\n\n  broken(", "a=1\n\n\nbroken("):
        state = record_failure("/s", variant)
    assert state["identical_repeats"] == 3
    assert state["exhausted"] is True


def test_token_changing_respacing_is_treated_as_a_new_script() -> None:
    """Documented limit of the fingerprint, pinned so a future change
    to ``_hash_code`` is a deliberate decision rather than a surprise."""
    record_failure("/s", "a=1\nbroken(")
    state = record_failure("/s", "a = 1\nbroken(")
    assert state["identical_repeats"] == 1
    # Still counted toward the consecutive-failure backstop.
    assert state["consecutive_failures"] == 2


def test_success_resets_the_budget() -> None:
    for i in range(MAX_CONSECUTIVE_FAILURES - 1):
        record_failure("/s", f"x{i}()")
    record_success("/s")
    state = record_failure("/s", "y()")
    assert state["consecutive_failures"] == 1
    assert state["exhausted"] is False


def test_sessions_are_independent() -> None:
    """One session's stuck loop must not nag a different session."""
    for i in range(MAX_CONSECUTIVE_FAILURES):
        record_failure("/session-a", f"x{i}()")
    state = record_failure("/session-b", "fresh()")
    assert state["consecutive_failures"] == 1
    assert state["exhausted"] is False


def test_never_raises_on_bad_input() -> None:
    assert record_failure(None, "")["exhausted"] is False
    assert record_failure("/s", None)["consecutive_failures"] >= 0  # type: ignore[arg-type]
    record_success(None)


def test_guidance_is_none_until_exhausted() -> None:
    assert guidance({"exhausted": False}) is None
    assert guidance({}) is None


def test_guidance_tolerates_malformed_persisted_counts() -> None:
    advice = guidance({
        "exhausted": True,
        "identical_repeats": {"not": "a count"},
        "consecutive_failures": float("nan"),
    })
    assert advice and "Stop retrying" in advice


def test_wired_into_failing_script_responses(tmp_path, monkeypatch) -> None:
    """End-to-end: a failing run attaches the advisory once the budget
    is exhausted, and an ok run clears it."""
    import json

    from sift.config import use_cwd
    from sift.store import get_store
    from sift.tools import _attach_status_metadata

    (tmp_path / ".sift").mkdir()
    store = get_store(tmp_path)

    class _Exec:
        error = "SyntaxError: unexpected EOF"
        exit_code = 1
        raw_stderr = "Traceback..."
        raw_stdout = ""
        run_dir = None
        environment = None
        warnings = []
        pre_user_stderr = ""
        user_stderr = "SyntaxError: unexpected EOF"

    code = "broken(\n"
    with use_cwd(tmp_path):
        for _ in range(MAX_CONSECUTIVE_FAILURES):
            response: dict = {}
            _attach_status_metadata(
                response, overall_status="execution_failed",
                exec_result=_Exec(), language="Python", label="t",
                code=code, script_run_id="r1", results=[], store=store,
            )
    assert "repair_budget" in response
    assert "Stop retrying" in response["repair_budget"]["instruction"]
    assert json.dumps(response)  # must stay JSON-serializable

    with use_cwd(tmp_path):
        ok_response: dict = {}
        _attach_status_metadata(
            ok_response, overall_status="ok", exec_result=_Exec(),
            language="Python", label="t", code=code, script_run_id="r2",
            results=[], store=store,
        )
        after = record_failure(tmp_path, "something_else()")
    assert after["consecutive_failures"] == 1   # success cleared it
