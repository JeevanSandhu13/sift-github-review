"""Sift — bounded repair budget for failing scripts.

When a script fails, the model reads the error excerpt and tries
again. That loop is usually right and usually short: a typo, a wrong
column name, a missing package. But nothing bounded it. A model that
misreads an error can resubmit a near-identical script indefinitely,
and every attempt costs a subprocess run and a full turn of tokens.
The researcher watches a spinner while their grant pays for it.

This module is the circuit breaker. It tracks consecutive failures
per session and, past a threshold, tells the model plainly to stop
retrying and talk to the researcher instead.

Design decisions worth keeping:

- **Consecutive, not cumulative.** A session that fails three times,
  succeeds, then fails again is not stuck — it is working. Only an
  unbroken run of failures counts, and any success resets it.
- **Similarity-aware.** Failing three times while genuinely changing
  approach is normal debugging. Failing three times by resubmitting
  the *same* script is a loop. Repeated identical submissions trip
  the breaker sooner, because they are the case where another attempt
  provably adds nothing.
- **Advisory, not enforced.** The breaker attaches an instruction to
  the tool response; it does not refuse to execute. Hard-blocking a
  tool the model believes it needs produces worse behaviour than
  telling it clearly why it should stop, and a researcher who *wants*
  another attempt can simply ask for one. This is a cost and
  attention guard, not a security boundary — the security boundaries
  are elsewhere and are enforced.
- **Never raises.** Budget tracking must not add a failure mode to
  the failure path.

State is per-session and in-memory only. A restart clears it, which
is the right default: a researcher who reopens a session is starting
a new attempt, and a stale breaker would nag about a problem they may
already have fixed.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass, field
from pathlib import Path

# Consecutive failures before the model is told to stop and consult
# the researcher. Three allows the ordinary fix-it-twice case through
# untouched; a fourth is where a human would start asking whether the
# approach is wrong rather than the syntax.
MAX_CONSECUTIVE_FAILURES = 3

# Identical resubmissions tolerated before advising a stop. Two is
# enough to absorb a genuine transient (a flaky external process); a
# third identical attempt cannot inform anything the first two did
# not already establish.
MAX_IDENTICAL_ATTEMPTS = 2


@dataclass
class _SessionState:
    consecutive_failures: int = 0
    recent_hashes: list[str] = field(default_factory=list)


_LOCK = threading.Lock()
_STATE: dict[str, _SessionState] = {}


def _key(cwd: Path | str | None) -> str:
    return str(cwd) if cwd is not None else "<no-session>"


def _hash_code(code: str) -> str:
    """Fingerprint a script for repeat detection.

    Indentation, line breaks and runs of spaces are normalised away,
    so reindenting a broken script or adding a blank line and
    resubmitting counts as the repeat that it is.

    Deliberately NOT normalised: spacing that changes tokenisation
    (``a=1`` versus ``a = 1``). Treating those as identical would
    require language-aware tokenising of R, Stata and Python, which is
    fragile and would occasionally call two genuinely different
    scripts the same. The consecutive-failure counter is the backstop
    for edits that slip past this — a model making only cosmetic
    changes still trips that limit.
    """
    normalized = " ".join((code or "").split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def record_failure(cwd: Path | str | None, code: str) -> dict[str, object]:
    """Record a failed script run and return the current budget state.

    Returns a dict with ``consecutive_failures``, ``identical_repeats``
    and ``exhausted``. Never raises.
    """
    try:
        key = _key(cwd)
        digest = _hash_code(code)
        with _LOCK:
            state = _STATE.setdefault(key, _SessionState())
            state.consecutive_failures += 1
            state.recent_hashes.append(digest)
            # Only the recent window matters; keep it small and bounded.
            del state.recent_hashes[:-MAX_IDENTICAL_ATTEMPTS - 2]
            identical = sum(1 for h in state.recent_hashes if h == digest)
            consecutive = state.consecutive_failures
        return {
            "consecutive_failures": consecutive,
            "identical_repeats": identical,
            "exhausted": (consecutive >= MAX_CONSECUTIVE_FAILURES
                          or identical > MAX_IDENTICAL_ATTEMPTS),
        }
    except Exception:  # noqa: BLE001 — must not add a failure mode here
        return {"consecutive_failures": 0, "identical_repeats": 0,
                "exhausted": False}


def record_success(cwd: Path | str | None) -> None:
    """Clear the budget after any successful run. Never raises."""
    try:
        with _LOCK:
            _STATE.pop(_key(cwd), None)
    except Exception:  # noqa: BLE001
        return


def guidance(state: dict[str, object]) -> str | None:
    """Return model-facing instruction text, or None while under budget.

    The wording deliberately routes the model to the researcher rather
    than to another attempt, and names what to report. A model told
    only "stop" tends to apologise and retry anyway.
    """
    if not state.get("exhausted"):
        return None
    def safe_count(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            return 0
        try:
            return max(0, int(value))
        except (TypeError, ValueError, OverflowError):
            return 0

    identical = safe_count(state.get("identical_repeats", 0))
    consecutive = safe_count(state.get("consecutive_failures", 0))
    if identical > MAX_IDENTICAL_ATTEMPTS:
        cause = (f"You have submitted essentially the same script "
                 f"{identical} times and it has failed every time.")
    else:
        cause = (f"{consecutive} consecutive script runs have failed in "
                 f"this session.")
    return (
        f"{cause} Stop retrying. Another attempt at the same approach "
        f"will not produce a different result. Tell the researcher "
        f"what you were trying to do, what the error says in plain "
        f"language, and what you need from them — a package they must "
        f"install, a column that is not what you expected, a decision "
        f"about how to handle the data. Ask before running another "
        f"script."
    )


def reset_all() -> None:
    """Clear all tracked state. For tests and session teardown."""
    with _LOCK:
        _STATE.clear()
