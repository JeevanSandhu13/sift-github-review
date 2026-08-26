"""Sift — per-session token accounting and cost estimation.

Academic users run on grant money with fixed, audited budgets. "How
much did that cost?" is a question a principal investigator has to be
able to answer, and an assistant that spends real money while showing
nothing is one a careful researcher will stop using — not because the
spend is large, but because it is unaccountable.

Two quantities, kept deliberately distinct because their epistemic
status differs:

- **Tokens.** Reported by the provider for every turn. Exact, and
  they never go stale. Persisted per session.
- **Cost.** Tokens multiplied by a published price. Prices change
  without notice and vary by contract, region and tier. Any figure
  here is therefore an *estimate*, is labelled as one, and carries
  the date its rates were recorded. When no rate is known for a
  model, the cost is reported as ``None`` rather than guessed — a
  missing number is honest, a fabricated one is not.

Design notes:

- Accumulation is append-then-aggregate on a small JSON file under
  ``.sift/``, written atomically. It never raises into the turn path:
  a broken meter must not take down analysis.
- Only counts and model identifiers are recorded. No prompt text, no
  result content, nothing about the data. The meter is not another
  copy of the transcript.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sift.file_lock import exclusive_file_lock

USAGE_FILENAME = "usage.json"
USAGE_HEALTH_FILENAME = "usage.health.json"

# Published list prices in USD per million tokens, recorded on the
# date below. Deliberately a plain table rather than a network lookup:
# a privacy tool should not phone a pricing API, and a stale number
# that is labelled stale is safer than a live number that is another
# outbound connection.
#
# Keys are matched case-insensitively against a substring of the model
# identifier, so ``claude-sonnet-5-20260101`` matches ``sonnet-5``.
# Update RATES_AS_OF whenever this table changes.
RATES_AS_OF = "2026-08-21"

# Upper bounds for a single accumulated session. Far above any real
# usage (a very long session is single-digit millions of tokens), far
# below the point where integer-to-float conversion overflows. These
# exist to contain corrupt or buggy inputs, not to limit researchers.
_MAX_PLAUSIBLE_TOKENS = 10**12
_MAX_PLAUSIBLE_COST_USD = 10**7

_RATES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    # Anthropic
    # Sift enables Anthropic's one-hour prompt cache, whose write price is
    # 2x base input (not the 1.25x five-minute-cache price).
    "opus-5": {
        "input": 5.0, "output": 25.0, "cache_read": 0.50,
        "cache_creation": 10.0,
    },
    # Promotional rate through 2026-08-31. RATES_AS_OF makes this temporary
    # value explicit; update to 3/15 (cache 0.30/6.00) on 2026-09-01.
    "sonnet-5": {
        "input": 2.0, "output": 10.0, "cache_read": 0.20,
        "cache_creation": 4.0,
    },
    "fable-5": {
        "input": 10.0, "output": 50.0, "cache_read": 1.0,
        "cache_creation": 20.0,
    },
    "haiku-4-5": {"input": 0.80, "output": 4.0, "cache_read": 0.08},
    # OpenAI
    "gpt-5.6-sol": {"input": 5.0, "output": 30.0, "cache_read": 0.50},
    "gpt-5.6-terra": {"input": 2.0, "output": 12.0, "cache_read": 0.20},
    # Gemini rates. Rates
    # sourced from ``provider/catalog.py``'s own documented pricing
    # comments (the catalog is this codebase's internal source of
    # truth for published rates — see that file's GEMINI_MODELS
    # comment for the full sourcing note):
    #   - gemini-3.7-flash: $0.75/$3.75 per MTok introductory rate,
    #     in effect through 2026-12-31; becomes $1.50/$7.50 after.
    #     Using the introductory rate as of RATES_AS_OF above — this
    #     row needs updating to 1.50/7.50 once the introductory
    #     window ends.
    #   - gemini-3.1-pro-preview: $2/$12 per MTok through 200k input,
    #     $4/$18 above it. The estimator selects that tier from the
    #     provider-reported prompt-token count.
    "gemini-3.7-flash": {
        "input": 0.75, "output": 3.75, "cache_read": 0.075,
    },
    # Standard tier steps up when the prompt exceeds 200k tokens.
    "gemini-3.1-pro-preview": {
        "input": 2.0, "output": 12.0, "cache_read": 0.20,
        "input_over_200k": 4.0, "output_over_200k": 18.0,
        "cache_read_over_200k": 0.40,
    },
}


def usage_path(cwd: Path) -> Path:
    return Path(cwd) / ".sift" / USAGE_FILENAME


def _usage_lock_path(cwd: Path) -> Path:
    return usage_path(cwd).with_suffix(".json.lock")


def _usage_health_path(cwd: Path) -> Path:
    return Path(cwd) / ".sift" / USAGE_HEALTH_FILENAME


def _read_usage_checked(cwd: Path) -> tuple[dict[str, Any], bool, str]:
    path = usage_path(Path(cwd))
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, True, "no usage file"
    except OSError as error:
        return {}, False, f"usage file unreadable ({type(error).__name__})"
    try:
        state = json.loads(raw)
    except (TypeError, ValueError):
        return {}, False, "usage file is malformed"
    if not isinstance(state, dict):
        return {}, False, "usage file root is not an object"
    return state, True, "ok"


def _read_usage_health(cwd: Path) -> dict[str, Any] | None:
    try:
        marker = json.loads(_usage_health_path(cwd).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return None
    if (
        isinstance(marker, dict)
        and isinstance(marker.get("id"), str)
        and isinstance(marker.get("count"), int)
        and marker["count"] > 0
    ):
        return marker
    return None


def _note_usage_failure(cwd: Path, error: BaseException) -> None:
    """Best-effort durable evidence that a usage event was not recorded."""
    try:
        from sift.config import ensure_private_sift_dir

        ensure_private_sift_dir(cwd)
        with exclusive_file_lock(_usage_lock_path(cwd)):
            current = _read_usage_health(cwd)
            marker = {
                "v": 1,
                "id": current["id"] if current else uuid.uuid4().hex,
                "count": int(current["count"]) + 1 if current else 1,
                "last_error_type": type(error).__name__,
                "updated_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds",
                ),
            }
            _write_atomic(_usage_health_path(cwd), marker)
    except Exception:  # noqa: BLE001 - metering must not break a turn
        return


def _rates_for(model: str | None) -> dict[str, float] | None:
    """Return the rate row for a model id, or None when unknown."""
    if not model:
        return None
    needle = str(model).lower()
    # Longest key first so ``gpt-5.6-sol`` wins over a hypothetical
    # shorter prefix that also matches.
    for key in sorted(_RATES_USD_PER_MTOK, key=len, reverse=True):
        if key in needle:
            return _RATES_USD_PER_MTOK[key]
    return None


def estimate_cost_usd(
    model: str | None,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> float | None:
    """Estimate spend for one turn, or ``None`` when rates are unknown.

    Cache *creation* is billed above the input rate by both providers;
    cache *reads* are billed well below it. Treating either as plain
    input would misstate the total for a long session, which is
    exactly the workload Sift produces.
    """
    rates = _rates_for(model)
    if rates is None:
        return None

    def _clean(v: Any) -> float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return 0.0
        try:
            fval = float(v)
        except (OverflowError, ValueError):
            # Python ints are unbounded; ``float()`` on one larger
            # than ~1.8e308 raises. Such a value is implausible by
            # many orders of magnitude, so clamp rather than reject —
            # rejecting would silently zero a real (if corrupted)
            # session total.
            return float(_MAX_PLAUSIBLE_TOKENS)
        if not math.isfinite(fval) or fval <= 0:
            return 0.0
        return min(fval, _MAX_PLAUSIBLE_TOKENS)

    # State written before these bounds existed, or hand-edited, can
    # still hold implausible counts. Clamp on read as well as on
    # write so a bad file degrades to a wrong-but-finite number
    # instead of raising into the UI.
    clean_input_tokens = _clean(input_tokens)
    clean_output_tokens = _clean(output_tokens)
    clean_cache_read_tokens = _clean(cache_read_tokens)
    clean_cache_creation_tokens = _clean(cache_creation_tokens)
    long_context = clean_input_tokens > 200_000 and "input_over_200k" in rates
    input_rate = rates.get("input_over_200k", rates["input"]) if long_context else rates["input"]
    output_rate = rates.get("output_over_200k", rates["output"]) if long_context else rates["output"]
    cache_read_rate = (
        rates.get("cache_read_over_200k", rates.get("cache_read", input_rate * 0.1))
        if long_context
        else rates.get("cache_read", input_rate * 0.1)
    )
    cache_creation_rate = rates.get("cache_creation", input_rate * 1.25)
    per_tok_in = input_rate / 1_000_000
    per_tok_out = output_rate / 1_000_000
    per_tok_cache_read = cache_read_rate / 1_000_000
    per_tok_cache_write = cache_creation_rate / 1_000_000
    total = (
        clean_input_tokens * per_tok_in
        + clean_output_tokens * per_tok_out
        + clean_cache_read_tokens * per_tok_cache_read
        + clean_cache_creation_tokens * per_tok_cache_write
    )
    return round(total, 6)


def record_turn(
    cwd: Path | None,
    *,
    model: str | None,
    provider: str | None = None,
    input_tokens: int | None = 0,
    output_tokens: int | None = 0,
    cache_read_tokens: int | None = 0,
    cache_creation_tokens: int | None = 0,
    reported_cost_usd: float | None = None,
) -> None:
    """Accumulate one turn's usage. Never raises.

    ``reported_cost_usd`` is the provider's own figure when it supplies
    one (the Anthropic API path does; compatible endpoints and OpenAI do not).
    It is authoritative and is accumulated separately from the
    rate-table estimate, so the UI can show a measured number where one
    exists rather than presenting an approximation as fact.
    """
    resolved_cwd: Path | None = None
    try:
        if cwd is None:
            return
        cwd = Path(cwd)
        resolved_cwd = cwd
        if not cwd.is_dir():
            return

        def _n(v: Any) -> int:
            """Coerce a provider-reported count to a sane integer.

            Rejects booleans, non-numerics, non-finite values, and
            counts beyond ``_MAX_PLAUSIBLE_TOKENS``. The bound matters:
            an absurd value (a provider bug, a corrupted field) is
            written to disk and accumulated, so it poisons every later
            read of the session — including a multiplication that
            raises ``OverflowError`` once the running total exceeds
            what a float can hold. Dropping the implausible value
            keeps the meter usable; recording it would break it
            permanently.
            """
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return 0
            try:
                fval = float(v)
            except (OverflowError, ValueError):
                return _MAX_PLAUSIBLE_TOKENS
            if not math.isfinite(fval) or fval <= 0:
                return 0
            return int(min(fval, _MAX_PLAUSIBLE_TOKENS))

        inp, outp = _n(input_tokens), _n(output_tokens)
        cread, ccreate = _n(cache_read_tokens), _n(cache_creation_tokens)
        if not any((inp, outp, cread, ccreate)):
            return

        from sift.config import ensure_private_sift_dir

        ensure_private_sift_dir(cwd)
        with exclusive_file_lock(_usage_lock_path(cwd)):
            _accumulate_turn(
                cwd,
                model=model,
                provider=provider,
                inp=inp,
                outp=outp,
                cread=cread,
                ccreate=ccreate,
                reported_cost_usd=reported_cost_usd,
            )
    except Exception as error:  # noqa: BLE001 — never break a turn
        if resolved_cwd is not None and resolved_cwd.is_dir():
            _note_usage_failure(resolved_cwd, error)
        return


def _accumulate_turn(
    cwd: Path,
    *,
    model: str | None,
    provider: str | None,
    inp: int,
    outp: int,
    cread: int,
    ccreate: int,
    reported_cost_usd: float | None,
) -> None:
    """Read-modify-write one usage event while the caller holds the lock."""
    state, state_ok, detail = _read_usage_checked(cwd)
    if not state_ok:
        raise ValueError(detail)
    pending = _read_usage_health(cwd)
    gap_ids = [
        value for value in state.get("accounting_gap_ids", []) if isinstance(value, str)
    ]
    if pending is not None and pending["id"] not in gap_ids:
        gap_ids.append(pending["id"])
        state["unrecorded_turns"] = int(state.get("unrecorded_turns", 0)) + int(
            pending["count"]
        )
        state["accounting_gap_ids"] = gap_ids[-100:]
    state["turns"] = int(state.get("turns", 0)) + 1
    state["input_tokens"] = int(state.get("input_tokens", 0)) + inp
    state["output_tokens"] = int(state.get("output_tokens", 0)) + outp
    state["cache_read_tokens"] = int(state.get("cache_read_tokens", 0)) + cread
    state["cache_creation_tokens"] = (
        int(state.get("cache_creation_tokens", 0)) + ccreate
    )
    state["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if (
        not isinstance(reported_cost_usd, bool)
        and isinstance(reported_cost_usd, (int, float))
        and math.isfinite(float(reported_cost_usd))
        and 0 <= float(reported_cost_usd) <= _MAX_PLAUSIBLE_COST_USD
    ):
        state["reported_cost_usd"] = round(
            float(state.get("reported_cost_usd", 0.0)) + float(reported_cost_usd), 6
        )
        state["reported_cost_turns"] = int(state.get("reported_cost_turns", 0)) + 1

    # Per-model breakdown: a session that switched models has a
    # cost that cannot be derived from the totals alone.
    by_model = state.setdefault("by_model", {})
    key = str(model or "unknown")
    row = by_model.setdefault(
        key,
        {
            "turns": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "provider": provider or "",
        },
    )
    row["turns"] += 1
    row["input_tokens"] += inp
    row["output_tokens"] += outp
    row["cache_read_tokens"] += cread
    row["cache_creation_tokens"] += ccreate
    if provider:
        row["provider"] = provider

    _write_atomic(usage_path(cwd), state)
    if pending is not None:
        try:
            _usage_health_path(cwd).unlink(missing_ok=True)
        except OSError:
            pass


def read_usage(cwd: Path) -> dict[str, Any]:
    """Return the accumulated usage state (empty dict when absent)."""
    state, ok, _detail = _read_usage_checked(Path(cwd))
    return state if ok else {}


def summarize(cwd: Path) -> dict[str, Any]:
    """Return a display-ready usage summary for the session.

    ``estimated_cost_usd`` is ``None`` when no model in the session
    has a known rate. Callers must render that as "unavailable", never
    as ``$0.00`` — a zero implies free, which is a different and false
    claim.
    """
    state, accounting_ok, accounting_detail = _read_usage_checked(Path(cwd))
    pending = _read_usage_health(Path(cwd))
    if pending is None and _usage_health_path(Path(cwd)).exists():
        accounting_ok = False
        accounting_detail = "usage health marker is malformed"
    unrecorded_turns = int(state.get("unrecorded_turns", 0)) if state else 0
    acknowledged_ids = {
        value for value in state.get("accounting_gap_ids", []) if isinstance(value, str)
    }
    if pending is not None and pending["id"] not in acknowledged_ids:
        accounting_ok = False
        accounting_detail = f"{pending['count']} usage event(s) could not be recorded"
        unrecorded_turns += int(pending["count"])
    if unrecorded_turns:
        accounting_ok = False
        if accounting_detail in {"ok", "no usage file"}:
            accounting_detail = f"{unrecorded_turns} usage event(s) were not recorded"
    if not state:
        return {
            "turns": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
            "reported_cost_usd": None,
            "reported_cost_turns": 0,
            "rates_as_of": RATES_AS_OF,
            "by_model": [],
            "complete": accounting_ok,
            "pricing_complete": True,
            "usage_accounting_complete": accounting_ok,
            "usage_accounting_detail": accounting_detail,
            "unrecorded_turns": unrecorded_turns,
        }

    total_tokens = sum(
        int(state.get(k, 0))
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
        )
    )

    rows: list[dict[str, Any]] = []
    total_cost = 0.0
    any_known = False
    all_known = True
    for model, row in (state.get("by_model") or {}).items():
        cost = estimate_cost_usd(
            model,
            input_tokens=row.get("input_tokens", 0),
            output_tokens=row.get("output_tokens", 0),
            cache_read_tokens=row.get("cache_read_tokens", 0),
            cache_creation_tokens=row.get("cache_creation_tokens", 0),
        )
        if cost is None:
            all_known = False
        else:
            any_known = True
            total_cost += cost
        rows.append(
            {
                "model": model,
                "provider": row.get("provider", ""),
                "turns": int(row.get("turns", 0)),
                "tokens": sum(
                    int(row.get(k, 0))
                    for k in (
                        "input_tokens",
                        "output_tokens",
                        "cache_read_tokens",
                        "cache_creation_tokens",
                    )
                ),
                "estimated_cost_usd": cost,
            }
        )

    rows.sort(key=lambda r: -r["tokens"])
    reported = state.get("reported_cost_usd")
    reported_turns = int(state.get("reported_cost_turns", 0))
    return {
        "turns": int(state.get("turns", 0)),
        "total_tokens": total_tokens,
        # Provider-reported spend, when the provider supplies it. This
        # is measured, not modelled; the UI should prefer it and say
        # so. ``reported_cost_turns`` lets the UI tell the difference
        # between "covers the session" and "covers part of it".
        "reported_cost_usd": (
            round(float(reported), 4) if isinstance(reported, (int, float)) else None
        ),
        "reported_cost_turns": reported_turns,
        "input_tokens": int(state.get("input_tokens", 0)),
        "output_tokens": int(state.get("output_tokens", 0)),
        "cache_read_tokens": int(state.get("cache_read_tokens", 0)),
        "cache_creation_tokens": int(state.get("cache_creation_tokens", 0)),
        "estimated_cost_usd": round(total_cost, 4) if any_known else None,
        # False when at least one model had no known rate, so the UI
        # can say the estimate is partial instead of implying it covers
        # the whole session.
        "complete": all_known and accounting_ok,
        "pricing_complete": all_known,
        "usage_accounting_complete": accounting_ok,
        "usage_accounting_detail": accounting_detail,
        "unrecorded_turns": unrecorded_turns,
        "rates_as_of": RATES_AS_OF,
        "by_model": rows,
        "updated_at": state.get("updated_at"),
    }


def _write_atomic(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
