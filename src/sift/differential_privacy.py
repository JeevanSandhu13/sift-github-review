"""Sift — differential privacy: scoped, honest, opt-in noise for counts.

**Scope, deliberately narrow.** Every other disclosure-control
mechanism in Sift (the sanitizer's threshold/suppression rules,
``data_request.py``'s min-N gates, privacy budgets/adaptive
suppression) is exact-value suppression: a fact is either released
unchanged or withheld entirely. This module adds a genuinely
different mechanism — calibrated random noise, honest epsilon-DP —
but ONLY for a single, well-understood primitive: a COUNT of records.

A count's sensitivity (how much the true value can change when one
record is added or removed) is exactly 1, always, regardless of the
data — this is what makes the textbook Laplace mechanism correct and
its epsilon guarantee HONEST here. Means, regressions, correlations,
and quantiles all have data-dependent or unbounded sensitivity (one
extreme outlier can move a mean or a regression coefficient by an
arbitrary amount) — calibrating real Laplace noise for those requires
either a hard-to-verify sensitivity bound or a separate mechanism
(e.g., clipping) this codebase does not implement. Extending DP
noise to those types without solving that honestly would be a
false privacy claim, not a stronger one — so this module doesn't
do it. See ``data_request.py``'s ``noisy_count`` request type for
where this is actually wired to a real column.

**Composition (why this module tracks spend, not just noise).** Each
``noisy_count`` call leaks a bounded amount of information governed
by its epsilon; running the SAME analysis repeatedly and averaging
the noisy answers lets an analyst recover the true count to
arbitrary precision (the noise cancels out over repeated draws) —
exactly the same combination-of-releases risk
``query_fingerprint.py`` / ``privacy_budget.py`` address for
suppression-based releases. Basic composition says the epsilons of
independent DP queries against the same dataset simply ADD; this
module tracks cumulative epsilon SPENT per dataset per session
(reading the existing release ledger, same pattern as
``privacy_budget.py``) against a fixed per-profile CAP, and callers
(``tools.py``) must refuse further ``noisy_count`` calls once the
cap would be exceeded — enforced as a Sift-owned orchestration
decision, not inside this module or the sanitizer.
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sift.release_ledger import same_dataset, verified_ledger_snapshot

# Sensitivity of a COUNT query: adding or removing one record changes
# the count by at most this much. This is the one fact that makes
# the Laplace mechanism below honest for this primitive specifically
# — it does NOT generalize to other statistics.
COUNT_SENSITIVITY = 1

# Bounds on a single query's epsilon. Both ends are meaningful, not
# arbitrary: above MAX_EPSILON the Laplace scale (sensitivity/epsilon)
# becomes small enough relative to realistic dataset counts that the
# "noisy" value is for practical purposes the true value — accepting
# it would let the tool claim a DP guarantee that provides no real
# protection. Below MIN_EPSILON the guarantee is very strong but the
# noise scale (1/epsilon >= 100) swamps any count a small-to-medium
# research dataset would realistically have, making the answer
# useless rather than private — allowing it wouldn't be dishonest,
# just pointless, so it's rejected as a likely misconfiguration
# rather than silently accepted.
MIN_EPSILON = 0.01
MAX_EPSILON = 10.0

# Each privacy profile's TOTAL epsilon budget for noisy_count calls
# against one dataset, for the whole session (basic composition: the
# sum of every granted call's epsilon against that dataset). ``None``
# means unbounded — only "public", matching
# ``policy.PRIVACY_BUDGET_BY_PROFILE``'s same reasoning. The other
# three profiles get a cap loosely calibrated so a handful of
# reasonable single-query epsilons (e.g. 5 queries at epsilon=0.2)
# fit comfortably within "regulated"'s cap, while a researcher
# spending it all on one or two queries at a much larger epsilon is
# still bounded.
EPSILON_BUDGET_BY_PROFILE: dict[str, float | None] = {
    "public": None,
    "internal": 10.0,
    "confidential": 3.0,
    "regulated": 1.0,
}


def validate_epsilon(epsilon: Any) -> str | None:
    """Return an error reason string if ``epsilon`` is not usable as
    a single query's privacy-loss parameter, or ``None`` if it's
    fine. Never raises — a non-numeric value is itself the error."""
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        return "epsilon must be a number"
    if not math.isfinite(epsilon):
        return "epsilon must be finite"
    if epsilon < MIN_EPSILON:
        return (
            f"epsilon {epsilon} is below the minimum {MIN_EPSILON} — "
            f"noise this large would make the result meaningless for "
            f"a typical research dataset"
        )
    if epsilon > MAX_EPSILON:
        return (
            f"epsilon {epsilon} exceeds the maximum {MAX_EPSILON} — "
            f"noise this small would not provide a meaningful privacy "
            f"guarantee"
        )
    return None


def laplace_sample(scale: float, rng: Any = None) -> float:
    """One draw from Laplace(0, scale) using the standard inverse-CDF
    construction from a uniform draw. ``rng`` is any object exposing
    ``.uniform(low, high)`` (a ``numpy.random.Generator`` in
    production; tests may pass a stub for determinism). A fresh
    the operating system's cryptographically secure random source is used
    when ``rng`` is ``None``.  Tests may still inject a seeded NumPy generator
    to make statistical checks reproducible; production never uses a fixed or
    user-visible seed because predictable noise defeats the mechanism.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    if rng is None:
        rng = secrets.SystemRandom()
    # u in (-0.5, 0.5); the endpoints are excluded so log(1 - 2|u|)
    # never evaluates log(0). Laplace inverse-CDF:
    #   x = -scale * sign(u) * ln(1 - 2|u|)
    u = rng.uniform(-0.5 + 1e-12, 0.5 - 1e-12)
    sign = 1.0 if u >= 0 else -1.0
    return -scale * sign * math.log(1.0 - 2.0 * abs(u))


def noisy_count(
    true_count: int, epsilon: float, rng: Any = None,
) -> tuple[int, float]:
    """Apply the Laplace mechanism to a count. Returns
    ``(reported_int, raw_noisy_float)`` — ``reported_int`` is the
    value a caller should actually disclose (rounded to the nearest
    integer, then clamped at 0 since a count can never be sensibly
    negative). The clamp is ordinary DP-safe POST-PROCESSING: it's a
    deterministic function of the already-noised value, not of the
    true count, so it adds no further information and doesn't need
    its own epsilon. ``raw_noisy_float`` is returned alongside purely
    for tests / diagnostics — callers must never surface it in place
    of the rounded, clamped value.
    """
    err = validate_epsilon(epsilon)
    if err is not None:
        raise ValueError(err)
    scale = COUNT_SENSITIVITY / epsilon
    raw = true_count + laplace_sample(scale, rng=rng)
    reported = max(0, round(raw))
    return reported, raw


@dataclass(frozen=True)
class EpsilonStatus:
    dataset: str
    privacy_profile: str
    cap: float | None
    spent: float
    accounting_ok: bool = True
    accounting_detail: str = "ok"

    @property
    def unbounded(self) -> bool:
        return self.cap is None

    @property
    def remaining(self) -> float | None:
        if self.cap is None:
            return None
        return max(0.0, self.cap - self.spent)


def epsilon_cap_for_profile(profile: str) -> float | None:
    """Session-long epsilon cap for ``profile``. An unrecognised
    profile fails closed to the strictest known cap — same posture
    as ``privacy_budget.budget_for_profile``."""
    if profile in EPSILON_BUDGET_BY_PROFILE:
        return EPSILON_BUDGET_BY_PROFILE[profile]
    return min(
        (b for b in EPSILON_BUDGET_BY_PROFILE.values() if b is not None),
        default=None,
    )


def epsilon_spent_for_dataset(
    records: list[dict[str, Any]], dataset: str, *, cwd: Path | None = None,
) -> float:
    """Sum of epsilon spent on GRANTED ``noisy_count`` calls against
    ``dataset``, from already-loaded ledger records. Only records
    with a numeric top-level ``epsilon`` fact and
    ``request_type == "noisy_count"`` count — a call that was denied
    (e.g. for being unauthorised, or itself over budget) spent
    nothing, since nothing about the true count was disclosed."""
    total = 0.0
    for rec in records:
        if rec.get("tool") != "request_data":
            continue
        args = rec.get("args") or {}
        if not same_dataset(
            args.get("dataset"), dataset, cwd=cwd,
            record_identity=args.get("dataset_identity"),
        ):
            continue
        if args.get("request_type") != "noisy_count":
            continue
        facts = rec.get("facts") or {}
        if facts.get("status") != "granted":
            continue
        eps = facts.get("epsilon")
        if isinstance(eps, (int, float)) and not isinstance(eps, bool):
            value = float(eps)
            # Corrupt/non-standard JSON can contain NaN or infinities.
            # Letting one into the sum makes every later comparison false
            # and silently disables the cap for the rest of the session.
            if math.isfinite(value) and value > 0:
                total += value
    return total


def epsilon_status_for_dataset(
    cwd: Path, dataset: str, privacy_profile: str,
) -> EpsilonStatus:
    """Full epsilon-budget status from verified accounting history.

    Unlike advisory fingerprinting, epsilon composition is a formal
    release gate.  Missing/corrupt history or a known recording gap is
    therefore surfaced as ``accounting_ok=False`` rather than silently
    becoming zero spend.  ``would_exceed_budget`` fails closed for every
    bounded profile when that flag is false.
    """
    accounting_ok = False
    accounting_detail = "privacy accounting history is unavailable"
    records, accounting_ok, accounting_detail = verified_ledger_snapshot(
        Path(cwd),
    )
    cap = epsilon_cap_for_profile(privacy_profile)
    spent = epsilon_spent_for_dataset(records, dataset, cwd=Path(cwd))
    return EpsilonStatus(dataset=dataset, privacy_profile=privacy_profile,
                         cap=cap, spent=spent,
                         accounting_ok=accounting_ok,
                         accounting_detail=accounting_detail)


def would_exceed_budget(status: EpsilonStatus, requested_epsilon: float) -> bool:
    """True iff granting a call spending ``requested_epsilon`` would
    push cumulative spend for this dataset past its profile's cap.
    Unbounded (``cap is None``) never exceeds."""
    cap = status.cap
    if cap is None:
        return False
    if not status.accounting_ok:
        return True
    return (status.spent + requested_epsilon) > cap
