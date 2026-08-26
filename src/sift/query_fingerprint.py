"""Sift — query fingerprinting: detecting repeated or modified
extraction attempts against a dataset.

**The threat model.** Every individual ``request_data`` answer and
every individual ``submit_script`` result is disclosure-controlled on
its own — that is the sanitizer's whole job, and it is stateless by
design (each call is judged only against its own payload). What the
sanitizer structurally CANNOT see is a SEQUENCE of otherwise-safe
releases that, taken together, narrow down more than any one of them
does alone. Two well-documented shapes of this:

- **Differencing / complementary disclosure**: two safe releases about
  overlapping-but-different subsets of the same population (e.g. a
  regression on the full dataset, then the same regression on a
  filtered subset) can be subtracted to isolate the excluded rows'
  contribution, even though both releases individually cleared every
  SDC threshold.
- **Combination of releases**: several different bounded facts about
  the SAME variable (a categorical breakdown, a missingness count, a
  percentile range, a correlation) each reveal little alone, but
  stacked together can narrow a variable's plausible values far more
  than the analyst who requested them may realise.

**What this module does and does not do.** It analyzes the EXISTING
release ledger (``release_ledger.py`` already records every tool call
and response fact that crosses to the model — this module adds no new
logging, it reads what is already durably, hash-chain-verified
recorded) and surfaces STRUCTURED, EXPLAINABLE findings: which
dataset/variable combinations were queried repeatedly, which
dataset/analysis-type combinations show an N-drift signature
consistent with subset differencing, and which variables have
accumulated many DIFFERENT kinds of bounded facts. It does not (and,
given what's actually recorded — aggregated facts, not full payload
content — safely cannot) prove an attack occurred; a legitimate
researcher exploring their own data trips every one of these patterns
constantly. Findings are advisory, surfaced to the researcher (via the
Privacy Inspector) and as a lightweight in-response note back to the
model when its OWN call completes a pattern — never a silent block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sift.release_ledger import dataset_key, verified_ledger_snapshot

# A dataset+variable combination queried via THIS many or more
# distinct request_data calls (regardless of request_type) is flagged
# as a repeated-extraction pattern. 3 is deliberately low — the
# finding is advisory, not a block, so erring toward surfacing more
# rather than fewer real patterns is the right trade for something a
# human reviews.
REPEATED_QUERY_THRESHOLD = 3

# A dataset+variable combination touched via this many DISTINCT
# request_types is a combination-of-releases candidate — each fact
# type reveals a different slice of the variable's distribution.
COMBINED_RELEASE_THRESHOLD = 3

# A dataset+analysis_type combination appearing with this many
# DISTINCT ``n`` values is flagged as a differencing candidate — the
# textbook signature of "same shape of analysis, different subset of
# rows" run in sequence against the same dataset.
DIFFERENCING_N_VALUES_THRESHOLD = 2
# ...but only once there are at least this many total observations of
# that (dataset, analysis_type) pair — two calls that happen to differ
# in N is barely a pattern; three or more starts to look deliberate.
DIFFERENCING_MIN_OBSERVATIONS = 3


@dataclass(frozen=True)
class RepeatedQueryFinding:
    dataset: str
    variable: str
    request_types: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class DifferencingFinding:
    dataset: str
    analysis_type: str
    distinct_n_values: tuple[float, ...]
    observation_count: int


@dataclass(frozen=True)
class CombinedReleaseFinding:
    dataset: str
    variable: str
    request_types: tuple[str, ...]


@dataclass(frozen=True)
class FingerprintReport:
    repeated_queries: tuple[RepeatedQueryFinding, ...] = ()
    differencing_candidates: tuple[DifferencingFinding, ...] = ()
    combined_release_variables: tuple[CombinedReleaseFinding, ...] = ()

    def is_empty(self) -> bool:
        return not (
            self.repeated_queries or self.differencing_candidates
            or self.combined_release_variables
        )


def _request_data_events(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract (dataset, variable, variable2, request_type) tuples for
    every ``request_data`` call in the ledger. Records with missing
    dataset/variable/request_type are skipped — best-effort, matching
    the ledger's own "never raise" posture."""
    out = []
    for rec in records:
        if rec.get("tool") != "request_data":
            continue
        args = rec.get("args") or {}
        facts = rec.get("facts")
        # Current ledgers record the response status.  Explicit denials and
        # errors disclose no result and must not contribute to cumulative
        # release findings.  A missing facts object is retained as a legacy
        # granted event so older ledgers remain analyzable.
        if isinstance(facts, dict) and facts.get("status") != "granted":
            continue
        dataset = args.get("dataset")
        variable = args.get("variable")
        request_type = args.get("request_type")
        if not (isinstance(dataset, str) and dataset
                and isinstance(variable, str) and variable
                and isinstance(request_type, str) and request_type):
            continue
        out.append({
            "dataset": dataset_key(dataset), "variable": variable,
            "variable2": args.get("variable2"),
            "request_type": request_type,
        })
    return out


def _submit_script_analysis_events(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract (source_dataset, analysis_type, n) tuples for every
    sanitizer-approved submit_script result fact recorded in the
    ledger — both the top-level facts (single-result payloads) and
    each entry of a batch's ``facts.results`` list. Rejected payloads
    remain audit records but are not releases and cannot participate
    in a differencing pattern."""
    out = []
    for rec in records:
        if rec.get("tool") != "submit_script":
            continue
        facts = rec.get("facts") or {}
        candidates = [facts] + list(facts.get("results") or [])
        for c in candidates:
            if not isinstance(c, dict):
                continue
            if c.get("status") != "ok":
                continue
            analysis_type = c.get("analysis_type")
            n = c.get("n")
            if not (isinstance(analysis_type, str) and analysis_type
                    and isinstance(n, (int, float))
                    and not isinstance(n, bool)
                    and math.isfinite(float(n)) and float(n) > 0):
                continue
            raw_sources = c.get("source_datasets")
            sources = list(raw_sources) if isinstance(raw_sources, list) else []
            if c.get("source_dataset"):
                sources.append(c["source_dataset"])
            seen: set[str] = set()
            for source in sources:
                if not isinstance(source, str) or not source:
                    continue
                normalized = dataset_key(source)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                out.append({
                    "dataset": normalized,
                    "analysis_type": analysis_type,
                    "n": n,
                })
    return out


def analyze_records(records: list[dict[str, Any]]) -> FingerprintReport:
    """Pure analysis over already-loaded ledger records. Split from
    ``analyze_ledger`` so tests can feed synthetic records without
    touching disk."""
    rd_events = _request_data_events(records)
    ss_events = _submit_script_analysis_events(records)

    # --- repeated_queries + combined_release_variables (request_data) ---
    # A call carrying ``variable2`` (currently only ``correlation_pair``)
    # discloses a joint fact about TWO variables, not one — the
    # correlation between income and region reveals something about
    # region's relationship to income just as much as the reverse.
    # Counting the release against ``variable`` alone left ``variable2``
    # invisible to every check below: a researcher (or a model acting
    # on their behalf) could extract many bivariate facts about one
    # sensitive variable by always placing it in the variable2 slot
    # against a different primary variable each time, and neither the
    # repeated-query nor combined-release detector would ever see it
    # cross a threshold. Each event now counts toward BOTH slots.
    by_var: dict[tuple[str, str], list[str]] = {}
    for e in rd_events:
        touched = [e["variable"]]
        v2 = e.get("variable2")
        if isinstance(v2, str) and v2 and v2 != e["variable"]:
            touched.append(v2)
        for v in touched:
            key = (e["dataset"], v)
            by_var.setdefault(key, []).append(e["request_type"])

    repeated: list[RepeatedQueryFinding] = []
    combined: list[CombinedReleaseFinding] = []
    for (dataset, variable), types in sorted(by_var.items()):
        if len(types) >= REPEATED_QUERY_THRESHOLD:
            repeated.append(RepeatedQueryFinding(
                dataset=dataset, variable=variable,
                request_types=tuple(types), count=len(types),
            ))
        distinct_types = tuple(sorted(set(types)))
        if len(distinct_types) >= COMBINED_RELEASE_THRESHOLD:
            combined.append(CombinedReleaseFinding(
                dataset=dataset, variable=variable,
                request_types=distinct_types,
            ))

    # --- differencing_candidates (submit_script analysis events) ---
    by_analysis: dict[tuple[str, str], list[float]] = {}
    for e in ss_events:
        key = (e["dataset"], e["analysis_type"])
        by_analysis.setdefault(key, []).append(float(e["n"]))

    differencing: list[DifferencingFinding] = []
    for (dataset, analysis_type), ns in sorted(by_analysis.items()):
        distinct_ns = tuple(sorted(set(ns)))
        if (len(ns) >= DIFFERENCING_MIN_OBSERVATIONS
                and len(distinct_ns) >= DIFFERENCING_N_VALUES_THRESHOLD):
            differencing.append(DifferencingFinding(
                dataset=dataset, analysis_type=analysis_type,
                distinct_n_values=distinct_ns,
                observation_count=len(ns),
            ))

    return FingerprintReport(
        repeated_queries=tuple(repeated),
        differencing_candidates=tuple(differencing),
        combined_release_variables=tuple(combined),
    )


def analyze_ledger(cwd: Path) -> FingerprintReport:
    """Load and analyze the release ledger for ``cwd``. Never raises —
    a missing or unreadable ledger yields an empty report, matching
    ``read_ledger``'s own "bad lines are skipped" posture."""
    records, ok, _detail = verified_ledger_snapshot(Path(cwd))
    if not ok:
        records = []
    return analyze_records(records)


def _note_for_single_variable(
    records: list[dict[str, Any]], *, dataset: str, target: str,
    request_type: str,
) -> str | None:
    """Threshold check for ONE variable name, counting prior events
    that touched it via EITHER slot (``variable`` or ``variable2``) —
    see ``analyze_records``' matching fix for why the variable2 slot
    must count. Factored out of ``note_for_new_request`` so it can be
    run once for ``variable`` and, when present, once more for
    ``variable2``, without duplicating the threshold logic."""
    key = dataset_key(dataset)
    prior = [
        e for e in _request_data_events(records)
        if e["dataset"] == key
        and (e["variable"] == target or e.get("variable2") == target)
    ]
    types_so_far = [e["request_type"] for e in prior] + [request_type]
    distinct_before = {e["request_type"] for e in prior}
    distinct_so_far = sorted(set(types_so_far))
    if len(types_so_far) == REPEATED_QUERY_THRESHOLD:
        return (
            f"privacy note: this is the {len(types_so_far)}th "
            f"request_data call touching {target!r} in this session. "
            f"Individually safe releases about the same variable can "
            f"combine to narrow its plausible values more than any "
            f"one release does alone — worth pausing to consider "
            f"whether a single script-based analysis would answer the "
            f"underlying question more directly."
        )
    # Fires exactly once: when THIS call is the one that pushes the
    # distinct-type count to the threshold, not on every subsequent
    # call that merely happens to still be sitting at that count.
    # Unlike ``types_so_far`` (the TOTAL count, which strictly
    # increases by exactly 1 every call, so an ``==`` check against it
    # is naturally a one-shot match), the DISTINCT count can plateau
    # for many calls in a row whenever a request reuses an
    # already-seen request_type -- an ``==`` check against it matched
    # on every one of those plateau calls, re-firing the identical
    # "has now been queried via 3 different types" note over and over
    # for calls 4, 5, 6, ... as long as no NEW distinct type appeared,
    # then falling silent forever once a 4th distinct type finally
    # did (found by fuzzing a realistic call sequence, not review).
    if (len(distinct_so_far) >= COMBINED_RELEASE_THRESHOLD
            and len(distinct_before) < COMBINED_RELEASE_THRESHOLD):
        return (
            f"privacy note: {target!r} has now been queried via "
            f"{len(distinct_so_far)} different request_data types "
            f"({', '.join(distinct_so_far)}) in this session."
        )
    return None


def note_for_new_request(
    cwd: Path, *, dataset: str, variable: str, request_type: str,
    variable2: str | None = None,
) -> str | None:
    """Best-effort, real-time check: if the request about to complete
    would push (dataset, variable) — or, for a two-variable request
    type like ``correlation_pair``, (dataset, variable2) — over the
    repeated-query or combined-release threshold, return a short
    advisory string to attach to THIS response — otherwise ``None``.

    Checks ``variable`` first and returns immediately if it fires;
    ``variable2`` is checked only when ``variable`` didn't already
    produce a note, so a call crossing both thresholds at once still
    surfaces exactly one advisory (the common case — a script or
    researcher exploring one relationship at a time — doesn't need
    two near-duplicate notes on the same response).

    Called with the ledger state BEFORE this call's own record_release
    (tools.py records the response only after building it), so the
    prior-call count reflects everything up to but not including the
    request currently being answered; the +1 in the message accounts
    for the current call.
    """
    records, ok, _detail = verified_ledger_snapshot(Path(cwd))
    if not ok:
        return None
    note = _note_for_single_variable(
        records, dataset=dataset, target=variable, request_type=request_type,
    )
    if note:
        return note
    if variable2 and variable2 != variable:
        return _note_for_single_variable(
            records, dataset=dataset, target=variable2,
            request_type=request_type,
        )
    return None
