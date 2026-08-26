"""Sift — ``request_data`` handler.

The MCP ``request_data`` tool lets the selected model ask for a specific,
bounded fact about a variable—more than the schema gives, less than a script
would produce. Every response goes through SDC rules before the model
sees it.

**Design principle:** each request type has a narrow, pre-approved
output shape. The set of request types is the entire model-visible surface
beyond the schema. Requests outside the set require ``submit_script`` and
let the sanitizer there handle it.

Supported request types:

- ``categorical_levels`` — returns the list of level *names* whose
  counts meet the threshold. Low-count levels are *hidden entirely*
  (their names and counts are both suppressed), only a tally of how
  many levels were suppressed is revealed. Level names themselves
  can be disclosive (rare-disease codes, specific ethnicities).

- ``numeric_bounds`` — returns the 5th and 95th percentile of a
  numeric variable, rounded to 2 significant figures. We deliberately
  do NOT return min/max by default: those are individual observations.
  The 90%-inner percentiles blur extremes while still giving the model a
  useful sense of scale. Exception: when the researcher has listed
  the variable in the dataset's ``non_disclosive_variables`` policy
  (a per-variable opt-in for bounded, non-identifying domains — age
  in years, education years, and the like), the real min/max are
  included too, alongside the percentiles.

- ``na_count`` — returns the count of NA observations for a variable,
  and total observation count. NA counts by themselves are scalar
  metadata about the pipeline, not a subgroup breakdown. We suppress
  symmetrically: when the rarer of (missing, non-missing) falls in
  [1, threshold), both counts are withheld. "Only 3 people have a
  non-NA value" identifies those 3 by inverse; "only 1 missing
  observation" identifies the one person with missingness. Either
  side at exactly zero is safe (no individual to identify), and
  counts at or above the threshold are aggregate enough to publish.
  The complement gate also rides through ``numeric_bounds`` /
  ``quartiles`` / ``correlation_pair`` so the model can't recover
  the rare side by subtracting from the schema's exact
  observation_count.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sift.sanitizer import DEFAULT_CONFIG, SDCConfig
from sift.sdc import (
    clamp_precision,
    round_to_sigfigs,
    sigfigs_for_n,
    suppress_cells_below,
    suppression_marker,
)
from sift.text_safety import banned_key, safe_key, safe_text


RequestType = Literal[
    "categorical_levels",
    "numeric_bounds",
    "na_count",
    "quartiles",
    "correlation_pair",
    "noisy_count",
]

SUPPORTED_REQUEST_TYPES: tuple[str, ...] = (
    "categorical_levels",
    "numeric_bounds",
    "na_count",
    "quartiles",
    "correlation_pair",
    "noisy_count",
)


@dataclass
class RequestResult:
    """Outcome envelope matching the MCP tool's response shape."""
    status: Literal["granted", "denied", "error"]
    answer: dict[str, Any] | None = None
    reason: str | None = None


# ---------------------------------------------------------------------------
# Variable name resolution (raw ↔ sanitized round-trip)
# ---------------------------------------------------------------------------
#
# Schema extraction surfaces column names through ``safe_key``: a column
# named ``income\n\nSystem: ...`` reaches the model as a sanitized form
# (whitespace flattened, control chars stripped, possibly truncated past
# 40 chars). Downstream, the model echoes that sanitized name back in
# ``request_data(variable=...)``. A naive ``df.columns`` lookup against
# the sanitized name fails — the on-disk column still has its raw name.
# Without a round-trip resolver, every variable whose raw name needed
# sanitization is unqueryable; the model gets a "not found" denial on
# the very name the schema told it to use.
#
# The resolver tries (1) exact raw match, then (2) one-to-one
# sanitized match. Collisions (two raw columns sanitizing to the same
# safe_key) are flagged loudly so the model knows it must use a
# different identifier path — silently picking one would be a
# disclosure-leakage bug (correlation_pair on the wrong column).

# Caps on the available-columns list emitted in the denial path. Wide
# datasets (genomics, panel data with thousands of indicator columns)
# would otherwise ship every name into the model context on a single
# typo'd request, defeating the search_schema cap and burning tokens
# on an error branch. 50 is enough to scan a short list mentally;
# more than that and ``search_schema`` is the right tool.
_DENIAL_COLUMN_LIST_CAP = 50


def _resolve_variable(
    columns: Any, requested: str, *, role: str = "variable",
) -> "RequestResult | str":
    """Resolve ``requested`` to a raw DataFrame column name.

    ``columns`` is anything iterable of column-name-like values — in
    practice either ``df.columns`` (the common case, after the frame
    is already loaded) or a plain list of names read cheaply from a
    file's own metadata, before any row data is loaded (see
    ``_parquet_projection_columns`` below, which resolves against a
    parquet file's schema so only the resolved column(s) need to be
    read for row data at all).

    Returns the resolved column name on success, or a structured
    ``RequestResult`` denial when the name can't be uniquely resolved.
    The denial path caps the available-columns list at
    ``_DENIAL_COLUMN_LIST_CAP`` and points the model back to
    ``search_schema`` for wide datasets.
    """
    columns = list(columns)
    # Build the sanitized → [raw_names] map first, with NO fast path
    # for "exact raw match." A prior version returned ``requested``
    # immediately when ``requested in df.columns``, but that bypassed
    # the collision check: with columns ``"A B"`` and ``"A\nB"`` (both
    # sanitize to ``"A B"``), a model-issued ``request_data(variable=
    # "A B")`` would silently resolve to the raw ``"A B"`` column even
    # though the lookup is genuinely ambiguous from the model's seat
    # (it only saw the sanitized name). The model must see the same
    # collision denial regardless of whether the sanitized form
    # happens to equal a raw column name.
    safe_to_raw: dict[str, list[str]] = {}
    for col in columns:
        safe_to_raw.setdefault(safe_key(str(col)), []).append(str(col))
    matches = safe_to_raw.get(requested, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Collision: the sanitized name is ambiguous. Surface the
        # collision count rather than picking one — the model needs to
        # know to use a different path (e.g., write a script that
        # references the column by its raw bytes via a DataFrame
        # method, or rename the column upstream).
        return RequestResult(
            status="denied",
            reason=(
                f"{role} {safe_key(str(requested))!r} matches "
                f"{len(matches)} columns whose sanitized names collide. "
                f"The raw column names cannot be safely echoed back to "
                f"you (data-origin strings are an injection surface), "
                f"so this lookup is ambiguous. Rename one of the "
                f"colliding columns in the dataset, or run a script "
                f"that references the column by index instead."
            ),
        )
    # No match. Return a structured denial with a capped column
    # listing — wide datasets must not ship the full column list
    # through the error path.
    safe_requested = safe_key(str(requested))
    safe_columns = [safe_key(str(c)) for c in columns]
    total = len(safe_columns)
    truncated = total > _DENIAL_COLUMN_LIST_CAP
    listed = safe_columns[:_DENIAL_COLUMN_LIST_CAP]
    if truncated:
        suffix = (
            f" {total - _DENIAL_COLUMN_LIST_CAP} more column(s) elided. "
            f"Use search_schema(query=...) to find the right column "
            f"on a wide dataset rather than scanning the full list."
        )
    else:
        suffix = ""
    return RequestResult(
        status="denied",
        reason=(
            f"{role} {safe_requested!r} not found in dataset. "
            f"Available columns ({len(listed)} of {total}): "
            f"{listed!r}.{suffix}"
        ),
    )


# ---------------------------------------------------------------------------
# Shared complement-leak suppression
# ---------------------------------------------------------------------------
#
# Schema extraction publishes exact ``observation_count`` at every
# schema depth (names_only included). Per-variable handlers below
# return ``n_nonmissing`` / ``n_complete``. The difference is the exact
# missing count — which ``_na_count`` already coarsens when one side
# is rare, because rare missingness identifies the small side directly.
# Publishing the COMPLEMENT (n_nonmissing or n_complete) without the
# same gate reopens the rare-missingness channel: schema says
# observation_count=1000, the bounds path says n_nonmissing=999, and
# the model trivially infers missing=1, which na_count would have
# refused to disclose. The helper below applies the symmetric gate
# both surfaces consume so the gate is named once and the policy
# can't drift between call sites.


def _check_not_banned(
    resolved_column: str, config: SDCConfig, *, role: str = "variable",
) -> "RequestResult | None":
    """Return a denial if ``resolved_column`` (a REAL, Sift-resolved
    DataFrame column name — never the model's raw requested string)
    is on the dataset's banned-variables list, else ``None``.

    Checked against ``banned_key(resolved_column)`` — safe_key
    normalization PLUS case-folding — matching how ``policy.
    banned_for``/``enterprise_policy``'s ``never_expose_fields`` both
    normalize entries on load. Using bare ``safe_key`` here (as this
    function used to) would let a policy's "SSN" silently fail to
    match a real column literally named "ssn", defeating the ban with
    no error anywhere.
    """
    if not config.banned_variables:
        return None
    if banned_key(resolved_column) in config.banned_variables:
        return RequestResult(
            status="denied",
            reason=(
                f"{role} is on this dataset's banned-variables list "
                f"(set via the privacy policy) — it cannot be "
                f"referenced in a request_data call at all, regardless "
                f"of request type."
            ),
        )
    return None


def _safe_count_pair(
    nonmissing: int, total: int, threshold: int,
) -> tuple[int | None, int | str, str | None]:
    """Return ``(nonmissing_field, missing_field, note)`` after applying
    the rare-side suppression rule.

    The two values that ride through to the model are the
    nonmissing-count field (an int, ``None`` when suppressed) and the
    missing-count field (an int when safe to publish, otherwise the
    suppression marker string). Either side being rare (in [1,
    threshold)) suppresses BOTH numbers — publishing the complement
    is what reopens the rare-missingness channel ``_na_count``
    already closes. Zero on either side is safe (no individual to
    identify).
    """
    missing = total - nonmissing
    rare_side = min(nonmissing, missing)
    if rare_side == 0 or rare_side >= threshold:
        return nonmissing, missing, None
    return (
        None,
        suppression_marker(threshold),
        (
            "Exact non-missing and missing counts suppressed: one "
            f"side is below the disclosure threshold of {threshold}, "
            "so publishing either count (or its complement) would "
            "identify the rarer subgroup by inverse."
        ),
    )


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def _parquet_projection_columns(
    dataset_path: Path,
    variable: str,
    request_type: str,
    variable2: str | None,
) -> list[str] | None:
    """Best-effort projection plan for columnar Arrow-family reads.

    Returns the raw column name(s) ``load_data`` should read, resolved
    from the file's own schema — a metadata-only pyarrow call that
    touches no row data — or ``None`` when projection can't be safely
    determined. ``None`` always means "fall back to reading every
    column," which is exactly what ``handle()`` did before this
    function existed, so returning it is never a correctness risk —
    only a missed optimization. It is returned, rather than raising or
    resolving partially, whenever:

    - the file isn't Parquet, Feather, Arrow IPC, or ORC;
    - the schema itself can't be read (corrupt file, permissions —
      the ordinary full-load path below will hit the same error and
      report it the normal way);
    - a variable name doesn't resolve unambiguously against the
      schema (not found, or a sanitized-name collision).

    That last case matters most: this function calls the SAME
    ``_resolve_variable`` used for real, on the same kind of name
    list (just schema-only instead of post-load), so "projects
    successfully" and "resolves successfully against the real loaded
    frame later" can never disagree — and when resolution here fails
    for any reason, this function does NOT return a denial itself
    (that would duplicate — and risk drifting from — the real
    resolution-and-denial logic in ``handle()`` and
    ``_correlation_pair``). It simply declines to project, and the
    unchanged full-load path re-resolves and produces the exact same
    denial it always did.
    """
    suffix = dataset_path.suffix.lower()
    if suffix not in (".parquet", ".feather", ".arrow", ".ipc", ".orc"):
        return None
    try:
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            names = list(pq.ParquetFile(str(dataset_path)).schema_arrow.names)
        elif suffix == ".orc":
            import pyarrow.orc as orc

            names = list(orc.ORCFile(str(dataset_path)).schema.names)
        else:
            import pyarrow as pa
            import pyarrow.ipc as ipc

            with pa.memory_map(str(dataset_path), "r") as source:
                names = list(ipc.open_file(source).schema.names)
    except Exception:  # noqa: BLE001 — advisory only, see docstring
        return None

    resolved = _resolve_variable(names, variable)
    if isinstance(resolved, RequestResult):
        return None
    wanted = {resolved}

    if request_type == "correlation_pair" and variable2:
        resolved2 = _resolve_variable(names, variable2, role="variable2")
        if isinstance(resolved2, RequestResult):
            return None
        wanted.add(resolved2)

    return sorted(wanted)


def handle(
    dataset_path: Path,
    request_type: str,
    variable: str,
    config: SDCConfig = DEFAULT_CONFIG,
    *,
    variable2: str | None = None,
    session_root: Path | None = None,
) -> RequestResult:
    """Compute the requested fact on real data and apply SDC rules.

    Returns the sanitized answer, or a structured denial / error. Never
    raises for normal failure modes (missing variable, unsupported
    type, etc.) — those become ``status=denied`` or ``status=error``
    with a ``reason`` the caller can forward to Claude.

    ``variable2`` is consumed only by the multi-variable request types
    (``correlation_pair``); single-variable types ignore it. Passing it
    to a single-variable type is silently OK rather than rejected so a
    caller composing requests dynamically doesn't need per-type
    branching just to set the field.
    """
    if request_type not in SUPPORTED_REQUEST_TYPES:
        return RequestResult(
            status="denied",
            reason=(
                f"request_type {request_type!r} is not in the allowlist. "
                f"Supported: {sorted(SUPPORTED_REQUEST_TYPES)}"
            ),
        )

    # Columnar-file projection: for a wide file, this narrows the
    # read to only the column(s) this request can possibly need,
    # computed from schema metadata alone (see the function's
    # docstring for why this can never diverge from — or bypass — the
    # ordinary resolve-then-load-then-check flow below; it is a pure
    # "read less data" optimization, never a second source of truth).
    projection = _parquet_projection_columns(
        dataset_path, variable, request_type, variable2,
    )
    try:
        from sift.canonical_dataset import load_canonical_data
        df = load_canonical_data(
            session_root or dataset_path.parent,
            dataset_path,
            selection={"worksheet": config.excel_sheet or 0}
            if dataset_path.suffix.casefold() in {".xlsx", ".xls", ".ods"}
            else {},
            columns=projection,
        )
    except Exception as e:  # ValueError from load_data or library errors
        # Library exception bodies (pandas ParserError, pyreadstat
        # ReadstatError, etc.) frequently quote the offending row text
        # or cell value. ``safe_text`` only enforces injection bounds
        # (length, control chars), not disclosure control. Match the
        # ``get_schema`` posture (tools.py): forward the exception
        # class name only and keep the underlying message in the
        # researcher's logs.
        logging.getLogger(__name__).warning(
            "request_data load_data failed for %s: %s",
            dataset_path, e, exc_info=True,
        )
        return RequestResult(
            status="error",
            reason=(
                f"could not read dataset: {e.__class__.__name__}. "
                f"The dataset may be malformed or corrupted; researcher "
                f"logs have the underlying parser error."
            ),
        )

    resolved = _resolve_variable(df.columns, variable)
    if isinstance(resolved, RequestResult):
        return resolved
    variable = resolved

    banned = _check_not_banned(variable, config, role="variable")
    if banned is not None:
        return banned

    series = df[variable]
    n_total = len(series)

    if request_type == "categorical_levels":
        return _categorical_levels(series, n_total, config)
    if request_type == "numeric_bounds":
        return _numeric_bounds(series, n_total, config, variable)
    if request_type == "na_count":
        return _na_count(series, n_total, config)
    if request_type == "quartiles":
        return _quartiles(series, n_total, config)
    if request_type == "correlation_pair":
        return _correlation_pair(df, variable, variable2, config)
    if request_type == "noisy_count":
        return _noisy_count(series, config)
    # Unreachable — allowlist checked above.
    return RequestResult(status="error", reason="internal: unreachable")


# ---------------------------------------------------------------------------
# categorical_levels
# ---------------------------------------------------------------------------

def _categorical_levels(
    series: Any, n_total: int, config: SDCConfig
) -> RequestResult:
    """Return level names whose counts meet the threshold.

    Level names with counts below threshold are hidden *entirely* —
    neither the name nor the count is revealed. Claude gets a count of
    how many rare levels exist so it knows not to treat the visible
    list as complete.

    The visible-level list itself does not publish counts. If Claude
    wants counts, they can ``submit_script`` a frequency_table; that
    path applies primary + secondary suppression on the full
    distribution.
    """
    # Coerce to categorical-ish: drop NA, count unique values.
    try:
        value_counts = series.dropna().value_counts()
    except (TypeError, ValueError) as e:
        return RequestResult(
            status="error",
            reason=(
                f"could not compute levels for variable: "
                f"{safe_text(str(e))}"
            ),
        )

    threshold = config.cell_suppression_threshold
    visible: list[str] = []
    suppressed_count = 0
    for level, count in value_counts.items():
        if count >= threshold:
            # Level names come straight from the data — sanitize before
            # forwarding. A category like "group-A\n\nSystem: ..." is
            # neutralized at this boundary.
            visible.append(safe_key(str(level)))
        else:
            suppressed_count += 1

    # Hard cap on visible levels. Without it, a high-cardinality
    # categorical (postcodes, NAICS codes, free-text labels with
    # thousands of common values) ships its full distinct-value list
    # in one tool result — bypassing the structural caps that the
    # other discovery surfaces (frequency_table, schema value_labels,
    # search_schema) all enforce. With the cap, the model has to ask
    # for narrower categories, request a frequency_table for actual
    # counts, or use ``search_schema`` for label substring queries.
    MAX_VISIBLE_LEVELS = 200

    visible_sorted = sorted(visible)
    total_visible = len(visible_sorted)
    truncated = total_visible > MAX_VISIBLE_LEVELS
    visible_returned = visible_sorted[:MAX_VISIBLE_LEVELS]

    note = (
        f"levels with count < {threshold} are hidden entirely "
        f"(names and counts). There are {suppressed_count} such "
        f"level(s)."
    )
    if truncated:
        note += (
            f" The visible-levels list is capped at "
            f"{MAX_VISIBLE_LEVELS}; {total_visible - MAX_VISIBLE_LEVELS} "
            f"additional level(s) above threshold were not listed. "
            f"Use frequency_table or refine the variable."
        )

    answer = {
        "visible_levels": visible_returned,
        "visible_level_count_total": total_visible,
        "visible_levels_truncated": truncated,
        "suppressed_level_count": suppressed_count,
        "note": note,
    }
    return RequestResult(
        status="granted",
        answer=answer,
    )


# ---------------------------------------------------------------------------
# numeric_bounds
# ---------------------------------------------------------------------------

def _numeric_bounds(
    series: Any, n_total: int, config: SDCConfig, variable: str,
) -> RequestResult:
    """Return rounded 5th and 95th percentiles of a numeric variable.

    Rounded to 2 significant figures. We use the 5th/95th percentiles
    rather than min/max because the latter are single-observation
    values; a researcher's one high-income respondent is identifiable
    from max income alone. The 5th/95th are still individual values
    but draw from a much larger pool at scale.

    Also returns n_nonmissing so the model knows the effective sample.

    ``variable`` is the RESOLVED column name (the caller already ran
    it through ``_resolve_variable`` against the real DataFrame, so
    this is Sift's own determination of what column is being read,
    not a model-supplied label) -- when it's in
    ``config.non_disclosive_variables`` (the researcher's per-dataset
    opt-in for variables they've judged safe to expose raw: bounded,
    non-identifying domains like age in years or education years),
    the answer also carries the real min/max alongside the
    percentiles. This is the one place ``non_disclosive_variables``
    is actually enforced -- see the field's doc comment on
    ``SDCConfig`` for why the sanitizer's descriptive-payload path
    can never safely do the same for a script-emitted result.
    """
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(series):
        # series.dtype is a pandas/numpy dtype object whose repr is
        # effectively controlled (no injection risk), but sanitize for
        # defense in depth at the boundary.
        return RequestResult(
            status="denied",
            reason=(
                "numeric_bounds requires a numeric variable; "
                f"this variable has dtype {safe_key(str(series.dtype))!r}"
            ),
        )

    non_na = series.dropna()
    n_effective = int(len(non_na))
    # Tail percentiles (5th / 95th) at small N are interpolations
    # adjacent to the min and max — at N=10, the 5th percentile sits
    # between the 1st and 2nd order statistic and rounds, even at 2
    # sig figs, to a value that effectively reveals the bottom
    # outlier. We require N >= 30 so the percentile is averaged over
    # roughly 1.5 - 2.5 observations on either tail rather than
    # essentially echoing back the extremes. This is stricter than
    # cell_suppression_threshold (10) on purpose: the extra factor
    # of 3 buys real interpolation breadth.
    NUMERIC_BOUNDS_MIN_N = 30
    if n_effective < NUMERIC_BOUNDS_MIN_N:
        # Don't echo ``n_effective`` in the reason. The denial itself
        # already reveals that the non-missing N is below
        # ``NUMERIC_BOUNDS_MIN_N``; spelling out e.g. "3 non-missing
        # observations" publishes the precise small subgroup size the
        # suppression was meant to hide. The threshold is safe to
        # disclose — it's a fixed configuration constant — the actual
        # count is not. Same posture as ``_na_count`` above.
        return RequestResult(
            status="denied",
            reason=(
                f"variable has fewer than {NUMERIC_BOUNDS_MIN_N} "
                f"non-missing observations — too few for tail-percentile "
                f"bounds. At small N the 5th and 95th percentiles "
                f"interpolate close to the min and max and would "
                f"identify the tail individuals."
            ),
        )

    p5 = float(non_na.quantile(0.05))
    p95 = float(non_na.quantile(0.95))
    # Same rare-side gate ``_na_count`` applies. Without this, schema
    # publishes observation_count=N exactly, this path publishes
    # n_nonmissing=N-1 exactly, and the model trivially recovers the
    # rare missing count by subtraction — defeating the na_count gate.
    nonmissing_field, missing_field, count_note = _safe_count_pair(
        n_effective, n_total, config.cell_suppression_threshold,
    )
    answer: dict[str, Any] = {
        "percentile_5": round_to_sigfigs(p5, 2),
        "percentile_95": round_to_sigfigs(p95, 2),
        "precision": "2 significant figures",
        "n_nonmissing": nonmissing_field,
        "missing_count": missing_field,
        "note": (
            "5th and 95th percentiles are returned instead of min/max "
            "to avoid revealing extreme individual observations."
        ),
    }
    if count_note is not None:
        answer["count_note"] = count_note
    # Per-variable opt-in: the researcher's own per-dataset policy has
    # judged this specific variable's true extremes non-identifying
    # (bounded domains like age in years or education years, not
    # something like income or a rare-disease indicator). Same N-floor
    # as the percentiles above -- the opt-in is what makes exposing
    # the extremes safe, not a separate small-N argument, so no
    # additional gating is applied beyond NUMERIC_BOUNDS_MIN_N.
    # ``banned_key`` normalization (safe_key + casefold), matching
    # ``_check_not_banned`` just above and ``policy.py``'s parsing of
    # this same set -- ``variable`` is Sift's resolved real column
    # name, which can differ in case alone from how the researcher
    # typed it into the policy file ("Age" vs a column literally
    # named "age"). Comparing bare would silently defeat the opt-in
    # on nothing but a case mismatch.
    if banned_key(variable) in config.non_disclosive_variables:
        exact_min = float(non_na.min())
        exact_max = float(non_na.max())
        answer["exact_min"] = round_to_sigfigs(exact_min, 2)
        answer["exact_max"] = round_to_sigfigs(exact_max, 2)
        answer["exact_bounds_note"] = (
            f"{safe_key(variable)!r} is configured in this dataset's "
            "policy as non-disclosive, so the real min/max are "
            "included above the usual percentile-only response."
        )
    return RequestResult(status="granted", answer=answer)


# ---------------------------------------------------------------------------
# na_count
# ---------------------------------------------------------------------------

def _na_count(series: Any, n_total: int, config: SDCConfig) -> RequestResult:
    """Return the NA count for a variable.

    Symmetric suppression: BOTH ``na_count`` and ``non_na_count``
    must clear the disclosure threshold (or be exactly zero). Same
    rule the schema summary uses (``schema._suppress_rare_count``):

      - "only 3 people have a non-NA value" identifies those 3 by
        inverse (the genuinely disclosive case);
      - "only 1 missing observation" identifies the one person with
        missingness — combined with other variables, it supports
        re-identification.

    A count of zero on either side is fine — "0 missing" doesn't
    pick out any individual. The suppression rule is "0 or
    >=threshold", same as a frequency_table cell. The previous gate
    only checked the non-NA side, reopening the rare-missing
    channel that the schema path already closed.
    """
    na_count = int(series.isna().sum())
    non_na_count = n_total - na_count
    threshold = config.cell_suppression_threshold
    # Skip suppression when one side is exactly zero (no missingness, or
    # all-missing): no rare subgroup exists to identify, same handling
    # as ``schema._suppress_rare_count``. The denial reason names "the
    # rarer side" generically rather than echoing whether it's NA or
    # non-NA — that distinction is itself an inference signal.
    rare = min(na_count, non_na_count)
    if rare > 0 and rare < threshold:
        # Don't echo ``rare`` in the reason. The denial itself already
        # reveals ``rare < threshold``; spelling out e.g. "7 observations
        # on the rarer side" leaks the precise small value the
        # suppression was meant to hide and contradicts the docstring
        # above ("the genuinely disclosive case"). The threshold is
        # safe to disclose — it's a fixed configuration constant.
        return RequestResult(
            status="denied",
            reason=(
                f"the rarer of (missing, non-missing) is below the "
                f"disclosure threshold of {threshold}. The count is "
                f"suppressed because reporting it would identify the "
                f"observations on the small side by inverse."
            ),
        )
    return RequestResult(
        status="granted",
        answer={
            "na_count": na_count,
            "non_na_count": non_na_count,
            "total": n_total,
        },
    )


# ---------------------------------------------------------------------------
# noisy_count — differential privacy, opt-in only
# ---------------------------------------------------------------------------

def _noisy_count(series: Any, config: SDCConfig) -> RequestResult:
    """Return a Laplace-mechanism-noised non-NA count for a variable.

    A SEPARATE disclosure mechanism from every other request type in
    this module: instead of exact-value suppression (grant unchanged
    or deny outright), this releases a randomly perturbed value with
    a formal epsilon-differential-privacy guarantee — see
    ``differential_privacy.py`` for why a count's sensitivity (1,
    always) is what makes that guarantee honest for this primitive
    specifically, and why it isn't extended to other statistics.

    Strictly opt-in: ``config.dp_epsilon is None`` (the default —
    nothing enables this by setting ``non_disclosive_variables`` or
    ``banned_variables`` alone) denies outright, regardless of
    whether the count itself would otherwise be safe to release. A
    present-but-invalid epsilon (out of
    ``[MIN_EPSILON, MAX_EPSILON]`` — including a value corrupted by a
    hand-edited policy file) is also denied rather than clamped: the
    researcher's actual configured epsilon is what the model's
    reported guarantee refers to, so silently substituting a
    different value would misrepresent the privacy guarantee that
    was actually applied.

    Cumulative epsilon-budget enforcement across a session (basic
    composition) happens ONE LEVEL UP, in ``tools.py`` — the same
    "Sift-owned orchestration checks the ledger, this module trusts
    the epsilon it's handed for a single call" split
    ``privacy_budget.py`` uses for adaptive suppression. This
    function has no access to the release ledger or session state by
    design, matching every other handler in this module.
    """
    from sift.differential_privacy import noisy_count, validate_epsilon

    if config.dp_epsilon is None:
        return RequestResult(
            status="denied",
            reason=(
                "differential privacy is not enabled for this dataset. "
                "noisy_count requires an explicit dp_epsilon opt-in in "
                "the dataset's policy — it is never enabled by default."
            ),
        )
    err = validate_epsilon(config.dp_epsilon)
    if err is not None:
        return RequestResult(
            status="denied",
            reason=f"dataset's configured dp_epsilon is invalid: {err}",
        )
    true_count = int(series.notna().sum())
    reported, _raw = noisy_count(true_count, config.dp_epsilon)
    return RequestResult(
        status="granted",
        answer={
            "noisy_non_na_count": reported,
            "epsilon": config.dp_epsilon,
            "mechanism": "laplace",
            "privacy_unit": "row",
            "note": (
                "this count has been perturbed with calibrated random "
                "noise (row-level epsilon-differential privacy under "
                "add-or-remove-one-row adjacency) and will differ from "
                "the true value by a random amount; if one person can "
                "contribute multiple rows, this is not automatically a "
                "person-level guarantee"
            ),
        },
    )


# ---------------------------------------------------------------------------
# quartiles
# ---------------------------------------------------------------------------

def _quartiles(
    series: Any, n_total: int, config: SDCConfig,
) -> RequestResult:
    """Return rounded 25th and 75th percentiles of a numeric variable.

    Pairs with ``numeric_bounds`` (5th / 95th) to give the model an IQR-
    style sense of the distribution's middle. The 50th percentile
    (median) is deliberately NOT returned — for any odd-N variable it
    is exactly an individual observation, and the system prompt's
    forbidden-fields rule already names ``min/max/median`` as
    disclosive at the row level. Rounding to 2 sig figs is the same
    posture as ``numeric_bounds``.
    """
    import pandas as pd

    if not pd.api.types.is_numeric_dtype(series):
        return RequestResult(
            status="denied",
            reason=(
                "quartiles requires a numeric variable; "
                f"this variable has dtype {safe_key(str(series.dtype))!r}"
            ),
        )

    non_na = series.dropna()
    n_effective = int(len(non_na))
    # Same N-floor as ``numeric_bounds`` (5th / 95th). Quartiles are
    # interpolations between adjacent sorted observations: at N=10
    # the 25th percentile sits at index 2.25, i.e. ``0.75 * x[2] +
    # 0.25 * x[3]`` — a weighted blend of two specific individuals.
    # After 2-sigfig rounding the published value can still echo
    # those individuals when they're close in value, especially in
    # narrow distributions. N>=30 puts ~7-8 observations on either
    # side of each quartile, so the percentile is interpolated over
    # roughly 1.5-2.5 observations and the rounded output no longer
    # ties to a specific 2-3 of them. This matches the stricter
    # bound the tail percentiles use; quartiles are less extreme but
    # the order-statistic interpolation argument is the same.
    QUARTILES_MIN_N = 30
    if n_effective < QUARTILES_MIN_N:
        # Don't echo ``n_effective`` — see ``_numeric_bounds`` and
        # ``_na_count`` above for the same lesson. The fact of the
        # denial plus the disclosed threshold is enough for the model
        # to back off; spelling the exact small N leaks it.
        return RequestResult(
            status="denied",
            reason=(
                f"variable has fewer than {QUARTILES_MIN_N} non-missing "
                f"observations — too few to publish quartiles without "
                f"identifying individuals. At smaller N, q25 and q75 "
                f"are weighted blends of 2-3 specific sorted "
                f"observations and 2-sigfig rounding doesn't reliably "
                f"hide them."
            ),
        )

    # pandas' default linear quantile returns the EXACT sorted
    # observation at position ``p = q*(n-1)`` whenever ``p`` is an
    # integer (n=33, 37, 41, ... for q=0.25, and the same set for
    # q=0.75 since (n-1)*0.75 is integer iff (n-1)*0.25 is). After
    # 2-sigfig rounding, exact-individual values for ages, years,
    # Likert scores, and small counts pass through unchanged. The
    # N>=30 floor was meant to put 1.5-2.5 observations either side
    # of the percentile (forcing interpolation), but at integer
    # positions no interpolation happens at all. Force a midpoint
    # blend with the neighbouring sorted observation in that case so
    # every published quartile is the average of two distinct
    # observations rather than one of them verbatim.
    def _blended_quartile(values_sorted: list[float], q: float) -> float:
        n_eff = len(values_sorted)
        if n_eff == 0:
            return float("nan")
        pos = q * (n_eff - 1)
        lo = int(pos)
        hi = min(lo + 1, n_eff - 1)
        frac = pos - lo
        if frac == 0.0:
            # Position lands exactly on sorted[lo]. Average with the
            # next observation (or the previous, at the upper edge)
            # so the published quartile is never an exact single
            # observation.
            if lo + 1 < n_eff:
                return 0.5 * (values_sorted[lo] + values_sorted[lo + 1])
            if lo - 1 >= 0:
                return 0.5 * (values_sorted[lo - 1] + values_sorted[lo])
            return float(values_sorted[lo])
        return (1.0 - frac) * values_sorted[lo] + frac * values_sorted[hi]

    sorted_vals = sorted(float(v) for v in non_na.tolist())
    q25 = _blended_quartile(sorted_vals, 0.25)
    q75 = _blended_quartile(sorted_vals, 0.75)
    # Compute the published IQR by SUBTRACTING the rounded quartiles
    # rather than independently rounding ``q75 - q25``. Independently
    # rounded triples over-determine the system: comparing
    # ``rounded(q75) - rounded(q25)`` against an independently-rounded
    # IQR recovers ~1 extra bit per quartile from the rounding-error
    # disagreement. Holding ``iqr == p75 - p25`` exactly removes that
    # channel — the model now sees three numbers that are mutually
    # consistent at the published precision, with no over-determined
    # constraint to invert.
    rounded_q25 = round_to_sigfigs(q25, 2)
    rounded_q75 = round_to_sigfigs(q75, 2)
    # Same rare-side gate as ``_numeric_bounds`` / ``_na_count``.
    nonmissing_field, missing_field, count_note = _safe_count_pair(
        n_effective, n_total, config.cell_suppression_threshold,
    )
    answer: dict[str, Any] = {
        "percentile_25": rounded_q25,
        "percentile_75": rounded_q75,
        "iqr": rounded_q75 - rounded_q25,
        "precision": "2 significant figures",
        "n_nonmissing": nonmissing_field,
        "missing_count": missing_field,
        "note": (
            "25th and 75th percentiles are returned. The 50th "
            "(median) is deliberately omitted: for any odd-N "
            "variable it is exactly an individual observation, "
            "which the SDC rules forbid at the row level."
        ),
    }
    if count_note is not None:
        answer["count_note"] = count_note
    return RequestResult(status="granted", answer=answer)


# ---------------------------------------------------------------------------
# correlation_pair
# ---------------------------------------------------------------------------

def _correlation_pair(
    df: Any, var1: str, var2: str | None, config: SDCConfig,
) -> RequestResult:
    """Pearson correlation between two numeric variables.

    Multi-variable correlation matrices have their own sanitizer type
    (``correlation_matrix`` via ``submit_script``) — this fast path is
    for the common "is X correlated with Y" question that doesn't
    warrant a full script.

    Returns the correlation coefficient (rounded), the complete-case N
    (rows with both variables observed), and the missing count. The
    correlation is a pure aggregate over sums-of-products; no per-row
    leak. We still gate on the same minimum N as ``numeric_bounds`` —
    at low N a near-perfect correlation is just "the three points are
    collinear" and could imply individual coordinates.
    """
    import pandas as pd

    if not var2:
        return RequestResult(
            status="denied",
            reason=(
                "correlation_pair requires both ``variable`` (the "
                "first variable) and ``variable2`` (the second). "
                "Pass both."
            ),
        )

    resolved = _resolve_variable(df.columns, var2, role="variable2")
    if isinstance(resolved, RequestResult):
        return resolved
    var2 = resolved

    banned = _check_not_banned(var2, config, role="variable2")
    if banned is not None:
        return banned

    if var1 == var2:
        return RequestResult(
            status="denied",
            reason=(
                "correlation_pair: variable and variable2 must differ. "
                "A variable's correlation with itself is always 1; "
                "the request is structurally redundant."
            ),
        )

    s1 = df[var1]
    s2 = df[var2]
    if not pd.api.types.is_numeric_dtype(s1):
        return RequestResult(
            status="denied",
            reason=(
                f"correlation_pair: ``variable`` ({safe_key(str(var1))!r}) "
                f"has dtype {safe_key(str(s1.dtype))!r}, not numeric"
            ),
        )
    if not pd.api.types.is_numeric_dtype(s2):
        return RequestResult(
            status="denied",
            reason=(
                f"correlation_pair: ``variable2`` ({safe_key(str(var2))!r}) "
                f"has dtype {safe_key(str(s2.dtype))!r}, not numeric"
            ),
        )

    pair = pd.concat([s1, s2], axis=1).dropna()
    n_complete = int(len(pair))
    # Same N floor as ``_numeric_bounds`` and ``_quartiles`` (30). The
    # docstring above already commits to this posture ("the same
    # minimum N as numeric_bounds"); the previous literal was 10,
    # which let near-perfect r values at N=10-29 imply the
    # coordinates of individual observations exactly as the comment
    # warns. As with the other tail-statistics, the denial reason
    # doesn't echo the exact ``n_complete`` — the fact of the denial
    # plus the disclosed threshold is enough.
    CORRELATION_MIN_N = 30
    if n_complete < CORRELATION_MIN_N:
        return RequestResult(
            status="denied",
            reason=(
                f"fewer than {CORRELATION_MIN_N} rows with both "
                f"variables observed — too few to publish a "
                f"correlation without identifying individuals (a "
                f"near-perfect r at small N usually just says 'these "
                f"few points are collinear')."
            ),
        )

    # Pearson is undefined when either column has zero variance
    # (constant column, or a perfectly-imputed series). Check before
    # asking pandas to divide by the standard deviations: computing the
    # correlation first returned NaN correctly but emitted NumPy warnings
    # and performed work for a request we already know must be denied.
    import math
    zero_var: list[str] = []
    for name, series in ((var1, pair[var1]), (var2, pair[var2])):
        try:
            if float(series.std(ddof=0)) == 0.0:
                zero_var.append(safe_key(str(name)))
        except (TypeError, ValueError):
            continue
    if zero_var:
        culprits = " and ".join(repr(v) for v in zero_var)
        return RequestResult(
            status="denied",
            reason=(
                f"correlation_pair: undefined because {culprits} "
                f"{'has' if len(zero_var) == 1 else 'have'} zero "
                f"variance on the complete-case rows. A constant "
                f"column has no correlation with anything."
            ),
        )

    r = float(pair[var1].corr(pair[var2]))
    if not math.isfinite(r):
        return RequestResult(
            status="denied",
            reason=(
                "correlation_pair: result is not finite (NaN/Inf). "
                "This usually means one of the two variables has zero "
                "variance on the complete-case rows. Drop the constant "
                "column or restrict the sample."
            ),
        )

    sigfigs = sigfigs_for_n(n_complete)
    # Use the caller-supplied SDCConfig so a stricter policy (e.g.
    # ``SDCConfig(cell_suppression_threshold=25)``) takes effect on
    # this surface too. Previously this path hard-coded
    # ``DEFAULT_CONFIG.cell_suppression_threshold``, so the same
    # config that tightened ``categorical_levels`` / ``na_count``
    # silently no-op'd on correlation_pair — a stricter site policy
    # would protect only some of the discovery surfaces.
    threshold = config.cell_suppression_threshold
    total_rows = int(len(s1))
    # Coarsen rare missingness symmetrically through both n_complete
    # AND missing_count. The previous fix only coarsened
    # missing_count, but kept n_complete exact — so the model still
    # recovered the rare missing count by subtracting from schema's
    # exact observation_count. ``_safe_count_pair`` is the shared
    # gate ``_numeric_bounds`` / ``_quartiles`` / ``_na_count`` use.
    n_complete_field, missing_field, count_note = _safe_count_pair(
        n_complete, total_rows, threshold,
    )
    answer: dict[str, Any] = {
        "variable": safe_key(str(var1)),
        "variable2": safe_key(str(var2)),
        "correlation": round_to_sigfigs(r, sigfigs),
        "method": "pearson",
        "n_complete": n_complete_field,
        "missing_count": missing_field,
        "note": (
            "Pearson correlation between the two variables on rows "
            "where BOTH are observed. For a multi-variable matrix "
            "use submit_script + sift$from_correlation / "
            "sift.from_correlation."
        ),
    }
    if count_note is not None:
        answer["count_note"] = count_note
    return RequestResult(status="granted", answer=answer)
