"""Statistical disclosure-control and runtime-library contract.

Research scripts emit structured payloads through Sift's Python, R, or Stata
runtime helper. ``sanitize()`` is the mandatory model-facing gate. The
dataclasses, per-shape allowlists, and sanitizer registry in this module define
the accepted contract; unknown fields are dropped and hard disclosure-control
violations reject the complete payload.

Design principles:
1. **Allowlist fields, don't blocklist.** If a field isn't listed as
   allowed for a type, it's dropped. This is safer than trying to
   enumerate forbidden fields — the attacker's attack surface is the set
   of field names Sift has NOT thought about, and there are infinitely
   many such names.
2. **Hard vs soft rules.** Hard (minimum-N violations, structural size
   cap overflows) reject the whole payload. Soft (precision clamping,
   cell suppression, undeclared-key drops) transform in place and log.
3. **Transformations are logged.** Every modification the sanitizer makes
   is recorded in `SanitizerResult.transformations`, so the researcher
   can audit what the model saw versus what the script produced.
4. **Structural size caps.** Each allowed dict / list field has an
   entry-count cap on top of the per-entry character cap enforced by
   `safe_key` / `safe_text`. The two limits together bound how much
   data a prompt-injected script can smuggle through an allowed field.
   See `_OLS_MAX_PREDICTORS`, `_FREQ_MAX_CELLS`, `_XTAB_MAX_CELLS`,
   `_MAGTAB_MAX_CELLS`.

Residual risks:

1. **`predictor_variables` has no upstream data authority.** The OLS
   coefficient-dict filter uses this list as the allowlist for inner
   keys; undeclared keys are dropped with a transformation log.
   But nothing ties `predictor_variables` itself back to the source
   dataset's columns — the script declares it, and the sanitizer
   trusts it. A prompt-injected script that emits both a fake
   `predictor_variables` and matching fake coefficient keys survives
   the filter. Closing this gap would require the sanitizer to read
   the source dataset's schema, which breaks its data-isolation
   invariant. A better fix lives in the runtime library: require a
   model object (not free-form args) and derive predictor_variables
   from the model's `xlevels`.

   **Partial mitigation in place.** Every variable-name-bearing field
   (`response_variable`, `cluster_variable`, `predictor_variables[*]`,
   `variable`, `row_variable`, `col_variable`, `value_variable`,
   correlation `variables[*]`) is gated by an identifier-shape regex
   (`_NAME_IDENT_RE`) before it reaches the model. Values that survive
   `safe_text` / `safe_key` (control-char strip, whitespace flatten,
   length cap) but don't match the column-name / coefficient-name
   character class — spaces, quotes, commas, semicolons, brackets,
   braces, equals, ampersand, slashes, dollar — are replaced with the
   empty string (scalars) or filtered out of the list (list-valued
   fields). This narrows the channel from "any 120-char arbitrary text"
   to "identifier-alphabet only", which blocks the dominant raw-data
   shapes (CSV rows, JSON dumps, error-message bodies). It does not
   close the gap for adversarial column names that already match the
   identifier shape — the runtime-library fix above is still the
   right long-term answer.

2. **`sift$result(...)` generic escape hatch.** The R and Stata
   runtime libraries expose a generic constructor that lets a script
   emit any supported-type payload with hand-crafted fields. Legit
   use case: bootstraps, custom statistics. But this is the path
   that bypasses gap #1 — without it, only `sift$from_lm(model)`
   would be available, and the model object would authoritatively
   define the variable names. Removing the escape hatch is a
   research-workflow and security trade-off documented in the threat model.

3. **Data-derived names are still a channel.** Category / level names
   in frequency tables, crosstabs, and magnitude tables originate in
   the researcher's data (reading dataset values). A prompt-injected
   script can fabricate category names to encode bits — each name
   capped at 40 chars by `safe_key`, total cell count capped by the
   structural caps. Bandwidth is bounded (≈8 KB per payload) but not
   zero. To eliminate entirely, the runtime library would need to
   verify category names come from the actual data (same constraint
   as #1 — requires data access during payload construction).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypeGuard

from sift.sdc import (
    DOMINANCE_THRESHOLD_DEFAULT,
    MinimumNViolation,
    clamp_dict_by_per_key_n,
    clamp_precision,
    clamp_precision_dict,
    dominance_fails,
    enforce_back_calc_safety,
    require_minimum_n,
    sigfigs_for_n,
    suppress_cells_below,
    suppression_marker,
)
from sift.text_safety import safe_key, safe_text


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SDCConfig:
    """Thresholds and policy knobs the sanitizer consults.

    Defaults are conservative. Deployments with stricter institutional rules
    can raise the minimum-N and suppression thresholds through policy.
    """
    # Minimum total N for any regression / descriptive result. Below this,
    # the whole payload is rejected — no precision clamp makes it safe.
    min_n_regression: int = 10
    min_n_descriptive: int = 10
    # Minimum group size for t-tests (applied to both groups for
    # two-sample and to the single group for one-sample).
    min_n_ttest_group: int = 10
    # Cell-size threshold for frequency-table primary suppression.
    cell_suppression_threshold: int = 10
    # Treated-cohort size threshold for DiD event-study suppression.
    # The new SDC primitive Callaway-Sant'Anna / de Chaisemartin /
    # Sun-Abraham introduce: min-N gate on the *treated-cohort size*
    # (carried in ``n_treated_per_group``), not on the cell count of
    # the ATT panel. A balanced panel can make cell counts look
    # comfortable (4 firms × 8 quarters = 32 cells) while the actual
    # disclosure unit is the 4 firms whose entire outcome trajectories
    # are summarized by the cohort's ATT series. Cohorts below this
    # threshold are dropped *whole* — partial-cell publication would
    # leak the cohort size through which cells survived.
    min_n_did_cohort: int = 10
    # Dominance threshold for magnitude tables. If any single contributor
    # in a group accounts for more than this fraction of the cell's
    # total, the cell's value is suppressed. See sdc.py.
    dominance_threshold: float = DOMINANCE_THRESHOLD_DEFAULT
    # Per-variable opt-in for real min / max via request_data's
    # ``numeric_bounds`` (see ``data_request.py::_numeric_bounds``) --
    # NOT this sanitizer, which never accepts min/max in a descriptive
    # payload from ANY variable (see ``_DESC_ALLOWED_NUMERIC_FIELDS``
    # for why: a script-emitted payload can't be trusted to have
    # actually computed over the column it names). Default empty:
    # every variable's true extremes are withheld because they can
    # identify outlier individuals (one $1.5M salary, one rare-disease
    # respondent) -- request_data instead returns the 5th/95th
    # percentile by default. The researcher can populate this set via
    # the per-dataset policy file (``.sift/policy.json``
    # ``non_disclosive_variables``) for variables they've judged safe
    # to expose raw — typical examples: ``age`` in years,
    # ``year_of_birth``, ``education_years``.
    non_disclosive_variables: frozenset[str] = field(
        default_factory=frozenset,
    )
    # Per-dataset variable BLOCK list (``.sift/policy.json``
    # ``banned_variables``) — carried on this config purely so
    # ``data_request.py`` (the ONLY consumer) can read it from the
    # same object every other policy-derived SDC knob already rides
    # on. Deliberately NOT read or enforced anywhere in THIS module.
    # The reason is the exact one documented on
    # ``_DESC_ALLOWED_NUMERIC_FIELDS`` above for why the min/max
    # opt-in couldn't be enforced here either: this sanitizer only
    # ever sees a payload's OWN label/key strings (``variable``,
    # coefficient names, row/column keys) — fields the model and the
    # researcher's own script fully control. Nothing here can prove
    # a payload naming ``variable="income"`` genuinely computed over
    # the ``income`` column rather than a relabeled ``ssn`` column; a
    # name-based block here would filter what a COOPERATIVE script
    # calls things, not what a script actually touches, and would
    # give a false sense of enforcement. ``data_request.py`` is a
    # sound enforcement point instead: it resolves a requested name
    # to a real DataFrame column ITSELF (Sift-owned, not model-
    # supplied) before this field is ever consulted, so the check
    # there binds to the actual data.
    banned_variables: frozenset[str] = field(
        default_factory=frozenset,
    )
    # Per-dataset opt-in epsilon for the dedicated ``noisy_count``
    # request_data type (``.sift/policy.json`` ``dp_epsilon``) —
    # differential privacy is a SEPARATE, explicitly-requested
    # mechanism from the suppression-based SDC rules the rest of this
    # config governs, not a blend of the two. ``None`` (the default)
    # means DP is disabled and ``noisy_count`` is denied outright,
    # regardless of what ``non_disclosive_variables`` /
    # ``banned_variables`` say — this is a fresh, separate opt-in a
    # researcher must set explicitly. Carried here purely so
    # ``data_request.py`` (the only consumer — see
    # ``differential_privacy.py`` for the actual noise mechanism and
    # session-level epsilon-composition accounting, both Sift-owned
    # orchestration in ``tools.py``, not this module) can read it off
    # the same object every other policy-derived SDC knob rides on.
    # Deliberately NOT read anywhere else in this module: exactly the
    # same "one sound Sift-owned enforcement point, not this
    # stateless per-payload sanitizer" reasoning already documented
    # on ``banned_variables`` above applies here too.
    dp_epsilon: float | None = None
    # Researcher's saved worksheet choice for a multi-sheet ``.xlsx``
    # dataset (``.sift/policy.json`` ``excel_sheet``) — carried here
    # for exactly the same reason as ``dp_epsilon`` immediately
    # above: this module never reads it, ``data_request.py`` is the
    # sole consumer (passed straight through to
    # ``schema.load_data(..., sheet=config.excel_sheet)``), and
    # ``None`` (the default) means "first worksheet," the behaviour
    # every ``.xlsx`` reader in this codebase had before this field
    # existed. Not a disclosure-control knob at all — it selects
    # WHICH data is read, not what is withheld from an already-read
    # result — so there is no suppression semantics to document here,
    # unlike every other field in this dataclass.
    excel_sheet: str | None = None


DEFAULT_CONFIG = SDCConfig()


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class SanitizerResult:
    """Outcome of running a raw payload through the sanitizer.

    - `ok=True, sanitized={...}` on success; Claude is shown `sanitized`.
    - `ok=False, rejection_reason=str` on hard rule violation; the raw
      payload is discarded.
    - `transformations` is a list of human-readable strings describing
      every soft-rule transformation applied. The researcher's TUI shows
      these so the gap between "raw" and "what Claude saw" is auditable.
    """
    ok: bool
    analysis_type: str | None
    sanitized: dict[str, Any] | None = None
    rejection_reason: str | None = None
    transformations: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-type schemas — the runtime-library contract
# ---------------------------------------------------------------------------

# Each schema entry lists fields by disposition. "Allowed" keys pass
# through (possibly transformed by SDC); everything else is dropped.

# --- Linear regression ------------------------------------------------------

# Required fields — if missing, the payload is rejected as malformed.
# Required fields. ``r_squared`` is intentionally NOT required: the
# ``linear_regression`` bucket spans every regression-shape model the
# runtime emits via ``sift_result_regress`` / ``from_lm`` (OLS, logit,
# probit, Poisson, Cox PH, etc.), and many of those don't have an R²:
#
#   - Cox PH (``stcox``): partial-likelihood model, no R² at all.
#     Fit is reported via log-likelihood, LR chi-squared, and
#     concordance (Harrell's C, computed via ``estat concordance``).
#   - Logit / probit / Poisson (``logit``, ``probit``, ``poisson``):
#     populate ``e(r2_p)`` (McFadden pseudo-R²), not ``e(r2)``. Stata
#     helper now emits this as ``pseudo_r_squared``.
#
# The previous required-set demanded ``r_squared`` and rejected every
# Cox PH payload as malformed — researcher could fit a tenure-survival
# model but couldn't read it back. Coefficients + SEs + n + variable
# names are the structural minimum that makes a regression payload
# meaningful and verifiable; fit metrics are model-family-specific
# and pass through when present (see the allowed-numeric set below).
_OLS_REQUIRED: frozenset[str] = frozenset(
    ("type", "n", "coefficients", "standard_errors",
     "response_variable", "predictor_variables")
)

# Fields we allow through after SDC. Anything else is dropped silently
# and logged as a transformation. Notably absent: residuals, fitted,
# leverage, influence, cook_distance, data — all indexed by observation
# and therefore disclosive by construction.
_OLS_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "r_squared", "adj_r_squared", "f_statistic", "f_p_value",
    "residual_std_error",
    # Non-OLS fit metrics. All aggregate scalars derived from the
    # likelihood or design matrix; no per-observation leak. Allowed
    # so logit / probit / Poisson / Cox PH payloads carry their
    # natural fit indicators through to the model:
    #   - pseudo_r_squared: McFadden's R² for logit / probit / Poisson
    #     (``e(r2_p)`` in Stata; statsmodels ``.prsquared``).
    #   - log_likelihood: final log-likelihood. Standard for any MLE.
    #   - aic / bic: information criteria for model comparison.
    #   - chi_squared / chi_squared_p_value: LR or Wald omnibus test
    #     (``e(chi2)`` / ``e(p)`` in Stata MLE commands).
    #   - concordance: Harrell's C-index for survival models. Only
    #     populated after ``estat concordance`` (stcox doesn't put it
    #     in e() automatically).
    "pseudo_r_squared", "log_likelihood",
    "aic", "bic",
    "chi_squared", "chi_squared_p_value",
    "concordance",
    # Aggregate diagnostics. All scalars derived from the design
    # matrix or residual sum-of-squares — no per-observation leak.
    # ``condition_number`` is kappa(X), the ratio of largest to
    # smallest singular value of the design; high values flag
    # numerical instability and hidden collinearity.
    "condition_number",
    # IV / 2SLS / GMM diagnostics. All bounded scalars; none reach
    # into per-observation data. Decision made in
    # ``docs/architecture.md`` "IV as regression-bucket extension":
    # 2SLS produces a single structural coefficient table plus a
    # handful of diagnostic scalars; the *first-stage* coefficient
    # table is rarely what the model needs (it just needs to know
    # whether the instruments are strong). Composite-shape territory
    # is reserved for genuine multi-stage estimators (3SLS,
    # mediation with separate exposure→mediator and mediator→outcome
    # regressions, control-function corrections).
    #   - first_stage_f: minimum F-statistic across endogenous
    #     variables, testing joint significance of excluded
    #     instruments in the first stage. Stock-Yogo rule of thumb
    #     flags weak instruments below ~10.
    #   - weak_instrument_p: p-value associated with first_stage_f
    #     (when computed) — sometimes more useful than the F itself
    #     for non-standard sample sizes.
    #   - hansen_j / hansen_j_p: overidentification test statistic
    #     and its p-value. Hansen J under heteroskedasticity;
    #     Sargan under homoskedasticity. Only defined when the
    #     number of instruments exceeds the number of endogenous
    #     regressors (overidentified case).
    #   - endogeneity_p: Wu-Hausman / Durbin-Wu-Hausman test
    #     p-value — whether OLS and IV estimates diverge enough to
    #     justify using IV at all.
    "first_stage_f", "weak_instrument_p",
    "hansen_j", "hansen_j_p",
    "endogeneity_p",
    # Intraclass correlation — fraction of total variance attributable
    # to the random-effects group. For a one-level model:
    # icc = sigma_u² / (sigma_u² + sigma_e²). Bounded [0, 1], no
    # disclosure surface beyond what ``random_effects_variance`` already
    # carries; emitted for inference adequacy so the model can cite
    # "ρ = 0.42, school explains 42% of variance" without the
    # researcher computing it post-hoc.
    "icc",
    # Panel-data post-estimation diagnostics. All four are scalar
    # test statistics + p-values derived from the fitted residual /
    # within-transformed design — pure aggregates, no per-observation
    # leak. Researchers report them alongside coefficient tables when
    # justifying FE vs RE / homoskedasticity / serial correlation
    # assumptions; without these slots the model has to ask the
    # researcher to re-run the test rather than reading it from the
    # payload.
    #
    #   * hausman_chi2 / hausman_p — Hausman test for FE vs RE.
    #     H_0: random effects estimator is consistent. Reject → use FE.
    #     R: plm::phtest(fixed, random); Python: linearmodels'
    #     PanelOLS / RandomEffects comparison; Stata: hausman fe re.
    #   * f_test_fe_chi2 / f_test_fe_p — F-test on the joint
    #     significance of the fixed effects. Tests pooled OLS
    #     versus FE; significant → FE is needed.
    #     R: plm::pFtest(fixed, pooled); Stata: regress + xtreg, fe
    #     reports as F( N-1, N(T-1)-k ) at the bottom of the table.
    #   * breusch_pagan_chi2 / breusch_pagan_p — Breusch-Pagan LM test
    #     for random effects. Significant → RE is preferred over
    #     pooled OLS. R: plm::plmtest; Stata: xttest0.
    #   * wooldridge_ar1_chi2 / wooldridge_ar1_p — Wooldridge test
    #     for first-order serial correlation in idiosyncratic errors
    #     in panel data. Significant → cluster SEs by panel unit are
    #     mandatory. R: plm::pwartest / pbgtest; Stata: xtserial.
    "hausman_chi2", "hausman_p",
    "f_test_fe_chi2", "f_test_fe_p",
    "breusch_pagan_chi2", "breusch_pagan_p",
    "wooldridge_ar1_chi2", "wooldridge_ar1_p",
))
_OLS_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n", "degrees_of_freedom",
    # Survival-specific sample metadata. ``n`` for stcox is the number
    # of records (post-stset, can include split episodes per subject);
    # ``n_subjects`` and ``n_failures`` are what the researcher
    # actually reads off a Cox table — "324 subjects, 178 events" — so
    # they need to reach the model alongside the coefficients.
    "n_subjects", "n_failures",
    # IV: count of instruments and count of endogenous regressors.
    # Both are dimension cardinalities of the design (overidentification
    # = ``n_instruments > n_endogenous``), not data-derived quantities.
    "n_instruments", "n_endogenous",
))
_OLS_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    # ``cluster_variable`` (singular) kept as back-compat — older
    # payloads in the SQLite store carry it. New helpers emit the
    # plural list form via ``cluster_variables`` below so multi-way
    # clustering (Cameron-Gelbach-Miller two-way and beyond) round-
    # trips natively without a schema change.
    "type", "response_variable", "robust_se_type", "cluster_variable",
    # Estimation method, primarily for mixed-effects: ``REML`` or
    # ``ML``. Also useful on GLM family fits if the caller wants to
    # surface the link function or scale-estimator choice. Free-text
    # is bounded to ~40 chars via ``safe_text``; for known enum
    # values the model interprets directly, others are still safe
    # because they go through text-safety.
    "fit_method",
    # Optional multiple-testing correction metadata. The adjusted values
    # live in ``adjusted_p_values`` below; this names the procedure (e.g.
    # benjamini_hochberg, holm, bonferroni).
    "p_adjustment_method",
    # Optimizer convergence status for iterative fits — see
    # ``_OLS_VALID_CONVERGED`` for the enum and why it exists.
    "converged",
))
_OLS_ALLOWED_DICT_NUMERIC: frozenset[str] = frozenset((
    "coefficients", "standard_errors", "t_statistics", "p_values",
    "adjusted_p_values",
    # Variance-inflation factors, one per predictor. Cross-field
    # key validation (further down in _sanitize_linear_regression)
    # restricts the keys to declared predictor names + the
    # intercept aliases, so this dict can't be used to smuggle
    # arbitrary numeric channels.
    "vif",
    # Absorbed fixed-effects cardinality, one entry per FE dimension.
    # Keys are FE-variable names (dataset column names); values are
    # the count of distinct levels in that dimension. The cardinality
    # is the disclosure-relevant quantity ("firm FE absorbed, 1,247
    # levels"); the level identities themselves are NOT in this dict
    # by construction — fixest emits sizes, not labels. Excluded from
    # the coefficient-name cross-field check below because FE-var keys
    # are exactly the names NOT in predictor_variables (they're
    # absorbed, not regressors of interest).
    "fixed_effects",
    # Cluster-robust SE cardinality, one entry per clustering
    # dimension. Same shape and disclosure profile as
    # ``fixed_effects`` — keys are clustering-variable names (dataset
    # columns the model already saw in the schema), values are
    # cluster counts ("clustered at firm, 1,247 clusters"; for two-
    # way clustering, both entries are present). Decision codified
    # here per the previous turn's "modifier vs new sub-shape" rule:
    # bounded aggregate scalars and counts go in the existing OLS
    # allowlist, structured shapes get their own type. Clustering
    # cardinalities are bounded counts → allowlist, not a new shape.
    "n_clusters",
    # Mixed-effects variance components, one entry per random-effect
    # group (and one ``residual`` entry for the residual variance).
    # Keys are random-effects-factor names (dataset column names);
    # values are variance components from the random-effects
    # covariance matrix. Random-slope models contribute multiple
    # entries per group keyed like ``school.x`` (slope on x within
    # school). The intercept-slope covariance is NOT emitted in this
    # field — keep the disclosure surface to variances only; the
    # full random-effects covariance is reachable through ``vcov``
    # if a researcher genuinely needs it. Same skip-coef-key-check
    # treatment as fixed_effects — keys are NOT predictor names.
    "random_effects_variance",
    # Per-level group counts for mixed-effects models — dataset
    # column names → number of distinct groups in that level. Two-
    # level model ``y ~ x + (1|school) + (1|classroom)`` emits
    # ``{school: 30, classroom: 60}``. Same disclosure profile and
    # treatment as ``fixed_effects`` and ``n_clusters``.
    "n_groups_per_level",
))
# Dict-numeric fields whose keys are NOT coefficient names, so the
# coefficient-name cross-field validation below must skip them.
# ``fixed_effects`` keys = absorbed-FE-var names; ``n_clusters`` keys
# = clustering-var names. Both are dataset column names that are
# deliberately NOT coefficients of interest. VIF, coefficients,
# standard_errors, t_statistics, p_values all use coefficient names
# and must remain inside the cross-field check.
_OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK: frozenset[str] = frozenset((
    "fixed_effects", "n_clusters",
    # Random-effects entries are keyed by RE-factor name (possibly
    # with a ``.term`` suffix for random slopes); never by coefficient
    # name. Same exclusion as fixed_effects.
    "random_effects_variance", "n_groups_per_level",
))
# Dict-numeric fields holding integer COUNTS (cardinalities) rather
# than data-precision measurements. Skipped from the sigfigs clamp
# (1247 firms shouldn't round to 1250 at sigfigs=3) and coerced to
# int rather than left as float. Same rule for FE level counts,
# cluster counts, and mixed-effects per-level group counts.
_OLS_DICT_FIELDS_INT_COUNTS: frozenset[str] = frozenset((
    "fixed_effects", "n_clusters", "n_groups_per_level",
))
# Canonical name of the regression-bucket payload type, plus its
# legacy alias. The bucket holds OLS / logit / probit / Poisson /
# negative binomial / Cox PH / fixest / 2SLS — anything that emits a
# coefficient table with associated fit statistics. The original
# name ``linear_regression`` misled both readers and the model into
# thinking the scope was OLS-only (see the audit arc that found
# Cox hard-failing through the helper and GLMs shipping no fit
# metrics). The descriptive name ``coefficient_table_with_fit_stats``
# is the new canonical; ``linear_regression`` is kept as an alias
# so payloads from older sessions, older helpers, and the existing
# SQLite stores on researcher disks still sanitize and render.
_REGRESSION_TYPE_CANONICAL: str = "coefficient_table_with_fit_stats"
_REGRESSION_TYPE_LEGACY: str = "linear_regression"
_REGRESSION_TYPE_ALIASES: frozenset[str] = frozenset((
    _REGRESSION_TYPE_CANONICAL, _REGRESSION_TYPE_LEGACY,
))

# Exact method identities the maintained Python/R ``from_lm`` helpers can
# stamp after inspecting the fitted object. This marker is deliberately not
# part of the public regression schema; generic emitters strip it and the
# sanitizer exposes only the validated, non-underscore form downstream.
_REGRESSION_REGISTRY_METHOD_IDS: frozenset[str] = frozenset((
    "linear_regression", "logistic_regression", "probit_regression",
    "poisson_regression", "negative_binomial_regression",
    "cox_proportional_hazards", "instrumental_variables",
    "linear_mixed_effects", "generalized_mixed_effects",
))


def _emitted_regression_type(raw: dict[str, Any]) -> str:
    """Return the type string to stamp on the sanitized output for a
    regression-bucket payload.

    Round-trips the input's ``type`` field if it's one of the
    recognised aliases. Defaults to the legacy name so the value
    never appears unset — but a well-formed payload always carries
    one of the aliases here because the dispatch table only routes
    those two strings to this sanitizer.
    """
    raw_type = raw.get("type")
    if isinstance(raw_type, str) and raw_type in _REGRESSION_TYPE_ALIASES:
        return raw_type
    return _REGRESSION_TYPE_LEGACY


_OLS_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "predictor_variables",
    # IV / 2SLS supplementary identifiers. ``instrument_variables``
    # names the excluded instruments (their cardinality goes through
    # ``n_instruments``); ``endogenous_variables`` names which of
    # the predictors are treated as endogenous. Both are bounded
    # name lists — each entry goes through ``safe_key`` (40-char
    # cap, control-char stripping) — so they live in the same
    # disclosure-budget envelope as ``predictor_variables``.
    "instrument_variables", "endogenous_variables",
    # Plural cluster-variable list — multi-way clustering shows up
    # by emitting a list rather than the singular string field.
    # Helpers should emit this going forward; ``cluster_variable``
    # singular is kept for back-compat with stored payloads.
    "cluster_variables",
))


# Robust SE flavour enum. ``robust_se_type`` is already in the
# string allowlist above; this set pins it to a small, documented
# vocabulary so the model can interpret the variance estimator at a
# glance without parsing free text. Helpers auto-detect from the
# fit object (statsmodels ``cov_type``, fixest / sandwich attrs,
# Stata ``e(vcetype)``) and map onto these canonical names; values
# outside the set are dropped with a transformation note rather
# than rejecting the whole payload (researchers running niche SE
# variants get the coefficients through; only the label is
# withheld).
#
#   * ``classical``       — model-based (homoskedastic) OLS SEs.
#   * ``hc0`` / ``hc1``   — White / MacKinnon-White heteroskedastic.
#   * ``hc2`` / ``hc3``   — leverage-adjusted heteroskedastic.
#   * ``hac_newey_west``  — Newey-West HAC.
#   * ``cluster``         — Cameron-Gelbach-Miller one/two-way.
#   * ``bootstrap``       — case / wild / pairs bootstrap.
_OLS_VALID_ROBUST_SE_TYPE: frozenset[str] = frozenset((
    "classical",
    "hc0", "hc1", "hc2", "hc3",
    "hac_newey_west",
    "cluster",
    "bootstrap",
))


# Convergence-status enum. Iterative fits (logistic / GLM via IRLS,
# mixed-effects via REML/ML, Cox PH via Newton-Raphson) can report
# whether the optimizer actually converged; a non-converged fit's
# coefficients and SEs are potentially meaningless even though every
# other field in the payload looks well-formed. Same small-vocabulary
# reasoning as ``_OLS_VALID_ROBUST_SE_TYPE``: verification.py needs
# to reliably detect "did not converge" without parsing free text, so
# this is an enum, not a ``fit_method``-style free string. Helpers
# map from whatever their library reports (statsmodels
# ``.mle_retvals['converged']``, lme4's ``@optinfo$conv``, R ``glm``'s
# ``$converged``, Stata's ``e(converged)``) onto these three values.
#
#   * ``converged``               — optimizer reported success.
#   * ``not_converged``           — optimizer reported failure; the
#     fit should not be trusted without investigation.
#   * ``converged_with_warnings`` — success, but the library also
#     emitted a convergence-adjacent warning (e.g. singular fit,
#     boundary estimate, gradient near-zero but Hessian
#     ill-conditioned) worth surfacing without treating it as an
#     outright failure.
_OLS_VALID_CONVERGED: frozenset[str] = frozenset((
    "converged",
    "not_converged",
    "converged_with_warnings",
))


# --- t-test ----------------------------------------------------------------

_TTEST_REQUIRED: frozenset[str] = frozenset(
    ("type", "test_type", "n1", "mean1", "t_statistic", "p_value")
)

_TTEST_VALID_SUBTYPES: frozenset[str] = frozenset(
    ("one_sample", "two_sample", "paired", "welch")
)
_TTEST_VALID_ALTERNATIVES: frozenset[str] = frozenset((
    "two_sided", "two-sided", "two.sided", "less", "greater",
))

_TTEST_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "mean1", "mean2", "sd1", "sd2", "mean_difference",
    "t_statistic", "p_value", "degrees_of_freedom",
))
_TTEST_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n1", "n2"))
_TTEST_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "test_type", "alternative")
)
_TTEST_ALLOWED_LIST_NUMERIC: frozenset[str] = frozenset(("confidence_interval",))


# --- Descriptive statistics ------------------------------------------------

_DESC_REQUIRED: frozenset[str] = frozenset(
    ("type", "variable", "n", "mean", "sd", "missing_count")
)

# Default-allowed numerics: mean and sd — pure aggregates, never
# disclosive at row level. min / max are individual observations and
# are NEVER accepted in this payload type. The opt-in path that
# previously let an explicitly-listed variable's min/max pass through
# was unsafe under the threat model: ``source_dataset``, ``variable``,
# and the values themselves are all model/script-controlled, and the
# sanitizer cannot prove that a payload labeled ``variable="age"``
# carries age's min/max rather than (eg) salary's. A typed helper
# stamping a provenance marker doesn't close it either — the marker
# only proves the helper was called; the caller still chooses which
# column to summarize and what to label it. Closing the channel
# requires a Sift-owned dataset-load path, which is out of scope here.
# ``non_disclosive_variables`` is INERT in this sanitizer specifically
# -- it stays on ``SDCConfig`` (see the field's doc comment above)
# only because ``data_request.py`` (a genuinely Sift-owned dataset-
# load path -- it resolves the requested name against the real
# DataFrame itself, not a model-supplied label) is where the opt-in
# is actually enforced: ``numeric_bounds`` returns a variable's real
# min/max, alongside the always-present 5th/95th percentiles, exactly
# when that variable is in this set. That is the "Sift-owned
# dataset-load path" this comment used to describe as out of scope --
# it now exists, just not here. Median / quartiles remain forbidden
# too -- use ``request_data`` ``quartiles`` for those (which is
# Sift-owned and IQR-clamped).
_DESC_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset(("mean", "sd"))
_DESC_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(
    ("n", "missing_count", "distinct_count")
)
_DESC_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(("type", "variable"))


# --- Frequency table (1D only at v0) ---------------------------------------

_FREQ_REQUIRED: frozenset[str] = frozenset(
    ("type", "variable", "counts", "n", "missing_count")
)
_FREQ_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n", "missing_count"))
_FREQ_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(("type", "variable"))


# --- Text extraction (local structure from a free-text column) -------------
#
# Emitted by ``from_text_extract``. The researcher's script
# runs a LOCAL, deterministic classifier + sentiment lexicon over a
# free-text column entirely inside the sandbox -- raw text NEVER
# appears in this payload's schema, only aggregated category counts
# (reusing frequency_table's proven cell-suppression machinery
# verbatim) and a per-category mean sentiment score, itself only
# published for categories that survive suppression.
_TEXTEXTRACT_REQUIRED: frozenset[str] = frozenset(
    ("type", "text_column", "categories", "category_sentiment",
     "n", "missing_count")
)
_TEXTEXTRACT_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n", "missing_count"))
_TEXTEXTRACT_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(("type", "text_column"))
# Fewer than frequency_table's 200 -- a text-extraction taxonomy is a
# small, researcher/skill-defined category list, not an open-ended
# tabulation of raw data values. 50 is generous headroom above any
# realistic taxonomy while still bounding the injection surface.
_TEXTEXTRACT_MAX_CATEGORIES = 50
# Sentiment scores are a bounded scale, not raw data -- clamp rather
# than reject so ordinary floating-point overshoot (-1.0000000002)
# doesn't bounce an otherwise-clean payload.
_TEXTEXTRACT_SENTIMENT_MIN = -1.0
_TEXTEXTRACT_SENTIMENT_MAX = 1.0


# --- Magnitude table (sum / mean of a numeric variable by group) -----------

# Required fields. Each cell is itself a dict with `value` (the
# aggregate), `n` (group size), and `max_share` (the dominance metric
# the runtime library computed on raw values — used internally for
# suppression, NEVER emitted to Claude).
_MAGTAB_REQUIRED: frozenset[str] = frozenset(
    ("type", "row_variable", "value_variable", "aggregation", "cells")
)
_MAGTAB_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "row_variable", "value_variable", "aggregation")
)
# Aggregation kinds we understand. The runtime library should only emit
# these; anything else is rejected as a schema violation.
_MAGTAB_VALID_AGGREGATIONS: frozenset[str] = frozenset(("sum", "mean"))

# Helper-provenance marker. Typed runtime helpers (Python's
# ``from_magnitude_table``, R's ``sift$from_magnitude_table``, Stata's
# ``sift_result_magnitude``) stamp this on payloads they emit. The
# generic runtime ``result()`` API strips the field from caller-passed
# kwargs, so a script can't forge it through the public entry point.
# Required for ``magnitude_table`` because cell-level ``max_share`` is
# consulted-only and stripped: without proof the metric came from
# raw-data computation, a script could publish a dominance-violating
# value with a forged ``max_share=0`` and skip the (1, k)-dominance
# gate. The token gate alone doesn't catch this — token validation
# proves the line passed through *some* runtime path (including the
# generic ``result()`` API), not specifically the typed helper. Same
# "raise the bar, not absolute guarantee" posture as the token: a
# script that hand-writes JSON to SIFT_RESULT_PATH (after reading
# sift._RUN_TOKEN) can still forge the marker, but trivial misuse
# of ``sift.result(type="magnitude_table", cells={..., "max_share":
# 0})`` is rejected.
_HELPER_PROVENANCE_FIELD = "_via_helper"
_MAGTAB_HELPER_VALUE = "from_magnitude_table"


# --- Structural size caps --------------------------------------------------
# Hard limits on the number of top-level entries each payload type can
# carry. Primarily a defense-in-depth measure: the per-entry key cap
# (40 chars via safe_key) plus these entry-count caps bound how much
# data a prompt-injected script can smuggle through an allowed field.
#
# Numbers picked to comfortably accommodate legitimate research
# output and reject anything that looks engineered — a regression
# with 60 predictors isn't interpretable statistics, a frequency
# table with 300 distinct levels isn't a useful summary.
_OLS_MAX_PREDICTORS = 50
_FREQ_MAX_CELLS = 200
_XTAB_MAX_CELLS = 2500          # allows up to ~50 × 50
_MAGTAB_MAX_CELLS = 200
_CORR_MAX_VARIABLES = 30        # NxN ⇒ up to 900 entries before clamping


# --- Correlation matrix ----------------------------------------------------

# Pairwise correlations among a list of numeric variables. Pure
# aggregate (sums of products / N), no per-row leak — but reject at
# low N where a near-perfect correlation is just "the three points
# are collinear" rather than a population property.
_CORR_REQUIRED: frozenset[str] = frozenset(
    ("type", "n", "variables", "correlations")
)
_CORR_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("n", "missing_count"))
_CORR_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "method", "label")
)
_CORR_ALLOWED_LIST_STRING: frozenset[str] = frozenset(("variables",))
# Allowed correlation types — Pearson is the linear default; Spearman /
# Kendall handle rank-based and ordinal data. Anything else is
# rejected as a schema violation rather than silently coerced.
_CORR_VALID_METHODS: frozenset[str] = frozenset(
    ("pearson", "spearman", "kendall")
)


# --- Crosstab (2D frequency table, no margins emitted) ---------------------

# Required fields. Note the deliberate absence of `n` — crosstabs do NOT
# expose a grand total at v0. Without margins (row totals, column totals,
# grand total), primary cell suppression alone is sufficient to prevent
# back-calculation; with margins, 2D requires LP-based secondary
# suppression (τ-ARGUS territory, deferred).
_XTAB_REQUIRED: frozenset[str] = frozenset(
    ("type", "row_variable", "col_variable", "counts")
)
# Margin-ish fields forbidden by name. `_collect_allowed` drops anything
# not on the allowlist, but naming these here is the documentation
# anchor: these are the fields that break the "no margins" invariant.
_XTAB_FORBIDDEN_MARGIN_FIELDS: frozenset[str] = frozenset((
    "n", "grand_total", "row_totals", "column_totals", "col_totals",
    "marginals",
))
_XTAB_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset(
    ("type", "row_variable", "col_variable")
)
# `missing_count` is kept optional — it's not a disclosive margin (it
# refers to observations with a missing value on one or both axes, which
# is a pipeline characteristic, not a cell-identifying quantity). The
# counts themselves are handled specially below.
_XTAB_ALLOWED_INT_FIELDS: frozenset[str] = frozenset(("missing_count",))


# ---------------------------------------------------------------------------
# Top-level dispatch
# ---------------------------------------------------------------------------

def sanitize(raw: dict[str, Any], config: SDCConfig = DEFAULT_CONFIG) -> SanitizerResult:
    """Validate a raw analysis payload and apply SDC rules.

    Dispatches on `raw["type"]`. Unknown types are rejected as malformed.
    All rejections carry a machine-readable reason; all transformations
    are logged.
    """
    if not isinstance(raw, dict):
        return SanitizerResult(
            ok=False, analysis_type=None,
            rejection_reason=f"payload must be a dict, got {type(raw).__name__}",
        )

    analysis_type = raw.get("type")
    if not isinstance(analysis_type, str):
        return SanitizerResult(
            ok=False, analysis_type=None,
            rejection_reason="payload missing 'type' field or type is not a string",
        )

    handler = _HANDLERS.get(analysis_type)
    if handler is None:
        # The script controls ``raw["type"]``; an adversarial payload
        # could set it to a raw cell value (a row, a cell, a JSON
        # blob) and trigger this branch to smuggle the value out
        # through ``rejection_reason``, which submit_script forwards
        # into both the inline result and the persisted diagnostic
        # row. Echo the type only after ``safe_key`` (40-char cap,
        # control-char strip) so the leak channel is bounded to a
        # short token, and store the same sanitized form in
        # ``analysis_type`` so the diagnostic row never carries the
        # raw value either.
        safe_type = safe_key(analysis_type)
        return SanitizerResult(
            ok=False, analysis_type=safe_type,
            rejection_reason=(
                f"unknown analysis type {safe_type!r}. Supported in v0: "
                f"{sorted(_HANDLERS.keys())}"
            ),
        )
    result = handler(raw, config)
    if result.ok and result.sanitized is not None:
        invalid = _invalid_statistical_range(result.sanitized)
        if invalid is not None:
            return SanitizerResult(
                ok=False,
                analysis_type=result.analysis_type,
                rejection_reason=(
                    f"sanitized result contains a mathematically invalid "
                    f"value in {invalid!r}; bounded statistics must be "
                    "finite and fall within their defined range"
                ),
            )
        inconsistent = _invalid_cross_field_invariant(result.sanitized)
        if inconsistent is not None:
            return SanitizerResult(
                ok=False,
                analysis_type=result.analysis_type,
                rejection_reason=(
                    "sanitized result is internally inconsistent in "
                    f"{inconsistent!r}; related estimates, intervals, "
                    "counts, and ordered statistics must agree"
                ),
            )
    return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _invalid_statistical_range(payload: dict[str, Any]) -> str | None:
    """Return the first known bounded field containing an invalid value.

    This runs after every shape-specific sanitizer, making the invariant
    uniform across regressions, t-tests, DiD, RDD, survival, factor analysis,
    clustering, and future result types. The walk is recursive because several
    result shapes keep probability or bounded statistics in nested mappings.
    Field names are schema-owned, so the rejection message cannot echo a
    caller-controlled value.
    """
    unit_interval_fields = {
        "r_squared",
        "concordance",
        "icc",
        "explained_variance_ratio",
        "cumulative_variance",
        "ss_ratio",
        "kmo",
    }
    signed_unit_fields = {
        "correlations",
        "silhouette_score",
        "silhouette_per_cluster",
    }
    nonnegative_fields = {
        "standard_errors",
        "f_statistic",
        "chi_square",
        "kmo",
        "explained_variance",
        "eigenvalues",
        "total_within_ss",
        "between_cluster_ss",
        "total_ss",
        "within_cluster_ss",
        "inertia",
        "calinski_harabasz_score",
        "davies_bouldin_score",
        "cut_height",
        "variance_components",
        "random_effects_variance",
        "f_statistic_per_variable",
        "residual_std_error",
        "rmsea",
        "median_survival_time",
        "median_survival_ci_lower",
        "median_survival_ci_upper",
        "bandwidth_left",
        "bandwidth_right",
        "bandwidth_bias_correction_left",
        "bandwidth_bias_correction_right",
        "first_stage_f",
        "degrees_of_freedom",
    }

    def leaves(value: Any):
        if isinstance(value, dict):
            for child in value.values():
                yield from leaves(child)
        elif isinstance(value, list):
            for child in value:
                yield from leaves(child)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            yield float(value)

    def bounds_for(
        field_name: str,
    ) -> tuple[float | None, float | None] | None:
        is_probability = (
            field_name in {"p", "p_value", "p_values", "adjusted_p_values"}
            or field_name.endswith(("_p", "_p_value"))
            or field_name.startswith("survival_at_")
        )
        if is_probability or field_name in unit_interval_fields:
            return (0.0, 1.0)
        if field_name in signed_unit_fields:
            return (-1.0, 1.0)
        if field_name == "adj_r_squared":
            return (None, 1.0)
        if (
            field_name in nonnegative_fields
            or field_name == "n"
            or field_name.startswith(("n_", "se_"))
            or field_name == "se"
            or field_name.endswith(("_se", "_chi2", "_chi_squared"))
        ):
            return (0.0, None)
        return None

    def visit(value: Any) -> str | None:
        if isinstance(value, dict):
            for field_name, child in value.items():
                if isinstance(field_name, str):
                    bounds = bounds_for(field_name)
                    if bounds is not None:
                        lower, upper = bounds
                        for number in leaves(child):
                            if (
                                not math.isfinite(number)
                                or (lower is not None and number < lower)
                                or (upper is not None and number > upper)
                            ):
                                return field_name
                invalid = visit(child)
                if invalid is not None:
                    return invalid
        elif isinstance(value, list):
            for child in value:
                invalid = visit(child)
                if invalid is not None:
                    return invalid
        return None

    return visit(payload)


def _invalid_cross_field_invariant(payload: dict[str, Any]) -> str | None:
    """Return a schema-owned label for the first impossible relationship.

    Per-field type/range checks cannot detect an interval whose lower bound is
    above its upper bound, a point estimate outside its own interval, or a
    survival curve that rises at a later horizon.  Those payloads are finite
    and individually in range, but they cannot describe the claimed result.
    Keep this pass centralized so every current and future helper language is
    held to the same contract after privacy transformations and rounding.
    """

    def number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    def ordered(lower: Any, upper: Any) -> bool:
        lo, hi = number(lower), number(upper)
        return lo is None or hi is None or lo <= hi

    # Standard two-element interval (currently t_test).
    ci = payload.get("confidence_interval")
    if isinstance(ci, list) and len(ci) == 2 and not ordered(ci[0], ci[1]):
        return "confidence_interval"
    if isinstance(ci, list) and len(ci) == 2:
        estimate = number(payload.get("mean_difference"))
        lo, hi = number(ci[0]), number(ci[1])
        if estimate is not None and lo is not None and hi is not None \
                and not lo <= estimate <= hi:
            return "mean_difference/confidence_interval"

    # Top-level ``prefix_ci_lower`` / ``prefix_ci_upper`` pairs, including
    # KM horizons and aggregate DiD intervals.
    for key, lower in payload.items():
        if not isinstance(key, str):
            continue
        if key.endswith("_ci_lower"):
            upper_key = key[:-len("_ci_lower")] + "_ci_upper"
        elif key.startswith("ci_lower_"):
            upper_key = "ci_upper_" + key[len("ci_lower_"):]
        else:
            continue
        if upper_key in payload and not ordered(lower, payload[upper_key]):
            return f"{key}/{upper_key}"

    # Flat per-estimand interval maps (marginal effects).
    lower_map = payload.get("ci_lower")
    upper_map = payload.get("ci_upper")
    estimate_map = payload.get("effects")
    if isinstance(lower_map, dict) and isinstance(upper_map, dict):
        for name in lower_map.keys() & upper_map.keys():
            lo, hi = number(lower_map[name]), number(upper_map[name])
            if lo is not None and hi is not None and lo > hi:
                return "ci_lower/ci_upper"
            if isinstance(estimate_map, dict) and name in estimate_map:
                est = number(estimate_map[name])
                if est is not None and lo is not None and hi is not None \
                        and not lo <= est <= hi:
                    return "effects/ci_lower/ci_upper"

    # Nested cohort -> event-time intervals (DiD event studies).
    if isinstance(lower_map, dict) and isinstance(upper_map, dict):
        for group in lower_map.keys() & upper_map.keys():
            lowers, uppers = lower_map[group], upper_map[group]
            if not isinstance(lowers, dict) or not isinstance(uppers, dict):
                continue
            estimates = payload.get("att", {}).get(group, {}) \
                if isinstance(payload.get("att"), dict) else {}
            for event_time in lowers.keys() & uppers.keys():
                lo, hi = number(lowers[event_time]), number(uppers[event_time])
                if lo is not None and hi is not None and lo > hi:
                    return "did_event_study confidence intervals"
                if isinstance(estimates, dict) and event_time in estimates:
                    est = number(estimates[event_time])
                    if est is not None and lo is not None and hi is not None \
                            and not lo <= est <= hi:
                        return "did_event_study estimate/interval"

    ptype = payload.get("type")
    if ptype == "kaplan_meier":
        horizons = ("1y", "3y", "5y", "10y")
        survival = [
            number(payload.get(f"survival_at_{h}")) for h in horizons
        ]
        observed = [value for value in survival if value is not None]
        if any(observed[i + 1] > observed[i]
               for i in range(len(observed) - 1)):
            return "survival_at_* monotonicity"
        n_subjects = number(payload.get("n_subjects"))
        for horizon in horizons:
            at_risk = number(payload.get(f"n_at_risk_{horizon}"))
            if at_risk is not None and n_subjects is not None \
                    and at_risk > n_subjects:
                return f"n_at_risk_{horizon}/n_subjects"
            estimate = number(payload.get(f"survival_at_{horizon}"))
            lo = number(payload.get(f"survival_at_{horizon}_ci_lower"))
            hi = number(payload.get(f"survival_at_{horizon}_ci_upper"))
            if estimate is not None and lo is not None and hi is not None \
                    and not lo <= estimate <= hi:
                return f"survival_at_{horizon} confidence interval"
        median = number(payload.get("median_survival_time"))
        median_lo = number(payload.get("median_survival_ci_lower"))
        median_hi = number(payload.get("median_survival_ci_upper"))
        if median is not None and median_lo is not None and median_hi is not None \
                and not median_lo <= median <= median_hi:
            return "median_survival_time confidence interval"

    if ptype == "rdd":
        for flavor in ("conventional", "bias_corrected", "robust"):
            estimate = number(payload.get(f"tau_{flavor}"))
            lo = number(payload.get(f"ci_lower_{flavor}"))
            hi = number(payload.get(f"ci_upper_{flavor}"))
            if estimate is not None and lo is not None and hi is not None \
                    and not lo <= estimate <= hi:
                return f"tau_{flavor} confidence interval"

    if ptype == "did_event_study":
        estimate = number(payload.get("aggregate_att"))
        lo = number(payload.get("aggregate_ci_lower"))
        hi = number(payload.get("aggregate_ci_upper"))
        if estimate is not None and lo is not None and hi is not None \
                and not lo <= estimate <= hi:
            return "aggregate_att confidence interval"

    return None

def _require_fields(
    raw: dict[str, Any], required: frozenset[str], analysis_type: str
) -> str | None:
    """Return a rejection reason if required fields are missing, else None."""
    missing = required - raw.keys()
    if missing:
        return (
            f"{analysis_type} payload missing required fields: "
            f"{sorted(missing)}"
        )
    return None


def _require_after_filter(
    out: dict[str, Any],
    required: frozenset[str],
    analysis_type: str,
    *,
    pre_validated: frozenset[str] = frozenset(),
) -> str | None:
    """Re-check required fields after ``_collect_allowed`` runs.

    ``_require_fields`` only checks that required keys are present in
    the raw payload; ``_collect_allowed`` then DROPS any field whose
    type doesn't match the schema (e.g. ``coefficients`` shipped as a
    string). The result is an ``ok=True`` payload missing structural
    fields — the model thinks the analysis succeeded with garbage. So
    callers re-check required fields against ``out`` after collection.

    ``pre_validated`` lists fields the caller already gates explicitly
    BEFORE ``_collect_allowed`` (e.g. ``n`` for OLS, ``cells`` for
    magnitude_table) — those are never inserted into ``out`` by
    ``_collect_allowed`` itself, so excluding them avoids spurious
    rejection. The handler is expected to assemble those fields
    elsewhere in ``out`` if they survive their pre-validation.
    """
    needs_check = required - pre_validated - {"type"}
    missing = needs_check - out.keys()
    if missing:
        return (
            f"{analysis_type} payload required field(s) had wrong "
            f"type and were dropped during sanitization: "
            f"{sorted(missing)}"
        )
    return None


def _is_finite_number(x: Any) -> TypeGuard[int | float]:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# Hard cap on transformation-log entries a single ``_collect_allowed``
# invocation may emit. The transformations list rides into the
# tool_result payload and is read by the model, so any loop that
# appends one entry per dropped key is a model-visible channel whose
# size is controlled by the payload author. Two such loops exist in
# this function: the outer ``for k in raw.items()`` (one entry per
# unknown field) and the inner ``for kk in v.items()`` inside the
# ``dict_numeric`` branch (one entry per malformed nested key).
# Without a cap, a payload with thousands of arbitrary keys would
# fill the model's context with megabytes of "dropped …" lines.
# 50 is enough that realistic shape-mismatch debugging is still
# legible; the trailing summary tells the model how many drops it
# isn't seeing in detail.
_COLLECT_ALLOWED_LOG_CAP = 50


def _collect_allowed(
    raw: dict[str, Any],
    *,
    numeric: frozenset[str] = frozenset(),
    integer: frozenset[str] = frozenset(),
    string: frozenset[str] = frozenset(),
    dict_numeric: frozenset[str] = frozenset(),
    list_string: frozenset[str] = frozenset(),
    list_numeric: frozenset[str] = frozenset(),
    transformations: list[str] | None = None,
) -> dict[str, Any]:
    """Filter a raw payload to just the allowed fields, validating shapes.

    Fields with wrong types are dropped with a log entry (not an error)
    — we treat malformed values as indistinguishable from untrusted
    input. Unknown fields are dropped silently (they're explicitly not
    in the allowlist, so logging every unknown field would be noisy).

    ``transformations`` is appended to in-place if passed. Field names
    and dict keys that originate in the researcher's data are passed
    through ``safe_key`` before being interpolated into log messages or
    returned as output keys — otherwise a maliciously-named variable
    could inject text into Claude's context through the transformations
    log or through a coefficient dict key.

    Per-invocation cap: at most ``_COLLECT_ALLOWED_LOG_CAP`` drop
    entries land in ``transformations``; surplus drops are summarised
    in a single tail line. See ``_COLLECT_ALLOWED_LOG_CAP`` for why.
    """
    t = transformations if transformations is not None else []
    out: dict[str, Any] = {}
    allowed = numeric | integer | string | dict_numeric | list_string | list_numeric

    # Track how many entries this call has emitted, separately from
    # ``len(t)`` — the caller may have prefilled ``t`` with notes from
    # earlier sanitiser stages, and we only want to bound THIS call's
    # contribution. ``surplus`` is appended once at the end as a
    # human-readable tail.
    emitted = 0
    surplus = 0

    def _log(msg: str) -> None:
        nonlocal emitted, surplus
        if emitted < _COLLECT_ALLOWED_LOG_CAP:
            t.append(msg)
            emitted += 1
        else:
            surplus += 1

    # Count rather than name unknown fields. Field names in ``raw``
    # but outside ``allowed`` are data-derived (an attacker-authored
    # script can encode raw row values as JSON field names and read
    # them back through ``transformations``). The single summary line
    # below emits the count only — the per-row store keeps the raw
    # payload for researcher audit, so naming the fields here was
    # exfil-without-benefit.
    unknown_field_count = 0
    # Same pattern for dict_numeric inner keys: non-string keys and
    # non-finite values inside an allowed dict-of-numeric field both
    # carry caller-controlled bytes if echoed. We collapse them into
    # one summary line per parent field.
    dict_drops: dict[str, int] = {}

    for k, v in raw.items():
        if k not in allowed:
            unknown_field_count += 1
            continue
        if k in integer:
            if not isinstance(v, int) or isinstance(v, bool):
                _log(f"dropped {k!r}: expected int, got {type(v).__name__}")
                continue
            out[k] = v
        elif k in numeric:
            if not _is_finite_number(v):
                _log(f"dropped {k!r}: not a finite number")
                continue
            out[k] = float(v)
        elif k in string:
            if not isinstance(v, str):
                _log(f"dropped {k!r}: expected str, got {type(v).__name__}")
                continue
            cleaned = safe_text(v)
            if cleaned != v:
                _log(f"sanitized scalar string field {k!r}")
            out[k] = cleaned
        elif k in dict_numeric:
            if not isinstance(v, dict):
                _log(f"dropped {k!r}: expected dict, got {type(v).__name__}")
                continue
            clean: dict[str, float] = {}
            inner_drops = 0
            inner_collisions = 0
            for kk, vv in v.items():
                if not isinstance(kk, str):
                    inner_drops += 1
                    continue
                if not _is_finite_number(vv):
                    inner_drops += 1
                    continue
                # safe_key on the key — e.g. coefficient names, which
                # originate in the data's variable names, cross to
                # Claude. Two raw keys that ``safe_key`` collapses to
                # the same form (newline → space, 40-char prefix
                # share) MUST NOT silently overwrite — the second
                # value would replace the first and the model would
                # see one value labelled by an ambiguous key. Drop
                # the duplicate; track a count for the transformation
                # log so the researcher can audit. The vcov path in
                # _sanitize_linear_regression detects collisions in
                # the same shape.
                safe_kk = safe_key(kk)
                if safe_kk in clean:
                    inner_collisions += 1
                    continue
                clean[safe_kk] = float(vv)
            if inner_drops:
                dict_drops[k] = inner_drops
            if inner_collisions:
                # Separate counter so the cause is auditable. Names
                # withheld — collisions identify pairs of data-derived
                # raw keys.
                _log(
                    f"dropped {inner_collisions} duplicate inner "
                    f"key(s) from {k!r} after sanitization "
                    f"(colliding names withheld)"
                )
            out[k] = clean
        elif k in list_string:
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                _log(f"dropped {k!r}: not a list[str]")
                continue
            # Each string element originates in the data — sanitize all.
            out[k] = [safe_key(x) for x in v]
        elif k in list_numeric:
            if not isinstance(v, list) or not all(_is_finite_number(x) for x in v):
                _log(f"dropped {k!r}: not a list of finite numbers")
                continue
            out[k] = [float(x) for x in v]

    if unknown_field_count:
        # Aggregate count only — the field names themselves were
        # data-derived and are deliberately not echoed. Researchers
        # who need to audit the dropped names can read the raw payload
        # from the per-row store; the model only sees the count.
        _log(
            f"dropped {unknown_field_count} unknown/forbidden "
            f"top-level field(s) (names withheld)"
        )
    if dict_drops:
        # Per-parent count for malformed dict-of-numeric inner entries.
        # Both non-string keys and non-finite values are collapsed —
        # the inner key is data-derived (a coefficient / VIF / vcov
        # row name) and a non-string key would otherwise be coerced to
        # str and echoed back.
        for parent, n_dropped in sorted(dict_drops.items()):
            _log(
                f"dropped {n_dropped} malformed entry(ies) from "
                f"{parent!r} (inner keys/values withheld)"
            )
    if surplus:
        # Single line that bounds the total log size at
        # ``_COLLECT_ALLOWED_LOG_CAP + 1`` regardless of payload size.
        t.append(
            f"… and {surplus} more drops omitted from this payload's log "
            f"(cap {_COLLECT_ALLOWED_LOG_CAP})"
        )
    return out


def _correlation_invariants_hold(
    corr: dict[str, dict[str, float]],
    declared: set[str],
) -> tuple[bool, str]:
    """Verify the aggregate invariants of a real correlation matrix.

    A correlation matrix from ``df.corr()`` is square, symmetric,
    has 1s on the main diagonal, and (Pearson) is bounded in
    [-1, 1]. The per-cell checks already clip to [-1, 1]; this
    function adds:

    * **Completeness**: every declared variable has a row whose
      columns cover every other declared variable. A partial matrix
      could otherwise be used to thread free cells past the
      declared-variables cap.
    * **Diagonals = 1**: a real Pearson / Spearman / Kendall
      diagonal is exactly 1 (or, with float round-trip noise,
      within a small epsilon of 1). Any other value indicates a
      hand-crafted matrix.
    * **Symmetry**: ``corr[i][j] == corr[j][i]`` within a small
      tolerance. Halves the bandwidth attacker-engineered values
      could otherwise use.

    Returns ``(ok, reason)``. ``reason`` is suitable for the
    sanitizer's ``rejection_reason``; it does not echo any specific
    correlation value back to the model (just structural facts).
    """
    REL_TOL = 1e-3
    ABS_TOL = 1e-9
    # Completeness: every declared variable must have its own row,
    # and that row must cover every other declared variable.
    for var in declared:
        row = corr.get(var)
        if row is None:
            return (False, f"missing row for declared variable {var!r}")
        for other in declared:
            if other not in row:
                return (
                    False,
                    f"row {var!r} missing column {other!r} "
                    f"(declared but unreported correlation)"
                )
    # Diagonals = 1.
    for var in declared:
        diag = corr[var][var]
        if abs(diag - 1.0) > ABS_TOL + REL_TOL:
            return (
                False,
                f"diagonal not 1.0: corr[{var!r}][{var!r}]={diag!r}"
            )
    # Symmetry.
    for row_var in declared:
        row = corr[row_var]
        for col_var in declared:
            if col_var == row_var:
                continue
            val = row[col_var]
            partner = corr[col_var][row_var]
            diff = abs(val - partner)
            scale = max(abs(val), abs(partner), 1.0)
            if diff > ABS_TOL + REL_TOL * scale:
                return (
                    False,
                    f"asymmetric: corr[{row_var!r}][{col_var!r}]={val!r} "
                    f"!= corr[{col_var!r}][{row_var!r}]={partner!r}"
                )
    return (True, "")


def _vcov_invariants_hold(
    vcov: dict[str, dict[str, float]],
    standard_errors: dict[str, float],
) -> tuple[bool, str]:
    """Verify the aggregate invariants of a variance-covariance matrix.

    Returns ``(ok, reason)``. A real cov matrix from σ²·(X'X)^-1 is
    symmetric (``vcov[i][j] == vcov[j][i]``) and its diagonals are
    the squared standard errors. Anything emitted through the
    generic ``result(type="linear_regression", vcov=...)`` escape
    hatch can carry arbitrary numeric cells; without these checks a
    hostile payload smuggles up to N² scalar values to the model
    through ``expand_result``. The checks reject the whole matrix
    rather than per-cell so an attacker can't slip a small number
    of inconsistent cells through.

    Tolerance: symmetry is checked against a small relative epsilon
    that accommodates float round-trip noise from JSON. Diagonals
    compare ``vcov[i][i]`` against ``standard_errors[i]**2`` with a
    similarly relaxed bound so legitimately-fit models with
    ill-conditioned designs still pass.
    """
    REL_TOL = 1e-3
    ABS_TOL = 1e-9
    # Symmetry: every off-diagonal cell must have a partner across
    # the main diagonal with a near-equal value.
    for row, inner in vcov.items():
        for col, val in inner.items():
            if row == col:
                continue
            partner_inner = vcov.get(col)
            if partner_inner is None or row not in partner_inner:
                return (
                    False,
                    f"asymmetric: vcov[{row!r}][{col!r}] present but "
                    f"vcov[{col!r}][{row!r}] missing"
                )
            partner = partner_inner[row]
            diff = abs(val - partner)
            scale = max(abs(val), abs(partner), 1.0)
            if diff > ABS_TOL + REL_TOL * scale:
                return (
                    False,
                    f"asymmetric: vcov[{row!r}][{col!r}]={val!r} "
                    f"!= vcov[{col!r}][{row!r}]={partner!r}"
                )
    # Diagonals match SE². Intercept aliases in ``standard_errors``
    # may use a different name than the vcov diagonal key (the
    # sanitizer accepts a few intercept aliases); skip the check
    # when no matching SE entry exists rather than over-reject.
    for row, inner in vcov.items():
        diag = inner.get(row)
        if diag is None:
            continue
        if diag < 0:
            return (
                False,
                f"negative variance: vcov[{row!r}][{row!r}]={diag!r}"
            )
        se = standard_errors.get(row)
        if se is None:
            continue
        expected = float(se) ** 2
        diff = abs(diag - expected)
        scale = max(abs(diag), abs(expected), 1.0)
        if diff > ABS_TOL + REL_TOL * scale:
            return (
                False,
                f"diagonal mismatch: vcov[{row!r}][{row!r}]={diag!r} "
                f"!= standard_errors[{row!r}]**2={expected!r}"
            )
    return (True, "")


def _coarsen_small_missing_count(
    out: dict[str, Any],
    transformations: list[str],
    config: SDCConfig,
) -> None:
    """Replace ``missing_count`` with the suppression marker when its
    exact value is itself disclosive (``0 < missing_count < threshold``).

    An exact small missingness count identifies the few records whose
    value on this variable is missing — combined with other variables
    it supports re-identification ("the one patient who declined to
    answer income"). Same threshold as cell suppression so the rule
    is uniform across payload kinds. Zero is left as 0 (no
    missingness, nothing to suppress).

    Mutates ``out`` in place. Appends one log line if coarsening
    fired. The schema-side ``request_data(na_count)`` path already
    enforces this gate symmetrically; this helper closes the gap on
    the stored-result path (descriptive / frequency_table /
    correlation_matrix / crosstab) where ``missing_count`` is
    allowlisted but was previously forwarded verbatim.
    """
    threshold = config.cell_suppression_threshold
    miss_raw = out.get("missing_count")
    if isinstance(miss_raw, int) and 0 < miss_raw < threshold:
        out["missing_count"] = suppression_marker(threshold)
        transformations.append(
            f"coarsened missing_count to {suppression_marker(threshold)} "
            f"(exact small missingness counts are themselves disclosive)"
        )


def _coarsen_small_distinct_count(
    out: dict[str, Any],
    transformations: list[str],
    config: SDCConfig,
) -> None:
    """Replace ``distinct_count`` with the suppression marker when its
    exact value is itself disclosive (``0 < distinct_count < threshold``).

    ``distinct_count`` is the number of unique values of the variable.
    A small exact value means the variable partitions the (>= min_n)
    analyzed records into very few groups — structurally the same
    disclosure surface as a frequency_table with a handful of cells,
    which we already cell-suppress. "523 records, 2 distinct employers"
    tells you the average group is ~260, but "12 records, 2 distinct"
    or a low count paired with other margins narrows membership the way
    a small cell does. The actual values aren't emitted, so the leak is
    cardinality, not identity — but we hold ``distinct_count`` to the
    same floor as cell suppression / ``missing_count`` so the descriptive
    path can't become a side channel for low-cardinality structure that
    ``from_table`` would have suppressed. Zero is left as-is (it only
    arises when there are no non-missing records, which the ``n`` gate
    already precludes).

    Mutates ``out`` in place. Appends one log line if coarsening fired.
    """
    threshold = config.cell_suppression_threshold
    distinct_raw = out.get("distinct_count")
    if isinstance(distinct_raw, int) and 0 < distinct_raw < threshold:
        out["distinct_count"] = suppression_marker(threshold)
        transformations.append(
            f"coarsened distinct_count to {suppression_marker(threshold)} "
            f"(exact small unique-value counts are themselves disclosive)"
        )


def _coarsen_small_cox_counts(
    out: dict[str, Any],
    transformations: list[str],
    config: SDCConfig,
) -> None:
    """Replace ``n_failures`` / ``n_subjects`` with the suppression
    marker when their exact values fall below ``cell_suppression_threshold``.

    Survival-specific Cox fits commonly report "n records / n subjects /
    n failures" together. ``n`` (top-level) is already gated by
    ``require_minimum_n(config.min_n_regression)`` upstream — typically
    a much higher floor than the cell-suppression threshold — so a
    Cox payload that survives to this point has at least
    ``min_n_regression`` records. But ``n_failures`` is a different
    quantity: it counts events (deaths, conversions, churn) and on a
    rare-outcome study can be tiny even when ``n`` is in the thousands.
    "324 subjects, 3 events" identifies those 3 specific individuals
    just as surely as a frequency_table cell with count 3 would.

    Apply the same threshold as cell suppression / missing_count for
    a uniform disclosure rule. Zero is left as 0 (no events — no
    individual to identify; same posture as ``_coarsen_small_missing_count``).
    ``n_subjects`` is coarsened for symmetry: in survival data it CAN
    differ from ``n`` (records can split into multiple per-subject
    episodes via ``stset``) and an analyst studying a panel of e.g.
    very rare patient subgroups could have a small ``n_subjects``
    even with many records.
    """
    threshold = config.cell_suppression_threshold
    for field in ("n_failures", "n_subjects"):
        raw = out.get(field)
        if isinstance(raw, int) and 0 < raw < threshold:
            out[field] = suppression_marker(threshold)
            transformations.append(
                f"coarsened {field} to {suppression_marker(threshold)} "
                f"(exact small Cox-style event/subject counts are "
                f"themselves disclosive)"
            )


# ---------------------------------------------------------------------------
# Identifier-shape gate for variable-name fields
# ---------------------------------------------------------------------------
#
# Background. Allowlisted string fields (``response_variable``,
# ``predictor_variables``, ``variable``, ``row_variable``,
# ``value_variable``, ``col_variable``, correlation ``variables``) are
# supposed to carry COLUMN NAMES — short identifiers chosen by the
# researcher in their data file. ``safe_text`` / ``safe_key`` neutralise
# prompt-injection text (control chars, newlines, length) but place no
# constraint on the character class. After whitespace-flattening, a
# value like ``"x SYSTEM: ignore previous"``, a CSV row
# ``'"John Smith",25,"Boston, MA",50000'``, or a JSON dump
# ``'{"id": 123, "ssn": "..."}'`` all survive ``safe_text`` and reach
# the model verbatim through the allowlist.
#
# Documented "Known gap #1" at the top of this file is exactly this:
# nothing ties these fields back to the dataset schema. The ideal fix
# lives in the runtime library (require a model object, derive names
# from ``xlevels``), but a partial sanitizer-side defence is cheap:
# require these fields to MATCH AN IDENTIFIER SHAPE before they're
# echoed back. The shape admits every character a legitimate column /
# coefficient name could plausibly contain (letters, digits,
# underscore, period for SQL/R/Python convention; parens, colon, hash,
# caret for R / Stata formula operators like ``factor(x)Asia``,
# ``I(age^2)``, ``age:sex``, ``c.age#c.sex``) and EXCLUDES the
# characters that raw CSV rows / JSON dumps / error-message bodies
# would carry (spaces, quotes, commas, semicolons, brackets, braces,
# equals, ampersand, slashes, dollar, asterisk).
#
# Threat narrowed, not eliminated: a script that already controls the
# dataset's column names (e.g. dataset prepared by the same hostile
# upstream) can still encode bits in shapes that pass — but bandwidth
# drops by an order of magnitude (40-char identifier alphabet vs.
# 120-char arbitrary-text alphabet), and the most common raw-data
# shapes (CSV rows, JSON, error bodies, secrets with ``=`` or ``-``
# separators) are filtered out.
#
# When a value fails the gate, it's replaced with the empty string and
# a transformation log entry records the drop. For LIST-valued fields
# (``predictor_variables``, correlation ``variables``) non-conforming
# entries are removed from the list — coefficient-dict keys are
# already filtered through the resulting list elsewhere in this
# module, so dropping a predictor here automatically drops its
# coefficient / SE / t / p / vif entries.
_NAME_IDENT_RE = re.compile(r"^[A-Za-z0-9_.(][A-Za-z0-9_.():^#]*$")

# ``safe_text`` / ``safe_key`` append this marker when an input
# exceeds the cap. A legitimate over-length identifier (rare but
# valid — long column names in user datasets) lands here as
# ``"<prefix>[TRUNCATED]"``; the bracket chars aren't in the regex
# character class above, so we strip the suffix before matching.
_TRUNCATION_TAIL = "[TRUNCATED]"


def _is_identifier_shape(value: str) -> bool:
    """True iff ``value`` matches the column-name / coefficient-name shape.

    Empty string is treated as non-identifier (callers that legitimately
    have already dropped a value should not re-enter this gate). The
    trailing ``[TRUNCATED]`` marker emitted by ``safe_text`` / ``safe_key``
    on over-length inputs is tolerated — the marker chars are not in
    the identifier alphabet, but their presence on the suffix is a
    sanitizer-controlled signal, not caller-controlled bytes.
    """
    if not value:
        return False
    body = (
        value[: -len(_TRUNCATION_TAIL)]
        if value.endswith(_TRUNCATION_TAIL)
        else value
    )
    return bool(_NAME_IDENT_RE.fullmatch(body))


def _enforce_identifier_string_fields(
    out: dict[str, Any],
    fields: frozenset[str],
    transformations: list[str],
    *,
    type_label: str,
) -> None:
    """Replace non-identifier-shape string fields with the empty string.

    Only fields actually present in ``out`` are checked. Mutates ``out``
    in place. Logs one transformation line per dropped field, naming
    the FIELD (which is sanitizer-controlled, not data-derived) and
    withholding the rejected VALUE (which is caller-controlled).
    """
    for field in fields:
        v = out.get(field)
        if not isinstance(v, str) or not v:
            continue
        if not _is_identifier_shape(v):
            out[field] = ""
            transformations.append(
                f"dropped {field!r} from {type_label} payload: value did "
                f"not match the column-name / coefficient-name identifier "
                f"shape (value withheld — caller-controlled)"
            )


def _enforce_identifier_list_field(
    out: dict[str, Any],
    field: str,
    transformations: list[str],
    *,
    type_label: str,
) -> None:
    """Filter a list-of-strings field to entries matching the identifier shape.

    Mutates ``out[field]`` in place. Logs a single transformation line
    with the COUNT of dropped entries (names withheld — entries are
    caller-controlled). If every entry is dropped the field becomes
    an empty list; downstream ``_require_after_filter`` may then
    reject the payload, which is the desired hard-fail behaviour.
    """
    raw_list = out.get(field)
    if not isinstance(raw_list, list):
        return
    kept = [x for x in raw_list if isinstance(x, str) and _is_identifier_shape(x)]
    dropped = len(raw_list) - len(kept)
    if dropped:
        out[field] = kept
        transformations.append(
            f"dropped {dropped} non-identifier-shape entry(ies) from "
            f"{field!r} in {type_label} payload (names withheld — "
            f"caller-controlled)"
        )


# ---------------------------------------------------------------------------
# Linear regression sanitizer
# ---------------------------------------------------------------------------

def _sanitize_linear_regression(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _OLS_REQUIRED, "linear_regression")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        # Don't echo ``n_raw`` itself — a malicious script could set
        # ``n`` to a raw cell value to smuggle it out via this
        # rejection_reason (which submit_script forwards into both
        # the inline result and the persisted diagnostic row). The
        # type name leaks zero bits of payload content.
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_regression, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=str(e),
        )

    # Structural size cap on predictor_variables. Each predictor name
    # goes through safe_key (40-char cap) already; this check bounds
    # the total number of names, so the data channel available through
    # this list is bounded in both dimensions. Also catches
    # accidentally-huge models that wouldn't be interpretable research
    # output anyway.
    raw_predictors = raw.get("predictor_variables")
    if isinstance(raw_predictors, list) and len(raw_predictors) > _OLS_MAX_PREDICTORS:
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=(
                f"predictor_variables has {len(raw_predictors)} entries; "
                f"the structural cap is {_OLS_MAX_PREDICTORS}. A regression "
                f"with that many predictors isn't interpretable output — "
                f"rejected as probable adversarial payload."
            ),
        )

    # Same structural cap on the other _OLS_ALLOWED_LIST_STRING name
    # lists. ``_collect_allowed``'s ``list_string`` branch only
    # validates element type (each entry still goes through
    # ``safe_key``'s 40-char cap), never list length — every OTHER
    # list_string field in this module (correlation's ``variables``,
    # DiD-event's ``groups``/``event_times``, marginal-effects'
    # ``variables``, factor analysis's ``variables``, clustering's
    # ``variables``) has its own explicit length cap checked here,
    # before ``_collect_allowed`` runs. These three were the one
    # gap: an adversarial payload could smuggle an unbounded number
    # of short strings through ``instrument_variables``,
    # ``endogenous_variables``, or ``cluster_variables`` and still
    # reach ``ok=True``. Same envelope as ``predictor_variables``
    # per the field's own doc comment above.
    for list_field in ("instrument_variables", "endogenous_variables",
                       "cluster_variables"):
        raw_list_field = raw.get(list_field)
        if (isinstance(raw_list_field, list)
                and len(raw_list_field) > _OLS_MAX_PREDICTORS):
            return SanitizerResult(
                ok=False, analysis_type=_emitted_regression_type(raw),
                rejection_reason=(
                    f"{list_field} has {len(raw_list_field)} entries; "
                    f"the structural cap is {_OLS_MAX_PREDICTORS}. "
                    f"Rejected as probable adversarial payload."
                ),
            )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_OLS_ALLOWED_NUMERIC_FIELDS,
        integer=_OLS_ALLOWED_INT_FIELDS,
        string=(
            _OLS_ALLOWED_STRING_FIELDS
            | frozenset(("_registry_method_id",))
        ),
        dict_numeric=_OLS_ALLOWED_DICT_NUMERIC,
        list_string=_OLS_ALLOWED_LIST_STRING,
        transformations=transformations,
    )

    raw_registry_method = raw.get("_registry_method_id")
    out.pop("_registry_method_id", None)
    if raw_registry_method is not None:
        if (not isinstance(raw_registry_method, str)
                or raw_registry_method not in _REGRESSION_REGISTRY_METHOD_IDS):
            return SanitizerResult(
                ok=False, analysis_type=_emitted_regression_type(raw),
                rejection_reason=(
                    "regression helper registry method marker is invalid"
                ),
            )
        out["registry_method_id"] = raw_registry_method

    # Re-check required fields after type filtering. ``_require_fields``
    # only verifies key presence in ``raw``; ``_collect_allowed`` then
    # silently drops any required field whose type doesn't match
    # (e.g. ``coefficients`` shipped as a string). Without this gate,
    # a wrong-typed ``coefficients`` / ``standard_errors`` /
    # ``response_variable`` survives to ``ok=True`` with the field
    # absent. ``n`` is already pre-validated above.
    missing_after_filter = _require_after_filter(
        out, _OLS_REQUIRED, "linear_regression",
        pre_validated=frozenset(("n",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate. ``response_variable`` and
    # ``cluster_variable`` carry single column names; each entry of
    # ``predictor_variables`` carries a coefficient name. See
    # ``_enforce_identifier_*`` above for the threat model. This gate
    # runs BEFORE the categorical-contrast / cross-field-key checks so
    # those downstream passes see the cleaned predictor list — a
    # predictor dropped here automatically loses its coefficient / SE /
    # t / p / vif entries because ``declared_predictors`` is recomputed
    # from ``out["predictor_variables"]`` below.
    _enforce_identifier_string_fields(
        out,
        frozenset(("response_variable", "cluster_variable")),
        transformations,
        type_label="linear_regression",
    )
    _enforce_identifier_list_field(
        out, "predictor_variables", transformations,
        type_label="linear_regression",
    )

    # Disclosure-control gate: refuse formula-categorical coefficient
    # names. statsmodels / patsy formula fits encode raw categorical
    # levels into coefficient names using contrast markers — e.g.
    # ``C(diagnosis)[T.diabetes]`` for treatment contrasts, with
    # ``[Sum.`` / ``[Diff.`` / ``[Helmert.`` for other coding schemes.
    # ``safe_key`` only enforces prompt-injection bounds (length,
    # control chars); it doesn't recognise the level value as data,
    # so a script can ``ols('y ~ C(secret)', data=df).fit()`` and
    # leak each unique level of ``secret`` through the regression
    # coefficient / SE / p-value keys (and through the predictor
    # list itself).
    #
    # Force the script to expand dummies explicitly via
    # ``pd.get_dummies(...)`` and pass them as named columns. That
    # moves the level-naming responsibility into the script proper
    # (where it's visible in the code the researcher reviews) and
    # the resulting predictor names go through ``predictor_variables``
    # like any other column name — same disclosure surface as a
    # normal regression on already-encoded data.
    _CATEGORICAL_CONTRAST_RE = re.compile(r"\[[A-Za-z]+\.")
    suspicious_keys: set[str] = set()
    for name in out.get("predictor_variables") or []:
        if isinstance(name, str) and _CATEGORICAL_CONTRAST_RE.search(name):
            suspicious_keys.add(name)
    for dict_field in _OLS_ALLOWED_DICT_NUMERIC:
        d = out.get(dict_field)
        if not isinstance(d, dict):
            continue
        for k in d:
            if isinstance(k, str) and _CATEGORICAL_CONTRAST_RE.search(k):
                suspicious_keys.add(k)
    if suspicious_keys:
        return SanitizerResult(
            ok=False, analysis_type=_emitted_regression_type(raw),
            rejection_reason=(
                "regression payload contains formula-categorical "
                "coefficient name(s) — patsy / statsmodels formula "
                "fits embed raw categorical level values into "
                "coefficient names (e.g. ``C(var)[T.level]``), "
                "which would surface those level values without "
                "going through the frequency-table cell suppression "
                "policy. Expand categorical predictors into named "
                "indicator columns before fitting (pandas: "
                "``pd.get_dummies(df, columns=[...], drop_first=True)``; "
                "R: build a model matrix with ``model.matrix`` then "
                "fit on the resulting numeric columns) so the "
                "predictor names you emit are plain identifiers."
            ),
        )

    # Cross-field integrity: each coefficient-dict key must name a
    # declared predictor OR the intercept. Without this, a prompt-
    # an injected model can smuggle arbitrary numbers out by emitting
    # e.g. ``coefficients: {leak_bit_0: 0.001, leak_bit_1: 0.002, …}``
    # — the inner dict accepts any well-formed key through
    # ``_collect_allowed``, and precision-clamping just rounds the
    # smuggled values, it doesn't reject them.
    #
    # R reports the intercept as ``(Intercept)``; Stata as ``_cons``.
    # We accept both plus a permissive ``intercept`` form in case
    # the runtime library normalizes. Any other key not declared in
    # ``predictor_variables`` is dropped with a transformation log
    # entry so the researcher can see what got stripped.
    declared_predictors = set(out.get("predictor_variables") or [])
    # Intercept aliases each runtime emits. R's lm() reports
    # "(Intercept)"; statsmodels formula fits report "Intercept";
    # statsmodels ``add_constant(X)`` reports "const"; Stata reports
    # "_cons". The lowercase "intercept" form is a permissive fallback
    # in case a future runtime normalizes naming.
    allowed_coefficient_keys = declared_predictors | {
        "(Intercept)", "_cons", "intercept", "Intercept", "const",
    }
    for dict_field in _OLS_ALLOWED_DICT_NUMERIC:
        if dict_field not in out:
            continue
        # ``fixed_effects`` (and any future dict-numeric field whose
        # keys aren't coefficient names) lives outside this check by
        # design — see _OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK.
        if dict_field in _OLS_DICT_FIELDS_SKIP_COEF_KEY_CHECK:
            continue
        d = out[dict_field]
        if not isinstance(d, dict):
            continue
        kept: dict[str, float] = {}
        dropped: list[str] = []
        for k, v in d.items():
            if k in allowed_coefficient_keys:
                kept[k] = v
            else:
                dropped.append(k)
        if dropped:
            # Names withheld by design: ``dropped`` contains keys
            # the caller-authored script put into the result dict
            # (coefficient / SE / t / p / vif keys originate from
            # the regression's design matrix column names, which
            # the script chooses freely). Echoing those names back
            # gives a script that ran on raw data ~30 strings ×
            # ~40 chars per submit_script call of attacker-chosen
            # content through the transformations log — a covert
            # channel for small high-value values (numeric IDs,
            # ZIP codes, salaries) that's both well-bounded enough
            # to fit in the safe_key cap and far easier than the
            # legitimate model-context channels. ``_collect_allowed``
            # uses this same "names withheld" treatment for unknown
            # top-level fields and for malformed inner dict values;
            # this matches it.
            transformations.append(
                f"dropped {len(dropped)} undeclared key(s) from "
                f"{dict_field!r} (names withheld — keys are caller-"
                f"controlled and could carry raw data bytes)"
            )
        out[dict_field] = kept

    # ``robust_se_type`` enum validation. The string allowlist above
    # accepts the field; this gate normalises the value to the
    # canonical small vocabulary so the model can interpret the
    # variance estimator at a glance. Free-text values still pass
    # ``safe_text`` and could fit in 40 chars, but a non-enum value
    # is more likely a typo or a niche flavour we haven't pinned
    # than something the model should reason on — drop it with a
    # transformation note instead of leaking ambiguous strings.
    rse = out.get("robust_se_type")
    if rse is not None and rse not in _OLS_VALID_ROBUST_SE_TYPE:
        transformations.append(
            f"dropped 'robust_se_type' value (must be one of "
            f"{sorted(_OLS_VALID_ROBUST_SE_TYPE)})"
        )
        del out["robust_se_type"]

    # ``converged`` enum validation — same reasoning and pattern as
    # ``robust_se_type`` immediately above.
    conv = out.get("converged")
    if conv is not None and conv not in _OLS_VALID_CONVERGED:
        transformations.append(
            f"dropped 'converged' value (must be one of "
            f"{sorted(_OLS_VALID_CONVERGED)})"
        )
        del out["converged"]

    # Variance-covariance matrix (vcov). Optional, dict-of-dict-of-
    # numeric keyed on the same coefficient names. Pure aggregate from
    # the design (sigma^2 * (X'X)^-1); the diagonals are SE^2 and the
    # off-diagonals enable Wald tests / joint hypothesis testing /
    # linear-combination CIs the model can compute itself. Each row
    # AND column key must reference a declared predictor (or
    # intercept alias); alien keys are dropped with the same defense
    # used on coefficients above.
    #
    # Sanitize each row/col key through ``safe_key`` BEFORE validation.
    # ``allowed_coefficient_keys`` was derived from already-sanitized
    # coefficient names (``out["predictor_variables"]`` was passed
    # through ``safe_key`` in ``_collect_allowed``). The raw vcov
    # keys haven't been sanitized yet, so any coefficient name that
    # changes under ``safe_key`` (length > 40, embedded control
    # chars, newlines) would compare unequal to its sanitized
    # counterpart and the entire row/col would be dropped as
    # "undeclared" — losing the matrix while keeping the coefficient
    # / SE entries intact (those went through the dict_numeric
    # branch in ``_collect_allowed``, which sanitizes keys). Apply
    # ``safe_key`` here so the comparison is apples-to-apples, and
    # detect post-sanitization collisions (two raw keys that map to
    # the same cleaned form) explicitly so a sanitized matrix
    # doesn't silently overwrite cells.
    raw_vcov = raw.get("vcov")
    if isinstance(raw_vcov, dict):
        sanitized_vcov: dict[str, dict[str, float]] = {}
        dropped_vcov: list[str] = []
        collisions: list[str] = []
        for row_key, row_value in raw_vcov.items():
            if not isinstance(row_key, str):
                dropped_vcov.append(f"row {row_key!r} (non-string key)")
                continue
            safe_row = safe_key(row_key)
            if safe_row not in allowed_coefficient_keys:
                dropped_vcov.append(f"row {safe_row!r}")
                continue
            if not isinstance(row_value, dict):
                dropped_vcov.append(f"row {safe_row!r} (non-dict)")
                continue
            sanitized_row: dict[str, float] = {}
            for col_key, val in row_value.items():
                if not isinstance(col_key, str):
                    dropped_vcov.append(
                        f"{safe_row}.{col_key!r} (non-string key)"
                    )
                    continue
                safe_col = safe_key(col_key)
                if safe_col not in allowed_coefficient_keys:
                    dropped_vcov.append(f"{safe_row}.{safe_col}")
                    continue
                if not _is_finite_number(val):
                    continue
                if safe_col in sanitized_row:
                    # Two raw column keys cleaned to the same name.
                    # Don't silently overwrite the earlier value;
                    # log the collision and skip the duplicate so
                    # the matrix degrades safely (the model sees the
                    # transformation log entry and can decide
                    # whether to re-fit with disambiguated names).
                    collisions.append(f"{safe_row}.{safe_col}")
                    continue
                sanitized_row[safe_col] = float(val)
            if sanitized_row:
                if safe_row in sanitized_vcov:
                    collisions.append(f"row {safe_row}")
                    continue
                sanitized_vcov[safe_row] = sanitized_row
        if dropped_vcov:
            # Names withheld for the same reason as the
            # ``dict_numeric`` log entry above — vcov row / col
            # keys originate in the regression's predictor names
            # and are caller-controlled bytes.
            transformations.append(
                f"dropped {len(dropped_vcov)} undeclared key(s) from "
                f"'vcov' (names withheld — keys are caller-controlled "
                f"and could carry raw data bytes)"
            )
        if collisions:
            # Collision labels are data-derived (two raw names that
            # both safe_key-cleaned to the same string). Withhold
            # them too: a script could craft colliding names whose
            # collision pattern itself encodes a payload.
            transformations.append(
                f"dropped {len(collisions)} 'vcov' cell(s) whose "
                f"sanitized keys collided (names withheld)"
            )
        if sanitized_vcov:
            # Aggregate-consistency check. A real variance-covariance
            # matrix from σ²·(X'X)^-1 is symmetric and its diagonals
            # are SE². Generic ``result(type="linear_regression",
            # vcov={...})`` bypasses the typed helper and can carry
            # arbitrary numeric cells; the key + finiteness checks
            # above don't catch that. Reject the whole vcov if either
            # invariant fails — a real model never produces such a
            # matrix, and accepting it would let the script smuggle
            # up to N² cells of attacker-shaped numeric data through
            # to the model via expand_result.
            vcov_ok, reject_reason = _vcov_invariants_hold(
                sanitized_vcov, out.get("standard_errors") or {},
            )
            if vcov_ok:
                out["vcov"] = sanitized_vcov
            else:
                transformations.append(
                    f"dropped vcov entirely: {reject_reason}"
                )

    # Precision clamp every numeric field and every dict-of-numeric
    # field. Clamp AFTER the cross-field key filter above so we only
    # pay the rounding cost on keys that survive the filter.
    n = out["n"]
    sigfigs = sigfigs_for_n(n)
    for key in _OLS_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n)
    for key in _OLS_ALLOWED_DICT_NUMERIC:
        if key in out:
            # Cardinality dicts (FE level counts, cluster counts)
            # carry integer counts describing dataset structure, not
            # data-derived measurements that scale precision with N.
            # Round to int rather than running through the sigfig
            # clamp, which would distort small counts (1247 → 1250
            # at sigfigs=3). Same rule applies to any future
            # cardinality-dict field — extend the set, not the branch.
            if key in _OLS_DICT_FIELDS_INT_COUNTS:
                out[key] = {
                    k: int(round(v)) for k, v in out[key].items()
                    if isinstance(v, (int, float)) and v >= 0
                }
            else:
                out[key] = clamp_precision_dict(out[key], n)
    # vcov is dict-of-dict; clamp each inner dict's values.
    if "vcov" in out:
        out["vcov"] = {
            row: clamp_precision_dict(inner, n)
            for row, inner in out["vcov"].items()
        }
    transformations.append(
        f"clamped all numeric fields to {sigfigs} significant figures (n={n})"
    )

    # Cox-style survival counts (``n_failures`` / ``n_subjects``) ride
    # in via the same payload type as OLS but aren't gated by
    # ``min_n_regression``. ``n_failures`` is the event count and is
    # commonly small on rare-outcome studies — "n=2000 records,
    # 3 deaths" identifies those 3 individuals. ``n_subjects`` can
    # also fall below the gate when records are split-episode rows
    # (stset can multiply rows per subject). The shared helper
    # ``_coarsen_small_cox_counts`` applies the same
    # cell-suppression rule we use for ``missing_count`` so the
    # disclosure floor is uniform across surfaces.
    _coarsen_small_cox_counts(out, transformations, config)

    return SanitizerResult(
        ok=True, analysis_type=_emitted_regression_type(raw),
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# t-test sanitizer
# ---------------------------------------------------------------------------

def _sanitize_t_test(raw: dict[str, Any], config: SDCConfig) -> SanitizerResult:
    missing_reason = _require_fields(raw, _TTEST_REQUIRED, "t_test")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="t_test", rejection_reason=missing_reason,
        )

    subtype = raw.get("test_type")
    if subtype not in _TTEST_VALID_SUBTYPES:
        # Same exfiltration concern as the dispatcher's unknown-type
        # branch: a script could set ``test_type`` to a raw cell value
        # to smuggle it through this rejection_reason. Bound the leak
        # to the type name (or, for strings, a 40-char ``safe_key``
        # which strips control chars and caps length).
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=(
                f"test_type must be one of {sorted(_TTEST_VALID_SUBTYPES)}, "
                f"got {type(subtype).__name__}"
            ),
        )

    n1 = raw.get("n1")
    if not isinstance(n1, int) or isinstance(n1, bool) or n1 < 0:
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=(
                f"n1 must be a non-negative int, got {type(n1).__name__}"
            ),
        )

    # For two-sample / welch, n2 is required; for paired, n1 is the number
    # of pairs (single effective sample size); for one-sample, n1 is N.
    n2 = raw.get("n2")
    needs_n2 = subtype in ("two_sample", "welch")
    validated_n2: int | None = None
    if needs_n2:
        if not isinstance(n2, int) or isinstance(n2, bool) or n2 < 0:
            return SanitizerResult(
                ok=False, analysis_type="t_test",
                rejection_reason=(
                    f"{subtype} requires integer n2 >= 0, got "
                    f"{type(n2).__name__}"
                ),
            )
        validated_n2 = n2

    try:
        require_minimum_n(n1, config.min_n_ttest_group, "n1")
        if validated_n2 is not None:
            require_minimum_n(validated_n2, config.min_n_ttest_group, "n2")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="t_test", rejection_reason=str(e),
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_TTEST_ALLOWED_NUMERIC_FIELDS,
        integer=_TTEST_ALLOWED_INT_FIELDS,
        string=_TTEST_ALLOWED_STRING_FIELDS,
        list_numeric=_TTEST_ALLOWED_LIST_NUMERIC,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``test_type`` and
    # ``n1`` are pre-validated above; ``mean1`` / ``t_statistic`` /
    # ``p_value`` would otherwise survive to ``ok=True`` if shipped
    # with a non-numeric type.
    missing_after_filter = _require_after_filter(
        out, _TTEST_REQUIRED, "t_test",
        pre_validated=frozenset(("n1", "test_type")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="t_test",
            rejection_reason=missing_after_filter,
        )

    alternative = out.get("alternative")
    if alternative is not None and alternative not in _TTEST_VALID_ALTERNATIVES:
        transformations.append(
            "dropped 'alternative' value (not a recognized test direction)"
        )
        del out["alternative"]

    # ``confidence_interval`` must be a 2-element [lower, upper]
    # list. The generic list_numeric filter accepts any length, so
    # without this check a 3+ element list would survive and could
    # smuggle arbitrary numbers out — one real bound plus arbitrary
    # extras. Drop any length != 2 with a transformation note.
    if "confidence_interval" in out:
        ci = out["confidence_interval"]
        if not isinstance(ci, list) or len(ci) != 2:
            transformations.append(
                f"dropped 'confidence_interval': expected a 2-element "
                f"[lower, upper] list, got "
                f"{len(ci) if isinstance(ci, list) else type(ci).__name__}"
            )
            del out["confidence_interval"]

    # Use the smallest group for conservative sig-fig scaling. If only
    # one n is present (one_sample, paired), use that.
    # ``n2`` is optional only for one-sample/paired tests.  Preserve that
    # distinction explicitly so precision scaling never operates on ``None``.
    n_for_precision = n1
    if needs_n2:
        if not isinstance(n2, int) or isinstance(n2, bool):
            return SanitizerResult(
                ok=False, analysis_type="t_test",
                rejection_reason="two-sample precision scaling requires integer n2",
            )
        n_for_precision = min(n1, n2)
    sigfigs = sigfigs_for_n(n_for_precision)
    for key in _TTEST_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_for_precision)
    if "confidence_interval" in out:
        # Length was already verified as 2 above.
        out["confidence_interval"] = [
            clamp_precision(x, n_for_precision)
            for x in out["confidence_interval"]
        ]
    transformations.append(
        f"clamped numeric fields to {sigfigs} significant figures "
        f"(smallest-group n={n_for_precision})"
    )

    return SanitizerResult(
        ok=True, analysis_type="t_test",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Descriptive statistics sanitizer
# ---------------------------------------------------------------------------

def _sanitize_descriptive(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _DESC_REQUIRED, "descriptive")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="descriptive",
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        return SanitizerResult(
            ok=False, analysis_type="descriptive",
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_descriptive, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="descriptive", rejection_reason=str(e),
        )

    transformations: list[str] = []
    # min_value / max_value are NEVER passed through here — see the
    # comment on ``_DESC_ALLOWED_NUMERIC_FIELDS`` for why. The opt-in
    # mechanism the prior code implemented (per-variable allowance
    # via ``config.non_disclosive_variables``) was unsafe because
    # nothing in the payload binds the reported values to the named
    # variable's actual column. Researchers who need a variable's
    # range should use a Sift-owned path (eg ``request_data``).
    numeric_allowlist = _DESC_ALLOWED_NUMERIC_FIELDS
    out = _collect_allowed(
        raw,
        numeric=numeric_allowlist,
        integer=_DESC_ALLOWED_INT_FIELDS,
        string=_DESC_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering — ``mean`` / ``sd``
    # / ``missing_count`` / ``variable`` would otherwise be silently
    # dropped on type mismatch and the response would still be
    # ``ok=True``.
    missing_after_filter = _require_after_filter(
        out, _DESC_REQUIRED, "descriptive",
        pre_validated=frozenset(("n",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="descriptive",
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate on ``variable`` (the single column name
    # this descriptive stat describes). Non-conforming values are
    # replaced with the empty string; see ``_enforce_identifier_*``
    # for the threat model.
    _enforce_identifier_string_fields(
        out, frozenset(("variable",)), transformations,
        type_label="descriptive",
    )

    n = out["n"]
    for key in numeric_allowlist:
        if key in out:
            out[key] = clamp_precision(out[key], n)
    _coarsen_small_missing_count(out, transformations, config)
    _coarsen_small_distinct_count(out, transformations, config)
    transformations.append(
        f"clamped numeric fields to {sigfigs_for_n(n)} significant "
        f"figures (n={n})"
    )

    return SanitizerResult(
        ok=True, analysis_type="descriptive",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Frequency-table sanitizer (1D only at v0)
# ---------------------------------------------------------------------------

def _sanitize_frequency_table(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _FREQ_REQUIRED, "frequency_table")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=missing_reason,
        )

    raw_counts = raw.get("counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=(
                "counts must be a non-empty dict of level→count"
            ),
        )
    if len(raw_counts) > _FREQ_MAX_CELLS:
        # Structural cap: 200 distinct levels is already more than any
        # readable frequency table. Bounds the data channel available
        # through level-name strings.
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=(
                f"counts has {len(raw_counts)} distinct levels; "
                f"the structural cap is {_FREQ_MAX_CELLS}. Collapse "
                f"rare levels, use a different summary, or rejected "
                f"as probable adversarial payload."
            ),
        )
    # Count values must be non-negative ints. Keys are level *names* — they
    # originate in the researcher's data (e.g. category strings in the
    # original CSV), so they're an injection surface.
    clean_counts: dict[str, int] = {}
    for k, v in raw_counts.items():
        if not isinstance(k, str):
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    f"count keys must be strings, got {type(k).__name__}"
                ),
            )
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            # Don't echo the level name (it's data-derived; even after
            # ``safe_key`` it carries up to 40 chars of attacker-
            # controlled bytes through ``rejection_reason``, which
            # ``submit_script`` forwards back to the model).
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    f"a count value is not a non-negative int "
                    f"(got {type(v).__name__}); level name withheld"
                ),
            )
        # safe_key neutralizes control chars / length / newline injections
        # in level names before they cross to Claude. But the same
        # normalisation also creates a collision surface: ``"A\nB"`` and
        # ``"A B"`` both sanitize to ``"A B"``, and two long labels
        # sharing the same 40-char prefix collapse to the same key. If
        # we silently overwrote, a small (suppressible) cell could be
        # hidden inside an aggregated total — defeating cell
        # suppression, since the post-merge count would be above
        # threshold even though one component was below it. Reject
        # the payload outright so the script has to disambiguate
        # before crossing the boundary.
        clean_key = safe_key(k)
        if clean_key in clean_counts:
            # The colliding key is data-derived; don't echo it back to
            # the model (each rejection would ship 40 chars of attacker-
            # controlled bytes through ``rejection_reason``).
            return SanitizerResult(
                ok=False, analysis_type="frequency_table",
                rejection_reason=(
                    "two distinct level names sanitize to the same "
                    "key (e.g. embedded newlines or shared 40-char "
                    "prefix). Collisions are rejected because "
                    "aggregating the counts would defeat cell "
                    "suppression on the smaller component. "
                    "Disambiguate the levels in the source data; "
                    "the colliding key is withheld."
                ),
            )
        clean_counts[clean_key] = v

    transformations: list[str] = []
    # `counts` is handled separately below (it gets SDC suppression). Strip
    # it from the raw dict before _collect_allowed so we don't spuriously
    # log "dropped 'counts'" — it isn't dropped, just routed specially.
    raw_minus_counts = {k: v for k, v in raw.items() if k != "counts"}
    out = _collect_allowed(
        raw_minus_counts,
        integer=_FREQ_ALLOWED_INT_FIELDS,
        string=_FREQ_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``counts`` is
    # validated above and routed in separately, so it sits in
    # ``pre_validated``. ``variable`` / ``n`` / ``missing_count``
    # would otherwise survive a type mismatch with ``ok=True``.
    missing_after_filter = _require_after_filter(
        out, _FREQ_REQUIRED, "frequency_table",
        pre_validated=frozenset(("counts",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="frequency_table",
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate on ``variable`` (the column this table
    # tabulates). LEVEL names in ``counts`` keys are data values, not
    # identifiers, and remain governed by ``safe_key`` + the structural
    # cell cap.
    _enforce_identifier_string_fields(
        out, frozenset(("variable",)), transformations,
        type_label="frequency_table",
    )

    # Primary cell suppression.
    primary = suppress_cells_below(
        clean_counts, config.cell_suppression_threshold
    )
    primary_suppressed = len(primary.suppressed_keys)
    if primary_suppressed:
        # Log the COUNT of suppressed cells, not their names. The level
        # names of suppressed cells are themselves disclosive (knowing
        # ``rare_diagnosis_X`` exists in this dataset identifies anyone
        # with that diagnosis), so they never leave this sanitizer.
        transformations.append(
            f"primary suppression: {primary_suppressed} cell(s) with "
            f"count < {config.cell_suppression_threshold} "
            f"(level names withheld — see [suppressed] bucket below)"
        )

    # Secondary suppression: when publishing `n`, exactly one primary-
    # suppressed cell is trivially back-calculable from the margin. The
    # fix is to also suppress the next-smallest cell so there are at
    # least two unknowns.
    has_total = "n" in out
    after_secondary = enforce_back_calc_safety(primary, total_n_present=has_total)
    secondary_added_count = (
        len(after_secondary.suppressed_keys) - len(primary.suppressed_keys)
    )
    if secondary_added_count:
        transformations.append(
            f"secondary suppression: also suppressed "
            f"{secondary_added_count} cell(s) because only one "
            f"primary-suppressed cell was back-calculable from the "
            f"total n (level name withheld)"
        )

    # Degenerate case: exactly one suppressed cell remains AND no other
    # cell was available for secondary (e.g. all cells < threshold, or
    # single-cell table). Without a sacrificial cell, the only way to
    # prevent back-calculation from the margin is to remove the margin
    # itself — drop `n` and `missing_count`.
    total_suppressed_distinct = len(after_secondary.suppressed_keys)
    if total_suppressed_distinct == 1 and has_total:
        # No secondary was added and we still have a single suppressed
        # cell + a published total. Strip the total. Note: this check
        # MUST run on the per-cell suppression result, before bucketing,
        # because bucketing collapses N suppressed cells into a single
        # entry — afterwards the dict no longer carries the count.
        for margin_field in ("n", "missing_count"):
            out.pop(margin_field, None)
        transformations.append(
            "stripped total n and missing_count: a single cell was "
            "suppressed and no secondary cell was available, so the "
            "margin would have made it back-calculable"
        )

    # Bucket every suppressed entry under a single ``[suppressed]``
    # key. The level names themselves are an SDC violation — knowing
    # ``rare_disease_X`` exists in the dataset identifies someone with
    # that diagnosis, regardless of whether the count is masked. The
    # bucket carries the suppression marker as its value (``<10``);
    # callers can read ``suppressed_cell_count`` for the count of
    # distinct levels collapsed here. The bucket aggregate is
    # back-calculable from ``n`` minus the visible cells, but only as
    # a SUM across all bucketed levels — no individual level's count
    # is recoverable.
    bucketed_counts: dict[str, int | str] = {
        k: v
        for k, v in after_secondary.counts.items()
        if isinstance(v, int)
    }
    if total_suppressed_distinct > 0:
        bucketed_counts["[suppressed]"] = suppression_marker(
            config.cell_suppression_threshold
        )
        out["suppressed_cell_count"] = total_suppressed_distinct
    out["counts"] = bucketed_counts

    # Coarsen any rare ``missing_count`` that survived the back-calc
    # strip above. ``submit_script`` can publish a frequency_table
    # with ``missing_count=1`` even when the cell suppression rule
    # otherwise fires cleanly — the schema-side ``request_data
    # (na_count)`` path already suppresses the same disclosure on
    # the discovery side, this closes the gap on the stored-result
    # side.
    _coarsen_small_missing_count(out, transformations, config)

    return SanitizerResult(
        ok=True, analysis_type="frequency_table",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Text extraction sanitizer -- local free-text structure
# ---------------------------------------------------------------------------

def _sanitize_text_extraction(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Cell-suppress category counts extracted locally from a free-text
    column, then align the per-category sentiment map to whatever
    survives suppression.

    Deliberately reuses ``suppress_cells_below`` /
    ``enforce_back_calc_safety`` -- the exact same proven primitives
    ``_sanitize_frequency_table`` uses -- rather than writing new
    suppression math for a second payload shape. The one genuinely
    new piece is step 5 below: a category whose count was suppressed
    must not leak its existence through a SEPARATE sentiment field
    that escaped the same gate.
    """
    missing_reason = _require_fields(raw, _TEXTEXTRACT_REQUIRED, "text_extraction")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="text_extraction",
            rejection_reason=missing_reason,
        )

    raw_categories = raw.get("categories")
    if not isinstance(raw_categories, dict) or not raw_categories:
        return SanitizerResult(
            ok=False, analysis_type="text_extraction",
            rejection_reason=(
                "categories must be a non-empty dict of "
                "category→count"
            ),
        )
    if len(raw_categories) > _TEXTEXTRACT_MAX_CATEGORIES:
        return SanitizerResult(
            ok=False, analysis_type="text_extraction",
            rejection_reason=(
                f"categories has {len(raw_categories)} distinct "
                f"entries; the structural cap is "
                f"{_TEXTEXTRACT_MAX_CATEGORIES}. Use a smaller, "
                f"more meaningful taxonomy."
            ),
        )

    raw_sentiment = raw.get("category_sentiment")
    if not isinstance(raw_sentiment, dict):
        return SanitizerResult(
            ok=False, analysis_type="text_extraction",
            rejection_reason=(
                "category_sentiment must be a dict of "
                "category→mean sentiment score"
            ),
        )

    # 1. Validate + safe_key-normalize categories (identical posture
    # to frequency_table's counts validation -- see that handler's
    # comments for the full collision-safety rationale).
    clean_counts: dict[str, int] = {}
    for k, v in raw_categories.items():
        if not isinstance(k, str):
            return SanitizerResult(
                ok=False, analysis_type="text_extraction",
                rejection_reason=(
                    f"category keys must be strings, got "
                    f"{type(k).__name__}"
                ),
            )
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return SanitizerResult(
                ok=False, analysis_type="text_extraction",
                rejection_reason=(
                    f"a category count is not a non-negative int "
                    f"(got {type(v).__name__}); category name withheld"
                ),
            )
        clean_key = safe_key(k)
        if clean_key in clean_counts:
            return SanitizerResult(
                ok=False, analysis_type="text_extraction",
                rejection_reason=(
                    "two distinct category names sanitize to the "
                    "same key. Disambiguate the taxonomy; the "
                    "colliding key is withheld."
                ),
            )
        clean_counts[clean_key] = v

    # 2. Normalize category_sentiment keys the same way, and clamp
    # values onto the fixed [-1, 1] scale. Values that aren't finite
    # numbers are dropped (not rejected) -- a missing sentiment score
    # for one category shouldn't sink the whole payload the way a
    # malformed COUNT does, since sentiment is secondary to the
    # category counts themselves.
    clean_sentiment: dict[str, float] = {}
    for k, v in raw_sentiment.items():
        if not isinstance(k, str):
            continue
        if not _is_finite_number(v):
            continue
        clamped = max(_TEXTEXTRACT_SENTIMENT_MIN,
                       min(_TEXTEXTRACT_SENTIMENT_MAX, float(v)))
        clean_sentiment[safe_key(k)] = round(clamped, 3)

    transformations: list[str] = []
    raw_minus_special = {
        k: v for k, v in raw.items()
        if k not in ("categories", "category_sentiment")
    }
    out = _collect_allowed(
        raw_minus_special,
        integer=_TEXTEXTRACT_ALLOWED_INT_FIELDS,
        string=_TEXTEXTRACT_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    missing_after_filter = _require_after_filter(
        out, _TEXTEXTRACT_REQUIRED, "text_extraction",
        pre_validated=frozenset(("categories", "category_sentiment")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="text_extraction",
            rejection_reason=missing_after_filter,
        )

    _enforce_identifier_string_fields(
        out, frozenset(("text_column",)), transformations,
        type_label="text_extraction",
    )

    # 3. Primary + secondary cell suppression on the category counts
    # -- byte-for-byte the same two calls frequency_table makes.
    primary = suppress_cells_below(
        clean_counts, config.cell_suppression_threshold
    )
    primary_suppressed = len(primary.suppressed_keys)
    if primary_suppressed:
        transformations.append(
            f"primary suppression: {primary_suppressed} categor"
            f"{'y' if primary_suppressed == 1 else 'ies'} with "
            f"count < {config.cell_suppression_threshold} "
            f"(category names withheld — see [suppressed] bucket below)"
        )

    has_total = "n" in out
    after_secondary = enforce_back_calc_safety(primary, total_n_present=has_total)
    secondary_added_count = (
        len(after_secondary.suppressed_keys) - len(primary.suppressed_keys)
    )
    if secondary_added_count:
        transformations.append(
            f"secondary suppression: also suppressed "
            f"{secondary_added_count} categor"
            f"{'y' if secondary_added_count == 1 else 'ies'} because "
            f"only one primary-suppressed category was back-"
            f"calculable from the total n (category name withheld)"
        )

    total_suppressed_distinct = len(after_secondary.suppressed_keys)
    if total_suppressed_distinct == 1 and has_total:
        for margin_field in ("n", "missing_count"):
            out.pop(margin_field, None)
        transformations.append(
            "stripped total n and missing_count: a single category "
            "was suppressed and no secondary category was available, "
            "so the margin would have made it back-calculable"
        )

    bucketed_counts: dict[str, int | str] = {
        k: v
        for k, v in after_secondary.counts.items()
        if isinstance(v, int)
    }
    if total_suppressed_distinct > 0:
        bucketed_counts["[suppressed]"] = suppression_marker(
            config.cell_suppression_threshold
        )
        out["suppressed_cell_count"] = total_suppressed_distinct
    out["categories"] = bucketed_counts

    # 4. Coarsen a small missing_count exactly like every other shape.
    _coarsen_small_missing_count(out, transformations, config)

    # 5. THE new part: align category_sentiment to the SURVIVING
    # category keys only. A category whose count was suppressed must
    # not have its existence (or its sentiment) leak through this
    # separate field -- computing the intersection against
    # ``bucketed_counts`` (which by construction contains only
    # non-suppressed integer cells plus the anonymous "[suppressed]"
    # bucket) is what enforces that.
    surviving_sentiment = {
        k: v for k, v in clean_sentiment.items()
        if k in bucketed_counts and isinstance(bucketed_counts[k], int)
    }
    dropped_sentiment = len(clean_sentiment) - len(surviving_sentiment)
    if dropped_sentiment:
        transformations.append(
            f"dropped sentiment score for {dropped_sentiment} "
            f"suppressed categor{'y' if dropped_sentiment == 1 else 'ies'} "
            f"(would have re-identified a suppressed cell)"
        )
    out["category_sentiment"] = surviving_sentiment

    # 6. overall_sentiment_mean only releases once the surviving n
    # clears the same descriptive floor as every other summary
    # statistic in Sift -- a mean over a handful of free-text rows is
    # exactly as disclosive as any other small-n descriptive.
    overall_raw = raw.get("overall_sentiment_mean")
    surviving_n = out.get("n")
    if (
        _is_finite_number(overall_raw)
        and isinstance(surviving_n, int)
        and surviving_n >= config.min_n_descriptive
    ):
        clamped_overall = max(
            _TEXTEXTRACT_SENTIMENT_MIN,
            min(_TEXTEXTRACT_SENTIMENT_MAX, float(overall_raw)),
        )
        out["overall_sentiment_mean"] = round(clamped_overall, 3)
    elif "overall_sentiment_mean" in raw:
        transformations.append(
            "dropped overall_sentiment_mean: below the minimum n for "
            "a released summary statistic, or n itself was stripped "
            "by suppression"
        )

    return SanitizerResult(
        ok=True, analysis_type="text_extraction",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Crosstab sanitizer (2D, no margins)
# ---------------------------------------------------------------------------

def _sanitize_crosstab(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Primary-suppress cells in a 2D contingency table.

    Structure: ``counts`` is a dict-of-dicts, ``counts[row_level][col_level]
    = int``. Cells below threshold are replaced with the suppression
    marker. No margins (row totals, column totals, grand total) are
    ever emitted; attempting to include them via any named field on the
    allowlist is impossible by construction, but we additionally log a
    loud drop message if the researcher's script tried to pass a known
    margin-field name like ``n`` or ``row_totals``.
    """
    missing_reason = _require_fields(raw, _XTAB_REQUIRED, "crosstab")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=missing_reason,
        )

    raw_counts = raw.get("counts")
    if not isinstance(raw_counts, dict) or not raw_counts:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=(
                "counts must be a non-empty dict-of-dicts "
                "(row_level → col_level → int)"
            ),
        )
    # Structural cap on the total cell count (sum over rows of inner
    # dict sizes). A 50×50 crosstab is already dense; anything bigger
    # isn't readable output.
    total_cells = 0
    for inner in raw_counts.values():
        if isinstance(inner, dict):
            total_cells += len(inner)
    if total_cells > _XTAB_MAX_CELLS:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=(
                f"counts contains {total_cells} cells; the structural "
                f"cap is {_XTAB_MAX_CELLS}. Pre-aggregate the table, "
                f"or rejected as probable adversarial payload."
            ),
        )

    # Validate nested shape and collect a flattened view for suppression.
    # Row + col keys are level *names* from the data — sanitize before
    # use. A prior version did ``clean_counts[(safe_row, safe_col)] = v``
    # unconditionally, which let two raw labels that sanitize to the
    # same safe_key SILENTLY OVERWRITE each other in the dict. Concrete
    # leak: raw row ``"A\nB"`` with count 2 (suppressible) overwritten
    # by raw row ``"A B"`` with count 100 (visible) leaves the model
    # seeing the visible count under a label that's actually ambiguous
    # — and worse, secondary suppression accounting now operates on
    # the wrong value.
    #
    # Detection: a duplicate (safe_row, safe_col) tuple in the build
    # loop means either (a) two raw row keys sanitized to the same
    # safe_row, or (b) two raw col keys within a row sanitized to
    # the same safe_col. Either is genuinely ambiguous; the SDC
    # posture is to deny rather than guess. The fix matches the
    # equivalent gate in ``data_request._resolve_variable``.
    clean_counts: dict[tuple[str, str], int] = {}
    col_levels: set[str] = set()
    for row_key, inner in raw_counts.items():
        if not isinstance(row_key, str):
            return SanitizerResult(
                ok=False, analysis_type="crosstab",
                rejection_reason=(
                    f"row keys must be strings; got {type(row_key).__name__}"
                ),
            )
        if not isinstance(inner, dict):
            # Row labels are data-derived; redact them from rejection
            # messages so the model can't trigger this branch with a
            # crafted label and read it back via ``rejection_reason``.
            return SanitizerResult(
                ok=False, analysis_type="crosstab",
                rejection_reason=(
                    f"a counts row is not a dict "
                    f"(col_level → int); got {type(inner).__name__}; "
                    f"row label withheld"
                ),
            )
        safe_row = safe_key(row_key)
        for col_key, v in inner.items():
            if not isinstance(col_key, str):
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        f"col keys must be strings; got "
                        f"{type(col_key).__name__}"
                    ),
                )
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                # Don't echo row/col labels — they are data-derived.
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        f"a counts cell is not a non-negative int "
                        f"(got {type(v).__name__}); row/col labels "
                        f"withheld"
                    ),
                )
            safe_col = safe_key(col_key)
            if (safe_row, safe_col) in clean_counts:
                # The colliding labels are data-derived; redact.
                return SanitizerResult(
                    ok=False, analysis_type="crosstab",
                    rejection_reason=(
                        "label collision after sanitization in "
                        "crosstab: two distinct raw row/col labels "
                        "sanitize to the same safe_key, which would "
                        "silently overwrite counts (and break "
                        "suppression accounting). Rename the "
                        "colliding levels in the source script — e.g. "
                        "strip embedded whitespace / control "
                        "characters before the crosstab — and re-run. "
                        "The colliding labels are withheld."
                    ),
                )
            clean_counts[(safe_row, safe_col)] = v
            col_levels.add(safe_col)

    transformations: list[str] = []

    # Strip known margin-ish fields loudly before _collect_allowed drops
    # unknown fields quietly. This makes violations of the no-margins
    # invariant visible in the transformation log.
    raw_pruned = dict(raw)
    raw_pruned.pop("counts", None)
    for field in _XTAB_FORBIDDEN_MARGIN_FIELDS:
        if field in raw_pruned:
            transformations.append(
                f"dropped margin field {field!r}: crosstabs do not emit "
                f"totals (no margins → no back-calc)"
            )
            raw_pruned.pop(field, None)

    out = _collect_allowed(
        raw_pruned,
        integer=_XTAB_ALLOWED_INT_FIELDS,
        string=_XTAB_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``counts`` is
    # validated above and reattached separately.
    missing_after_filter = _require_after_filter(
        out, _XTAB_REQUIRED, "crosstab",
        pre_validated=frozenset(("counts",)),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="crosstab",
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate on the row/col VARIABLE NAMES (column
    # names of the two factors being crosstabbed). Cell LEVEL names
    # in ``counts`` are data values and stay governed by ``safe_key``
    # + the structural cell cap.
    _enforce_identifier_string_fields(
        out, frozenset(("row_variable", "col_variable")), transformations,
        type_label="crosstab",
    )

    # Primary suppression on the flat view, then reshape back to
    # nested with bucketing. Suppressed (row, col) labels themselves
    # are disclosive — a row named ``rare_diagnosis`` whose only
    # column counts are below threshold leaks the existence of that
    # diagnosis even when its numbers are masked. We:
    #
    #   * Per surviving row: collapse all of its suppressed columns
    #     into a single ``[suppressed]`` entry.
    #   * Drop rows that have NO surviving (visible) cells entirely
    #     — including their row label — and account for them in a
    #     top-level ``suppressed_row_count`` field.
    #
    # The structural cap on table size keeps this loop cheap.
    threshold = config.cell_suppression_threshold
    marker = suppression_marker(threshold)

    # Primary status per cell: True == suppressed (below threshold).
    suppressed_status: dict[tuple[str, str], bool] = {
        key: v < threshold for key, v in clean_counts.items()
    }
    primary_count = sum(1 for s in suppressed_status.values() if s)

    # Secondary suppression to defend against per-row / per-column
    # back-calc when the model has an externally-known marginal.
    #
    # The attack: the model issues a separate ``request_data``
    # frequency_table on the row (or column) variable. That table
    # publishes per-level counts for visible levels — i.e. the row
    # / column marginal N_R (or N_C) of this crosstab. Then:
    #
    #   * If a surviving row R has exactly ONE suppressed cell, the
    #     row publishes a ``[suppressed]`` bucket whose sum equals
    #     ``N_R - sum(visible in R)`` — recovering the lone cell
    #     exactly.
    #   * Symmetrically for columns: the model sums visible cells
    #     in column C across the output and computes ``N_C -
    #     sum(visible in C)``. If exactly one cell in column C is
    #     hidden (suppressed in a surviving row, since dropped rows
    #     also have their column-C cell suppressed), that cell is
    #     recovered.
    #
    # Remedy is the standard SDC choice (ONS / Eurostat guidance):
    # promote additional visible cells to suppressed until every row
    # and every column with any suppression has either 0 or >=2
    # suppressed cells. Iterate to a fixed point — a row-side fix
    # can create a column-side violation and vice versa. The loop
    # is bounded by the total cell count.
    #
    # Victim choice: the smallest visible cell. This is the standard
    # data-utility-minimising choice — losing the smallest value
    # costs the least information to legitimate downstream
    # analysis. Ties broken by key for determinism (tests need
    # reproducibility).
    sorted_rows = sorted({r for (r, _) in clean_counts})
    sorted_cols = sorted({c for (_, c) in clean_counts})
    secondary_count = 0
    while True:
        target: tuple[str, str] | None = None
        # Row pass first.
        for r in sorted_rows:
            in_row = [c for c in sorted_cols if (r, c) in clean_counts]
            if not in_row:
                continue
            n_supp = sum(1 for c in in_row if suppressed_status[(r, c)])
            if n_supp != 1:
                continue
            visible = [
                (c, clean_counts[(r, c)]) for c in in_row
                if not suppressed_status[(r, c)]
            ]
            if not visible:
                continue
            victim_c, _ = min(visible, key=lambda cv: (cv[1], cv[0]))
            target = (r, victim_c)
            break
        # Column pass if row pass found nothing.
        if target is None:
            for c in sorted_cols:
                in_col = [r for r in sorted_rows if (r, c) in clean_counts]
                if not in_col:
                    continue
                n_supp = sum(1 for r in in_col if suppressed_status[(r, c)])
                if n_supp != 1:
                    continue
                visible = [
                    (r, clean_counts[(r, c)]) for r in in_col
                    if not suppressed_status[(r, c)]
                ]
                if not visible:
                    continue
                victim_r, _ = min(visible, key=lambda rv: (rv[1], rv[0]))
                target = (victim_r, c)
                break
        if target is None:
            break
        suppressed_status[target] = True
        secondary_count += 1

    nested_raw: dict[str, dict[str, int | str]] = {}
    suppressed_cell_count = 0
    for (r, c), v in clean_counts.items():
        if r not in nested_raw:
            nested_raw[r] = {}
        if suppressed_status[(r, c)]:
            nested_raw[r][c] = marker
            suppressed_cell_count += 1
        else:
            nested_raw[r][c] = v

    nested: dict[str, dict[str, int | str]] = {}
    suppressed_row_count = 0
    for row_label, row_cells in nested_raw.items():
        visible_cols: dict[str, int | str] = {
            c: v for c, v in row_cells.items() if isinstance(v, int)
        }
        if not visible_cols:
            # Every cell in this row was suppressed — drop the row
            # label too. Knowing the row exists (and is rare) is the
            # leak we're closing here.
            suppressed_row_count += 1
            continue
        n_suppressed_in_row = len(row_cells) - len(visible_cols)
        if n_suppressed_in_row:
            visible_cols["[suppressed]"] = marker
        nested[row_label] = visible_cols

    if primary_count:
        transformations.append(
            f"primary suppression: {primary_count} cell(s) "
            f"with count < {threshold} (cell labels withheld — "
            f"bucketed under '[suppressed]')"
        )
    if secondary_count:
        # Secondary cells were >= threshold originally but got
        # promoted to suppressed to defend against per-row /
        # per-column back-calc from an externally-known marginal
        # (a separate request_data on the row or column variable
        # publishes its level totals). The published bucket marker
        # stays ``<threshold`` for compactness; this log line is
        # the authoritative statement that some bucket entries do
        # NOT actually fall below threshold.
        transformations.append(
            f"secondary suppression: {secondary_count} additional "
            f"cell(s) promoted to '[suppressed]' to prevent "
            f"per-row/column back-calc when an external marginal "
            f"is known (smallest visible cells chosen)"
        )
    if suppressed_row_count:
        transformations.append(
            f"row suppression: {suppressed_row_count} row(s) had every "
            f"cell below threshold; row labels withheld since their "
            f"existence at this rarity is itself disclosive"
        )
        out["suppressed_row_count"] = suppressed_row_count
    if suppressed_cell_count:
        out["suppressed_cell_count"] = suppressed_cell_count

    # Strip ``missing_count`` when the cross-query back-calc is trivial.
    # ``n`` is already in ``_XTAB_FORBIDDEN_MARGIN_FIELDS`` so the
    # crosstab payload alone never exposes the grand total — but the
    # model can derive ``N`` from a separate descriptive query, then
    # compute ``sum(suppressed) = (N - missing_count) - sum(visible)``.
    # The unsafe configuration is "exactly one cell suppressed AND no
    # row was fully dropped": the surviving row's ``[suppressed]``
    # bucket then contains exactly that cell's count, the bucket sum
    # equals ``(N - missing_count) - sum(visible)`` exactly, and the
    # row label is published, so a single arithmetic step recovers the
    # cell. Mirrors the freq-table guard at ``_sanitize_frequency_table``
    # which drops both ``n`` and ``missing_count`` for the analogous
    # in-payload case. With a fully-dropped row in the mix, the dropped
    # row's cells contribute to the same sum but can't be separated, so
    # the cleanly-recoverable case requires ``suppressed_row_count == 0``.
    if (
        suppressed_cell_count == 1
        and suppressed_row_count == 0
        and "missing_count" in out
    ):
        out.pop("missing_count", None)
        transformations.append(
            "stripped missing_count: exactly one cell was suppressed "
            "and no row was dropped, so the published "
            "'[suppressed]' bucket would have been back-calculable "
            "from missing_count plus an externally-known N"
        )

    # Coarsen any rare ``missing_count`` that survived the back-calc
    # strip above. The exact small-missingness disclosure ("the one
    # row missing on either dimension") is independent of the
    # back-calc concern handled by the strip — the strip protects
    # the suppressed-cell bucket sum, this protects the missingness
    # cell itself.
    _coarsen_small_missing_count(out, transformations, config)

    out["counts"] = nested

    return SanitizerResult(
        ok=True, analysis_type="crosstab",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Magnitude-table sanitizer (sum/mean by group, with dominance rule)
# ---------------------------------------------------------------------------

def _sanitize_magnitude_table(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Suppress cells that fail either primary (n) or dominance (max_share).

    Unlike frequency tables, where the disclosure risk is "a cell of
    size 1 identifies someone", a magnitude cell can have n=100 and
    still be disclosive if one of those 100 contributors dominates —
    their value is effectively revealed by the cell's sum. That's what
    the (1, k)-dominance rule handles.

    Two suppression triggers per cell:
    - ``n < cell_suppression_threshold``: primary. Suppress.
    - ``dominance_fails(max_share, dominance_threshold)``: dominance.
      Suppress.

    The ``max_share`` field is computed by the runtime library on raw
    values (since the sanitizer has no access to them), consulted here
    for the suppression decision, and then **stripped from the output**.
    Emitting it would tell Claude "this cell has a dominant contributor"
    — information we don't need to publish.
    """
    missing_reason = _require_fields(
        raw, _MAGTAB_REQUIRED, "magnitude_table"
    )
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=missing_reason,
        )

    # Helper-provenance gate. ``max_share`` is caller-supplied and
    # consulted-only — the dominance gate trusts it. A script that
    # bypasses the typed helper (e.g. via the generic ``result()``)
    # could publish a dominance-violating value with a forged
    # ``max_share=0`` and skip the gate. Require the marker the typed
    # helper stamps. See the constant's docstring for threat-model
    # detail and the limits of this defense.
    if raw.get(_HELPER_PROVENANCE_FIELD) != _MAGTAB_HELPER_VALUE:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                "magnitude_table payloads must come through the "
                "typed runtime helper (Python: sift.from_magnitude_table; "
                "R: sift$from_magnitude_table; Stata: "
                "sift_result_magnitude). The generic sift.result() API "
                "is rejected for this type because cell-level max_share "
                "is consulted-only and a hand-crafted payload could "
                "publish a dominance-violating value with max_share=0 "
                "to skip the dominance gate."
            ),
        )

    aggregation = raw.get("aggregation")
    if aggregation not in _MAGTAB_VALID_AGGREGATIONS:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                f"aggregation must be one of "
                f"{sorted(_MAGTAB_VALID_AGGREGATIONS)}, got "
                f"{type(aggregation).__name__}"
            ),
        )

    raw_cells = raw.get("cells")
    if not isinstance(raw_cells, dict) or not raw_cells:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                "cells must be a non-empty dict of group_level → "
                "{value, n, max_share}"
            ),
        )
    if len(raw_cells) > _MAGTAB_MAX_CELLS:
        # Same rationale as the other table caps — bound the
        # data-channel bandwidth through group-name strings.
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=(
                f"cells has {len(raw_cells)} groups; the structural "
                f"cap is {_MAGTAB_MAX_CELLS}. Aggregate to fewer "
                f"groups, or rejected as probable adversarial payload."
            ),
        )

    transformations: list[str] = []

    # Strip the raw cells dict before _collect_allowed — we handle it
    # specially. Without this, the log spuriously says "dropped cells".
    # Also strip the helper-provenance marker so it doesn't appear in
    # the transformation log as a "dropped unknown field" — the marker
    # is internal to the runtime-library/sanitizer boundary and the
    # model has no business seeing its name.
    raw_pruned = {
        k: v for k, v in raw.items()
        if k != "cells" and k != _HELPER_PROVENANCE_FIELD
    }
    out = _collect_allowed(
        raw_pruned,
        string=_MAGTAB_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``cells`` and
    # ``aggregation`` are pre-validated above.
    missing_after_filter = _require_after_filter(
        out, _MAGTAB_REQUIRED, "magnitude_table",
        pre_validated=frozenset(("cells", "aggregation")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="magnitude_table",
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate on the row/value VARIABLE NAMES. Cell
    # GROUP labels (``cells`` keys) are data values and remain
    # governed by ``safe_key`` + structural cell cap.
    _enforce_identifier_string_fields(
        out,
        frozenset(("row_variable", "value_variable")),
        transformations,
        type_label="magnitude_table",
    )

    threshold_n = config.cell_suppression_threshold
    dom_threshold = config.dominance_threshold
    marker_n = suppression_marker(threshold_n)

    cleaned_cells: dict[str, Any] = {}
    # Counts only — never the group labels. See the
    # "suppressed cells leak labels" SDC fix: a cell whose label is
    # ``rare_industry`` is disclosive even when the count and value
    # are masked, since it tells the model that level exists in the
    # data at small N. We track aggregate counts for the
    # transformation log and bucket all suppressed groups under a
    # single ``[suppressed]`` entry below.
    n_suppressed_total = 0
    n_suppressed_by_n = 0
    n_suppressed_by_dominance = 0

    for raw_group, cell in raw_cells.items():
        if not isinstance(raw_group, str):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"cells keys must be strings; got {type(raw_group).__name__}"
                ),
            )
        if not isinstance(cell, dict):
            # Group labels are data-derived; redact from
            # ``rejection_reason`` (which crosses to the model via
            # ``submit_script``). ``cell`` here is the offending
            # dict-or-not from the raw payload, distinct from
            # ``cells`` (the parent dict) — no collision with the
            # parent name in the sanitizer itself.
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry is not a dict with keys "
                    f"value, n, max_share; got {type(cell).__name__}; "
                    f"group label withheld"
                ),
            )
        value = cell.get("value")
        n = cell.get("n")
        max_share = cell.get("max_share")

        if not _is_finite_number(value):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry's 'value' is not a finite "
                    f"number; got {type(value).__name__}; "
                    f"group label withheld"
                ),
            )
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry's 'n' is not a non-negative "
                    f"int; got {type(n).__name__}; "
                    f"group label withheld"
                ),
            )
        if not _is_finite_number(max_share):
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    f"a cells entry's 'max_share' is not a "
                    f"finite number; got {type(max_share).__name__}; "
                    f"group label withheld"
                ),
            )

        safe_group = safe_key(raw_group)

        fails_n = n < threshold_n
        fails_dominance = dominance_fails(float(max_share), threshold=dom_threshold)

        if fails_n or fails_dominance:
            # Don't emit a per-group entry at all — the group label
            # is itself disclosive (``rare_industry`` exists with
            # n < threshold identifies its members). Track counts
            # only.
            n_suppressed_total += 1
            if fails_n:
                n_suppressed_by_n += 1
            if fails_dominance:
                n_suppressed_by_dominance += 1
            continue
        # Reject ``safe_key`` collisions outright (mirrors crosstab /
        # frequency_table). Two raw group labels that sanitize to the
        # same form would silently overwrite — a small (suppressible)
        # cell could be replaced by a visible cell, or vice versa,
        # and the suppression accounting would never see the dropped
        # entry. Group labels are data-derived; rejection_reason
        # withholds them.
        if safe_group in cleaned_cells:
            return SanitizerResult(
                ok=False, analysis_type="magnitude_table",
                rejection_reason=(
                    "label collision after sanitization in "
                    "magnitude_table cells: two distinct raw group "
                    "labels sanitize to the same safe_key, which "
                    "would silently overwrite values (and break "
                    "dominance / primary suppression accounting). "
                    "Disambiguate the group labels in the source "
                    "script — e.g. strip embedded whitespace / "
                    "control characters before grouping — and "
                    "re-run. The colliding labels are withheld."
                ),
            )
        # Precision-clamp the value at sigfigs appropriate for n.
        cleaned_cells[safe_group] = {
            "value": clamp_precision(float(value), n),
            "n": n,
        }
        # NEVER emit max_share. It's only used internally above.

    if n_suppressed_by_n:
        transformations.append(
            f"primary suppression: {n_suppressed_by_n} cell(s) with "
            f"n < {threshold_n} (group labels withheld — bucketed "
            f"under '[suppressed]')"
        )
    if n_suppressed_by_dominance:
        transformations.append(
            f"dominance suppression: {n_suppressed_by_dominance} cell(s) "
            f"where one contributor exceeded {dom_threshold:.0%} of "
            f"the total (group labels withheld)"
        )
    transformations.append(
        "max_share stripped from every cell: dominance metric is internal "
        "to the sanitizer and never forwarded"
    )

    if n_suppressed_total:
        # Single bucketed entry for every suppressed group. The
        # marker tells the model these cells exist but their labels
        # and per-group n / value are deliberately withheld.
        cleaned_cells["[suppressed]"] = {
            "value": marker_n,
            "n": marker_n,
        }
        out["suppressed_cell_count"] = n_suppressed_total

    out["cells"] = cleaned_cells

    return SanitizerResult(
        ok=True, analysis_type="magnitude_table",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Correlation matrix sanitizer
# ---------------------------------------------------------------------------


def _sanitize_correlation_matrix(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    """Pairwise correlation matrix (Pearson / Spearman / Kendall).

    Privacy rationale: the matrix is a sums-of-products aggregate, so
    no per-row data crosses back. Three guardrails on top:

    1. Minimum N (``min_n_descriptive``) — at very low N a near-perfect
       correlation is just "the three points are collinear" and could
       imply individual coordinates, so reject below threshold.
    2. Variable-count cap — limits how much can be smuggled through
       even-well-formed payloads, mirroring the OLS predictor cap.
    3. Cross-field key validation — every row/column key in the
       correlations dict must be a declared variable. Without this,
       a prompt-injected script could smuggle channels via spurious
       keys like ``leak_bit_0`` carrying engineered values.

    Each correlation is precision-clamped (sigfigs scale with N), then
    clipped to [-1, 1] in case rounding pushed it past the boundary.
    """
    missing_reason = _require_fields(raw, _CORR_REQUIRED, "correlation_matrix")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )

    try:
        require_minimum_n(n_raw, config.min_n_descriptive, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=str(e),
        )

    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not raw_vars:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason="variables must be a non-empty list of strings",
        )
    if len(raw_vars) > _CORR_MAX_VARIABLES:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; the structural cap "
                f"is {_CORR_MAX_VARIABLES}. A correlation matrix that wide "
                f"isn't interpretable output — rejected as probable "
                f"adversarial payload."
            ),
        )

    # Method, if provided, must be one we recognise.
    method = raw.get("method")
    if method is not None and method not in _CORR_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"method must be one of {sorted(_CORR_VALID_METHODS)} or "
                f"omitted, got {type(method).__name__}"
            ),
        )

    correlations = raw.get("correlations")
    if not isinstance(correlations, dict) or not correlations:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason="correlations must be a non-empty dict of dicts",
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        integer=_CORR_ALLOWED_INT_FIELDS,
        string=_CORR_ALLOWED_STRING_FIELDS,
        list_string=_CORR_ALLOWED_LIST_STRING,
        transformations=transformations,
    )

    # Re-check required fields after type filtering. ``n``,
    # ``variables``, and ``correlations`` are pre-validated above
    # (``correlations`` is reattached after sanitization further down).
    missing_after_filter = _require_after_filter(
        out, _CORR_REQUIRED, "correlation_matrix",
        pre_validated=frozenset(("n", "variables", "correlations")),
    )
    if missing_after_filter:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=missing_after_filter,
        )

    # Identifier-shape gate on each entry of ``variables`` (column
    # names being correlated). Entries that fail are dropped before
    # ``declared`` is built — corresponding rows/cols in the
    # ``correlations`` dict will then be dropped as "undeclared" by
    # the cross-field validation below, with the same counters.
    _enforce_identifier_list_field(
        out, "variables", transformations,
        type_label="correlation_matrix",
    )

    # ``out["variables"]`` is the safe_key-transformed list (each
    # element passed through safe_key in _collect_allowed). The raw
    # ``correlations`` dict keys are not yet transformed. Compare on
    # safe_key both sides so a long or otherwise-transformed name
    # doesn't get spuriously "dropped as undeclared" simply because
    # the variables list shows the truncated form. Without this, a
    # legitimate matrix with a 50-char variable name returned with
    # ``correlations: {}`` and ``ok=True`` — silent empty success.
    sanitized_vars = list(out.get("variables") or [])
    # Reject sanitized-name collisions in the declared variables list.
    # Two raw names like ``"A B"`` / ``"A\nB"`` collapse to the same
    # ``safe_key`` and the previous ``set(...)`` silently merged
    # them — leaving a matrix where one declared label represents
    # two source variables (and the corresponding rows / columns
    # were dropped or merged in the per-key collision counters
    # below). The result was an ``ok=True`` matrix that was
    # ambiguous from the model's seat. Reject loudly so the script
    # has to disambiguate at the source. Same posture as the
    # frequency_table collision check above.
    if len(sanitized_vars) != len(set(sanitized_vars)):
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                "two or more variable names sanitize to the same "
                "key (e.g. embedded newlines or shared 40-char "
                "prefix). The declared variables list is ambiguous; "
                "rename the source variables to disambiguate. The "
                "colliding names are withheld — they're data-derived."
            ),
        )
    declared = set(sanitized_vars)

    sanitized_corr: dict[str, dict[str, float]] = {}
    # Counters only — row/col keys are data-derived (variable names),
    # so the per-key sample previously emitted in this transformation
    # leaked names back to the model. The per-row store keeps the raw
    # payload for researcher audit; the model sees totals.
    dropped_row_count = 0
    dropped_col_count = 0
    collision_row_count = 0
    collision_col_count = 0
    n = out["n"]
    for raw_row_key, row_value in correlations.items():
        if not isinstance(raw_row_key, str):
            dropped_row_count += 1
            continue
        row_key = safe_key(raw_row_key)
        if row_key not in declared:
            dropped_row_count += 1
            continue
        if not isinstance(row_value, dict):
            dropped_row_count += 1
            continue
        # Reject row-key collisions outright — two raw row keys that
        # ``safe_key`` collapses to the same form would silently
        # overwrite, replacing an earlier correlation row with a
        # later one and reporting it under an ambiguous label.
        if row_key in sanitized_corr:
            collision_row_count += 1
            continue
        kept_row: dict[str, float] = {}
        for raw_col_key, val in row_value.items():
            if not isinstance(raw_col_key, str):
                dropped_col_count += 1
                continue
            col_key = safe_key(raw_col_key)
            if col_key not in declared:
                dropped_col_count += 1
                continue
            if not _is_finite_number(val):
                continue
            # Same collision check on the column axis: skip duplicates
            # rather than overwrite. Counter only, no name echo.
            if col_key in kept_row:
                collision_col_count += 1
                continue
            kept_row[col_key] = clamp_precision(float(val), n)
        if kept_row:
            sanitized_corr[row_key] = kept_row
    if dropped_row_count or dropped_col_count:
        transformations.append(
            f"dropped {dropped_row_count} undeclared row(s) and "
            f"{dropped_col_count} undeclared column entry(ies) from "
            f"correlations (names withheld)"
        )
    if collision_row_count or collision_col_count:
        transformations.append(
            f"dropped {collision_row_count} duplicate row key(s) and "
            f"{collision_col_count} duplicate column key(s) from "
            f"correlations after sanitization (colliding names "
            f"withheld)"
        )
    if not sanitized_corr:
        # Every entry got dropped. Returning ok=True with an empty
        # matrix is misleading — the model would think "the analysis
        # ran but produced no correlations" when the truth is "the
        # payload's keys didn't line up with the declared variables."
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                "correlations dict is empty after sanitization — every "
                "row/column key was either not in the declared "
                "``variables`` list or had a non-finite value. The "
                "payload likely has a variables/correlations mismatch."
            ),
        )

    # Aggregate invariants for a real correlation matrix. The
    # per-cell checks above (declared keys, finiteness, clip to
    # [-1, 1]) don't catch a matrix that's asymmetric, has
    # off-diagonal cells without their transpose partner, or has
    # diagonals != 1. A real ``df.corr()`` always produces these
    # invariants; an attacker emitting through generic
    # ``result(type="correlation_matrix", ...)`` to smuggle numeric
    # values is the only realistic origin of a violation.
    #
    # Reject the whole payload rather than the matrix alone: unlike
    # vcov (which sits alongside coefficients/SE/etc.), the
    # correlations field IS the payload, and a correlation_matrix
    # without correlations is meaningless.
    invariants_ok, reject_reason = _correlation_invariants_hold(
        sanitized_corr, declared,
    )
    if not invariants_ok:
        return SanitizerResult(
            ok=False, analysis_type="correlation_matrix",
            rejection_reason=(
                f"correlation matrix failed aggregate-invariant check: "
                f"{reject_reason}. Real correlation matrices are "
                f"symmetric with 1s on the diagonal; a violation here "
                f"means the payload didn't come from a ``df.corr()``-"
                f"shaped computation, which is required by the SDC "
                f"posture for this result type."
            ),
        )
    out["correlations"] = sanitized_corr

    sigfigs = sigfigs_for_n(n)
    transformations.append(
        f"clamped correlation values to {sigfigs} significant figures (n={n})"
    )

    # Coarsen rare ``missing_count``. The complete-case correlation
    # path can publish a single-row-missing count exactly, which
    # identifies the one observation that's incomplete on at least
    # one of the variables in the matrix — same disclosure shape
    # the schema-side ``request_data(na_count)`` gate already
    # closes.
    _coarsen_small_missing_count(out, transformations, config)

    return SanitizerResult(
        ok=True, analysis_type="correlation_matrix",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# DiD event study (Callaway-Sant'Anna / de Chaisemartin-D'Haultfœuille /
# Sun-Abraham / TWFE event study)
# ---------------------------------------------------------------------------
#
# The modern-DiD literature has moved decisively to heterogeneous-
# treatment estimators that decompose into ATT(g, t) — average
# treatment effect on the treated, indexed by treatment cohort g and
# event time t (calendar time relative to treatment). Callaway-
# Sant'Anna (the ``did`` R package), de Chaisemartin-
# D'Haultfœuille (``DIDmultiplegt``), and Sun-Abraham (the
# ``fixest::sunab`` interaction-weighted estimator) all produce
# this shape, plus the older TWFE event-study with leads-and-lags
# coefficients indexed by event time.
#
# **The new SDC primitive this shape introduces** is min-N gated by
# the treated-cohort size, NOT by the cell count of the ATT panel.
# Concrete reason: in strategy / finance / applied micro, treated
# cohorts of 3-10 firms are normal (mergers, IPOs, regulatory
# events). A balanced panel can make the cell count of ATT(g, t)
# look comfortable (4 firms × 8 quarters = 32 "observations") while
# the actual disclosure unit is those 4 firms whose outcome
# trajectories are summarized by the ATT series for cohort g.
# Combined with knowledge that the cohort was treated at calendar
# time T (often public), the ATT series leaks firm-level outcome
# changes if the cohort is small.
#
# Suppression rule: any cohort g with ``n_treated_per_group[g] <
# min_n_did`` gets ALL its cells dropped from ``att`` and the
# per-cell SE / p / CI dicts. Whole-cohort suppression is
# mandatory — partial-cell publication would leak the cohort size
# through *which* cells survived. The cohort label itself is also
# withheld (the marker key is ``[suppressed]``), since the
# cohort label is data-derived (it's typically the treatment date
# / cohort id and identifies the cohort directly).
#
# Cross-field validation: ``att`` is a nested {group: {event_time:
# value}} dict. Every outer key must be in ``groups``; every inner
# key must be in ``event_times``. ``standard_errors`` / ``p_values``
# / ``ci_lower`` / ``ci_upper`` mirror that shape and validate the
# same way. ``n_treated_per_group`` outer keys must equal ``groups``.

# Required structural fields. ``att`` and ``n_treated_per_group``
# are load-bearing — without the latter the cohort-N gate has no
# input and SDC degenerates to "trust the script". The aggregate
# ATT block is optional (a study might only report the matrix).
_DID_EVENT_REQUIRED: frozenset[str] = frozenset((
    "type", "groups", "event_times", "att", "n_treated_per_group",
))
_DID_EVENT_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "aggregate_att", "aggregate_se", "aggregate_p_value",
    "aggregate_ci_lower", "aggregate_ci_upper",
    "pre_trends_chi_squared", "pre_trends_p_value",
))
_DID_EVENT_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    # Anticipation window the estimator was told to assume. Callaway-
    # Sant'Anna's ``anticipation`` arg specifies how many periods
    # before treatment the treatment effect may "leak in" — shifting
    # which pre-periods are usable as controls. A scalar count, no
    # disclosure risk; surfaced so the model can report "...assuming
    # zero anticipation" or call out a non-default value.
    "anticipation_periods",
    "n_pre_treatment_periods", "n_post_treatment_periods",
))
_DID_EVENT_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "estimator", "outcome_variable", "treatment_variable",
    "aggregation_method",
    # Which units serve as the control group during the differencing
    # step. CS / dCdH let you pick:
    #   * ``nevertreated`` — only units that NEVER receive treatment.
    #     Stricter but loses data when the never-treated cohort is
    #     small or absent.
    #   * ``notyettreated`` — units that will be treated later are
    #     also valid controls until their own treatment date.
    # Pinned by ``_DID_VALID_COMPARISON_GROUP`` below.
    "comparison_group",
    # CS / dCdH ``base_period`` rule: which pre-treatment period
    # serves as the reference for each cohort. ``varying`` (the R
    # ``did`` package default) re-bases for each (g, t) pair to the
    # immediately-pre-treatment period. ``universal`` fixes the base
    # period across all (g, t) pairs to a single period.
    "base_period",
))
_DID_VALID_COMPARISON_GROUP: frozenset[str] = frozenset((
    "nevertreated", "notyettreated",
    # snake_case variants — accept both so the helper doesn't have
    # to choose between the R package's no-underscore form and the
    # more readable Python idiom.
    "never_treated", "not_yet_treated",
))
_DID_VALID_BASE_PERIOD: frozenset[str] = frozenset((
    "varying", "universal",
))
_DID_EVENT_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "groups",
))
_DID_EVENT_ALLOWED_LIST_NUMERIC: frozenset[str] = frozenset((
    "event_times",
))
# Nested-dict (group → event_time → value) fields. Each gets the
# same cohort suppression and cross-field validation pass.
_DID_EVENT_NESTED_DICT_FIELDS: frozenset[str] = frozenset((
    "att", "standard_errors", "p_values", "ci_lower", "ci_upper",
))
# Flat per-group dicts. ``n_treated_per_group`` is the SDC-relevant
# one (drives the cohort gate); ``n_control_per_cell`` is optional
# secondary metadata if the script computed cell-level control N.
_DID_EVENT_PER_GROUP_INT_FIELDS: frozenset[str] = frozenset((
    "n_treated_per_group",
))
# Structural caps on the panel dimensions. A real Callaway-Sant'Anna
# study reports a handful of cohorts (treatment-year cohorts in a
# DiD design) over a window of event times (typically ±5 to ±10).
# A 50-cohort × 30-event-time panel is already 1500 cells of
# disclosure surface; bigger numbers are almost always a sign the
# script is shipping disaggregated data through this channel.
_DID_EVENT_MAX_GROUPS: int = 50
_DID_EVENT_MAX_EVENT_TIMES: int = 30
_DID_EVENT_VALID_AGGREGATION: frozenset[str] = frozenset((
    "overall", "by_group", "by_event_time", "simple",
    "dynamic", "calendar",  # Callaway-Sant'Anna aggregator names
    # ``event`` is the Python ``differences`` package's name for the
    # same event-time aggregation R's ``did`` calls ``dynamic``.
    # Accept both so the helper doesn't have to normalize.
    "event", "group",
))
_DID_EVENT_VALID_ESTIMATOR: frozenset[str] = frozenset((
    "callaway_santanna", "de_chaisemartin", "sun_abraham",
    "twfe_event_study", "twfe",
))


def _sanitize_did_event_study(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _DID_EVENT_REQUIRED, "did_event_study")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=missing_reason,
        )

    # Validate groups list shape and cap before anything else — the
    # cohort identifiers gate the cross-field key validation below.
    groups_raw = raw.get("groups")
    if not isinstance(groups_raw, list) or not all(
        isinstance(x, str) for x in groups_raw
    ):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "groups must be a list of strings (cohort identifiers); "
                f"got {type(groups_raw).__name__}"
            ),
        )
    if len(groups_raw) > _DID_EVENT_MAX_GROUPS:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"groups has {len(groups_raw)} cohorts; the structural "
                f"cap is {_DID_EVENT_MAX_GROUPS}. A real Callaway-Sant'Anna "
                f"/ event-study analysis ships a handful of cohorts; "
                f"larger payloads are rejected as probable adversarial."
            ),
        )
    if len(groups_raw) == 0:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason="groups is empty",
        )

    # Validate event_times — list of finite numbers (typically ints
    # but allow floats; sanitize to int when integer-valued for nice
    # JSON, otherwise keep float).
    event_times_raw = raw.get("event_times")
    if not isinstance(event_times_raw, list) or not all(
        _is_finite_number(x) for x in event_times_raw
    ):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "event_times must be a list of finite numbers; "
                f"got {type(event_times_raw).__name__}"
            ),
        )
    if len(event_times_raw) > _DID_EVENT_MAX_EVENT_TIMES:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"event_times has {len(event_times_raw)} entries; the "
                f"structural cap is {_DID_EVENT_MAX_EVENT_TIMES}."
            ),
        )

    # n_treated_per_group: required, dict[str, int]. This is the
    # SDC primitive's input — every cohort must declare its treated
    # size or the cohort-N gate can't run.
    n_treated_raw = raw.get("n_treated_per_group")
    if not isinstance(n_treated_raw, dict):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "n_treated_per_group must be a dict mapping cohort id "
                "to treated-unit count; "
                f"got {type(n_treated_raw).__name__}"
            ),
        )

    transformations: list[str] = []

    # Sanitize group labels and build the declared-cohort set after
    # safe_key normalization. Reject safe_key collisions outright
    # mirrors the magnitude_table / crosstab pattern.
    safe_groups: list[str] = []
    safe_groups_set: set[str] = set()
    for raw_g in groups_raw:
        sg = safe_key(raw_g)
        if sg in safe_groups_set:
            return SanitizerResult(
                ok=False, analysis_type="did_event_study",
                rejection_reason=(
                    "cohort label collision after sanitization in "
                    "did_event_study.groups: two distinct raw labels "
                    "sanitize to the same safe_key, which would silently "
                    "overwrite ATT entries. Disambiguate labels in the "
                    "source script. Colliding labels withheld."
                ),
            )
        safe_groups_set.add(sg)
        safe_groups.append(sg)

    # Sanitize event_time labels. Use a string form (so JSON keys are
    # stable: "-3", "-2", ...). Strip non-finite; coerce ints to int.
    safe_event_times: list[Any] = []
    safe_event_times_str_set: set[str] = set()
    for raw_t in event_times_raw:
        t = float(raw_t)
        if not math.isfinite(t):
            continue
        if t == int(t):
            t_norm: Any = int(t)
        else:
            t_norm = t
        safe_event_times.append(t_norm)
        safe_event_times_str_set.add(str(t_norm))
    if not safe_event_times:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason="event_times is empty",
        )
    if len(safe_event_times) != len(safe_event_times_str_set):
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "event_times contains duplicate values after numeric "
                "normalization; event-time keys would be ambiguous"
            ),
        )

    # Apply the cohort-N gate. Map safe-group → treated count; drop
    # the entire cohort when count < threshold (using the same
    # threshold as descriptive's min_n for consistency; the SDC
    # config has a single ``min_n`` that drives all suppression).
    cohort_min_n = config.min_n_did_cohort
    suppressed_cohorts: set[str] = set()
    cleaned_n_treated: dict[str, int] = {}

    for raw_g_key, count in n_treated_raw.items():
        if not isinstance(raw_g_key, str):
            continue
        sg = safe_key(raw_g_key)
        if sg not in safe_groups_set:
            # Cohort key not in declared groups — drop silently,
            # don't name it (would echo a data-derived label).
            continue
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            return SanitizerResult(
                ok=False, analysis_type="did_event_study",
                rejection_reason=(
                    "n_treated_per_group values must be non-negative "
                    f"ints; got {type(count).__name__} for one entry "
                    "(cohort label withheld)"
                ),
            )
        if count < cohort_min_n:
            suppressed_cohorts.add(sg)
        else:
            cleaned_n_treated[sg] = count

    # Every declared cohort must have a count entry. A cohort named
    # in ``groups`` but missing from ``n_treated_per_group`` would
    # bypass the gate — reject the payload rather than guess.
    declared_safe_in_n = {
        safe_key(k) for k in n_treated_raw.keys()
        if isinstance(k, str) and safe_key(k) in safe_groups_set
    }
    missing_n = [g for g in safe_groups if g not in declared_safe_in_n]
    if missing_n:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"{len(missing_n)} declared cohort(s) have no "
                f"n_treated_per_group entry. The cohort-N gate cannot "
                f"run without per-cohort sizes. Cohort labels withheld."
            ),
        )

    surviving_cohorts: set[str] = safe_groups_set - suppressed_cohorts
    if suppressed_cohorts:
        transformations.append(
            f"cohort suppression: {len(suppressed_cohorts)} cohort(s) "
            f"with n_treated < {cohort_min_n} dropped entirely (labels "
            f"withheld — cohort identities are disclosive)"
        )

    if not surviving_cohorts:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"all cohorts have n_treated < {cohort_min_n}; nothing "
                f"survives the cohort-N gate. No ATT panel published."
            ),
        )

    # Build the output. Top-level allowed fields first.
    out: dict[str, Any] = _collect_allowed(
        raw,
        numeric=_DID_EVENT_ALLOWED_NUMERIC_FIELDS,
        integer=_DID_EVENT_ALLOWED_INT_FIELDS,
        string=_DID_EVENT_ALLOWED_STRING_FIELDS,
        list_string=_DID_EVENT_ALLOWED_LIST_STRING,
        list_numeric=_DID_EVENT_ALLOWED_LIST_NUMERIC,
        transformations=transformations,
    )

    # Identifier-shape gate on the two column-name-bearing string
    # fields -- same mitigation every other analysis shape applies to
    # its variable-name fields (see ``_is_identifier_shape``'s module-
    # header rationale). ``outcome_variable``/``treatment_variable``
    # only ever went through ``safe_text``'s 120-char control-strip
    # here, unlike ``response_variable``/``predictor_variables`` in
    # the OLS shape, which is a real gap this closes.
    _enforce_identifier_string_fields(
        out, frozenset(("outcome_variable", "treatment_variable")),
        transformations, type_label="did_event_study",
    )

    # Validate string-enum fields (aggregation_method, estimator,
    # comparison_group, base_period).
    aggm = out.get("aggregation_method")
    if aggm is not None and aggm not in _DID_EVENT_VALID_AGGREGATION:
        transformations.append(
            f"dropped 'aggregation_method' value (not in valid set)"
        )
        del out["aggregation_method"]
    est = out.get("estimator")
    if est is not None and est not in _DID_EVENT_VALID_ESTIMATOR:
        transformations.append(
            f"dropped 'estimator' value (not in valid set)"
        )
        del out["estimator"]
    cmp_group = out.get("comparison_group")
    if cmp_group is not None and cmp_group not in _DID_VALID_COMPARISON_GROUP:
        transformations.append(
            "dropped 'comparison_group' value (must be one of "
            "nevertreated / notyettreated / never_treated / not_yet_treated)"
        )
        del out["comparison_group"]
    bperiod = out.get("base_period")
    if bperiod is not None and bperiod not in _DID_VALID_BASE_PERIOD:
        transformations.append(
            "dropped 'base_period' value (must be 'varying' or 'universal')"
        )
        del out["base_period"]

    # Re-place the sanitized identifier lists (use safe forms).
    out["groups"] = sorted(surviving_cohorts)
    out["event_times"] = safe_event_times
    out["n_treated_per_group"] = cleaned_n_treated

    # Process each nested dict-of-dict field (att / SE / p / CI
    # lower+upper). Apply: (a) outer key must be a surviving cohort,
    # (b) inner key must be in event_times, (c) values finite, then
    # precision-clamp by EACH COHORT'S OWN treated N -- not the
    # study-wide total. The disclosure unit for a cohort's ATT series
    # is that cohort's own treated units (a cohort with n_treated=12
    # sitting in a study whose OTHER cohorts are much larger must not
    # get its smallest, most disclosure-sensitive series published at
    # the precision the LARGE cohorts would justify). This mirrors
    # ``cluster_analysis``'s ``clamp_dict_by_per_key_n`` treatment of
    # the exact same per-group-N-varies shape. The aggregate top-level
    # scalars below (aggregate_att / pre_trends_* etc.) legitimately
    # summarize across every cohort, so THOSE still clamp by the
    # aggregate total -- only the per-cohort nested cells change here.
    total_treated_n = sum(cleaned_n_treated.values())
    sigfigs_n = total_treated_n if total_treated_n > 0 else cohort_min_n

    for field in _DID_EVENT_NESTED_DICT_FIELDS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected nested dict, "
                f"got {type(v).__name__}"
            )
            continue
        cleaned: dict[str, dict[str, float]] = {}
        dropped_outer = 0
        dropped_inner = 0
        for outer_k, inner_v in v.items():
            if not isinstance(outer_k, str):
                dropped_outer += 1
                continue
            sg = safe_key(outer_k)
            if sg not in surviving_cohorts:
                # Either the cohort was suppressed or it's not in
                # ``groups`` at all. Either way: drop, don't name.
                dropped_outer += 1
                continue
            if not isinstance(inner_v, dict):
                dropped_outer += 1
                continue
            cleaned_inner: dict[str, float] = {}
            for inner_k, cell_v in inner_v.items():
                # Inner key may be str or int (event times); normalize
                # to str-form to match safe_event_times_str_set.
                k_str = str(inner_k)
                if k_str not in safe_event_times_str_set:
                    dropped_inner += 1
                    continue
                if not _is_finite_number(cell_v):
                    dropped_inner += 1
                    continue
                # Per-cohort sigfigs: this cohort's own treated N,
                # never the study-wide aggregate. ``sg`` is guaranteed
                # present in ``cleaned_n_treated`` here -- it's a
                # member of ``surviving_cohorts``, and every surviving
                # cohort has a ``cleaned_n_treated`` entry by
                # construction (that's exactly what "surviving"
                # means: it passed the ``cohort_min_n`` gate above).
                cleaned_inner[k_str] = clamp_precision(
                    float(cell_v), cleaned_n_treated[sg]
                )
            if cleaned_inner:
                cleaned[sg] = cleaned_inner
        if dropped_outer:
            transformations.append(
                f"dropped {dropped_outer} undeclared/suppressed outer "
                f"key(s) from {field!r} (cohort labels withheld)"
            )
        if dropped_inner:
            transformations.append(
                f"dropped {dropped_inner} undeclared event-time key(s) "
                f"from {field!r}"
            )
        out[field] = cleaned

    # ``att`` is the result itself, not optional metadata.  A declared
    # cohort whose entire ATT row disappears after key/finiteness filtering
    # would otherwise produce an ok payload that silently misrepresents an
    # incomplete event study as a complete one.
    att = out.get("att")
    if not isinstance(att, dict) or not att:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                "att is empty after sanitization; no declared cohort/event-"
                "time estimate survived"
            ),
        )
    missing_att_rows = surviving_cohorts - set(att)
    if missing_att_rows:
        return SanitizerResult(
            ok=False, analysis_type="did_event_study",
            rejection_reason=(
                f"{len(missing_att_rows)} surviving cohort(s) have no ATT "
                "entries after sanitization; cohort labels withheld"
            ),
        )

    # Aggregate-att scalars: precision-clamp at total-treated-N.
    for key in _DID_EVENT_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], sigfigs_n)

    transformations.append(
        f"clamped numeric fields to precision matching total treated "
        f"n={sigfigs_n}"
    )

    return SanitizerResult(
        ok=True, analysis_type="did_event_study",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Regression discontinuity design (RDD)
# ---------------------------------------------------------------------------
#
# The RDD shape ships the local-polynomial point estimate(s) plus
# bandwidth and effective-N diagnostics — what the model needs to
# evaluate an RDD design. The Calonico-Cattaneo-Titiunik (rdrobust)
# convention reports three flavors of τ at one fit: the conventional
# local-polynomial estimate, the bias-corrected variant, and the
# robust variant whose standard error accounts for bias-correction
# noise. The model sees all three so it can report the standard
# table conventional / bc / robust users expect.
#
# **Privacy carve-out, made structural via the allowlist:**
# Two RDD diagnostics are deliberately excluded from the shape:
#
#   1.  The **McCrary density test**'s estimated density curve.
#       The test statistic itself is a single scalar (log-discontinuity
#       in density at the cutoff); the *curve* — density evaluated at
#       a grid of points around the cutoff — is essentially a
#       histogram of the running variable in the most identifying
#       region (a few bandwidths either side of c). The running
#       variable in an RDD is by construction sensitive: income at a
#       tax-credit cutoff, test score at an admissions threshold,
#       date-of-birth at a school-entry cutoff. Surfacing the density
#       curve to the model would invert the privacy claim ("the
#       model never sees a raw cell value") on the exact slice where
#       individual identification is most likely. Even the bare
#       statistic at the cutoff has a cutoff-scan attack: re-run at
#       placebo cutoffs c±δ and the sequence of statistics maps the
#       density. We exclude the test ENTIRELY from this shape.
#
#   2.  **Binscatter near the cutoff** — by construction, bins shrink
#       toward the cutoff to make the discontinuity visible. Small
#       bins mean cells of small N over the most sensitive variable
#       slice. Same disclosure surface as McCrary's curve.
#
# Both are researcher-only by construction — they have no field in
# the ``rdd`` allowlist below. A script can still produce them
# visually for the researcher (the executor's raw-log panel and the
# helper-error JSONL surface stay intact), but no path through the
# sanitizer carries them to the model. This matches the helper-
# allowlist precedent set for plot vision (the only paths to the
# model run through ``plot_residuals`` / ``plot_interaction`` /
# ``plot_coefficients`` / ``plot_estimate_comparison`` — bespoke
# plots stay researcher-only).
#
# Binscatter AWAY from cutoffs, with min-N-per-bin guarantees, fits
# the existing ``magnitude_table`` shape and can ship through that
# channel. The exclusion here is specifically for the cutoff-
# proximity case where binwidths shrink by construction.

_RDD_REQUIRED: frozenset[str] = frozenset((
    "type", "running_variable", "cutoff",
    "tau_robust", "se_robust",
    "effective_n_left", "effective_n_right",
))
_RDD_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # CCT three-flavor estimates
    "tau_conventional", "tau_bias_corrected", "tau_robust",
    "se_conventional", "se_bias_corrected", "se_robust",
    "p_conventional", "p_bias_corrected", "p_robust",
    "ci_lower_conventional", "ci_upper_conventional",
    "ci_lower_bias_corrected", "ci_upper_bias_corrected",
    "ci_lower_robust", "ci_upper_robust",
    # Cutoff (the threshold value) — a researcher-chosen constant,
    # not a data-derived quantity. Surfaced so the model can echo
    # "discontinuity at age 65 = ..." rather than guessing.
    "cutoff",
    # Bandwidth(s). Left/right bandwidth differ when the optimal
    # MSE-minimizing bandwidth is computed separately on each side
    # (rdrobust's default).
    "bandwidth_left", "bandwidth_right",
    "bandwidth_bias_correction_left", "bandwidth_bias_correction_right",
    # Fuzzy-RDD diagnostic: first-stage F-statistic for joint
    # significance of the cutoff dummy in the first-stage regression
    # of the endogenous-treatment indicator on the running variable.
    # Below ~10 flags a weak first-stage and renders the Wald-ratio
    # τ unstable. Same primitive as IV's ``first_stage_f``; the
    # field name is duplicated here so RDD payloads don't have to
    # route through the regression-bucket schema. Whether the fit
    # is sharp or fuzzy is communicated via the ``estimator`` enum
    # below (``fuzzy_2sls`` vs ``local_polynomial``).
    "first_stage_f",
))
_RDD_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    # Effective N inside the bandwidth window — the SDC-relevant
    # quantity. RDD inference is local; the effective sample sizes
    # are what bound how tightly the local fit can identify
    # individuals. Required, so the min-N gate has its inputs.
    "effective_n_left", "effective_n_right", "effective_n_total",
    "polynomial_order",
))
_RDD_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "estimator", "running_variable", "outcome_variable",
    "kernel",
    # rdrobust's bandwidth-selection rule. Each is a documented CCT /
    # CER selector with different optimality criteria:
    #   * ``mserd``: single MSE-optimal bandwidth, same on both sides.
    #   * ``msetwo``: MSE-optimal bandwidth selected separately per side.
    #   * ``msesum`` / ``msecomb1`` / ``msecomb2``: MSE variants.
    #   * ``cerrd`` / ``certwo`` / ``cercomb1`` / ``cercomb2``: coverage-
    #     error-rate (CER) optimal — narrower bandwidth, lower bias,
    #     wider CIs.
    #   * ``manual``: caller-supplied bandwidth (no automatic selection).
    # The validation set below pins these as the only legal values; the
    # field passes ``safe_text`` regardless, but limiting to known
    # selectors prevents a script from smuggling free-text into the
    # model via this slot.
    "bandwidth_selector",
))
_RDD_VALID_ESTIMATOR: frozenset[str] = frozenset((
    "local_polynomial", "sharp_parametric", "fuzzy_2sls",
    "rdrobust", "rdlocrand",
))
_RDD_VALID_KERNEL: frozenset[str] = frozenset((
    "triangular", "uniform", "epanechnikov",
))
_RDD_VALID_BANDWIDTH_SELECTOR: frozenset[str] = frozenset((
    "mserd", "msetwo", "msesum", "msecomb1", "msecomb2",
    "cerrd", "certwo", "cercomb1", "cercomb2",
    "manual",
))


def _sanitize_rdd(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _RDD_REQUIRED, "rdd")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="rdd",
            rejection_reason=missing_reason,
        )

    # Effective-N gate. Local-polynomial RDD identifies τ from
    # observations inside the bandwidth on each side of the cutoff;
    # the local sample sizes must each pass the min-N threshold or
    # the inference isn't trustworthy. Apply per-side, not just to
    # the total — a 50/2 split can hit total ≥ threshold while one
    # side has unbounded uncertainty.
    for side_field in ("effective_n_left", "effective_n_right"):
        v = raw.get(side_field)
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return SanitizerResult(
                ok=False, analysis_type="rdd",
                rejection_reason=(
                    f"{side_field} must be a non-negative int; "
                    f"got {type(v).__name__}"
                ),
            )
        try:
            require_minimum_n(v, config.min_n_regression, side_field)
        except MinimumNViolation as e:
            return SanitizerResult(
                ok=False, analysis_type="rdd",
                rejection_reason=str(e),
            )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_RDD_ALLOWED_NUMERIC_FIELDS,
        integer=_RDD_ALLOWED_INT_FIELDS,
        string=_RDD_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    # Validate enum string fields.
    est = out.get("estimator")
    if est is not None and est not in _RDD_VALID_ESTIMATOR:
        transformations.append(
            "dropped 'estimator' value (not in valid set)"
        )
        del out["estimator"]
    krn = out.get("kernel")
    if krn is not None and krn not in _RDD_VALID_KERNEL:
        transformations.append(
            "dropped 'kernel' value (not in valid set)"
        )
        del out["kernel"]
    bwsel = out.get("bandwidth_selector")
    if bwsel is not None and bwsel not in _RDD_VALID_BANDWIDTH_SELECTOR:
        transformations.append(
            "dropped 'bandwidth_selector' value (not in valid set — "
            "must be one of mserd / msetwo / msesum / msecomb1 / "
            "msecomb2 / cerrd / certwo / cercomb1 / cercomb2 / manual)"
        )
        del out["bandwidth_selector"]
    po = out.get("polynomial_order")
    if po is not None:
        if po < 0 or po > 4:
            transformations.append(
                "dropped 'polynomial_order' value (must be 0..4; "
                "local-polynomial RDD with degree > 4 is suspect)"
            )
            del out["polynomial_order"]

    # ``running_variable`` is identifier-shape gated. ``rejection_reason``
    # withholds bad names — the running variable is data-derived (the
    # column name the researcher chose).
    _enforce_identifier_string_fields(
        out, frozenset(("running_variable", "outcome_variable")),
        transformations, type_label="rdd",
    )

    # Re-check required after type filtering.
    missing_after = _require_after_filter(
        out, _RDD_REQUIRED, "rdd",
        pre_validated=frozenset(("effective_n_left", "effective_n_right")),
    )
    if missing_after:
        return SanitizerResult(
            ok=False, analysis_type="rdd",
            rejection_reason=missing_after,
        )

    # Precision clamp by total effective N (sum of sides if total
    # absent). Total drives the precision of τ; per-side N drives
    # the per-side gate already enforced above.
    # Total effective N is a derived invariant, never a caller-controlled
    # precision knob.  Trusting an inflated supplied total increased the
    # significant figures released for every RDD statistic and let a
    # malformed payload contradict its own left/right counts.
    derived_n_total = out["effective_n_left"] + out["effective_n_right"]
    supplied_n_total = out.get("effective_n_total")
    if supplied_n_total is not None and supplied_n_total != derived_n_total:
        return SanitizerResult(
            ok=False, analysis_type="rdd",
            rejection_reason=(
                "effective_n_total must equal effective_n_left + "
                "effective_n_right"
            ),
        )
    n_total = derived_n_total
    out["effective_n_total"] = n_total
    for key in _RDD_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_total)

    transformations.append(
        f"clamped numeric fields to precision matching effective n={n_total}"
    )

    return SanitizerResult(
        ok=True, analysis_type="rdd",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Kaplan-Meier (safe-form: scalars at preset horizons, no curve)
# ---------------------------------------------------------------------------
#
# The full KM step function — survival probability at every observed
# event time — is too granular near small risk sets. With small
# n_at_risk, the survival drop from a single event identifies that
# event's timing and (combined with covariate distribution) the
# individual. The shape published here is the *safe form* described
# in ``docs/architecture.md``: median survival with CI, plus survival
# at a small set of preset horizons (e.g., 1y, 3y, 5y) each gated by
# its own n_at_risk threshold. Dedicated shape, not a coefficient
# table sub-type, because the cross-field invariant is different
# (per-horizon N gate, not per-coefficient name match).
#
# The curve itself (per-event-time S(t) values, the Greenwood SE
# series, KM-by-group log-rank chi²) is researcher-only by
# construction — it has no field in this allowlist. Same exclusion
# pattern as McCrary in the RDD shape: structurally absent from the
# allowlist means it can't be smuggled through the generic
# ``result(type="kaplan_meier", ...)`` path.

_KM_REQUIRED: frozenset[str] = frozenset((
    "type", "time_variable", "event_variable",
    "n_subjects", "n_failures",
))
_KM_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    "median_survival_time", "median_survival_ci_lower",
    "median_survival_ci_upper",
    # Pre-specified horizon survival probabilities. We allowlist a
    # small fixed set — enough to cover the conventional 1y / 3y /
    # 5y reporting plus a couple extra — rather than letting the
    # caller name arbitrary horizons (which would be a covert-channel
    # surface for raw time values).
    "survival_at_1y", "survival_at_3y", "survival_at_5y",
    "survival_at_10y",
    "survival_at_1y_ci_lower", "survival_at_1y_ci_upper",
    "survival_at_3y_ci_lower", "survival_at_3y_ci_upper",
    "survival_at_5y_ci_lower", "survival_at_5y_ci_upper",
    "survival_at_10y_ci_lower", "survival_at_10y_ci_upper",
    # Log-rank omnibus across groups (when KM-by-group requested).
    "logrank_chi_squared", "logrank_p_value",
))
_KM_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n_subjects", "n_failures",
    # n_at_risk at each pre-specified horizon — drives the per-
    # horizon gate. Each horizon needs ≥ min_n_regression at-risk
    # subjects or its S(t) is dropped.
    "n_at_risk_1y", "n_at_risk_3y", "n_at_risk_5y", "n_at_risk_10y",
    # Group count for log-rank-by-group setups.
    "n_groups",
))
_KM_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "time_variable", "event_variable", "group_variable",
))


def _sanitize_kaplan_meier(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _KM_REQUIRED, "kaplan_meier")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=missing_reason,
        )

    n_sub = raw.get("n_subjects")
    if not isinstance(n_sub, int) or isinstance(n_sub, bool) or n_sub < 0:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=(
                f"n_subjects must be a non-negative int; "
                f"got {type(n_sub).__name__}"
            ),
        )
    try:
        require_minimum_n(n_sub, config.min_n_regression, "n_subjects")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=str(e),
        )

    n_fail = raw.get("n_failures")
    if not isinstance(n_fail, int) or isinstance(n_fail, bool) or n_fail < 0:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=(
                f"n_failures must be a non-negative int; "
                f"got {type(n_fail).__name__}"
            ),
        )
    if n_fail > n_sub:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=(
                f"n_failures ({n_fail}) cannot exceed n_subjects ({n_sub})"
            ),
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_KM_ALLOWED_NUMERIC_FIELDS,
        integer=_KM_ALLOWED_INT_FIELDS,
        string=_KM_ALLOWED_STRING_FIELDS,
        transformations=transformations,
    )

    _enforce_identifier_string_fields(
        out, frozenset(("time_variable", "event_variable", "group_variable")),
        transformations, type_label="kaplan_meier",
    )

    missing_after = _require_after_filter(
        out, _KM_REQUIRED, "kaplan_meier",
        pre_validated=frozenset(("n_subjects", "n_failures")),
    )
    if missing_after:
        return SanitizerResult(
            ok=False, analysis_type="kaplan_meier",
            rejection_reason=missing_after,
        )

    # Per-horizon n_at_risk gate. For each horizon h whose S(h) field
    # is populated, the corresponding n_at_risk_h must be present
    # and pass min_n_regression. If the gate fails, drop the S(h)
    # AND its CI bounds — partial publication leaks at-risk count
    # through "this horizon survives, that one doesn't".
    horizons = ("1y", "3y", "5y", "10y")
    for h in horizons:
        s_field = f"survival_at_{h}"
        if s_field not in out:
            continue
        n_risk_field = f"n_at_risk_{h}"
        n_risk = out.get(n_risk_field)
        if (n_risk is None
            or not isinstance(n_risk, int)
            or n_risk < config.min_n_regression):
            # Drop this horizon's S(h), its CI bounds, AND the
            # n_at_risk count that triggered the drop -- together.
            # Publishing survival_at_h=None while leaving
            # n_at_risk_h=3 on the payload doesn't withhold anything:
            # the exact tiny at-risk count is itself the disclosive
            # quantity this gate exists to protect (see the module
            # comment above), and it would still be visible to
            # anyone reading the sanitized payload even with S(h)
            # gone.
            dropped_fields = [s_field, n_risk_field]
            for suffix in ("_ci_lower", "_ci_upper"):
                key = s_field + suffix
                if key in out:
                    dropped_fields.append(key)
                    del out[key]
            del out[s_field]
            if n_risk_field in out:
                del out[n_risk_field]
            transformations.append(
                f"dropped horizon {h}: n_at_risk_{h} below "
                f"min_n_regression ({config.min_n_regression}) "
                f"or absent"
            )

    # Coarsen rare event/subject counts on the same rule the Cox path
    # uses (see ``_coarsen_small_cox_counts``). KM and Cox carry the
    # same disclosure surface here: "324 subjects, 3 deaths" identifies
    # those 3 individuals regardless of which estimator produced the
    # payload. The ``n_subjects`` branch is effectively a no-op for
    # well-formed KM (the upstream ``require_minimum_n`` gate rejects
    # n_subjects below ``min_n_regression``, typically ≥
    # ``cell_suppression_threshold``), but is kept in the call so the
    # Cox / KM behaviour stays symmetric if the two thresholds ever
    # diverge.
    _coarsen_small_cox_counts(out, transformations, config)

    # Precision clamp. Two different disclosure units apply here, not
    # one: ``survival_at_{h}`` (and its CI bounds) is a HORIZON-
    # specific statistic whose real disclosure unit is that horizon's
    # own at-risk set (``n_at_risk_{h}``) -- the horizon-drop gate
    # just above this exists precisely because "with small n_at_risk,
    # the survival drop from a single event identifies that event's
    # timing" (see that gate's own comment). KM risk sets shrink with
    # follow-up time from censoring/events, so n_at_risk_{h} for a
    # surviving horizon can be far smaller than the enrolled cohort
    # n_subjects -- clamping a horizon's survival probability by the
    # global n_subjects would publish it at a precision the horizon's
    # own, much smaller, at-risk count doesn't justify. Every other
    # field here (median survival, the log-rank omnibus test) is a
    # genuine whole-sample statistic, so those still clamp by n_sub.
    horizon_numeric_fields = frozenset(
        f"survival_at_{h}{suffix}"
        for h in horizons
        for suffix in ("", "_ci_lower", "_ci_upper")
    )
    for key in _KM_ALLOWED_NUMERIC_FIELDS:
        if key not in out:
            continue
        if key in horizon_numeric_fields:
            # key looks like "survival_at_{h}" or
            # "survival_at_{h}_ci_lower"/"_ci_upper" -- recover h and
            # read that horizon's own at-risk count. Guaranteed
            # present: the horizon-drop gate above already deleted
            # every survival_at_{h}* field whose n_at_risk_{h} was
            # missing or below min_n_regression before this point.
            body = key[len("survival_at_"):]
            h = body.split("_", 1)[0]
            n_risk = out.get(f"n_at_risk_{h}")
            clamp_n = n_risk if isinstance(n_risk, int) and n_risk > 0 else n_sub
            out[key] = clamp_precision(out[key], clamp_n)
        else:
            out[key] = clamp_precision(out[key], n_sub)
    transformations.append(
        f"clamped whole-sample numeric fields to precision matching "
        f"n_subjects={n_sub}; clamped each surviving horizon's "
        f"survival_at_* fields to precision matching that horizon's "
        f"own n_at_risk"
    )

    return SanitizerResult(
        ok=True, analysis_type="kaplan_meier",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Marginal effects — per-variable AME / MEM / at-representative
# ---------------------------------------------------------------------------
#
# Distinct from the regression bucket: marginal effects are scalars
# of interest *derived* from a fitted model rather than the model's
# raw coefficients. For a non-linear estimator (logit, probit,
# Poisson, mixed-effects with non-identity link), the coefficient is
# on the link scale; the model wants the marginal effect on the
# response scale to interpret magnitude. Methods:
#
#   * **AME (average marginal effect)**: average of ∂E[y|x]/∂x_j
#     across the sample. The de facto default for applied work; it
#     reports the typical effect under the observed covariate
#     distribution.
#   * **MEM (marginal effect at the means)**: ∂E[y|x]/∂x_j evaluated
#     at the sample mean covariate vector. Cheaper to compute and
#     interpret but less honest when covariates have heavy-tailed
#     distributions.
#   * **At representative values**: evaluated at a caller-specified
#     covariate vector (e.g. "treatment effect for a 45-year-old
#     female"). The representative values are researcher-chosen
#     constants — they ride alongside the effects as ``at_values``
#     so the model knows the conditioning point.
#
# Wire shape: per-variable scalars in flat dicts keyed by variable
# name. Cross-field validation pins every dict's keys to the
# declared ``variables`` list — same defense as the regression
# bucket's coefficient-key gate. Privacy: the per-variable AME /
# MEM / SE / p / CI are pure aggregates over the fitted model and
# the sample's covariate distribution; no per-observation leak.
# Required ``n`` drives the min-N gate; precision clamps by ``n``.
#
# Helper coverage at v0: R via ``marginaleffects::avg_slopes`` (the
# actively-maintained successor to ``margins``) and Python via
# ``statsmodels`` ``get_margeff()``. Stata's ``margins`` covers the
# same surface but needs ``e()`` post-estimation parsing; deferral
# pattern matching the other Stata gaps.

_ME_REQUIRED: frozenset[str] = frozenset((
    "type", "n", "method", "variables", "effects",
))

_ME_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # No scalar floats at top level today. Per-variable scalars all
    # ride through the dict-numeric slots below. Reserved for a
    # future joint-test scalar (Wald χ² across all marginal effects)
    # if researcher demand surfaces it.
))
_ME_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n",
))
_ME_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "method",
    # Names of the underlying fit's response + family for context.
    # ``outcome_variable`` is the dependent variable; ``model_family``
    # is the estimator that produced the fit (``logit`` / ``probit``
    # / ``poisson`` / ``ols`` / ``glm`` / …). Both are short
    # identifiers the model needs to interpret the marginal effect
    # scale (probability change for logit, count change for Poisson,
    # etc.).
    "outcome_variable", "model_family",
))
_ME_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "variables",
))
_ME_ALLOWED_DICT_NUMERIC: frozenset[str] = frozenset((
    "effects",
    "standard_errors",
    "z_statistics",
    "p_values",
    "ci_lower", "ci_upper",
    # Representative-values dict for ``method="at_representative"``.
    # Keys must be in ``variables``; values are the covariate vector
    # the marginal effect was evaluated at.
    #
    # **Disclosure threat.** ``at_values`` is script-controlled, so a
    # researcher (or prompt-injected helper call) could in principle
    # pass an exact-precision value pulled from a single row
    # (``income=847239`` identifies the one observation with that
    # income; combined with the other conditioning variables it pins
    # the individual). Naïve pass-through would let a script publish
    # near-identifiers under the legitimate-looking
    # ``method="at_representative"`` slot.
    #
    # **Structural rule.** Each at_values entry is precision-clamped
    # by the sample N — the same ``clamp_precision_dict`` pass the
    # other dict-numeric fields use, gated by ``sigfigs_for_n(n)``.
    # At n=1000 that's 4 sigfigs; at n=100 it drops to 3. The
    # clamp transformation lands in the log so the model and the
    # researcher both see "conditioned at income=847,200 (clamped
    # from 847,239)" — they can decide if the conditioning point
    # is still meaningful at that precision. Higher-precision
    # caller-supplied values are rounded; they cannot survive as
    # raw bytes through this slot.
    #
    # **Researcher-side guidance** (system prompt and docstrings):
    # pass interpretable summary points — means, medians, percentiles,
    # round reference values from the literature. Don't pass
    # exact-value rows from individual observations. The helper
    # signature on both languages stays a thin pass-through; the
    # disclosure floor is enforced here, structurally, by the
    # precision clamp.
    "at_values",
))

_ME_VALID_METHODS: frozenset[str] = frozenset((
    "ame", "mem", "at_representative",
))

# Structural cap. A real marginal-effects table on a paper-grade
# regression covers a handful of focal variables, not the whole
# design — same envelope as the regression bucket's predictor cap
# (50) for consistency. Bigger payloads are almost always engineered.
_ME_MAX_VARIABLES = 50


def _sanitize_marginal_effects(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _ME_REQUIRED, "marginal_effects")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=missing_reason,
        )

    n_raw = raw.get("n")
    if not isinstance(n_raw, int) or isinstance(n_raw, bool) or n_raw < 0:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=(
                f"n must be a non-negative int, got {type(n_raw).__name__}"
            ),
        )
    try:
        require_minimum_n(n_raw, config.min_n_regression, "n")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=str(e),
        )

    method = raw.get("method")
    if not isinstance(method, str) or method not in _ME_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=(
                f"method must be one of {sorted(_ME_VALID_METHODS)}; "
                f"got {method!r}"
            ),
        )

    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not all(
        isinstance(v, str) for v in raw_vars
    ):
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason="variables must be a list of strings",
        )
    if len(raw_vars) == 0:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason="variables list is empty",
        )
    if len(raw_vars) > _ME_MAX_VARIABLES:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; the structural "
                f"cap is {_ME_MAX_VARIABLES}. A marginal-effects table "
                f"with that many entries isn't interpretable output."
            ),
        )

    transformations: list[str] = []
    out = _collect_allowed(
        raw,
        numeric=_ME_ALLOWED_NUMERIC_FIELDS,
        integer=_ME_ALLOWED_INT_FIELDS,
        string=_ME_ALLOWED_STRING_FIELDS,
        dict_numeric=_ME_ALLOWED_DICT_NUMERIC,
        list_string=_ME_ALLOWED_LIST_STRING,
        transformations=transformations,
    )

    # ``effects`` is required-after-filter — if it shipped as
    # something other than a dict, ``_collect_allowed`` dropped it
    # and we'd otherwise return an ok-but-empty payload.
    missing_after = _require_after_filter(
        out, _ME_REQUIRED, "marginal_effects",
        pre_validated=frozenset(("n", "method", "variables")),
    )
    if missing_after:
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=missing_after,
        )

    # Identifier-shape gates on the variable list and the outcome /
    # model_family scalars. Same primitives as the regression
    # bucket — keeps the disclosure profile uniform.
    _enforce_identifier_string_fields(
        out, frozenset(("outcome_variable", "model_family")),
        transformations, type_label="marginal_effects",
    )
    _enforce_identifier_list_field(
        out, "variables", transformations, type_label="marginal_effects",
    )

    # Cross-field key validation. Each dict-of-numeric field's keys
    # must reference a declared variable; alien keys are dropped
    # with a transformations-log entry. Same defense as the OLS
    # coefficient-name gate.
    declared = set(out.get("variables") or [])
    for dict_field in _ME_ALLOWED_DICT_NUMERIC:
        if dict_field not in out:
            continue
        d = out[dict_field]
        if not isinstance(d, dict):
            continue
        kept: dict[str, float] = {}
        dropped: list[str] = []
        for k, v in d.items():
            if k in declared:
                kept[k] = v
            else:
                dropped.append(k)
        if dropped:
            # Names withheld — keys are caller-controlled and could
            # carry raw data bytes if echoed.
            transformations.append(
                f"dropped {len(dropped)} undeclared key(s) from "
                f"{dict_field!r} (names withheld — keys are caller-"
                f"controlled and could carry raw data bytes)"
            )
        out[dict_field] = kept

    # Method-specific gates.
    #
    # ``at_representative`` requires the ``at_values`` dict so the
    # model can interpret the marginal effect at a specific point
    # (otherwise "MEM evaluated at … nothing?" is an interpretation
    # trap). For ``ame`` / ``mem``, ``at_values`` is structurally
    # absent — drop it with a transformation note if present, since
    # there's no conditioning point to interpret it against.
    if method == "at_representative":
        if "at_values" not in out or not out.get("at_values"):
            return SanitizerResult(
                ok=False, analysis_type="marginal_effects",
                rejection_reason=(
                    "method='at_representative' requires non-empty "
                    "at_values (the covariate vector the effect was "
                    "evaluated at)"
                ),
            )
    else:
        if "at_values" in out:
            transformations.append(
                f"dropped 'at_values' (method={method!r} has no "
                f"conditioning point)"
            )
            del out["at_values"]

    # ``effects`` non-empty after cross-field filter — a payload
    # whose only kept effect keys were undeclared would otherwise
    # ship with effects={} and look successful. Make the failure
    # explicit.
    if not out.get("effects"):
        return SanitizerResult(
            ok=False, analysis_type="marginal_effects",
            rejection_reason=(
                "effects dict empty after sanitization — keys did "
                "not match the declared variables list"
            ),
        )

    # Precision clamp by total sample n. Per-variable marginal
    # effects share precision with the underlying coefficients
    # they're derived from; the same sigfigs scaling applies.
    n = out["n"]
    sigfigs = sigfigs_for_n(n)
    for key in _ME_ALLOWED_DICT_NUMERIC:
        if key in out:
            out[key] = clamp_precision_dict(out[key], n)
    transformations.append(
        f"clamped all numeric fields to {sigfigs} significant figures (n={n})"
    )

    return SanitizerResult(
        ok=True, analysis_type="marginal_effects",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Factor decomposition — PCA + factor analysis as one shape
# ---------------------------------------------------------------------------
#
# Covers principal-components analysis (PCA), classical factor
# analysis (factanal / sklearn.decomposition.FactorAnalyzer), and
# maximum-likelihood factor analysis. The disclosure-relevant
# quantities are all aggregates over the full sample:
#   * Loadings (variable × component matrix) — eigenvectors of the
#     correlation / covariance matrix. Bounded roughly [-1, 1] for
#     standardized inputs. The variable names are dataset columns
#     the model has already seen; component names ("PC1", "factor1",
#     etc.) are synthetic.
#   * Eigenvalues / explained variance / cumulative variance — one
#     scalar per component. Aggregate scalars.
#   * Communalities / uniqueness — one scalar per variable.
#   * Goodness-of-fit (KMO, Bartlett, chi²) — aggregate test stats.
#
# Privacy carve-out, structural: factor SCORES (per-observation
# projections onto the components) are NOT in this allowlist. Scores
# are essentially raw observations transformed; emitting them would
# undo the privacy claim on the exact axis PCA/FA defines. Stays
# researcher-only by construction.
#
# The same shape carries PCA, classical FA, and ML-FA outputs;
# ``method`` distinguishes. Helpers per method × language; the
# sanitizer doesn't dispatch on it.

_FACTOR_REQUIRED: frozenset[str] = frozenset((
    "type", "method", "n_observations", "n_variables", "n_components",
    "variables", "loadings",
))
_FACTOR_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # Goodness-of-fit scalars (mostly for ML factor analysis):
    "kmo",                       # Kaiser-Meyer-Olkin sampling adequacy
    "bartlett_chi_squared",      # Bartlett's test of sphericity
    "bartlett_p_value",
    "chi_squared",               # ML-FA goodness-of-fit
    "chi_squared_p_value",
    "log_likelihood",
    "rmsea",                     # Root mean square error of approximation
    "tli",                       # Tucker-Lewis index
))
_FACTOR_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n_observations", "n_variables", "n_components",
    "degrees_of_freedom",
))
_FACTOR_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "method", "rotation",
))
_FACTOR_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    # Variable names participating in the decomposition. Each goes
    # through ``safe_key``; the list is bounded by ``_FACTOR_MAX_VARIABLES``.
    "variables",
    # Component labels ("PC1", "PC2", ..., or "factor1", "factor2", ...).
    # Synthetic — generated by the helper rather than data-derived —
    # but allowlisted for consistency so the sanitizer's cross-field
    # check has the component keys to validate against.
    "components",
))
# Nested-dict-of-dict field: loadings is {variable: {component: value}}.
# Processed separately after the top-level filter, mirroring the
# ``did_event_study`` pattern.
_FACTOR_NESTED_DICT_FIELDS: frozenset[str] = frozenset((
    "loadings",
))
# Flat dict-numeric fields. Two key conventions live here:
#   * Component-keyed: ``explained_variance`` / ``explained_variance_ratio``
#     / ``cumulative_variance`` / ``eigenvalues``. Keys must match the
#     declared ``components`` list.
#   * Variable-keyed: ``communalities`` / ``uniqueness``. Keys must
#     match the declared ``variables`` list.
_FACTOR_PER_COMPONENT_DICTS: frozenset[str] = frozenset((
    "explained_variance", "explained_variance_ratio",
    "cumulative_variance", "eigenvalues",
))
_FACTOR_PER_VARIABLE_DICTS: frozenset[str] = frozenset((
    "communalities", "uniqueness",
))
_FACTOR_VALID_METHODS: frozenset[str] = frozenset((
    "pca",                          # principal components
    "factor_analysis",              # generic
    "principal_factor",             # principal-factor extraction
    "maximum_likelihood",           # ML factor analysis
    "minimum_residual",             # MinRes
))
_FACTOR_VALID_ROTATIONS: frozenset[str] = frozenset((
    "none", "varimax", "promax", "oblimin", "quartimax",
    "equamax", "geomin", "bentlerT", "bifactor",
))
# Structural caps. A real PCA / FA published in a paper uses ≤ ~50
# variables and ≤ ~20 components; bigger shapes are almost always
# data-shaped objects masquerading as aggregates.
_FACTOR_MAX_VARIABLES: int = 100
_FACTOR_MAX_COMPONENTS: int = 50


def _sanitize_factor_decomposition(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _FACTOR_REQUIRED, "factor_decomposition")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=missing_reason,
        )

    # n_observations gates the precision clamp; ``min_n_descriptive``
    # is the same threshold used for descriptive payloads (PCA / FA on
    # tiny samples is statistically meaningless anyway).
    n_obs = raw.get("n_observations")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 0:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_observations must be a non-negative int; "
                f"got {type(n_obs).__name__}"
            ),
        )
    try:
        require_minimum_n(n_obs, config.min_n_descriptive, "n_observations")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=str(e),
        )

    # Method enum check before any further work.
    method = raw.get("method")
    if not isinstance(method, str) or method not in _FACTOR_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"method must be one of {sorted(_FACTOR_VALID_METHODS)}; "
                f"got {method!r}"
            ),
        )

    # Variable list — required, bounded, names go through safe_key.
    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not all(
        isinstance(v, str) for v in raw_vars
    ):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                "variables must be a list of strings (dataset column names)"
            ),
        )
    if len(raw_vars) == 0:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason="variables list is empty",
        )
    if len(raw_vars) > _FACTOR_MAX_VARIABLES:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; structural cap "
                f"is {_FACTOR_MAX_VARIABLES}"
            ),
        )

    # n_components claim must match the actual declared structure.
    n_comp_claim = raw.get("n_components")
    if not isinstance(n_comp_claim, int) or n_comp_claim <= 0:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_components must be a positive int; got {n_comp_claim!r}"
            ),
        )
    if n_comp_claim > _FACTOR_MAX_COMPONENTS:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_components is {n_comp_claim}; structural cap is "
                f"{_FACTOR_MAX_COMPONENTS}"
            ),
        )

    # n_variables claim must match the variables list length.
    n_var_claim = raw.get("n_variables")
    if not isinstance(n_var_claim, int) or n_var_claim != len(raw_vars):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"n_variables claim ({n_var_claim}) does not match "
                f"variables list length ({len(raw_vars)})"
            ),
        )

    transformations: list[str] = []

    # Sanitize variable + component label lists; reject safe_key
    # collisions outright (the same pattern as crosstab / DiD).
    #
    # Identifier-shape gate (matches the column-name-bearing fields in
    # every other analysis shape -- see ``_is_identifier_shape``'s
    # module-header rationale): ``variables`` here is documented as
    # dataset columns the model has already seen, not free text, and
    # it becomes the ``loadings``/``communalities``/``uniqueness`` dict
    # keys the model reads directly. A rejected entry here fails the
    # whole payload (rather than silently dropping just that variable,
    # the way ``correlation_matrix`` handles the same gate) because
    # ``safe_vars`` is positionally load-bearing for every per-variable
    # row built below -- filtering it independently of the raw
    # ``loadings`` structure would reopen the declared/undeclared
    # misalignment class of bug the crosstab/correlation_matrix code
    # comments describe guarding against, for no offsetting benefit.
    safe_vars: list[str] = []
    safe_var_set: set[str] = set()
    for raw_v in raw_vars:
        sv = safe_key(raw_v)
        if not _is_identifier_shape(sv):
            return SanitizerResult(
                ok=False, analysis_type="factor_decomposition",
                rejection_reason=(
                    "a 'variables' entry did not match the column-name "
                    "identifier shape after sanitization (value "
                    "withheld -- caller-controlled)"
                ),
            )
        if sv in safe_var_set:
            return SanitizerResult(
                ok=False, analysis_type="factor_decomposition",
                rejection_reason=(
                    "variable label collision after sanitization "
                    "(two distinct raw labels sanitize to the same "
                    "safe_key; would silently overwrite loadings rows). "
                    "Labels withheld."
                ),
            )
        safe_var_set.add(sv)
        safe_vars.append(sv)

    raw_comps = raw.get("components") or [f"PC{i+1}" for i in range(n_comp_claim)]
    if not isinstance(raw_comps, list) or not all(isinstance(c, str) for c in raw_comps):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason="components must be a list of strings",
        )
    if len(raw_comps) != n_comp_claim:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"components list length ({len(raw_comps)}) does not "
                f"match n_components ({n_comp_claim})"
            ),
        )
    safe_comps: list[str] = []
    safe_comp_set: set[str] = set()
    for raw_c in raw_comps:
        sc = safe_key(raw_c)
        if sc in safe_comp_set:
            return SanitizerResult(
                ok=False, analysis_type="factor_decomposition",
                rejection_reason=(
                    "component label collision after sanitization"
                ),
            )
        safe_comp_set.add(sc)
        safe_comps.append(sc)

    out: dict[str, Any] = _collect_allowed(
        raw,
        numeric=_FACTOR_ALLOWED_NUMERIC_FIELDS,
        integer=_FACTOR_ALLOWED_INT_FIELDS,
        string=_FACTOR_ALLOWED_STRING_FIELDS,
        list_string=_FACTOR_ALLOWED_LIST_STRING,
        transformations=transformations,
    )
    out["type"] = "factor_decomposition"
    out["method"] = method
    out["variables"] = safe_vars
    out["components"] = safe_comps
    out["n_observations"] = n_obs
    out["n_variables"] = n_var_claim
    out["n_components"] = n_comp_claim

    # Validate rotation enum if supplied.
    rot = out.get("rotation")
    if rot is not None and rot not in _FACTOR_VALID_ROTATIONS:
        transformations.append(
            f"dropped 'rotation' value (must be one of "
            f"{sorted(_FACTOR_VALID_ROTATIONS)})"
        )
        del out["rotation"]

    # Loadings: nested {variable: {component: value}}. Outer keys
    # must be in safe_vars; inner keys must be in safe_comps.
    loadings_raw = raw.get("loadings")
    if not isinstance(loadings_raw, dict):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                f"loadings must be a nested dict {{variable: {{component: value}}}};"
                f" got {type(loadings_raw).__name__}"
            ),
        )
    cleaned_loadings: dict[str, dict[str, float]] = {}
    dropped_outer = 0
    dropped_inner = 0
    for outer_k, inner_v in loadings_raw.items():
        if not isinstance(outer_k, str):
            dropped_outer += 1
            continue
        sv = safe_key(outer_k)
        if sv not in safe_var_set:
            dropped_outer += 1
            continue
        if not isinstance(inner_v, dict):
            dropped_outer += 1
            continue
        cleaned_inner: dict[str, float] = {}
        for inner_k, val in inner_v.items():
            if not isinstance(inner_k, str):
                dropped_inner += 1
                continue
            sc = safe_key(inner_k)
            if sc not in safe_comp_set:
                dropped_inner += 1
                continue
            if not _is_finite_number(val):
                dropped_inner += 1
                continue
            cleaned_inner[sc] = clamp_precision(float(val), n_obs)
        if cleaned_inner:
            cleaned_loadings[sv] = cleaned_inner
    if dropped_outer:
        transformations.append(
            f"dropped {dropped_outer} undeclared variable(s) from loadings"
        )
    if dropped_inner:
        transformations.append(
            f"dropped {dropped_inner} undeclared component entry(ies) from loadings"
        )
    if not cleaned_loadings:
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                "loadings dict empty after sanitization — keys didn't match "
                "the declared variables/components"
            ),
        )
    if set(cleaned_loadings) != safe_var_set or any(
        set(row) != safe_comp_set for row in cleaned_loadings.values()
    ):
        return SanitizerResult(
            ok=False, analysis_type="factor_decomposition",
            rejection_reason=(
                "loadings must contain one complete component row for every "
                "declared variable; partial loading matrices are ambiguous"
            ),
        )
    out["loadings"] = cleaned_loadings

    # Per-component dicts (explained_variance, eigenvalues, …) — outer
    # keys must be in safe_comps.
    for field in _FACTOR_PER_COMPONENT_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sc = safe_key(k)
            if sc not in safe_comp_set:
                continue
            if not _is_finite_number(val):
                continue
            cleaned[sc] = clamp_precision(float(val), n_obs)
        if cleaned:
            out[field] = cleaned

    # Per-variable dicts (communalities, uniqueness) — outer keys
    # must be in safe_vars.
    for field in _FACTOR_PER_VARIABLE_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sv = safe_key(k)
            if sv not in safe_var_set:
                continue
            if not _is_finite_number(val):
                continue
            cleaned[sv] = clamp_precision(float(val), n_obs)
        if cleaned:
            out[field] = cleaned

    # Precision-clamp scalar numeric fields.
    for key in _FACTOR_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_obs)

    # PCA/FA variance accounting has invariants beyond per-value bounds.
    ratios = out.get("explained_variance_ratio")
    if isinstance(ratios, dict) and sum(ratios.values()) > 1.001:
        return SanitizerResult(
            ok=False,
            analysis_type="factor_decomposition",
            rejection_reason=(
                "explained_variance_ratio values sum above 1; the component "
                "variance decomposition is mathematically inconsistent"
            ),
        )
    cumulative = out.get("cumulative_variance")
    if isinstance(cumulative, dict):
        ordered = [cumulative[c] for c in safe_comps if c in cumulative]
        if any(right < left for left, right in zip(ordered, ordered[1:])):
            return SanitizerResult(
                ok=False,
                analysis_type="factor_decomposition",
                rejection_reason=(
                    "cumulative_variance decreases across declared components; "
                    "the variance decomposition is mathematically inconsistent"
                ),
            )

    transformations.append(
        f"clamped numeric fields to precision matching n_observations={n_obs}"
    )

    return SanitizerResult(
        ok=True, analysis_type="factor_decomposition",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Cluster analysis (k-means, k-medoids, hierarchical, …)
# ---------------------------------------------------------------------------
#
# The shape ships per-cluster centroids and quality metrics from a
# fitted clustering. Two new SDC primitives the existing shapes
# don't have:
#
#   1.  **Whole-cluster suppression by size.** Clusters below
#       ``min_n_cluster`` are dropped entirely — their entry in
#       ``cluster_sizes``, their centroid row, their within-cluster
#       SS, every per-cluster dict. Partial publication would leak
#       the cluster size through which clusters survived. Same
#       pattern as DiD's cohort gate.
#
#   2.  **Per-cluster precision clamping on centroids.** Centroids
#       are means over the cluster's members; their precision
#       scales with that cluster's N, not the global N. A centroid
#       of a 12-person cluster on income should be clamped to
#       ~3 sigfigs; a centroid of a 12,000-person cluster on income
#       can carry ~5. The existing shapes all clamp by global N
#       (``clamp_precision_dict(d, n_total)``); this shape
#       walks per-row and clamps with the row-specific N.
#
# Privacy carve-out, structural: per-observation cluster assignments
# (sklearn's ``labels_``, R kmeans's ``$cluster``) are NOT in this
# allowlist. Assignments are per-row data — emitting them would tell
# the model which row went where, which combined with the centroid
# is enough to identify individuals in small clusters. The
# researcher's local R / Python session sees them; the model
# doesn't.

_CLUSTER_REQUIRED: frozenset[str] = frozenset((
    "type", "method", "n_observations", "n_clusters", "n_features",
    "variables", "cluster_labels", "cluster_sizes",
    # ``centroids`` is conditionally required — required for every
    # method that has centroids by construction (kmeans / hierarchical
    # / pam / agglomerative), absent-OK for DBSCAN / HDBSCAN (density-
    # based; no centroids by design). The conditional check fires
    # below in ``_sanitize_cluster_analysis``. If a DBSCAN payload
    # includes centroids anyway (caller computed them post-hoc from
    # the labels array), the field is still validated against the
    # cross-field rules.
))
_CLUSTER_METHODS_WITHOUT_CENTROIDS: frozenset[str] = frozenset((
    "dbscan", "hdbscan",
))
_CLUSTER_ALLOWED_NUMERIC_FIELDS: frozenset[str] = frozenset((
    # Sum-of-squares decomposition. Scalars over the whole fit;
    # aggregate quantities, no per-row leak.
    "total_within_ss", "between_cluster_ss", "total_ss",
    "ss_ratio",                # between / total — fraction explained
    "inertia",                 # sklearn alias for total_within_ss
    # Cluster quality scalars.
    "silhouette_score",        # global mean silhouette
    "calinski_harabasz_score",
    "davies_bouldin_score",
    # Hierarchical-specific: the dendrogram cut height that produced
    # the n_clusters partition. A scalar over the data; the
    # dendrogram itself (linkage matrix / merge heights series) is
    # structurally absent from the allowlist.
    "cut_height",
))
_CLUSTER_ALLOWED_INT_FIELDS: frozenset[str] = frozenset((
    "n_observations", "n_clusters", "n_features", "n_iterations",
    # DBSCAN / HDBSCAN: count of points labeled noise (outside any
    # cluster). Aggregate scalar; the noise points' identities don't
    # cross — same disclosure profile as ``n_clusters``.
    "n_noise_points",
))
_CLUSTER_ALLOWED_STRING_FIELDS: frozenset[str] = frozenset((
    "type", "method", "distance_metric", "linkage",
))
_CLUSTER_ALLOWED_LIST_STRING: frozenset[str] = frozenset((
    "variables",
    "cluster_labels",          # synthetic identifiers like "cluster_1"
))
# Per-cluster flat dicts. ``cluster_sizes`` drives the suppression
# gate so it's required. The others are optional metrics; their keys
# must match the declared (surviving) cluster labels.
_CLUSTER_PER_CLUSTER_INT_DICTS: frozenset[str] = frozenset((
    "cluster_sizes",
))
_CLUSTER_PER_CLUSTER_NUMERIC_DICTS: frozenset[str] = frozenset((
    "within_cluster_ss", "silhouette_per_cluster",
))
# Per-variable numeric dict: ``f_statistic_per_variable`` carries
# the between-cluster F-statistic for each input variable —
# diagnostic for which variables most discriminate clusters.
# Aggregate scalar per variable; keys validated against the
# declared ``variables`` list (not cluster_sizes).
_CLUSTER_PER_VARIABLE_NUMERIC_DICTS: frozenset[str] = frozenset((
    "f_statistic_per_variable",
))
# Nested-dict (cluster × variable) — centroids. Per-cluster precision
# clamping fires on this field.
_CLUSTER_NESTED_DICT_FIELDS: frozenset[str] = frozenset((
    "centroids",
))
_CLUSTER_VALID_METHODS: frozenset[str] = frozenset((
    "kmeans",
    "hierarchical",      # use the ``linkage`` field for ward / complete /
                         # average / single / centroid / median
    "agglomerative",     # alias for hierarchical bottom-up; same payload
    "pam",               # partitioning around medoids — literature-standard
                         # name for k-medoids
    "kmedoids",          # legacy alias retained; ``pam`` is the canonical
    "dbscan",            # density-based — no centroids by construction;
                         # centroids field becomes optional below
    "hdbscan",           # hierarchical density-based; same payload shape
    # Gaussian-mixture and spectral clustering are intentionally absent.
    # This contract has no component covariance/weight fields for GMM and no
    # affinity/eigenspace diagnostics for spectral clustering. Accepting a
    # method name without the quantities needed to interpret it created a
    # false capability claim.
))
_CLUSTER_VALID_LINKAGE: frozenset[str] = frozenset((
    "ward", "complete", "average", "single", "centroid", "median",
))
_CLUSTER_VALID_DISTANCE: frozenset[str] = frozenset((
    "euclidean", "manhattan", "cosine", "mahalanobis", "chebyshev",
    "minkowski", "hamming", "jaccard",
))
_CLUSTER_MAX_CLUSTERS: int = 50
_CLUSTER_MAX_FEATURES: int = 100


def _sanitize_cluster_analysis(
    raw: dict[str, Any], config: SDCConfig
) -> SanitizerResult:
    missing_reason = _require_fields(raw, _CLUSTER_REQUIRED, "cluster_analysis")
    if missing_reason:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=missing_reason,
        )

    n_obs = raw.get("n_observations")
    if not isinstance(n_obs, int) or isinstance(n_obs, bool) or n_obs < 0:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_observations must be a non-negative int; "
                f"got {type(n_obs).__name__}"
            ),
        )
    try:
        require_minimum_n(n_obs, config.min_n_descriptive, "n_observations")
    except MinimumNViolation as e:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=str(e),
        )

    method = raw.get("method")
    if not isinstance(method, str) or method not in _CLUSTER_VALID_METHODS:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"method must be one of {sorted(_CLUSTER_VALID_METHODS)}; "
                f"got {method!r}"
            ),
        )

    raw_vars = raw.get("variables")
    if not isinstance(raw_vars, list) or not all(
        isinstance(v, str) for v in raw_vars
    ):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="variables must be a list of strings",
        )
    if len(raw_vars) == 0:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="variables list is empty",
        )
    if len(raw_vars) > _CLUSTER_MAX_FEATURES:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"variables has {len(raw_vars)} entries; structural cap "
                f"is {_CLUSTER_MAX_FEATURES}"
            ),
        )

    n_clusters_claim = raw.get("n_clusters")
    if not isinstance(n_clusters_claim, int) or n_clusters_claim <= 0:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_clusters must be a positive int; got {n_clusters_claim!r}"
            ),
        )
    if n_clusters_claim > _CLUSTER_MAX_CLUSTERS:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_clusters is {n_clusters_claim}; structural cap is "
                f"{_CLUSTER_MAX_CLUSTERS}"
            ),
        )

    n_features_claim = raw.get("n_features")
    if not isinstance(n_features_claim, int) or n_features_claim != len(raw_vars):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"n_features ({n_features_claim}) does not match "
                f"variables list length ({len(raw_vars)})"
            ),
        )

    raw_labels = raw.get("cluster_labels")
    if not isinstance(raw_labels, list) or not all(
        isinstance(c, str) for c in raw_labels
    ):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="cluster_labels must be a list of strings",
        )
    if len(raw_labels) != n_clusters_claim:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"cluster_labels length ({len(raw_labels)}) does not "
                f"match n_clusters ({n_clusters_claim})"
            ),
        )

    transformations: list[str] = []

    # Sanitize variable + cluster labels.
    #
    # Identifier-shape gate -- same rationale as ``factor_decomposition``
    # just above: ``variables`` is documented as dataset column names,
    # not free text, and becomes the ``centroids``/
    # ``f_statistic_per_variable`` dict keys the model reads directly.
    # Reject the whole payload on a non-identifier-shape entry rather
    # than filtering it, since ``safe_vars`` is positionally load-
    # bearing for every per-variable row built below.
    safe_vars: list[str] = []
    safe_var_set: set[str] = set()
    for raw_v in raw_vars:
        sv = safe_key(raw_v)
        if not _is_identifier_shape(sv):
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "a 'variables' entry did not match the column-name "
                    "identifier shape after sanitization (value "
                    "withheld -- caller-controlled)"
                ),
            )
        if sv in safe_var_set:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "variable label collision after sanitization"
                ),
            )
        safe_var_set.add(sv)
        safe_vars.append(sv)

    safe_labels: list[str] = []
    safe_label_set: set[str] = set()
    for raw_c in raw_labels:
        sl = safe_key(raw_c)
        if sl in safe_label_set:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "cluster label collision after sanitization"
                ),
            )
        safe_label_set.add(sl)
        safe_labels.append(sl)

    # Validate cluster_sizes structure + run the whole-cluster
    # suppression gate. Clusters below ``min_n_descriptive`` are
    # dropped whole — their entry in cluster_sizes, their centroid
    # row, their within-cluster SS, every per-cluster entry.
    raw_sizes = raw.get("cluster_sizes")
    if not isinstance(raw_sizes, dict):
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason="cluster_sizes must be a dict",
        )

    cluster_n: dict[str, int] = {}
    declared_sizes: set[str] = set()
    for raw_k, raw_n in raw_sizes.items():
        if not isinstance(raw_k, str):
            continue
        sl = safe_key(raw_k)
        if sl not in safe_label_set:
            continue
        declared_sizes.add(sl)
        if not isinstance(raw_n, int) or isinstance(raw_n, bool) or raw_n < 0:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "cluster_sizes values must be non-negative ints"
                ),
            )
        cluster_n[sl] = raw_n

    # Every declared cluster_label needs a size entry — the gate
    # can't run otherwise.
    missing_sizes = [c for c in safe_labels if c not in declared_sizes]
    if missing_sizes:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"{len(missing_sizes)} cluster_label(s) have no "
                f"cluster_sizes entry. Labels withheld."
            ),
        )

    min_n_cluster = config.min_n_descriptive
    suppressed: set[str] = set()
    surviving: list[str] = []
    cleaned_sizes: dict[str, int] = {}
    for sl in safe_labels:
        size = cluster_n[sl]
        if size < min_n_cluster:
            suppressed.add(sl)
        else:
            surviving.append(sl)
            cleaned_sizes[sl] = size

    if suppressed:
        transformations.append(
            f"cluster suppression: {len(suppressed)} cluster(s) with "
            f"size < {min_n_cluster} dropped entirely (labels withheld "
            f"— cluster identities are disclosive when small)"
        )

    if not surviving:
        return SanitizerResult(
            ok=False, analysis_type="cluster_analysis",
            rejection_reason=(
                f"all clusters have size < {min_n_cluster}; nothing "
                f"survives the cluster-size gate"
            ),
        )

    out: dict[str, Any] = _collect_allowed(
        raw,
        numeric=_CLUSTER_ALLOWED_NUMERIC_FIELDS,
        integer=_CLUSTER_ALLOWED_INT_FIELDS,
        string=_CLUSTER_ALLOWED_STRING_FIELDS,
        list_string=_CLUSTER_ALLOWED_LIST_STRING,
        transformations=transformations,
    )
    out["type"] = "cluster_analysis"
    out["method"] = method
    out["variables"] = safe_vars
    out["n_observations"] = n_obs
    out["n_variables"] = n_features_claim  # back-compat synonym
    out["n_features"] = n_features_claim
    # cluster_labels list is the surviving set (alphabetized to
    # match the per-dict keys' order).
    out["cluster_labels"] = sorted(surviving)
    out["n_clusters"] = len(surviving)
    if len(surviving) != n_clusters_claim:
        transformations.append(
            f"n_clusters reduced from {n_clusters_claim} to "
            f"{len(surviving)} after cluster-size suppression"
        )
    out["cluster_sizes"] = cleaned_sizes

    # Linkage / distance_metric enum validation.
    lk = out.get("linkage")
    if lk is not None and lk not in _CLUSTER_VALID_LINKAGE:
        transformations.append(
            f"dropped 'linkage' value (must be one of "
            f"{sorted(_CLUSTER_VALID_LINKAGE)})"
        )
        del out["linkage"]
    dm = out.get("distance_metric")
    if dm is not None and dm not in _CLUSTER_VALID_DISTANCE:
        transformations.append(
            f"dropped 'distance_metric' value (must be one of "
            f"{sorted(_CLUSTER_VALID_DISTANCE)})"
        )
        del out["distance_metric"]

    # Centroids: nested {cluster: {variable: value}}. Outer keys
    # must be surviving clusters; inner keys in safe_vars.
    #
    # Conditionally required: methods in
    # ``_CLUSTER_METHODS_WITHOUT_CENTROIDS`` (DBSCAN, HDBSCAN) have
    # no centroids by construction — absent-OK. If a DBSCAN payload
    # ships centroids anyway (caller computed them post-hoc from the
    # labels), the structure is still validated below: cross-field
    # rules apply identically so the field can't smuggle anything
    # past the gate.
    #
    # Per-cluster precision clamping fires on the surviving
    # centroids — each value clamped by the cluster's OWN N rather
    # than the global n_observations. The 12-member cluster gets
    # fewer sigfigs than the 12,000-member cluster.
    centroids_raw = raw.get("centroids")
    centroids_absent_ok = method in _CLUSTER_METHODS_WITHOUT_CENTROIDS

    if centroids_raw is None:
        if not centroids_absent_ok:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    f"centroids is required for method={method!r}; "
                    f"only DBSCAN-family methods may omit centroids"
                ),
            )
        # DBSCAN-family with no centroids — skip the centroid block.
    else:
        if not isinstance(centroids_raw, dict):
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason="centroids must be a nested dict",
            )
        cleaned_centroids: dict[str, dict[str, float]] = {}
        dropped_outer = 0
        dropped_inner = 0
        surviving_set = set(surviving)
        for outer_k, inner_v in centroids_raw.items():
            if not isinstance(outer_k, str):
                dropped_outer += 1
                continue
            sl = safe_key(outer_k)
            if sl not in surviving_set:
                dropped_outer += 1
                continue
            if not isinstance(inner_v, dict):
                dropped_outer += 1
                continue
            cleaned_inner: dict[str, float] = {}
            for inner_k, val in inner_v.items():
                if not isinstance(inner_k, str):
                    dropped_inner += 1
                    continue
                sv = safe_key(inner_k)
                if sv not in safe_var_set:
                    dropped_inner += 1
                    continue
                if not _is_finite_number(val):
                    dropped_inner += 1
                    continue
                cleaned_inner[sv] = float(val)
            if cleaned_inner:
                # Per-cluster clamp via the named primitive: each
                # variable's centroid value gets clamped by THIS
                # cluster's N rather than the global n_observations.
                cleaned_centroids[sl] = clamp_precision_dict(
                    cleaned_inner, cleaned_sizes[sl],
                )
        if dropped_outer:
            transformations.append(
                f"dropped {dropped_outer} undeclared/suppressed cluster(s) "
                f"from centroids (labels withheld)"
            )
        if dropped_inner:
            transformations.append(
                f"dropped {dropped_inner} undeclared variable entry(ies) "
                f"from centroids"
            )
        # For non-DBSCAN methods, centroids must be non-empty after
        # the gate; for DBSCAN, an empty centroids dict (e.g. all
        # centroid clusters were sub-min-N) is acceptable since
        # centroids weren't required.
        if not cleaned_centroids and not centroids_absent_ok:
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "centroids dict empty after sanitization — keys did "
                    "not match declared (and surviving) "
                    "clusters/variables"
                ),
            )
        if not centroids_absent_ok and (
            set(cleaned_centroids) != set(surviving)
            or any(set(row) != safe_var_set
                   for row in cleaned_centroids.values())
        ):
            return SanitizerResult(
                ok=False, analysis_type="cluster_analysis",
                rejection_reason=(
                    "centroids must contain one complete variable row for "
                    "every surviving cluster"
                ),
            )
        if cleaned_centroids:
            out["centroids"] = cleaned_centroids
            transformations.append(
                "centroid precision clamped per-cluster (each value's "
                "sigfigs scales with that cluster's size, not global N)"
            )

    # Per-cluster numeric dicts (within_cluster_ss, silhouette_per_cluster):
    # keys must be surviving clusters; values precision-clamped by
    # the cluster's OWN N via the named ``clamp_dict_by_per_key_n``
    # primitive. Each metric is an aggregate over the cluster's
    # members (within-SS is a sum over the cluster's points;
    # silhouette is a mean over them), so local N is the right
    # precision floor — same reasoning as the centroid clamp.
    for field in _CLUSTER_PER_CLUSTER_NUMERIC_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sl = safe_key(k)
            if sl not in set(surviving):
                continue
            if not _is_finite_number(val):
                continue
            cleaned[sl] = float(val)
        if cleaned:
            # Per-cluster precision clamp via the named primitive.
            out[field] = clamp_dict_by_per_key_n(cleaned, cleaned_sizes)

    # Per-variable numeric dicts (f_statistic_per_variable): keys
    # must be declared variable names; values are aggregate scalars
    # (between-cluster F-stat per variable) — global-N clamp is the
    # right precision floor since the F is computed over the full
    # sample, not a subgroup.
    for field in _CLUSTER_PER_VARIABLE_NUMERIC_DICTS:
        v = raw.get(field)
        if v is None:
            continue
        if not isinstance(v, dict):
            transformations.append(
                f"dropped {field!r}: expected dict, got {type(v).__name__}"
            )
            continue
        cleaned_var: dict[str, float] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                continue
            sv = safe_key(k)
            if sv not in safe_var_set:
                continue
            if not _is_finite_number(val):
                continue
            cleaned_var[sv] = clamp_precision(float(val), n_obs)
        if cleaned_var:
            out[field] = cleaned_var

    # Precision-clamp scalar numeric fields (total_within_ss, etc.)
    # by global N.
    for key in _CLUSTER_ALLOWED_NUMERIC_FIELDS:
        if key in out:
            out[key] = clamp_precision(out[key], n_obs)

    return SanitizerResult(
        ok=True, analysis_type="cluster_analysis",
        sanitized=out, transformations=transformations,
    )


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_METHOD_RESULT_MAX_FIELDS = 100
_METHOD_RESULT_NUMERIC_MAPS = frozenset((
    "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper", "metrics",
))
_METHOD_DIAGNOSTIC_STATUS = frozenset(("pass", "warn", "fail", "not_applicable"))


def _sanitize_method_result(
    raw: dict[str, Any], config: SDCConfig,
) -> SanitizerResult:
    """Sanitize the universal aggregate result for registry-backed methods.

    The method registry supplies the allowlisted method ID, mandatory
    diagnostics, and claim rule. Payload authors cannot invent any of those
    policy fields. Observation-level arrays, labels, predictions, residuals,
    and case diagnostics have no slot in this schema.
    """
    from sift.methodology import METHODS

    required = frozenset(("type", "method_id", "n", "diagnostics"))
    missing = _require_fields(raw, required, "method_result")
    if missing:
        return SanitizerResult(False, "method_result", rejection_reason=missing)
    method_id = raw.get("method_id")
    if not isinstance(method_id, str) or method_id not in METHODS:
        return SanitizerResult(
            False, "method_result",
            rejection_reason="method_id is not in the supported methodology registry",
        )
    method = METHODS[method_id]
    n = raw.get("n")
    if isinstance(n, bool) or not isinstance(n, int) or n < config.min_n_regression:
        return SanitizerResult(
            False, "method_result",
            rejection_reason="method_result n is below the configured minimum or has the wrong type",
        )
    diagnostics = raw.get("diagnostics")
    if not isinstance(diagnostics, dict) or len(diagnostics) > _METHOD_RESULT_MAX_FIELDS:
        return SanitizerResult(
            False, "method_result",
            rejection_reason="diagnostics must be a bounded object",
        )
    required_diagnostics = set(method.diagnostics)
    if not required_diagnostics.issubset(diagnostics):
        return SanitizerResult(
            False, "method_result",
            rejection_reason=(
                "method_result is missing required methodology diagnostics: "
                f"{sorted(required_diagnostics - set(diagnostics))}"
            ),
        )

    transformations: list[str] = []
    clean_diagnostics: dict[str, Any] = {}
    for key in sorted(required_diagnostics):
        value = diagnostics.get(key)
        if isinstance(value, str) and value in _METHOD_DIAGNOSTIC_STATUS:
            clean_diagnostics[key] = value
        elif isinstance(value, bool):
            clean_diagnostics[key] = value
        elif _is_finite_number(value):
            clean_diagnostics[key] = clamp_precision(float(value), n)
        else:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"required diagnostic {key!r} has an invalid value type",
            )

    out: dict[str, Any] = {
        "type": "method_result", "method_id": method.id,
        "method_family": method.family, "n": n,
        "diagnostics": clean_diagnostics,
        "claim_rule": method.claim_rule,
        "output_schema": method.output_schema,
    }
    analysis_id = raw.get("analysis_id")
    if analysis_id is not None:
        cleaned_analysis_id = safe_key(str(analysis_id))
        if not _is_identifier_shape(cleaned_analysis_id):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="analysis_id must be an identifier-shaped workflow analysis id",
            )
        out["analysis_id"] = cleaned_analysis_id
    any_quantities = False
    cleaned_maps: dict[str, dict[str, float]] = {}
    for field in _METHOD_RESULT_NUMERIC_MAPS:
        value = raw.get(field)
        if value is None:
            continue
        if not isinstance(value, dict) or len(value) > _METHOD_RESULT_MAX_FIELDS:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{field} must be a bounded numeric object",
            )
        clean: dict[str, float] = {}
        raw_keys_by_safe_key: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = safe_key(str(raw_key))
            if not _is_identifier_shape(key) or not _is_finite_number(raw_value):
                transformations.append(f"dropped invalid entry from {field!r}")
                continue
            raw_text = str(raw_key)
            if key in raw_keys_by_safe_key and raw_keys_by_safe_key[key] != raw_text:
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason=f"{field} contains colliding quantity identifiers",
                )
            raw_keys_by_safe_key[key] = raw_text
            number = float(raw_value)
            if field == "p_values" and not 0 <= number <= 1:
                transformations.append("dropped out-of-range p-value")
                continue
            clean[key] = clamp_precision(number, n)
        if clean:
            out[field] = clean
            cleaned_maps[field] = clean
            any_quantities = True

    # Every interval must be ordered and, when an estimate exists, contain it.
    lower = cleaned_maps.get("ci_lower", {})
    upper = cleaned_maps.get("ci_upper", {})
    estimates = cleaned_maps.get("estimates", {})
    if set(lower) != set(upper):
        return SanitizerResult(False, "method_result",
                               rejection_reason="confidence-interval bounds have different keys")
    for key in lower:
        if lower[key] > upper[key] or (
            key in estimates and not lower[key] <= estimates[key] <= upper[key]
        ):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="invalid method_result confidence interval")

    for field in (
        "subjects", "events", "records", "clusters", "imputations", "folds", "seed",
        "burn_in", "matching_donors",
        "treated", "controls", "donors", "pre_periods", "post_periods",
        "frequency", "training_observations", "validation_observations",
        "test_observations", "evaluated_observations", "bootstrap_replicates",
        "replicates", "crs_epsg",
    ):
        value = raw.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{field} must be a non-negative integer")
        if field not in {
            "seed", "imputations", "folds", "replicates", "crs_epsg",
            "bootstrap_replicates",
        } and value > n:
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{field} cannot exceed n")
        out[field] = value

    for field, allowed in {
        "uncertainty_type": {
            "classical", "robust", "cluster_robust", "bootstrap", "posterior",
            "design_based", "multiple_imputation",
        },
        "multiple_testing": {"none", "holm", "bonferroni", "benjamini_hochberg"},
        "evaluation_split": {"none", "held_out", "cross_validation", "grouped", "rolling_origin"},
        "estimand": {
            "ate", "att", "atc", "sample_att", "unit_time_att",
            "average_predicted_cate", "robustness_value",
        },
        "design": {
            "propensity_nearest_neighbor", "inverse_probability_weighting",
            "synthetic_control", "honest_t_learner", "omitted_variable_sensitivity",
            "panel_entity_fixed_effects", "two_by_two_panel_did",
        },
        "interval_method": {
            "model_based_gaussian", "ets_state_space_exact",
            "holtwinters_state_space", "heldout_case_bootstrap",
            "clopper_pearson_binomial",
        },
        "test_alternative": {"two_sided", "larger"},
        "spatial_weight_rule": {"distance_band_binary"},
        "stability_type": {"document_resampling"},
    }.items():
        value = raw.get(field)
        if value is not None:
            if not isinstance(value, str) or value not in allowed:
                return SanitizerResult(False, "method_result",
                                       rejection_reason=f"{field} has an unsupported value")
            out[field] = value

    for field, allowed in {
        "imputation_scope": {
            "prediction_preprocessing", "deterministic_nuisance_covariate",
        },
        "imputation_model": {
            "simple_deterministic", "mice_predictive_mean_matching",
        },
        "mnar_model": {"delta_adjusted_pattern_mixture"},
        "split_strategy": {
            "train_validation_test", "cross_validation",
            "train_test_calibration_cv",
        },
        "baseline_model": {
            "simple_dummy", "uncalibrated_classifier_and_prevalence",
        },
        "calibration_method": {"not_applicable", "nested_sigmoid"},
        "imbalance_strategy": {"not_applicable", "balanced_weight", "none"},
    }.items():
        value = raw.get(field)
        if value is not None:
            if not isinstance(value, str) or value not in allowed:
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason=f"{field} has an unsupported value",
                )
            out[field] = value

    causal_methods = {
        "matching", "propensity_weighting", "synthetic_control",
        "treatment_effect_heterogeneity", "causal_sensitivity",
        "difference_in_differences",
    }
    if method.id in causal_methods and any(field not in out for field in ("estimand", "design")):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="causal method results require declared estimand and design",
        )
    expected_design = {
        "matching": {"propensity_nearest_neighbor"},
        "propensity_weighting": {"inverse_probability_weighting"},
        "synthetic_control": {"synthetic_control"},
        "treatment_effect_heterogeneity": {"honest_t_learner"},
        "causal_sensitivity": {"omitted_variable_sensitivity"},
        "difference_in_differences": {"two_by_two_panel_did"},
    }
    if method.id in expected_design and out.get("design") not in expected_design[method.id]:
        return SanitizerResult(
            False, "method_result",
            rejection_reason=f"{method.id} result declares an incompatible design",
        )
    causal_required_metrics = {
        "matching": {
            "effect", "max_abs_smd_before", "max_abs_smd_after",
            "overlap_fraction", "effective_sample_size",
            "treated_score_p05", "treated_score_p95",
            "control_score_p05", "control_score_p95",
        },
        "propensity_weighting": {
            "effect", "max_abs_smd_before", "max_abs_smd_after",
            "overlap_fraction", "effective_sample_size", "max_weight",
            "treated_score_p05", "treated_score_p95",
            "control_score_p05", "control_score_p95",
        },
        "synthetic_control": {
            "effect", "pre_rmse", "post_rmse", "placebo_p_value",
            "max_donor_weight",
        },
        "treatment_effect_heterogeneity": {
            "average_cate", "cate_sd", "q4_q1_contrast", "calibration_correlation",
            "overlap_fraction", "max_abs_smd_before",
        },
        "causal_sensitivity": {
            "robustness_value_zero", "robustness_value_alpha", "t_statistic",
            "q", "alpha", "margin_equal_r2_01", "margin_equal_r2_05",
            "margin_equal_r2_10",
        },
    }
    if method.id in causal_required_metrics:
        metric_keys = set(cleaned_maps.get("metrics", {}))
        missing_metrics = causal_required_metrics[method.id] - metric_keys
        if missing_metrics:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    f"{method.id} is missing required aggregate design metrics: "
                    f"{sorted(missing_metrics)}"
                ),
            )
    if method.id in {"matching", "propensity_weighting"}:
        if any(field not in out for field in ("treated", "controls")):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} must report treated and control counts",
            )
        if out["treated"] < config.min_n_ttest_group or out["controls"] < config.min_n_ttest_group:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} treatment arms are below the configured group minimum",
            )
        if out["treated"] + out["controls"] != n:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} treated and control counts must sum to n",
            )
        if any(field in cleaned_maps for field in ("standard_errors", "ci_lower", "ci_upper")):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    f"{method.id} does not accept analytic uncertainty from the "
                    "typed propensity-design helper; bootstrap the full design "
                    "before reporting intervals"
                ),
            )
        if clean_diagnostics.get("effect_uncertainty") != "not_applicable":
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} must mark unsupported analytic uncertainty not_applicable",
            )
        if method.id == "matching" and out.get("estimand") != "att":
            return SanitizerResult(False, "method_result",
                                   rejection_reason="matching currently supports the ATT estimand only")
        if method.id == "propensity_weighting" and out.get("estimand") not in {"ate", "att"}:
            return SanitizerResult(False, "method_result",
                                   rejection_reason="propensity weighting supports ATE or ATT")
        design_metrics = cleaned_maps["metrics"]
        overlap = design_metrics["overlap_fraction"]
        before = design_metrics["max_abs_smd_before"]
        after = design_metrics["max_abs_smd_after"]
        ess = design_metrics["effective_sample_size"]
        if not (0 <= overlap <= 1 and before >= 0 and after >= 0 and 0 < ess <= n):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} design diagnostics are outside valid domains",
            )
        for arm in ("treated", "control"):
            p05 = design_metrics[f"{arm}_score_p05"]
            p95 = design_metrics[f"{arm}_score_p95"]
            if not 0 <= p05 <= p95 <= 1:
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason=f"{method.id} propensity quantiles are invalid",
                )
        if method.id == "propensity_weighting" and design_metrics["max_weight"] <= 0:
            return SanitizerResult(False, "method_result",
                                   rejection_reason="propensity weights must be positive")
        estimate_key = str(out["estimand"])
        if estimate_key not in cleaned_maps.get("estimates", {}):
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} estimate key must match estimand")
    if method.id == "synthetic_control" and any(
        field not in out for field in ("donors", "pre_periods", "post_periods")
    ):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="synthetic control must report donor and pre/post period counts",
        )
    if method.id == "synthetic_control":
        if out.get("estimand") != "unit_time_att":
            return SanitizerResult(False, "method_result",
                                   rejection_reason="synthetic control requires unit_time_att estimand")
        if n != (out["donors"] + 1) * (out["pre_periods"] + out["post_periods"]):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="synthetic-control counts are internally inconsistent")
        design_metrics = cleaned_maps["metrics"]
        if (design_metrics["pre_rmse"] < 0 or design_metrics["post_rmse"] < 0
                or not 0 <= design_metrics["placebo_p_value"] <= 1
                or not 0 <= design_metrics["max_donor_weight"] <= 1):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="synthetic-control metrics are outside valid domains")
        if any(field in cleaned_maps for field in ("standard_errors", "ci_lower", "ci_upper")):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="synthetic-control donor placebos are not analytic standard errors")
        if clean_diagnostics.get("effect_uncertainty") != "not_applicable":
            return SanitizerResult(False, "method_result",
                                   rejection_reason="synthetic-control uncertainty must be marked not_applicable")
    if method.id == "treatment_effect_heterogeneity":
        if (out.get("estimand") != "average_predicted_cate"
                or clean_diagnostics.get("honest_sample_splitting") != "pass"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="heterogeneity output requires honest-split average predicted CATE")
        heterogeneity_metrics = cleaned_maps["metrics"]
        calibration = heterogeneity_metrics["calibration_correlation"]
        if (not -1 <= calibration <= 1 or heterogeneity_metrics["cate_sd"] < 0
                or heterogeneity_metrics["q4_q1_contrast"] < 0
                or not 0 <= heterogeneity_metrics["overlap_fraction"] <= 1
                or heterogeneity_metrics["max_abs_smd_before"] < 0):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="heterogeneity calibration must be a correlation")
        if any(field in cleaned_maps for field in ("standard_errors", "ci_lower", "ci_upper")):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="T-learner prediction spread is not an effect standard error")
        if clean_diagnostics.get("effect_uncertainty") != "not_applicable":
            return SanitizerResult(False, "method_result",
                                   rejection_reason="heterogeneity effect uncertainty must be marked not_applicable")
    if method.id == "causal_sensitivity":
        if out.get("estimand") != "robustness_value":
            return SanitizerResult(False, "method_result",
                                   rejection_reason="causal sensitivity requires robustness_value estimand")
        if "uncertainty_type" in out:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "a robustness value is a sensitivity summary, not robust sampling uncertainty"
                ),
            )
        sensitivity_metrics = cleaned_maps["metrics"]
        rv0 = sensitivity_metrics["robustness_value_zero"]
        rva = sensitivity_metrics["robustness_value_alpha"]
        if not (0 <= rva <= rv0 <= 1 and sensitivity_metrics["t_statistic"] >= 0):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="causal-sensitivity metrics are outside valid domains")
        if (sensitivity_metrics["q"] <= 0
                or not 0 < sensitivity_metrics["alpha"] < 1
                or clean_diagnostics.get("assumption_grid") != "pass"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="causal-sensitivity assumption grid is invalid")

    for field, allowed in {
        "model_form": {
            "ordered_logit", "ordered_probit", "multinomial_logit",
            "zero_inflated_poisson", "zero_inflated_negative_binomial",
            "regression_spline", "polynomial_regression",
        },
        "link": {"logit", "probit", "log", "identity"},
        "basis": {
            "bspline", "natural_spline", "restricted_cubic_spline",
            "polynomial",
        },
        "weight_type": {"probability"},
        "variance_method": {
            "taylor_linearization", "brr", "fay", "jackknife", "bootstrap",
        },
        "lonely_psu_handling": {"fail", "adjust", "certainty"},
    }.items():
        value = raw.get(field)
        if value is not None:
            if not isinstance(value, str) or value not in allowed:
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason=f"{field} has an unsupported value",
                )
            out[field] = value

    executable_regressions = {
        "ordinal_regression", "multinomial_regression",
        "zero_inflated_model", "spline_regression",
    }
    if method.id in executable_regressions:
        estimate_keys = set(cleaned_maps.get("estimates", {}))
        if not estimate_keys:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} requires aggregate fitted parameters",
            )
        for field in ("standard_errors", "p_values", "ci_lower", "ci_upper"):
            keys = set(cleaned_maps.get(field, {}))
            if not keys.issubset(estimate_keys):
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason=f"{field} contains an undeclared fitted parameter",
                )

    metrics = cleaned_maps.get("metrics", {})
    if method.id == "ordinal_regression":
        categories = metrics.get("category_count")
        thresholds = metrics.get("threshold_count")
        threshold_keys = {
            key for key in cleaned_maps.get("estimates", {})
            if key.startswith("threshold_")
        }
        ordered_values_optional = [
            cleaned_maps["estimates"].get(f"threshold_{index}")
            for index in range(1, len(threshold_keys) + 1)
        ]
        ordered_values = [
            value for value in ordered_values_optional if value is not None
        ]
        if (out.get("model_form") not in {"ordered_logit", "ordered_probit"}
                or clean_diagnostics.get("proportional_odds") != "warn"
                or categories is None or categories < 3
                or thresholds != categories - 1
                or len(threshold_keys) != int(thresholds)
                or len(ordered_values) != len(ordered_values_optional)
                or any(right <= left for left, right in zip(ordered_values, ordered_values[1:]))):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="ordinal regression has inconsistent categories or ordered thresholds",
            )
    if method.id == "multinomial_regression":
        categories = metrics.get("category_count")
        equations = metrics.get("equation_count")
        min_category_n = metrics.get("min_category_n")
        equation_prefixes = {
            key.split("#", 1)[0] for key in cleaned_maps.get("estimates", {})
            if "#" in key
        }
        if (out.get("model_form") != "multinomial_logit"
                or categories is None or categories < 3
                or equations != categories - 1
                or len(equation_prefixes) != int(equations)
                or min_category_n is None
                or min_category_n < config.min_n_regression):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="multinomial regression has inconsistent equations or insufficient class support",
            )
    if method.id == "zero_inflated_model":
        zero_fraction = metrics.get("zero_fraction")
        count_mean = metrics.get("count_mean")
        ratio = metrics.get("variance_mean_ratio")
        has_both_parts = (
            any(key.startswith("inflate_") for key in cleaned_maps.get("estimates", {}))
            and any(not key.startswith("inflate_") for key in cleaned_maps.get("estimates", {}))
        )
        if (out.get("model_form") not in {
                "zero_inflated_poisson", "zero_inflated_negative_binomial"}
                or zero_fraction is None or not 0 <= zero_fraction <= 1
                or count_mean is None or count_mean < 0
                or ratio is None or ratio < 0 or not has_both_parts):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="zero-inflated result lacks coherent count and inflation aggregates",
            )
    if method.id == "spline_regression":
        basis_df = metrics.get("basis_df")
        basis_parameter_count = metrics.get("basis_parameter_count")
        parameter_count = metrics.get("parameter_count")
        if (out.get("model_form") not in {
                "regression_spline", "polynomial_regression"}
                or out.get("basis") is None or basis_df is None or basis_df < 2
                or basis_df >= n or parameter_count is None
                or basis_parameter_count != basis_df
                or parameter_count != len(cleaned_maps.get("estimates", {}))):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="spline result has inconsistent basis degrees of freedom or parameter count",
            )
    if method.id in {"survey_mean", "survey_proportion", "survey_regression"}:
        estimate_keys = set(cleaned_maps.get("estimates", {}))
        if (not estimate_keys
                or set(cleaned_maps.get("standard_errors", {})) != estimate_keys
                or set(cleaned_maps.get("p_values", {})) != estimate_keys
                or set(lower) != estimate_keys):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="survey result requires matching estimate, SE, p-value, and interval keys",
            )
        effective_n = metrics.get("effective_sample_size")
        weight_cv = metrics.get("weight_cv")
        design_df = metrics.get("design_df")
        strata_count = metrics.get("strata_count")
        psu_count = metrics.get("psu_count")
        replicate_count = metrics.get("replicate_count")
        stage_count = metrics.get("stage_count")
        secondary_psu_count = metrics.get("secondary_psu_count")
        lonely_count = metrics.get("lonely_strata_count")
        lonely_certainty = metrics.get("lonely_certainty_count")
        lonely_adjusted = metrics.get("lonely_adjusted_count")
        fpc_min = metrics.get("fpc_fraction_min")
        fpc_max = metrics.get("fpc_fraction_max")
        variance_method = out.get("variance_method")
        if (out.get("weight_type") != "probability"
                or out.get("uncertainty_type") != "design_based"
                or effective_n is None or not 1 <= effective_n <= n
                or weight_cv is None or weight_cv < 0
                or design_df is None or design_df < 1
                or strata_count is None or strata_count < 0
                or psu_count is None or psu_count < 0
                or replicate_count is None or replicate_count < 0
                or stage_count is None or stage_count < 0
                or secondary_psu_count is None or secondary_psu_count < 0
                or lonely_count is None or lonely_count < 0
                or lonely_certainty is None or lonely_certainty < 0
                or lonely_adjusted is None or lonely_adjusted < 0
                or fpc_min is None or fpc_max is None
                or not 0 <= fpc_min <= fpc_max <= 1):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="survey design metadata is incomplete or mathematically inconsistent",
            )
        if lonely_certainty + lonely_adjusted != lonely_count:
            return SanitizerResult(
                False, "method_result",
                rejection_reason="survey lonely-PSU counts are mathematically inconsistent",
            )
        if variance_method == "taylor_linearization":
            if (replicate_count != 0 or psu_count < strata_count or strata_count < 1
                    or stage_count not in {1, 2}
                    or (stage_count == 1 and secondary_psu_count != 0)
                    or (stage_count == 2 and secondary_psu_count < psu_count)):
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason="Taylor survey variance has inconsistent strata, PSU, or replicate counts",
                )
            policy = out.get("lonely_psu_handling")
            if (policy == "certainty" and lonely_certainty != lonely_count):
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason="certainty lonely-PSU handling requires every singleton to have a census FPC",
                )
            if policy != "adjust" and lonely_adjusted != 0:
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason="lonely-PSU adjustment count is incompatible with the declared policy",
                )
        elif (replicate_count < 2 or strata_count != 0 or psu_count != 0
                or fpc_max != 0 or stage_count != 0 or secondary_psu_count != 0):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="replicate-weight variance has inconsistent replicate or Taylor-design metadata",
            )
        else:
            replicate_mse = metrics.get("replicate_mse")
            replicate_scale = metrics.get("replicate_scale")
            rscale_min = metrics.get("replicate_rscale_min")
            rscale_max = metrics.get("replicate_rscale_max")
            if (replicate_mse not in {0, 1}
                    or replicate_scale is None or replicate_scale <= 0
                    or rscale_min is None or rscale_max is None
                    or not 0 < rscale_min <= rscale_max):
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason="replicate-weight result lacks a valid centering/scale/rscales contract",
                )
        ses = cleaned_maps["standard_errors"]
        if any(value < 0 for value in ses.values()):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="survey standard errors must be non-negative",
            )
        if method.id in {"survey_mean", "survey_proportion"}:
            expected_name = "proportion" if method.id == "survey_proportion" else "mean"
            variance = metrics.get("variance")
            reference = metrics.get("reference_variance")
            deff = metrics.get("design_effect")
            if (estimate_keys != {expected_name}
                    or variance is None or variance < 0
                    or reference is None or reference < 0
                    or deff is None or deff < 0
                    or abs(variance - ses[expected_name] ** 2) > max(1e-8, abs(variance) * 0.01)
                    or (method.id == "survey_proportion"
                        and not 0 <= cleaned_maps["estimates"][expected_name] <= 1)):
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason="survey mean/proportion variance or estimand is inconsistent",
                )
        else:
            for key in estimate_keys:
                variance = metrics.get(f"variance#{key}")
                deff = metrics.get(f"deff#{key}")
                if (variance is None or variance < 0 or deff is None or deff < 0
                        or abs(variance - ses[key] ** 2) > max(1e-8, abs(variance) * 0.01)):
                    return SanitizerResult(
                        False, "method_result",
                        rejection_reason="survey regression variance/design-effect maps are inconsistent",
                    )
    if method.id == "reliability":
        reliability_keys = {"alpha", "omega_total"}
        reliability_metrics = {
            "item_count", "reversed_item_count", "min_item_rest_correlation",
            "bootstrap_replicates", "bootstrap_success_count",
        }
        estimates_map = cleaned_maps.get("estimates", {})
        if (set(estimates_map) != reliability_keys
                or set(lower) != reliability_keys
                or set(cleaned_maps.get("metrics", {})) != reliability_metrics
                or any(field in cleaned_maps for field in ("standard_errors", "p_values"))
                or out.get("uncertainty_type") != "bootstrap"
                or "seed" not in out):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="reliability requires alpha/omega bootstrap intervals and exact aggregate diagnostics",
            )
        reliability = cleaned_maps["metrics"]
        item_count = reliability["item_count"]
        reversed_count = reliability["reversed_item_count"]
        repetitions = reliability["bootstrap_replicates"]
        successes = reliability["bootstrap_success_count"]
        min_item_rest = reliability["min_item_rest_correlation"]
        if (any(not 0 <= estimates_map[key] <= 1 for key in reliability_keys)
                or any(not 0 <= lower[key] <= estimates_map[key] <= upper[key] <= 1
                       for key in reliability_keys)
                or item_count < 3 or item_count > 100 or item_count != int(item_count)
                or reversed_count < 0 or reversed_count > item_count
                or reversed_count != int(reversed_count)
                or repetitions < 200 or repetitions > 5000
                or repetitions != int(repetitions)
                or successes < math.ceil(0.9 * repetitions)
                or successes > repetitions or successes != int(successes)
                or not 0 <= min_item_rest <= 1
                or clean_diagnostics.get("item_count") != item_count
                or clean_diagnostics.get("omega_or_alpha_interval") != "pass"
                or clean_diagnostics.get("item_direction") != "pass"):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="reliability coefficients, intervals, item direction, or bootstrap stability are inconsistent",
            )
    if method.id == "confirmatory_factor_analysis":
        cfa_metrics = {
            "factor_count", "indicator_count", "loading_count",
            "degrees_of_freedom", "chi_square", "cfi", "tli", "rmsea", "srmr",
        }
        cfa_estimates = cleaned_maps.get("estimates", {})
        metric_map = cleaned_maps.get("metrics", {})
        if (not cfa_estimates or any(not key.startswith("loading_") for key in cfa_estimates)
                or set(metric_map) != cfa_metrics
                or any(not set(cleaned_maps.get(field, {})).issubset(cfa_estimates)
                       for field in ("standard_errors", "p_values", "ci_lower", "ci_upper"))
                or out.get("uncertainty_type") != "classical"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="CFA requires synthetic loadings and exact maintained fit-index aggregates")
        factors = metric_map["factor_count"]
        indicators = metric_map["indicator_count"]
        loadings = metric_map["loading_count"]
        if (factors < 1 or factors != int(factors)
                or indicators < 3 or indicators != int(indicators)
                or loadings != len(cfa_estimates) or loadings < 3 * factors
                or metric_map["degrees_of_freedom"] <= 0
                or metric_map["chi_square"] < 0
                or not -1 <= metric_map["cfi"] <= 1.2
                or not -1 <= metric_map["tli"] <= 1.2
                or not 0 <= metric_map["rmsea"] <= 10
                or not 0 <= metric_map["srmr"] <= 1
                or any(not _is_finite_number(clean_diagnostics.get(key))
                       for key in ("cfi", "tli", "rmsea", "srmr"))
                or any(abs(clean_diagnostics[key] - metric_map[key]) > 0.002
                       for key in ("cfi", "tli", "rmsea", "srmr"))):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="CFA fit indices or model dimensions are inconsistent")
    if method.id == "measurement_invariance":
        change_keys = {
            "metric_delta_cfi", "metric_delta_rmsea",
            "scalar_delta_cfi", "scalar_delta_rmsea",
        }
        invariance_metrics = {"group_count", "indicator_count"} | {
            f"{model}_{field}"
            for model in ("configural", "metric", "scalar")
            for field in ("cfi", "rmsea", "chisq", "df")
        }
        changes = cleaned_maps.get("estimates", {})
        nested_p = cleaned_maps.get("p_values", {})
        metric_map = cleaned_maps.get("metrics", {})
        if (set(changes) != change_keys
                or set(nested_p) != {"metric_nested", "scalar_nested"}
                or set(metric_map) != invariance_metrics
                or out.get("uncertainty_type") != "classical"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="measurement invariance requires exact nested-model aggregates")
        expected_changes = {
            "metric_delta_cfi": metric_map["configural_cfi"] - metric_map["metric_cfi"],
            "metric_delta_rmsea": metric_map["metric_rmsea"] - metric_map["configural_rmsea"],
            "scalar_delta_cfi": metric_map["metric_cfi"] - metric_map["scalar_cfi"],
            "scalar_delta_rmsea": metric_map["scalar_rmsea"] - metric_map["metric_rmsea"],
        }
        if (metric_map["group_count"] < 2 or metric_map["group_count"] != int(metric_map["group_count"])
                or metric_map["indicator_count"] < 3
                or metric_map["indicator_count"] != int(metric_map["indicator_count"])
                or not (0 < metric_map["configural_df"] < metric_map["metric_df"] < metric_map["scalar_df"])
                or any(metric_map[f"{model}_chisq"] < 0 for model in ("configural", "metric", "scalar"))
                or any(not -1 <= metric_map[f"{model}_cfi"] <= 1.2 for model in ("configural", "metric", "scalar"))
                or any(not 0 <= metric_map[f"{model}_rmsea"] <= 10 for model in ("configural", "metric", "scalar"))
                or any(abs(changes[key] - expected_changes[key]) > 0.002 for key in change_keys)
                or clean_diagnostics.get("configural_fit") != (
                    "pass" if metric_map["configural_cfi"] >= 0.90
                    and metric_map["configural_rmsea"] <= 0.08 else "warn"
                )
                or clean_diagnostics.get("metric_change") != (
                    "pass" if changes["metric_delta_cfi"] <= 0.01
                    and changes["metric_delta_rmsea"] <= 0.015 else "warn"
                )
                or clean_diagnostics.get("scalar_change") != (
                    "pass" if changes["scalar_delta_cfi"] <= 0.01
                    and changes["scalar_delta_rmsea"] <= 0.015 else "warn"
                )):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="measurement-invariance fit sequence or changes are inconsistent")
    if method.id == "latent_class":
        latent_metrics = {
            "class_count", "start_count", "stable_start_count",
            "min_expected_class_n", "normalized_entropy", "best_log_likelihood",
            "second_best_gap", "likelihood_tolerance", "aic", "bic",
        }
        proportions = cleaned_maps.get("estimates", {})
        metric_map = cleaned_maps.get("metrics", {})
        classes = metric_map.get("class_count")
        expected_keys = (
            {f"class_{index}" for index in range(1, int(classes) + 1)}
            if (isinstance(classes, (int, float)) and 2 <= classes <= 20
                and classes == int(classes))
            else set()
        )
        if (set(metric_map) != latent_metrics or set(proportions) != expected_keys
                or any(not 0 <= value <= 1 for value in proportions.values())
                or abs(sum(proportions.values()) - 1) > 0.01
                or metric_map.get("start_count", 0) < 5
                or metric_map["start_count"] != int(metric_map["start_count"])
                or metric_map["stable_start_count"] < 2
                or metric_map["stable_start_count"] > metric_map["start_count"]
                or metric_map["stable_start_count"] != int(metric_map["stable_start_count"])
                or metric_map["min_expected_class_n"] < config.min_n_regression
                or abs(metric_map["min_expected_class_n"]
                       - n * min(proportions.values(), default=0)) > 1.0
                or not 0 <= metric_map["normalized_entropy"] <= 1
                or metric_map["likelihood_tolerance"] <= 0
                or not 0 <= metric_map["second_best_gap"] <= metric_map["likelihood_tolerance"] + 0.002
                or clean_diagnostics.get("class_sizes") != "pass"
                or clean_diagnostics.get("entropy") != (
                    "pass" if metric_map["normalized_entropy"] >= 0.6 else "warn"
                )
                or clean_diagnostics.get("solution_stability") != "pass"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="latent-class solution lacks stable multi-start or class-support evidence")

    if method.id == "panel_fixed_effects":
        expected_metrics = {
            "within_r_squared", "min_within_variation_ratio",
            "fixed_effect_f_statistic", "fixed_effect_p_value",
            "entity_count", "period_count", "predictor_count",
        }
        metrics = cleaned_maps.get("metrics", {})
        estimates = cleaned_maps.get("estimates", {})
        coefficient_keys = set(estimates)
        inference_keys_match = (
            bool(coefficient_keys)
            and all(set(cleaned_maps.get(field, {})) == coefficient_keys
                    for field in ("standard_errors", "p_values", "ci_lower", "ci_upper"))
        )
        entities = metrics.get("entity_count")
        periods = metrics.get("period_count")
        predictors = metrics.get("predictor_count")
        within_ratio = metrics.get("min_within_variation_ratio")
        fixed_p = metrics.get("fixed_effect_p_value")
        structurally_valid = (
            raw.get("_via_helper") == "panel_fixed_effects_v1"
            and raw.get("design") == "panel_entity_fixed_effects"
            and out.get("uncertainty_type") == "cluster_robust"
            and set(metrics) == expected_metrics and inference_keys_match
            and isinstance(entities, (int, float))
            and entities == out.get("clusters") and out.get("records") == n
            and isinstance(periods, (int, float)) and periods >= 2
            and periods == int(periods) and entities * periods == n
            and predictors == len(coefficient_keys)
            and isinstance(within_ratio, (int, float)) and 0 < within_ratio <= 1
            and isinstance(metrics.get("within_r_squared"), (int, float))
            and metrics["within_r_squared"] <= 1
            and isinstance(metrics.get("fixed_effect_f_statistic"), (int, float))
            and metrics["fixed_effect_f_statistic"] >= 0
            and isinstance(fixed_p, (int, float)) and 0 <= fixed_p <= 1
            and clean_diagnostics.get("cluster_count") == entities
            and clean_diagnostics.get("cluster_size") == periods
            and clean_diagnostics.get("convergence") is True
            and clean_diagnostics.get("balanced_panel") == "pass"
            and clean_diagnostics.get("clustered_uncertainty") == "pass"
            and abs(clean_diagnostics.get("within_variation", -1) - within_ratio) <= 0.002
            and abs(clean_diagnostics.get("fixed_effect_test", -1) - fixed_p) <= 0.002
        )
        if not structurally_valid:
            return SanitizerResult(
                False, "method_result",
                rejection_reason="panel fixed-effects fit, within-variation proof, or clustered inference is inconsistent",
            )

    if method.id == "difference_in_differences":
        expected_metrics = {
            "control_pre_mean", "control_post_mean", "treated_pre_mean",
            "treated_post_mean", "raw_did", "entity_count", "period_count",
        }
        metrics = cleaned_maps.get("metrics", {})
        estimates = cleaned_maps.get("estimates", {})
        expected_did = (
            metrics.get("treated_post_mean", 0) - metrics.get("treated_pre_mean", 0)
            - metrics.get("control_post_mean", 0) + metrics.get("control_pre_mean", 0)
        )
        structurally_valid = (
            raw.get("_via_helper") == "difference_in_differences_v1"
            and out.get("design") == "two_by_two_panel_did"
            and out.get("estimand") == "att"
            and out.get("uncertainty_type") == "cluster_robust"
            and set(metrics) == expected_metrics and set(estimates) == {"att"}
            and all(set(cleaned_maps.get(field, {})) == {"att"}
                    for field in ("standard_errors", "p_values", "ci_lower", "ci_upper"))
            and out.get("records") == n and out.get("clusters", 0) * 2 == n
            and out.get("treated", 0) >= config.min_n_ttest_group
            and out.get("controls", 0) >= config.min_n_ttest_group
            and out.get("treated", 0) + out.get("controls", 0) == out.get("clusters")
            and metrics.get("entity_count") == out.get("clusters")
            and metrics.get("period_count") == 2
            and abs(metrics.get("raw_did", 0) - expected_did) <= 0.01
            and abs(estimates.get("att", 0) - metrics.get("raw_did", 0)) <= 0.01
            and clean_diagnostics.get("parallel_pretrends") == "not_applicable"
            and clean_diagnostics.get("treatment_timing") == "pass"
            and clean_diagnostics.get("balanced_two_period_panel") == "pass"
            and clean_diagnostics.get("clustered_uncertainty") == "pass"
            and clean_diagnostics.get("effect_uncertainty") == "pass"
            and clean_diagnostics.get("design_specific_falsification") == "not_applicable"
        )
        if not structurally_valid:
            return SanitizerResult(
                False, "method_result",
                rejection_reason="two-period DiD contrast, panel structure, or clustered inference is inconsistent",
            )

    if method.id == "probability_calibration":
        expected_metrics = {
            "feature_count", "brier_score", "uncalibrated_brier_score",
            "prevalence_brier_score", "uncalibrated_brier_improvement",
            "baseline_improvement", "calibration_slope", "calibration_intercept",
            "expected_calibration_error", "max_calibration_gap",
            "nonempty_calibration_bins", "minimum_calibration_bin_count",
            "minority_fraction", "roc_auc_context",
        }
        metrics = cleaned_maps.get("metrics", {})
        estimates = cleaned_maps.get("estimates", {})
        brier = metrics.get("brier_score")
        brier_value = float(brier) if isinstance(brier, (int, float)) else math.nan
        probability_metrics = (
            brier, metrics.get("uncalibrated_brier_score"),
            metrics.get("prevalence_brier_score"),
            metrics.get("expected_calibration_error"),
            metrics.get("max_calibration_gap"), metrics.get("minority_fraction"),
            metrics.get("roc_auc_context"),
        )
        calibration_ok = (
            isinstance(metrics.get("calibration_slope"), (int, float))
            and 0.8 <= metrics["calibration_slope"] <= 1.2
            and abs(metrics.get("calibration_intercept", 2)) <= 0.5
        )
        structurally_valid = (
            raw.get("_via_helper") == "probability_calibration_v1"
            and set(metrics) == expected_metrics and set(estimates) == {"brier_score"}
            and estimates.get("brier_score") == brier
            and all(isinstance(value, (int, float)) and 0 <= value <= 1
                    for value in probability_metrics)
            and metrics.get("feature_count", 0) >= 1
            and metrics.get("nonempty_calibration_bins", 0) >= 2
            and metrics.get("nonempty_calibration_bins", 0) <= 10
            and metrics.get("nonempty_calibration_bins") == int(metrics.get("nonempty_calibration_bins", -1))
            and metrics.get("minimum_calibration_bin_count", 0) >= 5
            and metrics.get("minimum_calibration_bin_count") == int(metrics.get("minimum_calibration_bin_count", -1))
            and abs(metrics.get("uncalibrated_brier_improvement", 0)
                    - (metrics.get("uncalibrated_brier_score", 0) - brier_value)) <= 0.002
            and abs(metrics.get("baseline_improvement", 0)
                    - (metrics.get("prevalence_brier_score", 0) - brier_value)) <= 0.002
            and out.get("evaluation_split") == "held_out"
            and out.get("split_strategy") == "train_test_calibration_cv"
            and out.get("training_observations", 0) >= config.min_n_regression
            and out.get("test_observations", 0) >= config.min_n_regression
            and out.get("training_observations", 0) + out.get("test_observations", 0) == n
            and 3 <= out.get("folds", 0) <= 10
            and out.get("bootstrap_replicates", 0) >= 200
            and out.get("uncertainty_type") == "bootstrap"
            and out.get("interval_method") == "heldout_case_bootstrap"
            and set(lower) == set(upper) == {"brier_score"}
            and out.get("baseline_model") == "uncalibrated_classifier_and_prevalence"
            and out.get("calibration_method") == "nested_sigmoid"
            and out.get("imbalance_strategy") in {"balanced_weight", "none"}
            and (metrics.get("minority_fraction", 0) >= 0.4
                 or out.get("imbalance_strategy") == "balanced_weight")
            and clean_diagnostics.get("held_out_performance") == "pass"
            and clean_diagnostics.get("split_integrity") == "pass"
            and clean_diagnostics.get("preprocessing_inside_split") == "pass"
            and clean_diagnostics.get("calibration_nested") == "pass"
            and clean_diagnostics.get("class_balance") == "pass"
            and clean_diagnostics.get("uncertainty") == "pass"
            and clean_diagnostics.get("baseline_comparison") == (
                "pass" if metrics.get("baseline_improvement", -1) >= 0 else "warn"
            )
            and clean_diagnostics.get("calibration") == (
                "pass" if calibration_ok else "warn"
            )
            and abs(clean_diagnostics.get("calibration_curve", -1)
                    - metrics.get("expected_calibration_error", -2)) <= 0.002
            and abs(clean_diagnostics.get("brier_score", -1) - brier) <= 0.002
        )
        if not structurally_valid:
            return SanitizerResult(
                False, "method_result",
                rejection_reason="probability calibration split, aggregate metrics, or nested-fit provenance is inconsistent",
            )

    if method.id in {"predictive_regression", "predictive_classification"}:
        if raw.get("_via_helper") != "predictive_workflow_v1":
            return SanitizerResult(
                False, "method_result",
                rejection_reason="predictive results require the typed leakage-resistant workflow",
            )
        metrics = cleaned_maps.get("metrics", {})
        estimates = cleaned_maps.get("estimates", {})
        held_out = out.get("evaluation_split") == "held_out"
        cross_validation = out.get("evaluation_split") == "cross_validation"
        if out.get("split_strategy") != (
            "train_validation_test" if held_out else "cross_validation"
        ):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="predictive split strategy is inconsistent",
            )
        if held_out:
            raw_counts = [
                out.get("training_observations"), out.get("validation_observations"),
                out.get("test_observations"),
            ]
            counts = [value for value in raw_counts if isinstance(value, int)]
            split_valid = (
                len(counts) == len(raw_counts)
                and all(value >= config.min_n_regression for value in counts)
                and sum(counts) == n and out.get("bootstrap_replicates", 0) >= 200
                and out.get("uncertainty_type") == "bootstrap"
                and out.get("interval_method") == "heldout_case_bootstrap"
                and set(lower) == set(upper) == set(estimates)
            )
        else:
            split_valid = (
                cross_validation and out.get("evaluated_observations") == n
                and 3 <= out.get("folds", 0) <= 10
                and out.get("bootstrap_replicates") == 0
                and "uncertainty_type" not in out and not lower and not upper
            )
        expected_diagnostics = {
            "held_out_performance": "pass", "split_integrity": "pass",
            "preprocessing_inside_split": "pass",
            "uncertainty": "pass" if held_out else "not_applicable",
        }
        base_valid = (
            split_valid and "seed" in out
            and out.get("baseline_model") == "simple_dummy"
            and all(clean_diagnostics.get(key) == value
                    for key, value in expected_diagnostics.items())
            and clean_diagnostics.get("baseline_comparison") == (
                "pass" if metrics.get("baseline_improvement", -1) >= 0 else "warn"
            )
            and metrics.get("feature_count", 0) >= 1
        )
        if method.id == "predictive_regression":
            expected_metrics = {
                "feature_count", "rmse", "mae", "r2", "baseline_rmse",
                "baseline_mae", "calibration_slope", "calibration_intercept",
                "baseline_improvement",
            } | ({"validation_rmse"} if held_out else set())
            rmse = metrics.get("rmse")
            baseline_rmse = metrics.get("baseline_rmse")
            calibration_ok = (
                isinstance(metrics.get("calibration_slope"), (int, float))
                and 0.8 <= metrics["calibration_slope"] <= 1.2
            )
            method_valid = (
                set(metrics) == expected_metrics and set(estimates) == {"rmse"}
                and estimates.get("rmse") == rmse
                and isinstance(rmse, (int, float)) and rmse >= 0
                and isinstance(baseline_rmse, (int, float)) and baseline_rmse >= 0
                and abs(metrics.get("baseline_improvement", 0)
                        - (baseline_rmse - rmse)) <= 0.01
                and metrics.get("mae", -1) >= 0 and metrics.get("baseline_mae", -1) >= 0
                and out.get("calibration_method") == "not_applicable"
                and out.get("imbalance_strategy") == "not_applicable"
                and clean_diagnostics.get("calibration") == (
                    "pass" if calibration_ok else "warn"
                )
            )
        else:
            expected_metrics = {
                "feature_count", "roc_auc", "average_precision",
                "balanced_accuracy", "brier_score", "baseline_brier_score",
                "baseline_auc", "calibration_slope", "calibration_intercept",
                "minority_fraction", "baseline_improvement",
            } | ({"validation_auc", "validation_brier"} if held_out else set())
            probabilities = [
                metrics.get("roc_auc"), metrics.get("average_precision"),
                metrics.get("balanced_accuracy"), metrics.get("brier_score"),
                metrics.get("baseline_brier_score"), metrics.get("minority_fraction"),
            ]
            calibration_ok = (
                isinstance(metrics.get("calibration_slope"), (int, float))
                and 0.8 <= metrics["calibration_slope"] <= 1.2
                and metrics.get("brier_score", 2) <= metrics.get("baseline_brier_score", -1)
            )
            method_valid = (
                set(metrics) == expected_metrics and set(estimates) == {"roc_auc"}
                and estimates.get("roc_auc") == metrics.get("roc_auc")
                and all(isinstance(value, (int, float)) and 0 <= value <= 1
                        for value in probabilities)
                and isinstance(metrics.get("baseline_auc"), (int, float))
                and 0 <= metrics["baseline_auc"] <= 1
                and abs(metrics.get("baseline_improvement", 0)
                        - (metrics.get("roc_auc", 0) - metrics["baseline_auc"])) <= 0.01
                and out.get("calibration_method") == "nested_sigmoid"
                and out.get("imbalance_strategy") in {"balanced_weight", "none"}
                and (metrics.get("minority_fraction", 0) >= 0.4
                     or out.get("imbalance_strategy") == "balanced_weight")
                and clean_diagnostics.get("discrimination") == metrics.get("roc_auc")
                and clean_diagnostics.get("class_balance") == "pass"
                and clean_diagnostics.get("calibration") == (
                    "pass" if calibration_ok else "warn"
                )
            )
        if not base_valid or not method_valid:
            return SanitizerResult(
                False, "method_result",
                rejection_reason="predictive workflow metrics, split, calibration, or baseline are inconsistent",
            )
    if method.family == "predictive" and out.get("evaluation_split") not in {
        "held_out", "cross_validation", "grouped", "rolling_origin",
    }:
        return SanitizerResult(False, "method_result",
                               rejection_reason="predictive results require out-of-sample evaluation")
    if method.id == "forecast_backtest" and out.get("evaluation_split") != "rolling_origin":
        return SanitizerResult(False, "method_result",
                               rejection_reason="forecast backtests require rolling-origin evaluation")
    time_series_metrics = {
        "stationarity_diagnostic": {"stationarity_statistic"},
        "seasonal_decomposition": {
            "trend_strength", "seasonal_strength", "residual_sd",
            "residual_variance_share",
        },
        "arima": {
            "rmse", "mae", "prediction_interval_coverage",
            "prediction_interval_mean_width", "nominal_coverage",
            "mean_forecast", "mean_actual", "aic", "bic", "ljung_box_p_value",
        },
        "exponential_smoothing": {
            "rmse", "mae", "prediction_interval_coverage",
            "prediction_interval_mean_width", "nominal_coverage",
            "mean_forecast", "mean_actual", "residual_sd", "ljung_box_p_value",
        },
        "interrupted_time_series": {"aic", "bic", "ljung_box_p_value"},
        "forecast_backtest": {
            "rmse", "mae", "prediction_interval_coverage",
            "prediction_interval_mean_width", "nominal_coverage",
            "mean_forecast", "mean_actual", "baseline_rmse", "origins",
        },
    }
    if method.id in time_series_metrics:
        if "frequency" not in out:
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} requires declared frequency")
        metrics = cleaned_maps.get("metrics", {})
        cadence_proof = {"cadence_min_ratio", "cadence_max_ratio", "time_span_steps"}
        missing_metrics = (time_series_metrics[method.id] | cadence_proof) - set(metrics)
        if missing_metrics:
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} is missing time-series metrics: {sorted(missing_metrics)}")
        if (clean_diagnostics.get("temporal_order") != "pass"
                or clean_diagnostics.get("regular_frequency") != "pass"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} requires ordered regular observations")
        if (abs(metrics["cadence_min_ratio"] - 1) > 1e-6
                or abs(metrics["cadence_max_ratio"] - 1) > 1e-6
                or abs(metrics["time_span_steps"] - (n - 1)) > max(1e-6, n * 1e-6)):
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} cadence proof is inconsistent")
    if method.id in {"arima", "exponential_smoothing", "forecast_backtest"}:
        metrics = cleaned_maps["metrics"]
        if (any(field not in out for field in ("training_observations", "test_observations"))
                or out["training_observations"] + out["test_observations"] != n):
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} temporal split counts are inconsistent")
        if method.id != "forecast_backtest" and out.get("evaluation_split") != "held_out":
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} requires chronological held-out evaluation")
        if "interval_method" not in out:
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} must declare its prediction-interval method")
        if (not 0 <= metrics["prediction_interval_coverage"] <= 1
                or metrics["prediction_interval_mean_width"] < 0
                or not 0 < metrics["nominal_coverage"] < 1
                or metrics["rmse"] < 0 or metrics["mae"] < 0):
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} forecast metrics are outside valid domains")
        if clean_diagnostics.get("temporal_order") != "pass" or clean_diagnostics.get("holdout_leakage") != "pass":
            return SanitizerResult(False, "method_result",
                                   rejection_reason=f"{method.id} requires ordered leakage-free evaluation")
    if method.id == "arima" and (
        clean_diagnostics.get("ar_stationarity") not in {"pass", "warn"}
        or clean_diagnostics.get("ma_invertibility") not in {"pass", "warn"}
    ):
        return SanitizerResult(False, "method_result",
                               rejection_reason="ARIMA requires separate AR-stationarity and MA-invertibility diagnostics")
    if method.id == "exponential_smoothing":
        metrics = cleaned_maps["metrics"]
        if ("first_interval_width" not in metrics or "last_interval_width" not in metrics
                or metrics["first_interval_width"] <= 0
                or metrics["last_interval_width"] + 1e-8 < metrics["first_interval_width"]):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="exponential smoothing requires non-shrinking state-space interval widths")
    if method.id == "forecast_backtest":
        if out.get("folds") != out.get("test_observations"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="forecast backtest folds must equal evaluated origins")
        if cleaned_maps["metrics"]["origins"] != out["folds"]:
            return SanitizerResult(False, "method_result",
                                   rejection_reason="forecast backtest origins metric is inconsistent")
    if method.id == "seasonal_decomposition":
        metrics = cleaned_maps["metrics"]
        if (not 0 <= metrics["trend_strength"] <= 1
                or not 0 <= metrics["seasonal_strength"] <= 1
                or metrics["residual_sd"] < 0
                or metrics["residual_variance_share"] < 0):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="decomposition metrics are outside valid domains")
    if method.id == "stationarity_diagnostic":
        p_value = cleaned_maps["metrics"].get("adf_p_value")
        if p_value is not None and not 0 <= p_value <= 1:
            return SanitizerResult(False, "method_result",
                                   rejection_reason="stationarity p-values must be probabilities")
    if method.id == "interrupted_time_series":
        pretrend_p = cleaned_maps["metrics"].get("pretrend_stability_p_value")
        expected_pretrend = "pass" if pretrend_p is not None and pretrend_p > .05 else "warn"
        if (pretrend_p is None or not 0 <= pretrend_p <= 1
                or clean_diagnostics.get("pre_intervention_trend") != expected_pretrend):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="interrupted time series requires a consistent pretrend stability diagnostic")
        if (any(field not in out for field in ("pre_periods", "post_periods"))
                or out["pre_periods"] + out["post_periods"] != n):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="interrupted-series period counts are inconsistent")
        if clean_diagnostics.get("temporal_order") != "pass":
            return SanitizerResult(False, "method_result",
                                   rejection_reason="interrupted series requires ordered observations")
        if not {"level_change", "slope_change"}.issubset(cleaned_maps.get("estimates", {})):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="interrupted series requires level and slope changes")
    if method.id == "multiple_testing_correction" and out.get("multiple_testing") in {None, "none"}:
        return SanitizerResult(False, "method_result",
                               rejection_reason="multiple-testing results must name the applied correction")
    if method.id == "descriptive_confidence_interval":
        estimate_keys = set(cleaned_maps.get("estimates", {}))
        if (not estimate_keys
                or set(cleaned_maps.get("standard_errors", {})) != estimate_keys
                or set(lower) != estimate_keys):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "descriptive confidence intervals require matching estimate, "
                    "standard-error, and confidence-bound keys"
                ),
            )
        missingness = clean_diagnostics.get("missingness")
        effective_n = clean_diagnostics.get("effective_sample_size")
        confidence = clean_diagnostics.get("confidence_level")
        if ((isinstance(missingness, (int, float)) and missingness < 0)
                or effective_n != n
                or not isinstance(confidence, (int, float))
                or not 0 < confidence < 1):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "descriptive confidence-interval diagnostics require "
                    "non-negative missingness, effective_sample_size=n, and "
                    "0<confidence_level<1"
                ),
            )
    if method.id in {
        "nonparametric_test", "proportion_test", "anova", "ancova",
        "repeated_measures_test",
    } and not cleaned_maps.get("p_values"):
        return SanitizerResult(
            False, "method_result",
            rejection_reason=f"{method.id} requires at least one aggregate p-value",
        )
    if method.id in {"anova", "ancova", "repeated_measures_test"} and not cleaned_maps.get("metrics"):
        return SanitizerResult(
            False, "method_result",
            rejection_reason=f"{method.id} requires at least one aggregate test statistic",
        )
    if method.id == "repeated_measures_test" and any(
        field not in out for field in ("subjects", "records")
    ):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="repeated-measures results must distinguish subjects and records",
        )
    if method.id == "repeated_measures_test":
        cluster_count = clean_diagnostics.get("cluster_count")
        cluster_size = clean_diagnostics.get("cluster_size")
        if (cluster_count != out["subjects"]
                or not isinstance(cluster_size, (int, float))
                or cluster_size <= 0):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "repeated-measures diagnostics must bind cluster_count to "
                    "subjects and report a positive cluster_size"
                ),
            )
    if method.id == "multiple_testing_correction":
        raw_p = cleaned_maps.get("estimates", {})
        adjusted_p = cleaned_maps.get("p_values", {})
        metrics = cleaned_maps.get("metrics", {})
        if not raw_p or set(raw_p) != set(adjusted_p):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "multiple-testing results require matching raw and adjusted "
                    "p-value hypothesis keys"
                ),
            )
        if any(not 0 <= value <= 1 for value in raw_p.values()):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="multiple-testing raw p-values must be probabilities",
            )
        hypothesis_count = metrics.get("hypothesis_count")
        rejection_count = metrics.get("rejection_count")
        alpha = metrics.get("alpha")
        if (hypothesis_count != len(raw_p)
                or rejection_count is None or not 0 <= rejection_count <= len(raw_p)
                or alpha is None or not 0 < alpha < 1):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "multiple-testing metrics must report consistent hypothesis_count, "
                    "rejection_count, and alpha"
                ),
            )
    if method.id == "missingness_pattern":
        metrics = cleaned_maps.get("metrics", {})
        required_metrics = {
            "variable_count", "missing_fraction", "complete_case_rate",
            "complete_case_warning_threshold", "missingness_pattern_count",
            "largest_pattern_fraction",
        }
        variables = metrics.get("variable_count")
        missing_fraction = metrics.get("missing_fraction")
        complete_rate = metrics.get("complete_case_rate")
        threshold = metrics.get("complete_case_warning_threshold")
        patterns = metrics.get("missingness_pattern_count")
        largest = metrics.get("largest_pattern_fraction")
        warning = clean_diagnostics.get("complete_case_warning")
        expected_warning = (
            "warn" if isinstance(complete_rate, (int, float))
            and isinstance(threshold, (int, float))
            and 1 - complete_rate >= threshold else "pass"
        )
        if (set(metrics) != required_metrics
                or not isinstance(variables, (int, float))
                or variables != int(variables) or not 1 <= variables <= 100
                or not isinstance(missing_fraction, (int, float))
                or not 0 <= missing_fraction <= 1
                or not isinstance(complete_rate, (int, float))
                or not 0 <= complete_rate <= 1
                or clean_diagnostics.get("complete_case_rate") != complete_rate
                or not isinstance(threshold, (int, float))
                or not 0 < threshold < 1
                or warning != expected_warning
                or clean_diagnostics.get("missingness_pattern") != "pass"
                or not isinstance(patterns, (int, float))
                or patterns != int(patterns) or not 1 <= patterns <= n
                or not isinstance(largest, (int, float))
                or not 1 / n <= largest <= 1):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="missingness-pattern aggregates or complete-case warning are inconsistent",
            )
    if method.id == "single_imputation":
        metrics = cleaned_maps.get("metrics", {})
        required_metrics = {
            "feature_count", "output_feature_count", "missing_fraction",
            "affected_row_fraction", "imputed_cell_count",
        }
        features = metrics.get("feature_count")
        output_features = metrics.get("output_feature_count")
        missing_fraction = metrics.get("missing_fraction")
        affected_fraction = metrics.get("affected_row_fraction")
        imputed_cells = metrics.get("imputed_cell_count")
        inferential_maps = {
            "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper",
        }
        if (set(metrics) != required_metrics
                or inferential_maps.intersection(cleaned_maps)
                or "uncertainty_type" in out
                or out.get("imputation_model") != "simple_deterministic"
                or out.get("imputation_scope") not in {
                    "prediction_preprocessing", "deterministic_nuisance_covariate",
                }
                or any(clean_diagnostics.get(key) != "pass" for key in (
                    "missingness_pattern", "imputation_scope",
                    "inferential_uncertainty_not_claimed",
                ))
                or not isinstance(features, (int, float))
                or features != int(features) or not 1 <= features <= 100
                or output_features != features
                or not isinstance(imputed_cells, (int, float))
                or imputed_cells != int(imputed_cells)
                or not 1 <= imputed_cells <= n * features
                or not isinstance(missing_fraction, (int, float))
                or abs(missing_fraction - imputed_cells / (n * features)) > 0.01
                or not isinstance(affected_fraction, (int, float))
                or not 0 < affected_fraction <= 1
                or affected_fraction + 0.01 < imputed_cells / (n * features)
                or affected_fraction > min(1.0, imputed_cells / n) + 0.01):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="single-imputation audit exceeds its deterministic preprocessing boundary",
            )
    if method.id == "multiple_imputation":
        estimates = cleaned_maps.get("estimates", {})
        ses = cleaned_maps.get("standard_errors", {})
        p_values = cleaned_maps.get("p_values", {})
        metrics = cleaned_maps.get("metrics", {})
        mi_keys = set(estimates)
        map_keys_match = (
            bool(mi_keys) and set(ses) == mi_keys and set(p_values) == mi_keys
            and set(lower) == mi_keys and set(upper) == mi_keys
        )
        required_summary = {
            "parameter_count", "missing_fraction", "complete_case_rate",
            "missingness_pattern_count", "max_fraction_missing_information",
            "mean_fraction_missing_information",
            "max_lambda_missing_information", "mean_between_imputation_variance",
            "imputed_mean_trace_drift", "variable_count",
        }
        component_keys = {
            f"{prefix}#{key}" for key in mi_keys
            for prefix in ("within", "between", "lambda", "fmi", "df", "complete_df")
        }
        components_valid = True
        for key in mi_keys:
            within = metrics.get(f"within#{key}")
            between = metrics.get(f"between#{key}")
            fmi = metrics.get(f"fmi#{key}")
            lambda_missing = metrics.get(f"lambda#{key}")
            degrees = metrics.get(f"df#{key}")
            complete_df = metrics.get(f"complete_df#{key}")
            if (not isinstance(within, (int, float)) or within < 0
                    or not isinstance(between, (int, float)) or between < 0
                    or not isinstance(fmi, (int, float)) or not 0 <= fmi <= 1
                    or not isinstance(lambda_missing, (int, float))
                    or not 0 <= lambda_missing <= 1
                    or not isinstance(degrees, (int, float)) or degrees <= 0
                    or not isinstance(complete_df, (int, float)) or complete_df <= 0
                    or ses.get(key, -1) <= 0):
                components_valid = False
                continue
            total = within + (1 + 1 / out.get("imputations", 1)) * between
            missing_variance = (1 + 1 / out.get("imputations", 1)) * between
            expected_lambda = missing_variance / total
            ratio = missing_variance / max(within, 1e-300)
            old_df = ((out.get("imputations", 1) - 1) * (1 + 1 / max(ratio, 1e-300)) ** 2)
            observed_df = ((complete_df + 1) / (complete_df + 3)) * complete_df * (1 - expected_lambda)
            expected_df = (
                observed_df if missing_variance <= 1e-15
                else 1 / (1 / old_df + 1 / observed_df)
            )
            expected_fmi = (ratio + 2 / (expected_df + 3)) / (ratio + 1)
            if (abs(ses[key] ** 2 - total) > max(1e-7, total * 0.02)
                    or abs(lambda_missing - expected_lambda) > 0.02
                    or abs(degrees - expected_df) > max(0.02, expected_df * 0.02)
                    or abs(fmi - expected_fmi) > 0.02):
                components_valid = False
        fmis_optional = [metrics.get(f"fmi#{key}") for key in mi_keys]
        fmis = [value for value in fmis_optional if value is not None]
        lambdas_optional = [metrics.get(f"lambda#{key}") for key in mi_keys]
        lambdas = [value for value in lambdas_optional if value is not None]
        betweens_optional = [metrics.get(f"between#{key}") for key in mi_keys]
        betweens = [value for value in betweens_optional if value is not None]
        if (not map_keys_match or set(metrics) != required_summary | component_keys
                or not components_valid
                or out.get("uncertainty_type") != "multiple_imputation"
                or out.get("imputation_model") != "mice_predictive_mean_matching"
                or out.get("imputations", 0) < 2 or out.get("burn_in", 0) < 1
                or out.get("matching_donors", 0) < 1
                or clean_diagnostics.get("missingness_pattern") != "pass"
                or clean_diagnostics.get("seed_recorded") != "pass"
                or clean_diagnostics.get("rubin_pooling") != "pass"
                or not isinstance(clean_diagnostics.get("imputation_trace_stability"), (int, float))
                or clean_diagnostics["imputation_trace_stability"] < 0
                or clean_diagnostics["imputation_trace_stability"]
                    != metrics.get("imputed_mean_trace_drift")
                or not fmis or len(fmis) != len(fmis_optional)
                or len(lambdas) != len(lambdas_optional)
                or metrics.get("parameter_count") != len(mi_keys)
                or metrics.get("max_fraction_missing_information") != max(fmis)
                or metrics.get("max_lambda_missing_information")
                    != max(lambdas)
                or abs(metrics.get("mean_fraction_missing_information", -1)
                       - sum(fmis) / len(fmis)) > 0.01
                or not betweens or len(betweens) != len(betweens_optional)
                or abs(metrics.get("mean_between_imputation_variance", -1)
                       - sum(betweens) / len(betweens)) > 0.01
                or clean_diagnostics.get("between_imputation_variance")
                    != metrics.get("mean_between_imputation_variance")
                or clean_diagnostics.get("fraction_missing_information")
                    != metrics.get("max_fraction_missing_information")
                or not 0 <= metrics.get("missing_fraction", -1) <= 1
                or not 0 <= metrics.get("complete_case_rate", -1) <= 1
                or metrics.get("missingness_pattern_count", 0) < 1
                or metrics.get("variable_count", 0) < 1):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="multiple-imputation result is not a coherent Rubin-pooled fit",
            )
    if method.id == "mnar_sensitivity":
        estimates = cleaned_maps.get("estimates", {})
        ses = cleaned_maps.get("standard_errors", {})
        p_values = cleaned_maps.get("p_values", {})
        metrics = cleaned_maps.get("metrics", {})
        mnar_keys = [f"scenario_{index + 1}" for index in range(len(estimates))]
        same_keys = (
            list(sorted(estimates, key=lambda key: int(key.split("_")[-1]))) == mnar_keys
            if estimates and all(
                re.fullmatch(r"scenario_[1-9][0-9]*", key) for key in estimates
            ) else False
        )
        same_keys = same_keys and all(
            set(mapping) == set(mnar_keys) for mapping in (ses, p_values, lower, upper)
        )
        delta_optional = [metrics.get(f"delta#{key}") for key in mnar_keys]
        deltas = [value for value in delta_optional if value is not None]
        mnar_fmi_optional = [metrics.get(f"fmi#{key}") for key in mnar_keys]
        mnar_fmis = [value for value in mnar_fmi_optional if value is not None]
        classifications = [
            -1 if upper[key] < 0 else (1 if lower[key] > 0 else 0)
            for key in mnar_keys
        ] if same_keys else []
        mnar_required_metrics = {
            "scenario_count", "delta_min", "delta_max", "baseline_estimate",
            "estimate_range", "max_fraction_missing_information",
            "delta_applied_fraction",
            *(f"delta#{key}" for key in mnar_keys),
            *(f"fmi#{key}" for key in mnar_keys),
        }
        if (not 3 <= len(mnar_keys) <= 21 or not same_keys
                or set(metrics) != mnar_required_metrics
                or out.get("uncertainty_type") != "multiple_imputation"
                or out.get("imputation_model") != "mice_predictive_mean_matching"
                or out.get("mnar_model") != "delta_adjusted_pattern_mixture"
                or out.get("imputations", 0) < 2 or out.get("burn_in", 0) < 1
                or out.get("matching_donors", 0) < 1
                or "seed" not in out
                or clean_diagnostics.get("delta_grid") != "pass"
                or clean_diagnostics.get("baseline_included") != "pass"
                or clean_diagnostics.get("sensitivity_parameter_justification") != "warn"
                or len(deltas) != len(delta_optional)
                or deltas != sorted(deltas) or len(set(deltas)) != len(deltas)
                or not (deltas[0] < 0 < deltas[-1]) or 0 not in deltas
                or len(mnar_fmis) != len(mnar_fmi_optional)
                or any(not 0 <= value <= 1 for value in mnar_fmis)
                or metrics.get("scenario_count") != len(mnar_keys)
                or metrics.get("delta_min") != deltas[0]
                or metrics.get("delta_max") != deltas[-1]
                or metrics.get("baseline_estimate")
                    != estimates[mnar_keys[deltas.index(0)]]
                or abs(metrics.get("estimate_range", -1)
                       - (max(estimates.values()) - min(estimates.values()))) > 0.01
                or metrics.get("max_fraction_missing_information") != max(mnar_fmis)
                or not 0 < metrics.get("delta_applied_fraction", 0) < 1
                or clean_diagnostics.get("conclusion_stability")
                    != (len(set(classifications)) == 1)):
            return SanitizerResult(
                False, "method_result",
                rejection_reason="MNAR sensitivity result has an incoherent pooled delta grid",
            )
    if method.id == "geospatial_analysis":
        metrics = cleaned_maps.get("metrics", {})
        geospatial_required = {
            "moran_i", "expected_moran_i", "permutation_p_value", "permutation_mcse",
            "neighbor_links", "mean_neighbors", "island_fraction",
            "distance_threshold_crs_units", "distance_threshold_metres",
            "crs_linear_unit_to_metre",
        }
        p_value = metrics.get("permutation_p_value")
        reps = out.get("replicates")
        links = metrics.get("neighbor_links")
        if (set(metrics) != geospatial_required or any(key in cleaned_maps for key in (
                "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper"))
                or out.get("spatial_weight_rule") != "distance_band_binary"
                or out.get("crs_epsg", 0) <= 0 or reps is None or not 199 <= reps <= 9999
                or "seed" not in out or clean_diagnostics.get("crs_validity") != "pass"
                or clean_diagnostics.get("spatial_weights") != "pass"
                or clean_diagnostics.get("privacy_aggregation") != "pass"
                or not isinstance(p_value, (int, float)) or not 0 <= p_value <= 1
                or clean_diagnostics.get("spatial_autocorrelation") != p_value
                or not isinstance(links, (int, float)) or links != int(links)
                or not 1 <= links <= n * (n - 1) / 2
                or abs(metrics.get("mean_neighbors", -1) - 2 * links / n) > .01
                or not 0 <= metrics.get("island_fraction", -1) <= 1
                or metrics.get("distance_threshold_crs_units", 0) <= 0
                or metrics.get("crs_linear_unit_to_metre", 0) <= 0
                or abs(metrics.get("distance_threshold_metres", -1)
                       - metrics["distance_threshold_crs_units"] * metrics["crs_linear_unit_to_metre"])
                    > max(.01, metrics.get("distance_threshold_metres", 0) * .01)
                or abs(metrics.get("permutation_mcse", -1)
                       - math.sqrt(p_value * (1 - p_value) / (reps + 1))) > .01):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="geospatial CRS, weights, permutation, or aggregate privacy contract is inconsistent")
    if method.id == "network_analysis":
        metrics = cleaned_maps.get("metrics", {})
        network_required = {"node_count", "edge_count", "density", "mean_degree", "component_count",
                    "largest_component_fraction", "isolate_fraction", "transitivity"}
        edges = metrics.get("edge_count")
        if (set(metrics) != network_required or any(key in cleaned_maps for key in (
                "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper"))
                or metrics.get("node_count") != n or not isinstance(edges, (int, float))
                or edges != int(edges) or not 1 <= edges <= n * (n - 1) / 2
                or abs(metrics.get("density", -1) - 2 * edges / (n * (n - 1))) > .01
                or abs(metrics.get("mean_degree", -1) - 2 * edges / n) > .01
                or metrics.get("component_count", 0) != int(metrics.get("component_count", .5))
                or not 1 <= metrics.get("component_count", 0) <= n
                or not 0 < metrics.get("largest_component_fraction", 0) <= 1
                or not 0 <= metrics.get("isolate_fraction", -1) <= 1
                or not 0 <= metrics.get("transitivity", -1) <= 1
                or clean_diagnostics != {"graph_definition": "pass", "graph_symmetry": "pass",
                    "dependence_aware_uncertainty": "not_applicable", "privacy_aggregation": "pass"}):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="network result must be a coherent aggregate-only simple graph summary")
    if method.id == "text_analysis":
        metrics = cleaned_maps.get("metrics", {})
        text_required = {"document_count", "vocabulary_size", "matrix_sparsity", "mean_document_norm",
                    "initialization_stability_ari", "resampling_stability_ari",
                    "cluster_count", "minimum_cluster_fraction"}
        if (set(metrics) != text_required or any(key in cleaned_maps for key in (
                "estimates", "standard_errors", "p_values", "ci_lower", "ci_upper"))
                or "seed" not in out or metrics.get("document_count") != n
                or not 1 <= metrics.get("vocabulary_size", 0) <= 10000
                or not 0 <= metrics.get("matrix_sparsity", -1) < 1
                or not 0 < metrics.get("mean_document_norm", 0) <= 1.01
                or not -1 <= metrics.get("initialization_stability_ari", -2) <= 1
                or not -1 <= metrics.get("resampling_stability_ari", -2) <= 1
                or out.get("stability_type") != "document_resampling"
                or clean_diagnostics.get("held_out_or_stability_check") != metrics.get("resampling_stability_ari")
                or clean_diagnostics.get("tokenization_specification") != "pass"
                or clean_diagnostics.get("document_privacy") != "pass"
                or clean_diagnostics.get("vocabulary_privacy") != "pass"
                or metrics.get("cluster_count", 0) != int(metrics.get("cluster_count", .5))
                or not 2 <= metrics.get("cluster_count", 0) <= min(10, n // 5)
                or not 0 < metrics.get("minimum_cluster_fraction", 0) <= 1):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="text result must be a stable aggregate with no documents or vocabulary")
    if method.id == "bayesian_model":
        rhat = clean_diagnostics.get("rhat"); bulk = clean_diagnostics.get("bulk_ess")
        tail = clean_diagnostics.get("tail_ess"); divergences = clean_diagnostics.get("divergences")
        ppc = clean_diagnostics.get("posterior_predictive_check")
        estimate_keys = set(cleaned_maps.get("estimates", {}))
        metrics = cleaned_maps.get("metrics", {})
        bayesian_required_metrics = {
            "chains", "draws_per_chain", "parameter_count",
            "posterior_predictive_replicates",
        }
        chains = metrics.get("chains")
        draws = metrics.get("draws_per_chain")
        posterior_replicates = metrics.get("posterior_predictive_replicates")
        bayesian_rejection = (
            "Bayesian output requires validated R-hat, ESS, zero divergences, "
            "and posterior predictive evidence"
        )
        # Narrow every required scalar independently.  This also makes a
        # malformed diagnostic fail before any arithmetic or ``int`` coercion.
        if not _is_finite_number(rhat):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(bulk):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(tail):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(divergences):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(ppc):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(chains):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(draws):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if not _is_finite_number(posterior_replicates):
            return SanitizerResult(False, "method_result", rejection_reason=bayesian_rejection)
        if (not estimate_keys or set(lower) != estimate_keys or set(upper) != estimate_keys
                or out.get("uncertainty_type") != "posterior"
                or set(metrics) != bayesian_required_metrics
                or chains != int(chains) or chains < 4
                or draws != int(draws) or draws < 100
                or metrics.get("parameter_count") != len(estimate_keys)
                or posterior_replicates != chains * draws
                or not 0.9 <= rhat < 1.01 or bulk < 400 or tail < 400
                or divergences != int(divergences)
                or not 0 <= divergences <= posterior_replicates
                or divergences != 0 or not .01 <= ppc <= .99):
            return SanitizerResult(
                False, "method_result", rejection_reason=bayesian_rejection,
            )
    if method.id == "power_precision":
        estimates = cleaned_maps.get("estimates", {}); metrics = cleaned_maps.get("metrics", {})
        power_keys = [f"scenario_{index}" for index in range(1, len(estimates) + 1)]
        expected_metrics = {"scenario_count", "alpha", "target_power", "allocation_ratio",
                            *(f"effect_size#{key}" for key in power_keys),
                            *(f"group1_n#{key}" for key in power_keys),
                            *(f"group2_n#{key}" for key in power_keys)}
        effect_optional = [metrics.get(f"effect_size#{key}") for key in power_keys]
        effects = [value for value in effect_optional if value is not None]
        coherent = (bool(power_keys) and set(estimates) == set(power_keys)
                    and set(metrics) == expected_metrics)
        for key in power_keys:
            coherent = coherent and estimates.get(key) == metrics.get(f"group1_n#{key}", -1) + metrics.get(f"group2_n#{key}", -1)
        if (raw.get("_via_helper") != "power_precision_v1"
                or out.get("test_alternative") not in {"two_sided", "larger"}
                or not coherent
                or len(effects) != len(effect_optional)
                or any(not .01 <= value <= 5 for value in effects)
                or effects != sorted(set(effects))
                or metrics.get("scenario_count") != len(power_keys)
                or not 0 < metrics.get("alpha", 0) < 1 or not .5 < metrics.get("target_power", 0) < 1
                or clean_diagnostics != {"effect_size_scenarios": "pass", "alpha_and_power": "pass",
                    "prospective_design": "pass"}):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="power calculation must bind prospective effect, alpha, power, and allocation scenarios")
    if method.id == "simulation_design":
        metrics = cleaned_maps.get("metrics", {})
        estimate = cleaned_maps.get("estimates", {}).get("empirical_power")
        reps = out.get("replicates")
        rejected = metrics.get("rejection_count")
        mcse = metrics.get("monte_carlo_standard_error")
        simulation_required = {"effect_size", "group_n", "alpha", "replications", "rejection_count",
                    "monte_carlo_standard_error", "analytic_power", "absolute_analytic_difference"}
        if (raw.get("_via_helper") != "simulation_design_v1"
                or set(metrics) != simulation_required
                or set(cleaned_maps.get("estimates", {})) != {"empirical_power"}
                or set(lower) != {"empirical_power"} or set(upper) != {"empirical_power"}
                or not isinstance(reps, int) or not 1000 <= reps <= 20000
                or metrics.get("replications") != reps
                or out.get("interval_method") != "clopper_pearson_binomial"
                or not .01 <= metrics.get("effect_size", 0) <= 5
                or "seed" not in out or not isinstance(estimate, (int, float))
                or not isinstance(rejected, (int, float)) or rejected != int(rejected)
                or not 0 <= rejected <= reps or abs(estimate - rejected / reps) > .002
                or not isinstance(mcse, (int, float))
                or abs(mcse - math.sqrt(estimate * (1 - estimate) / reps)) > .002
                or abs(metrics.get("absolute_analytic_difference", -1)
                       - abs(estimate - metrics.get("analytic_power", -2))) > .002
                or not lower["empirical_power"] <= estimate <= upper["empirical_power"]
                or clean_diagnostics.get("seed_recorded") != "pass"
                or clean_diagnostics.get("replication_count") != reps
                or clean_diagnostics.get("monte_carlo_standard_error") != mcse
                or clean_diagnostics.get("scenario_sensitivity") != "warn"):
            return SanitizerResult(False, "method_result",
                                   rejection_reason="simulation design must bind seed, replications, Monte Carlo error, and prospective assumptions")
    if method.id in {"multiple_imputation", "simulation_design", "reliability"} and "seed" not in out:
        return SanitizerResult(False, "method_result",
                               rejection_reason=f"{method.id} requires a recorded random seed")
    if method.id == "multiple_imputation" and out.get("imputations", 0) < 2:
        return SanitizerResult(False, "method_result",
                               rejection_reason="multiple imputation requires at least two imputations")
    if method.family == "survival" and any(field not in out for field in ("subjects", "events", "records")):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="survival method results must distinguish subjects, events, and records",
        )
    longitudinal_fits = {
        "growth_curve", "gee", "panel_fixed_effects", "panel_random_effects",
    }
    if method.id in longitudinal_fits:
        if any(field not in out for field in ("clusters", "records")):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    f"{method.id} must distinguish cluster and record counts"
                ),
            )
        cluster_count = clean_diagnostics.get("cluster_count")
        cluster_size = clean_diagnostics.get("cluster_size")
        if (cluster_count != out["clusters"]
                or out["records"] != n
                or not isinstance(cluster_size, (int, float))
                or cluster_size <= 0
                or abs(cluster_size * out["clusters"] - n) > 1.0):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    f"{method.id} diagnostics do not match its cluster/record counts"
                ),
            )
        if clean_diagnostics.get("convergence") in {False, "fail"}:
            return SanitizerResult(
                False, "method_result",
                rejection_reason=f"{method.id} did not converge",
            )
    if (method.id == "growth_curve"
            and clean_diagnostics.get("random_effect_structure") != "pass"):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="growth-curve random-effect structure was not fitted",
        )
    if method.id == "gee" and out.get("uncertainty_type") != "robust":
        return SanitizerResult(
            False, "method_result",
            rejection_reason="GEE results require robust sandwich uncertainty",
        )
    if method.id == "gee":
        sensitivity = clean_diagnostics.get("working_correlation_sensitivity")
        reported = cleaned_maps.get("metrics", {}).get(
            "working_correlation_max_abs_change"
        )
        if isinstance(sensitivity, (int, float)):
            if sensitivity < 0 or reported is None or abs(reported - sensitivity) > 0.002:
                return SanitizerResult(
                    False, "method_result",
                    rejection_reason=(
                        "GEE working-correlation sensitivity must match its "
                        "reported coefficient-change metric"
                    ),
                )
    if method.id == "panel_random_effects":
        hausman = clean_diagnostics.get("hausman")
        metric = cleaned_maps.get("metrics", {}).get("hausman_p_value")
        if (not isinstance(hausman, (int, float)) or not 0 <= hausman <= 1
                or metric is None or abs(metric - hausman) > 0.002):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "panel random-effects results require a matching Hausman p-value"
                ),
            )
    if method.family == "survival":
        subjects = out.get("subjects")
        events = out.get("events")
        records = out.get("records")
        if (subjects is None or events is None or records is None
                or records != n or subjects > records or events > records
                or clean_diagnostics.get("subject_count") != subjects
                or clean_diagnostics.get("event_count") != events):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "survival diagnostics do not match subject, event, and record counts"
                ),
            )
    if method.id == "competing_risks":
        cif = cleaned_maps.get("estimates", {})
        if (clean_diagnostics.get("cause_specific_events") != "pass"
                or not cif
                or any(not key.startswith("cause_") or not key.endswith("_final")
                       for key in cif)
                or any(not 0 <= value <= 1 for value in cif.values())
                or sum(cif.values()) > 1.001):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    "competing-risks results require supported cause-specific "
                    "events and final cumulative-incidence probabilities"
                ),
            )
    if method.id in {"recurrent_events", "time_varying_survival"}:
        if (out.get("clusters") != out.get("subjects")
                or out.get("records", 0) <= out.get("subjects", 0)
                or out.get("uncertainty_type") != "cluster_robust"):
            return SanitizerResult(
                False, "method_result",
                rejection_reason=(
                    f"{method.id} requires subject-clustered uncertainty"
                ),
            )
    if (method.id == "recurrent_events"
            and clean_diagnostics.get("within_subject_dependence") != "pass"):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="recurrent-event data contain no supported recurrence",
        )
    if (method.id == "time_varying_survival"
            and clean_diagnostics.get("interval_integrity") != "pass"):
        return SanitizerResult(
            False, "method_result",
            rejection_reason="time-varying survival intervals failed integrity checks",
        )

    # Diagnostic-only methods are legitimate, but every other method must
    # return at least one aggregate estimate or metric.
    diagnostic_only = {"stationarity_diagnostic", "missingness_pattern"}
    if not any_quantities and method.id not in diagnostic_only:
        return SanitizerResult(False, "method_result",
                               rejection_reason="method_result has no aggregate estimates or metrics")
    return SanitizerResult(True, "method_result", sanitized=out,
                           transformations=transformations[:_COLLECT_ALLOWED_LOG_CAP])

_HANDlerFn = Callable[[dict[str, Any], SDCConfig], SanitizerResult]

_HANDLERS: dict[str, _HANDlerFn] = {
    # Regression bucket — canonical name and legacy alias both
    # dispatch to the same sanitizer. The output's ``analysis_type``
    # mirrors whichever name the input used, so existing stored
    # payloads keep their old name on read and new emissions carry
    # the canonical name.
    _REGRESSION_TYPE_CANONICAL: _sanitize_linear_regression,
    _REGRESSION_TYPE_LEGACY:    _sanitize_linear_regression,
    "t_test": _sanitize_t_test,
    "descriptive": _sanitize_descriptive,
    "frequency_table": _sanitize_frequency_table,
    "crosstab": _sanitize_crosstab,
    "magnitude_table": _sanitize_magnitude_table,
    "correlation_matrix": _sanitize_correlation_matrix,
    "did_event_study": _sanitize_did_event_study,
    "rdd": _sanitize_rdd,
    "kaplan_meier": _sanitize_kaplan_meier,
    "factor_decomposition": _sanitize_factor_decomposition,
    "cluster_analysis": _sanitize_cluster_analysis,
    "marginal_effects": _sanitize_marginal_effects,
    "text_extraction": _sanitize_text_extraction,
    "method_result": _sanitize_method_result,
}


def supported_types() -> list[str]:
    """Return the list of analysis types the sanitizer currently accepts."""
    return sorted(_HANDLERS.keys())
