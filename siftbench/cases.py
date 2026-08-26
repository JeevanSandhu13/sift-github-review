"""SiftBench seed cases.

Each :class:`BenchCase` bundles a synthetic-data generator with a
KNOWN ground-truth answer (fixed RNG seed — deterministic, not just
"probably close"), a hand-written REFERENCE script that a competent
researcher (or a well-behaved model) would plausibly write to answer
the ``prompt``, and a ``score`` function that checks Sift's actual
output against that ground truth.

Every case runs entirely through Sift's real pipeline — the sandboxed
executor, the disclosure-control sanitizer, the result store — nothing
here is mocked. See ``siftbench/__init__.py`` for why this seed scores
the PIPELINE against ground truth rather than a live model's judgment.

Adding a case: write a reference script, know its answer analytically
or by construction, write a ``score`` function that reads the stored
result and checks it. Keep tolerances honest — wide enough that a
correct-but-differently-coded script wouldn't spuriously fail, tight
enough that a real bug (wrong formula, off-by-one, unit error) would
actually get caught. Where a case is about testing PLUMBING rather
than a number (e.g. "low n gets suppressed"), the fixture at
``score_status`` below is usually the right building block.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoreResult:
    passed: bool
    message: str


@dataclass
class BenchCase:
    id: str
    description: str
    # Written as if handed to a researcher-facing model — not used by
    # this seed's deterministic runner, but kept so a future live-
    # model runner (see siftbench/__init__.py) can reuse these same
    # cases without redefining them.
    prompt: str
    reference_script: str
    score: Callable[[dict[str, Any], Any], ScoreResult]
    language: str = "Python"
    source_dataset: str = ""


def _payload_for(body: dict[str, Any], store: Any) -> dict[str, Any] | None:
    """Fetch the FULL sanitized payload for a case's first 'ok'
    result from the store (not the trimmed inline ``payload`` field
    on the tool response, which is compacted for a coefficient
    table). Returns None if there is no such result."""
    for r in body.get("results", []):
        if r.get("status") == "ok":
            row = store.get(r["result_id"])
            if row is not None:
                return row.sanitized_payload
    return None


def score_status(expected_status: str) -> Callable[[dict, Any], ScoreResult]:
    """Build a scorer that only checks the envelope ``status`` —
    for cases that test a structural/disclosure-control property
    (e.g. "sub-threshold n gets rejected") rather than a number."""
    def _score(body: dict[str, Any], store: Any) -> ScoreResult:
        got = body.get("status")
        if got == expected_status:
            return ScoreResult(True, f"status == {expected_status!r} as expected")
        return ScoreResult(
            False,
            f"expected status {expected_status!r}, got {got!r} "
            f"(body: {body})",
        )
    return _score


def _close(actual: float | None, expected: float, tol: float, name: str) -> str | None:
    """Return an error message if ``actual`` isn't within ``tol`` of
    ``expected``, else None."""
    if actual is None:
        return f"{name} missing from sanitized payload"
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        return f"{name} is not numeric: {actual!r}"
    if not math.isfinite(float(actual)):
        return f"{name} is not finite: {actual!r}"
    if abs(actual - expected) > tol:
        return f"{name} = {actual!r}, expected {expected!r} ± {tol}"
    return None


# ---------------------------------------------------------------------------
# Case 1: correlation — known true r
# ---------------------------------------------------------------------------

# Exact sample statistic produced by the fixed seed below.  Comparing to the
# population parameter (0.6) with a wide sampling tolerance can let a broken
# implementation pass; this benchmark is a deterministic pipeline regression
# gate, so its oracle should be the deterministic sample answer.
_CORRELATION_EXPECTED_R = 0.6068824805390957

def _score_correlation(body: dict[str, Any], store: Any) -> ScoreResult:
    payload = _payload_for(body, store)
    if payload is None:
        return ScoreResult(False, f"no successful result found: {body}")
    errs = []
    if payload.get("n") != 500:
        errs.append(f"n = {payload.get('n')!r}, expected 500")
    got_r = payload.get("correlations", {}).get("x", {}).get("y")
    err = _close(got_r, _CORRELATION_EXPECTED_R, 0.001, "correlation(x, y)")
    if err:
        errs.append(err)
    if errs:
        return ScoreResult(False, "; ".join(errs))
    return ScoreResult(
        True,
        f"r = {got_r:.4f} (expected sample r = {_CORRELATION_EXPECTED_R:.4f})",
    )


CORRELATION_KNOWN_R = BenchCase(
    id="correlation_known_r",
    description=(
        "Pearson correlation on synthetic data drawn with a known "
        "true correlation of 0.6 (n=500, seed=42)."
    ),
    prompt=(
        "The dataset has two numeric columns, x and y. Compute the "
        "Pearson correlation between them and report it."
    ),
    reference_script=(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import sift\n"
        "\n"
        "rng = np.random.default_rng(42)\n"
        "data = rng.multivariate_normal([0.0, 0.0], "
        "[[1.0, 0.6], [0.6, 1.0]], size=500)\n"
        "df = pd.DataFrame(data, columns=['x', 'y'])\n"
        "sift.from_correlation(df, variables=['x', 'y'])\n"
    ),
    score=_score_correlation,
)


# ---------------------------------------------------------------------------
# Case 2: linear regression — known slope and intercept
# ---------------------------------------------------------------------------

_REGRESSION_EXPECTED_INTERCEPT = 1.98424528
_REGRESSION_EXPECTED_SLOPE = 3.00370597

def _score_regression(body: dict[str, Any], store: Any) -> ScoreResult:
    payload = _payload_for(body, store)
    if payload is None:
        return ScoreResult(False, f"no successful result found: {body}")
    coefs = payload.get("coefficients", {})
    errs = []
    if payload.get("n") != 500:
        errs.append(f"n = {payload.get('n')!r}, expected 500")
    err = _close(
        coefs.get("Intercept"),
        _REGRESSION_EXPECTED_INTERCEPT,
        0.002,
        "Intercept",
    )
    if err:
        errs.append(err)
    err = _close(coefs.get("x"), _REGRESSION_EXPECTED_SLOPE, 0.002, "slope on x")
    if err:
        errs.append(err)
    if errs:
        return ScoreResult(False, "; ".join(errs))
    return ScoreResult(
        True,
        f"Intercept={coefs.get('Intercept'):.3f}, x={coefs.get('x'):.3f} "
        f"(expected sample: {_REGRESSION_EXPECTED_INTERCEPT:.4f}, "
        f"{_REGRESSION_EXPECTED_SLOPE:.4f})",
    )


LINEAR_REGRESSION_KNOWN_SLOPE = BenchCase(
    id="linear_regression_known_slope",
    description=(
        "OLS on y = 2.0 + 3.0*x + noise, n=500, seed=7. Coefficients "
        "should recover the true intercept and slope."
    ),
    prompt=(
        "The dataset has columns x and y. Fit y as a linear function "
        "of x and report the coefficients."
    ),
    reference_script=(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import statsmodels.formula.api as smf\n"
        "import sift\n"
        "\n"
        "rng = np.random.default_rng(7)\n"
        "x = rng.normal(0, 1, 500)\n"
        "y = 2.0 + 3.0 * x + rng.normal(0, 1.0, 500)\n"
        "df = pd.DataFrame({'x': x, 'y': y})\n"
        "model = smf.ols('y ~ x', data=df).fit()\n"
        "sift.from_lm(model)\n"
    ),
    score=_score_regression,
)


# ---------------------------------------------------------------------------
# Case 3: two-sample t-test — known mean difference
# ---------------------------------------------------------------------------

_TTEST_EXPECTED_MEAN1 = 49.94195641175042
_TTEST_EXPECTED_MEAN2 = 54.707802845518124

def _score_ttest(body: dict[str, Any], store: Any) -> ScoreResult:
    payload = _payload_for(body, store)
    if payload is None:
        return ScoreResult(False, f"no successful result found: {body}")
    m1, m2 = payload.get("mean1"), payload.get("mean2")
    errs = []
    if m1 is None or m2 is None:
        errs.append(f"mean1/mean2 missing: {payload}")
    else:
        for actual, expected, label in (
            (m1, _TTEST_EXPECTED_MEAN1, "mean1"),
            (m2, _TTEST_EXPECTED_MEAN2, "mean2"),
        ):
            err = _close(actual, expected, 0.02, label)
            if err:
                errs.append(err)
    if payload.get("n1") != 200 or payload.get("n2") != 200:
        errs.append(f"n1/n2 = {payload.get('n1')!r}/{payload.get('n2')!r}, expected 200/200")
    if errs:
        return ScoreResult(False, "; ".join(errs))
    return ScoreResult(
        True,
        f"mean1={m1:.3f}, mean2={m2:.3f} match fixed-seed sample truth",
    )


TWO_SAMPLE_TTEST_KNOWN_DIFF = BenchCase(
    id="two_sample_ttest_known_diff",
    description=(
        "Two-sample t-test, group means 50 and 55 by construction "
        "(sd=10, n=200 each, seed=99) — true difference is 5."
    ),
    prompt=(
        "The dataset has a numeric outcome split into two groups. "
        "Run a two-sample t-test comparing the groups' means."
    ),
    reference_script=(
        "import numpy as np\n"
        "from scipy import stats\n"
        "import sift\n"
        "\n"
        "rng = np.random.default_rng(99)\n"
        "a = rng.normal(50, 10, 200)\n"
        "b = rng.normal(55, 10, 200)\n"
        "res = stats.ttest_ind(a, b)\n"
        "sift.from_t_test(res, n1=len(a), n2=len(b), "
        "mean1=float(a.mean()), mean2=float(b.mean()), "
        "test_type='two_sample')\n"
    ),
    score=_score_ttest,
)


# ---------------------------------------------------------------------------
# Case 4: descriptive summary — exact pass-through fields
# ---------------------------------------------------------------------------

# The deterministic raw sample mean is 10.049725973369908. Sift's
# disclosure boundary publishes descriptive values at the precision allowed
# for n=50, so the independently expected *released* value is 10.0. The
# benchmark must score the public contract, not compare a deliberately rounded
# sanitized payload with the pre-sanitization statistic.
_DESCRIPTIVE_RAW_MEAN = 10.049725973369908
_DESCRIPTIVE_PUBLISHED_MEAN = 10.0

def _score_descriptive(body: dict[str, Any], store: Any) -> ScoreResult:
    payload = _payload_for(body, store)
    if payload is None:
        return ScoreResult(False, f"no successful result found: {body}")
    errs = []
    if payload.get("n") != 50:
        errs.append(f"n = {payload.get('n')!r}, expected 50")
    if payload.get("missing_count") != 12:
        errs.append(f"missing_count = {payload.get('missing_count')!r}, expected 12")
    err = _close(
        payload.get("mean"), _DESCRIPTIVE_PUBLISHED_MEAN, 0.001,
        "disclosure-controlled mean",
    )
    if err:
        errs.append(err)
    if errs:
        return ScoreResult(False, "; ".join(errs))
    return ScoreResult(
        True,
        "n and missing_count match construction; mean matches the "
        f"disclosure-controlled release of raw {_DESCRIPTIVE_RAW_MEAN:.6f}",
    )


DESCRIPTIVE_EXACT_MISSING_COUNT = BenchCase(
    id="descriptive_exact_missing_count",
    description=(
        "Descriptive summary with a missing_count set explicitly by "
        "construction (12) — the sanitizer must pass it through "
        "unchanged, not recompute or drop it."
    ),
    prompt=(
        "Summarize the 'score' variable: sample size, mean, standard "
        "deviation, and how many values are missing."
    ),
    reference_script=(
        "import numpy as np\n"
        "import sift\n"
        "\n"
        "rng = np.random.default_rng(3)\n"
        "values = rng.normal(loc=10.0, scale=2.0, size=50)\n"
        "sift.from_summarize('score', n=len(values), "
        "mean=float(values.mean()), sd=float(values.std(ddof=1)), "
        "missing_count=12, distinct_count=50)\n"
    ),
    score=_score_descriptive,
)


# ---------------------------------------------------------------------------
# Case 5: sub-threshold n — must be rejected by disclosure control
# ---------------------------------------------------------------------------

SMALL_N_CORRELATION_REJECTED = BenchCase(
    id="small_n_correlation_rejected",
    description=(
        "Correlation on only 6 rows, deliberately below SDC's minimum-n "
        "threshold (10). This is a plumbing check, not a numeric one: "
        "the point is that Sift's disclosure-control layer refuses to "
        "release a correlation computed from a handful of points, "
        "regardless of what the number would have been."
    ),
    prompt=(
        "Compute the correlation between columns a and b in this "
        "6-row dataset."
    ),
    reference_script=(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import sift\n"
        "\n"
        "rng = np.random.default_rng(1)\n"
        "data = rng.normal(size=(6, 2))\n"
        "df = pd.DataFrame(data, columns=['a', 'b'])\n"
        "sift.from_correlation(df, variables=['a', 'b'])\n"
    ),
    score=score_status("rejected_by_sanitizer"),
)


SEED_CASES: list[BenchCase] = [
    CORRELATION_KNOWN_R,
    LINEAR_REGRESSION_KNOWN_SLOPE,
    TWO_SAMPLE_TTEST_KNOWN_DIFF,
    DESCRIPTIVE_EXACT_MISSING_COUNT,
    SMALL_N_CORRELATION_REJECTED,
]
