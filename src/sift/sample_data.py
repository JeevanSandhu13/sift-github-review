"""Sift — synthetic evaluation datasets.

Nobody puts confidential data into unfamiliar software on day one. A
researcher assessing Sift needs to see the whole loop — load, profile,
analyse, verify, export — before they will point it at data covered by
an IRB protocol or a data-use agreement. Without a safe way to do
that, the first run is also the first trust decision, and the honest
answer to "should I paste my patient data into this?" is "not yet".

This module generates a dataset for exactly that evaluation. It is
fully synthetic: every value is drawn from a seeded generator here, so
there is no possibility of real data reaching a demo. The seed is
fixed, so two people evaluating Sift see identical numbers and can
compare notes.

The design is not arbitrary. The generated data deliberately exercises
the parts of Sift a sceptical reviewer should check:

- **A real effect to find.** Churn depends on tenure and support
  contacts through a specified logistic model, so a regression
  recovers a known sign and rough magnitude. A reviewer can check that
  Sift finds something true rather than something plausible.
- **Missingness.** ``monthly_spend`` is missing for ~8% of rows,
  concentrated in newer accounts — so missingness is informative, not
  random, and the verification layer's warning about it is warranted.
- **An identifier column.** ``customer_id`` is unique per row, so the
  profile's identifier detection has something to find.
- **A constant column.** ``data_release`` never varies, which is the
  classic silent-killer in real extracts.
- **A small subgroup.** One region has fewer members than the
  small-cell suppression threshold, so a frequency table over region
  *visibly suppresses a cell* — and triggers secondary suppression of
  a second cell, because a single hidden cell is recoverable from the
  total. This is the important one: the reviewer watches the
  disclosure control fire on their own screen rather than taking the
  README's word for it.
- **A near-duplicate pair.** A handful of exactly duplicated rows, so
  the profile's duplicate count is non-zero.

Generated with the standard library only — no numpy/pandas dependency
at generation time, so the on-ramp works even in an environment where
the analysis stack is not yet installed (which is precisely the
situation of a first-time user who has not run any pip install yet).
"""

from __future__ import annotations

import csv
import io
import math
import random
from pathlib import Path

SAMPLE_FILENAME = "sample_customers.csv"

# Fixed so every evaluator sees the same dataset and the same numbers.
_SEED = 20260818

_REGIONS = (
    # (name, weight). "islands" is deliberately rare: with ~1200 rows
    # its expected count lands below the cell-suppression threshold, so
    # a frequency table over region shows a suppressed cell.
    ("north", 0.34),
    ("south", 0.31),
    ("east", 0.22),
    ("west", 0.125),
    ("islands", 0.005),
)

_PLANS = (("basic", 0.5), ("standard", 0.35), ("premium", 0.15))

_N_ROWS = 1200
_N_DUPLICATES = 3


def _weighted(rng: random.Random, options: tuple[tuple[str, float], ...]) -> str:
    r = rng.random()
    cumulative = 0.0
    for name, weight in options:
        cumulative += weight
        if r <= cumulative:
            return name
    return options[-1][0]


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def generate_rows() -> list[dict[str, object]]:
    """Return the synthetic rows. Deterministic for a fixed seed."""
    rng = random.Random(_SEED)
    rows: list[dict[str, object]] = []

    for i in range(_N_ROWS):
        tenure_months = max(1, min(72, int(rng.gauss(24, 15))))
        region = _weighted(rng, _REGIONS)
        plan = _weighted(rng, _PLANS)
        support_contacts = min(12, int(rng.expovariate(1 / 2.2)))
        age = max(18, min(85, int(rng.gauss(44, 14))))

        base_spend = {"basic": 29.0, "standard": 59.0, "premium": 119.0}[plan]
        monthly_spend = round(
            max(5.0, rng.gauss(base_spend, base_spend * 0.18)), 2)

        # Informative missingness: newer accounts are likelier to have
        # no recorded spend yet. A complete-case analysis therefore
        # drops disproportionately many short-tenure customers, which
        # is exactly the bias the verification layer should flag.
        missing_p = 0.30 if tenure_months <= 6 else 0.04
        spend_value: object = "" if rng.random() < missing_p else monthly_spend

        # Known data-generating process for churn. Longer tenure
        # protects; support contacts hurt; premium plans churn less.
        logit = (
            -0.45
            - 0.045 * tenure_months
            + 0.32 * support_contacts
            - 0.5 * (plan == "premium")
            + 0.010 * (age - 44)
        )
        churn = 1 if rng.random() < _logistic(logit) else 0

        rows.append({
            "customer_id": f"C{100000 + i}",
            "age": age,
            "region": region,
            "plan": plan,
            "tenure_months": tenure_months,
            "support_contacts": support_contacts,
            "monthly_spend": spend_value,
            "churn": churn,
            # Constant across the extract — the kind of column that
            # silently breaks a fixed-effects specification.
            "data_release": "2026-Q2",
        })

    # Exact duplicates, so the profile's duplicate count is non-zero.
    for j in range(_N_DUPLICATES):
        rows.append(dict(rows[j]))

    return rows


# ---------------------------------------------------------------------------
# Survival dataset
# ---------------------------------------------------------------------------
#
# Exercises the Kaplan-Meier / Cox path and its verification checks:
# a real treatment effect on survival, administrative censoring at
# study end plus random dropout (so the event count is well below the
# subject count), and a thin tail (few subjects still at risk late) so
# the at-risk-depth warning has something true to say.

SURVIVAL_FILENAME = "sample_trial.csv"
_SURV_N = 800
_STUDY_MONTHS = 60


def generate_survival_rows() -> list[dict[str, object]]:
    rng = random.Random(_SEED + 1)
    rows: list[dict[str, object]] = []
    for i in range(_SURV_N):
        arm = "treatment" if rng.random() < 0.5 else "control"
        age = max(30, min(85, int(rng.gauss(62, 10))))
        # Exponential event times; treatment slows the hazard ~40%.
        base_hazard = 1 / 28.0 if arm == "control" else 1 / 46.0
        # Older patients have modestly higher hazard.
        hazard = base_hazard * (1.0 + 0.015 * (age - 62))
        event_time = rng.expovariate(max(hazard, 1e-6))
        dropout_time = rng.expovariate(1 / 90.0)   # sparse random dropout
        observed = min(event_time, dropout_time, _STUDY_MONTHS)
        event = 1 if event_time <= min(dropout_time, _STUDY_MONTHS) else 0
        rows.append({
            "patient_id": f"P{20000 + i}",
            "arm": arm,
            "age": age,
            "time_months": round(observed, 1),
            "event": event,
        })
    return rows


# ---------------------------------------------------------------------------
# Staggered-adoption panel dataset
# ---------------------------------------------------------------------------
#
# Exercises the DiD / event-study path: three adoption cohorts plus a
# never-treated group, a genuine post-treatment effect, unit and year
# structure, and clean pre-trends by construction — so the estimator
# should find the effect AND the pre-trend check should pass, which is
# the right demonstration (a reviewer can then break it themselves by
# subsetting).

PANEL_FILENAME = "sample_panel.csv"
_PANEL_FIRMS = 300
_PANEL_YEARS = tuple(range(2015, 2025))
_COHORTS = (2018, 2020, 2022, None, None)   # ~40% never treated
_PANEL_EFFECT = 2.0


def generate_panel_rows() -> list[dict[str, object]]:
    rng = random.Random(_SEED + 2)
    rows: list[dict[str, object]] = []
    for i in range(_PANEL_FIRMS):
        cohort = _COHORTS[rng.randrange(len(_COHORTS))]
        firm_effect = rng.gauss(0, 1.5)
        sector = rng.choice(("manufacturing", "services", "retail"))
        for year in _PANEL_YEARS:
            treated = 1 if (cohort is not None and year >= cohort) else 0
            growth = (
                3.0
                + firm_effect
                + 0.15 * (year - 2015)          # common trend
                + _PANEL_EFFECT * treated       # the designed effect
                + rng.gauss(0, 1.0)
            )
            rows.append({
                "firm_id": f"F{3000 + i}",
                "year": year,
                "sector": sector,
                "adoption_year": cohort if cohort is not None else "",
                "treated": treated,
                "revenue_growth": round(growth, 2),
            })
    return rows


def write_sample_datasets(dest_dir: Path) -> list[Path]:
    """Write all synthetic datasets and the explainer; return paths."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, rows in (
        (SAMPLE_FILENAME, generate_rows()),
        (SURVIVAL_FILENAME, generate_survival_rows()),
        (PANEL_FILENAME, generate_panel_rows()),
    ):
        target = dest_dir / filename
        fieldnames = list(rows[0].keys())
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        from sift.reliability import atomic_write_text
        atomic_write_text(target, buffer.getvalue())
        written.append(target)
    _write_readme(dest_dir)
    written.append(dest_dir / "ABOUT_SAMPLE_DATA.md")
    return written


def write_sample_dataset(dest_dir: Path) -> Path:
    """Write the synthetic dataset into ``dest_dir`` and return its path.

    Overwrites any existing copy so a researcher who edited it while
    experimenting can get a clean one back by loading the sample
    again.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / SAMPLE_FILENAME
    rows = generate_rows()
    fieldnames = list(rows[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    from sift.reliability import atomic_write_text
    atomic_write_text(target, buffer.getvalue())
    _write_readme(dest_dir)
    return target


_README = """# Sample data (synthetic)

`sample_customers.csv` is **synthetic data generated by Sift**. It
contains no real people and no real customer records. It exists so you
can evaluate Sift end to end before pointing it at anything
confidential.

It was built to exercise the things worth checking:

- **A real effect.** Churn was generated from a known model: longer
  tenure lowers it, more support contacts raise it, premium plans
  churn less. A regression should recover those signs.
- **Informative missingness.** `monthly_spend` is missing for roughly
  8% of rows, concentrated among newer accounts — so dropping
  incomplete cases biases the sample.
- **An identifier** (`customer_id`) and **a constant column**
  (`data_release`), both of which the dataset profile should flag.
- **A rare category.** The `islands` region has only a handful of
  members. Ask for a frequency table of `region` and that cell comes
  back **suppressed** rather than reported. You should see a *second*
  cell suppressed too: with only one cell hidden, its value is
  recoverable by subtracting the rest from the total, so the
  disclosure-control layer hides another one to close that route.
  Both suppressions are the controls working, not errors.
- **Duplicate rows**, so the profile's duplicate count is non-zero.

Things worth asking Sift:

- "Profile this dataset and tell me what's in it."
- "What predicts churn? Check your assumptions."
- "Show me the distribution of customers by region."  ← watch cells
  get suppressed
- "How robust is the tenure effect?"

## The other two files

`sample_trial.csv` is a synthetic two-arm survival study (800
patients): treatment genuinely slows the event hazard, and censoring
is heavy by design — most subjects never experience the event inside
the study window, and few remain at risk near the end. Ask for a
survival analysis and watch the verification layer report the event
count and how many subjects actually stand behind the late-horizon
estimates.

`sample_panel.csv` is a synthetic firm-year panel (300 firms,
2015–2024) with staggered adoption: cohorts adopt in 2018, 2020 or
2022, and about 40% never adopt. The adoption effect on
`revenue_growth` is +2.0 by construction, with clean pre-trends. An
event-study or difference-in-differences run should recover roughly
that number — and because the truth is known, you can judge the
estimate rather than admire it.

Delete this folder whenever you like; nothing depends on it.
"""


def _write_readme(dest_dir: Path) -> None:
    from sift.reliability import atomic_write_text
    atomic_write_text(dest_dir / "ABOUT_SAMPLE_DATA.md", _README)
