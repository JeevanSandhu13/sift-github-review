"""Synthetic sample dataset — the evaluation on-ramp.

The sample exists so a reviewer can exercise Sift end to end without
touching confidential data. That only works if the claims made about
it in ABOUT_SAMPLE_DATA.md are actually true of the generated file,
so these tests check the documented properties rather than just
"a file appeared". A demo that quietly stops demonstrating what it
promises is worse than no demo.
"""

from __future__ import annotations

import csv
from pathlib import Path

from sift.sample_data import SAMPLE_FILENAME, generate_rows, write_sample_dataset


def _rows(tmp_path: Path):
    path = write_sample_dataset(tmp_path)
    with path.open(encoding="utf-8") as fh:
        return path, list(csv.DictReader(fh))


def test_writes_dataset_and_explainer(tmp_path: Path) -> None:
    path = write_sample_dataset(tmp_path)
    assert path.name == SAMPLE_FILENAME and path.is_file()
    about = tmp_path / "ABOUT_SAMPLE_DATA.md"
    assert about.is_file()
    text = about.read_text(encoding="utf-8")
    # The file must say plainly that it is synthetic.
    assert "synthetic" in text.lower()
    assert "no real people" in text.lower()


def test_is_deterministic(tmp_path: Path) -> None:
    """Two evaluators must see identical numbers so they can compare
    notes and so documentation about it stays true."""
    a = write_sample_dataset(tmp_path / "a").read_bytes()
    b = write_sample_dataset(tmp_path / "b").read_bytes()
    assert a == b


def test_generation_needs_no_analysis_stack() -> None:
    """A first-time user may not have pandas installed yet; generation
    must not depend on it."""
    import inspect

    import sift.sample_data as mod
    src = inspect.getsource(mod)
    for heavy in ("import pandas", "import numpy", "import pyarrow"):
        assert heavy not in src


def test_documented_structural_features_are_real(tmp_path: Path) -> None:
    _, rows = _rows(tmp_path)
    assert len(rows) > 1000

    # Identifier column: unique per row except the deliberate dupes.
    ids = [r["customer_id"] for r in rows]
    assert len(set(ids)) < len(ids)          # duplicates exist
    # Constant column.
    assert len({r["data_release"] for r in rows}) == 1
    # Missingness present, and concentrated in short-tenure rows.
    missing = [r for r in rows if r["monthly_spend"] == ""]
    assert 0 < len(missing) < len(rows)
    short = [r for r in missing if int(r["tenure_months"]) <= 6]
    assert len(short) / len(missing) > 0.3   # informative, not random


def test_rare_category_is_below_suppression_threshold(tmp_path: Path) -> None:
    """The headline demonstration: a region rare enough that the SDC
    layer suppresses its cell."""
    from sift.sanitizer import DEFAULT_CONFIG

    threshold = DEFAULT_CONFIG.cell_suppression_threshold
    _, rows = _rows(tmp_path)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["region"]] = counts.get(r["region"], 0) + 1
    smallest = min(counts.values())
    assert 0 < smallest < threshold


def test_suppression_actually_fires_through_the_real_sanitizer(
        tmp_path: Path) -> None:
    """End-to-end against the production sanitizer, not a mock: the
    reviewer's promised experience must be what the code does."""
    from sift.sanitizer import DEFAULT_CONFIG, sanitize

    _, rows = _rows(tmp_path)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["region"]] = counts.get(r["region"], 0) + 1

    result = sanitize({
        "type": "frequency_table", "variable": "region",
        "n": len(rows), "counts": counts, "missing_count": 0,
        "sift_token": "t",
    }, DEFAULT_CONFIG)

    assert result.ok
    model_visible = result.sanitized["counts"]
    # The rare level's own name must not reach the model.
    assert "islands" not in model_visible
    # A suppression marker must be present.
    assert any(isinstance(v, str) and v.startswith("<")
               for v in model_visible.values())
    # And secondary suppression must have fired too — with one cell
    # hidden, its value is recoverable from the total.
    joined = " ".join(result.transformations).lower()
    assert "secondary suppression" in joined


def test_effect_is_recoverable(tmp_path: Path) -> None:
    """A reviewer should find something true, not merely plausible.
    Checked with a simple mean comparison so the test carries no
    modelling dependency."""
    _, rows = _rows(tmp_path)
    long_t = [int(r["churn"]) for r in rows if int(r["tenure_months"]) >= 30]
    short_t = [int(r["churn"]) for r in rows if int(r["tenure_months"]) <= 10]
    assert long_t and short_t
    # Churn was generated to fall with tenure; the gap should be clear.
    assert sum(short_t) / len(short_t) > sum(long_t) / len(long_t) + 0.15


def test_rows_have_stable_schema() -> None:
    rows = generate_rows()
    expected = {
        "customer_id", "age", "region", "plan", "tenure_months",
        "support_contacts", "monthly_spend", "churn", "data_release",
    }
    assert set(rows[0].keys()) == expected


# --------------------------------------------------------------------
# Survival + panel datasets
# --------------------------------------------------------------------

def test_all_sample_files_written_and_deterministic(tmp_path: Path) -> None:
    from sift.sample_data import write_sample_datasets

    a = write_sample_datasets(tmp_path / "a")
    b = write_sample_datasets(tmp_path / "b")
    assert [p.name for p in a] == [
        "sample_customers.csv", "sample_trial.csv", "sample_panel.csv",
        "ABOUT_SAMPLE_DATA.md"]
    for pa, pb in zip(a, b):
        assert pa.read_bytes() == pb.read_bytes(), pa.name


def test_survival_dataset_properties(tmp_path: Path) -> None:
    """The documented claims must be true of the file: a real
    treatment effect, heavy censoring, a thin late tail."""
    import statistics

    from sift.sample_data import write_sample_datasets

    write_sample_datasets(tmp_path)
    with (tmp_path / "sample_trial.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 800
    by_arm: dict = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    assert set(by_arm) == {"treatment", "control"}
    med = {a: statistics.median(float(r["time_months"]) for r in rs)
           for a, rs in by_arm.items()}
    assert med["treatment"] > med["control"] * 1.2   # designed effect
    events = sum(int(r["event"]) for r in rows)
    assert 0 < events < len(rows) * 0.8               # censoring present
    late = sum(1 for r in rows if float(r["time_months"]) >= 55)
    assert late < len(rows) * 0.2                     # thin tail


def test_panel_dataset_properties(tmp_path: Path) -> None:
    """Staggered cohorts, a never-treated group, and a recoverable
    effect near the designed +2.0."""
    import statistics

    from sift.sample_data import write_sample_datasets

    write_sample_datasets(tmp_path)
    with (tmp_path / "sample_panel.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 300 * 10
    cohorts = {r["adoption_year"] for r in rows}
    assert {"2018", "2020", "2022", ""} <= cohorts
    never = [r for r in rows if r["adoption_year"] == ""]
    assert len(never) > len(rows) * 0.2               # real control group
    # treated flag consistent with adoption year
    for r in rows[:500]:
        if r["adoption_year"]:
            expected = int(int(r["year"]) >= int(r["adoption_year"]))
            assert int(r["treated"]) == expected

    def mean(rs):
        return statistics.mean(float(r["revenue_growth"]) for r in rs)
    tr = [r for r in rows if r["adoption_year"]]
    nv = never
    did = ((mean([r for r in tr if int(r["treated"])])
            - mean([r for r in tr if not int(r["treated"])]))
           - (mean([r for r in nv if int(r["year"]) >= 2020])
              - mean([r for r in nv if int(r["year"]) < 2020])))
    assert 1.3 < did < 2.7


def test_about_documents_all_three(tmp_path: Path) -> None:
    from sift.sample_data import write_sample_datasets

    write_sample_datasets(tmp_path)
    about = (tmp_path / "ABOUT_SAMPLE_DATA.md").read_text(encoding="utf-8")
    for name in ("sample_customers.csv", "sample_trial.csv",
                 "sample_panel.csv"):
        assert name in about
