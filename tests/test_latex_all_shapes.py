"""LaTeX export across every analysis shape.

A malformed table is worse than no table: it fails to compile at
submission time, or — far worse — compiles into something subtly
wrong that reaches a reader. So these tests check structure
(balanced environments, consistent column counts) and escaping
(data-origin text escaped, Sift-authored math preserved) for every
shape, rather than just asserting a string came back.
"""

from __future__ import annotations

import pytest

from sift.research_export import render_latex_table


def _columns(line: str) -> int:
    r"""Count column separators in a LaTeX row.

    Only *unescaped* ``&`` separates columns. An escaped ``\&`` is a
    literal ampersand inside a cell — counting it would make a
    correctly-escaped table look ragged.
    """
    return line.replace("\\&", "").count("&")


PAYLOADS = {
    "linear_regression": {
        "type": "linear_regression", "n": 28194,
        "coefficients": {"(Intercept)": 0.41, "share_%": -0.12},
        "standard_errors": {"(Intercept)": 0.02, "share_%": 0.03},
        "p_values": {"(Intercept)": 1e-9, "share_%": 0.04},
        "r_squared": 0.148, "robust_se_type": "cluster",
    },
    "descriptive": {"type": "descriptive", "variable": "age", "n": 500,
                    "mean": 41.2, "sd": 12.1, "missing_count": 3},
    "frequency_table": {"type": "frequency_table", "variable": "region",
                        "counts": {"north": 400, "islands": "<10"}},
    "t_test": {"type": "t_test", "t_statistic": 2.1, "p_value": 0.03,
               "degrees_of_freedom": 88, "mean_difference": 1.4},
    "correlation_matrix": {
        "type": "correlation_matrix", "n": 500,
        "variables": ["age", "income_%"],
        "correlations": {"age|income_%": 0.31},
    },
    "crosstab": {
        "type": "crosstab", "row_variable": "region",
        "column_variable": "plan", "n": 900,
        "counts": {"north|basic": 120, "north|premium": 80,
                   "south|basic": 300, "south|premium": "<10"},
    },
    "magnitude_table": {
        "type": "magnitude_table", "row_variable": "region",
        "value_variable": "revenue", "aggregation": "sum",
        "cells": {"north": 120000, "south": 98000},
    },
    "did_event_study": {
        "type": "did_event_study", "groups": ["2019"],
        "event_times": [0, 1], "att": {"2019": {"0": 0.11, "1": 0.18}},
        "standard_errors": {"2019": {"0": 0.03, "1": 0.05}},
        "n_treated_per_group": {"2019": 420}, "aggregate_att": 0.145,
        "aggregate_se": 0.031, "pre_trends_p_value": 0.62,
    },
    "rdd": {
        "type": "rdd", "running_variable": "score", "cutoff": 50,
        "tau_robust": 2.1, "se_robust": 0.4, "p_robust": 0.0001,
        "tau_conventional": 2.0, "se_conventional": 0.38,
        "p_conventional": 0.0002, "effective_n_left": 420,
        "effective_n_right": 480, "bandwidth_left": 5.0,
        "bandwidth_right": 5.4,
    },
    "kaplan_meier": {
        "type": "kaplan_meier", "time_variable": "t",
        "event_variable": "d", "n_subjects": 900, "n_failures": 260,
        "median_survival_time": 48, "median_survival_ci_lower": 41,
        "median_survival_ci_upper": 55, "survival_at_5y": 0.62,
        "logrank_p_value": 0.004,
    },
    "cluster_analysis": {
        "type": "cluster_analysis", "method": "kmeans",
        "n_observations": 5000, "n_clusters": 3, "n_features": 4,
        "variables": ["a"], "cluster_labels": ["1"],
        "cluster_sizes": {"1": 1200, "2": "<10", "3": 1100},
        "silhouette_score": 0.58,
    },
    "factor_decomposition": {
        "type": "factor_decomposition", "method": "pca",
        "n_observations": 1200, "n_variables": 2, "n_components": 2,
        "variables": ["income_%", "age"],
        "loadings": {"PC1": {"income_%": 0.81, "age": 0.12},
                     "PC2": {"income_%": 0.09, "age": 0.77}},
        "kmo": 0.86,
    },
    "marginal_effects": {
        "type": "marginal_effects", "n": 5000, "method": "AME",
        "variables": ["x"], "effects": {"x_1": 0.12},
        "standard_errors": {"x_1": 0.03}, "p_values": {"x_1": 0.0001},
    },
}


def test_canonical_regression_type_name_renders_identically() -> None:
    """Regression test for architecture-audit finding J:
    ``render_latex_table`` checked ``ptype == "linear_regression"``
    only -- the LEGACY alias. Every regression payload the current
    R / Python / Stata helpers actually emit is stamped
    "coefficient_table_with_fit_stats" (see verification.py's
    matching fix and sanitizer.py's ``_REGRESSION_TYPE_CANONICAL``);
    without this, every CURRENT regression result silently got no
    LaTeX table at all in any export (replication package, PDF,
    PowerPoint) -- render_latex_table fell through every branch and
    returned None. The canonical-named payload must render to
    exactly the same LaTeX as the legacy-named one.
    """
    legacy = dict(PAYLOADS["linear_regression"])
    canonical = dict(legacy, type="coefficient_table_with_fit_stats")
    tex_legacy = render_latex_table(legacy, caption="x", label="tab:x")
    tex_canonical = render_latex_table(canonical, caption="x", label="tab:x")
    assert tex_canonical is not None
    assert tex_canonical == tex_legacy


@pytest.mark.parametrize("shape", sorted(PAYLOADS))
def test_every_shape_renders(shape) -> None:
    assert render_latex_table(PAYLOADS[shape]) is not None, \
        f"{shape} has no LaTeX renderer"


@pytest.mark.parametrize("shape", sorted(PAYLOADS))
def test_structure_is_well_formed(shape) -> None:
    tex = render_latex_table(PAYLOADS[shape], caption=shape,
                             label=f"tab:{shape}")
    # Environments balanced.
    for env in ("table", "tabular"):
        assert tex.count(f"\\begin{{{env}}}") == 1
        assert tex.count(f"\\end{{{env}}}") == 1
    for rule in ("\\toprule", "\\midrule", "\\bottomrule"):
        assert rule in tex
    # Every body row has the same number of columns as the header, and
    # matches the column spec. A mismatch is a compile error.
    body = [ln for ln in tex.splitlines() if ln.rstrip().endswith("\\\\")]
    assert body
    widths = {_columns(ln) for ln in body}
    assert len(widths) == 1, f"{shape}: ragged columns {widths}"
    spec = tex.split("\\begin{tabular}{", 1)[1].split("}", 1)[0]
    assert len(spec) == _columns(body[0]) + 1, \
        f"{shape}: colspec {spec!r} vs {_columns(body[0]) + 1} columns"


@pytest.mark.parametrize("shape", sorted(PAYLOADS))
def test_no_double_escaping(shape) -> None:
    """Sift-authored markup must survive; a stray \\textbackslash in
    output means our own LaTeX got escaped as if it were data."""
    tex = render_latex_table(PAYLOADS[shape])
    assert "\\textbackslash{}\\%" not in tex
    assert "\\$p\\$" not in tex


@pytest.mark.parametrize("shape,raw_name", [
    ("linear_regression", "share_%"),
    ("correlation_matrix", "income_%"),
    ("factor_decomposition", "income_%"),
])
def test_data_origin_text_is_escaped_everywhere(shape, raw_name) -> None:
    """A variable name containing LaTeX specials must not break
    compilation in any shape that can carry one."""
    tex = render_latex_table(PAYLOADS[shape])
    escaped = raw_name.replace("_", "\\_").replace("%", "\\%")
    assert escaped in tex, f"{shape}: {raw_name!r} not escaped"
    # The bare form must never survive outside the escaped sequence.
    assert raw_name not in tex.replace(escaped, "")


def test_suppression_markers_survive_into_every_shape() -> None:
    """A suppressed cell must remain visibly suppressed in the paper,
    never silently rendered as a number or a blank."""
    for shape in ("frequency_table", "crosstab", "cluster_analysis"):
        tex = render_latex_table(PAYLOADS[shape])
        assert "<10" in tex, f"{shape} lost its suppression marker"


def test_unknown_and_malformed_return_none_not_broken_latex() -> None:
    for payload in (
        {}, {"type": "no_such_shape"},
        {"type": "rdd"},                       # no estimates at all
        {"type": "did_event_study", "att": {}},
        {"type": "factor_decomposition", "loadings": {}},
        {"type": "cluster_analysis", "cluster_sizes": {}},
        {"type": "marginal_effects", "effects": {}},
        {"type": "crosstab", "counts": {"no-pipe-key": 1}},
    ):
        assert render_latex_table(payload) is None


def test_math_mode_used_for_relational_operators() -> None:
    """A bare ``<`` typesets as an inverted exclamation in text mode."""
    tex = render_latex_table(PAYLOADS["linear_regression"])
    assert "$<$0.001" in tex
    assert "& <0.001" not in tex


@pytest.mark.parametrize("hostile", [
    "north\nsouth",            # newline splits the row
    "a\tb",                    # tab
    "line1\r\nline2",          # CRLF
    "  padded  ",              # leading/trailing space
    "\\end{table}",            # attempted environment break-out
    "% comment",               # LaTeX comment character
    "a & b \\\\ c",            # column and row separators
])
def test_hostile_cell_values_cannot_break_table_structure(hostile) -> None:
    """A data-origin value must never alter the table's structure.

    Whitespace is flattened (a newline would split the row) and LaTeX
    specials are escaped (``\\end{table}`` must not close the
    environment early). Structure is asserted, not just absence of a
    crash.
    """
    tex = render_latex_table({
        "type": "frequency_table", "variable": hostile,
        "counts": {hostile: 40, "other": 10},
    })
    assert tex is not None
    assert tex.count("\\begin{table}") == 1
    assert tex.count("\\end{table}") == 1
    assert tex.count("\\begin{tabular}") == 1
    body = [ln for ln in tex.splitlines() if ln.rstrip().endswith("\\\\")]
    assert len({_columns(ln) for ln in body}) == 1, "ragged rows"
    spec = tex.split("\\begin{tabular}{", 1)[1].split("}", 1)[0]
    assert len(spec) == _columns(body[0]) + 1
