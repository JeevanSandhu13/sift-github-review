"""Regression-verification checks for convergence and panel/IV diagnostics.

These fields (converged, hausman_p, breusch_pagan_p, f_test_fe_p,
wooldridge_ar1_p, hansen_j_p, endogeneity_p, icc) were already carried
through the sanitizer's OLS-bucket schema but never consulted by
verification.py -- a real, silent gap. Pins:

1. Each check appears only when its input is present (the module's
   core honesty rule).
2. Each fires the correct verdict on the correct side of its p<0.05
   (or enum) boundary, with the DIRECTION of "warn" matching the
   test's actual statistical meaning -- getting a direction backwards
   here would be worse than not having the check at all.
3. sanitizer.py actually carries ``converged``/the panel-diagnostic
   fields through unchanged, and drops an invalid ``converged`` value
   via the same enum-gate pattern ``robust_se_type`` already uses.
"""

from __future__ import annotations

from sift.sanitizer import sanitize
from sift.verification import verify_payload


def _regression(**over):
    base = {
        "type": "linear_regression",
        "n": 5000,
        "coefficients": {"(Intercept)": 1.2, "tenure": -0.4, "spend": 0.1},
        "standard_errors": {"(Intercept)": 0.1, "tenure": 0.05, "spend": 0.02},
        "response_variable": "churn",
        "predictor_variables": ["tenure", "spend"],
    }
    base.update(over)
    return base


def _ids(block):
    return {c["id"]: c for c in block["checks"]}


# ---------------------------------------------------------------------------
# Absence: no check when the field isn't present
# ---------------------------------------------------------------------------


def test_no_diagnostic_checks_when_fields_absent():
    block = verify_payload(_regression())
    ids = _ids(block)
    for check_id in ("convergence", "fe_vs_re", "re_vs_pooled",
                     "fe_vs_pooled", "serial_correlation",
                     "overidentification", "endogeneity",
                     "intraclass_correlation"):
        assert check_id not in ids


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------


def test_convergence_pass():
    block = verify_payload(_regression(converged="converged"))
    assert _ids(block)["convergence"]["status"] == "pass"


def test_convergence_not_converged_warns():
    block = verify_payload(_regression(converged="not_converged"))
    assert _ids(block)["convergence"]["status"] == "warn"


def test_convergence_with_warnings_warns():
    block = verify_payload(_regression(converged="converged_with_warnings"))
    assert _ids(block)["convergence"]["status"] == "warn"


# ---------------------------------------------------------------------------
# fe_vs_re (Hausman)
# ---------------------------------------------------------------------------


def test_hausman_significant_prefers_fe():
    block = verify_payload(_regression(hausman_p=0.01))
    c = _ids(block)["fe_vs_re"]
    assert c["status"] == "warn"
    assert "fixed-effects" in c["detail"]


def test_hausman_not_significant_passes():
    block = verify_payload(_regression(hausman_p=0.5))
    assert _ids(block)["fe_vs_re"]["status"] == "pass"


# ---------------------------------------------------------------------------
# re_vs_pooled (panel Breusch-Pagan)
# ---------------------------------------------------------------------------


def test_panel_breusch_pagan_significant_prefers_re():
    block = verify_payload(_regression(breusch_pagan_p=0.001))
    c = _ids(block)["re_vs_pooled"]
    assert c["status"] == "warn"
    assert "random effects are" in c["detail"]


def test_panel_breusch_pagan_not_significant_passes():
    block = verify_payload(_regression(breusch_pagan_p=0.3))
    assert _ids(block)["re_vs_pooled"]["status"] == "pass"


# ---------------------------------------------------------------------------
# fe_vs_pooled (F-test on FE joint significance)
# ---------------------------------------------------------------------------


def test_f_test_fe_significant_warns():
    block = verify_payload(_regression(f_test_fe_p=0.0001))
    c = _ids(block)["fe_vs_pooled"]
    assert c["status"] == "warn"
    assert "jointly significant" in c["detail"]


def test_f_test_fe_not_significant_passes():
    block = verify_payload(_regression(f_test_fe_p=0.4))
    assert _ids(block)["fe_vs_pooled"]["status"] == "pass"


# ---------------------------------------------------------------------------
# serial_correlation (Wooldridge AR1)
# ---------------------------------------------------------------------------


def test_wooldridge_significant_warns_cluster_needed():
    block = verify_payload(_regression(wooldridge_ar1_p=0.02))
    c = _ids(block)["serial_correlation"]
    assert c["status"] == "warn"
    assert "clustered" in c["detail"]


def test_wooldridge_not_significant_passes():
    block = verify_payload(_regression(wooldridge_ar1_p=0.6))
    assert _ids(block)["serial_correlation"]["status"] == "pass"


# ---------------------------------------------------------------------------
# overidentification (Hansen J)
# ---------------------------------------------------------------------------


def test_hansen_j_significant_warns_invalid_instruments():
    block = verify_payload(_regression(hansen_j_p=0.01))
    c = _ids(block)["overidentification"]
    assert c["status"] == "warn"
    assert "may be invalid" in c["detail"]


def test_hansen_j_not_significant_passes():
    block = verify_payload(_regression(hansen_j_p=0.7))
    assert _ids(block)["overidentification"]["status"] == "pass"


# ---------------------------------------------------------------------------
# endogeneity (Wu-Hausman)
# ---------------------------------------------------------------------------


def test_endogeneity_significant_warns_iv_justified():
    block = verify_payload(_regression(endogeneity_p=0.005))
    c = _ids(block)["endogeneity"]
    assert c["status"] == "warn"
    assert "IV" in c["detail"]


def test_endogeneity_not_significant_passes():
    block = verify_payload(_regression(endogeneity_p=0.8))
    assert _ids(block)["endogeneity"]["status"] == "pass"


# ---------------------------------------------------------------------------
# ICC: always informational (no warn threshold)
# ---------------------------------------------------------------------------


def test_icc_always_reports_pass():
    for value in (0.05, 0.5, 0.95):
        block = verify_payload(_regression(icc=value))
        c = _ids(block)["intraclass_correlation"]
        assert c["status"] == "pass"
        assert f"{value:.3f}" in c["detail"]


# ---------------------------------------------------------------------------
# All diagnostics fire together without interfering
# ---------------------------------------------------------------------------


def test_all_new_diagnostics_fire_independently_in_one_payload():
    block = verify_payload(_regression(
        converged="not_converged",
        hausman_p=0.01,
        breusch_pagan_p=0.4,
        f_test_fe_p=0.5,
        wooldridge_ar1_p=0.01,
        hansen_j_p=0.9,
        endogeneity_p=0.6,
        icc=0.2,
    ))
    ids = _ids(block)
    assert ids["convergence"]["status"] == "warn"
    assert ids["fe_vs_re"]["status"] == "warn"
    assert ids["re_vs_pooled"]["status"] == "pass"
    assert ids["fe_vs_pooled"]["status"] == "pass"
    assert ids["serial_correlation"]["status"] == "warn"
    assert ids["overidentification"]["status"] == "pass"
    assert ids["endogeneity"]["status"] == "pass"
    assert ids["intraclass_correlation"]["status"] == "pass"
    assert block["warnings"] == 3  # convergence, fe_vs_re, serial_correlation


# ---------------------------------------------------------------------------
# sanitizer.py: converged enum round-trips and fails closed
# ---------------------------------------------------------------------------


def _raw_regression(**over):
    base = {
        "type": "linear_regression",
        "response_variable": "y",
        "predictor_variables": ["x1"],
        "n": 100,
        "coefficients": {"x1": 0.5, "intercept": 1.0},
        "standard_errors": {"x1": 0.1, "intercept": 0.2},
    }
    base.update(over)
    return base


def test_sanitizer_passes_through_valid_converged_value():
    r = sanitize(_raw_regression(converged="converged"))
    assert r.ok
    assert r.sanitized["converged"] == "converged"


def test_sanitizer_drops_invalid_converged_value():
    r = sanitize(_raw_regression(converged="totally_fine_i_promise"))
    assert r.ok  # whole payload isn't rejected, just the one field
    assert "converged" not in r.sanitized
    assert any("converged" in t for t in r.transformations)


def test_sanitizer_passes_through_panel_diagnostic_fields():
    r = sanitize(_raw_regression(
        hausman_chi2=12.3, hausman_p=0.02,
        breusch_pagan_chi2=4.1, breusch_pagan_p=0.3,
        f_test_fe_chi2=8.8, f_test_fe_p=0.001,
        wooldridge_ar1_chi2=6.0, wooldridge_ar1_p=0.04,
        hansen_j=2.1, hansen_j_p=0.6,
        endogeneity_p=0.01,
        icc=0.35,
    ))
    assert r.ok
    for key in ("hausman_chi2", "hausman_p", "breusch_pagan_chi2",
               "breusch_pagan_p", "f_test_fe_chi2", "f_test_fe_p",
               "wooldridge_ar1_chi2", "wooldridge_ar1_p",
               "hansen_j", "hansen_j_p", "endogeneity_p", "icc"):
        assert key in r.sanitized, key
