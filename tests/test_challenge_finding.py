"""Challenge Finding — deterministic agreement across alternative specs.

``challenge_summary`` is the code-computed half of "is this finding
robust": given a baseline result plus a batch of alternative
specifications, does the sign of the shared estimates hold? The
critical property under test is when it stays silent — an ordinary
multi-result batch (unrelated tables) must never get a fabricated
ROBUST/FRAGILE label just because it happened to run through the same
script.
"""

from __future__ import annotations

from sift.verification import challenge_summary
from sift.verification import independent_challenge_pass


def _reg(coefs: dict[str, float]) -> dict:
    return {"type": "linear_regression", "coefficients": coefs, "n": 500}


def test_all_alternatives_agree_is_robust() -> None:
    baseline = _reg({"treatment": 4.2, "age": 0.1})
    alts = [
        _reg({"treatment": 3.8, "age": 0.12}),
        _reg({"treatment": 5.1, "age": 0.09}),
        _reg({"treatment": 3.1}),  # dropped a control; still shares "treatment"
    ]
    out = challenge_summary([baseline] + alts)
    assert out is not None
    assert out["verdict"] == "ROBUST"
    assert out["agreeing"] == out["total"] == 3


def test_sign_flip_is_fragile_and_names_the_variable() -> None:
    baseline = _reg({"treatment": 4.2})
    alts = [
        _reg({"treatment": 3.8}),
        _reg({"treatment": -1.5}),  # flips sign
    ]
    out = challenge_summary([baseline] + alts)
    assert out is not None
    assert out["verdict"] == "FRAGILE"
    assert out["agreeing"] == 1 and out["total"] == 2
    assert "treatment" in out["note"]


def test_one_shared_estimate_flip_prevents_robust_claim() -> None:
    """The payload does not identify the focal coefficient, so a majority
    rule could hide a treatment-effect flip behind stable nuisance controls.
    Every shared estimate must retain direction for the batch-level claim."""
    baseline = _reg({"treatment": 4.2, "age": 0.1, "income": 0.02})
    alt = _reg({"treatment": 3.9, "age": -0.05, "income": 0.03})
    out = challenge_summary([baseline, alt])
    assert out is not None
    assert out["comparisons"][0]["agrees"] is False
    assert out["verdict"] == "FRAGILE"


def test_unrelated_batch_returns_none_not_a_fabricated_verdict() -> None:
    """The core safety property: a script that emits a regression,
    then a completely unrelated crosstab, must not get a ROBUST/
    FRAGILE label — there is nothing shared to compare."""
    regression = _reg({"treatment": 4.2})
    crosstab = {"type": "crosstab", "counts": {"a": 10, "b": 20}}
    out = challenge_summary([regression, crosstab])
    assert out is None


def test_single_result_returns_none() -> None:
    assert challenge_summary([_reg({"treatment": 1.0})]) is None


def test_empty_list_returns_none() -> None:
    assert challenge_summary([]) is None


def test_baseline_with_no_estimate_dict_returns_none() -> None:
    baseline = {"type": "descriptive", "mean": 5.0, "n": 100}
    alt = _reg({"treatment": 1.0})
    assert challenge_summary([baseline, alt]) is None


def test_marginal_effects_estimate_field_supported() -> None:
    baseline = {"type": "marginal_effects", "effects": {"x1": 0.3, "x2": -0.1}}
    alt = {"type": "marginal_effects", "effects": {"x1": 0.25, "x2": -0.2}}
    out = challenge_summary([baseline, alt])
    assert out is not None
    assert out["verdict"] == "ROBUST"


def test_zero_valued_estimates_excluded_from_comparison() -> None:
    """A coefficient that's exactly zero has no sign to agree or
    disagree on — it must not silently count as either."""
    baseline = _reg({"treatment": 4.2, "noise": 0.0})
    alt = _reg({"treatment": 3.9, "noise": 0.0})
    out = challenge_summary([baseline, alt])
    assert out is not None
    assert out["comparisons"][0]["shared_estimates"] == 1  # only "treatment"


def test_non_finite_estimates_do_not_crash(caplog=None) -> None:
    baseline = _reg({"treatment": float("nan")})
    alt = _reg({"treatment": float("inf")})
    # Neither payload yields a usable estimate dict (nan/inf excluded
    # by ``_finite``), so this must return None, not raise.
    assert challenge_summary([baseline, alt]) is None


def test_malformed_inputs_do_not_crash() -> None:
    assert challenge_summary(None) is None  # type: ignore[arg-type]
    assert challenge_summary("not a list") is None  # type: ignore[arg-type]
    assert challenge_summary([None, None]) is None  # type: ignore[list-item]
    assert challenge_summary([{"type": "linear_regression"}, {}]) is None


def test_three_of_four_agree_reports_exact_counts() -> None:
    baseline = _reg({"effect": 2.0})
    alts = [
        _reg({"effect": 1.5}),
        _reg({"effect": 1.8}),
        _reg({"effect": -0.5}),
    ]
    out = challenge_summary([baseline] + alts)
    assert out["agreeing"] == 2
    assert out["total"] == 3
    assert out["verdict"] == "FRAGILE"


def test_independent_pass_compares_workflow_roles_and_detects_contradictions() -> None:
    rows = [
        {
            "result_id": "M1", "payload": _reg({"treatment": 2.0}),
            "provenance": {"schema_verified": True, "analysis_role": "primary"},
        },
        {
            "result_id": "M2", "payload": _reg({"treatment": -1.0}),
            "provenance": {"schema_verified": True, "analysis_role": "sensitivity"},
        },
    ]
    result = independent_challenge_pass(rows)
    assert result["status"] == "warn"
    assert result["alternative_specification_comparison"]["verdict"] == "FRAGILE"
    assert result["contradictions"][0]["estimate_names"] == ["treatment"]
    assert result["limitations"]


def test_independent_pass_never_claims_schema_verified_without_provenance() -> None:
    result = independent_challenge_pass([{
        "result_id": "M1", "payload": _reg({"x": 1.0}), "provenance": {},
    }])
    assert result["status"] == "warn"
    assert result["checks"][0]["schema_verified"] is False


# ---------------------------------------------------------------------------
# Envelope wiring — the ``submit_script`` response actually carries it
# ---------------------------------------------------------------------------

def _fake_exec_result(tmp_path):
    from sift.executor import ExecutionResult
    return ExecutionResult(
        ok=True, language="Python", raw_stdout="", raw_stderr="",
        exit_code=0, result_payloads=[], error=None, run_dir=tmp_path,
        script_path=None, duration_seconds=0.1, warnings=[],
        environment=None,
    )


def test_envelope_carries_challenge_summary_for_a_real_batch(tmp_path) -> None:
    from sift.tools import _build_response_envelope

    results = [
        {"status": "ok", "result_id": "r1", "label": "baseline",
         "payload": _reg({"treatment": 4.2})},
        {"status": "ok", "result_id": "r2", "label": "drop outliers",
         "payload": _reg({"treatment": 3.9})},
        {"status": "ok", "result_id": "r3", "label": "cluster SE",
         "payload": _reg({"treatment": -0.2})},
    ]
    envelope = _build_response_envelope(
        overall_status="ok", script_run_id="run-1", results=results,
        exec_result=_fake_exec_result(tmp_path), language="Python",
        sanitize_seconds=0.0, store_seconds=0.0,
        row_count_audit_seconds=0.0,
    )
    assert "challenge_summary" in envelope
    assert envelope["challenge_summary"]["verdict"] == "FRAGILE"
    assert envelope["challenge_summary"]["agreeing"] == 1
    assert envelope["challenge_summary"]["total"] == 2


def test_envelope_carries_challenge_summary_even_when_payload_trim_fires(
    tmp_path,
) -> None:
    """Regression: ``_build_response_envelope`` used to compute
    ``challenge_summary`` AFTER ``_trim_oversize_inline_payloads``,
    which deletes ``entry["payload"]`` in place on every ok result
    once the batch crosses ``_INLINE_PAYLOAD_BUDGET``. Reading
    ``r.get("payload")`` on already-trimmed entries returned a list
    of ``None``s, so ``challenge_summary`` came back ``None`` and the
    field silently vanished -- on exactly the wide multi-spec
    robustness batches (many predictors -> large payloads -> most
    likely to cross the trim budget) this deterministic verdict is
    supposed to cover.

    Builds a batch whose per-result payloads are individually large
    enough (100 extra coefficients each, on top of the shared
    ``treatment`` estimate) that the combined JSON crosses the real
    ``_INLINE_PAYLOAD_BUDGET`` and the trim actually fires -- this
    test would not have caught the bug with small payloads like the
    other envelope tests in this file use."""
    from sift.tools import _build_response_envelope

    def _wide_reg(treatment_effect: float) -> dict:
        coefs = {
            f"control_variable_number_{i}": float(i) for i in range(150)
        }
        coefs["treatment"] = treatment_effect
        return {"type": "linear_regression", "coefficients": coefs, "n": 500}

    payloads = [_wide_reg(4.2), _wide_reg(3.9), _wide_reg(-0.2)]
    # Ground truth: what challenge_summary reports when run directly
    # against the untrimmed payloads. Computed independently rather
    # than hand-predicting ROBUST/FRAGILE, since this batch's verdict
    # depends on majority agreement across every shared coefficient
    # (150 wide controls plus "treatment"), not just the flipping
    # "treatment" estimate alone.
    expected = challenge_summary(payloads)
    assert expected is not None, "test setup must produce a real verdict"

    results = [
        {"status": "ok", "result_id": "r1", "label": "baseline",
         "payload": payloads[0]},
        {"status": "ok", "result_id": "r2", "label": "drop outliers",
         "payload": payloads[1]},
        {"status": "ok", "result_id": "r3", "label": "cluster SE",
         "payload": payloads[2]},
    ]
    envelope = _build_response_envelope(
        overall_status="ok", script_run_id="run-trim", results=results,
        exec_result=_fake_exec_result(tmp_path), language="Python",
        sanitize_seconds=0.0, store_seconds=0.0,
        row_count_audit_seconds=0.0,
    )
    # Confirm the trim actually fired -- otherwise this test isn't
    # exercising the interaction it claims to.
    assert envelope.get("_inline_payload_omitted") is True
    assert "payload" not in results[0]
    # The verdict itself must still be present and match what
    # challenge_summary computes directly from the pre-trim payloads.
    assert "challenge_summary" in envelope
    assert envelope["challenge_summary"] == expected


def test_envelope_omits_challenge_summary_for_unrelated_batch(tmp_path) -> None:
    """An ordinary multi-result batch (a regression, then a crosstab)
    must not get a fabricated ROBUST/FRAGILE label."""
    from sift.tools import _build_response_envelope

    results = [
        {"status": "ok", "result_id": "r1", "label": "regression",
         "payload": _reg({"treatment": 4.2})},
        {"status": "ok", "result_id": "r2", "label": "breakdown",
         "payload": {"type": "crosstab", "counts": {"a": 10, "b": 20}}},
    ]
    envelope = _build_response_envelope(
        overall_status="ok", script_run_id="run-2", results=results,
        exec_result=_fake_exec_result(tmp_path), language="Python",
        sanitize_seconds=0.0, store_seconds=0.0,
        row_count_audit_seconds=0.0,
    )
    assert "challenge_summary" not in envelope


def test_envelope_omits_challenge_summary_for_single_result(tmp_path) -> None:
    from sift.tools import _build_response_envelope

    results = [
        {"status": "ok", "result_id": "r1", "label": "only one",
         "payload": _reg({"treatment": 4.2})},
    ]
    envelope = _build_response_envelope(
        overall_status="ok", script_run_id="run-3", results=results,
        exec_result=_fake_exec_result(tmp_path), language="Python",
        sanitize_seconds=0.0, store_seconds=0.0,
        row_count_audit_seconds=0.0,
    )
    assert "challenge_summary" not in envelope
