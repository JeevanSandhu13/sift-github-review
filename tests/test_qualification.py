from __future__ import annotations

import json
import os
from pathlib import Path

from sift.qualification import run_qualification
from sift.ui import SiftBridge


def _checks(report: dict) -> dict[str, dict]:
    return {row["id"]: row for row in report["checks"]}


def test_backend_qualification_is_json_safe_without_live_services() -> None:
    report = run_qualification(include_runtime=False)
    json.dumps(report)
    assert report["overall"] == "warning"
    assert report["live_external_services_tested"] is False
    assert report["model_access_included"] is False
    assert report["adversarial_generated_code_certified"] is False
    assert _checks(report)["researcher-supplied-models"]["status"] == "pass"
    assert _checks(report)["generated-code-trust-boundary"]["status"] == "warning"


def test_doctor_ok_status_is_normalized_to_qualification_pass(monkeypatch) -> None:
    from sift import doctor

    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda: doctor.DoctorReport(
            runtimes=[doctor.RuntimeReport("sandbox", "ok", "healthy")],
            rejected_python_candidates=[],
            blocked=False,
        ),
    )
    report = run_qualification(include_runtime=True)
    assert _checks(report)["runtime.sandbox"]["status"] == "pass"
    assert report["overall"] == "warning"  # generated-code trust warning remains


def test_optional_runtime_unavailable_is_a_qualification_warning(monkeypatch) -> None:
    from sift import doctor

    monkeypatch.setattr(
        doctor,
        "run_doctor",
        lambda: doctor.DoctorReport(
            runtimes=[
                doctor.RuntimeReport("Python", "ok", "healthy"),
                doctor.RuntimeReport("Stata", "unavailable", "not installed"),
            ],
            rejected_python_candidates=[],
            blocked=False,
        ),
    )
    report = run_qualification(include_runtime=True)
    assert _checks(report)["runtime.Python"]["status"] == "pass"
    assert _checks(report)["runtime.Stata"]["status"] == "warning"


def test_clean_empty_session_qualifies(tmp_path: Path) -> None:
    report = run_qualification(tmp_path, include_runtime=False)
    assert report["overall"] == "warning"
    assert _checks(report)["session.directory"]["status"] == "pass"


def test_corrupt_session_artifacts_fail_qualification(tmp_path: Path) -> None:
    metadata = tmp_path / ".sift"
    metadata.mkdir(mode=0o700)
    (metadata / "chat_history.jsonl").write_text("not-json\n", encoding="utf-8")
    (metadata / "staged_files.json").write_text("{bad", encoding="utf-8")
    (metadata / "session_state.json").write_text("{bad", encoding="utf-8")

    report = run_qualification(tmp_path, include_runtime=False)
    checks = _checks(report)
    assert report["overall"] == "fail"
    assert checks["session.transcript"]["status"] == "fail"
    assert checks["session.provenance"]["status"] == "fail"
    assert checks["session.state"]["status"] == "fail"


def test_insecure_metadata_permissions_fail_on_posix(tmp_path: Path) -> None:
    metadata = tmp_path / ".sift"
    metadata.mkdir(mode=0o755)
    os.chmod(metadata, 0o755)
    report = run_qualification(tmp_path, include_runtime=False)
    if os.name == "posix":
        assert _checks(report)["session.permissions"]["status"] == "fail"


def test_bridge_exposes_qualification_contract(tmp_path: Path) -> None:
    payload = SiftBridge(cwd=tmp_path).get_qualification_report()
    assert payload["ok"] is True
    assert payload["schema_version"] == 1
    assert isinstance(payload["checks"], list)


def test_session_probe_failure_returns_verdict_not_exception(
    tmp_path: Path, monkeypatch,
) -> None:
    from sift import qualification

    def _boom(_cwd: Path):
        raise RuntimeError("simulated race")

    monkeypatch.setattr(qualification, "_session_checks", _boom)
    report = run_qualification(tmp_path, include_runtime=False)
    assert report["overall"] == "fail"
    assert _checks(report)["session.unavailable"]["status"] == "fail"
    assert "simulated race" not in _checks(report)["session.unavailable"]["detail"]
