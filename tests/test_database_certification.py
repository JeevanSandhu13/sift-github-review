from __future__ import annotations

import hashlib
import json
import os
import runpy
import subprocess
import sys
import zipfile
from pathlib import Path

from sift.database_certification import (
    DATABASE_CERTIFICATION_SCENARIOS,
    certification_environment_status,
    certification_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
DATABASE_QUALIFICATION = ROOT / "scripts" / "database_qualification.py"


def test_authentication_evidence_hash_tracks_identity_not_context() -> None:
    namespace = runpy.run_path(str(DATABASE_QUALIFICATION))
    proof_hash = namespace["_authentication_proof_sha256"]
    first = json.dumps({
        "authenticated_identity": "fixture-one",
        "authentication_context": "oauth",
    })
    changed_context = json.dumps({
        "authenticated_identity": "fixture-one",
        "authentication_context": "password",
    })
    other_identity = json.dumps({
        "authenticated_identity": "fixture-two",
        "authentication_context": "oauth",
    })
    assert proof_hash(first) == proof_hash(changed_context)
    assert proof_hash(first) == proof_hash(json.dumps({
        "authenticated_identity": " fixture-one ",
        "authentication_context": "oauth",
    }))
    assert proof_hash(first) != proof_hash(other_identity)
    assert proof_hash(json.dumps({"authentication_context": "oauth"})) is None


def test_fixture_and_runtime_provenance_are_secret_free_and_stable() -> None:
    namespace = runpy.run_path(str(DATABASE_QUALIFICATION))
    fixture_hash = namespace["_fixture_definition_sha256"]
    runtime = namespace["_runtime_provenance"]()
    scenario = next(
        value for value in DATABASE_CERTIFICATION_SCENARIOS
        if value.step_id == "S04-100"
    )
    required = scenario.required_environment()
    first = {name: "fixture-definition" for name in required}
    first[next(name for name in required if name.endswith("_URI"))] = (
        "databricks://user:secret@host"
    )
    second = dict(first)
    second[next(name for name in required if name.endswith("_URI"))] = (
        "databricks://other:changed@other-host"
    )
    second[next(name for name in required if name.endswith("_CLIENT_SECRET"))] = (
        "changed-client-secret"
    )
    assert fixture_hash(scenario, first) == fixture_hash(scenario, second)
    provenance_digest = runtime.pop("sha256")
    assert provenance_digest == hashlib.sha256(json.dumps(
        runtime, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    assert "secret" not in json.dumps(runtime).casefold()


def _qualification_root(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    """Create a minimal source/release pair for evidence-binding tests."""
    root = tmp_path / "qualification-root"
    package = root / "src" / "sift"
    package.mkdir(parents=True)
    source = b"VALUE = 'qualification fixture'\n"
    (package / "marker.py").write_bytes(source)
    (root / "pyproject.toml").write_text("[project]\nname='sift'\n", encoding="utf-8")
    (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    harness_files = {
        "scripts/database_qualification.py": b"# qualification fixture\n",
        "tests/live/test_database_live.py": b"# live fixture\n",
    }
    for relative, content in harness_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    wheel = root / "dist" / "sift-0.0.0-py3-none-any.whl"
    wheel.parent.mkdir(parents=True)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("sift/marker.py", source)
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight",
            "--output", str(root / "report.json"),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    preflight = json.loads(completed.stdout)
    assert preflight["qualification_context_ready"] is True
    context = preflight["qualification_context"]
    assert isinstance(context, dict)
    return root, context


def _bound_report(
    context: dict[str, object], rows: list[dict[str, object]],
) -> dict[str, object]:
    digest = str(context["qualification_context_sha256"])
    for row in rows:
        if row.get("status") == "passed":
            row["evidence_context_sha256"] = digest
        authentication = row.get("authentication_results", [])
        if isinstance(authentication, list):
            for result in authentication:
                if isinstance(result, dict) and result.get("status") == "passed":
                    result["evidence_context_sha256"] = digest
                    result.setdefault(
                        "identity_proof_sha256",
                        hashlib.sha256(
                            str(result.get("variant", "")).encode("utf-8"),
                        ).hexdigest(),
                    )
    return {
        "schema_version": 5,
        "qualification_context": context,
        "remote_scenarios": rows,
    }


def test_registry_covers_every_remote_database_checklist_step_exactly_once() -> None:
    ids = [scenario.step_id for scenario in DATABASE_CERTIFICATION_SCENARIOS]
    assert ids == [f"S04-{number:03d}" for number in range(58, 105)]
    assert len(ids) == len(set(ids)) == 47


def test_every_scenario_has_specific_inputs_and_supported_mode() -> None:
    for scenario in DATABASE_CERTIFICATION_SCENARIOS:
        required = scenario.required_environment()
        assert required and len(required) == len(set(required))
        assert all(name.startswith(f"SIFT_LIVE_{scenario.step_id.replace('-', '_')}_") for name in required)
        assert scenario.claim.endswith(".")
        if scenario.mode == "auth":
            assert scenario.variants
            assert len(required) >= 2 * len(scenario.variants)
            for variant in scenario.variants:
                variant_prefix = f"{scenario.env_prefix}_{variant.upper()}"
                assert f"{variant_prefix}_URI" in required
                assert f"{variant_prefix}_EXPECTED_ROW_JSON" in required
        if scenario.mode == "types":
            assert scenario.required_fields
            assert f"{scenario.env_prefix}_EXPECTED_ROW_JSON" in required


def test_environment_report_is_content_free_and_fail_closed() -> None:
    first = DATABASE_CERTIFICATION_SCENARIOS[0]
    environment = {name: "secret-value" for name in first.required_environment()}
    rows = certification_environment_status(environment)
    assert len(rows) == 47
    assert rows[0]["configured"] is True
    assert rows[1]["configured"] is False
    assert "secret-value" not in repr(rows)


def test_preflight_lists_only_configuration_state_and_variable_names() -> None:
    first = DATABASE_CERTIFICATION_SCENARIOS[0]
    environment = {
        "SIFT_LIVE_DATABASES_DISPOSABLE": "1",
        "SIFT_LIVE_DATABASE_WRITE_PROBE_ACK": "1",
        **{name: "private-value" for name in first.required_environment()},
    }
    preflight = certification_preflight(environment)
    assert preflight["configured_scenarios"] == 1
    assert preflight["total_scenarios"] == 47
    assert preflight["disposable_scope_confirmed"] is True
    assert preflight["write_probe_acknowledged"] is True
    assert preflight["remote_scenarios"][0]["configured"] is True
    type_scenario = next(
        row for row in preflight["remote_scenarios"] if row["step_id"] == "S04-061"
    )
    assert "jsonb_value" in type_scenario["required_fixture_fields"]
    auth_scenario = next(
        row for row in preflight["remote_scenarios"] if row["step_id"] == "S04-060"
    )
    assert auth_scenario["authentication_variants"] == [
        "password", "certificate", "managed",
    ]
    assert auth_scenario["authentication_assurance"] == {
        "password": "session_principal",
        "certificate": "mechanism",
        "managed": "session_principal",
    }
    assert "does not distinguish" in auth_scenario[
        "authentication_context_semantics"
    ]
    # The report intentionally publishes required variable names, including
    # provider-side cancellation proof inputs, but never their values.
    assert "private-value" not in repr(preflight)
    assert "private-value" not in repr(preflight)


def test_preflight_cli_is_read_only_and_content_free(tmp_path: Path) -> None:
    output = tmp_path / "existing-report.json"
    output.write_text('{"keep":"this"}\n', encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/database_qualification.py",
            "--preflight",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert output.read_text(encoding="utf-8") == '{"keep":"this"}\n'
    assert report["total_scenarios"] == 47
    assert report["schema_version"] == 2
    assert report["currently_configured_scenarios"] == 0
    assert "configured_scenarios" not in report
    assert "missing_environment_variable_count" not in report
    assert "durable_evidence_path" not in report
    assert str(tmp_path) not in completed.stdout
    assert "SIFT_LIVE_S04_058_URI" in repr(report)
    assert "postgresql://" not in completed.stdout


def test_preflight_cli_excludes_durably_completed_inputs(tmp_path: Path) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    core_node = (
        "tests/live/test_database_live.py::"
        "test_remote_database_requirement[S04-058]"
    )
    auth_node = (
        "tests/live/test_database_live.py::"
        "test_remote_database_auth_variant[S04-060-password]"
    )
    output.write_text(json.dumps(_bound_report(context, [
            {
                "step_id": "S04-058",
                "status": "passed",
                "configured": True,
                "missing_environment": [],
                "test_node": core_node,
                "checked_at": "2026-08-22T00:00:00+00:00",
            },
            {
                "step_id": "S04-060",
                "authentication_results": [{
                    "variant": "password",
                    "configured": True,
                    "missing_environment": [],
                    "status": "passed",
                    "test_node": auth_node,
                    "checked_at": "2026-08-22T00:00:00+00:00",
                }],
            },
        ])), encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(DATABASE_QUALIFICATION),
            "--root", str(root),
            "--preflight",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    rows = {row["step_id"]: row for row in report["remote_scenarios"]}
    assert report["durably_passed_scenarios"] == 1
    assert report["remaining_scenarios"] == 46
    assert report["durable_evidence_source"] == "existing_certification_report"
    assert "durable_evidence_path" not in report
    assert rows["S04-058"]["durably_passed"] is True
    assert rows["S04-058"]["remaining_environment"] == []
    assert rows["S04-060"]["durably_passed_authentication_variants"] == [
        "password"
    ]
    assert rows["S04-060"]["remaining_authentication_variants"] == [
        "certificate", "managed",
    ]
    assert rows["S04-060"]["remaining_environment"] == [
        "SIFT_LIVE_S04_060_CERTIFICATE_URI",
        "SIFT_LIVE_S04_060_CERTIFICATE_EXPECTED_ROW_JSON",
        "SIFT_LIVE_S04_060_MANAGED_URI",
        "SIFT_LIVE_S04_060_MANAGED_EXPECTED_ROW_JSON",
    ]
    assert "SIFT_LIVE_S04_060_PASSWORD_URI" not in rows["S04-060"][
        "remaining_environment"
    ]


def test_selected_scenario_cli_records_missing_input_without_erasing_proven_passes(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    prior_node = (
        "tests/live/test_database_live.py::"
        "test_remote_database_requirement[S04-058]"
    )
    output.write_text(json.dumps(_bound_report(context, [
            {
                "step_id": "S04-058",
                "status": "passed",
                "configured": True,
                "missing_environment": [],
                "test_node": prior_node,
                "checked_at": "2026-08-22T00:00:00+00:00",
            },
            {"step_id": "not-a-real-step", "status": "passed"},
        ])), encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(DATABASE_QUALIFICATION),
            "--root", str(root),
            "--scenario",
            "S04-061",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["step_id"]: row for row in report["remote_scenarios"]}
    assert report["schema_version"] == 5
    assert report["status"] == "blocked"
    assert report["selected_scenarios"] == ["S04-061"]
    assert rows["S04-058"]["status"] == "passed"
    assert rows["S04-058"]["configured"] is True
    assert rows["S04-058"]["missing_environment"] == []
    assert rows["S04-058"]["test_node"] == prior_node
    assert rows["S04-058"]["checked_at"] == "2026-08-22T00:00:00+00:00"
    assert rows["S04-061"]["status"].startswith("blocked_")
    assert "not-a-real-step" not in rows


def test_selected_scenario_does_not_trust_a_bare_prior_pass_label(
    tmp_path: Path,
) -> None:
    root, _context = _qualification_root(tmp_path)
    output = root / "report.json"
    output.write_text(json.dumps({
        "remote_scenarios": [{"step_id": "S04-058", "status": "passed"}],
    }), encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(DATABASE_QUALIFICATION),
            "--root", str(root),
            "--scenario",
            "S04-061",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    rows = {row["step_id"]: row for row in report["remote_scenarios"]}
    assert rows["S04-058"]["status"].startswith("blocked_")


def test_selected_read_only_scenario_requires_explicit_write_ack(tmp_path: Path) -> None:
    root, _context = _qualification_root(tmp_path)
    output = root / "report.json"
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    environment.update({
        "SIFT_LIVE_DATABASES_DISPOSABLE": "1",
        "SIFT_LIVE_S04_062_URI": "configured-but-never-executed",
    })
    completed = subprocess.run(
        [
            sys.executable,
            str(DATABASE_QUALIFICATION),
            "--root", str(root),
            "--scenario",
            "S04-062",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    selected = next(
        row for row in report["remote_scenarios"] if row["step_id"] == "S04-062"
    )
    assert selected["status"].startswith("blocked_")
    assert report["selected_test_exit_codes"] == {"S04-062": None}


def test_auth_variant_cli_preserves_proof_and_reports_only_unfinished_variants(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    node = (
        "tests/live/test_database_live.py::"
        "test_remote_database_auth_variant[S04-060-password]"
    )
    output.write_text(json.dumps(_bound_report(context, [{
            "step_id": "S04-060",
            "authentication_results": [{
                "variant": "password",
                "configured": True,
                "missing_environment": [],
                "status": "passed",
                "test_node": node,
                "checked_at": "2026-08-22T00:00:00+00:00",
            }],
        }])), encoding="utf-8")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    completed = subprocess.run(
        [
            sys.executable,
            str(DATABASE_QUALIFICATION),
            "--root", str(root),
            "--auth-variant",
            "S04-060:managed",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    row = next(
        value for value in report["remote_scenarios"]
        if value["step_id"] == "S04-060"
    )
    results = {value["variant"]: value for value in row["authentication_results"]}
    assert report["schema_version"] == 5
    assert report["selected_authentication_variants"] == ["S04-060:managed"]
    assert results["password"]["status"] == "passed"
    assert results["password"]["test_node"] == node
    assert results["managed"]["status"].startswith("blocked_")
    assert row["status"] == "partial_authentication"
    assert row["remaining_authentication_variants"] == ["certificate", "managed"]
    assert row["missing_environment"] == [
        "SIFT_LIVE_S04_060_CERTIFICATE_URI",
        "SIFT_LIVE_S04_060_CERTIFICATE_EXPECTED_ROW_JSON",
        "SIFT_LIVE_S04_060_MANAGED_URI",
        "SIFT_LIVE_S04_060_MANAGED_EXPECTED_ROW_JSON",
    ]


def test_strict_cli_preserves_partial_authentication_evidence(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    node = (
        "tests/live/test_database_live.py::"
        "test_remote_database_auth_variant[S04-060-password]"
    )
    output.write_text(json.dumps(_bound_report(context, [{
        "step_id": "S04-060",
        "status": "partial_authentication",
        "authentication_results": [{
            "variant": "password",
            "configured": True,
            "missing_environment": [],
            "status": "passed",
            "test_node": node,
            "checked_at": "2026-08-22T00:00:00+00:00",
        }],
    }])), encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--output", str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    row = next(
        value for value in report["remote_scenarios"]
        if value["step_id"] == "S04-060"
    )
    assert report["schema_version"] == 5
    assert row["status"] == "partial_authentication"
    assert row["authentication_results"][0]["test_node"] == node
    assert row["remaining_authentication_variants"] == [
        "certificate", "managed",
    ]


def test_incremental_authentication_results_form_a_durable_composite(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    results = [
        {
            "variant": variant,
            "configured": True,
            "missing_environment": [],
            "status": "passed",
            "test_node": (
                "tests/live/test_database_live.py::"
                f"test_remote_database_auth_variant[S04-060-{variant}]"
            ),
            "checked_at": f"2026-08-22T00:0{index}:00+00:00",
        }
        for index, variant in enumerate(("password", "certificate", "managed"))
    ]
    output.write_text(json.dumps(_bound_report(context, [{
        "step_id": "S04-060",
        "status": "partial_authentication",
        "authentication_results": results,
    }])), encoding="utf-8")
    environment = {
        key: value for key, value in os.environ.items()
        if not key.startswith("SIFT_LIVE_")
    }
    first = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--output", str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 2
    row = next(
        value for value in json.loads(output.read_text(encoding="utf-8"))[
            "remote_scenarios"
        ] if value["step_id"] == "S04-060"
    )
    assert row["status"] == "passed"
    assert row["test_node"] == "aggregate:authentication_results[S04-060]"
    assert row["remaining_authentication_variants"] == []

    preflight = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    preflight_report = json.loads(preflight.stdout)
    composite = next(
        value for value in preflight_report["remote_scenarios"]
        if value["step_id"] == "S04-060"
    )
    assert composite["durably_passed"] is True
    assert composite["remaining_environment"] == []


def test_malformed_authentication_proof_hash_is_not_durable(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    report = _bound_report(context, [{
        "step_id": "S04-060",
        "authentication_results": [{
            "variant": "password",
            "configured": True,
            "missing_environment": [],
            "status": "passed",
            "test_node": (
                "tests/live/test_database_live.py::"
                "test_remote_database_auth_variant[S04-060-password]"
            ),
            "checked_at": "2026-08-22T00:00:00+00:00",
        }],
    }])
    authentication = report["remote_scenarios"][0]["authentication_results"]
    authentication[0]["identity_proof_sha256"] = "z" * 64
    output.write_text(json.dumps(report), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    preflight = json.loads(completed.stdout)
    row = next(
        value for value in preflight["remote_scenarios"]
        if value["step_id"] == "S04-060"
    )
    assert row["durably_passed_authentication_variants"] == []
    assert "SIFT_LIVE_S04_060_PASSWORD_EXPECTED_ROW_JSON" in row[
        "remaining_environment"
    ]


def test_duplicate_authentication_proofs_reopen_collided_variants(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    report = _bound_report(context, [{
        "step_id": "S04-060",
        "authentication_results": [
            {
                "variant": variant,
                "configured": True,
                "missing_environment": [],
                "status": "passed",
                "test_node": (
                    "tests/live/test_database_live.py::"
                    f"test_remote_database_auth_variant[S04-060-{variant}]"
                ),
                "checked_at": "2026-08-22T00:00:00+00:00",
            }
            for variant in ("password", "certificate")
        ],
    }])
    authentication = report["remote_scenarios"][0]["authentication_results"]
    duplicate = "a" * 64
    for result in authentication:
        result["identity_proof_sha256"] = duplicate
    output.write_text(json.dumps(report), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    preflight = json.loads(completed.stdout)
    row = next(
        value for value in preflight["remote_scenarios"]
        if value["step_id"] == "S04-060"
    )
    assert row["durably_passed"] is False
    assert row["durably_passed_authentication_variants"] == []
    assert row["remaining_authentication_variants"] == [
        "password", "certificate", "managed",
    ]
    assert len(row["remaining_environment"]) == 6


def test_durable_evidence_is_invalidated_by_source_or_release_drift(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    output.write_text(json.dumps(_bound_report(context, [{
        "step_id": "S04-058",
        "status": "passed",
        "configured": True,
        "missing_environment": [],
        "test_node": (
            "tests/live/test_database_live.py::"
            "test_remote_database_requirement[S04-058]"
        ),
        "checked_at": "2026-08-22T00:00:00+00:00",
    }])), encoding="utf-8")
    (root / "src" / "sift" / "marker.py").write_text(
        "VALUE = 'changed after certification'\n", encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    row = next(
        value for value in report["remote_scenarios"]
        if value["step_id"] == "S04-058"
    )
    assert report["qualification_context_ready"] is False
    assert report["durably_passed_scenarios"] == 0
    assert row["durably_passed"] is False


def test_durable_evidence_is_invalidated_by_release_artifact_drift(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    output.write_text(json.dumps(_bound_report(context, [{
        "step_id": "S04-058",
        "status": "passed",
        "configured": True,
        "missing_environment": [],
        "test_node": (
            "tests/live/test_database_live.py::"
            "test_remote_database_requirement[S04-058]"
        ),
        "checked_at": "2026-08-22T00:00:00+00:00",
    }])), encoding="utf-8")
    wheel = root / str(context["release_artifact_path"])
    with wheel.open("ab") as handle:
        handle.write(b"changed after certification")
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["qualification_context_ready"] is True
    assert report["qualification_context"]["release_artifact_sha256"] != (
        context["release_artifact_sha256"]
    )
    assert report["durably_passed_scenarios"] == 0


def test_durable_evidence_is_invalidated_by_harness_drift(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    output.write_text(json.dumps(_bound_report(context, [{
        "step_id": "S04-058",
        "status": "passed",
        "configured": True,
        "missing_environment": [],
        "test_node": (
            "tests/live/test_database_live.py::"
            "test_remote_database_requirement[S04-058]"
        ),
        "checked_at": "2026-08-22T00:00:00+00:00",
    }])), encoding="utf-8")
    harness = root / "tests" / "live" / "test_database_live.py"
    harness.write_text("# semantic assertion changed\n", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["qualification_context_ready"] is True
    assert report["qualification_context"]["certification_harness_sha256"] != (
        context["certification_harness_sha256"]
    )
    assert report["durably_passed_scenarios"] == 0


def test_proof_complete_looking_row_without_binding_is_not_durable(
    tmp_path: Path,
) -> None:
    root, context = _qualification_root(tmp_path)
    output = root / "report.json"
    output.write_text(json.dumps({
        "schema_version": 5,
        "qualification_context": context,
        "remote_scenarios": [{
            "step_id": "S04-058",
            "status": "passed",
            "configured": True,
            "missing_environment": [],
            "test_node": (
                "tests/live/test_database_live.py::"
                "test_remote_database_requirement[S04-058]"
            ),
            "checked_at": "2026-08-22T00:00:00+00:00",
        }],
    }), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--preflight", "--output", str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["qualification_context_ready"] is True
    assert report["durably_passed_scenarios"] == 0


def test_stale_release_blocks_execution_without_overwriting_report(
    tmp_path: Path,
) -> None:
    root, _context = _qualification_root(tmp_path)
    output = root / "report.json"
    original = b'{"existing":"evidence"}\n'
    output.write_bytes(original)
    (root / "src" / "sift" / "marker.py").write_text(
        "VALUE = 'source newer than wheel'\n", encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable, str(DATABASE_QUALIFICATION),
            "--root", str(root), "--scenario", "S04-058",
            "--output", str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert result == {
        "status": "blocked",
        "reason": "missing_or_stale_release_artifact",
        "report_preserved": True,
    }
    assert output.read_bytes() == original
