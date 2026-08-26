#!/usr/bin/env python3
"""Run and record the optional disposable live-database compatibility program."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, cast
from importlib.metadata import PackageNotFoundError, version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sift.database_certification import (
    DATABASE_CERTIFICATION_SCENARIOS,
    certification_environment_status,
    certification_preflight,
)
from sift.reliability import atomic_write_json


class CertificationReportRow(TypedDict, total=False):
    step_id: str
    backend: str
    mode: str
    claim: str
    authentication_variants: list[str]
    authentication_assurance: dict[str, object]
    authentication_context_semantics: str
    required_fixture_fields: list[str]
    configured: bool
    missing_environment: list[str]
    status: str
    test_node: str
    checked_at: str
    authentication_results: list[dict[str, object]]
    remaining_authentication_variants: list[str]
    evidence_context_sha256: str
    fixture_definition_sha256: str


_DRIVER_DISTRIBUTIONS = (
    "SQLAlchemy", "psycopg", "PyMySQL", "pyodbc", "oracledb",
    "snowflake-sqlalchemy", "sqlalchemy-bigquery", "redshift-connector",
    "databricks-sql-connector", "databricks-sqlalchemy",
    "databricks-sdk",
)


def _runtime_provenance() -> dict[str, object]:
    packages: dict[str, str] = {}
    for distribution in _DRIVER_DISTRIBUTIONS:
        try:
            packages[distribution] = version(distribution)
        except PackageNotFoundError:
            packages[distribution] = "not_installed"
    body: dict[str, object] = {
        "schema_version": 1,
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "driver_distributions": packages,
    }
    body["sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return body


def _fixture_definition_sha256(
    scenario: object,
    environment: dict[str, str],
) -> str | None:
    """Hash synthetic proof definitions while excluding endpoints and secrets."""
    required_environment = getattr(scenario, "required_environment")()
    included: dict[str, str] = {}
    secret_markers = (
        "_URI", "_PRIVATE_KEY", "_CLIENT_SECRET", "_PASSWORD", "_TOKEN",
    )
    for name in required_environment:
        if any(marker in name for marker in secret_markers):
            continue
        value = environment.get(name)
        if value:
            included[name] = value
    proof_query = getattr(scenario, "proof_query", None)
    if proof_query:
        included["code_owned_proof_query"] = str(proof_query)
    if not included:
        return None
    return hashlib.sha256(json.dumps(
        included, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _authentication_proof_sha256(raw_expected_row: str) -> str | None:
    """Hash the authenticated identity without persisting its contents.

    Context values can legitimately differ between credential paths for the
    same account, so hashing the whole row would not enforce the fixture rule
    that every authentication variant use a distinct disposable identity.
    """
    try:
        value = json.loads(raw_expected_row)
    except (TypeError, ValueError):
        return None
    if not isinstance(value, dict):
        return None
    identity = value.get("authenticated_identity")
    if not isinstance(identity, str) or not identity.strip():
        return None
    return hashlib.sha256(
        identity.strip().encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_manifest_sha256(entries: list[tuple[str, bytes]]) -> str:
    """Hash a path-aware content manifest independent of file mtimes."""
    digest = hashlib.sha256()
    for relative, content in sorted(entries):
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _source_package_sha256(root: Path) -> str | None:
    source = root / "src" / "sift"
    if source.is_symlink() or not source.is_dir():
        return None
    entries: list[tuple[str, bytes]] = []
    for path in sorted(source.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(source)
        if (
            "__pycache__" in relative.parts
            or path.suffix in {".pyc", ".pyo"}
            or path.name == ".DS_Store"
        ):
            continue
        if path.is_symlink() or not path.is_file():
            return None
        entries.append((relative.as_posix(), path.read_bytes()))
    return _content_manifest_sha256(entries) if entries else None


def _certification_harness_sha256(root: Path) -> str | None:
    """Bind live proof to the exact runner and semantic assertions used."""
    relative_paths = (
        Path("scripts/database_qualification.py"),
        Path("tests/live/test_database_live.py"),
    )
    entries: list[tuple[str, bytes]] = []
    for relative in relative_paths:
        path = root / relative
        if path.is_symlink() or not path.is_file():
            return None
        entries.append((relative.as_posix(), path.read_bytes()))
    return _content_manifest_sha256(entries)


def _wheel_package_sha256(wheel: Path) -> str | None:
    try:
        with zipfile.ZipFile(wheel) as archive:
            names = [
                name for name in archive.namelist()
                if name.startswith("sift/") and not name.endswith("/")
            ]
            if not names or len(names) != len(set(names)):
                return None
            entries: list[tuple[str, bytes]] = []
            for name in names:
                relative = Path(name).relative_to("sift")
                if ".." in relative.parts:
                    return None
                entries.append((relative.as_posix(), archive.read(name)))
    except (OSError, ValueError, zipfile.BadZipFile):
        return None
    return _content_manifest_sha256(entries)


def _qualification_context(root: Path) -> dict[str, object] | None:
    """Bind evidence to the exact package source, lock, and release wheel."""
    source_sha256 = _source_package_sha256(root)
    harness_sha256 = _certification_harness_sha256(root)
    pyproject = root / "pyproject.toml"
    lock = root / "uv.lock"
    release_root = root / "dist"
    wheels = sorted(release_root.glob("sift-*.whl")) if release_root.is_dir() else []
    if (
        source_sha256 is None or harness_sha256 is None
        or pyproject.is_symlink() or not pyproject.is_file()
        or lock.is_symlink() or not lock.is_file()
        or len(wheels) != 1
        or wheels[0].is_symlink() or not wheels[0].is_file()
    ):
        return None
    wheel = wheels[0]
    wheel_package_sha256 = _wheel_package_sha256(wheel)
    if wheel_package_sha256 is None:
        return None
    body: dict[str, object] = {
        "schema_version": 1,
        "source_package_sha256": source_sha256,
        "certification_harness_sha256": harness_sha256,
        "pyproject_sha256": _sha256_file(pyproject),
        "dependency_lock_sha256": _sha256_file(lock),
        "release_artifact_path": wheel.relative_to(root).as_posix(),
        "release_artifact_sha256": _sha256_file(wheel),
        "release_package_sha256": wheel_package_sha256,
        "release_package_matches_source": wheel_package_sha256 == source_sha256,
    }
    body["qualification_context_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return body


def _context_ready(context: dict[str, object] | None) -> bool:
    return bool(context and context.get("release_package_matches_source") is True)


def _load_bound_report(
    output: Path, context: dict[str, object] | None,
) -> dict[str, object] | None:
    if not _context_ready(context):
        return None
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("qualification_context") != context
    ):
        return None
    return cast(dict[str, object], payload)


def _auth_aggregate_node(step_id: str) -> str:
    return f"aggregate:authentication_results[{step_id}]"


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _authentication_result_is_bound_pass(
    value: object,
    *,
    step_id: str,
    variant: str,
    evidence_context_sha256: str,
) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("variant") == variant
        and value.get("status") == "passed"
        and value.get("configured") is True
        and value.get("missing_environment") == []
        and value.get("test_node") == _auth_variant_node(step_id, variant)
        and value.get("evidence_context_sha256") == evidence_context_sha256
        and _is_sha256(value.get("identity_proof_sha256"))
        and isinstance(value.get("checked_at"), str)
        and bool(value["checked_at"])
    )


def _previous_passes(
    output: Path, context: dict[str, object] | None,
) -> dict[str, CertificationReportRow]:
    """Load only complete, independently verifiable prior pass records.

    A bare ``status=passed`` is not durable evidence.  Preserve a prior result
    only when the report also records that all required inputs were configured,
    none were missing, and the exact scenario node ran.  This both prevents a
    hand-written label from becoming release evidence and avoids erasing real
    proof when scenarios are certified one at a time.
    """
    payload = _load_bound_report(output, context)
    if payload is None or context is None:
        return {}
    evidence_context_sha256 = str(context["qualification_context_sha256"])
    definitions = {
        scenario.step_id: scenario for scenario in DATABASE_CERTIFICATION_SCENARIOS
    }
    rows = payload.get("remote_scenarios", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    result: dict[str, CertificationReportRow] = {}
    for value in rows:
        if not isinstance(value, dict):
            continue
        step_id = str(value.get("step_id", ""))
        definition = definitions.get(step_id)
        expected_node = (
            "tests/live/test_database_live.py::"
            f"test_remote_database_requirement[{step_id}]"
        )
        authentication_results = value.get("authentication_results", [])
        aggregate_authentication_pass = bool(
            definition is not None
            and definition.mode == "auth"
            and value.get("test_node") == _auth_aggregate_node(step_id)
            and isinstance(authentication_results, list)
            and len(authentication_results) == len(definition.variants)
            and {
                str(result.get("variant"))
                for result in authentication_results
                if isinstance(result, dict)
            } == set(definition.variants)
            and all(
                any(
                    _authentication_result_is_bound_pass(
                        result,
                        step_id=step_id,
                        variant=variant,
                        evidence_context_sha256=evidence_context_sha256,
                    )
                    for result in authentication_results
                )
                for variant in definition.variants
            )
            and len({
                str(result.get("identity_proof_sha256"))
                for result in authentication_results
                if isinstance(result, dict) and result.get("status") == "passed"
            }) == len(definition.variants)
        )
        if (
            definition is not None
            and value.get("status") == "passed"
            and value.get("configured") is True
            and value.get("missing_environment") == []
            and (
                value.get("test_node") == expected_node
                or aggregate_authentication_pass
            )
            and value.get("evidence_context_sha256") == evidence_context_sha256
            and isinstance(value.get("checked_at"), str)
            and bool(value["checked_at"])
        ):
            result[step_id] = cast(CertificationReportRow, dict(value))
    return result


def _restore_prior_pass(
    row: CertificationReportRow,
    prior: CertificationReportRow,
) -> None:
    """Restore proof fields while retaining the current registry metadata."""
    row["configured"] = True
    row["missing_environment"] = []
    row["status"] = "passed"
    row["test_node"] = prior["test_node"]
    row["checked_at"] = prior["checked_at"]
    row["evidence_context_sha256"] = prior["evidence_context_sha256"]
    if "fixture_definition_sha256" in prior:
        row["fixture_definition_sha256"] = prior["fixture_definition_sha256"]


def _auth_variant_node(step_id: str, variant: str) -> str:
    return (
        "tests/live/test_database_live.py::"
        f"test_remote_database_auth_variant[{step_id}-{variant}]"
    )


def _previous_authentication_passes(
    output: Path, context: dict[str, object] | None,
) -> dict[str, dict[str, dict[str, object]]]:
    """Load proof-complete prior authentication-variant results only."""
    payload = _load_bound_report(output, context)
    if payload is None or context is None:
        return {}
    evidence_context_sha256 = str(context["qualification_context_sha256"])
    definitions = {
        scenario.step_id: scenario
        for scenario in DATABASE_CERTIFICATION_SCENARIOS
        if scenario.mode == "auth"
    }
    rows = payload.get("remote_scenarios", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict[str, dict[str, object]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        step_id = str(row.get("step_id", ""))
        definition = definitions.get(step_id)
        values = row.get("authentication_results", [])
        if definition is None or not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, dict):
                continue
            variant = str(value.get("variant", ""))
            if (
                variant in definition.variants
                and _authentication_result_is_bound_pass(
                    value,
                    step_id=step_id,
                    variant=variant,
                    evidence_context_sha256=evidence_context_sha256,
                )
            ):
                result.setdefault(step_id, {})[variant] = dict(value)
    for step_id, variants in list(result.items()):
        proof_counts: dict[str, int] = {}
        for value in variants.values():
            proof = str(value["identity_proof_sha256"])
            proof_counts[proof] = proof_counts.get(proof, 0) + 1
        result[step_id] = {
            variant: value
            for variant, value in variants.items()
            if proof_counts[str(value["identity_proof_sha256"])] == 1
        }
        if not result[step_id]:
            del result[step_id]
    return result


def _durable_preflight(
    environment: dict[str, str], root: Path,
    output: Path,
) -> dict[str, object]:
    """Return the next-input inventory after accounting for durable evidence."""
    base = certification_preflight(environment)
    currently_configured = base.pop("configured_scenarios")
    current_environment_missing = base.pop("missing_environment_variable_count")
    context = _qualification_context(root)
    prior_passes = _previous_passes(output, context)
    prior_authentication = _previous_authentication_passes(output, context)
    definitions = {
        scenario.step_id: scenario for scenario in DATABASE_CERTIFICATION_SCENARIOS
    }
    rows = base["remote_scenarios"]
    if not isinstance(rows, (tuple, list)):
        raise RuntimeError(
            "database certification preflight returned an invalid scenario inventory"
        )
    remaining_input_count = 0
    enriched: list[dict[str, object]] = []
    for value in rows:
        row = dict(value)
        step_id = str(row["step_id"])
        definition = definitions[step_id]
        durably_passed = step_id in prior_passes
        passed_variants = sorted(
            prior_authentication.get(step_id, {}),
            key=(
                definition.variants.index
                if definition.mode == "auth"
                else lambda variant: variant
            ),
        )
        remaining_variants = (
            [variant for variant in definition.variants if variant not in passed_variants]
            if definition.mode == "auth" and not durably_passed
            else []
        )
        if durably_passed:
            remaining_environment: list[str] = []
        elif definition.mode == "auth":
            remaining_environment = [
                name
                for variant in remaining_variants
                for name in definition.required_authentication_environment(variant)
            ]
        else:
            remaining_environment = list(definition.required_environment())
        missing_remaining = [
            name for name in remaining_environment if not environment.get(name)
        ]
        remaining_input_count += len(missing_remaining)
        row.update({
            "durably_passed": durably_passed,
            "durably_passed_authentication_variants": passed_variants,
            "remaining_authentication_variants": remaining_variants,
            "remaining_environment": remaining_environment,
            "missing_remaining_environment": missing_remaining,
        })
        enriched.append(row)
    base.update({
        "schema_version": 2,
        "remote_scenarios": enriched,
        "currently_configured_scenarios": currently_configured,
        "current_environment_missing_variable_count": current_environment_missing,
        "durably_passed_scenarios": len(prior_passes),
        "remaining_scenarios": len(DATABASE_CERTIFICATION_SCENARIOS) - len(prior_passes),
        "missing_remaining_environment_variable_count": remaining_input_count,
        "durable_evidence_source": "existing_certification_report",
        "qualification_context_ready": _context_ready(context),
        "qualification_context": context,
    })
    return base


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("docs/live_database_certification.json"))
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="print a secret-free provisioning inventory without running or recording certification",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.step_id for scenario in DATABASE_CERTIFICATION_SCENARIOS],
        help=(
            "run and durably record one scenario; repeat for multiple scenarios. "
            "Without this option the strict all-scenario release gate runs."
        ),
    )
    auth_variant_choices = [
        f"{scenario.step_id}:{variant}"
        for scenario in DATABASE_CERTIFICATION_SCENARIOS
        if scenario.mode == "auth"
        for variant in scenario.variants
    ]
    parser.add_argument(
        "--auth-variant",
        action="append",
        choices=auth_variant_choices,
        help=(
            "run and durably record one authentication variant without "
            "claiming the composite scenario is complete"
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    environment = dict(os.environ)
    qualification_context = _qualification_context(root)
    context_ready = _context_ready(qualification_context)
    evidence_context_sha256 = (
        str(qualification_context["qualification_context_sha256"])
        if qualification_context is not None else None
    )
    if args.preflight:
        print(json.dumps(
            _durable_preflight(environment, root, output), indent=2,
        ))
        return 0
    if not context_ready:
        print(json.dumps({
            "status": "blocked",
            "reason": "missing_or_stale_release_artifact",
            "report_preserved": output.is_file(),
        }, indent=2))
        return 2
    scenarios = [
        cast(CertificationReportRow, dict(row))
        for row in certification_environment_status(environment)
    ]
    disposable = environment.get("SIFT_LIVE_DATABASES_DISPOSABLE") == "1"
    write_probe_acknowledged = environment.get("SIFT_LIVE_DATABASE_WRITE_PROBE_ACK") == "1"
    selected = tuple(dict.fromkeys(args.scenario or ()))
    selected_auth = tuple(dict.fromkeys(args.auth_variant or ()))
    if selected or selected_auth:
        prior_passes = _previous_passes(output, qualification_context)
        prior_authentication = _previous_authentication_passes(
            output, qualification_context,
        )
        definitions = {scenario.step_id: scenario for scenario in DATABASE_CERTIFICATION_SCENARIOS}
        exit_codes: dict[str, int | None] = {}
        checked_at = datetime.now(timezone.utc).isoformat()
        for row in scenarios:
            step_id = str(row["step_id"])
            if step_id not in selected:
                if step_id in prior_passes:
                    _restore_prior_pass(row, prior_passes[step_id])
                else:
                    row["status"] = (
                        "configured_not_run" if row["configured"] else
                        "blocked_missing_disposable_instance_credentials_or_fixture"
                    )
                continue
            definition = definitions[step_id]
            ready = bool(row["configured"]) and disposable and context_ready
            if definition.mode == "read_only":
                ready = ready and write_probe_acknowledged
            if not ready:
                row["status"] = (
                    "blocked_missing_or_stale_release_artifact"
                    if not context_ready else
                    "blocked_missing_disposable_instance_credentials_or_fixture"
                )
                exit_codes[step_id] = None
                continue
            run_environment = {**environment, "SIFT_REQUIRE_LIVE_DATABASES": "1"}
            node = (
                "tests/live/test_database_live.py::"
                f"test_remote_database_requirement[{step_id}]"
            )
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", node],
                cwd=root, env=run_environment, check=False,
            )
            exit_codes[step_id] = completed.returncode
            row["status"] = "passed" if completed.returncode == 0 else "configured_test_run_failed"
            row["test_node"] = node
            row["checked_at"] = checked_at
            if completed.returncode == 0 and evidence_context_sha256 is not None:
                row["evidence_context_sha256"] = evidence_context_sha256
                fixture_sha256 = _fixture_definition_sha256(definition, environment)
                if fixture_sha256 is not None:
                    row["fixture_definition_sha256"] = fixture_sha256

        rows_by_step = {str(row["step_id"]): row for row in scenarios}
        authentication_results = {
            step_id: dict(results)
            for step_id, results in prior_authentication.items()
        }
        for selection in selected_auth:
            step_id, variant = selection.split(":", 1)
            definition = definitions[step_id]
            required_environment = list(
                definition.required_authentication_environment(variant)
            )
            missing_environment = [
                name for name in required_environment if not environment.get(name)
            ]
            configured = not missing_environment
            node = _auth_variant_node(step_id, variant)
            result: dict[str, object] = {
                "variant": variant,
                "configured": configured,
                "missing_environment": missing_environment,
                "status": "blocked_missing_disposable_identity_or_credential",
            }
            ready = configured and disposable and context_ready
            if ready:
                run_environment = {**environment, "SIFT_REQUIRE_LIVE_DATABASES": "1"}
                completed = subprocess.run(
                    [sys.executable, "-m", "pytest", "-q", node],
                    cwd=root, env=run_environment, check=False,
                )
                exit_codes[selection] = completed.returncode
                result["status"] = (
                    "passed" if completed.returncode == 0
                    else "configured_test_run_failed"
                )
                result["test_node"] = node
                result["checked_at"] = checked_at
                if completed.returncode == 0 and evidence_context_sha256 is not None:
                    result["evidence_context_sha256"] = evidence_context_sha256
                    proof_sha256 = _authentication_proof_sha256(
                        environment[required_environment[1]],
                    )
                    if proof_sha256 is None:
                        result["status"] = "configured_test_run_failed"
                    else:
                        result["identity_proof_sha256"] = proof_sha256
                        fixture_sha256 = _fixture_definition_sha256(
                            definition, environment,
                        )
                        if fixture_sha256 is not None:
                            result["fixture_definition_sha256"] = fixture_sha256
            else:
                if not context_ready:
                    result["status"] = "blocked_missing_or_stale_release_artifact"
                exit_codes[selection] = None
            authentication_results.setdefault(step_id, {})[variant] = result

        for definition in definitions.values():
            if definition.mode != "auth":
                continue
            row = rows_by_step[definition.step_id]
            results = authentication_results.get(definition.step_id, {})
            ordered = [
                results[variant]
                for variant in definition.variants
                if variant in results
            ]
            row["authentication_results"] = ordered
            proof_counts: dict[str, int] = {}
            for result in ordered:
                if result.get("status") != "passed":
                    continue
                proof = str(result.get("identity_proof_sha256", ""))
                proof_counts[proof] = proof_counts.get(proof, 0) + 1
            for result in ordered:
                proof = str(result.get("identity_proof_sha256", ""))
                if (
                    result.get("status") == "passed"
                    and (
                        not _is_sha256(proof)
                        or proof_counts.get(proof, 0) != 1
                    )
                ):
                    result["status"] = "configured_test_run_failed"
                    result["proof_error"] = "identity_proof_not_distinct"
            passed_variants = {
                str(result["variant"])
                for result in ordered
                if result.get("status") == "passed"
            }
            remaining = [
                variant for variant in definition.variants
                if variant not in passed_variants
            ]
            distinct_proofs = {
                str(result.get("identity_proof_sha256"))
                for result in ordered
                if result.get("status") == "passed"
            }
            row["remaining_authentication_variants"] = remaining
            row["missing_environment"] = [
                name
                for variant in remaining
                for name in definition.required_authentication_environment(variant)
            ]
            if row.get("status") != "passed" and passed_variants:
                row["status"] = "partial_authentication"
            if (
                not remaining
                and definition.variants
                and len(distinct_proofs) == len(definition.variants)
            ):
                row["status"] = "passed"
                row["configured"] = True
                row["missing_environment"] = []
                row["checked_at"] = max(
                    str(result["checked_at"]) for result in ordered
                )
                row["test_node"] = _auth_aggregate_node(definition.step_id)
                if evidence_context_sha256 is not None:
                    row["evidence_context_sha256"] = evidence_context_sha256

        selected_rows = [row for row in scenarios if row["step_id"] in selected]
        selected_auth_results = [
            authentication_results[selection.split(":", 1)[0]][selection.split(":", 1)[1]]
            for selection in selected_auth
        ]
        failed = (
            any(row["status"] == "configured_test_run_failed" for row in selected_rows)
            or any(
                row["status"] == "configured_test_run_failed"
                for row in selected_auth_results
            )
        )
        all_selected_passed = (
            all(row["status"] == "passed" for row in selected_rows)
            and all(row["status"] == "passed" for row in selected_auth_results)
        )
        all_remote_passed = all(row["status"] == "passed" for row in scenarios)
        report_status = (
            "fail" if failed else "pass" if all_remote_passed else
            "partial" if all_selected_passed else "blocked"
        )
        report = {
            "schema_version": 5,
            "checked_at": checked_at,
            "scope": "Optional disposable live-database compatibility certification",
            "program": "optional_external_compatibility",
            "product_release_blocking": False,
            "provisioning_responsibility": "researcher_or_database_operator",
            "status": report_status,
            "run_mode": "selected_scenarios",
            "selected_scenarios": list(selected),
            "selected_authentication_variants": list(selected_auth),
            "disposable_scope_confirmed": disposable,
            "write_probe_acknowledged": write_probe_acknowledged,
            "strict_test_exit_code": None,
            "selected_test_exit_codes": exit_codes,
            "qualification_context": qualification_context,
            "runtime_provenance": _runtime_provenance(),
            "local": {
                "sqlite": "covered_by_tests/live/test_database_live.py",
                "duckdb": "covered_by_tests/live/test_database_live.py",
            },
            "remote_scenarios": scenarios,
            "security_note": (
                "Connection values are never serialized. Remote certification requires "
                "synthetic fixtures, disposable identities, and explicit acknowledgement "
                "before any write probe."
            ),
        }
        atomic_write_json(output, report)
        print(json.dumps({
            "status": report_status,
            "selected_scenarios": list(selected),
            "selected_authentication_variants": list(selected_auth),
            "passed": (
                sum(row["status"] == "passed" for row in selected_rows)
                + sum(row["status"] == "passed" for row in selected_auth_results)
            ),
        }, indent=2))
        return 0 if all_selected_passed else 2

    prior_passes = _previous_passes(output, qualification_context)
    prior_authentication = _previous_authentication_passes(
        output, qualification_context,
    )
    configured = (
        disposable and write_probe_acknowledged and context_ready
        and all(bool(row["configured"]) for row in scenarios)
    )
    exit_code: int | None = None
    if configured:
        run_environment = {**environment, "SIFT_REQUIRE_LIVE_DATABASES": "1"}
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/live/test_database_live.py"],
            cwd=root, env=run_environment, check=False,
        )
        exit_code = completed.returncode
    passed = configured and exit_code == 0
    for row in scenarios:
        step_id = str(row["step_id"])
        if passed:
            row["status"] = "passed"
            row["test_node"] = (
                "tests/live/test_database_live.py::"
                f"test_remote_database_requirement[{step_id}]"
            )
            row["checked_at"] = datetime.now(timezone.utc).isoformat()
            if evidence_context_sha256 is not None:
                row["evidence_context_sha256"] = evidence_context_sha256
                definition = next(
                    scenario for scenario in DATABASE_CERTIFICATION_SCENARIOS
                    if scenario.step_id == step_id
                )
                fixture_sha256 = _fixture_definition_sha256(definition, environment)
                if fixture_sha256 is not None:
                    row["fixture_definition_sha256"] = fixture_sha256
        elif configured:
            row["status"] = "configured_test_run_failed"
        elif step_id in prior_passes:
            _restore_prior_pass(row, prior_passes[step_id])
        else:
            row["status"] = "blocked_missing_disposable_instance_credentials_or_fixture"
    rows_by_step = {str(row["step_id"]): row for row in scenarios}
    for definition in DATABASE_CERTIFICATION_SCENARIOS:
        if definition.mode != "auth":
            continue
        row = rows_by_step[definition.step_id]
        if row.get("status") == "passed":
            continue
        results = prior_authentication.get(definition.step_id, {})
        ordered = [
            results[variant] for variant in definition.variants
            if variant in results
        ]
        if not ordered:
            continue
        row["authentication_results"] = ordered
        passed_variants = {
            str(result["variant"]) for result in ordered
            if result.get("status") == "passed"
        }
        remaining = [
            variant for variant in definition.variants
            if variant not in passed_variants
        ]
        distinct_proofs = {
            str(result.get("identity_proof_sha256"))
            for result in ordered
            if result.get("status") == "passed"
        }
        row["remaining_authentication_variants"] = remaining
        row["missing_environment"] = [
            name
            for variant in remaining
            for name in definition.required_authentication_environment(variant)
        ]
        if passed_variants:
            row["status"] = "partial_authentication"
        if not remaining and len(distinct_proofs) == len(definition.variants):
            row["status"] = "passed"
            row["configured"] = True
            row["missing_environment"] = []
            row["checked_at"] = max(
                str(result["checked_at"]) for result in ordered
            )
            row["test_node"] = _auth_aggregate_node(definition.step_id)
            if evidence_context_sha256 is not None:
                row["evidence_context_sha256"] = evidence_context_sha256
    all_remote_passed = all(row["status"] == "passed" for row in scenarios)
    report = {
        "schema_version": 5,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Optional disposable live-database compatibility certification",
        "program": "optional_external_compatibility",
        "product_release_blocking": False,
        "provisioning_responsibility": "researcher_or_database_operator",
        "status": "pass" if all_remote_passed else "blocked" if not configured else "fail",
        "run_mode": "strict_all_scenarios",
        "selected_scenarios": [],
        "selected_authentication_variants": [],
        "disposable_scope_confirmed": disposable,
        "write_probe_acknowledged": write_probe_acknowledged,
        "strict_test_exit_code": exit_code,
        "selected_test_exit_codes": {},
        "qualification_context": qualification_context,
        "runtime_provenance": _runtime_provenance(),
        "local": {
            "sqlite": "covered_by_tests/live/test_database_live.py",
            "duckdb": "covered_by_tests/live/test_database_live.py",
        },
        "remote_scenarios": scenarios,
        "security_note": (
            "Connection values are never serialized. Remote certification requires "
            "synthetic fixtures, disposable identities, and explicit write-probe acknowledgement."
        ),
    }
    atomic_write_json(output, report)
    print(json.dumps({"status": report["status"], "scenarios": len(scenarios)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
