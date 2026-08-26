"""Non-destructive backend qualification for releases and future frontends.

The report intentionally distinguishes product-contract validity, runtime
readiness, and per-session evidence integrity.  It never tests live model or
database credentials and never claims that an external provider's data terms
are configured correctly; those remain researcher-controlled concerns.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from sift.capabilities import GENERATED_CODE_TRUST, MODEL_SUPPLY, product_contract

QualificationStatus = Literal["pass", "warning", "fail", "skipped"]
QUALIFICATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class QualificationCheck:
    id: str
    status: QualificationStatus
    detail: str
    scope: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_product_contract() -> list[QualificationCheck]:
    try:
        contract = product_contract()
        json.dumps(contract, sort_keys=True)
        count = len(contract["capabilities"])
    except Exception as exc:  # noqa: BLE001 - qualification must return a verdict
        return [QualificationCheck(
            "product-contract", "fail",
            f"contract construction failed ({type(exc).__name__})", "backend",
        )]
    checks = [QualificationCheck(
        "product-contract", "pass",
        f"runtime-derived contract is valid ({count} capabilities)", "backend",
    )]
    byo = (
        MODEL_SUPPLY.get("models_included") is False
        and MODEL_SUPPLY.get("model_proxy_operated_by_sift") is False
        and MODEL_SUPPLY.get("credentials") == "researcher_supplied"
    )
    checks.append(QualificationCheck(
        "researcher-supplied-models",
        "pass" if byo else "fail",
        (
            "models, accounts, credentials, endpoints, and billing are "
            "researcher supplied"
            if byo else "model-supply contract is inconsistent"
        ),
        "backend",
    ))
    semantics_attested = GENERATED_CODE_TRUST.get(
        "semantics_cryptographically_attested"
    ) is True
    checks.append(QualificationCheck(
        "generated-code-trust-boundary",
        "pass" if semantics_attested else "warning",
        (
            "generated-code semantics are cryptographically attested"
            if semantics_attested else
            "runtime tokens detect framing errors and trivial bypasses, but "
            "code running in the same interpreter can fabricate aggregate-"
            "shaped results; adversarial generated code is not certified"
        ),
        "backend",
    ))
    return checks


def _runtime_checks() -> list[QualificationCheck]:
    from sift.doctor import run_doctor

    try:
        report = run_doctor()
    except Exception as exc:  # noqa: BLE001
        return [QualificationCheck(
            "runtime", "fail", f"runtime probe failed ({type(exc).__name__})",
            "host",
        )]
    status_map: dict[str, QualificationStatus] = {
        "ok": "pass",
        "warning": "warning",
        # Optional language runtimes are capabilities, not installation
        # failures. The overall doctor gate below still fails when none of
        # the languages is usable.
        "unavailable": "warning",
        "blocked": "fail",
    }
    checks = [QualificationCheck(
        f"runtime.{row.runtime}",
        status_map[row.status],
        row.detail,
        "host",
    ) for row in report.runtimes]
    # A missing optional language is a warning when at least one language and
    # the sandbox are usable; DoctorReport.blocked is the authoritative gate.
    if not report.blocked:
        for index, check in enumerate(checks):
            if check.status == "fail" and check.id.startswith("runtime."):
                checks[index] = QualificationCheck(
                    check.id, "warning", check.detail, check.scope,
                )
    checks.append(QualificationCheck(
        "runtime.overall", "fail" if report.blocked else "pass",
        "script execution is blocked" if report.blocked else "script execution is usable",
        "host",
    ))
    return checks


def _sqlite_integrity(db_path: Path) -> QualificationCheck:
    try:
        conn = sqlite3.connect(str(db_path), timeout=5.0)
        try:
            rows = [str(row[0]) for row in conn.execute("PRAGMA integrity_check")]
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        return QualificationCheck(
            "session.results", "fail",
            f"result store could not be checked ({type(exc).__name__})", "session",
        )
    ok = rows == ["ok"]
    return QualificationCheck(
        "session.results", "pass" if ok else "fail",
        "SQLite integrity check passed" if ok else "SQLite integrity check failed",
        "session",
    )


def _provenance_check(metadata_dir: Path) -> QualificationCheck:
    from sift.file_provenance import MANIFEST_FILENAME, MANIFEST_VERSION

    path = metadata_dir / MANIFEST_FILENAME
    if not path.exists():
        return QualificationCheck(
            "session.provenance", "pass", "no staged-file manifest", "session",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError
        version = payload.get("version", 1)
        if version == 1:
            names = payload.get("names")
            valid = isinstance(names, list) and all(isinstance(x, str) for x in names)
            detail = "legacy provenance manifest is parseable; restage files to fingerprint"
            status: QualificationStatus = "warning"
        elif version == MANIFEST_VERSION:
            entries = payload.get("entries")
            valid = isinstance(entries, list) and all(
                isinstance(row, dict)
                and isinstance(row.get("name"), str)
                and isinstance(row.get("sha256"), str)
                and len(row["sha256"]) == 64
                and isinstance(row.get("size_bytes"), int)
                and row["size_bytes"] >= 0
                for row in entries
            )
            detail = f"fingerprinted provenance manifest is valid ({len(entries or [])} entries)"
            status = "pass"
        else:
            valid = False
            detail = "provenance manifest has an unsupported version"
            status = "fail"
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
        detail = "provenance manifest is malformed or unreadable"
        status = "fail"
    return QualificationCheck(
        "session.provenance", status if valid else "fail", detail, "session",
    )


def _session_checks(cwd: Path) -> list[QualificationCheck]:
    from sift.chat_history import history_health
    from sift.release_ledger import ledger_snapshot
    from sift.session_state import SESSION_STATE_FILENAME, read_session_state
    from sift.store import DB_FILENAME, STORE_SUBDIR
    from sift.usage_meter import summarize

    root = Path(cwd)
    if not root.is_dir():
        return [QualificationCheck(
            "session.directory", "fail", "session directory is unavailable", "session",
        )]
    checks = [QualificationCheck(
        "session.directory", "pass", "session directory is available", "session",
    )]
    metadata = root / STORE_SUBDIR
    if not metadata.exists():
        checks.append(QualificationCheck(
            "session.metadata", "pass", "session has no local metadata yet", "session",
        ))
        return checks
    if not metadata.is_dir():
        checks.append(QualificationCheck(
            "session.metadata", "fail", ".sift metadata path is not a directory", "session",
        ))
        return checks

    if os.name == "posix":
        try:
            mode = stat.S_IMODE(metadata.stat().st_mode)
        except OSError as exc:
            checks.append(QualificationCheck(
                "session.permissions", "fail",
                f".sift permissions could not be checked ({type(exc).__name__})",
                "session",
            ))
        else:
            checks.append(QualificationCheck(
                "session.permissions", "pass" if mode & 0o077 == 0 else "fail",
                f".sift permissions are {oct(mode)}",
                "session",
            ))

    transcript = history_health(root)
    checks.append(QualificationCheck(
        "session.transcript", "pass" if transcript.ok else "fail",
        transcript.detail, "session",
    ))
    ledger = ledger_snapshot(root)
    checks.append(QualificationCheck(
        "session.release-ledger", "pass" if ledger.integrity_ok else "fail",
        ledger.detail, "session",
    ))
    usage = summarize(root)
    accounting_ok = bool(usage.get("usage_accounting_complete", False))
    checks.append(QualificationCheck(
        "session.usage-accounting", "pass" if accounting_ok else "fail",
        str(usage.get("usage_accounting_detail", "unknown")), "session",
    ))
    if not usage.get("pricing_complete", True):
        checks.append(QualificationCheck(
            "session.usage-pricing", "warning",
            "at least one selected model has no local price estimate", "session",
        ))
    checks.append(_provenance_check(metadata))

    state_path = metadata / SESSION_STATE_FILENAME
    state = read_session_state(root) if state_path.exists() else None
    checks.append(QualificationCheck(
        "session.state",
        "pass" if not state_path.exists() or state is not None else "fail",
        (
            "session state is absent or valid"
            if not state_path.exists() or state is not None
            else "session state is malformed"
        ),
        "session",
    ))
    db_path = metadata / DB_FILENAME
    if db_path.exists():
        checks.append(_sqlite_integrity(db_path))
    else:
        checks.append(QualificationCheck(
            "session.results", "pass", "no result store yet", "session",
        ))
    return checks


def run_qualification(
    cwd: Path | None = None, *, include_runtime: bool = True,
) -> dict[str, Any]:
    """Return a stable, JSON-safe qualification report without live API calls."""
    checks = _check_product_contract()
    if include_runtime:
        checks.extend(_runtime_checks())
    else:
        checks.append(QualificationCheck(
            "runtime", "skipped", "runtime probes were not requested", "host",
        ))
    if cwd is not None:
        try:
            checks.extend(_session_checks(Path(cwd)))
        except Exception as exc:  # noqa: BLE001 - report must remain available
            checks.append(QualificationCheck(
                "session.unavailable", "fail",
                f"session qualification failed ({type(exc).__name__})", "session",
            ))
    failures = sum(check.status == "fail" for check in checks)
    warnings = sum(check.status == "warning" for check in checks)
    overall = "fail" if failures else "warning" if warnings else "pass"
    return {
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "overall": overall,
        "failures": failures,
        "warnings": warnings,
        "checks": [check.as_dict() for check in checks],
        "live_external_services_tested": False,
        "model_access_included": False,
        "adversarial_generated_code_certified": False,
    }


__all__ = [
    "QUALIFICATION_SCHEMA_VERSION",
    "QualificationCheck",
    "run_qualification",
]
