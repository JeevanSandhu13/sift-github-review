from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from sift.backend_api import (
    API_VERSION,
    BACKEND_API_V1_CONTRACT_SHA256,
    ENDPOINTS,
    EVENT_SCHEMAS,
    BackendApplication,
    backend_contract,
    backend_contract_sha256,
    migrate_request,
    secret_safe_response,
    validate_schema,
)
from sift.config import WorkspaceScopeError
from sift.integration_core import CancellationToken
from sift.store import close_store, get_store


@pytest.fixture
def backend(tmp_path: Path):
    database = tmp_path / "research.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE observations (id INTEGER, value REAL)")
    connection.executemany(
        "INSERT INTO observations VALUES (?, ?)",
        [(index, index * 0.5) for index in range(20)],
    )
    connection.commit()
    connection.close()
    row = get_store(tmp_path).insert(
        label="Known summary", analysis_type="descriptive",
        sanitized_payload={
            "type": "descriptive", "variable": "value", "n": 20,
            "mean": 4.75, "sd": 2.958, "missing_count": 0,
        }, language="Python", script_code="", transformations=[],
    )
    app = BackendApplication(tmp_path)
    try:
        yield app, f"sqlite:///{database}", row.id
    finally:
        close_store(tmp_path)


def test_frozen_contract_hash_and_version_are_pinned() -> None:
    assert BACKEND_API_V1_CONTRACT_SHA256 != "TO_BE_FROZEN"
    assert backend_contract_sha256() == BACKEND_API_V1_CONTRACT_SHA256
    contract = backend_contract()
    assert contract["api_version"] == API_VERSION
    assert contract["frozen"] is True
    assert contract["contract_sha256"] == contract["frozen_sha256"]


def test_every_backend_endpoint_has_a_contract_and_handler(backend) -> None:
    app, connection, result_id = backend
    app._operations["cancel-me"] = CancellationToken()
    requests = {
        "contract.get": {},
        "capabilities.list": {},
        "qualification.get": {"include_runtime": False},
        "integrations.list": {},
        "integrations.setup.schema": {"integration_id": "openai", "kind": "model"},
        "credentials.entry.schema": {"integration_id": "openai"},
        "integrations.connection.test": {"connection": connection},
        "integrations.catalog.discover": {"connection": connection},
        "extractions.database.run": {
            "operation_id": "extract-contract-test", "connection": connection,
            "sql": "SELECT * FROM observations", "dataset_name": "contract_rows",
        },
        "operations.cancel": {"operation_id": "cancel-me"},
        "research.plan.evaluate": {
            "method_id": "linear_regression", "research_specification": {},
        },
        "evidence.verify": {"result_id": result_id},
        "privacy.warning.review": {
            "text": "Patient date of birth is 2000-01-01",
            "attachment_names": [], "field_names": [],
        },
    }
    assert set(requests) == set(ENDPOINTS)
    for endpoint, request in requests.items():
        events = []
        response = app.call(endpoint, request, progress=events.append)
        assert response["ok"] is True, (endpoint, response)
        assert response["endpoint"] == endpoint
        json.dumps(response)
        for event in events:
            validate_schema(event, EVENT_SCHEMAS[event["type"]], path="event")


def test_request_validation_and_all_errors_are_structured(backend) -> None:
    app, _connection, _result_id = backend
    for endpoint, request in (
        ("missing.endpoint", {}),
        ("integrations.connection.test", {}),
        ("capabilities.list", {"unknown": True}),
        ("capabilities.list", ["not", "an", "object"]),
    ):
        response = app.call(endpoint, request)  # type: ignore[arg-type]
        assert response["ok"] is False
        assert set(response["error"]) >= {"code", "message", "action", "retryable"}
        json.dumps(response)


def test_legacy_request_migration_is_backwards_compatible_and_nonmutating() -> None:
    legacy = {"uri": "sqlite:///research.sqlite"}
    migrated = migrate_request("integrations.connection.test", legacy, "0.9")
    assert legacy == {"uri": "sqlite:///research.sqlite"}
    assert migrated == {"connection": "sqlite:///research.sqlite"}
    conflict = BackendApplication().call(
        "integrations.connection.test",
        {"uri": "sqlite:///a", "connection": "sqlite:///b"},
        request_version="0.9",
    )
    assert conflict["ok"] is False
    assert conflict["error"]["code"] == "ambiguous_migration"


def test_credential_entry_is_write_only_and_secret_free(backend) -> None:
    app, _connection, _result_id = backend
    response = app.call("credentials.entry.schema", {"integration_id": "openai"})
    assert response["ok"] is True
    secret_methods = [
        row for row in response["data"]["methods"] if row["secret_bearing"]
    ]
    assert secret_methods
    assert all(row["input"]["write_only"] for row in secret_methods)
    assert all(row["input"]["echo_in_response"] is False for row in secret_methods)


def test_final_response_guard_removes_exact_and_pattern_shaped_secrets() -> None:
    secret = "super-secret-value"
    value = {
        "nested": [
            f"request failed with {secret}",
            "Authorization: Bearer abcdef123456",
            "Authorization: Basic dXNlcjpwYXNzd29yZA==",
            "Set-Cookie: session=browser-secret; Path=/; Secure",
            "password=hunter2",
        ],
        "api_key": "novel-secret-format",
    }
    safe = secret_safe_response(value, secrets=(secret,))
    encoded = json.dumps(safe)
    assert secret not in encoded
    assert "abcdef123456" not in encoded
    assert "dXNlcjpwYXNzd29yZA==" not in encoded
    assert "browser-secret" not in encoded
    assert "hunter2" not in encoded
    assert "novel-secret-format" not in encoded
    assert encoded.count("***") >= 6
    assert secret_safe_response("", secrets=("x",)) == ""
    assert secret_safe_response("ordinary response", secrets=()) == "ordinary response"


def test_final_response_guard_is_cycle_and_depth_safe() -> None:
    cycle: dict[str, object] = {}
    cycle["child"] = cycle
    assert secret_safe_response(cycle) == {
        "child": "[cyclic response omitted]",
    }

    nested: object = "leaf"
    for _ in range(100):
        nested = [nested]
    assert "invalid nested response omitted" in json.dumps(
        secret_safe_response(nested)
    )


def test_backend_application_rejects_dangerously_broad_workspace() -> None:
    with pytest.raises(WorkspaceScopeError):
        BackendApplication(Path(Path.cwd().anchor))
    app = BackendApplication()
    with pytest.raises(WorkspaceScopeError):
        app.set_cwd(Path.home())


def test_connection_password_cannot_reappear_in_structured_error(monkeypatch, backend) -> None:
    app, _connection, _result_id = backend
    secret = "top-secret-password"

    def leak(*_args, **_kwargs):
        raise RuntimeError(f"password={secret}; Authorization: Bearer token-value")

    monkeypatch.setattr(app, "_handle", leak)
    response = app.call(
        "integrations.connection.test",
        {"connection": f"postgresql://alice:{secret}@db.invalid/research"},
    )
    encoded = json.dumps(response)
    assert response["ok"] is False
    assert secret not in encoded and "token-value" not in encoded
    assert response["error"]["code"] == "backend_failure"


def test_extraction_progress_and_cross_thread_cancellation(monkeypatch, tmp_path: Path) -> None:
    from sift import connectors

    app = BackendApplication(tmp_path)
    started = threading.Event()
    progress_events: list[dict] = []

    def slow_extract(_cwd, *, cancellation, progress, **_kwargs):
        progress(connectors.ExtractionProgress("starting", 0, 0))
        started.set()
        while not cancellation.wait(0.01):
            pass
        raise connectors.ConnectorError(
            "database extraction cancelled", code="cancelled",
            action="Start a new extraction when ready.",
        )

    monkeypatch.setattr(connectors, "run_extract", slow_extract)
    result: list[dict] = []

    def run() -> None:
        result.append(app.call(
            "extractions.database.run", {
                "operation_id": "slow-operation", "connection": "sqlite:///:memory:",
                "sql": "SELECT 1", "dataset_name": "slow",
            }, progress=progress_events.append,
        ))

    thread = threading.Thread(target=run)
    thread.start()
    assert started.wait(2)
    cancelled = app.call("operations.cancel", {"operation_id": "slow-operation"})
    thread.join(timeout=2)
    assert cancelled["ok"] is True
    assert cancelled["data"]["event"]["type"] == "operation.cancelled"
    assert result and result[0]["ok"] is False
    assert result[0]["error"]["code"] == "cancelled"
    assert progress_events[0]["type"] == "extraction.progress"


def test_extraction_stays_bound_to_submission_project(monkeypatch, tmp_path: Path) -> None:
    from sift import connectors

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    app = BackendApplication(first)
    started = threading.Event()
    release = threading.Event()
    observed: list[Path] = []

    def slow_extract(cwd, **_kwargs):
        started.set()
        assert release.wait(2)
        observed.append(Path(cwd))
        return connectors.ExtractResult(
            dataset_path=Path(cwd) / "rows.parquet",
            rows=0,
            columns=0,
            truncated=False,
            backend="sqlite",
            connection_display="sqlite",
            query_sha256="a" * 64,
            dataset_sha256="b" * 64,
        )

    monkeypatch.setattr(connectors, "run_extract", slow_extract)
    result: list[dict] = []
    thread = threading.Thread(target=lambda: result.append(app.call(
        "extractions.database.run",
        {
            "connection": "sqlite:///:memory:",
            "sql": "SELECT 1",
            "dataset_name": "rows",
        },
    )))
    thread.start()
    assert started.wait(2)
    app.set_cwd(second)
    release.set()
    thread.join(timeout=2)
    assert observed == [first.resolve()]
    assert result and result[0]["ok"] is True


def test_legacy_webview_bridge_is_only_a_thin_backend_adapter(tmp_path: Path) -> None:
    from sift.ui import SiftBridge

    bridge = SiftBridge(cwd=tmp_path)
    contract = bridge.get_backend_api_contract()
    response = bridge.call_backend_api("capabilities.list", {})
    assert contract["ok"] is True
    assert contract["contract_sha256"] == BACKEND_API_V1_CONTRACT_SHA256
    assert response["ok"] is True
    assert response["api_version"] == API_VERSION
