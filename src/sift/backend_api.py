"""Frozen, UI-neutral backend contract for Sift's future GUI.

This module is an application-service boundary, not a web server and not a
GUI.  Any future desktop, browser, or assistive frontend can call the same
versioned endpoints and render the returned view-ready data.  Business logic
remains in Sift's domain modules; this layer validates requests, coordinates
operations, normalizes errors/events, and enforces a final secret-redaction
boundary on every response.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlsplit

from sift.config import validate_workspace
from sift.integration_core import CancellationToken, IntegrationError

API_VERSION = "1.0"
API_SCHEMA_VERSION = 1
MINIMUM_REQUEST_VERSION = "0.9"
SUPPORTED_REQUEST_VERSIONS = (MINIMUM_REQUEST_VERSION, API_VERSION)
BACKEND_API_V1_CONTRACT_SHA256 = "9effa75f93ca69f9db4de31d6bb13e8515e96895fa4d9a0c7772ecc7ca84ce71"

JsonSchema = dict[str, Any]


@dataclass(frozen=True)
class EndpointSpec:
    id: str
    description: str
    request_schema: JsonSchema
    response_data_schema: JsonSchema
    events: tuple[str, ...] = ()
    mutates_local_state: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendEvent:
    type: str
    operation_id: str
    sequence: int
    terminal: bool
    payload: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_version": API_VERSION,
            "event_schema_version": API_SCHEMA_VERSION,
            "type": self.type,
            "operation_id": self.operation_id,
            "sequence": self.sequence,
            "terminal": self.terminal,
            **dict(self.payload),
        }


class BackendContractError(Exception):
    """Structured, user-actionable application-service failure."""

    def __init__(
        self, code: str, message: str, *, action: str,
        retryable: bool = False, field: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.action = action
        self.retryable = retryable
        self.field = field

    def as_dict(self) -> dict[str, Any]:
        value = {
            "code": self.code, "message": str(self), "action": self.action,
            "retryable": self.retryable,
        }
        if self.field:
            value["field"] = self.field
        return value


def _object(
    properties: Mapping[str, JsonSchema], *, required: Sequence[str] = (),
    additional: bool = False,
) -> JsonSchema:
    return {
        "type": "object", "properties": dict(properties),
        "required": list(required), "additionalProperties": additional,
    }


_STRING = {"type": "string", "maxLength": 10_000}
_SHORT_STRING = {"type": "string", "maxLength": 255}
_BOOL = {"type": "boolean"}
_NONNEGATIVE_INT = {"type": "integer", "minimum": 0}
_OBJECT_ANY = {"type": "object", "additionalProperties": True}

ERROR_SCHEMA = _object({
    "code": _SHORT_STRING, "message": _STRING, "action": _STRING,
    "retryable": _BOOL, "field": _SHORT_STRING,
}, required=("code", "message", "action", "retryable"), additional=False)

RESPONSE_ENVELOPE_SCHEMA = _object({
    "api_version": {"type": "string", "const": API_VERSION},
    "schema_version": {"type": "integer", "const": API_SCHEMA_VERSION},
    "request_id": _SHORT_STRING, "endpoint": _SHORT_STRING, "ok": _BOOL,
    "data": _OBJECT_ANY, "error": ERROR_SCHEMA,
}, required=("api_version", "schema_version", "request_id", "endpoint", "ok"))

EVENT_SCHEMAS: dict[str, JsonSchema] = {
    "extraction.progress": _object({
        "api_version": _SHORT_STRING, "event_schema_version": _NONNEGATIVE_INT,
        "type": {"type": "string", "const": "extraction.progress"},
        "operation_id": _SHORT_STRING, "sequence": _NONNEGATIVE_INT,
        "terminal": _BOOL,
        "stage": {"type": "string", "enum": [
            "starting", "querying", "materializing", "finalizing", "complete",
        ]},
        "rows_materialized": _NONNEGATIVE_INT,
        "bytes_buffered": _NONNEGATIVE_INT,
    }, required=("api_version", "event_schema_version", "type", "operation_id",
                 "sequence", "terminal", "stage", "rows_materialized",
                 "bytes_buffered")),
    "operation.cancelled": _object({
        "api_version": _SHORT_STRING, "event_schema_version": _NONNEGATIVE_INT,
        "type": {"type": "string", "const": "operation.cancelled"},
        "operation_id": _SHORT_STRING, "sequence": _NONNEGATIVE_INT,
        "terminal": _BOOL,
    }, required=("api_version", "event_schema_version", "type", "operation_id",
                 "sequence", "terminal")),
    "research.plan": _object({
        "api_version": _SHORT_STRING, "event_schema_version": _NONNEGATIVE_INT,
        "type": {"type": "string", "const": "research.plan"},
        "operation_id": _SHORT_STRING, "sequence": _NONNEGATIVE_INT,
        "terminal": _BOOL, "valid": _BOOL,
        "method_id": _SHORT_STRING,
        "clarifications": {"type": "array", "items": _STRING, "maxItems": 100},
    }, required=("api_version", "event_schema_version", "type", "operation_id",
                 "sequence", "terminal", "valid", "method_id", "clarifications")),
    "evidence.verification": _object({
        "api_version": _SHORT_STRING, "event_schema_version": _NONNEGATIVE_INT,
        "type": {"type": "string", "const": "evidence.verification"},
        "operation_id": _SHORT_STRING, "sequence": _NONNEGATIVE_INT,
        "terminal": _BOOL, "result_id": _SHORT_STRING,
        "verification": _OBJECT_ANY,
    }, required=("api_version", "event_schema_version", "type", "operation_id",
                 "sequence", "terminal", "result_id", "verification")),
    "privacy.warning": _object({
        "api_version": _SHORT_STRING, "event_schema_version": _NONNEGATIVE_INT,
        "type": {"type": "string", "const": "privacy.warning"},
        "operation_id": _SHORT_STRING, "sequence": _NONNEGATIVE_INT,
        "terminal": _BOOL, "warn": _BOOL,
        "findings": {"type": "array", "items": _OBJECT_ANY, "maxItems": 100},
    }, required=("api_version", "event_schema_version", "type", "operation_id",
                 "sequence", "terminal", "warn", "findings")),
}


ENDPOINTS: dict[str, EndpointSpec] = {
    "contract.get": EndpointSpec(
        "contract.get", "Return this frozen backend contract.",
        _object({}), _OBJECT_ANY,
    ),
    "capabilities.list": EndpointSpec(
        "capabilities.list", "Return view-ready product capabilities.",
        _object({}), _object({
            "compatibility_version": _NONNEGATIVE_INT,
            "categories": {"type": "array", "items": _OBJECT_ANY},
            "capabilities": {"type": "array", "items": _OBJECT_ANY},
        }, required=("compatibility_version", "categories", "capabilities")),
    ),
    "qualification.get": EndpointSpec(
        "qualification.get", "Return backend/runtime/session qualification.",
        _object({"include_runtime": _BOOL}), _OBJECT_ANY,
    ),
    "integrations.list": EndpointSpec(
        "integrations.list", "Return integration trust and readiness data.",
        _object({}), _OBJECT_ANY,
    ),
    "integrations.setup.schema": EndpointSpec(
        "integrations.setup.schema", "Describe setup and authentication fields.",
        _object({"integration_id": _SHORT_STRING, "kind": _SHORT_STRING},
                required=("integration_id",)),
        _object({"integration": _OBJECT_ANY, "credential_entry": _OBJECT_ANY},
                required=("integration", "credential_entry")),
    ),
    "credentials.entry.schema": EndpointSpec(
        "credentials.entry.schema", "Return a write-only credential-entry contract.",
        _object({"integration_id": _SHORT_STRING}, required=("integration_id",)),
        _OBJECT_ANY,
    ),
    "integrations.connection.test": EndpointSpec(
        "integrations.connection.test", "Test a database connection without row reads.",
        _object({"connection": _STRING}, required=("connection",)),
        _object({
            "backend": _SHORT_STRING, "connection_display": _STRING,
            "latency_ms": _NONNEGATIVE_INT, "server_version": {
                "type": ["string", "null"], "maxLength": 255,
            }, "read_only_enforcement": _STRING, "sampled_rows": _NONNEGATIVE_INT,
        }, required=("backend", "connection_display", "latency_ms",
                     "server_version", "read_only_enforcement", "sampled_rows")),
    ),
    "integrations.catalog.discover": EndpointSpec(
        "integrations.catalog.discover", "Discover bounded database metadata only.",
        _object({
            "connection": _STRING, "schema": _SHORT_STRING,
            "object_name": _SHORT_STRING,
        }, required=("connection",)), _OBJECT_ANY,
    ),
    "extractions.database.run": EndpointSpec(
        "extractions.database.run", "Materialize a bounded read-only query locally.",
        _object({
            "operation_id": _SHORT_STRING, "connection": _STRING,
            "sql": _STRING, "dataset_name": _SHORT_STRING,
            "row_limit": {"type": "integer", "minimum": 1, "maximum": 1_000_000},
        }, required=("connection", "sql", "dataset_name")),
        _OBJECT_ANY, events=("extraction.progress",), mutates_local_state=True,
    ),
    "operations.cancel": EndpointSpec(
        "operations.cancel", "Cooperatively cancel one active backend operation.",
        _object({"operation_id": _SHORT_STRING}, required=("operation_id",)),
        _object({"cancelled": _BOOL, "event": _OBJECT_ANY},
                required=("cancelled", "event")),
        events=("operation.cancelled",), mutates_local_state=True,
    ),
    "research.plan.evaluate": EndpointSpec(
        "research.plan.evaluate", "Validate a method against a research specification.",
        _object({
            "method_id": _SHORT_STRING, "research_specification": _OBJECT_ANY,
        }, required=("method_id", "research_specification")),
        _object({"evaluation": _OBJECT_ANY, "event": _OBJECT_ANY},
                required=("evaluation", "event")), events=("research.plan",),
    ),
    "evidence.verify": EndpointSpec(
        "evidence.verify", "Verify one stored sanitized result.",
        _object({"result_id": _SHORT_STRING}, required=("result_id",)),
        _object({"result_id": _SHORT_STRING, "verification": _OBJECT_ANY,
                 "event": _OBJECT_ANY},
                required=("result_id", "verification", "event")),
        events=("evidence.verification",),
    ),
    "privacy.warning.review": EndpointSpec(
        "privacy.warning.review", "Locally review a prospective provider disclosure.",
        _object({
            "text": _STRING,
            "attachment_names": {"type": "array", "items": _SHORT_STRING,
                                 "maxItems": 100},
            "field_names": {"type": "array", "items": _SHORT_STRING,
                            "maxItems": 10_000},
            "organization_sensitive_fields": {
                "type": "array", "items": _SHORT_STRING, "maxItems": 10_000,
            }, "enabled": _BOOL,
        }, required=("text",)),
        _object({"review": _OBJECT_ANY, "event": _OBJECT_ANY},
                required=("review", "event")), events=("privacy.warning",),
    ),
}


_LEGACY_RENAMES: dict[str, dict[str, str]] = {
    "integrations.connection.test": {"uri": "connection"},
    "integrations.catalog.discover": {
        "uri": "connection", "table": "object_name",
    },
    "extractions.database.run": {
        "uri": "connection", "query": "sql", "name": "dataset_name",
    },
}


def migrate_request(
    endpoint: str, request: Mapping[str, Any], from_version: str,
) -> dict[str, Any]:
    """Migrate a supported older request without changing the caller object."""
    if from_version not in SUPPORTED_REQUEST_VERSIONS:
        raise BackendContractError(
            "unsupported_api_version",
            f"Request version {from_version!r} is not supported.",
            action=f"Send {API_VERSION} requests or migrate from {MINIMUM_REQUEST_VERSION}.",
        )
    value = dict(request)
    if from_version == MINIMUM_REQUEST_VERSION:
        for old, new in _LEGACY_RENAMES.get(endpoint, {}).items():
            if old in value:
                if new in value:
                    raise BackendContractError(
                        "ambiguous_migration",
                        f"Both legacy field {old!r} and current field {new!r} were supplied.",
                        action="Send only the current field name.", field=new,
                    )
                value[new] = value.pop(old)
    return value


def _type_matches(value: Any, expected: Any) -> bool:
    kinds = expected if isinstance(expected, list) else [expected]
    for kind in kinds:
        if kind == "null" and value is None:
            return True
        if kind == "object" and isinstance(value, Mapping):
            return True
        if kind == "array" and isinstance(value, list):
            return True
        if kind == "string" and isinstance(value, str):
            return True
        if kind == "boolean" and isinstance(value, bool):
            return True
        if kind == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if kind == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
    return False


def validate_schema(value: Any, schema: Mapping[str, Any], *, path: str = "request") -> None:
    expected = schema.get("type")
    if expected is not None and not _type_matches(value, expected):
        raise BackendContractError(
            "invalid_request", f"{path} has the wrong type.",
            action="Correct the highlighted field and try again.", field=path,
        )
    if "const" in schema and value != schema["const"]:
        raise BackendContractError(
            "invalid_request", f"{path} does not match the required constant.",
            action="Use the value declared by the current API contract.", field=path,
        )
    if "enum" in schema and value not in schema["enum"]:
        raise BackendContractError(
            "invalid_request", f"{path} is not an allowed value.",
            action="Choose one of the values in the endpoint schema.", field=path,
        )
    if isinstance(value, str):
        if len(value) > int(schema.get("maxLength", len(value))):
            raise BackendContractError(
                "request_too_large", f"{path} exceeds its length limit.",
                action="Shorten the field and try again.", field=path,
            )
        if len(value) < int(schema.get("minLength", 0)):
            raise BackendContractError(
                "invalid_request", f"{path} is too short.",
                action="Provide the required value.", field=path,
            )
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            raise BackendContractError(
                "invalid_request", f"{path} must be finite.",
                action="Provide a finite numeric value.", field=path,
            )
        if "minimum" in schema and value < schema["minimum"]:
            raise BackendContractError(
                "invalid_request", f"{path} is below its minimum.",
                action="Use a value within the documented range.", field=path,
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise BackendContractError(
                "invalid_request", f"{path} exceeds its maximum.",
                action="Use a value within the documented range.", field=path,
            )
    if isinstance(value, list):
        if len(value) > int(schema.get("maxItems", len(value))):
            raise BackendContractError(
                "request_too_large", f"{path} has too many items.",
                action="Narrow the request and try again.", field=path,
            )
        item_schema = schema.get("items")
        if isinstance(item_schema, Mapping):
            for index, item in enumerate(value):
                validate_schema(item, item_schema, path=f"{path}[{index}]")
    if isinstance(value, Mapping):
        properties = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in value:
                raise BackendContractError(
                    "missing_field", f"{path}.{field} is required.",
                    action="Provide the required field and try again.",
                    field=f"{path}.{field}",
                )
        if schema.get("additionalProperties") is False:
            unknown = sorted(str(key) for key in value if key not in properties)
            if unknown:
                raise BackendContractError(
                    "unknown_field", f"{path} contains unsupported fields: {', '.join(unknown)}.",
                    action="Remove fields not declared by the endpoint schema.",
                    field=f"{path}.{unknown[0]}",
                )
        for field, item in value.items():
            child = properties.get(field)
            if isinstance(child, Mapping):
                validate_schema(item, child, path=f"{path}.{field}")


_SECRET_KEYS = frozenset({
    "api_key", "password", "passwd", "pwd", "token", "access_token",
    "refresh_token", "client_secret", "authorization", "cookie",
    "private_key", "passphrase", "connection", "uri",
})
_SECRET_VALUE_KEYS = _SECRET_KEYS - {"connection", "uri"}
_RESPONSE_AUTHORIZATION_RE = re.compile(
    r"(?i)(\b(?:proxy-authorization|authorization)\s*[:=]\s*)[^\r\n]+",
)
_RESPONSE_COOKIE_RE = re.compile(
    r"(?i)(\b(?:set-cookie|cookie)\s*[:=]\s*)[^\r\n]+",
)
_RESPONSE_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|"
    r"client[_-]?secret|password|passwd|pwd)\b\s*[:=]\s*[\"']?)"
    r"[^\"'\s,;&}]+",
)
_RESPONSE_URL_USERINFO_RE = re.compile(
    r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s/@:]+:)[^\s/@]+(@)",
)


def _normalized_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")


def _request_secrets(
    value: Any,
    *,
    key: str = "",
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> tuple[str, ...]:
    """Collect exact request credentials without trusting object structure."""
    if _depth > 64:
        return ()
    if _seen is None:
        _seen = set()
    found: list[str] = []
    normalized = _normalized_key(key)
    if isinstance(value, str):
        if normalized in _SECRET_VALUE_KEYS and value:
            found.append(value)
        if normalized in {"connection", "uri"}:
            try:
                parsed = urlsplit(value)
                if parsed.password:
                    found.append(unquote(parsed.password))
                for parameter, item in parse_qsl(parsed.query, keep_blank_values=True):
                    query_key = re.sub(
                        r"[^a-z0-9]+", "_", parameter.casefold(),
                    ).strip("_")
                    if query_key in _SECRET_KEYS and item:
                        found.append(item)
            except (TypeError, ValueError):
                pass
    elif isinstance(value, Mapping):
        identity = id(value)
        if identity in _seen:
            return ()
        _seen.add(identity)
        try:
            for child_key, child in value.items():
                found.extend(_request_secrets(
                    child, key=str(child_key), _seen=_seen, _depth=_depth + 1,
                ))
        finally:
            _seen.discard(identity)
    elif isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in _seen:
            return ()
        _seen.add(identity)
        try:
            for child in value:
                found.extend(_request_secrets(
                    child, key=key, _seen=_seen, _depth=_depth + 1,
                ))
        finally:
            _seen.discard(identity)
    return tuple(dict.fromkeys(found))


def secret_safe_response(value: Any, *, secrets: Sequence[str] = ()) -> Any:
    """Recursively redact known values and credential-shaped diagnostics.

    Secret-named response fields are write-only even when an integration
    returns a novel secret format. Traversal is bounded so an invalid adapter
    response cannot escape the structured backend-error boundary.
    """
    exact = tuple(secret for secret in secrets if isinstance(secret, str) and secret)
    active: set[int] = set()

    def scrub(item: Any, *, key: str = "", depth: int = 0) -> Any:
        if _normalized_key(key) in _SECRET_VALUE_KEYS:
            return "***"
        if depth > 64:
            return "[invalid nested response omitted]"
        if isinstance(item, str):
            text = item
            for secret in exact:
                text = text.replace(secret, "***")
            text = _RESPONSE_AUTHORIZATION_RE.sub(r"\1***", text)
            text = _RESPONSE_COOKIE_RE.sub(r"\1***", text)
            text = _RESPONSE_SECRET_ASSIGNMENT_RE.sub(r"\1***", text)
            text = _RESPONSE_URL_USERINFO_RE.sub(r"\1***\2", text)
            return text
        if isinstance(item, Mapping):
            identity = id(item)
            if identity in active:
                return "[cyclic response omitted]"
            active.add(identity)
            try:
                return {
                    str(child_key): scrub(
                        child, key=str(child_key), depth=depth + 1,
                    )
                    for child_key, child in item.items()
                }
            finally:
                active.discard(identity)
        if isinstance(item, (tuple, list)):
            identity = id(item)
            if identity in active:
                return "[cyclic response omitted]"
            active.add(identity)
            try:
                return [scrub(child, depth=depth + 1) for child in item]
            finally:
                active.discard(identity)
        return item

    return scrub(value)


def credential_entry_contract(integration_id: str) -> dict[str, Any]:
    from sift.integrations import integration_contracts

    matches = [row for row in integration_contracts() if row.id == integration_id]
    if not matches:
        raise BackendContractError(
            "integration_not_found", "The requested integration is not registered.",
            action="Choose an integration returned by integrations.list.",
            field="request.integration_id",
        )
    methods = []
    for auth in matches[0].authentication:
        methods.append({
            "id": auth.id, "kind": auth.kind,
            "secret_bearing": auth.secret_bearing,
            "storage": auth.storage, "description": auth.description,
            "input": ({
                "type": "secret", "write_only": True, "echo_in_response": False,
                "autocomplete": "off", "clear_after_submit": True,
            } if auth.secret_bearing else None),
        })
    return {
        "integration_id": integration_id, "methods": methods,
        "secrets_persisted_in_application_metadata": False,
        "responses_may_contain_secret": False,
    }


def view_ready_capabilities() -> dict[str, Any]:
    from sift.capabilities import CAPABILITY_COMPATIBILITY_VERSION, capabilities

    rows = [row.as_dict() for row in capabilities()]
    category_ids = sorted({str(row["category"]) for row in rows})
    categories = [{
        "id": category, "label": category.replace("_", " ").title(),
        "count": sum(row["category"] == category for row in rows),
    } for category in category_ids]
    return {
        "compatibility_version": CAPABILITY_COMPATIBILITY_VERSION,
        "categories": categories,
        "capabilities": [{
            "id": row["id"], "category": row["category"], "label": row["label"],
            "status": row["status"], "claim": row["claim"],
            "limitations": list(row["limitations"]),
            "acceptance_criteria": list(row["acceptance_criteria"]),
        } for row in rows],
    }


def _canonical_contract_payload() -> dict[str, Any]:
    return {
        "api_version": API_VERSION,
        "schema_version": API_SCHEMA_VERSION,
        "minimum_request_version": MINIMUM_REQUEST_VERSION,
        "supported_request_versions": list(SUPPORTED_REQUEST_VERSIONS),
        "response_envelope_schema": RESPONSE_ENVELOPE_SCHEMA,
        "event_schemas": EVENT_SCHEMAS,
        "endpoints": {name: spec.as_dict() for name, spec in ENDPOINTS.items()},
        "compatibility_policy": {
            "additive_optional_fields_allowed": True,
            "removing_or_retyping_fields_requires_new_major_version": True,
            "legacy_requests_migrated_without_mutating_input": True,
        },
        "frozen": True,
    }


def backend_contract_sha256() -> str:
    encoded = json.dumps(
        _canonical_contract_payload(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def backend_contract() -> dict[str, Any]:
    value = _canonical_contract_payload()
    value["contract_sha256"] = backend_contract_sha256()
    value["frozen_sha256"] = BACKEND_API_V1_CONTRACT_SHA256
    return value


ProgressCallback = Callable[[dict[str, Any]], None]


class BackendApplication:
    """Stateful application-service coordinator shared by any frontend."""

    def __init__(self, cwd: Path | None = None) -> None:
        self.cwd = validate_workspace(Path(cwd)) if cwd is not None else None
        self._operations: dict[str, CancellationToken] = {}
        self._lock = threading.RLock()

    def set_cwd(self, cwd: Path | None) -> None:
        self.cwd = validate_workspace(Path(cwd)) if cwd is not None else None

    def _require_cwd(self) -> Path:
        if self.cwd is None or not self.cwd.is_dir():
            raise BackendContractError(
                "no_active_session", "No active research session is selected.",
                action="Select or create a research session and try again.",
            )
        return self.cwd

    def _event(self, event_type: str, operation_id: str, sequence: int,
               terminal: bool, **payload: Any) -> dict[str, Any]:
        event = BackendEvent(event_type, operation_id, sequence, terminal, payload).as_dict()
        validate_schema(event, EVENT_SCHEMAS[event_type], path="event")
        return event

    def _handle(self, endpoint: str, request: Mapping[str, Any],
                progress: ProgressCallback | None) -> dict[str, Any]:
        if endpoint == "contract.get":
            return backend_contract()
        if endpoint == "capabilities.list":
            return view_ready_capabilities()
        if endpoint == "qualification.get":
            from sift.qualification import run_qualification

            return run_qualification(
                self.cwd, include_runtime=bool(request.get("include_runtime", True)),
            )
        if endpoint == "integrations.list":
            from sift.integrations import list_integration_manifests

            return list_integration_manifests()
        if endpoint in {"integrations.setup.schema", "credentials.entry.schema"}:
            from sift.integrations import integration_contracts

            integration_id = str(request["integration_id"])
            kind = request.get("kind")
            matches = [row for row in integration_contracts()
                       if row.id == integration_id and (kind is None or row.kind == kind)]
            if not matches:
                raise BackendContractError(
                    "integration_not_found", "The requested integration is not registered.",
                    action="Choose an integration returned by integrations.list.",
                    field="request.integration_id",
                )
            credential = credential_entry_contract(integration_id)
            if endpoint == "credentials.entry.schema":
                return credential
            return {"integration": matches[0].as_dict(), "credential_entry": credential}
        if endpoint == "integrations.connection.test":
            from sift.connectors import check_connection

            connection_result = check_connection(
                self._require_cwd(), connection=str(request["connection"]),
            )
            return {
                "backend": connection_result.backend,
                "connection_display": connection_result.connection_display,
                "latency_ms": connection_result.latency_ms,
                "server_version": connection_result.server_version,
                "read_only_enforcement": connection_result.read_only_enforcement,
                "sampled_rows": 0,
            }
        if endpoint == "integrations.catalog.discover":
            from sift.connectors import inspect_database

            catalog_result = inspect_database(
                self._require_cwd(), connection=str(request["connection"]),
                schema=request.get("schema"), object_name=request.get("object_name"),
            )
            return asdict(catalog_result)
        if endpoint == "extractions.database.run":
            from sift.connectors import run_extract

            # Bind the operation to the project selected at submission time.
            # A frontend may switch focus while the database driver is opening;
            # using ``self.cwd`` later could materialize into another project.
            operation_cwd = self._require_cwd()
            operation_id = str(request.get("operation_id") or uuid.uuid4().hex)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", operation_id):
                raise BackendContractError(
                    "invalid_operation_id", "The operation id is invalid.",
                    action="Use a short identifier containing letters, digits, dot, dash, or underscore.",
                    field="request.operation_id",
                )
            token = CancellationToken()
            with self._lock:
                if operation_id in self._operations:
                    raise BackendContractError(
                        "operation_conflict", "An operation with this id is already active.",
                        action="Wait for it to finish or use a different operation id.",
                        field="request.operation_id",
                    )
                self._operations[operation_id] = token
            sequence = 0

            def emit(update: Any) -> None:
                nonlocal sequence
                event = self._event(
                    "extraction.progress", operation_id, sequence,
                    update.stage == "complete", stage=update.stage,
                    rows_materialized=int(update.rows_materialized),
                    bytes_buffered=int(update.bytes_buffered),
                )
                sequence += 1
                if progress is not None:
                    try:
                        progress(event)
                    except Exception:  # noqa: BLE001 - observation must not abort extraction
                        pass

            try:
                kwargs: dict[str, Any] = {}
                if "row_limit" in request:
                    kwargs["row_limit"] = int(request["row_limit"])
                extract_result = run_extract(
                    operation_cwd, connection=str(request["connection"]),
                    sql=str(request["sql"]), dataset_name=str(request["dataset_name"]),
                    cancellation=token, progress=emit, **kwargs,
                )
                return {
                    "operation_id": operation_id,
                    "dataset": extract_result.dataset_path.name,
                    "rows": extract_result.rows,
                    "columns": extract_result.columns,
                    "truncated": extract_result.truncated,
                    "backend": extract_result.backend,
                    "connection_display": extract_result.connection_display,
                    "query_sha256": extract_result.query_sha256,
                    "dataset_sha256": extract_result.dataset_sha256,
                    "canonical_fingerprint": extract_result.canonical_fingerprint,
                }
            finally:
                with self._lock:
                    self._operations.pop(operation_id, None)
        if endpoint == "operations.cancel":
            operation_id = str(request["operation_id"])
            with self._lock:
                cancel_token = self._operations.get(operation_id)
            if cancel_token is None:
                raise BackendContractError(
                    "operation_not_found", "No active operation has this id.",
                    action="Refresh active operations before trying to cancel.",
                    field="request.operation_id",
                )
            cancel_token.cancel()
            event = self._event("operation.cancelled", operation_id, 0, True)
            if progress is not None:
                try:
                    progress(event)
                except Exception:  # noqa: BLE001 - cancellation already took effect
                    pass
            return {"cancelled": True, "event": event}
        if endpoint == "research.plan.evaluate":
            from sift.methodology import evaluate_method

            evaluated = evaluate_method(
                str(request["method_id"]), request["research_specification"],
            )
            event = self._event(
                "research.plan", uuid.uuid4().hex, 0, True,
                valid=bool(evaluated.get("valid")),
                method_id=str(evaluated.get("method_id", request["method_id"])),
                clarifications=list(evaluated.get("clarifications", [])),
            )
            return {"evaluation": evaluated, "event": event}
        if endpoint == "evidence.verify":
            from sift.store import get_store
            from sift.verification import verify_payload

            result_id = str(request["result_id"])
            row = get_store(self._require_cwd()).get(result_id)
            if row is None:
                raise BackendContractError(
                    "evidence_not_found", "The requested evidence is unavailable.",
                    action="Choose an active result from the evidence list.",
                    field="request.result_id",
                )
            verification = verify_payload(row.sanitized_payload) or {
                "status": "not_applicable", "checks": [],
            }
            event = self._event(
                "evidence.verification", uuid.uuid4().hex, 0, True,
                result_id=result_id, verification=verification,
            )
            return {"result_id": result_id, "verification": verification, "event": event}
        if endpoint == "privacy.warning.review":
            from sift.security_assurance import review_pre_provider_disclosure

            review = review_pre_provider_disclosure(
                str(request["text"]),
                attachment_names=request.get("attachment_names", []),
                field_names=request.get("field_names", []),
                organization_sensitive_fields=request.get(
                    "organization_sensitive_fields", [],
                ), enabled=bool(request.get("enabled", True)),
            )
            event = self._event(
                "privacy.warning", uuid.uuid4().hex, 0, True,
                warn=bool(review.get("warn")), findings=list(review.get("findings", [])),
            )
            return {"review": review, "event": event}
        raise BackendContractError(
            "endpoint_not_found", "The requested backend endpoint does not exist.",
            action="Use an endpoint returned by contract.get.",
        )

    def call(
        self, endpoint: str, request: Mapping[str, Any] | None = None, *,
        request_version: str = API_VERSION,
        request_id: str | None = None,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        """Validate, dispatch, and secret-scrub one endpoint invocation."""
        rid = request_id if isinstance(request_id, str) and re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", request_id,
        ) else uuid.uuid4().hex
        request_is_mapping = request is None or isinstance(request, Mapping)
        raw = dict(request or {}) if request_is_mapping else {}
        secrets = _request_secrets(raw)
        try:
            if not request_is_mapping:
                raise BackendContractError(
                    "invalid_request", "The endpoint request must be an object.",
                    action="Send a JSON object matching the endpoint request schema.",
                    field="request",
                )
            spec = ENDPOINTS.get(endpoint)
            if spec is None:
                raise BackendContractError(
                    "endpoint_not_found", "The requested backend endpoint does not exist.",
                    action="Use an endpoint returned by contract.get.",
                )
            migrated = migrate_request(endpoint, raw, request_version)
            validate_schema(migrated, spec.request_schema)
            data = self._handle(endpoint, migrated, progress)
            validate_schema(data, spec.response_data_schema, path="response.data")
            response: dict[str, Any] = {
                "api_version": API_VERSION, "schema_version": API_SCHEMA_VERSION,
                "request_id": rid, "endpoint": endpoint, "ok": True,
                "data": data,
            }
        except BackendContractError as exc:
            response = {
                "api_version": API_VERSION, "schema_version": API_SCHEMA_VERSION,
                "request_id": rid, "endpoint": endpoint, "ok": False,
                "error": exc.as_dict(),
            }
        except IntegrationError as exc:
            response = {
                "api_version": API_VERSION, "schema_version": API_SCHEMA_VERSION,
                "request_id": rid, "endpoint": endpoint, "ok": False,
                "error": {
                    "code": exc.code, "message": str(exc), "action": exc.action,
                    "retryable": exc.retryable,
                },
            }
        except Exception as exc:  # noqa: BLE001 - endpoint errors remain structured
            response = {
                "api_version": API_VERSION, "schema_version": API_SCHEMA_VERSION,
                "request_id": rid, "endpoint": endpoint, "ok": False,
                "error": {
                    "code": "backend_failure",
                    "message": f"The backend operation failed ({type(exc).__name__}).",
                    "action": "Review the request and local qualification report, then try again.",
                    "retryable": False,
                },
            }
        safe = secret_safe_response(response, secrets=secrets)
        try:
            validate_schema(safe, RESPONSE_ENVELOPE_SCHEMA, path="response")
        except BackendContractError:
            safe = {
                "api_version": API_VERSION, "schema_version": API_SCHEMA_VERSION,
                "request_id": rid, "endpoint": endpoint if isinstance(endpoint, str) else "",
                "ok": False, "error": {
                    "code": "response_contract_failure",
                    "message": "The backend could not produce a contract-valid response.",
                    "action": "Run backend qualification and report this endpoint failure.",
                    "retryable": False,
                },
            }
        return safe


__all__ = [
    "API_SCHEMA_VERSION", "API_VERSION", "BACKEND_API_V1_CONTRACT_SHA256",
    "ENDPOINTS", "EVENT_SCHEMAS", "MINIMUM_REQUEST_VERSION",
    "RESPONSE_ENVELOPE_SCHEMA", "SUPPORTED_REQUEST_VERSIONS",
    "BackendApplication", "BackendContractError", "BackendEvent",
    "EndpointSpec", "backend_contract", "backend_contract_sha256",
    "credential_entry_contract", "migrate_request", "secret_safe_response",
    "validate_schema", "view_ready_capabilities",
]
