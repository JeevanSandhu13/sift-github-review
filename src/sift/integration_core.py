"""Provider/source-neutral contracts for every external integration.

This module contains no SDK imports and performs no network I/O.  It is the
small common layer used by model providers, databases, object stores, and
future research services to describe authentication, data flow, lifecycle,
retention/residency, timeouts, cancellation, retries, and safe errors.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable


IntegrationKind = Literal[
    "model", "database", "object_storage", "research_service"
]
IntegrationState = Literal[
    "unconfigured", "needs_credentials", "needs_configuration",
    "blocked_by_policy", "ready", "active", "degraded", "unavailable",
]

# Model turns can legitimately be long (especially against a local model that
# is loading weights), but they must never inherit an SDK's unbounded/default
# policy.  A conversation turn is not safe to replay automatically: the first
# request may have succeeded even when its response was lost.  Providers use
# these values at client construction so retries cannot happen below Sift's
# operation layer.
MODEL_REQUEST_TIMEOUT_SECONDS = 300.0
MODEL_SDK_MAX_RETRIES = 0
AuthenticationKind = Literal[
    "api_key", "password", "certificate", "oauth", "managed_identity",
    "workload_identity", "local_permissions", "none",
]


@dataclass(frozen=True)
class AuthenticationMethod:
    id: str
    kind: AuthenticationKind
    secret_bearing: bool
    storage: str
    description: str


@dataclass(frozen=True)
class DataFlowContract:
    host_reads_raw_data: bool
    generated_code_reads_raw_data: bool
    generated_code_network_access: bool
    remote_receives_prompts: bool
    remote_receives_sanitized_results: bool
    remote_receives_explicit_attachments: bool
    remote_receives_raw_dataset: bool


@dataclass(frozen=True)
class RetentionContract:
    sift_persists_remote_content: bool
    controlled_by: str
    account_setting_verifiable_by_sift: bool
    disclosure: str


@dataclass(frozen=True)
class ResidencyContract:
    mode: Literal["local", "provider_defined", "region_configurable", "unverified"]
    region: str | None
    enforced_by: str
    disclosure: str


@dataclass(frozen=True)
class OperationPolicy:
    timeout_seconds: float
    cancellation_supported: bool
    automatic_retries: int = 0
    retry_condition: Literal["never", "idempotent_transient_only"] = "never"

    def __post_init__(self) -> None:
        if not 0 < self.timeout_seconds <= 86_400:
            raise ValueError("integration timeout must be in (0, 86400]")
        if not 0 <= self.automatic_retries <= 10:
            raise ValueError("automatic retries must be in [0, 10]")
        if self.automatic_retries and self.retry_condition == "never":
            raise ValueError("retry attempts require an explicit safe retry condition")


@dataclass(frozen=True)
class IntegrationContract:
    id: str
    kind: IntegrationKind
    label: str
    maturity: Literal["supported", "preview", "experimental", "internal"]
    authentication: tuple[AuthenticationMethod, ...]
    data_flow: DataFlowContract
    retention: RetentionContract
    residency: ResidencyContract
    capabilities: tuple[str, ...]
    lifecycle: tuple[IntegrationState, ...]
    operations: Mapping[str, OperationPolicy]
    policy_boundary: str
    credential_boundary: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["operations"] = {
            name: asdict(policy) for name, policy in self.operations.items()
        }
        return payload


@dataclass(frozen=True)
class IntegrationReadiness:
    integration_id: str
    state: IntegrationState
    issues: tuple[str, ...] = ()
    diagnostics: Mapping[str, str | int | float | bool | None] = field(
        default_factory=dict
    )

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    def as_dict(self) -> dict[str, Any]:
        return {
            "integration_id": self.integration_id,
            "state": self.state,
            "ready": self.ready,
            "issues": list(self.issues),
            "diagnostics": dict(self.diagnostics),
        }


class CancellationToken:
    """Thread-safe cooperative cancellation shared by host integrations."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation without exposing the mutable event."""
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise IntegrationCancelled()


class IntegrationError(Exception):
    """Typed, researcher-actionable error containing no raw SDK exception."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        integration_id: str = "",
        retryable: bool = False,
        action: str = "Review the integration configuration and try again.",
    ) -> None:
        super().__init__(message)
        self.code = code
        self.integration_id = integration_id
        self.retryable = retryable
        self.action = action

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": str(self),
            "integration_id": self.integration_id,
            "retryable": self.retryable,
            "action": self.action,
        }


class IntegrationCancelled(IntegrationError):
    def __init__(self) -> None:
        super().__init__(
            "cancelled",
            "The integration operation was cancelled.",
            retryable=False,
            action="Start a new operation when ready.",
        )


class IntegrationDeadlineExceeded(IntegrationError):
    def __init__(self, timeout_seconds: float) -> None:
        super().__init__(
            "deadline_exceeded",
            f"The integration operation exceeded its {timeout_seconds:g}-second timeout.",
            retryable=True,
            action="Narrow the operation or increase its bounded timeout.",
        )


class Deadline:
    def __init__(self, timeout_seconds: float) -> None:
        if not 0 < timeout_seconds <= 86_400:
            raise ValueError("deadline must be in (0, 86400]")
        self.timeout_seconds = float(timeout_seconds)
        self._ends_at = time.monotonic() + self.timeout_seconds

    @property
    def remaining(self) -> float:
        return max(0.0, self._ends_at - time.monotonic())

    def check(self, cancellation: CancellationToken | None = None) -> None:
        if cancellation is not None:
            cancellation.raise_if_cancelled()
        if self.remaining <= 0:
            raise IntegrationDeadlineExceeded(self.timeout_seconds)


@dataclass(frozen=True)
class CredentialSpec:
    integration_id: str
    keyring_service: str
    keyring_account: str
    environment_variables: tuple[str, ...] = ()
    allow_environment: bool = True
    prefer_environment: bool = False


@dataclass(frozen=True)
class ResolvedCredential:
    integration_id: str
    method: Literal["keyring", "environment", "managed_identity", "none"]
    secret: str | None = field(default=None, repr=False)

    @property
    def configured(self) -> bool:
        return self.secret is not None or self.method in {"managed_identity", "none"}


def keyring_backend_is_secure(
    backend: Any,
    _seen: set[int] | None = None,
) -> bool:
    """Reject plaintext, null, failed, and unsafe chained keyring backends."""
    if backend is None:
        return False
    seen = _seen if _seen is not None else set()
    identity = id(backend)
    if identity in seen:
        return False
    seen.add(identity)
    backend_name = (
        f"{type(backend).__module__}.{type(backend).__qualname__}"
    ).casefold()
    if any(marker in backend_name for marker in (
        "keyrings.alt", "plaintext", ".null.", ".fail.",
    )):
        return False
    nested = getattr(backend, "backends", None)
    if nested is not None:
        try:
            children = tuple(nested)
        except (TypeError, RuntimeError):
            return False
        return bool(children) and all(
            keyring_backend_is_secure(child, seen) for child in children
        )
    try:
        return float(getattr(backend, "priority", 0)) > 0
    except (TypeError, ValueError, RuntimeError):
        return False


def keyring_module_is_secure(keyring_module: Any) -> bool:
    """Return whether a keyring module is backed by a reviewed secret store."""
    try:
        backend = keyring_module.get_keyring()
    except Exception:  # noqa: BLE001 - backend discovery APIs vary
        return False
    return keyring_backend_is_secure(backend)


def resolve_credential(
    spec: CredentialSpec,
    *,
    keyring_backend: Any | None = None,
    environment: Mapping[str, str] | None = None,
) -> ResolvedCredential:
    """Resolve a secret without logging, serializing, or exposing it in repr."""
    env = os.environ if environment is None else environment

    def from_environment() -> ResolvedCredential | None:
        if not spec.allow_environment:
            return None
        for name in spec.environment_variables:
            value = env.get(name)
            if isinstance(value, str) and value.strip():
                return ResolvedCredential(
                    spec.integration_id, "environment", value.strip()
                )
        return None

    if spec.prefer_environment:
        resolved_env = from_environment()
        if resolved_env is not None:
            return resolved_env
    if keyring_backend is None:
        try:
            import keyring as keyring_backend  # type: ignore[no-redef]
        except ImportError:
            keyring_backend = None
    if keyring_backend is not None:
        try:
            value = keyring_backend.get_password(
                spec.keyring_service, spec.keyring_account
            )
        except Exception:  # noqa: BLE001 - caller gets a deterministic miss
            value = None
        if isinstance(value, str) and value.strip():
            return ResolvedCredential(spec.integration_id, "keyring", value.strip())
    resolved_env = from_environment()
    if resolved_env is not None:
        return resolved_env
    return ResolvedCredential(spec.integration_id, "none", None)


T = TypeVar("T")


def run_with_safe_retries(
    operation: Callable[[], T],
    *,
    policy: OperationPolicy,
    idempotent: bool,
    is_transient: Callable[[Exception], bool],
    cancellation: CancellationToken | None = None,
) -> T:
    """Retry only an explicitly idempotent operation after transient errors."""
    if policy.automatic_retries and not idempotent:
        raise ValueError("automatic retry refused for a non-idempotent operation")
    deadline = Deadline(policy.timeout_seconds)
    attempt = 0
    while True:
        deadline.check(cancellation)
        try:
            return operation()
        except IntegrationCancelled:
            raise
        except Exception as exc:
            if (
                attempt >= policy.automatic_retries
                or policy.retry_condition != "idempotent_transient_only"
                or not is_transient(exc)
            ):
                raise
            attempt += 1


@runtime_checkable
class DataSourceAdapter(Protocol):
    """Source-neutral host interface; raw data never enters model tools."""

    integration_id: str

    def connection_test(
        self, *, cancellation: CancellationToken | None = None
    ) -> IntegrationReadiness: ...

    def discover(
        self,
        *,
        limit: int,
        cancellation: CancellationToken | None = None,
    ) -> Mapping[str, Any]: ...

    def materialize(
        self,
        destination: Any,
        *,
        cancellation: CancellationToken | None = None,
    ) -> Mapping[str, Any]: ...


INTEGRATION_CONFIG_VERSION = 1


def migrate_integration_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize versioned non-secret configuration; reject future schemas."""
    version = payload.get("version", 0)
    if version == 0:
        values = dict(payload)
        values.pop("version", None)
        return {"version": INTEGRATION_CONFIG_VERSION, "integrations": values}
    if version != INTEGRATION_CONFIG_VERSION:
        raise IntegrationError(
            "unsupported_config_version",
            "The integration configuration uses an unsupported version.",
            action="Upgrade Sift or restore a compatible configuration export.",
        )
    integrations = payload.get("integrations")
    if not isinstance(integrations, Mapping):
        raise IntegrationError(
            "invalid_config", "The integration configuration is malformed."
        )
    return {"version": INTEGRATION_CONFIG_VERSION, "integrations": dict(integrations)}


__all__ = [
    "AuthenticationMethod", "CancellationToken", "CredentialSpec",
    "DataFlowContract", "DataSourceAdapter", "Deadline",
    "INTEGRATION_CONFIG_VERSION", "IntegrationCancelled",
    "IntegrationContract", "IntegrationDeadlineExceeded", "IntegrationError",
    "IntegrationReadiness", "MODEL_REQUEST_TIMEOUT_SECONDS",
    "MODEL_SDK_MAX_RETRIES", "OperationPolicy", "ResidencyContract",
    "ResolvedCredential", "RetentionContract", "migrate_integration_config",
    "keyring_backend_is_secure", "keyring_module_is_secure",
    "resolve_credential", "run_with_safe_retries",
]
