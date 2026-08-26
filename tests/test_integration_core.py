from __future__ import annotations

from dataclasses import asdict

import pytest

from sift.integration_core import (
    CancellationToken,
    CredentialSpec,
    IntegrationCancelled,
    IntegrationError,
    OperationPolicy,
    migrate_integration_config,
    resolve_credential,
    run_with_safe_retries,
)


class FakeKeyring:
    def __init__(self, value: str | None = None, *, fails: bool = False) -> None:
        self.value = value
        self.fails = fails

    def get_password(self, service: str, account: str) -> str | None:
        assert service == "sift-test"
        assert account == "provider"
        if self.fails:
            raise RuntimeError("backend said secret-canary")
        return self.value


def _spec(*, allow_environment: bool = True) -> CredentialSpec:
    return CredentialSpec(
        "provider",
        "sift-test",
        "provider",
        ("SIFT_TEST_API_KEY",),
        allow_environment,
    )


def test_credential_resolution_prefers_keyring_and_secret_repr_is_safe() -> None:
    resolved = resolve_credential(
        _spec(),
        keyring_backend=FakeKeyring("keyring-secret"),
        environment={"SIFT_TEST_API_KEY": "environment-secret"},
    )
    assert resolved.method == "keyring"
    assert resolved.secret == "keyring-secret"
    assert "keyring-secret" not in repr(resolved)
    # dataclasses.asdict is intentionally not used by the public integration
    # manifests for credential results; this assertion pins the repr boundary.
    assert asdict(resolved)["secret"] == "keyring-secret"


def test_credential_resolution_environment_can_be_disabled() -> None:
    resolved = resolve_credential(
        _spec(allow_environment=False),
        keyring_backend=FakeKeyring(),
        environment={"SIFT_TEST_API_KEY": "environment-secret"},
    )
    assert resolved.method == "none"
    assert resolved.secret is None


def test_credential_resolution_can_honor_process_override_order() -> None:
    spec = CredentialSpec(
        "provider",
        "sift-test",
        "provider",
        ("SIFT_TEST_API_KEY",),
        prefer_environment=True,
    )
    resolved = resolve_credential(
        spec,
        keyring_backend=FakeKeyring("keyring-secret"),
        environment={"SIFT_TEST_API_KEY": "environment-secret"},
    )
    assert resolved.method == "environment"
    assert resolved.secret == "environment-secret"


def test_keyring_error_does_not_surface_backend_message_or_secret() -> None:
    resolved = resolve_credential(
        _spec(), keyring_backend=FakeKeyring(fails=True), environment={}
    )
    assert resolved.method == "none"
    assert "secret-canary" not in repr(resolved)


def test_cancellation_is_typed_and_actionable() -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(IntegrationCancelled) as raised:
        token.raise_if_cancelled()
    payload = raised.value.as_dict()
    assert payload["code"] == "cancelled"
    assert payload["retryable"] is False
    assert payload["action"]


def test_retry_refuses_non_idempotent_operation() -> None:
    policy = OperationPolicy(
        timeout_seconds=10,
        automatic_retries=1,
        retry_condition="idempotent_transient_only",
        cancellation_supported=True,
    )
    with pytest.raises(ValueError, match="non-idempotent"):
        run_with_safe_retries(
            lambda: None,
            policy=policy,
            idempotent=False,
            is_transient=lambda _exc: True,
        )


def test_retry_only_repeats_transient_idempotent_failures() -> None:
    calls = 0

    def operation() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError
        return "ok"

    result = run_with_safe_retries(
        operation,
        policy=OperationPolicy(
            timeout_seconds=10,
            automatic_retries=1,
            retry_condition="idempotent_transient_only",
            cancellation_supported=True,
        ),
        idempotent=True,
        is_transient=lambda exc: isinstance(exc, TimeoutError),
    )
    assert result == "ok"
    assert calls == 2


def test_operation_policy_is_bounded_and_consistent() -> None:
    with pytest.raises(ValueError):
        OperationPolicy(timeout_seconds=0, cancellation_supported=False)
    with pytest.raises(ValueError):
        OperationPolicy(
            timeout_seconds=10,
            cancellation_supported=False,
            automatic_retries=1,
        )


def test_config_migration_accepts_legacy_and_rejects_future() -> None:
    assert migrate_integration_config({"openai": {"model": "x"}}) == {
        "version": 1,
        "integrations": {"openai": {"model": "x"}},
    }
    assert migrate_integration_config({"version": 1, "integrations": {}}) == {
        "version": 1,
        "integrations": {},
    }
    with pytest.raises(IntegrationError) as raised:
        migrate_integration_config({"version": 99, "integrations": {}})
    assert raised.value.code == "unsupported_config_version"
