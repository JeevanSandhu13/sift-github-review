from __future__ import annotations

import pytest

from sift import enterprise_policy
from sift.integration_ids import MODEL_PROVIDER_IDS
from sift.provider import (
    SUPPORTED_PROVIDERS,
    all_provider_readiness,
    provider_readiness,
)
from sift.provider.catalog import PROVIDER_DEFAULTS
from sift.provider.openai_compatible import ENV_BASE_URL, ENV_MODEL
from sift.ui import SiftBridge


def test_all_provider_registries_share_one_canonical_order():
    assert SUPPORTED_PROVIDERS == MODEL_PROVIDER_IDS
    assert tuple(PROVIDER_DEFAULTS) == MODEL_PROVIDER_IDS


def test_openai_compatible_readiness_distinguishes_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(enterprise_policy, "load_enterprise_policy", lambda: None)

    missing = provider_readiness("openai_compatible")
    assert missing["state"] == "needs_configuration"
    assert missing["issues"] == ["base_url_required", "model_name_required"]

    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:11434/v1")
    assert provider_readiness("openai_compatible")["issues"] == ["model_name_required"]

    monkeypatch.setenv(ENV_MODEL, "llama3.1")
    ready = provider_readiness("openai_compatible")
    assert ready["ready"] is True
    assert ready["issues"] == []


def test_readiness_reports_policy_block_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        enterprise_policy,
        "load_enterprise_policy",
        lambda: enterprise_policy.EnterprisePolicy(
            allowed_model_providers=frozenset({"gemini"})
        ),
    )

    state = provider_readiness("openai")

    assert state["ready"] is False
    assert state["state"] == "blocked_by_policy"
    assert "blocked_by_enterprise_policy" in state["issues"]


def test_readiness_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        provider_readiness("not-a-provider")


def test_all_readiness_loads_enterprise_policy_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"n": 0}

    def load():
        calls["n"] += 1

    monkeypatch.setattr(enterprise_policy, "load_enterprise_policy", load)
    monkeypatch.setattr("sift.provider.detect_auth", lambda _provider: "api_key")

    states = all_provider_readiness()

    assert tuple(states) == MODEL_PROVIDER_IDS
    assert calls["n"] == 1


def test_auth_payload_only_allows_policy_permitted_ready_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift import auth

    monkeypatch.setattr(
        enterprise_policy,
        "load_enterprise_policy",
        lambda: enterprise_policy.EnterprisePolicy(
            allowed_model_providers=frozenset({"gemini"})
        ),
    )
    monkeypatch.setattr("sift.provider.detect_auth", lambda _provider: "api_key")
    monkeypatch.setattr(
        auth,
        "credential_state",
        lambda _provider, force_refresh=False: auth.AUTH_STATE_CONFIGURED,
    )
    monkeypatch.setattr(auth, "has_credential", lambda _provider: True)

    bridge = SiftBridge.__new__(SiftBridge)
    bridge._default_provider = "anthropic"
    bridge._active_runner = lambda: None

    payload = bridge._auth_status_payload()

    assert payload["any_authed"] is True
    assert payload["providers"]["gemini"]["configured"] is True
    assert payload["providers"]["openai"]["configured"] is False
    assert payload["providers"]["openai"]["authenticated"] is True
    assert payload["providers"]["openai"]["readiness"] == "blocked_by_policy"
