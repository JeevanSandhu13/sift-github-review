from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from sift import enterprise_policy
from sift.provider import ProviderSession, open_session
from sift.provider.azure_openai import AzureOpenAISession
from sift.provider.bedrock_anthropic import (
    BedrockAnthropicConfig,
    BedrockAnthropicSession,
)
from sift.provider.enterprise_common import (
    PRIVACY_MANIFESTS,
    ManagedDeploymentConfigError,
    translate_managed_error,
)
from sift.provider.gemini import build_gemini_tools
from sift.provider.managed_claude import build_claude_tools, verify_claude_lockdown
from sift.provider.tool_schemas import build_tool_specs
from sift.provider.vertex_anthropic import (
    VertexAnthropicConfig,
    VertexAnthropicSession,
)
from sift.provider.vertex_gemini import VertexGeminiConfig, VertexGeminiSession

MANAGED_ENV = (
    "SIFT_AZURE_OPENAI_ENDPOINT",
    "SIFT_AZURE_OPENAI_DEPLOYMENT",
    "SIFT_AZURE_OPENAI_REGION",
    "SIFT_AZURE_OPENAI_RESOURCE",
    "SIFT_AZURE_OPENAI_AUTH",
    "AZURE_OPENAI_API_KEY",
    "SIFT_VERTEX_GEMINI_PROJECT",
    "SIFT_VERTEX_GEMINI_LOCATION",
    "SIFT_VERTEX_GEMINI_MODEL",
    "SIFT_BEDROCK_REGION",
    "SIFT_BEDROCK_ANTHROPIC_MODEL",
    "SIFT_AWS_ACCOUNT_ID",
    "SIFT_AWS_PROFILE",
    "SIFT_VERTEX_ANTHROPIC_PROJECT",
    "SIFT_VERTEX_ANTHROPIC_LOCATION",
    "SIFT_VERTEX_ANTHROPIC_MODEL",
)


@pytest.fixture(autouse=True)
def _managed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in MANAGED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(enterprise_policy, "load_enterprise_policy", lambda: None)


def _azure_env(monkeypatch: pytest.MonkeyPatch, *, auth: str = "api_key") -> None:
    monkeypatch.setenv("SIFT_AZURE_OPENAI_ENDPOINT", "https://lab.openai.azure.com")
    monkeypatch.setenv("SIFT_AZURE_OPENAI_DEPLOYMENT", "research-gpt")
    monkeypatch.setenv("SIFT_AZURE_OPENAI_REGION", "canadacentral")
    monkeypatch.setenv("SIFT_AZURE_OPENAI_RESOURCE", "lab")
    monkeypatch.setenv("SIFT_AZURE_OPENAI_AUTH", auth)


def _vertex_gemini_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIFT_VERTEX_GEMINI_PROJECT", "research-prod")
    monkeypatch.setenv("SIFT_VERTEX_GEMINI_LOCATION", "northamerica-northeast1")
    monkeypatch.setenv("SIFT_VERTEX_GEMINI_MODEL", "gemini-3.7-flash")


def _bedrock_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIFT_BEDROCK_REGION", "ca-central-1")
    monkeypatch.setenv("SIFT_BEDROCK_ANTHROPIC_MODEL", "ca.anthropic.claude-sonnet-5")
    monkeypatch.setenv("SIFT_AWS_ACCOUNT_ID", "123456789012")


def _vertex_anthropic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIFT_VERTEX_ANTHROPIC_PROJECT", "research-prod")
    monkeypatch.setenv("SIFT_VERTEX_ANTHROPIC_LOCATION", "us")
    monkeypatch.setenv("SIFT_VERTEX_ANTHROPIC_MODEL", "claude-sonnet-5")


def test_managed_providers_are_distinct_protocol_implementations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    _azure_env(monkeypatch)
    _vertex_gemini_env(monkeypatch)
    _bedrock_env(monkeypatch)
    _vertex_anthropic_env(monkeypatch)
    rows = (
        ("azure_openai", "azure-openai-deployment", AzureOpenAISession),
        ("vertex_gemini", "vertex-gemini-model", VertexGeminiSession),
        ("bedrock_anthropic", "bedrock-anthropic-model", BedrockAnthropicSession),
        ("vertex_anthropic", "vertex-anthropic-model", VertexAnthropicSession),
    )
    for provider, model, expected in rows:
        session = open_session(provider, tmp_path, model, "system")
        assert type(session) is expected
        assert isinstance(session, ProviderSession)


def test_azure_rejects_untrusted_endpoint_and_invalid_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift.provider.azure_openai import load_config

    _azure_env(monkeypatch)
    monkeypatch.setenv("SIFT_AZURE_OPENAI_ENDPOINT", "https://evil.example")
    with pytest.raises(ManagedDeploymentConfigError, match="approved cloud domain"):
        load_config()
    _azure_env(monkeypatch)
    monkeypatch.setenv("SIFT_AZURE_OPENAI_AUTH", "openai_key")
    with pytest.raises(ManagedDeploymentConfigError, match="managed_identity.*api_key"):
        load_config()


def test_direct_provider_keys_are_never_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift import auth
    from sift.provider import azure_openai

    _azure_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-direct-must-not-be-used")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-" + "ant-direct-must-not-be-used")
    monkeypatch.setenv("GEMINI_API_KEY", "AI" + "za-direct-must-not-be-used")
    monkeypatch.setattr(auth, "get_credential", lambda _provider: None)
    assert azure_openai.detect_auth() == "unknown"
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-resource-key")
    assert azure_openai.detect_auth() == "api_key"


def test_azure_managed_identity_uses_default_credential_and_ai_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider.azure_openai import _MANAGED_IDENTITY_SENTINEL

    _azure_env(monkeypatch, auth="managed_identity")
    calls: dict[str, Any] = {}

    class DefaultCredential:
        pass

    def token_provider(credential: Any, scope: str) -> Any:
        calls.update(credential=credential, scope=scope)
        return lambda: "entra-token"

    azure_module = types.ModuleType("azure")
    identity_module = types.ModuleType("azure.identity")
    identity_module.DefaultAzureCredential = DefaultCredential
    identity_module.get_bearer_token_provider = token_provider
    azure_module.identity = identity_module
    monkeypatch.setitem(sys.modules, "azure", azure_module)
    monkeypatch.setitem(sys.modules, "azure.identity", identity_module)

    session = AzureOpenAISession(tmp_path, "azure-openai-deployment", "system")
    session._config = __import__(
        "sift.provider.azure_openai", fromlist=["load_config"]
    ).load_config()
    options = session._client_options(_MANAGED_IDENTITY_SENTINEL)
    assert isinstance(calls["credential"], DefaultCredential)
    assert calls["scope"] == "https://ai.azure.com/.default"
    assert callable(options["api_key"])


def test_enterprise_region_project_and_account_allowlists_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sift.provider.bedrock_anthropic import load_config as load_bedrock
    from sift.provider.vertex_gemini import load_config as load_vertex

    _vertex_gemini_env(monkeypatch)
    policy = enterprise_policy.EnterprisePolicy(
        allowed_model_providers=frozenset({"vertex_gemini"}),
        allowed_regions=frozenset({"us-central1"}),
        allowed_cloud_projects=frozenset({"approved-project"}),
    )
    monkeypatch.setattr(enterprise_policy, "load_enterprise_policy", lambda: policy)
    with pytest.raises(ManagedDeploymentConfigError, match="region"):
        load_vertex()

    _bedrock_env(monkeypatch)
    policy = enterprise_policy.EnterprisePolicy(
        allowed_model_providers=frozenset({"bedrock_anthropic"}),
        allowed_regions=frozenset({"ca-central-1"}),
        allowed_cloud_accounts=frozenset({"000000000000"}),
    )
    monkeypatch.setattr(enterprise_policy, "load_enterprise_policy", lambda: policy)
    with pytest.raises(ManagedDeploymentConfigError, match="account"):
        load_bedrock()


def test_privacy_manifests_are_distinct_and_honest() -> None:
    assert set(PRIVACY_MANIFESTS) == {
        "azure_openai", "vertex_gemini", "bedrock_anthropic", "vertex_anthropic",
    }
    assert len({manifest.processor for manifest in PRIVACY_MANIFESTS.values()}) == 4
    for manifest in PRIVACY_MANIFESTS.values():
        payload = manifest.as_dict()
        assert payload["authentication_boundary"]
        assert payload["processing_location"]
        assert payload["retention"]
        assert payload["not_verified_by_sift"]


def test_managed_claude_tool_surface_is_exact_and_fail_closed() -> None:
    tools = build_claude_tools()
    canonical = {spec.name for spec in build_tool_specs()}
    assert {tool["name"] for tool in tools} == canonical
    verify_claude_lockdown(tools)
    with pytest.raises(RuntimeError, match="lockdown"):
        verify_claude_lockdown(tools + [{
            "name": "computer", "description": "hosted", "input_schema": {},
        }])


class _OpenAIItem:
    type = "message"
    content: ClassVar[list[Any]] = [
        SimpleNamespace(type="output_text", text="azure ok")
    ]

    def model_dump(self) -> dict[str, Any]:
        return {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "azure ok"}],
        }


class _OpenAIResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            output=[_OpenAIItem()],
            usage=SimpleNamespace(input_tokens=12, output_tokens=3),
            status="completed",
        )


class _OpenAIClient:
    instances: ClassVar[list[_OpenAIClient]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.responses = _OpenAIResponses()
        self.instances.append(self)

    async def close(self) -> None:
        return None


def test_azure_conversation_uses_deployment_store_false_and_local_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import openai

    _azure_env(monkeypatch)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-only-key")
    monkeypatch.setattr(openai, "AsyncOpenAI", _OpenAIClient)
    _OpenAIClient.instances.clear()
    session = AzureOpenAISession(tmp_path, "azure-openai-deployment", "system")

    async def run() -> None:
        assert [type(event).__name__ async for event in session.send("first")][-1] == "TurnDone"
        assert [type(event).__name__ async for event in session.send("second")][-1] == "TurnDone"

    asyncio.run(run())
    client = _OpenAIClient.instances[-1]
    assert client.kwargs["api_key"] == "azure-only-key"
    assert client.kwargs["base_url"] == "https://lab.openai.azure.com/openai/v1/"
    assert [call["model"] for call in client.responses.calls] == ["research-gpt"] * 2
    assert all(call["store"] is False for call in client.responses.calls)
    assert len(client.responses.calls[1]["input"]) > len(client.responses.calls[0]["input"])


class _BedrockClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "bedrock ok"}]}},
            "usage": {"inputTokens": 20, "outputTokens": 4},
            "stopReason": "end_turn",
        }


def test_bedrock_conversation_uses_converse_region_model_and_sift_tools(
    tmp_path: Path,
) -> None:
    session = BedrockAnthropicSession(tmp_path, "bedrock-anthropic-model", "system")
    client = _BedrockClient()
    session._client = client
    session._config = BedrockAnthropicConfig(
        "ca-central-1", "ca.anthropic.claude-sonnet-5", "123456789012", None,
    )

    async def run() -> list[str]:
        return [type(event).__name__ async for event in session.send("question")]

    assert asyncio.run(run())[-1] == "TurnDone"
    call = client.calls[0]
    assert call["modelId"] == "ca.anthropic.claude-sonnet-5"
    assert {row["toolSpec"]["name"] for row in call["toolConfig"]["tools"]} == {
        spec.name for spec in build_tool_specs()
    }
    assert "guardrailConfig" not in call


class _VertexMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(
                type="text",
                text="vertex claude ok",
                model_dump=lambda exclude_none=True: {
                    "type": "text", "text": "vertex claude ok",
                },
            )],
            usage=SimpleNamespace(input_tokens=21, output_tokens=5),
            stop_reason="end_turn",
        )


def test_vertex_claude_conversation_is_local_and_uses_partner_model(
    tmp_path: Path,
) -> None:
    session = VertexAnthropicSession(tmp_path, "vertex-anthropic-model", "system")
    messages = _VertexMessages()
    session._client = SimpleNamespace(messages=messages)
    session._config = VertexAnthropicConfig("research-prod", "us", "claude-sonnet-5")

    async def run() -> None:
        assert [type(e).__name__ async for e in session.send("one")][-1] == "TurnDone"
        assert [type(e).__name__ async for e in session.send("two")][-1] == "TurnDone"

    asyncio.run(run())
    assert [call["model"] for call in messages.calls] == ["claude-sonnet-5"] * 2
    assert len(messages.calls[1]["messages"]) > len(messages.calls[0]["messages"])
    assert all({tool["name"] for tool in call["tools"]} == {
        spec.name for spec in build_tool_specs()
    } for call in messages.calls)


class _VertexGeminiChat:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._curated_history: list[Any] = []
        self._comprehensive_history: list[Any] = []

    async def send_message(self, parts: Any, config: Any = None) -> Any:
        self.calls.append({"parts": parts, "config": config})
        response = SimpleNamespace(
            candidates=[SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(
                    text="vertex gemini ok", function_call=None, thought=False,
                )]),
                finish_reason=None,
                finish_message=None,
            )],
            usage_metadata=SimpleNamespace(
                prompt_token_count=18,
                candidates_token_count=4,
                thoughts_token_count=0,
                cached_content_token_count=0,
            ),
            prompt_feedback=None,
        )
        self._curated_history.extend([parts, response])
        self._comprehensive_history.extend([parts, response])
        return response


def test_vertex_gemini_conversation_uses_vertex_session_without_direct_key(
    tmp_path: Path,
) -> None:
    session = VertexGeminiSession(tmp_path, "vertex-gemini-model", "system")
    chat = _VertexGeminiChat()
    session._client = object()
    session._chat = chat
    session._tool = build_gemini_tools()
    session._config = VertexGeminiConfig(
        "research-prod", "northamerica-northeast1", "gemini-3.7-flash",
    )

    async def run() -> list[str]:
        return [type(event).__name__ async for event in session.send("question")]

    assert asyncio.run(run())[-1] == "TurnDone"
    assert len(chat.calls) == 1
    config = chat.calls[0]["config"]
    assert len(config.tools) == 1
    assert config.google_search is None if hasattr(config, "google_search") else True


def test_vertex_gemini_open_failure_closes_created_transports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A post-client setup failure must not orphan either SDK transport."""
    _vertex_gemini_env(monkeypatch)
    calls: list[str] = []

    class Chats:
        def create(self, **_kwargs: Any) -> Any:
            raise RuntimeError("chat setup failed")

    class Aio:
        chats = Chats()

        async def aclose(self) -> None:
            calls.append("async")

    class Client:
        aio = Aio()

        def close(self) -> None:
            calls.append("sync")

    import google.genai as genai

    monkeypatch.setattr(genai, "Client", lambda **_kwargs: Client())
    session = VertexGeminiSession(
        tmp_path, "vertex-gemini-model", "system",
    )

    asyncio.run(session.open())

    assert session._client is None
    assert session._chat is None
    assert calls == ["async", "sync"]
    assert "chat setup failed" in (session._configuration_error or "")


@pytest.mark.parametrize(
    "error, expected_type, expected_text",
    [
        (RuntimeError("AccessDeniedException"), "AuthFailure", "IAM/RBAC"),
        (RuntimeError("ThrottlingException quota"), "TurnError", "quota"),
        (RuntimeError("model NotFound"), "TurnError", "could not find"),
        (TimeoutError("timed out"), "TurnError", "not retried"),
    ],
)
def test_managed_error_translation_is_actionable_and_secret_safe(
    error: Exception, expected_type: str, expected_text: str,
) -> None:
    event = translate_managed_error(
        error,
        provider_label="Cloud",
        resource_label="deployment",
        secrets=("secret-value",),
    )
    assert type(event).__name__ == expected_type
    rendered = getattr(event, "reason", None) or getattr(event, "message", "")
    assert expected_text in rendered
    assert "secret-value" not in rendered
