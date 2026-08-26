"""Stage 2 gate: every model family shares Sift's hard contract."""

from __future__ import annotations

from pathlib import Path

from sift.integrations import integration_contracts, model_integration
from sift.provider import ProviderSession, open_session
from sift.provider.anthropic import AnthropicSession
from sift.provider.gemini import build_gemini_tools
from sift.provider.openai import build_openai_tools
from sift.provider.openai_compatible import build_chat_completion_tools
from sift.provider.tool_schemas import build_tool_specs
from sift.tools import ALLOWED_TOOL_NAMES

FAMILIES = (
    ("openai", "gpt-5.6-sol"),
    ("anthropic", "claude-sonnet-5[1m]"),
    ("gemini", "gemini-3.7-flash"),
    ("openai_compatible", "openai-compatible-custom"),
)


def test_all_families_implement_the_same_conversation_protocol(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )
    for provider, model in FAMILIES:
        session = open_session(
            provider,
            cwd=tmp_path,
            model=model,
            system_prompt="system",
        )
        assert isinstance(session, ProviderSession)
        assert callable(session.open)
        assert callable(session.send)
        assert callable(session.set_model)
        assert callable(session.set_effort)
        assert callable(session.close)


def test_all_families_expose_exactly_the_canonical_sift_tools() -> None:
    canonical = {spec.name for spec in build_tool_specs()}
    assert {name.rsplit("__", 1)[-1] for name in ALLOWED_TOOL_NAMES} == canonical
    assert {tool["name"] for tool in build_openai_tools()} == canonical
    assert {
        tool["function"]["name"] for tool in build_chat_completion_tools()
    } == canonical
    assert {
        declaration.name
        for declaration in build_gemini_tools().function_declarations
    } == canonical


def test_all_families_share_cancellation_retry_and_privacy_guarantees(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "SIFT_OPENAI_COMPATIBLE_BASE_URL", "https://gateway.example/v1",
    )
    contracts = {contract.id: contract for contract in integration_contracts()}
    for provider, _model in FAMILIES:
        contract = contracts[provider]
        turn = contract.operations["conversation_turn"]
        assert turn.timeout_seconds == 300.0
        assert turn.cancellation_supported is True
        assert turn.automatic_retries == 0
        trust = model_integration(provider)
        assert trust.raw_dataset_access is False
        assert any("network" in guarantee.casefold() for guarantee in trust.guarantees)


def test_anthropic_hosted_capabilities_are_deny_by_default(tmp_path: Path) -> None:
    session = AnthropicSession(
        tmp_path, "claude-sonnet-5[1m]", "system",
    )
    options = session._build_options()
    assert options.strict_mcp_config is True
    assert options.setting_sources == []
    assert options.tools == []
    assert options.permission_mode == "default"
    assert options.can_use_tool is not None
