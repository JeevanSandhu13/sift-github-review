"""Sift provider package — multi-provider session interface.

Public surface (importable as ``from sift.provider import ...``):

- ``ProviderSession``: the protocol both Anthropic and OpenAI sessions
  satisfy. Wraps everything a chat turn needs (lifecycle, send,
  set_model) so the rest of Sift doesn't reach into provider SDKs
  directly.
- ``Event``-types (``AssistantText``, ``ToolCall``, …): the
  provider-neutral stream every session yields.
- ``ModelInfo`` + catalog helpers: which models exist, which provider
  owns each one, defaults.
- ``open_session(provider, cwd, model, system_prompt)``: factory that
  returns the right session class for the given provider id, with
  lazy import so missing optional deps (the OpenAI SDK) don't break
  Anthropic-only installs.
- ``detect_auth(provider)``: per-provider auth detection. Returns
  ``"api_key"``, ``"endpoint"``, or ``"unknown"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sift.integration_ids import MODEL_PROVIDER_IDS
from sift.provider.availability import ModelAvailability, check_model_availability
from sift.provider.base import (
    AssistantText,
    AssistantThinking,
    AuthFailure,
    AuthMode,
    Event,
    ProviderSession,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from sift.provider.catalog import (
    ALL_MODELS,
    ANTHROPIC_MODELS,
    AZURE_OPENAI_MODELS,
    BEDROCK_ANTHROPIC_MODELS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    GEMINI_MODELS,
    OPENAI_MODELS,
    PROVIDER_DEFAULTS,
    VERTEX_ANTHROPIC_MODELS,
    VERTEX_GEMINI_MODELS,
    ModelInfo,
    get_model,
    models_for_provider,
    provider_for_model,
)

# Provider IDs the rest of the codebase recognises.
SUPPORTED_PROVIDERS: tuple[str, ...] = MODEL_PROVIDER_IDS


def open_session(
    provider: str,
    cwd: Path,
    model: str,
    system_prompt: str,
    **kwargs: Any,
) -> ProviderSession:
    """Construct (but don't yet ``open()``) a provider session.

    Lazy-imports the per-provider module so an Anthropic-only install
    doesn't trip on a missing ``openai`` dep.
    """
    # Governance is enforced at the provider factory, not only in the UI.
    # CLI callers and future frontends therefore cannot bypass an admin's
    # provider allowlist/local-only requirement.
    from sift import enterprise_policy

    ent = enterprise_policy.load_enterprise_policy()
    if not enterprise_policy.model_provider_allowed(provider, ent):
        requirement = (
            "; this deployment requires a verified localhost model endpoint"
            if ent is not None and ent.require_local_model
            else ""
        )
        raise PermissionError(
            f"model provider {provider!r} is blocked by enterprise policy{requirement}"
        )

    if provider == "anthropic":
        from sift.provider.anthropic import AnthropicSession

        return AnthropicSession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
    if provider == "openai":
        # Lazy import so the openai SDK only loads when actually used.
        from sift.provider.openai import OpenAISession

        return OpenAISession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
    if provider == "openai_compatible":
        # Lazy import for the same reason as the openai branch above
        # — this module also depends on the openai SDK (it's the
        # client library used to reach the target endpoint's Chat
        # Completions API).
        from sift.provider.openai_compatible import OpenAICompatibleSession

        return OpenAICompatibleSession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
    if provider == "gemini":
        # Lazy import so the google-genai SDK only loads when Gemini
        # is actually selected — same rationale as the openai branch.
        from sift.provider.gemini import GeminiSession

        return GeminiSession(
            cwd=cwd,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
    if provider == "azure_openai":
        from sift.provider.azure_openai import AzureOpenAISession

        return AzureOpenAISession(cwd, model, system_prompt, **kwargs)
    if provider == "vertex_gemini":
        from sift.provider.vertex_gemini import VertexGeminiSession

        return VertexGeminiSession(cwd, model, system_prompt, **kwargs)
    if provider == "bedrock_anthropic":
        from sift.provider.bedrock_anthropic import BedrockAnthropicSession

        return BedrockAnthropicSession(cwd, model, system_prompt, **kwargs)
    if provider == "vertex_anthropic":
        from sift.provider.vertex_anthropic import VertexAnthropicSession

        return VertexAnthropicSession(cwd, model, system_prompt, **kwargs)
    raise ValueError(
        f"unknown provider: {provider!r}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
    )


def detect_auth(provider: str) -> AuthMode:
    """Per-provider auth detection.

    Remote providers use researcher-supplied API keys from environment or OS
    keyring. OpenAI-compatible local endpoints may be auth-free.
    """
    if provider == "anthropic":
        from sift.provider.anthropic import detect_auth as _a

        return _a()
    if provider == "openai":
        from sift.provider.openai import detect_auth as _o

        return _o()
    if provider == "openai_compatible":
        from sift.provider.openai_compatible import detect_auth as _oc

        return _oc()
    if provider == "gemini":
        from sift.provider.gemini import detect_auth as _g

        return _g()
    if provider == "azure_openai":
        from sift.provider.azure_openai import detect_auth as _az

        return _az()
    if provider == "vertex_gemini":
        from sift.provider.vertex_gemini import detect_auth as _vg

        return _vg()
    if provider == "bedrock_anthropic":
        from sift.provider.bedrock_anthropic import detect_auth as _ba

        return _ba()
    if provider == "vertex_anthropic":
        from sift.provider.vertex_anthropic import detect_auth as _va

        return _va()
    raise ValueError(f"unknown provider: {provider!r}")


def _provider_readiness(provider: str, ent: Any) -> dict[str, Any]:
    """Build readiness using an already-loaded enterprise policy."""
    from sift import enterprise_policy

    allowed = enterprise_policy.model_provider_allowed(provider, ent)
    issues: list[str] = []
    if not allowed:
        issues.append("blocked_by_enterprise_policy")

    if provider == "openai_compatible":
        from sift.provider.openai_compatible import configuration_issues

        issues.extend(configuration_issues())
    elif provider in {
        "azure_openai",
        "vertex_gemini",
        "bedrock_anthropic",
        "vertex_anthropic",
    }:
        try:
            module_name = {
                "azure_openai": "azure_openai",
                "vertex_gemini": "vertex_gemini",
                "bedrock_anthropic": "bedrock_anthropic",
                "vertex_anthropic": "vertex_anthropic",
            }[provider]
            import importlib

            importlib.import_module(f"sift.provider.{module_name}").load_config()
        except (ImportError, ValueError, PermissionError):
            # Readiness intentionally exposes only a stable issue code, never
            # an endpoint, project, account, or SDK exception string.
            issues.append("managed_deployment_configuration_required")
        required_module = {
            "azure_openai": "azure.identity",
            "vertex_gemini": "google.genai",
            "bedrock_anthropic": "boto3",
            "vertex_anthropic": "anthropic",
        }[provider]
        try:
            import importlib.util

            sdk_present = importlib.util.find_spec(required_module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            sdk_present = False
        if not sdk_present:
            issues.append("managed_provider_sdk_required")

    auth_mode = detect_auth(provider)
    if provider != "openai_compatible" and auth_mode == "unknown":
        issues.append("credential_required")

    if not allowed:
        state = "blocked_by_policy"
    elif any(item.endswith("_required") for item in issues):
        state = "needs_configuration"
    elif auth_mode == "unknown":
        state = "needs_credentials"
    else:
        state = "ready"
    return {
        "provider": provider,
        "ready": state == "ready",
        "state": state,
        "auth_mode": auth_mode,
        "issues": issues,
    }


def provider_readiness(provider: str) -> dict[str, Any]:
    """Return a secret-free readiness state for one model integration.

    Authentication and configuration are different states.  This matters for
    OpenAI-compatible targets, where an optional saved key does not make the
    integration usable until both an endpoint URL and a model name exist.
    Enterprise policy is evaluated here as well so every future frontend gets
    the same answer.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"unknown provider: {provider!r}. "
            f"Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )
    from sift import enterprise_policy

    return _provider_readiness(provider, enterprise_policy.load_enterprise_policy())


def all_provider_readiness() -> dict[str, dict[str, Any]]:
    """Return every readiness state while loading enterprise policy once."""
    from sift import enterprise_policy

    ent = enterprise_policy.load_enterprise_policy()
    return {
        provider: _provider_readiness(provider, ent) for provider in SUPPORTED_PROVIDERS
    }


__all__ = [
    "ALL_MODELS",
    "ANTHROPIC_MODELS",
    "AZURE_OPENAI_MODELS",
    "BEDROCK_ANTHROPIC_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_PROVIDER",
    "GEMINI_MODELS",
    "OPENAI_MODELS",
    "PROVIDER_DEFAULTS",
    "SUPPORTED_PROVIDERS",
    "VERTEX_ANTHROPIC_MODELS",
    "VERTEX_GEMINI_MODELS",
    "AssistantText",
    "AssistantThinking",
    "AuthFailure",
    "AuthMode",
    "Event",
    "ModelAvailability",
    "ModelInfo",
    "ProviderSession",
    "ToolCall",
    "ToolCallResult",
    "TurnDone",
    "TurnError",
    "all_provider_readiness",
    "check_model_availability",
    "detect_auth",
    "get_model",
    "models_for_provider",
    "open_session",
    "provider_for_model",
    "provider_readiness",
]
