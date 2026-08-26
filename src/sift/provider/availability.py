"""Explicit, credential-safe model availability diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sift.integration_core import CancellationToken, IntegrationCancelled
from sift.provider.catalog import current_model_id, get_model
from sift.provider.error_safety import provider_error_message
from sift.provider.response_limits import read_bounded_json_object

MAX_CATALOG_BYTES = 2 * 1024 * 1024
MAX_CATALOG_MODELS = 1_000


@dataclass(frozen=True)
class ModelAvailability:
    provider: str
    requested_model: str
    current_model: str
    lifecycle: str
    reachable: bool
    available: bool | None
    discovered_count: int
    catalog_truncated: bool
    issue: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bounded_payload(response: Any) -> dict[str, Any]:
    return read_bounded_json_object(
        response,
        max_bytes=MAX_CATALOG_BYTES,
        label="provider model catalog",
    )


def _remote_catalog_request(provider: str, key: str) -> tuple[str, dict[str, str]]:
    if provider == "openai":
        return "https://api.openai.com/v1/models", {
            "authorization": f"Bearer {key}",
        }
    if provider == "anthropic":
        return "https://api.anthropic.com/v1/models?limit=1000", {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
    if provider == "gemini":
        # Keep the key in a header, never a URL or diagnostic string.
        return (
            "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
            {"x-goog-api-key": key},
        )
    raise ValueError(f"unsupported remote provider: {provider!r}")


def _ids_from_payload(provider: str, payload: dict[str, Any]) -> tuple[list[str], bool]:
    key = "models" if provider == "gemini" else "data"
    rows = payload.get(key)
    if not isinstance(rows, list):
        raise TypeError("provider model catalog did not contain a model list")
    ids: list[str] = []
    for row in rows[:MAX_CATALOG_MODELS]:
        if not isinstance(row, dict):
            continue
        value = row.get("name" if provider == "gemini" else "id")
        if isinstance(value, str):
            if provider == "gemini" and value.startswith("models/"):
                value = value.removeprefix("models/")
            if 0 < len(value) <= 300:
                ids.append(value)
    has_more = bool(payload.get("has_more") or payload.get("nextPageToken"))
    return ids, len(rows) > MAX_CATALOG_MODELS or has_more


def _catalog_wire_id(provider: str, model_id: str) -> str:
    # ``[1m]`` is the Claude Agent SDK/CLI context selector. Anthropic's
    # Models API catalogs the underlying model id without that local suffix.
    if provider == "anthropic" and model_id.endswith("[1m]"):
        return model_id[:-4]
    return model_id


def check_model_availability(
    provider: str,
    model_id: str,
    *,
    timeout_seconds: float = 10.0,
    cancellation: CancellationToken | None = None,
    transport: Any | None = None,
) -> ModelAvailability:
    """Check one catalog model without sending prompts or research data.

    This is an explicit setup diagnostic, not an automatic background call.
    Only the configured credential and requested provider model catalog cross
    the network boundary.
    """
    try:
        replacement = current_model_id(model_id)
    except KeyError:
        raise ValueError(f"unknown model id: {model_id!r}") from None
    info = get_model(replacement)
    if info.provider != provider:
        raise ValueError(
            f"model {model_id!r} belongs to {info.provider!r}, not {provider!r}"
        )
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    if provider == "openai_compatible":
        import os

        from sift import auth
        from sift.provider.openai_compatible import (
            ENV_API_KEY,
            ENV_BASE_URL,
            ENV_MODEL,
            resolve_context_window,
        )
        from sift.provider.openai_compatible_probe import (
            probe_openai_compatible_endpoint,
        )

        target_model = os.environ.get(ENV_MODEL) or ""
        probe = probe_openai_compatible_endpoint(
            base_url=os.environ.get(ENV_BASE_URL) or "",
            model=target_model,
            context_window=resolve_context_window(),
            api_key=auth.resolve_provider_credential(provider, (ENV_API_KEY,)),
            timeout_seconds=timeout_seconds,
            cancellation=cancellation,
            transport=transport,
        )
        return ModelAvailability(
            provider=provider,
            requested_model=model_id,
            current_model=replacement,
            lifecycle=info.lifecycle,
            reachable=probe.reachable,
            available=probe.configured_model_present,
            discovered_count=len(probe.discovered_models),
            catalog_truncated=probe.catalog_truncated,
            issue=";".join(probe.issues) or None,
        )

    managed_modules = {
        "azure_openai": "azure_openai",
        "vertex_gemini": "vertex_gemini",
        "bedrock_anthropic": "bedrock_anthropic",
        "vertex_anthropic": "vertex_anthropic",
    }
    if provider in managed_modules:
        # Managed catalogs are project/region/account scoped and require the
        # cloud identity chain.  The non-network availability API validates
        # configuration honestly but does not claim reachability without a
        # cloud control/data-plane request.  Provider-specific connection tests
        # can perform that explicit call from the setup flow.
        import importlib

        try:
            importlib.import_module(
                f"sift.provider.{managed_modules[provider]}"
            ).load_config()
        except Exception:
            return ModelAvailability(
                provider, model_id, replacement, info.lifecycle,
                False, None, 0, False, "managed_deployment_configuration_required",
            )
        return ModelAvailability(
            provider, model_id, replacement, info.lifecycle,
            False, None, 0, False, "managed_deployment_configured_not_probed",
        )

    from sift import auth
    env_names = {
        "openai": ("OPENAI_API_KEY",),
        "anthropic": ("ANTHROPIC_API_KEY",),
        "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    }
    key = auth.resolve_provider_credential(provider, env_names[provider])
    if not key:
        return ModelAvailability(
            provider, model_id, replacement, info.lifecycle,
            False, None, 0, False, "credential_required",
        )
    import httpx
    url, headers = _remote_catalog_request(provider, key)
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            transport=transport,
        ) as client:
            with client.stream(
                "GET", url, headers={"accept": "application/json", **headers},
            ) as response:
                response.raise_for_status()
                ids, truncated = _ids_from_payload(
                    provider, _bounded_payload(response)
                )
        wire_id = _catalog_wire_id(provider, replacement)
        return ModelAvailability(
            provider, model_id, replacement, info.lifecycle,
            True, wire_id in ids, len(ids), truncated,
            None if wire_id in ids else "model_not_available_to_credential",
        )
    except IntegrationCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - HTTP errors vary
        safe = provider_error_message(exc, secrets=(key,))
        return ModelAvailability(
            provider, model_id, replacement, info.lifecycle,
            False, None, 0, False, f"availability_check_failed:{safe}",
        )


__all__ = ["ModelAvailability", "check_model_availability"]
