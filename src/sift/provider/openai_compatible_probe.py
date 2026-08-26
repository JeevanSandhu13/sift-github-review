"""Bounded capability probe for OpenAI-compatible model endpoints.

The probe is explicit and sends synthetic content only. It never includes a
research prompt, dataset value, schema, attachment, or persisted chat state.
"""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from typing import Any

from sift.integration_core import CancellationToken, IntegrationCancelled
from sift.provider.error_safety import provider_error_message
from sift.provider.response_limits import (
    read_bounded_json_object,
    read_bounded_response,
)

MAX_PROBE_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class CompatibleEndpointProbe:
    reachable: bool
    server_family: str
    discovered_models: tuple[str, ...]
    catalog_truncated: bool
    configured_model_present: bool | None
    tool_calling: bool | None
    vision: bool | None
    streaming: bool | None
    context_window: int
    discovered_context_window: int | None
    context_window_match: bool | None
    certification_scope: str | None
    certified: bool
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _server_family(headers: Any) -> str:
    signal = " ".join(
        str(headers.get(name, ""))
        for name in ("server", "x-powered-by", "x-model-server")
    ).casefold()
    for needle, family in (
        ("ollama", "ollama"),
        ("lm studio", "lm_studio"),
        ("lmstudio", "lm_studio"),
        ("vllm", "vllm"),
        ("llama.cpp", "llama_cpp"),
        ("llama-cpp", "llama_cpp"),
    ):
        if needle in signal:
            return family
    return "generic"


def _bounded_json(response: Any) -> dict[str, Any]:
    return read_bounded_json_object(
        response,
        max_bytes=MAX_PROBE_RESPONSE_BYTES,
        label="endpoint probe response",
    )


def _tool_call_present(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    calls = message.get("tool_calls") if isinstance(message, dict) else None
    return isinstance(calls, list) and bool(calls)


def _completion_present(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    return isinstance(message, dict) and isinstance(message.get("content"), str)


def probe_openai_compatible_endpoint(
    *,
    base_url: str,
    model: str,
    context_window: int,
    api_key: str | None = None,
    deep: bool = False,
    timeout_seconds: float = 10.0,
    cancellation: CancellationToken | None = None,
    transport: Any | None = None,
) -> CompatibleEndpointProbe:
    """Discover models and optionally test tools, vision, and streaming."""
    from sift import enterprise_policy
    from sift.provider.openai_compatible import validate_base_url

    issues = list(validate_base_url(base_url))
    if issues:
        return CompatibleEndpointProbe(
            False, "unknown", (), False, None, None, None, None,
            context_window, None, None, None, False, tuple(issues),
        )
    policy = enterprise_policy.load_enterprise_policy()
    if not enterprise_policy.integration_endpoint_allowed(base_url, policy):
        return CompatibleEndpointProbe(
            False, "unknown", (), False, None, None, None, None,
            context_window, None, None, None, False,
            ("blocked_by_enterprise_policy",),
        )
    if cancellation is not None:
        cancellation.raise_if_cancelled()

    import httpx

    headers = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    endpoint = base_url.rstrip("/")
    family = "generic"
    models: list[str] = []
    tool_calling: bool | None = None
    vision: bool | None = None
    streaming: bool | None = None
    discovered_context: int | None = None
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        ) as client:
            with client.stream(
                "GET", f"{endpoint}/models", headers=headers,
            ) as response:
                family = _server_family(response.headers)
                response.raise_for_status()
                payload = _bounded_json(response)
            rows = payload.get("data")
            if not isinstance(rows, list):
                raise TypeError("model catalog did not contain a data list")
            for row in rows[:501]:
                value = row.get("id") if isinstance(row, dict) else None
                if isinstance(value, str) and 0 < len(value) <= 300:
                    models.append(value)
                    if value == model:
                        for key in (
                            "context_length", "max_model_len",
                            "max_context_length", "context_window",
                        ):
                            candidate = row.get(key)
                            if (
                                isinstance(candidate, int)
                                and not isinstance(candidate, bool)
                                and 0 < candidate <= 100_000_000
                            ):
                                discovered_context = candidate
                                break
            catalog_truncated = len(models) > 500 or len(rows) > 500
            models = models[:500]

            if deep:
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                common = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Call sift_probe once."}],
                    "tools": [{
                        "type": "function",
                        "function": {
                            "name": "sift_probe",
                            "description": "Synthetic compatibility probe.",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }],
                    "tool_choice": "required",
                    "max_tokens": 16,
                }
                with client.stream(
                    "POST", f"{endpoint}/chat/completions",
                    headers=headers, json=common,
                ) as tool_response:
                    tool_response.raise_for_status()
                    tool_calling = _tool_call_present(_bounded_json(tool_response))

                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                with client.stream(
                    "POST", f"{endpoint}/chat/completions",
                    headers=headers, json={**common, "stream": True},
                ) as stream_response:
                    stream_response.raise_for_status()
                    stream_body = read_bounded_response(
                        stream_response,
                        max_bytes=MAX_PROBE_RESPONSE_BYTES,
                        label="stream probe response",
                    )
                lines = [
                    line.strip()
                    for line in stream_body.decode("utf-8").splitlines()
                ]
                chunks = [line[5:].strip() for line in lines if line.startswith("data:")]
                streaming = bool(chunks) and chunks[-1] == "[DONE]"
                for chunk in chunks[:-1]:
                    if chunk:
                        json.loads(chunk)

                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                # Valid 1x1 transparent PNG.  An incomplete signature/header
                # can make a vision-capable endpoint fail for the wrong reason.
                png = base64.b64encode(base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                    "QVQImWNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
                )).decode("ascii")
                with client.stream(
                    "POST", f"{endpoint}/chat/completions",
                    headers=headers, json={
                        "model": model,
                        "messages": [{"role": "user", "content": [
                            {"type": "text", "text": "Reply OK."},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{png}"
                            }},
                        ]}],
                        "max_tokens": 4,
                    },
                ) as vision_response:
                    vision_response.raise_for_status()
                    vision = _completion_present(_bounded_json(vision_response))
        present = model in models
        if not present:
            issues.append("configured_model_not_discovered")
        if deep and not tool_calling:
            issues.append("required_tool_calling_not_observed")
        if deep and not streaming:
            issues.append("compatible_streaming_not_observed")
        context_match = (
            discovered_context == context_window
            if discovered_context is not None
            else None
        )
        if context_match is False:
            issues.append("configured_context_window_mismatch")
        elif discovered_context is None:
            issues.append("context_window_not_reported_by_endpoint")
        known_family = family in {"ollama", "lm_studio", "vllm", "llama_cpp"}
        certified = bool(
            deep and known_family and present and tool_calling and streaming
            and context_match is not False
        )
        return CompatibleEndpointProbe(
            True, family, tuple(models), catalog_truncated, present,
            tool_calling, vision, streaming, context_window,
            discovered_context, context_match,
            ("observed_openai_protocol" if certified else None), certified,
            tuple(issues),
        )
    except IntegrationCancelled:
        raise
    except Exception as exc:  # noqa: BLE001 - SDK/transport errors vary
        safe = provider_error_message(exc, secrets=(api_key,))
        return CompatibleEndpointProbe(
            False, family, tuple(models), False,
            (model in models if models else None), tool_calling, vision,
            streaming, context_window, discovered_context,
            (
                discovered_context == context_window
                if discovered_context is not None else None
            ),
            None, False,
            (*issues, f"probe_failed:{safe}"),
        )


__all__ = ["CompatibleEndpointProbe", "probe_openai_compatible_endpoint"]
