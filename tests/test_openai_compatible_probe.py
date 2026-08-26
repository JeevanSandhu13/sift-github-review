from __future__ import annotations

import json

import httpx
import pytest

from sift.integration_core import CancellationToken, IntegrationCancelled
from sift.provider.openai_compatible_probe import (
    MAX_PROBE_RESPONSE_BYTES,
    probe_openai_compatible_endpoint,
)


class _CountingBody(httpx.SyncByteStream):
    def __init__(self, chunks: int) -> None:
        self.chunks = chunks
        self.yielded = 0

    def __iter__(self):
        for _ in range(self.chunks):
            self.yielded += 1
            yield b"x" * 65_536


@pytest.mark.parametrize(
    "server_header,family",
    [
        ("Ollama", "ollama"),
        ("LM Studio", "lm_studio"),
        ("vLLM", "vllm"),
        ("llama.cpp", "llama_cpp"),
        ("gateway", "generic"),
    ],
)
def test_deep_probe_discovers_and_certifies_observed_protocol(
    monkeypatch, server_header: str, family: str,
) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        headers = {"server": server_header}
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, headers=headers, json={"data": [{"id": "research-model"}]},
            )
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(
                200, headers=headers,
                text='data: {"choices":[]}\n\ndata: [DONE]\n\n',
            )
        if body.get("tools"):
            return httpx.Response(200, headers=headers, json={"choices": [{
                "message": {"tool_calls": [{"id": "c1", "type": "function"}]}
            }]})
        return httpx.Response(200, headers=headers, json={"choices": [{
            "message": {"content": "OK"}
        }]})

    result = probe_openai_compatible_endpoint(
        base_url="http://localhost:11434/v1",
        model="research-model",
        context_window=32768,
        deep=True,
        transport=httpx.MockTransport(handler),
    )
    assert result.reachable is True
    assert result.server_family == family
    assert result.discovered_models == ("research-model",)
    assert result.tool_calling is True
    assert result.streaming is True
    assert result.vision is True
    assert result.certified is (family != "generic")
    assert result.certification_scope == (
        "observed_openai_protocol" if family != "generic" else None
    )
    assert result.context_window_match is None
    assert len(calls) == 4


def test_probe_redacts_api_key_from_transport_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )
    secret = "probe-secret-canary"

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"authorization Bearer {secret}")

    result = probe_openai_compatible_endpoint(
        base_url="https://gateway.example/v1",
        model="research-model",
        context_window=32768,
        api_key=secret,
        transport=httpx.MockTransport(handler),
    )
    assert result.reachable is False
    assert secret not in str(result)


def test_probe_rejects_unsafe_endpoint_before_transport(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )
    result = probe_openai_compatible_endpoint(
        base_url="file:///tmp/model",
        model="x",
        context_window=1,
    )
    assert result.reachable is False
    assert result.issues


def test_probe_propagates_cancellation_before_network(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )
    token = CancellationToken()
    token.cancel()
    with pytest.raises(IntegrationCancelled):
        probe_openai_compatible_endpoint(
            base_url="http://localhost:11434/v1",
            model="x",
            context_window=1,
            cancellation=token,
            transport=httpx.MockTransport(
                lambda request: pytest.fail("cancelled probe performed I/O")
            ),
        )


def test_probe_does_not_claim_vision_for_empty_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, headers={"server": "Ollama"},
                json={"data": [{"id": "research-model"}]},
            )
        body = json.loads(request.content)
        if body.get("stream"):
            return httpx.Response(200, text="data: [DONE]\n\n")
        if body.get("tools"):
            return httpx.Response(200, json={"choices": [{"message": {
                "tool_calls": [{"id": "c1", "type": "function"}]
            }}]})
        return httpx.Response(200, json={"choices": []})

    result = probe_openai_compatible_endpoint(
        base_url="http://localhost:11434/v1",
        model="research-model",
        context_window=32768,
        deep=True,
        transport=httpx.MockTransport(handler),
    )
    assert result.reachable is True
    assert result.vision is False


def test_probe_detects_server_context_configuration_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"server": "vLLM"}, json={
            "data": [{"id": "research-model", "max_model_len": 8192}],
        })

    result = probe_openai_compatible_endpoint(
        base_url="http://localhost:8000/v1",
        model="research-model",
        context_window=32768,
        transport=httpx.MockTransport(handler),
    )
    assert result.discovered_context_window == 8192
    assert result.context_window_match is False
    assert "configured_context_window_mismatch" in result.issues


def test_probe_stops_incremental_body_at_decoded_response_cap(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.enterprise_policy.load_enterprise_policy", lambda: None,
    )
    chunks = MAX_PROBE_RESPONSE_BYTES // 65_536 + 20
    body = _CountingBody(chunks)

    result = probe_openai_compatible_endpoint(
        base_url="http://localhost:11434/v1",
        model="research-model",
        context_window=32768,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, stream=body)
        ),
    )
    assert result.reachable is False
    assert any("safety limit" in issue for issue in result.issues)
    # One chunk beyond the exact cap proves rejection; the unread malicious
    # tail is never buffered.
    assert body.yielded <= (MAX_PROBE_RESPONSE_BYTES // 65_536) + 1
    assert body.yielded < chunks
