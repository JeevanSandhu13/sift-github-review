from __future__ import annotations

import httpx
import pytest

from sift.provider.availability import MAX_CATALOG_BYTES, check_model_availability


class _CountingBody(httpx.SyncByteStream):
    def __init__(self, chunks: int) -> None:
        self.chunks = chunks
        self.yielded = 0

    def __iter__(self):
        for _ in range(self.chunks):
            self.yielded += 1
            yield b"x" * 65_536


@pytest.mark.parametrize(
    "provider,model,payload",
    [
        ("openai", "gpt-5.6-sol", {"data": [{"id": "gpt-5.6-sol"}]}),
        (
            "anthropic", "claude-sonnet-5[1m]",
            {"data": [{"id": "claude-sonnet-5"}]},
        ),
        (
            "gemini", "gemini-3.7-flash",
            {"models": [{"name": "models/gemini-3.7-flash"}]},
        ),
    ],
)
def test_remote_availability_uses_header_credential_and_finds_model(
    monkeypatch, provider: str, model: str, payload: dict,
) -> None:
    secret = f"{provider}-secret-canary"
    monkeypatch.setattr(
        "sift.auth.resolve_provider_credential",
        lambda selected, names: secret,
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert secret not in str(request.url)
        assert any(secret in value for value in request.headers.values())
        return httpx.Response(200, json=payload)

    result = check_model_availability(
        provider, model, transport=httpx.MockTransport(handler),
    )
    assert result.reachable is True
    assert result.available is True
    assert result.issue is None
    assert len(requests) == 1


def test_retired_model_is_checked_using_documented_replacement(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.auth.resolve_provider_credential", lambda provider, names: "secret",
    )
    result = check_model_availability(
        "openai",
        "gpt-5.5",
        transport=httpx.MockTransport(lambda request: httpx.Response(
            200, json={"data": [{"id": "gpt-5.6-sol"}]},
        )),
    )
    assert result.current_model == "gpt-5.6-sol"
    assert result.available is True


def test_availability_failure_redacts_credential(monkeypatch) -> None:
    secret = "availability-secret-canary"
    monkeypatch.setattr(
        "sift.auth.resolve_provider_credential", lambda provider, names: secret,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"authorization Bearer {secret}")

    result = check_model_availability(
        "openai", "gpt-5.6-sol", transport=httpx.MockTransport(handler),
    )
    assert result.available is None
    assert secret not in str(result)


def test_missing_credential_does_not_touch_network(monkeypatch) -> None:
    monkeypatch.setattr(
        "sift.auth.resolve_provider_credential", lambda provider, names: None,
    )
    result = check_model_availability(
        "gemini", "gemini-3.7-flash",
        transport=httpx.MockTransport(
            lambda request: pytest.fail("network should not be used")
        ),
    )
    assert result.issue == "credential_required"
    assert result.available is None


def test_availability_rejects_declared_oversize_before_reading_body(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "sift.auth.resolve_provider_credential", lambda provider, names: "secret",
    )
    body = _CountingBody(100)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": str(MAX_CATALOG_BYTES + 1)},
            stream=body,
        )

    result = check_model_availability(
        "openai", "gpt-5.6-sol", transport=httpx.MockTransport(handler),
    )
    assert result.reachable is False
    assert "safety limit" in (result.issue or "")
    assert body.yielded == 0
