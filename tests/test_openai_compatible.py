"""OpenAI-compatible endpoint provider.

Covers ``provider/openai_compatible.py`` -- the Chat-Completions-API
session that lets Sift talk to local models (Ollama, vLLM, LM Studio)
and third-party gateways (OpenRouter et al.), as distinct from
``provider/openai.py`` which is built on OpenAI's proprietary
Responses API that essentially no third-party server implements.

Test strategy mirrors ``test_openai_lockdown.py``: monkeypatch
``openai.AsyncOpenAI`` with a fake client that records every
``chat.completions.create()`` call and returns scripted Chat-
Completions-shaped responses, so the real ``send()`` tool loop runs
against a stand-in rather than a real network call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sift.provider.base import (
    AssistantText,
    AuthFailure,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from sift.provider.openai_compatible import (
    DEFAULT_CONTEXT_WINDOW,
    ENV_ALLOW_INSECURE_REMOTE,
    ENV_API_KEY,
    ENV_BASE_URL,
    ENV_CONTEXT_WINDOW,
    ENV_MODEL,
    MAX_COMPATIBLE_OUTPUT_TOKENS,
    MAX_COMPATIBLE_RESPONSE_BYTES,
    MAX_COMPATIBLE_TOOL_ARGUMENT_BYTES,
    OpenAICompatibleSession,
    _bounded_chat_completion_create,
    _verify_lockdown,
    build_chat_completion_tools,
    configuration_issues,
    detect_auth,
    resolve_context_window,
)
from sift.provider.response_limits import ProviderResponseTooLarge
from sift.provider.tool_schemas import build_tool_specs

# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with none of this provider's env vars set,
    so a stray value from the real environment (or a previous test)
    can't leak in."""
    for var in (
        ENV_BASE_URL,
        ENV_MODEL,
        ENV_API_KEY,
        ENV_CONTEXT_WINDOW,
        ENV_ALLOW_INSECURE_REMOTE,
    ):
        monkeypatch.delenv(var, raising=False)


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.type = "function"
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(
        self,
        content: str | None,
        tool_calls: list[_FakeToolCall] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _FakeUsage:
    def __init__(self, prompt_tokens: int = 12, completion_tokens: int = 4) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeChoice:
    def __init__(
        self,
        message: _FakeMessage,
        finish_reason: str | None = None,
    ) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeChatCompletion:
    def __init__(
        self,
        message: _FakeMessage,
        usage: _FakeUsage | None = None,
        *,
        finish_reason: str | None = None,
    ) -> None:
        self.choices = [_FakeChoice(message, finish_reason)]
        self.usage = usage or _FakeUsage()


class _FakeCompletionsAPI:
    """Records every ``create()`` call and returns responses from a
    scripted queue (or a fixed default if none is scripted)."""

    def __init__(self, scripted: list[_FakeChatCompletion] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._scripted = list(scripted or [])
        self._default = _FakeChatCompletion(_FakeMessage("ok"))

    async def create(self, **kwargs: Any) -> _FakeChatCompletion:
        # Snapshot ``messages`` rather than keeping the caller's live
        # list reference — a real HTTP request body is serialized at
        # call time, so later in-place mutations to
        # ``self._messages`` (the provider appends to the SAME list
        # across rounds) must not retroactively change what an
        # earlier recorded call looks like.
        snapshot = dict(kwargs)
        if "messages" in snapshot:
            snapshot["messages"] = [dict(m) for m in snapshot["messages"]]
        self.calls.append(snapshot)
        if self._scripted:
            return self._scripted.pop(0)
        return self._default


class _FakeChatAPI:
    def __init__(self, scripted: list[_FakeChatCompletion] | None = None) -> None:
        self.completions = _FakeCompletionsAPI(scripted)


class _FakeAsyncOpenAI:
    """Stand-in for ``openai.AsyncOpenAI``, constructed the way the
    real SDK is: ``AsyncOpenAI(api_key=..., base_url=...)``."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **options: Any,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.options = options
        self.chat = _FakeChatAPI(_SCRIPTED_RESPONSES.get(id(self), None))

    async def close(self) -> None:
        return None


# Scripted-response registry keyed by fake-client instance id, set by
# the test before constructing the session (the session builds its
# own client instance inside open(), so tests can't hand one in
# directly -- see _install_fake_client below).
_SCRIPTED_RESPONSES: dict[int, list[_FakeChatCompletion]] = {}


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    scripted: list[_FakeChatCompletion] | None = None,
) -> list[_FakeAsyncOpenAI]:
    """Patch ``openai.AsyncOpenAI`` so the next client constructed
    inside ``OpenAICompatibleSession.open()`` is a fake that returns
    ``scripted`` responses in order. Returns a list that gets the
    constructed fake client appended to it, so the test can reach in
    and inspect ``.chat.completions.calls`` afterward."""
    constructed: list[_FakeAsyncOpenAI] = []

    def _factory(
        api_key: str | None = None, base_url: str | None = None, **options: Any
    ) -> _FakeAsyncOpenAI:
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _FakeChatAPI(list(scripted or []))
        constructed.append(client)
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)
    return constructed


def _configure_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    monkeypatch.setenv(
        ENV_BASE_URL, overrides.get("base_url", "http://localhost:11434/v1")
    )
    monkeypatch.setenv(ENV_MODEL, overrides.get("model", "llama3.1"))
    if "api_key" in overrides:
        monkeypatch.setenv(ENV_API_KEY, overrides["api_key"])


def _run(coro_iter) -> list[Any]:
    async def _collect():
        return [e async for e in coro_iter]

    return asyncio.run(_collect())


# ---------------------------------------------------------------------------
# Tool schema shape (Chat Completions, nested under "function")
# ---------------------------------------------------------------------------


def test_chat_completion_tools_match_canonical_specs():
    tools = build_chat_completion_tools()
    assert len(tools) == len(build_tool_specs())
    assert all(t.get("type") == "function" for t in tools)
    names = {t["function"]["name"] for t in tools}
    expected = {s.name for s in build_tool_specs()}
    assert names == expected


def test_lockdown_rejects_missing_or_duplicate_function() -> None:
    tools = build_chat_completion_tools()
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(tools[:-1])
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(tools + [dict(tools[0])])


def test_chat_completion_tool_shape_is_nested_not_flat():
    """Distinguishes this from the Responses-API shape
    (as_openai_tool): name/description/parameters live under
    ``function``, not top-level."""
    tools = build_chat_completion_tools()
    t = tools[0]
    assert "name" not in t  # not the flat Responses shape
    assert set(t["function"]) >= {"name", "description", "parameters"}


def test_lockdown_rejects_non_function_type():
    tools = build_chat_completion_tools()
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(tools + [{"type": "web_search"}])


def test_lockdown_rejects_unknown_function_name():
    bad = [{"type": "function", "function": {"name": "not_a_real_tool"}}]
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(bad)


def test_lockdown_passes_the_real_tool_list():
    _verify_lockdown(build_chat_completion_tools())  # must not raise


# ---------------------------------------------------------------------------
# Configuration / auth detection
# ---------------------------------------------------------------------------


def test_detect_auth_unknown_with_nothing_configured():
    assert detect_auth() == "unknown"


def test_detect_auth_requires_base_url_and_model(monkeypatch):
    """A no-key local server is ready only once both settings exist."""
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:11434/v1")
    assert detect_auth() == "unknown"
    monkeypatch.setenv(ENV_MODEL, "llama3.1")
    assert detect_auth() == "endpoint"

    monkeypatch.setenv(ENV_API_KEY, "local-optional-key")
    assert detect_auth() == "api_key"


def test_detect_auth_is_not_configured_by_api_key_alone(monkeypatch):
    monkeypatch.setenv(ENV_API_KEY, "sk-or-something")
    assert detect_auth() == "unknown"


def test_resolve_context_window_default():
    assert resolve_context_window() == DEFAULT_CONTEXT_WINDOW


def test_resolve_context_window_from_env(monkeypatch):
    monkeypatch.setenv(ENV_CONTEXT_WINDOW, "128000")
    assert resolve_context_window() == 128_000


def test_resolve_context_window_ignores_garbage(monkeypatch):
    monkeypatch.setenv(ENV_CONTEXT_WINDOW, "not-a-number")
    assert resolve_context_window() == DEFAULT_CONTEXT_WINDOW


def test_resolve_context_window_ignores_non_positive(monkeypatch):
    monkeypatch.setenv(ENV_CONTEXT_WINDOW, "-5")
    assert resolve_context_window() == DEFAULT_CONTEXT_WINDOW


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.com/v1",
        "not-a-url",
        "https://user:password@example.com/v1",
        "https://example.com/v1?token=secret",
        "https://example.com/v1#fragment",
    ],
)
def test_configuration_rejects_unsafe_or_ambiguous_base_urls(
    monkeypatch: pytest.MonkeyPatch, base_url: str,
) -> None:
    monkeypatch.setenv(ENV_BASE_URL, base_url)
    monkeypatch.setenv(ENV_MODEL, "research-model")
    assert configuration_issues()
    assert detect_auth() == "unknown"


def test_remote_plaintext_http_requires_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_BASE_URL, "http://models.example.org/v1")
    monkeypatch.setenv(ENV_MODEL, "research-model")
    assert configuration_issues() == [
        "insecure_remote_http_requires_explicit_opt_in",
    ]

    monkeypatch.setenv(ENV_ALLOW_INSECURE_REMOTE, "1")
    assert configuration_issues() == []


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
        "https://models.example.org/v1",
    ],
)
def test_loopback_http_and_remote_https_are_safe_by_default(
    monkeypatch: pytest.MonkeyPatch, base_url: str,
) -> None:
    monkeypatch.setenv(ENV_BASE_URL, base_url)
    monkeypatch.setenv(ENV_MODEL, "research-model")
    assert configuration_issues() == []


# ---------------------------------------------------------------------------
# send(): missing config -> AuthFailure, never an exception
# ---------------------------------------------------------------------------


def test_send_without_config_yields_auth_failure(tmp_path: Path):
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hello"))
    assert len(events) == 1
    assert isinstance(events[0], AuthFailure)
    assert ENV_BASE_URL in events[0].reason
    assert ENV_MODEL in events[0].reason


def test_send_with_only_base_url_still_reports_missing_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:11434/v1")
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hello"))
    assert isinstance(events[0], AuthFailure)
    assert ENV_MODEL in events[0].reason
    assert ENV_BASE_URL not in events[0].reason


# ---------------------------------------------------------------------------
# send(): text-only round trip
# ---------------------------------------------------------------------------


def test_send_simple_text_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_env(monkeypatch)
    constructed = _install_fake_client(
        monkeypatch,
        scripted=[_FakeChatCompletion(_FakeMessage("Hello researcher"))],
    )
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="you are sift",
    )
    events = _run(sess.send("hi"))

    assert any(
        isinstance(e, AssistantText) and e.text == "Hello researcher" for e in events
    )
    done = [e for e in events if isinstance(e, TurnDone)]
    assert len(done) == 1
    assert done[0].input_tokens == 12
    assert done[0].output_tokens == 4
    assert done[0].post_turn_tokens == 16

    client = constructed[0]
    from sift.integration_core import (
        MODEL_REQUEST_TIMEOUT_SECONDS,
        MODEL_SDK_MAX_RETRIES,
    )

    assert client.options == {
        "timeout": MODEL_REQUEST_TIMEOUT_SECONDS,
        "max_retries": MODEL_SDK_MAX_RETRIES,
    }
    assert len(client.chat.completions.calls) == 1
    call = client.chat.completions.calls[0]
    assert call["model"] == "llama3.1"  # the REAL configured model, not the catalog id
    # System prompt is the first message; user message follows.
    assert call["messages"][0] == {"role": "system", "content": "you are sift"}
    assert call["messages"][1] == {"role": "user", "content": "hi"}
    assert call["tools"] == build_chat_completion_tools()
    assert call["max_tokens"] == MAX_COMPATIBLE_OUTPUT_TOKENS


def test_oversized_decoded_response_fails_closed_and_rolls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    canary = "endpoint-private-canary"
    _install_fake_client(monkeypatch, scripted=[
        _FakeChatCompletion(
            _FakeMessage("x" * (MAX_COMPATIBLE_RESPONSE_BYTES + 1) + canary)
        )
    ])
    sess = OpenAICompatibleSession(
        cwd=tmp_path, model="openai-compatible-custom", system_prompt="sys",
    )
    events = _run(sess.send("sensitive question"))
    assert isinstance(events[-1], TurnError)
    assert "2 MB" in events[-1].message
    assert canary not in events[-1].message
    assert sess._messages == [{"role": "system", "content": "sys"}]


def test_oversized_tool_arguments_never_reach_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    called = False

    async def must_not_run(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    from sift.provider import openai_compatible as mod

    monkeypatch.setitem(mod.HANDLERS, "get_schema", must_not_run)
    _install_fake_client(monkeypatch, scripted=[_FakeChatCompletion(_FakeMessage(
        None,
        [_FakeToolCall(
            "call_oversize", "get_schema",
            "x" * (MAX_COMPATIBLE_TOOL_ARGUMENT_BYTES + 1),
        )],
    ))])
    sess = OpenAICompatibleSession(
        cwd=tmp_path, model="openai-compatible-custom", system_prompt="sys",
    )
    events = _run(sess.send("go"))
    assert isinstance(events[-1], TurnError)
    assert "512 KB" in events[-1].message
    assert called is False
    assert sess._messages == [{"role": "system", "content": "sys"}]


def test_history_ceiling_rejects_before_sdk_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    from sift.provider import openai_compatible as mod

    monkeypatch.setattr(mod, "MAX_COMPATIBLE_HISTORY_BYTES", 64)
    constructed = _install_fake_client(monkeypatch)
    sess = OpenAICompatibleSession(
        cwd=tmp_path, model="openai-compatible-custom", system_prompt="system",
    )
    events = _run(sess.send("sensitive" * 20))
    assert isinstance(events[-1], TurnError)
    assert "history limit" in events[-1].message
    assert constructed[0].chat.completions.calls == []
    assert sess._messages == [{"role": "system", "content": "system"}]


def test_raw_sdk_response_is_incrementally_capped_before_json_decode() -> None:
    chunks = MAX_COMPATIBLE_RESPONSE_BYTES // 65_536 + 20

    class RawResponse:
        headers: dict[str, str] = {}

        def __init__(self) -> None:
            self.yielded = 0

        async def iter_bytes(self, chunk_size: int):
            assert chunk_size == 65_536
            for _ in range(chunks):
                self.yielded += 1
                yield b"x" * 65_536

    raw = RawResponse()

    class Context:
        async def __aenter__(self):
            return raw

        async def __aexit__(self, *exc: object) -> None:
            return None

    class Streaming:
        def create(self, **kwargs: Any) -> Context:
            return Context()

    class API:
        with_streaming_response = Streaming()

        async def create(self, **kwargs: Any) -> Any:
            pytest.fail("bounded raw-response path was bypassed")

    with pytest.raises(ProviderResponseTooLarge, match="2 MB"):
        asyncio.run(_bounded_chat_completion_create(API(), model="m"))
    assert raw.yielded <= (MAX_COMPATIBLE_RESPONSE_BYTES // 65_536) + 1
    assert raw.yielded < chunks


def test_raw_sdk_content_length_rejects_without_reading_body() -> None:
    class RawResponse:
        headers = {"content-length": str(MAX_COMPATIBLE_RESPONSE_BYTES + 1)}
        yielded = 0

        async def iter_bytes(self, chunk_size: int):
            self.yielded += 1
            yield b"should-not-be-read"

    raw = RawResponse()

    class Context:
        async def __aenter__(self):
            return raw

        async def __aexit__(self, *exc: object) -> None:
            return None

    class Streaming:
        def create(self, **kwargs: Any) -> Context:
            return Context()

    class API:
        with_streaming_response = Streaming()

    with pytest.raises(ProviderResponseTooLarge, match="2 MB"):
        asyncio.run(_bounded_chat_completion_create(API(), model="m"))
    assert raw.yielded == 0


def test_session_consumes_bounded_raw_sdk_json_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    body = json.dumps({
        "choices": [{
            "message": {"content": "bounded raw response", "tool_calls": []},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4},
    }).encode()
    calls: list[dict[str, Any]] = []

    class RawResponse:
        headers = {"content-length": str(len(body))}

        async def iter_bytes(self, chunk_size: int):
            yield body

    class Context:
        async def __aenter__(self):
            return RawResponse()

        async def __aexit__(self, *exc: object) -> None:
            return None

    class Streaming:
        def create(self, **kwargs: Any) -> Context:
            calls.append(kwargs)
            return Context()

    completions = SimpleNamespace(with_streaming_response=Streaming())
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
        close=lambda: None,
    )
    sess = OpenAICompatibleSession(
        cwd=tmp_path, model="openai-compatible-custom", system_prompt="sys",
    )
    sess._client = client
    sess._resolved_model = "research-model"
    sess._messages = [{"role": "system", "content": "sys"}]
    sess._tools = build_chat_completion_tools()

    events = _run(sess.send("hello"))
    assert any(
        isinstance(event, AssistantText)
        and event.text == "bounded raw response"
        for event in events
    )
    assert isinstance(events[-1], TurnDone)
    assert events[-1].post_turn_tokens == 7
    assert calls[0]["max_tokens"] == MAX_COMPATIBLE_OUTPUT_TOKENS


def test_two_turns_resend_the_complete_local_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    constructed = _install_fake_client(monkeypatch, scripted=[
        _FakeChatCompletion(_FakeMessage("first answer")),
        _FakeChatCompletion(_FakeMessage("second answer")),
    ])
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="system",
    )
    assert isinstance(_run(sess.send("first question"))[-1], TurnDone)
    assert isinstance(_run(sess.send("second question"))[-1], TurnDone)
    calls = constructed[0].chat.completions.calls
    assert len(calls) == 2
    assert calls[1]["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_incomplete_finish_reason_is_error_and_rolls_back_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    finish_reason: str,
) -> None:
    _configure_env(monkeypatch)
    _install_fake_client(
        monkeypatch,
        scripted=[
            _FakeChatCompletion(
                _FakeMessage("partial answer"),
                finish_reason=finish_reason,
            )
        ],
    )
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("sensitive research question"))

    assert any(isinstance(event, AssistantText) for event in events)
    assert isinstance(events[-1], TurnError)
    assert finish_reason in events[-1].message
    assert not any(isinstance(event, TurnDone) for event in events)
    assert sess._messages == [{"role": "system", "content": "sys"}]


def test_tool_finish_reason_without_calls_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)
    _install_fake_client(
        monkeypatch,
        scripted=[
            _FakeChatCompletion(
                _FakeMessage(None),
                finish_reason="tool_calls",
            )
        ],
    )
    sess = OpenAICompatibleSession(
        cwd=tmp_path, model="openai-compatible-custom", system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert isinstance(events[-1], TurnError)
    assert "without returning" in events[-1].message
    assert sess._messages == [{"role": "system", "content": "sys"}]


def test_missing_and_duplicate_tool_call_ids_are_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_env(monkeypatch)

    async def fake_handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ok"}]}

    from sift.provider import openai_compatible as mod

    monkeypatch.setitem(mod.HANDLERS, "get_schema", fake_handler)
    tool_calls = [
        _FakeToolCall("", "get_schema", "{}"),
        _FakeToolCall("duplicate", "get_schema", "{}"),
        _FakeToolCall("duplicate", "get_schema", "{}"),
    ]
    _install_fake_client(
        monkeypatch,
        scripted=[
            _FakeChatCompletion(
                _FakeMessage(None, tool_calls),
                finish_reason="tool_calls",
            ),
            _FakeChatCompletion(_FakeMessage("done"), finish_reason="stop"),
        ],
    )
    sess = OpenAICompatibleSession(
        cwd=tmp_path, model="openai-compatible-custom", system_prompt="sys",
    )
    events = _run(sess.send("go"))
    calls = [event for event in events if isinstance(event, ToolCall)]
    results = [event for event in events if isinstance(event, ToolCallResult)]
    assert len({event.call_id for event in calls}) == 3
    assert [event.call_id for event in results] == [
        event.call_id for event in calls
    ]
    assert isinstance(events[-1], TurnDone)


def test_send_uses_configured_base_url_and_api_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(
        monkeypatch,
        base_url="http://localhost:8000/v1",
        model="Qwen/Qwen2.5-32B",
        api_key="my-secret",
    )
    constructed = _install_fake_client(monkeypatch)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    _run(sess.send("hi"))
    client = constructed[0]
    assert client.base_url == "http://localhost:8000/v1"
    assert client.api_key == "my-secret"


def test_send_defaults_api_key_placeholder_when_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """No key configured (the common local-server case) must not
    crash client construction -- a harmless placeholder is used."""
    _configure_env(monkeypatch)
    constructed = _install_fake_client(monkeypatch)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    _run(sess.send("hi"))
    assert constructed[0].api_key  # some non-empty placeholder


# ---------------------------------------------------------------------------
# send(): tool-call round trip
# ---------------------------------------------------------------------------


def test_send_dispatches_tool_call_and_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)

    async def fake_handler(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "content": [
                {"type": "text", "text": json.dumps({"status": "ok", "echo": args})}
            ]
        }

    from sift.provider import openai_compatible as mod

    monkeypatch.setitem(mod.HANDLERS, "get_schema", fake_handler)

    tool_call = _FakeToolCall("call_1", "get_schema", json.dumps({"dataset": "x.csv"}))
    scripted = [
        _FakeChatCompletion(_FakeMessage(None, [tool_call])),
        _FakeChatCompletion(_FakeMessage("Here's what I found.")),
    ]
    constructed = _install_fake_client(monkeypatch, scripted=scripted)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("what columns does x.csv have?"))

    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(calls) == 1
    assert calls[0].name == "get_schema"
    assert calls[0].input == {"dataset": "x.csv"}
    assert calls[0].call_id == "call_1"
    assert len(results) == 1
    assert results[0].is_error is False

    done = [e for e in events if isinstance(e, TurnDone)]
    assert len(done) == 1

    # Two rounds: one that produced the tool call, one with the final answer.
    client = constructed[0]
    assert len(client.chat.completions.calls) == 2
    second_call_messages = client.chat.completions.calls[1]["messages"]
    # The assistant's tool_calls message and the tool-role reply must
    # both be present, in that order, before the second request.
    roles = [m["role"] for m in second_call_messages]
    assert roles[-2:] == ["assistant", "tool"]
    assert second_call_messages[-1]["tool_call_id"] == "call_1"


def test_send_unknown_tool_name_reports_error_without_crashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)
    tool_call = _FakeToolCall("call_1", "not_a_real_tool", "{}")
    scripted = [
        _FakeChatCompletion(_FakeMessage(None, [tool_call])),
        _FakeChatCompletion(_FakeMessage("ok")),
    ]
    _install_fake_client(monkeypatch, scripted=scripted)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert results[0].is_error is True
    assert "unknown tool" in results[0].text


def test_send_malformed_tool_arguments_reports_parse_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)
    tool_call = _FakeToolCall("call_1", "get_schema", "{not valid json")
    scripted = [
        _FakeChatCompletion(_FakeMessage(None, [tool_call])),
        _FakeChatCompletion(_FakeMessage("ok")),
    ]
    _install_fake_client(monkeypatch, scripted=scripted)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert results[0].is_error is True
    assert "not valid JSON" in results[0].text


def test_send_handler_exception_reports_bounded_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)

    async def exploding_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("some internal detail that should not leak: /secret/path")

    from sift.provider import openai_compatible as mod

    monkeypatch.setitem(mod.HANDLERS, "get_schema", exploding_handler)

    tool_call = _FakeToolCall("call_1", "get_schema", "{}")
    scripted = [
        _FakeChatCompletion(_FakeMessage(None, [tool_call])),
        _FakeChatCompletion(_FakeMessage("ok")),
    ]
    _install_fake_client(monkeypatch, scripted=scripted)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert results[0].is_error is True
    assert "/secret/path" not in results[0].text
    assert "ValueError" in results[0].text


# ---------------------------------------------------------------------------
# send(): error classification
# ---------------------------------------------------------------------------


def test_send_authentication_error_yields_auth_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)

    class _RaisingCompletionsAPI:
        async def create(self, **kwargs: Any) -> Any:
            import openai as openai_pkg

            raise openai_pkg.AuthenticationError(
                message="invalid api key",
                response=_fake_httpx_response(401),
                body=None,
            )

    class _RaisingChatAPI:
        def __init__(self) -> None:
            self.completions = _RaisingCompletionsAPI()

    def _factory(api_key=None, base_url=None, **options):
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _RaisingChatAPI()
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert len(events) == 1
    assert isinstance(events[0], AuthFailure)


def test_send_timeout_error_yields_actionable_turn_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """``APITimeoutError`` (subclasses APIConnectionError -- must be
    checked first) gets a distinct message from a bare connection
    refusal: a large local model can legitimately take a while to
    load weights on first request, and that's a different
    troubleshooting story than "server isn't running"."""
    _configure_env(monkeypatch)

    class _RaisingCompletionsAPI:
        async def create(self, **kwargs: Any) -> Any:
            import openai as openai_pkg

            raise openai_pkg.APITimeoutError(request=None)

    class _RaisingChatAPI:
        def __init__(self) -> None:
            self.completions = _RaisingCompletionsAPI()

    def _factory(api_key=None, base_url=None, **options):
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _RaisingChatAPI()
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert "timed out" in events[0].message.lower()
    assert "loading weights" in events[0].message.lower()


def test_send_connection_error_names_configured_base_url(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The single most common first-use failure for this provider:
    the local server isn't running yet, wrong port, or a typo'd base
    URL. The message must name the ACTUAL configured base URL (not a
    generic string) and point at the likely causes -- this is the
    error a researcher setting up Ollama for the first time is most
    likely to see."""
    _configure_env(monkeypatch, base_url="http://localhost:11434/v1")

    class _RaisingCompletionsAPI:
        async def create(self, **kwargs: Any) -> Any:
            import openai as openai_pkg

            raise openai_pkg.APIConnectionError(
                message="Connection error.", request=None
            )

    class _RaisingChatAPI:
        def __init__(self) -> None:
            self.completions = _RaisingCompletionsAPI()

    def _factory(api_key=None, base_url=None, **options):
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _RaisingChatAPI()
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert "http://localhost:11434/v1" in events[0].message
    assert (
        "hasn't been started" in events[0].message
        or "not running" in events[0].message.lower()
        or "server" in events[0].message.lower()
    )


def test_send_not_found_error_names_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """A 404 from the endpoint most often means
    SIFT_OPENAI_COMPATIBLE_MODEL doesn't match a model name the
    server actually has -- the message must name the configured
    model and suggest the concrete check (``ollama list`` etc.),
    not read as a generic routing failure."""
    _configure_env(monkeypatch, model="llama3.1:not-a-real-tag")

    class _RaisingCompletionsAPI:
        async def create(self, **kwargs: Any) -> Any:
            import openai as openai_pkg

            raise openai_pkg.NotFoundError(
                "model not found", response=_fake_httpx_response(404), body=None
            )

    class _RaisingChatAPI:
        def __init__(self) -> None:
            self.completions = _RaisingCompletionsAPI()

    def _factory(api_key=None, base_url=None, **options):
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _RaisingChatAPI()
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert "llama3.1:not-a-real-tag" in events[0].message
    assert "ollama list" in events[0].message


def test_send_rate_limit_error_yields_retry_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Hosted gateways (OpenRouter, Together, Groq) enforce real rate
    limits, unlike most local servers -- worth a distinct message
    over a generic request failure."""
    _configure_env(monkeypatch)

    class _RaisingCompletionsAPI:
        async def create(self, **kwargs: Any) -> Any:
            import openai as openai_pkg

            raise openai_pkg.RateLimitError(
                "rate limited", response=_fake_httpx_response(429), body=None
            )

    class _RaisingChatAPI:
        def __init__(self) -> None:
            self.completions = _RaisingCompletionsAPI()

    def _factory(api_key=None, base_url=None, **options):
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _RaisingChatAPI()
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert (
        "rate-limited" in events[0].message.lower()
        or "rate limited" in events[0].message.lower()
    )
    assert "OpenRouter" in events[0].message


def test_send_generic_error_yields_turn_error_not_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)

    class _RaisingCompletionsAPI:
        async def create(self, **kwargs: Any) -> Any:
            raise ConnectionError("connection refused")

    class _RaisingChatAPI:
        def __init__(self) -> None:
            self.completions = _RaisingCompletionsAPI()

    def _factory(api_key=None, base_url=None, **options):
        client = _FakeAsyncOpenAI(
            api_key=api_key, base_url=base_url, **options
        )
        client.chat = _RaisingChatAPI()
        return client

    import openai as openai_pkg

    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _factory, raising=True)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert "connection refused" in events[0].message
    assert sess._messages == [{"role": "system", "content": "sys"}]


# ---------------------------------------------------------------------------
# Tool-loop convergence bound
# ---------------------------------------------------------------------------


def test_tool_loop_reports_error_when_never_converging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)
    monkeypatch.setenv("SIFT_OPENAI_COMPATIBLE_MAX_TOOL_ROUNDS", "2")

    async def fake_handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "{}"}]}

    from sift.provider import openai_compatible as mod

    monkeypatch.setitem(mod.HANDLERS, "get_schema", fake_handler)

    def _never_ending_call(i: int) -> _FakeChatCompletion:
        return _FakeChatCompletion(
            _FakeMessage(None, [_FakeToolCall(f"call_{i}", "get_schema", "{}")]),
        )

    scripted = [_never_ending_call(1), _never_ending_call(2), _never_ending_call(3)]
    _install_fake_client(monkeypatch, scripted=scripted)

    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    events = _run(sess.send("hi"))
    errors = [e for e in events if isinstance(e, TurnError)]
    assert len(errors) == 1
    assert "did not converge" in errors[0].message
    assert sess._messages == [{"role": "system", "content": "sys"}]


# ---------------------------------------------------------------------------
# set_model / set_effort
# ---------------------------------------------------------------------------


def test_set_model_accepts_the_one_catalog_entry(tmp_path: Path):
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    result = asyncio.run(sess.set_model("openai-compatible-custom"))
    assert result["ok"] is True


def test_set_model_rejects_unknown_id(tmp_path: Path):
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    result = asyncio.run(sess.set_model("not-a-real-model"))
    assert result["ok"] is False


def test_set_model_rejects_a_different_providers_model(tmp_path: Path):
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    result = asyncio.run(sess.set_model("gpt-5.6-sol"))
    assert result["ok"] is False
    assert "not 'openai_compatible'" in result["reason"]


def test_set_effort_always_succeeds_but_is_marked_unsupported(tmp_path: Path):
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    result = asyncio.run(sess.set_effort("xhigh"))
    assert result["ok"] is True
    assert result["unsupported"] is True
    assert sess.effort == "xhigh"


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_open_is_a_noop_without_config(tmp_path: Path):
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    asyncio.run(sess.open())
    assert sess._client is None


def test_resolved_model_name_is_none_before_open(tmp_path: Path):
    """Before config resolves (or when it never does), the accessor
    used by usage accounting must report ``None`` rather than the
    catalog placeholder — a caller falling back to ``self.model`` on
    ``None`` is the whole point."""
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    assert sess.resolved_model_name is None
    asyncio.run(sess.open())  # no config -> still unresolved
    assert sess.resolved_model_name is None


def test_resolved_model_name_reflects_the_real_configured_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Once ``open()`` resolves against real config, the accessor
    must return the ACTUAL model name (what ``SIFT_OPENAI_COMPATIBLE_
    MODEL`` names, "llama3.1" here) — never the fixed catalog
    placeholder ``self.model`` was constructed with. This is the
    value ``runner.py``'s usage accounting reads instead of
    ``self.model`` for this provider."""
    _configure_env(monkeypatch)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    asyncio.run(sess.open())
    assert sess.resolved_model_name == "llama3.1"
    assert sess.model == "openai-compatible-custom"  # unchanged, as documented


def test_resolved_model_name_clears_on_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _configure_env(monkeypatch)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    asyncio.run(sess.open())
    assert sess.resolved_model_name == "llama3.1"
    asyncio.run(sess.close())
    assert sess.resolved_model_name is None


def test_close_resets_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _configure_env(monkeypatch)
    _install_fake_client(monkeypatch)
    sess = OpenAICompatibleSession(
        cwd=tmp_path,
        model="openai-compatible-custom",
        system_prompt="sys",
    )
    asyncio.run(sess.open())
    assert sess._client is not None
    asyncio.run(sess.close())
    assert sess._client is None
    assert sess._messages == []
    assert sess._tools == []


def _fake_httpx_response(status_code: int) -> Any:
    """Minimal stand-in for the httpx.Response the openai SDK's
    exception constructors expect."""

    class _Resp:
        def __init__(self, code: int) -> None:
            self.status_code = code
            self.headers: dict[str, str] = {}
            self.request = None

        def json(self) -> dict[str, Any]:
            return {"error": {"message": "invalid api key"}}

    return _Resp(status_code)
