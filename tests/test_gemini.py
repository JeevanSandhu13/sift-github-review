"""Gemini provider coverage.

Covers ``provider/gemini.py`` -- the ``google-genai``-backed
``ProviderSession`` implementation. Test strategy mirrors
``test_openai_compatible.py`` (the equivalent adapter coverage, cited
in ``gemini.py``'s own module docstring): monkeypatch
``google.genai.Client`` with a fake that records every
``chat.send_message()`` call and returns scripted, SDK-shaped
responses, so the real ``send()`` tool loop runs against a stand-in
rather than a live network call. No outbound network access to
Google's API happens anywhere in this file.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from sift.provider.base import (
    AssistantText,
    AssistantThinking,
    AuthFailure,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from sift.provider.gemini import (
    GeminiSession,
    _build_user_parts,
    _extract_hints,
    _mcp_payload_to_text,
    _rewrite_for_dropped_image,
)
from sift.provider.tool_schemas import build_tool_specs

# ---------------------------------------------------------------------------
# Fixtures / fakes -- SDK-shaped stand-ins for google.genai objects
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "SIFT_GEMINI_MAX_TOOL_ROUNDS"):
        monkeypatch.delenv(var, raising=False)


class _FakeUsage:
    def __init__(
        self,
        prompt_token_count: int = 10,
        candidates_token_count: int = 5,
        thoughts_token_count: int = 0,
        cached_content_token_count: int = 0,
    ) -> None:
        self.prompt_token_count = prompt_token_count
        self.candidates_token_count = candidates_token_count
        self.thoughts_token_count = thoughts_token_count
        self.cached_content_token_count = cached_content_token_count


class _FakeFunctionCall:
    def __init__(
        self, name: str, args: dict[str, Any], id: str | None = None,
    ) -> None:
        self.name = name
        self.args = args
        self.id = id


class _FakePart:
    def __init__(
        self,
        text: str | None = None,
        function_call: _FakeFunctionCall | None = None,
        thought: bool = False,
    ) -> None:
        self.text = text
        self.function_call = function_call
        self.thought = thought


class _FakeContent:
    def __init__(self, parts: list[_FakePart]) -> None:
        self.parts = parts


class _FakeCandidate:
    def __init__(
        self,
        parts: list[_FakePart],
        finish_reason: str | None = None,
        finish_message: str | None = None,
    ) -> None:
        self.content = _FakeContent(parts)
        self.finish_reason = finish_reason
        self.finish_message = finish_message


class _FakeResponse:
    """SDK-shaped stand-in for the object ``chat.send_message()``
    resolves to: ``.candidates[0].content.parts`` and
    ``.usage_metadata``, matching exactly what ``send()`` reads via
    ``getattr``."""

    def __init__(
        self,
        parts: list[_FakePart],
        usage: _FakeUsage | None = None,
        *,
        finish_reason: str | None = None,
        finish_message: str | None = None,
    ) -> None:
        self.candidates = [
            _FakeCandidate(parts, finish_reason, finish_message)
        ]
        self.usage_metadata = usage or _FakeUsage()
        self.prompt_feedback = None


def _text_response(text: str, **usage_kwargs: int) -> _FakeResponse:
    usage = _FakeUsage(**usage_kwargs) if usage_kwargs else None
    return _FakeResponse([_FakePart(text=text)], usage)


def _thought_and_text_response(thought: str, text: str) -> _FakeResponse:
    return _FakeResponse([
        _FakePart(text=thought, thought=True),
        _FakePart(text=text),
    ])


def _tool_call_response(name: str, args: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse([_FakePart(function_call=_FakeFunctionCall(name, args))])


def _no_candidates_response() -> _FakeResponse:
    resp = _FakeResponse([])
    resp.candidates = []
    return resp


class _FakeChat:
    """Stand-in for ``google.genai.chats.AsyncChat``. Returns
    responses from a scripted queue in order and records every
    ``send_message()`` call's kwargs for assertions."""

    def __init__(
        self,
        scripted: list[Any] | None = None,
        history: list[Any] | None = None,
    ) -> None:
        self._queue: list[Any] = list(scripted or [])
        self.calls: list[dict[str, Any]] = []
        self.model: str | None = None
        self.config: Any = None
        self._curated_history = list(history or [])
        self._comprehensive_history = list(history or [])

    async def send_message(self, message_parts: Any, config: Any = None) -> Any:
        self.calls.append({"message_parts": message_parts, "config": config})
        if not self._queue:
            raise RuntimeError("test queue exhausted -- add more scripted responses")
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        self._curated_history.extend([message_parts, item])
        self._comprehensive_history.extend([message_parts, item])
        return item

    def get_history(self, curated: bool = False) -> list[Any]:
        history = self._curated_history if curated else self._comprehensive_history
        return list(history)


class _FakeChats:
    def __init__(self, factory: Any) -> None:
        self._factory = factory

    def create(
        self,
        *,
        model: str,
        config: Any = None,
        history: list[Any] | None = None,
    ) -> _FakeChat:
        return self._factory(model, config, history)


class _FakeAio:
    def __init__(self, chats: _FakeChats) -> None:
        self.chats = chats


class _FakeClient:
    """Stand-in for ``google.genai.Client``. Every
    ``aio.chats.create()`` call returns a fresh ``_FakeChat`` seeded
    from the same scripted queue supplied at construction (tests
    that need per-chat queues build a fresh client per test, which is
    the normal case since ``GeminiSession`` builds exactly one client
    per session lifetime)."""

    def __init__(
        self,
        api_key: str | None = None,
        scripted: list[Any] | None = None,
        http_options: Any = None,
    ) -> None:
        self.api_key = api_key
        self.http_options = http_options
        self._scripted = scripted
        self.created_chats: list[_FakeChat] = []

    def _make_chat(
        self,
        model: str,
        config: Any,
        history: list[Any] | None = None,
    ) -> _FakeChat:
        chat = _FakeChat(self._scripted, history)
        chat.model = model
        chat.config = config
        self.created_chats.append(chat)
        return chat

    @property
    def aio(self) -> _FakeAio:
        return _FakeAio(_FakeChats(self._make_chat))


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch, scripted: list[Any] | None = None,
) -> list[_FakeClient]:
    """Patch ``google.genai.Client`` so the next client constructed
    inside ``GeminiSession.open()`` is a fake seeded with
    ``scripted`` responses. Returns a list the constructed fake
    client gets appended to."""
    constructed: list[_FakeClient] = []

    def _factory(
        api_key: str | None = None, http_options: Any = None,
    ) -> _FakeClient:
        client = _FakeClient(
            api_key=api_key, scripted=scripted, http_options=http_options,
        )
        constructed.append(client)
        return client

    import google.genai as genai_pkg
    monkeypatch.setattr(genai_pkg, "Client", _factory, raising=True)
    return constructed


def _run(coro_iter: Any) -> list[Any]:
    async def _collect() -> list[Any]:
        return [e async for e in coro_iter]
    return asyncio.run(_collect())


# ---------------------------------------------------------------------------
# Auth / lifecycle
# ---------------------------------------------------------------------------


def test_open_with_no_key_defers_client_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: None)

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    asyncio.run(sess.open())
    assert sess._client is None
    assert sess._chat is None


def test_send_with_no_key_yields_auth_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: None)

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hello"))
    assert events, "send() must yield at least one event on missing key"
    assert isinstance(events[0], AuthFailure)
    assert "Gemini API key" in events[0].reason


def test_open_constructs_client_and_verified_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    constructed = _install_fake_client(monkeypatch, scripted=[_text_response("ok")])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    asyncio.run(sess.open())

    assert len(constructed) == 1
    assert constructed[0].api_key == "AIza-test"
    assert constructed[0].http_options.timeout == 300_000
    assert constructed[0].http_options.retry_options.attempts == 1
    assert sess._chat is not None
    # The tool built at open() must be the lockdown-clean Sift tool.
    expected_names = {s.name for s in build_tool_specs()}
    sent_names = {d.name for d in sess._tool.function_declarations}
    assert sent_names == expected_names


def test_close_resets_all_session_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[_text_response("ok")])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    asyncio.run(sess.open())
    asyncio.run(sess.close())
    assert sess._client is None
    assert sess._chat is None
    assert sess._tool is None


def test_close_releases_both_google_transport_layers(tmp_path: Path) -> None:
    calls: list[str] = []

    class AsyncClient:
        async def aclose(self) -> None:
            calls.append("async")

    class Client:
        aio = AsyncClient()

        def close(self) -> None:
            calls.append("sync")

    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift",
    )
    sess._client = Client()
    sess._chat = object()
    sess._tool = object()

    asyncio.run(sess.close())

    assert calls == ["async", "sync"]
    assert sess._client is None
    assert sess._chat is None
    assert sess._tool is None


# ---------------------------------------------------------------------------
# send() -- plain text turn, no tool calls
# ---------------------------------------------------------------------------


def test_send_plain_turn_yields_text_then_turndone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        _text_response("hello researcher", prompt_token_count=12, candidates_token_count=4),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hi"))

    texts = [e for e in events if isinstance(e, AssistantText)]
    assert len(texts) == 1
    assert texts[0].text == "hello researcher"
    assert isinstance(events[-1], TurnDone)
    assert events[-1].input_tokens == 12
    assert events[-1].output_tokens == 4


def test_two_turns_reuse_one_client_side_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    constructed = _install_fake_client(monkeypatch, scripted=[
        _text_response("first"), _text_response("second"),
    ])
    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift",
    )
    assert isinstance(_run(sess.send("one"))[-1], TurnDone)
    assert isinstance(_run(sess.send("two"))[-1], TurnDone)
    assert len(constructed[0].created_chats) == 1
    assert len(constructed[0].created_chats[0].calls) == 2


def test_send_thought_part_yields_assistantthinking_not_assistanttext(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        _thought_and_text_response("reasoning about the data...", "final answer"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hi"))

    thinking = [e for e in events if isinstance(e, AssistantThinking)]
    texts = [e for e in events if isinstance(e, AssistantText)]
    assert len(thinking) == 1
    assert thinking[0].text == "reasoning about the data..."
    assert len(texts) == 1
    assert texts[0].text == "final answer"


def test_send_only_passes_the_lockdown_clean_tool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Drive a real send() and assert the ``config.tools`` sent on
    the request contains exactly the Sift tool -- the test that
    would catch a future PR turning on a Gemini built-in
    (google_search, code_execution, ...)."""
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    constructed = _install_fake_client(monkeypatch, scripted=[_text_response("ok")])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    _run(sess.send("hi"))

    chat = constructed[0].created_chats[-1]
    assert len(chat.calls) == 1
    config = chat.calls[0]["config"]
    tools = config.tools
    assert len(tools) == 1
    tool = tools[0]
    expected_names = {s.name for s in build_tool_specs()}
    sent_names = {d.name for d in tool.function_declarations}
    assert sent_names == expected_names
    for field_name in type(tool).model_fields:
        if field_name == "function_declarations":
            continue
        assert getattr(tool, field_name, None) is None

    # Automatic function calling must be explicitly disabled -- Sift
    # always dispatches through HANDLERS, never the SDK's own
    # convenience executor.
    assert config.automatic_function_calling.disable is True


# ---------------------------------------------------------------------------
# send() -- tool-call round trip
# ---------------------------------------------------------------------------


def test_send_dispatches_function_call_and_returns_function_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    async def _fake_list_results(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": '{"status":"ok","results":[]}'}]}

    monkeypatch.setitem(HANDLERS, "list_results", _fake_list_results)

    constructed = _install_fake_client(monkeypatch, scripted=[
        _tool_call_response("list_results", {}),
        _text_response("done"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("show me the results"))

    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(calls) == 1
    assert calls[0].name == "list_results"
    assert len(results) == 1
    assert results[0].is_error is False
    assert isinstance(events[-1], TurnDone)

    chat = constructed[0].created_chats[-1]
    assert len(chat.calls) == 2
    # Round 2's message_parts must be exactly the function_response
    # part built from round 1's tool call, not a replay of the
    # user's original prompt (the chat's own internal history
    # carries that already).
    round2_parts = chat.calls[1]["message_parts"]
    assert len(round2_parts) == 1
    fr = round2_parts[0].function_response
    assert fr.name == "list_results"
    assert fr.response == {"result": '{"status":"ok","results":[]}'}


def test_send_dispatches_multiple_calls_in_one_round_by_position(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Gemini's ``FunctionCall.id`` is optional and frequently
    absent; the dispatch loop matches function_response parts to
    function_call parts by ORDER within the round. Two calls in one
    round must come back in the same order."""
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    async def _handler_a(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "A-result"}]}

    async def _handler_b(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "B-result"}]}

    monkeypatch.setitem(HANDLERS, "list_results", _handler_a)
    monkeypatch.setitem(HANDLERS, "get_schema", _handler_b)

    two_call_response = _FakeResponse([
        _FakePart(function_call=_FakeFunctionCall("list_results", {})),
        _FakePart(function_call=_FakeFunctionCall("get_schema", {"dataset": "x"})),
    ])
    constructed = _install_fake_client(monkeypatch, scripted=[
        two_call_response,
        _text_response("done"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    _run(sess.send("go"))

    chat = constructed[0].created_chats[-1]
    round2_parts = chat.calls[1]["message_parts"]
    assert len(round2_parts) == 2
    assert round2_parts[0].function_response.name == "list_results"
    assert round2_parts[0].function_response.response == {"result": "A-result"}
    assert round2_parts[1].function_response.name == "get_schema"
    assert round2_parts[1].function_response.response == {"result": "B-result"}


def test_parallel_calls_to_the_same_tool_get_distinct_call_ids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Audit pass 2 finding: the model can issue TWO calls to the
    SAME tool in one round (e.g. ``get_schema`` for two different
    datasets). Before the fix, ``call_id`` was the bare tool name, so
    both ToolCall events -- and both ToolCallResult events -- shared
    one call_id. The frontend keys a tool-result card off call_id
    (``card.dataset.callId`` / the matching ``.find()`` in app.js) to
    route each result back to the card its own call created; with a
    shared call_id, ``.find()`` always resolves to the FIRST card, so
    the first result silently overwrites the same card twice and the
    second card is left stuck "pending" forever.

    This simulates that exact frontend routing logic directly against
    the real event stream, rather than re-deriving app.js in Python,
    and asserts each of the two cards receives its OWN correct
    result.
    """
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    async def _fake_get_schema(args: dict[str, Any]) -> dict[str, Any]:
        dataset = args.get("dataset", "?")
        return {"content": [{"type": "text", "text": f"schema-for-{dataset}"}]}

    monkeypatch.setitem(HANDLERS, "get_schema", _fake_get_schema)

    two_call_response = _FakeResponse([
        _FakePart(function_call=_FakeFunctionCall(
            "get_schema", {"dataset": "a.csv"})),
        _FakePart(function_call=_FakeFunctionCall(
            "get_schema", {"dataset": "b.csv"})),
    ])
    _install_fake_client(monkeypatch, scripted=[
        two_call_response,
        _text_response("done"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("go"))

    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(calls) == 2
    assert len(results) == 2

    # The bug, made concrete: call_id must differ across the two
    # calls to the same tool.
    assert calls[0].call_id != calls[1].call_id, (
        "two parallel calls to the same tool got the SAME call_id -- "
        "this is exactly the UI-corrupting bug: the frontend cannot "
        "tell the two tool cards apart"
    )
    assert results[0].call_id != results[1].call_id

    # Simulate app.js's card-routing logic: one "card" created per
    # ToolCall (keyed by call_id), then each ToolCallResult is routed
    # to the card whose call_id matches -- ``.find()`` returns the
    # FIRST match, exactly like Array.prototype.find in app.js.
    cards = [{"call_id": c.call_id, "input": c.input, "result": None} for c in calls]
    for r in results:
        card = next((c for c in cards if c["call_id"] == r.call_id), None)
        assert card is not None, f"no card found for result call_id {r.call_id!r}"
        card["result"] = r.text

    assert cards[0]["result"] is not None, "first card was never filled in"
    assert cards[1]["result"] is not None, (
        "second card was never filled in -- it would be stuck "
        "'pending' forever in the real UI"
    )
    # And each card got the RESULT MATCHING ITS OWN ARGS, not a
    # copy of the other call's result.
    assert cards[0]["result"] == "schema-for-a.csv"
    assert cards[1]["result"] == "schema-for-b.csv"


def test_gemini_supplied_function_call_id_is_used_verbatim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When the SDK DOES populate ``FunctionCall.id`` (real Gemini
    responses sometimes do), Sift must use it directly rather than
    synthesizing its own -- preferring the SDK's own id keeps Sift's
    event stream aligned with whatever the SDK itself considers the
    call's identity."""
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    async def _fake_list_results(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setitem(HANDLERS, "list_results", _fake_list_results)

    resp = _FakeResponse([_FakePart(function_call=_FakeFunctionCall(
        "list_results", {}, id="fc-abc123"))])
    _install_fake_client(monkeypatch, scripted=[resp, _text_response("done")])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("go"))

    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert calls[0].call_id == "fc-abc123"
    assert results[0].call_id == "fc-abc123"


def test_synthesized_call_id_is_stable_and_round_scoped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When ``FunctionCall.id`` is absent (the common case), the
    synthesized call_id must still be deterministic given the same
    round/position/tool-name shape -- not e.g. a fresh random value
    that would make snapshot-based debugging impossible -- and must
    incorporate enough of the round/position to disambiguate two
    calls to the same tool within one round (this is checked more
    directly by the parallel-calls test above; here we just pin the
    format so a future change to it is a deliberate, visible diff)."""
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    async def _fake_list_results(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setitem(HANDLERS, "list_results", _fake_list_results)

    _install_fake_client(monkeypatch, scripted=[
        _tool_call_response("list_results", {}),
        _text_response("done"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("go"))

    calls = [e for e in events if isinstance(e, ToolCall)]
    assert len(calls) == 1
    assert calls[0].call_id.startswith("list_results:")
    assert calls[0].call_id != "list_results", (
        "call_id must not be the bare tool name -- that's the exact "
        "shape the case-collision bug used"
    )


def test_unknown_tool_name_yields_error_result_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        _tool_call_response("not_a_real_tool", {}),
        _text_response("done"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("go"))

    results = [e for e in events if isinstance(e, ToolCallResult)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert isinstance(events[-1], TurnDone)


def test_handler_exception_does_not_leak_message_to_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Mirrors the OpenAI provider's equivalent guard: a raised
    handler exception's ``str(e)`` must never ride the
    function_response back to the model -- it can carry raw data
    values a per-tool redaction pass would have scrubbed."""
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    SENSITIVE = "row 42: name=Jane Doe income=$487192 email=jane.doe@example.com"

    async def _raising_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(SENSITIVE)

    monkeypatch.setitem(HANDLERS, "list_results", _raising_handler)

    constructed = _install_fake_client(monkeypatch, scripted=[
        _tool_call_response("list_results", {}),
        _text_response("done"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    _run(sess.send("go"))

    chat = constructed[0].created_chats[-1]
    round2_parts = chat.calls[1]["message_parts"]
    out = round2_parts[0].function_response.response["result"]
    assert SENSITIVE not in out
    assert "Jane Doe" not in out
    assert "jane.doe@example.com" not in out
    assert "$487192" not in out
    assert "RuntimeError" in out


def test_send_yields_turnerror_when_tool_loop_does_not_converge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setitem(HANDLERS, "list_results", _handler)
    monkeypatch.setenv("SIFT_GEMINI_MAX_TOOL_ROUNDS", "3")

    scripted = [_tool_call_response("list_results", {}) for _ in range(3)]
    constructed = _install_fake_client(monkeypatch, scripted=scripted)

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("loop forever"))

    assert not any(isinstance(e, TurnDone) for e in events)
    assert isinstance(events[-1], TurnError)
    chat = constructed[0].created_chats[-1]
    assert len(chat.calls) == 3


def test_no_candidates_yields_turnerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[_no_candidates_response()])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hi"))
    assert isinstance(events[-1], TurnError)
    assert "no candidates" in events[-1].message.lower()


@pytest.mark.parametrize(
    "finish_reason",
    [
        "MAX_TOKENS",
        "SAFETY",
        "RECITATION",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "TOO_MANY_TOOL_CALLS",
    ],
)
def test_non_stop_finish_reason_rolls_back_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    finish_reason: str,
) -> None:
    """Provider terminal reasons other than STOP are incomplete or
    blocked turns, never successful answers and never valid history."""
    from sift.provider import gemini as gemini_provider

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    response = _FakeResponse(
        [_FakePart(text="partial")],
        finish_reason=finish_reason,
        finish_message="provider stopped generation",
    )
    constructed = _install_fake_client(monkeypatch, scripted=[response])
    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift",
    )
    events = _run(sess.send("hi"))

    assert isinstance(events[-1], TurnError)
    assert finish_reason in events[-1].message
    assert not any(isinstance(event, TurnDone) for event in events)
    chat = constructed[0].created_chats[-1]
    assert chat.get_history(curated=True) == []
    assert chat.get_history(curated=False) == []


def test_failed_turn_restores_existing_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Rollback returns to the precise pre-turn checkpoint rather than
    clearing earlier successful research context."""
    from sift.provider import gemini as gemini_provider

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    response = _FakeResponse(
        [_FakePart(text="partial")],
        finish_reason="MAX_TOKENS",
    )
    constructed = _install_fake_client(monkeypatch, scripted=[response])
    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift",
    )
    asyncio.run(sess.open())
    chat = constructed[0].created_chats[-1]
    chat._curated_history[:] = ["prior-user", "prior-model"]
    chat._comprehensive_history[:] = ["prior-user", "prior-model"]

    _run(sess.send("new turn"))
    assert chat.get_history(curated=True) == ["prior-user", "prior-model"]
    assert chat.get_history(curated=False) == ["prior-user", "prior-model"]


def test_malformed_function_arguments_do_not_invoke_handler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Coercing malformed arguments to an empty object can invoke a
    state-changing handler with unintended defaults; return a tool error."""
    from sift.provider import gemini as gemini_provider
    from sift.tools import HANDLERS

    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    called = False

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        nonlocal called
        called = True
        return {"content": [{"type": "text", "text": "should not run"}]}

    monkeypatch.setitem(HANDLERS, "list_results", _handler)
    malformed = _FakeResponse([
        _FakePart(function_call=_FakeFunctionCall("list_results", None)),  # type: ignore[arg-type]
    ])
    _install_fake_client(
        monkeypatch,
        scripted=[malformed, _text_response("recovered")],
    )
    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift",
    )
    events = _run(sess.send("go"))

    assert called is False
    errors = [event for event in events if isinstance(event, ToolCallResult)]
    assert len(errors) == 1 and errors[0].is_error is True
    assert "not a JSON object" in errors[0].text
    assert isinstance(events[-1], TurnDone)


# ---------------------------------------------------------------------------
# send() -- request failures translated to provider-neutral events
# ---------------------------------------------------------------------------


def test_auth_error_during_send_message_yields_authfailure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        RuntimeError("401 Unauthorized: invalid API key"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hi"))
    assert isinstance(events[-1], AuthFailure)


def test_context_window_error_translated_to_turnerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        RuntimeError("400 Bad Request: input token count exceeds context limit"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hi"))
    assert isinstance(events[-1], TurnError)
    assert "context" in events[-1].message.lower()
    assert "recall_conversation" in events[-1].message


def test_generic_request_failure_yields_turnerror(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        RuntimeError("503 Service Unavailable"),
    ])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="you are sift")
    events = _run(sess.send("hi"))
    assert isinstance(events[-1], TurnError)
    assert "Gemini request failed" in events[-1].message


# ---------------------------------------------------------------------------
# set_model / set_effort
# ---------------------------------------------------------------------------


def test_set_model_rejects_unknown_model(tmp_path: Path) -> None:
    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x")
    result = asyncio.run(sess.set_model("not-a-real-model"))
    assert result["ok"] is False


def test_set_model_rejects_cross_provider_model(tmp_path: Path) -> None:
    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x")
    result = asyncio.run(sess.set_model("gpt-5.6-sol"))
    assert result["ok"] is False
    assert "not Gemini" in result["reason"]


def test_set_model_unchanged_when_already_active(tmp_path: Path) -> None:
    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x")
    result = asyncio.run(sess.set_model("gemini-3.7-flash"))
    assert result["ok"] is True
    assert result.get("unchanged") is True


def test_set_model_switches_and_rebuilds_chat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    constructed = _install_fake_client(monkeypatch, scripted=[_text_response("ok")])

    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x")
    asyncio.run(sess.open())
    first_chat = sess._chat
    first_chat._curated_history[:] = ["prior-user", "prior-model"]
    first_chat._comprehensive_history[:] = ["prior-user", "prior-model"]

    result = asyncio.run(sess.set_model("gemini-3.1-pro-preview"))
    assert result["ok"] is True
    assert sess.model == "gemini-3.1-pro-preview"
    assert sess._chat is not first_chat
    assert len(constructed[0].created_chats) == 2
    assert constructed[0].created_chats[-1].model == "gemini-3.1-pro-preview"
    assert constructed[0].created_chats[-1].get_history(curated=True) == [
        "prior-user", "prior-model",
    ]


def test_set_effort_rejects_level_outside_geminis_ladder(tmp_path: Path) -> None:
    sess = GeminiSession(cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x")
    result = asyncio.run(sess.set_effort("max"))
    assert result["ok"] is False


def test_set_effort_unchanged_when_already_active(tmp_path: Path) -> None:
    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x", effort="high",
    )
    result = asyncio.run(sess.set_effort("high"))
    assert result["ok"] is True
    assert result.get("unchanged") is True


def test_set_effort_switches_and_next_request_uses_new_thinking_level(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    constructed = _install_fake_client(monkeypatch, scripted=[_text_response("ok")])

    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x", effort="low",
    )
    result = asyncio.run(sess.set_effort("high"))
    assert result["ok"] is True
    assert sess.effort == "high"

    _run(sess.send("hi"))
    chat = constructed[0].created_chats[-1]
    config = chat.calls[0]["config"]
    assert config.thinking_config.thinking_level == "HIGH"


def test_construction_clamps_default_effort_to_geminis_ceiling(tmp_path: Path) -> None:
    """A fresh session with no explicit effort (or the cross-provider
    default ``xhigh``) must clamp DOWN to Gemini's own ceiling
    (``high``), not silently land on a middle rung."""
    sess = GeminiSession(
        cwd=tmp_path, model="gemini-3.7-flash", system_prompt="x", effort="xhigh",
    )
    assert sess.effort == "high"


# ---------------------------------------------------------------------------
# Pure helper functions (exercise the real google-genai types)
# ---------------------------------------------------------------------------


def test_build_user_parts_text_only() -> None:
    parts = _build_user_parts("hello", None)
    assert len(parts) == 1
    assert parts[0].text == "hello"


def test_build_user_parts_with_image() -> None:
    import base64
    data = base64.b64encode(b"fake-png-bytes").decode()
    parts = _build_user_parts("look at this", [{"mime": "image/png", "data": data}])
    assert len(parts) == 2
    assert parts[0].text == "look at this"
    assert parts[1].inline_data.mime_type == "image/png"


def test_build_user_parts_skips_malformed_image_data() -> None:
    parts = _build_user_parts("hi", [{"mime": "image/png", "data": "not-valid-base64!!"}])
    # Malformed image is dropped, text part still present.
    assert len(parts) == 1
    assert parts[0].text == "hi"


def test_mcp_payload_to_text_plain_text_passthrough() -> None:
    payload = {"content": [{"type": "text", "text": '{"status":"ok"}'}]}
    assert _mcp_payload_to_text(payload) == '{"status":"ok"}'


def test_mcp_payload_to_text_rewrites_dropped_image() -> None:
    import json as _json
    descriptor = _json.dumps({
        "status": "ok", "name": "plot.png", "kind": "image",
        "ext": ".png", "mime": "image/png", "size": 100,
    })
    payload = {
        "content": [
            {"type": "image", "data": "BASE64", "mimeType": "image/png"},
            {"type": "text", "text": descriptor},
        ]
    }
    out = _mcp_payload_to_text(payload)
    parsed = _json.loads(out)
    assert parsed["status"] == "image_not_supported_on_provider"
    assert "plot.png" in parsed["reason"]
    assert "@mention" in parsed["reason"]


def test_rewrite_for_dropped_image_falls_back_on_non_json() -> None:
    assert _rewrite_for_dropped_image("not json") == "not json"


def test_extract_hints_pulls_run_dir_and_language() -> None:
    import json as _json
    text = _json.dumps({"_run_dir": "/tmp/run1", "_language": "python"})
    rd, lang = _extract_hints(text)
    assert rd == "/tmp/run1"
    assert lang == "python"


def test_extract_hints_returns_none_on_malformed_json() -> None:
    assert _extract_hints("not json") == (None, None)


# ---------------------------------------------------------------------------
# _translate_send_exception -- structured error translation
# ---------------------------------------------------------------------------
#
# Provider readiness hardening: chat.send_message() failures are now
# translated from the SDK's own STRUCTURED exception fields
# (google.genai.errors.APIError's .code/.status/.message, and the
# underlying httpx transport exceptions for connection-level
# failures) rather than only regexing str(e). These tests construct
# real errors.ClientError / errors.ServerError / httpx exception
# instances -- the actual types the SDK raises -- rather than stand-
# ins, so a mismatch between what this function expects and what the
# SDK really raises would show up here.


def _client_error(code: int, status: str, message: str):
    from google.genai import errors
    return errors.ClientError(code, {"error": {"status": status, "message": message}})


def _server_error(code: int, status: str, message: str):
    from google.genai import errors
    return errors.ServerError(code, {"error": {"status": status, "message": message}})


def test_translate_401_client_error_to_auth_failure() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _client_error(401, "UNAUTHENTICATED", "API key not valid")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, AuthFailure)
    assert "401" in out.reason
    assert "API key not valid" in out.reason


def test_translate_403_permission_denied_to_auth_failure() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _client_error(403, "PERMISSION_DENIED", "caller lacks permission")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, AuthFailure)


def test_translate_404_to_turnerror_naming_model(
) -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _client_error(404, "NOT_FOUND", "model not found")
    out = _translate_send_exception(e, "gemini-9000-nonexistent")
    assert isinstance(out, TurnError)
    assert "gemini-9000-nonexistent" in out.message
    assert "404" in out.message


def test_translate_429_to_turnerror_rate_limit_guidance() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _client_error(429, "RESOURCE_EXHAUSTED", "quota exceeded")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "429" in out.message
    assert "quota" in out.message.lower() or "rate" in out.message.lower()


def test_translate_5xx_server_error_to_turnerror_retry_guidance() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _server_error(503, "UNAVAILABLE", "server overloaded")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "503" in out.message
    assert "retry" in out.message.lower()


def test_translate_400_context_length_to_context_reset_guidance() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _client_error(
        400, "INVALID_ARGUMENT",
        "The input token count exceeds the maximum context length "
        "for this model.",
    )
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "context" in out.message.lower()
    assert "recall_conversation" in out.message


def test_translate_generic_400_to_turnerror() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = _client_error(400, "INVALID_ARGUMENT", "malformed request body")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "400" in out.message
    assert "malformed request body" in out.message


def test_translate_httpx_timeout_to_turnerror() -> None:
    import httpx

    from sift.provider.gemini import _translate_send_exception
    e = httpx.ConnectTimeout("connection timed out")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "timed out" in out.message.lower()


def test_translate_httpx_connect_error_to_turnerror() -> None:
    import httpx

    from sift.provider.gemini import _translate_send_exception
    e = httpx.ConnectError("connection refused")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "connect" in out.message.lower()
    assert "generativelanguage.googleapis.com" in out.message


def test_translate_unrecognized_exception_falls_back_to_substring_heuristic() -> None:
    """A bare RuntimeError (no structured .code/.status, not an httpx
    transport exception) must still fall through to the original
    substring-matching heuristic -- preserves compatibility with
    anything that doesn't shape itself as one of the SDK's own
    typed exceptions."""
    from sift.provider.gemini import _translate_send_exception
    e = RuntimeError("401 Unauthorized: invalid API key")
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, AuthFailure)


def test_translate_unrecognized_exception_context_fallback() -> None:
    from sift.provider.gemini import _translate_send_exception
    e = RuntimeError(
        "context token limit exceeded, request too long for this model"
    )
    out = _translate_send_exception(e, "gemini-3.7-flash")
    assert isinstance(out, TurnError)
    assert "context" in out.message.lower()


def test_send_translates_structured_404_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """End-to-end: a real errors.ClientError raised from inside
    chat.send_message() during a live send() call reaches the caller
    as a TurnError naming the model, not a raw/unhandled exception or
    a generic message."""
    from sift.provider import gemini as gemini_provider
    monkeypatch.setattr(gemini_provider, "_resolve_api_key", lambda: "AIza-test")
    _install_fake_client(monkeypatch, scripted=[
        _client_error(404, "NOT_FOUND", "model not found"),
    ])

    sess = GeminiSession(
        cwd=tmp_path, model="gemini-9000-nonexistent", system_prompt="x",
    )
    events = _run(sess.send("hi"))
    assert isinstance(events[-1], TurnError)
    assert "gemini-9000-nonexistent" in events[-1].message
