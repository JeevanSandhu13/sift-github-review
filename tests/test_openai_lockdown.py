"""Lockdown tests for the OpenAI provider.

Mirrors the spirit of Sift's existing Claude SDK lockdown coverage:
no matter what changes upstream in the OpenAI Responses API, the
``tools`` field Sift sends must contain EXACTLY the Sift function
tools (the canonical list in ``build_tool_specs()``) and no built-in
types (web_search, code_interpreter, file_search, image_generation,
mcp, computer_use_preview, …).

Without these guards, a future "let's enable web_search to help with
literature lookups" PR could silently punch a hole in the privacy
boundary — the model would gain a way to talk to the open internet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sift.provider.openai import (
    FORBIDDEN_BUILTIN_TYPES,
    OpenAISession,
    _mcp_payload_to_text,
    _verify_lockdown,
    build_openai_tools,
)
from sift.provider.tool_schemas import build_tool_specs

# ---------------------------------------------------------------------------
# Static checks on the tool list
# ---------------------------------------------------------------------------

def test_tool_list_contains_only_function_tools():
    tools = build_openai_tools()
    # Sanity floor: drop-out below this would mean the schema list
    # got truncated. We don't pin an exact count any more because
    # new tools land here naturally as the surface grows.
    assert len(tools) >= 6
    # Match the canonical specs exactly. That's the real invariant
    # (covered separately by test_tool_list_names_match_canonical_specs)
    # and stops a stray duplicate from sneaking in.
    assert len(tools) == len(build_tool_specs())
    assert all(t.get("type") == "function" for t in tools), (
        "every Sift tool sent to OpenAI must be a function tool"
    )


def test_tool_list_names_match_canonical_specs():
    tools = build_openai_tools()
    sent_names = {t["name"] for t in tools}
    expected_names = {s.name for s in build_tool_specs()}
    assert sent_names == expected_names


def test_tool_list_excludes_every_known_builtin():
    tools = build_openai_tools()
    sent_types = {t.get("type") for t in tools}
    # Every entry must be the "function" type — none of the built-in
    # types may ever appear.
    forbidden_seen = sent_types & FORBIDDEN_BUILTIN_TYPES
    assert not forbidden_seen, (
        f"built-in types leaked into the tool list: {forbidden_seen}"
    )


def test_lockdown_verifier_rejects_each_forbidden_type():
    """Synthetically inject every known built-in into the tool list
    and confirm the verifier raises. Mirrors the SDK lockdown test
    that ensures we know how to spot each kind of escape."""
    base = build_openai_tools()
    for builtin in FORBIDDEN_BUILTIN_TYPES:
        bad = base + [{"type": builtin}]
        with pytest.raises(RuntimeError, match="lockdown"):
            _verify_lockdown(bad)


def test_lockdown_verifier_rejects_unknown_function_name():
    """A non-Sift function tool name must also be rejected — covers
    the case of someone "helpfully" appending a custom helper."""
    base = build_openai_tools()
    bad = base + [{
        "type": "function",
        "name": "exfiltrate_data",
        "description": "evil",
        "parameters": {"type": "object", "properties": {}},
    }]
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(bad)


def test_lockdown_verifier_rejects_missing_or_duplicate_function() -> None:
    base = build_openai_tools()
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(base[:-1])
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(base + [dict(base[0])])


# ---------------------------------------------------------------------------
# Image-bearing tool results — pixels can't ride function_call_output
# ---------------------------------------------------------------------------


def test_text_only_payload_passes_through_unchanged():
    """A normal tool result with only a text content block is
    forwarded verbatim — no rewrite, no provider-specific noise."""
    payload = {
        "content": [
            {"type": "text", "text": '{"status":"ok","result_id":"M1"}'},
        ]
    }
    assert _mcp_payload_to_text(payload) == (
        '{"status":"ok","result_id":"M1"}'
    )


def test_image_payload_descriptor_is_rewritten_for_openai():
    """``read_attached_file`` returns a hedged "if your provider
    doesn't support images" descriptor alongside an image block. On
    OpenAI the image bytes can't ride the function_call_output, so
    the descriptor must be rewritten to tell the model definitively
    that the image was dropped — and to point at the recovery path
    (re-@mention, which uses the user-message vision channel)."""
    descriptor = json.dumps({
        "status": "ok",
        "name": "residuals.png",
        "kind": "image",
        "ext": ".png",
        "mime": "image/png",
        "size": 12345,
        "note": (
            "The image is attached as an inline content block. "
            "If your provider doesn't support image tool results, "
            "ask the researcher to re-@mention the file in their "
            "next message."
        ),
    })
    payload = {
        "content": [
            {"type": "image", "data": "BASE64...", "mimeType": "image/png"},
            {"type": "text", "text": descriptor},
        ]
    }
    rewritten = _mcp_payload_to_text(payload)
    parsed = json.loads(rewritten)
    assert parsed["status"] == "image_not_supported_on_provider"
    assert parsed["name"] == "residuals.png"
    # The model is told what to ask the researcher to do, definitively.
    assert "re-@mention" in parsed["reason"]
    assert "residuals.png" in parsed["reason"]
    # The hedged "If your provider doesn't support" wording must NOT
    # leak through — that conditional was the whole problem.
    assert "If your provider" not in rewritten


def test_malformed_descriptor_falls_back_to_original_text():
    """If the descriptor isn't the JSON shape we expect (e.g. an
    older tool, a hand-written test fixture, an MCP server we don't
    own), the rewrite path must NOT raise — fall back to the
    original text. The rewrite is best-effort polish, not a load-
    bearing parse."""
    payload = {
        "content": [
            {"type": "image", "data": "BASE64...", "mimeType": "image/png"},
            {"type": "text", "text": "not json"},
        ]
    }
    out = _mcp_payload_to_text(payload)
    assert out == "not json"


# ---------------------------------------------------------------------------
# Lockdown is checked on every request
# ---------------------------------------------------------------------------


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5


class _FakeResponse:
    """Minimal Responses-API-shaped object: an ``output`` list with
    one finished message, plus a usage block. Used to short-circuit
    the tool-loop in send() so the test only verifies what was sent,
    not what comes back."""

    def __init__(self) -> None:
        # One assistant text item, no function_calls — send() exits
        # the tool loop after the first round.
        class _Block:
            type = "output_text"
            text = "ok"

        class _Item:
            type = "message"
            content = [_Block()]

            def model_dump(self) -> dict[str, Any]:
                return {"type": "message", "content": [{"type": "output_text", "text": "ok"}]}

        self.output = [_Item()]
        self.usage = _FakeUsage()


class _FakeResponsesAPI:
    """Captures every ``create()`` call's kwargs so the test can
    assert what was sent."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeAsyncOpenAI:
    """Stand-in for ``openai.AsyncOpenAI``. The test substitutes one
    of these for the real client by monkey-patching the SDK import
    inside ``OpenAISession.open()``."""

    def __init__(self, api_key: str | None = None, **options: Any) -> None:
        self.api_key = api_key
        self.options = options
        self.responses = _FakeResponsesAPI()

    async def close(self) -> None:
        return None


def test_send_only_passes_function_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Drive a real send() against a fake OpenAI client and assert
    the ``tools`` kwarg contains exactly Sift's function tools, and
    nothing else. This is the test that catches a future PR adding
    a built-in (web_search, code_interpreter, file_search,
    image_generation, MCP) to the request."""
    import asyncio

    # Stub out the keyring resolver so we don't need a real OpenAI key.
    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    # Make AsyncOpenAI inside the lazy import resolve to our fake.
    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _FakeAsyncOpenAI, raising=True)

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("hello"):
            pass

    asyncio.run(_drive())

    from sift.integration_core import (
        MODEL_REQUEST_TIMEOUT_SECONDS,
        MODEL_SDK_MAX_RETRIES,
    )

    assert sess._client.options == {  # type: ignore[union-attr]
        "timeout": MODEL_REQUEST_TIMEOUT_SECONDS,
        "max_retries": MODEL_SDK_MAX_RETRIES,
    }
    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 1, "expected exactly one Responses-API call"
    call = api.calls[0]

    # Lockdown assertions. Floor of 6 (the original locked surface)
    # plus exact match against the canonical spec count, so a future
    # PR adding a tool flows through naturally but a duplicate or a
    # stray built-in does not.
    tools = call.get("tools")
    assert tools is not None and len(tools) >= 6
    assert len(tools) == len(build_tool_specs())
    assert all(t.get("type") == "function" for t in tools)
    sent_names = {t["name"] for t in tools}
    expected = {s.name for s in build_tool_specs()}
    assert sent_names == expected
    assert not any(t.get("type") in FORBIDDEN_BUILTIN_TYPES for t in tools)

    # Privacy-first default: no server-side application state. Opaque
    # encrypted reasoning items are requested so Sift can replay context
    # locally without reading the model's hidden reasoning.
    assert call.get("store") is False
    assert call.get("include") == ["reasoning.encrypted_content"]


# ---------------------------------------------------------------------------
# Local replay and server-storage refusal
# ---------------------------------------------------------------------------
#
# Every request must use ``store=false`` and replay committed conversation
# items locally. The retired opt-in environment variable cannot weaken it.


class _ScriptedResponse:
    """Drop-in for ``_FakeResponse`` that lets the test specify the
    response id and whether the round emits a function_call (loop
    continues) vs. a plain message (loop exits)."""

    def __init__(
        self, response_id: str, *, with_tool_call: bool,
        tool_arguments: str = "{}",
    ) -> None:
        self.id = response_id
        self.usage = _FakeUsage()
        if with_tool_call:
            arguments_value = tool_arguments

            class _Call:
                type = "function_call"
                name = "list_results"
                call_id = "call_xyz"
                arguments = arguments_value

                def model_dump(self) -> dict[str, Any]:
                    return {
                        "type": "function_call",
                        "name": "list_results",
                        "call_id": "call_xyz",
                        "arguments": arguments_value,
                    }

            self.output = [_Call()]
        else:
            class _Block:
                type = "output_text"
                text = "ok"

            class _Msg:
                type = "message"
                content = [_Block()]

                def model_dump(self) -> dict[str, Any]:
                    return {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }

            self.output = [_Msg()]


class _ScriptedResponsesAPI:
    """Returns a queue of pre-built responses in order, capturing the
    kwargs of each call for assertions."""

    def __init__(self, responses: list[_ScriptedResponse]) -> None:
        self._queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> _ScriptedResponse:
        self.calls.append(kwargs)
        if not self._queue:
            raise RuntimeError("test queue exhausted")
        return self._queue.pop(0)


class _ScriptedAsyncOpenAI:
    def __init__(
        self, api_key: str | None = None, *, responses: list[_ScriptedResponse],
        **options: Any,
    ) -> None:
        self.api_key = api_key
        self.options = options
        self.responses = _ScriptedResponsesAPI(responses)

    async def close(self) -> None:
        return None


def test_server_storage_env_cannot_override_local_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Even the retired opt-in variable cannot enable server storage."""
    import asyncio

    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_OPENAI_STORE_RESPONSES", "1")
    # Disable the first-turn few-shot so this test isolates local replay.
    monkeypatch.setenv("SIFT_DISABLE_FEWSHOT", "1")

    # Three scripted responses driving two turns:
    scripted = [
        _ScriptedResponse("resp_1", with_tool_call=True),    # turn 1 r1
        _ScriptedResponse("resp_2", with_tool_call=False),   # turn 1 r2
        _ScriptedResponse("resp_3", with_tool_call=False),   # turn 2 r1
    ]

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options
        ),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("turn one"):
            pass
        async for _ in sess.send("turn two"):
            pass

    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 3, (
        f"expected 3 round-trips (2 in turn 1, 1 in turn 2); got {len(api.calls)}"
    )

    c1, c2, c3 = api.calls

    # Call 1: fresh chain, no prior id.
    assert "previous_response_id" not in c1, (
        "first call of a fresh session must NOT carry previous_response_id"
    )
    assert len(c1["input"]) == 1
    assert c1["input"][0]["role"] == "user"

    assert all(c["store"] is False for c in (c1, c2, c3))
    assert all("previous_response_id" not in c for c in (c1, c2, c3))
    assert [item.get("type") for item in c2["input"]] == [
        None, "function_call", "function_call_output",
    ]
    assert c3["input"][-1]["role"] == "user"


def test_default_replays_context_locally_without_response_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Default mode carries the same tool/conversation context as explicit
    response chaining, but every request is store=false and has no server
    response pointer."""
    import asyncio

    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.delenv("SIFT_OPENAI_STORE_RESPONSES", raising=False)
    monkeypatch.setenv("SIFT_DISABLE_FEWSHOT", "1")
    scripted = [
        _ScriptedResponse("resp_1", with_tool_call=True),
        _ScriptedResponse("resp_2", with_tool_call=False),
        _ScriptedResponse("resp_3", with_tool_call=False),
    ]
    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options
        ),
        raising=True,
    )
    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("turn one"):
            pass
        async for _ in sess.send("turn two"):
            pass

    asyncio.run(_drive())
    calls = sess._client.responses.calls  # type: ignore[union-attr]
    assert len(calls) == 3
    assert all(c["store"] is False for c in calls)
    assert all("previous_response_id" not in c for c in calls)
    assert all(c["include"] == ["reasoning.encrypted_content"] for c in calls)
    assert len(calls[0]["input"]) == 1
    assert [item.get("type") for item in calls[1]["input"]] == [
        None, "function_call", "function_call_output",
    ]
    # Successful turn 1 is committed locally and replayed before turn 2.
    assert len(calls[2]["input"]) == 5
    assert calls[2]["input"][-1]["role"] == "user"


def test_first_turn_prepends_fewshot_demonstration_then_user_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Round 1 of a fresh session prepends a 4-item few-shot exchange
    (user, function_call, function_call_output, assistant) BEFORE the
    real user message. This is the structural table-preference nudge
    that biases the model toward pasting markdown payloads verbatim
    rather than re-narrating numbers. Subsequent turns ride the
    demonstration via local replay without duplicating it.

    The companion storage-refusal test
    (``test_server_storage_env_cannot_override_local_replay``)
    sets SIFT_DISABLE_FEWSHOT=1 to isolate replay from this prepend;
    here we leave the env unset so the
    prepend fires and pin its shape."""
    import asyncio

    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_OPENAI_STORE_RESPONSES", "1")

    scripted = [
        _ScriptedResponse("resp_1", with_tool_call=False),   # turn 1 r1
        _ScriptedResponse("resp_2", with_tool_call=False),   # turn 2 r1
    ]

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options
        ),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("turn one"):
            pass
        async for _ in sess.send("turn two"):
            pass

    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 2
    c1, c2 = api.calls

    # Round 1: few-shot prefix + real user. Item types in order:
    # user (few-shot Q), function_call, function_call_output,
    # message (few-shot assistant), user (real).
    inp = c1["input"]
    assert len(inp) == 5, (
        f"round 1 should carry 4 few-shot items + 1 real user; got {len(inp)}"
    )
    assert inp[0].get("role") == "user"
    assert inp[1].get("type") == "function_call"
    assert inp[1].get("name") == "submit_script"
    assert inp[2].get("type") == "function_call_output"
    assert inp[3].get("type") == "message"
    assert inp[3].get("role") == "assistant"
    # The real user message is last and carries the prompt verbatim.
    assert inp[4].get("role") == "user"
    real_user_text = inp[4]["content"][0]["text"]
    assert real_user_text == "turn one"

    # Turn 2 replays locally; few-shot does not get appended twice.
    assert "previous_response_id" not in c2
    assert c2["store"] is False
    assert len(c2["input"]) == 7
    assert c2["input"][-1]["role"] == "user"


def test_disable_fewshot_env_var_skips_prepend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``SIFT_DISABLE_FEWSHOT=1`` is the documented A/B opt-out. With
    it set, round 1 carries only the real user message. Used by the
    chain-pointer lockdown test above and available to researchers
    who want to measure the few-shot's behavioral effect."""
    import asyncio

    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_DISABLE_FEWSHOT", "1")

    scripted = [_ScriptedResponse("resp_1", with_tool_call=False)]

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options
        ),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("turn one"):
            pass

    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 1
    inp = api.calls[0]["input"]
    assert len(inp) == 1
    assert inp[0]["role"] == "user"
    assert inp[0]["content"][0]["text"] == "turn one"


def test_request_failure_does_not_advance_committed_local_history(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A failed turn cannot partially enter locally replayed history."""
    import asyncio

    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_OPENAI_STORE_RESPONSES", "1")

    class _FailingResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []
            self._call_count = 0
            self._first_response = _ScriptedResponse("resp_good", with_tool_call=False)

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            self._call_count += 1
            if self._call_count == 1:
                return self._first_response
            # Second turn fails; committed local history must remain.
            raise RuntimeError("server returned 500: simulated failure")

    class _FlakyAsyncOpenAI:
        def __init__(self, api_key: str | None = None, **options: Any) -> None:
            self.api_key = api_key
            self.options = options
            self.responses = _FailingResponsesAPI()

        async def close(self) -> None:
            return None

    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "AsyncOpenAI", _FlakyAsyncOpenAI, raising=True)

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("first"):
            pass
        # Second send fails; just drain the events.
        async for _ in sess.send("second"):
            pass

    asyncio.run(_drive())

    assert len(sess._history_items) == 6
    assert sess._history_items[-1].get("type") == "message"


@pytest.mark.parametrize(
    ("status", "detail_attr", "detail_value"),
    [
        ("incomplete", "incomplete_details", "max_output_tokens"),
        ("failed", "error", "provider execution failed"),
        ("cancelled", "error", "request cancelled"),
    ],
)
def test_terminal_response_status_is_not_committed_or_reported_done(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status: str,
    detail_attr: str,
    detail_value: str,
) -> None:
    """A successful HTTP exchange is not necessarily a completed model
    response. Failed terminal states may expose partial text, but they
    cannot advance conversation history, dispatch tools, or emit done."""
    import asyncio
    from types import SimpleNamespace

    from sift.provider import openai as openai_provider
    from sift.provider.base import AssistantText, TurnDone, TurnError

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.delenv("SIFT_OPENAI_STORE_RESPONSES", raising=False)
    monkeypatch.setenv("SIFT_DISABLE_FEWSHOT", "1")

    response = _ScriptedResponse("resp_bad", with_tool_call=False)
    response.status = status
    if detail_attr == "incomplete_details":
        response.incomplete_details = SimpleNamespace(reason=detail_value)
    else:
        response.error = SimpleNamespace(message=detail_value)

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg,
        "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=[response], **options,
        ),
        raising=True,
    )
    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="you are sift",
    )

    async def _drive() -> list[Any]:
        return [event async for event in sess.send("hello")]

    events = asyncio.run(_drive())
    assert any(isinstance(event, AssistantText) for event in events)
    errors = [event for event in events if isinstance(event, TurnError)]
    assert len(errors) == 1
    assert status in errors[0].message
    assert detail_value in errors[0].message
    assert not any(isinstance(event, TurnDone) for event in events)
    assert sess._history_items == []


def test_failed_response_does_not_dispatch_returned_tool_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Even a malformed failed response containing a function call must
    stop at the provider boundary; running it could mutate local data."""
    import asyncio
    from types import SimpleNamespace

    from sift.provider import openai as openai_provider
    from sift.provider.base import ToolCallResult, TurnError

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.delenv("SIFT_OPENAI_STORE_RESPONSES", raising=False)
    monkeypatch.setenv("SIFT_DISABLE_FEWSHOT", "1")
    response = _ScriptedResponse("resp_bad", with_tool_call=True)
    response.status = "failed"
    response.error = SimpleNamespace(message="generation failed")

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg,
        "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=[response], **options,
        ),
        raising=True,
    )
    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="you are sift",
    )

    async def _drive() -> list[Any]:
        return [event async for event in sess.send("hello")]

    events = asyncio.run(_drive())
    assert any(isinstance(event, TurnError) for event in events)
    assert not any(isinstance(event, ToolCallResult) for event in events)
    assert sess._history_items == []


def test_response_not_found_cannot_create_a_server_chain_reset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A response-shaped 404 is ordinary failure in local-only mode."""
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.provider.base import TurnError
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_OPENAI_STORE_RESPONSES", "1")

    class _ExpiringResponsesAPI:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            self.calls.append(kwargs)
            raise RuntimeError(
                "openai.NotFoundError: previous_response_id "
                "'resp_expired' not found (404)"
            )

    class _ExpiringAsyncOpenAI:
        def __init__(self, api_key: str | None = None, **options: Any) -> None:
            self.api_key = api_key
            self.options = options
            self.responses = _ExpiringResponsesAPI()

        async def close(self) -> None:
            return None

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI", _ExpiringAsyncOpenAI, raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )
    await_open = sess.open()
    asyncio.run(await_open)

    events: list[Any] = []

    async def _drive() -> None:
        async for ev in sess.send("continue the analysis"):
            events.append(ev)

    asyncio.run(_drive())

    errors = [e for e in events if isinstance(e, TurnError)]
    assert len(errors) == 1, (
        f"expected exactly one TurnError; got {len(errors)}"
    )
    err = errors[0]
    assert err.context_reset is False
    api = sess._client.responses  # type: ignore[union-attr]
    assert api.calls[0]["store"] is False
    assert "previous_response_id" not in api.calls[0]


def test_handler_exception_does_not_leak_message_to_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """When a tool handler raises, the OpenAI tool loop's catch-all
    fallback must NOT interpolate ``str(e)`` into the model-visible
    reason. The exception message can include parser excerpts,
    file paths, or raw data values that per-tool error handling
    would have redacted. Return only the exception class (a bounded
    identifier) plus a generic recovery hint; log full details
    locally.

    The sentinel text is constructed to look like the kind of
    content the per-tool redaction pass would scrub: a row of
    data with a name, dollar amount, and email. If any of those
    tokens reach the function_call_output the fix didn't hold.
    """
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.tools import HANDLERS
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_OPENAI_STORE_RESPONSES", "1")

    SENSITIVE = (
        "row 42: name=Jane Doe income=$487192 email=jane.doe@example.com"
    )

    async def _raising_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(SENSITIVE)

    # The scripted tool call names ``list_results``; route THAT handler
    # to our raising one for the test only. monkeypatch undoes the
    # change at teardown so other tests aren't affected.
    monkeypatch.setitem(HANDLERS, "list_results", _raising_handler)

    # Two scripted responses: round 1 emits the function_call; round 2
    # (after our raising handler) emits a clean message so the loop
    # exits and we can inspect what the provider sent back in
    # ``function_call_output``.
    scripted = [
        _ScriptedResponse("resp_call", with_tool_call=True),
        _ScriptedResponse("resp_done", with_tool_call=False),
    ]
    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options
        ),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("trigger the failing tool"):
            pass
    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 2, "expected exactly two round-trips"
    # The function_call_output that travels back to the model on
    # round 2 carries the tool result.
    round2 = api.calls[1]
    func_outputs = [
        item for item in round2["input"]
        if item.get("type") == "function_call_output"
        and item.get("call_id") == "call_xyz"
    ]
    assert func_outputs, "round 2 must carry the tool's output back"
    output_text = func_outputs[0]["output"]
    # Critical: the raw exception message must NOT appear.
    assert SENSITIVE not in output_text, (
        f"sensitive exception text leaked into model-visible tool "
        f"result: {output_text!r}"
    )
    assert "Jane Doe" not in output_text
    assert "jane.doe@example.com" not in output_text
    assert "$487192" not in output_text
    # The class name is a bounded identifier and stays — the model
    # can route on it without seeing the redacted detail.
    assert "RuntimeError" in output_text
    # Generic recovery hint surfaces.
    assert "Retry" in output_text or "fall back" in output_text


def test_handler_exception_diagnostic_gated_behind_debug_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression test for architecture-audit finding P: the LOCAL
    diagnostic logged when a tool handler raises (str(e) + full
    traceback, written to <cwd>/.sift-usage.log) carries exactly the
    kind of parser-excerpt / file-path / raw-data content the
    model-visible path above is careful never to leak -- but unlike
    every other diagnostic-logging path in this file (all gated
    behind SIFT_DEBUG_USAGE), this one used to write unconditionally,
    for every researcher, on every unexpected tool error. It must now
    stay out of the persistent log by default and only appear with
    the env var set.
    """
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.provider.usage_log import append_usage_line  # noqa: F401 (path sanity)
    from sift.tools import HANDLERS
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    SENSITIVE = "row 42: name=Jane Doe income=$487192"

    async def _raising_handler(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(SENSITIVE)

    monkeypatch.setitem(HANDLERS, "list_results", _raising_handler)

    def _run_once() -> str:
        scripted = [
            _ScriptedResponse("resp_call", with_tool_call=True),
            _ScriptedResponse("resp_done", with_tool_call=False),
        ]
        import openai as openai_pkg
        monkeypatch.setattr(
            openai_pkg, "AsyncOpenAI",
            lambda api_key=None, **options: _ScriptedAsyncOpenAI(
                api_key, responses=scripted, **options
            ),
            raising=True,
        )
        sess = OpenAISession(
            cwd=tmp_path, model="gpt-5.6-sol", system_prompt="you are sift",
        )

        async def _drive() -> None:
            async for _ in sess.send("trigger the failing tool"):
                pass
        asyncio.run(_drive())
        log_path = tmp_path / ".sift-usage.log"
        return log_path.read_text(encoding="utf-8") if log_path.is_file() else ""

    monkeypatch.delenv("SIFT_DEBUG_USAGE", raising=False)
    log_default = _run_once()
    assert SENSITIVE not in log_default, (
        "raw exception text reached .sift-usage.log without "
        "SIFT_DEBUG_USAGE set"
    )
    assert "RuntimeError" in log_default
    assert "SIFT_DEBUG_USAGE" in log_default

    (tmp_path / ".sift-usage.log").unlink(missing_ok=True)
    monkeypatch.setenv("SIFT_DEBUG_USAGE", "1")
    log_debug = _run_once()
    assert SENSITIVE in log_debug, (
        "full diagnostic should appear once SIFT_DEBUG_USAGE=1 is set"
    )


def test_send_yields_turnerror_when_tool_loop_does_not_converge(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A model that emits a function_call on every one of
    MAX_TOOL_ROUNDS rounds must produce a TurnError, not TurnDone.

    Before the fix, the for-loop's natural exhaustion fell through to
    the same TurnDone branch as a clean exit, so a runaway tool-using
    model looked indistinguishable from a normal completion to the UI.
    """
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.provider.base import TurnDone, TurnError
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    # Sixteen scripted responses, each requesting a function_call. This
    # is exactly the number range(MAX_TOOL_ROUNDS) gives us, so the
    # loop exhausts on iteration 16 without ever seeing a plain message
    # — the path the old code mishandled.
    scripted = [
        _ScriptedResponse(f"resp_{i}", with_tool_call=True)
        for i in range(16)
    ]

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options
        ),
        raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )
    prior_history = [{"type": "message", "role": "assistant", "content": []}]
    sess._history_items = list(prior_history)

    events: list[Any] = []

    async def _drive() -> None:
        async for ev in sess.send("loop forever"):
            events.append(ev)

    asyncio.run(_drive())

    assert events, "send() yielded no events"
    assert not any(isinstance(e, TurnDone) for e in events), (
        "exhausted tool loop must NOT emit TurnDone; that's the silent "
        "truncation the fix prevents"
    )
    assert isinstance(events[-1], TurnError), (
        f"last event must be TurnError; got {type(events[-1]).__name__}"
    )
    # Loop ran exactly MAX_TOOL_ROUNDS times before giving up — no
    # short-circuit, no overrun.
    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 16
    assert sess._history_items == prior_history, (
        "exhausted tool loop must not commit a partial local conversation"
    )


# ---------------------------------------------------------------------------
# Missing API key surfaces as AuthFailure, not RuntimeError
# ---------------------------------------------------------------------------

def test_send_with_no_api_key_yields_auth_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``SessionRunner.ensure_session`` awaits ``OpenAISession.open()``
    BEFORE any event stream exists. If ``open()`` raises on a missing
    key, the runner translates the RuntimeError into a generic
    ``turn_error`` — diverging from the Anthropic provider, whose
    open() succeeds and whose send() yields AuthFailure on the 401.

    Verify the deferred path: open() with no key no-ops, and the
    first ``send()`` round yields ``AuthFailure`` so the runner emits
    the same provider-neutral ``auth_failure`` event the API-call
    path produces for 401 responses."""
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.provider.base import AuthFailure

    # No keyring credential, no env key. _resolve_api_key returns None.
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: None)

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    # open() must NOT raise — that was the pre-fix behaviour that
    # short-circuited the auth_failure event path.
    asyncio.run(sess.open())
    assert sess._client is None, (
        "open() with no key must defer client construction"
    )

    events: list[Any] = []

    async def _drive() -> None:
        async for ev in sess.send("hi"):
            events.append(ev)

    asyncio.run(_drive())

    assert events, "send() must yield at least one event on missing key"
    assert isinstance(events[0], AuthFailure), (
        f"first event must be AuthFailure; got "
        f"{type(events[0]).__name__}. Without this the runner emits a "
        f"generic turn_error and the UI never enters the auth-failure "
        f"flow that prompts the researcher to fix credentials."
    )
    assert "OpenAI API key" in events[0].reason


def test_runner_ensure_session_does_not_raise_on_missing_openai_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """End-to-end: the runner's ``ensure_session`` must not blow up
    on a missing OpenAI key. The translation happens at send-time,
    not at open-time. Mirrors the Anthropic posture (whose
    ClaudeSDKClient construction also doesn't fail on missing
    credentials)."""
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.runner import SessionRunner

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: None)

    runner = SessionRunner(
        cwd=tmp_path,
        provider="openai",
        model="gpt-5.6-sol",
    )

    # ensure_session must succeed and return a ProviderSession whose
    # ._client is None; the runner's turn loop will then iterate
    # send() and receive AuthFailure.
    sess = asyncio.run(runner.ensure_session())
    assert sess is not None
    assert sess._client is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# parallel_tool_calls is pinned explicitly
# ---------------------------------------------------------------------------

def test_request_kwargs_pin_parallel_tool_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The OpenAI docstring claims ``parallel_tool_calls`` is on. The
    request payload must SET it explicitly so a future SDK/API default
    change can't silently flip the behaviour. Pin the field at the
    request boundary so a regression is caught by this test rather
    than by surprised researchers."""
    import asyncio

    from sift.provider import openai as openai_provider
    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")

    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg, "AsyncOpenAI", _FakeAsyncOpenAI, raising=True,
    )

    sess = OpenAISession(
        cwd=tmp_path,
        model="gpt-5.6-sol",
        system_prompt="you are sift",
    )

    async def _drive() -> None:
        async for _ in sess.send("hello"):
            pass

    asyncio.run(_drive())

    api = sess._client.responses  # type: ignore[union-attr]
    assert len(api.calls) == 1
    call = api.calls[0]
    assert call.get("parallel_tool_calls") is True, (
        "parallel_tool_calls must be pinned to True on every request; "
        "without this the model loses concurrent tool calls if the "
        "SDK/API default ever flips"
    )


# ---------------------------------------------------------------------------
# OpenAI tool description for read_attached_file warns away from images
# ---------------------------------------------------------------------------

def test_read_attached_file_openai_description_warns_about_images() -> None:
    """The canonical Anthropic description for ``read_attached_file``
    says images come back as a vision content block. On OpenAI that
    contract is impossible — function_call_output is text-only, and
    the response is rewritten to ``image_not_supported_on_provider``.
    The OpenAI-specific description must spell this out so the model
    doesn't waste a tool call inviting the rewrite path.
    """
    from sift.provider.tool_schemas import build_tool_specs

    specs = {s.name: s for s in build_tool_specs()}
    spec = specs["read_attached_file"]
    assert spec.openai_description is not None, (
        "read_attached_file must have a provider-specific OpenAI "
        "description so the image-not-supported guidance reaches the "
        "model"
    )
    desc = spec.openai_description
    # Must definitively say image bytes can't return.
    assert "image bytes" in desc.lower() or "cannot" in desc.lower()
    # Must point at the recovery path (re-@mention via user message).
    assert "@mention" in desc, (
        "OpenAI description must direct the model to the @mention "
        "recovery path so the researcher can re-send the file as "
        "user-side vision input"
    )
    assert "vision content block" not in desc


@pytest.mark.parametrize(
    "status,message,expected",
    [
        (429, "quota exceeded", "retry"),
        (404, "model not found", "retired"),
        (503, "service unavailable", "transient"),
        (400, "maximum context length exceeded", "context"),
    ],
)
def test_openai_failures_have_common_actionable_categories(
    status: int, message: str, expected: str,
) -> None:
    from sift.provider.base import TurnError
    from sift.provider.openai import _translate_request_failure

    class _StatusError(Exception):
        def __init__(self) -> None:
            super().__init__(message)
            self.status_code = status

    event = _translate_request_failure(
        _StatusError(), model_id="gpt-5.6-sol", safe_message=message,
    )
    assert isinstance(event, TurnError)
    assert expected in event.message.lower()


def test_openai_timeout_is_not_automatically_retried() -> None:
    from sift.provider.base import TurnError
    from sift.provider.openai import _translate_request_failure

    class APITimeoutError(Exception):
        pass

    event = _translate_request_failure(
        APITimeoutError("timed out"),
        model_id="gpt-5.6-sol",
        safe_message="timed out",
    )
    assert isinstance(event, TurnError)
    assert "not retried automatically" in event.message


def test_malformed_openai_tool_arguments_return_error_without_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import asyncio

    from sift.provider import openai as openai_provider
    from sift.provider.base import ToolCallResult, TurnDone

    monkeypatch.setattr(openai_provider, "_resolve_api_key", lambda: "sk-test")
    monkeypatch.setenv("SIFT_DISABLE_FEWSHOT", "1")
    scripted = [
        _ScriptedResponse(
            "bad", with_tool_call=True, tool_arguments="{not valid json",
        ),
        _ScriptedResponse("done", with_tool_call=False),
    ]
    import openai as openai_pkg
    monkeypatch.setattr(
        openai_pkg,
        "AsyncOpenAI",
        lambda api_key=None, **options: _ScriptedAsyncOpenAI(
            api_key, responses=scripted, **options,
        ),
        raising=True,
    )
    sess = OpenAISession(
        cwd=tmp_path, model="gpt-5.6-sol", system_prompt="system",
    )

    async def drive() -> list[Any]:
        return [event async for event in sess.send("go")]

    events = asyncio.run(drive())
    results = [event for event in events if isinstance(event, ToolCallResult)]
    assert len(results) == 1
    assert results[0].is_error is True
    assert "not valid JSON" in results[0].text
    assert isinstance(events[-1], TurnDone)
