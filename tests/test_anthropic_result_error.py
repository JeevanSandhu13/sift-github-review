"""Anthropic provider — ResultMessage error signal must reach the caller.

Regression coverage for architecture-audit finding O: AnthropicSession.
send() read ResultMessage for usage/cost accounting but never checked
its own is_error field, so a turn the SDK itself reported as failed
(rate limit surfaced mid-stream, "error_during_execution" subtype,
etc.) still produced a plain TurnDone -- the researcher saw no error,
and usage_meter recorded accounting for a turn that may not have
produced anything real.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, ToolUseBlock

from sift.provider.anthropic import AnthropicSession
from sift.provider.base import TurnDone, TurnError


class _FakeClient:
    """Stand-in for claude_agent_sdk.ClaudeSDKClient: query() is a
    no-op, receive_response() yields exactly the messages the test
    hands it."""

    def __init__(self, messages: list) -> None:
        self._messages = messages
        self.queried_with: list = []

    async def query(self, prompt) -> None:
        self.queried_with.append(prompt)

    async def receive_response(self):
        for m in self._messages:
            yield m


def _result_message(*, is_error: bool, subtype: str = "success",
                    result: str | None = None) -> ResultMessage:
    return ResultMessage(
        subtype=subtype,
        duration_ms=100,
        duration_api_ms=90,
        is_error=is_error,
        num_turns=1,
        session_id="s1",
        usage={"input_tokens": 10, "output_tokens": 5},
        total_cost_usd=0.01,
        result=result,
    )


def _drive(sess: AnthropicSession) -> list:
    events = []

    async def _run() -> None:
        async for evt in sess.send("hello"):
            events.append(evt)

    asyncio.run(_run())
    return events


def test_result_message_is_error_yields_turn_error(tmp_path: Path) -> None:
    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _FakeClient([
        _result_message(
            is_error=True, subtype="error_during_execution",
            result="the model overloaded",
        ),
    ])

    events = _drive(sess)

    assert not any(isinstance(e, TurnDone) for e in events), (
        "an is_error=True ResultMessage must never produce a TurnDone"
    )
    errors = [e for e in events if isinstance(e, TurnError)]
    assert len(errors) == 1
    assert "error_during_execution" in errors[0].message
    assert "the model overloaded" in errors[0].message


def test_result_message_success_still_yields_turn_done(tmp_path: Path) -> None:
    """Sanity check the other direction: a normal, non-error result
    must be unaffected by this fix."""
    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _FakeClient([
        _result_message(is_error=False),
    ])

    events = _drive(sess)

    assert not any(isinstance(e, TurnError) for e in events)
    done = [e for e in events if isinstance(e, TurnDone)]
    assert len(done) == 1
    assert done[0].input_tokens == 10
    assert done[0].output_tokens == 5


def test_two_turns_reuse_one_agent_conversation(tmp_path: Path) -> None:
    client = _FakeClient([_result_message(is_error=False)])
    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = client
    assert isinstance(_drive(sess)[-1], TurnDone)
    assert isinstance(_drive(sess)[-1], TurnDone)
    assert len(client.queried_with) == 2
    assert sess._client is client


def test_stream_exception_resets_session_and_yields_terminal_error(
    tmp_path: Path,
) -> None:
    class _StreamFailureClient(_FakeClient):
        async def receive_response(self):
            if False:  # pragma: no cover - makes this an async generator
                yield None
            raise RuntimeError("stream disconnected")

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _StreamFailureClient([])
    events = _drive(sess)

    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert events[0].context_reset is True
    assert "stream disconnected" in events[0].message
    assert sess._client is None


def test_stream_without_result_is_error_not_silent_completion(
    tmp_path: Path,
) -> None:
    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _FakeClient([])
    events = _drive(sess)

    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert events[0].context_reset is True
    assert "without a terminal result" in events[0].message
    assert sess._client is None


def test_assistant_rate_limit_is_terminal_and_resets_session(
    tmp_path: Path,
) -> None:
    message = AssistantMessage(
        content=[], model="claude-opus-5", error="rate_limit",
    )
    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _FakeClient([message])
    events = _drive(sess)

    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert events[0].context_reset is True
    assert "rate_limit" in events[0].message
    assert "retry" in events[0].message.lower()
    assert sess._client is None


@pytest.mark.parametrize(
    "detail, expected",
    [
        ("429 rate_limit", "retry"),
        ("maximum context token window exceeded", "context"),
        ("404 model not found", "retired"),
    ],
)
def test_query_failures_have_category_specific_recovery(
    tmp_path: Path, detail: str, expected: str,
) -> None:
    class _QueryFailureClient(_FakeClient):
        async def query(self, prompt) -> None:
            raise RuntimeError(detail)

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _QueryFailureClient([])
    events = _drive(sess)
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert expected in events[0].message.lower()
    assert events[0].context_reset is True


def test_query_exception_redacts_api_key_and_resets_session(
    tmp_path: Path, monkeypatch,
) -> None:
    secret = "sk-ant-sensitive-test-key"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)

    class _QueryFailureClient(_FakeClient):
        async def query(self, prompt) -> None:
            raise RuntimeError(f"server error api_key={secret}")

    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _QueryFailureClient([])
    events = _drive(sess)

    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert secret not in events[0].message
    assert "***" in events[0].message
    assert events[0].context_reset is True
    assert sess._client is None


@pytest.mark.parametrize(
    "block",
    [
        ToolUseBlock(id="c1", name="Bash", input={}),
        ToolUseBlock(id="", name="mcp__sift__list_results", input={}),
        ToolUseBlock(
            id="c1", name="mcp__sift__list_results", input="bad",  # type: ignore[arg-type]
        ),
    ],
)
def test_malformed_or_unapproved_tool_use_fails_closed(
    tmp_path: Path, block: ToolUseBlock,
) -> None:
    sess = AnthropicSession(
        cwd=tmp_path, model="claude-opus-5", system_prompt="x",
    )
    sess._client = _FakeClient([
        AssistantMessage(content=[block], model="claude-opus-5"),
    ])
    events = _drive(sess)
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert "No tool was executed" in events[0].message
    assert events[0].context_reset is True
    assert sess._client is None
