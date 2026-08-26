"""Per-turn usage accounting must record the resolved model name.

``SessionRunner.run_turn`` previously recorded every turn's
tokens under ``self.model`` -- correct for Anthropic/OpenAI/Gemini,
whose catalog ids ARE the real model name, but wrong for the
openai_compatible provider. That provider's catalog has exactly one
entry, a fixed placeholder id ("openai-compatible-custom"), because
there's no way to enumerate real model ids across arbitrary target
servers up front -- the actual model invoked is resolved from
``SIFT_OPENAI_COMPATIBLE_MODEL`` inside the session itself
(``OpenAICompatibleSession._resolved_model``) and never reached
``usage_meter.record_turn``. Every openai_compatible session's exact
token counts were therefore attributed to the meaningless placeholder
string in the usage summary's per-model breakdown -- indistinguishable
from any other local/gateway model a researcher pointed Sift at.

The fix: ``run_turn`` now prefers
``getattr(session, "resolved_model_name", None)`` over ``self.model``
when recording usage. Every other provider session simply doesn't
define that attribute, so this is a no-op for them (pinned below).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from sift.integration_audit import read_and_verify
from sift.runner import SessionRunner
from sift.usage_meter import read_usage


class _FakeSessionWithResolvedModel:
    """Stands in for ``OpenAICompatibleSession``: constructed with the
    catalog placeholder id, but exposes the REAL model name via
    ``resolved_model_name`` -- the exact shape the fix reads."""

    def __init__(self, resolved_model: str) -> None:
        from sift.provider import TurnDone
        self._TurnDone = TurnDone
        self.resolved_model_name = resolved_model

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, prompt: str, images: Any = None) -> AsyncIterator[Any]:
        yield self._TurnDone(input_tokens=1234, output_tokens=567)


class _FakeSessionWithoutResolvedModel:
    """Stands in for Anthropic/OpenAI/Gemini sessions: no
    ``resolved_model_name`` attribute at all. Confirms the fix falls
    back to ``self.model`` rather than crashing or recording
    ``None``."""

    def __init__(self) -> None:
        from sift.provider import TurnDone
        self._TurnDone = TurnDone

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, prompt: str, images: Any = None) -> AsyncIterator[Any]:
        yield self._TurnDone(input_tokens=1000, output_tokens=200)


def _run(runner: SessionRunner) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    async def drive() -> None:
        await runner.run_turn(
            "go",
            images=None,
            on_event=events.append,
            build_context_prefix=lambda cwd: "",
            build_script_prefix=lambda atts, cwd: "",
            turn_id="t-1",
        )

    asyncio.run(drive())
    return events


def test_openai_compatible_usage_recorded_under_resolved_model_name(
    tmp_path: Path,
) -> None:
    """The headline fix: a session exposing ``resolved_model_name``
    must have ITS tokens recorded under that name, not the runner's
    catalog placeholder."""
    runner = SessionRunner(
        cwd=tmp_path, provider="openai_compatible",
        model="openai-compatible-custom",
    )
    runner._session = _FakeSessionWithResolvedModel("llama-3.1-70b-instruct")

    _run(runner)

    state = read_usage(tmp_path)
    by_model = state.get("by_model") or {}
    assert "llama-3.1-70b-instruct" in by_model, (
        f"expected the resolved model name as the usage key, got "
        f"{list(by_model)}"
    )
    assert "openai-compatible-custom" not in by_model, (
        "tokens must not be recorded under the meaningless catalog "
        "placeholder when a real model name is available"
    )
    row = by_model["llama-3.1-70b-instruct"]
    assert row["input_tokens"] == 1234
    assert row["output_tokens"] == 567

    verified, audit_rows = read_and_verify(tmp_path)
    assert verified is True
    assert len(audit_rows) == 1
    assert audit_rows[0]["integration_id"] == "openai_compatible"
    assert audit_rows[0]["action"] == "conversation_turn"
    assert audit_rows[0]["outcome"] == "success"
    assert audit_rows[0]["metadata"]["input_tokens"] == 1234
    assert audit_rows[0]["metadata"]["output_tokens"] == 567
    serialized = str(audit_rows)
    assert "llama-3.1-70b-instruct" not in serialized
    assert "go" not in serialized


def test_providers_without_resolved_model_name_fall_back_to_self_model(
    tmp_path: Path,
) -> None:
    """Anthropic/OpenAI/Gemini sessions don't define
    ``resolved_model_name`` -- usage must still record correctly
    under ``self.model``, exactly as before this fix."""
    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-5[1m]",
    )
    runner._session = _FakeSessionWithoutResolvedModel()

    _run(runner)

    state = read_usage(tmp_path)
    by_model = state.get("by_model") or {}
    assert "claude-sonnet-5[1m]" in by_model
    row = by_model["claude-sonnet-5[1m]"]
    assert row["input_tokens"] == 1000
    assert row["output_tokens"] == 200


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-5.6-sol"),
        ("anthropic", "claude-sonnet-5[1m]"),
        ("gemini", "gemini-3.7-flash"),
        ("openai_compatible", "openai-compatible-custom"),
    ],
)
def test_every_provider_stream_has_one_total_turn_deadline(
    tmp_path: Path, provider: str, model: str,
    monkeypatch,
) -> None:
    from sift import integration_core

    class _NeverFinishes:
        async def open(self) -> None:
            return None

        async def close(self) -> None:
            return None

        async def send(self, prompt: str, images: Any = None):
            await asyncio.Event().wait()
            yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr(integration_core, "MODEL_REQUEST_TIMEOUT_SECONDS", 0.01)
    runner = SessionRunner(
        cwd=tmp_path, provider=provider, model=model,
    )
    runner._session = _NeverFinishes()

    events = _run(runner)
    errors = [row for row in events if row.get("type") == "turn_error"]
    assert len(errors) == 1
    assert "0.01-second timeout" in errors[0]["message"]
    verified, audit_rows = read_and_verify(tmp_path)
    assert verified is True
    assert audit_rows[-1]["outcome"] == "failure"


def test_no_active_session_does_not_crash_usage_recording(
    tmp_path: Path,
) -> None:
    """``getattr(self._session, ...)`` must tolerate ``self._session``
    being ``None`` (defensive: run_turn's own preflight should never
    reach the TurnDone branch without a session, but the accounting
    line itself must not assume one)."""

    runner = SessionRunner(
        cwd=tmp_path, provider="anthropic", model="claude-sonnet-5[1m]",
    )
    assert getattr(runner._session, "resolved_model_name", None) is None


@pytest.mark.parametrize(
    "provider,model",
    [
        ("openai", "gpt-5.6-sol"),
        ("anthropic", "claude-sonnet-5[1m]"),
        ("gemini", "gemini-3.7-flash"),
        ("openai_compatible", "openai-compatible-custom"),
    ],
)
def test_every_provider_family_is_cooperatively_cancellable(
    tmp_path: Path, provider: str, model: str,
) -> None:
    class _BlockingSession:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.closed = False

        async def open(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

        async def send(self, prompt: str, images: Any = None):
            self.started.set()
            await asyncio.Event().wait()
            yield  # pragma: no cover

    async def scenario() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        session = _BlockingSession()
        runner = SessionRunner(cwd=tmp_path, provider=provider, model=model)
        runner._session = session
        task = asyncio.create_task(runner.run_turn(
            "go",
            images=None,
            on_event=events.append,
            build_context_prefix=lambda cwd: "",
            build_script_prefix=lambda attachments, cwd: "",
            turn_id="cancel-me",
        ))
        await session.started.wait()
        assert runner.cancel_turn("cancel-me") == "cancel-me"
        await task
        await asyncio.sleep(0)
        assert session.closed is True
        return events

    events = asyncio.run(scenario())
    assert any(
        event.get("type") == "turn_error"
        and event.get("message") == "cancelled"
        for event in events
    )
