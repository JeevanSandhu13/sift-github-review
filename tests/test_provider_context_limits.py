from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from sift.provider import TurnDone
from sift.provider.context_limits import (
    ContextBudgetExceeded,
    conservative_text_tokens,
    enforce_context_budget,
)
from sift.runner import SessionRunner


def test_estimate_is_utf8_safe_and_context_rejects_before_provider() -> None:
    assert conservative_text_tokens("é") >= 2
    with pytest.raises(ContextBudgetExceeded, match="cannot fit"):
        enforce_context_budget(
            model_id="openai-compatible-custom",
            provider="openai_compatible",
            occupied_tokens=31_000,
            prompt="x" * 100,
        )


class _CountingSession:
    def __init__(self, post_turn_tokens: int = 100) -> None:
        self.calls = 0
        self.post_turn_tokens = post_turn_tokens

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def send(self, prompt: str, images: Any = None) -> AsyncIterator[Any]:
        self.calls += 1
        yield TurnDone(
            input_tokens=80,
            output_tokens=20,
            post_turn_tokens=self.post_turn_tokens,
        )


def _drive(runner: SessionRunner, text: str, turn_id: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    asyncio.run(runner.run_turn(
        text,
        images=None,
        on_event=events.append,
        build_context_prefix=lambda cwd: "",
        build_script_prefix=lambda attachments, cwd: "",
        turn_id=turn_id,
    ))
    return events


def test_runner_tracks_clean_provider_context_and_blocks_oversize_turn(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.setenv("SIFT_OPENAI_COMPATIBLE_CONTEXT_WINDOW", "32000")
    runner = SessionRunner(
        cwd=tmp_path,
        provider="openai_compatible",
        model="openai-compatible-custom",
    )
    session = _CountingSession(post_turn_tokens=30_000)
    runner._session = session
    assert any(e["type"] == "turn_done" for e in _drive(runner, "ok", "t1"))
    assert runner._last_context_tokens == 30_000

    events = _drive(runner, "x" * 100, "t2")
    assert session.calls == 1
    assert any(
        e["type"] == "turn_error" and "cannot fit" in e["message"]
        for e in events
    )


def test_failed_or_unknown_usage_does_not_erase_last_exact_context(
    tmp_path: Path,
) -> None:
    runner = SessionRunner(
        cwd=tmp_path, provider="gemini", model="gemini-test",
    )
    runner._last_context_tokens = 777
    session = _CountingSession(post_turn_tokens=900)
    runner._session = session
    _drive(runner, "ok", "t1")
    assert runner._last_context_tokens == 900
