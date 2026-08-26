from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import pytest

from sift.provider import TurnError
from sift.provider.anthropic import AnthropicSession, _image_message_iter
from sift.provider.gemini import GeminiSession, _build_user_parts
from sift.provider.openai import OpenAISession, _build_user_content
from sift.provider.openai_compatible import (
    OpenAICompatibleSession,
    _build_user_message,
)

PNG_DATA = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
IMAGE = {"mime": "image/png", "data": PNG_DATA}


def test_every_provider_encodes_the_same_explicit_png() -> None:
    openai = _build_user_content("look", [IMAGE])
    assert openai[1]["type"] == "input_image"
    assert PNG_DATA in openai[1]["image_url"]

    compatible = _build_user_message("look", [IMAGE])
    assert PNG_DATA in compatible["content"][1]["image_url"]["url"]

    gemini = _build_user_parts("look", [IMAGE])
    assert gemini[1].inline_data.data == base64.b64decode(PNG_DATA)

    async def anthropic_message() -> dict[str, Any]:
        return await anext(_image_message_iter("look", [IMAGE]))

    anthropic = asyncio.run(anthropic_message())
    source = anthropic["message"]["content"][1]["source"]
    assert source["media_type"] == "image/png"
    assert source["data"] == PNG_DATA


@pytest.mark.parametrize(
    "provider",
    ["openai", "anthropic", "gemini", "openai_compatible"],
)
def test_every_provider_rejects_non_image_before_sdk_use(
    tmp_path: Path, provider: str,
) -> None:
    sessions: dict[str, Any] = {
        "openai": OpenAISession(tmp_path, "gpt-5.6-sol", "system"),
        "anthropic": AnthropicSession(
            tmp_path, "claude-sonnet-5[1m]", "system",
        ),
        "gemini": GeminiSession(tmp_path, "gemini-3.7-flash", "system"),
        "openai_compatible": OpenAICompatibleSession(
            tmp_path, "openai-compatible-custom", "system",
        ),
    }
    session = sessions[provider]
    # A truthy sentinel prevents lazy auth construction. If validation drifted
    # after SDK use this object would fail loudly on attribute access.
    session._client = object()
    if provider == "gemini":
        session._chat = object()

    async def drive() -> list[Any]:
        return [event async for event in session.send(
            "look", images=[{
                "mime": "application/pdf",
                "data": base64.b64encode(b"%PDF-1.7").decode("ascii"),
            }],
        )]

    events = asyncio.run(drive())
    assert len(events) == 1
    assert isinstance(events[0], TurnError)
    assert "unsupported type" in events[0].message
