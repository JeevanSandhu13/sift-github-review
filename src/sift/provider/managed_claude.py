"""Provider-neutral Claude tool loop for managed cloud deployments.

Bedrock Converse and AnthropicVertex have different wire shapes, but both are
driven through the same fail-closed local conversation and Sift-tool loop.
Subclasses translate one canonical Messages-like request/response at the SDK
boundary.  No hosted cloud tools are ever registered.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sift.provider.attachments import (
    AttachmentValidationError,
    validate_explicit_images,
)
from sift.provider.base import (
    AssistantText,
    AssistantThinking,
    AuthFailure,
    Event,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from sift.provider.openai import _extract_hints, _mcp_payload_to_text
from sift.provider.tool_schemas import build_tool_specs
from sift.tools import HANDLERS


@dataclass(frozen=True)
class ManagedClaudeResponse:
    content: tuple[dict[str, Any], ...]
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


def build_claude_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in build_tool_specs()
    ]


def verify_claude_lockdown(tools: list[dict[str, Any]]) -> None:
    expected = {spec.name for spec in build_tool_specs()}
    seen: list[str] = []
    for tool in tools:
        if set(tool) != {"name", "description", "input_schema"}:
            raise RuntimeError("Sift lockdown violation: managed Claude tool has extra fields")
        name = tool.get("name")
        if not isinstance(name, str) or name not in expected:
            raise RuntimeError(f"Sift lockdown violation: unknown managed Claude tool {name!r}")
        if not isinstance(tool.get("input_schema"), dict):
            raise TypeError(f"Sift lockdown violation: invalid schema for {name!r}")
        seen.append(name)
    if len(seen) != len(expected) or set(seen) != expected:
        raise RuntimeError("Sift lockdown violation: managed Claude tool set is incomplete or duplicated")


def _user_content(prompt: str, images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for image in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": image["mime"],
                "data": image["data"],
            },
        })
    return content


class ManagedClaudeSession(ABC):
    """Abstract local-history Claude Messages session for managed clouds."""

    PROVIDER = ""
    PROVIDER_LABEL = "Managed Claude"

    def __init__(
        self,
        cwd: Path,
        model: str,
        system_prompt: str,
        continue_conversation: bool = False,
        effort: str | None = None,
    ) -> None:
        from sift.provider.catalog import clamp_effort

        self.cwd = cwd
        self.model = model
        self._system_prompt = system_prompt
        self.effort = clamp_effort(effort, self.PROVIDER)
        del continue_conversation
        self._client: Any = None
        self._history: list[dict[str, Any]] = []
        self._tools = build_claude_tools()
        self._configuration_error: str | None = None

    @abstractmethod
    async def open(self) -> None:
        """Open the concrete managed-provider client."""

    @abstractmethod
    async def _request(
        self,
        messages: list[dict[str, Any]],
    ) -> ManagedClaudeResponse:
        """Translate and execute one provider-specific request."""

    @abstractmethod
    def _failure_event(self, error: Exception) -> Event:
        """Translate a provider-specific failure into Sift's event contract."""

    def _missing_auth_reason(self) -> str:
        return self._configuration_error or f"{self.PROVIDER_LABEL} is not configured"

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._history = []
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:  # noqa: BLE001, S110 - best-effort SDK close
                    pass

    async def set_model(self, model_id: str) -> dict[str, Any]:
        from sift.provider.catalog import get_model

        try:
            info = get_model(model_id)
        except KeyError:
            return {"ok": False, "reason": f"unknown model: {model_id}"}
        if info.provider != self.PROVIDER:
            return {
                "ok": False,
                "reason": (
                    f"model {model_id!r} belongs to provider {info.provider!r}, "
                    f"not {self.PROVIDER_LABEL}"
                ),
            }
        unchanged = model_id == self.model
        self.model = model_id
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
            **({"unchanged": True} if unchanged else {}),
        }

    async def set_effort(self, effort: str) -> dict[str, Any]:
        from sift.provider.catalog import effort_levels_for_provider, get_effort

        if effort not in effort_levels_for_provider(self.PROVIDER):
            return {
                "ok": False,
                "reason": f"{self.PROVIDER_LABEL} does not support effort {effort!r}",
            }
        info = get_effort(effort)
        unchanged = effort == self.effort
        self.effort = effort
        return {
            "ok": True,
            "effort": effort,
            "label": info.label,
            **({"unchanged": True} if unchanged else {}),
        }

    async def send(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        await self.open()
        if self._client is None:
            yield AuthFailure(reason=self._missing_auth_reason())
            return
        try:
            validated_images = validate_explicit_images(images)
        except AttachmentValidationError as exc:
            yield TurnError(message=str(exc))
            return

        verify_claude_lockdown(self._tools)
        turn_messages: list[dict[str, Any]] = [
            {"role": "user", "content": _user_content(prompt, validated_images)}
        ]
        try:
            rounds = int(os.environ.get("SIFT_MANAGED_CLAUDE_MAX_TOOL_ROUNDS", "16"))
        except (TypeError, ValueError):
            rounds = 16
        if not 1 <= rounds <= 64:
            rounds = 16
        last_input = 0
        last_output = 0

        try:
            for round_index in range(rounds):
                verify_claude_lockdown(self._tools)
                try:
                    response = await self._request(self._history + turn_messages)
                except asyncio.CancelledError:
                    # Closing the SDK client interrupts an in-flight socket where
                    # the SDK supports it; cancellation always propagates.
                    await self.close()
                    raise
                except Exception as exc:  # noqa: BLE001 - SDK exception families vary
                    yield self._failure_event(exc)
                    return
                last_input = response.input_tokens
                last_output = response.output_tokens

                pending: list[dict[str, Any]] = []
                for block in response.content:
                    kind = block.get("type")
                    if kind == "text":
                        text = block.get("text")
                        if isinstance(text, str) and text.strip():
                            yield AssistantText(text=text)
                    elif kind in {"thinking", "redacted_thinking"}:
                        trace = block.get("thinking") or block.get("text")
                        if isinstance(trace, str) and trace.strip():
                            yield AssistantThinking(text=trace)
                    elif kind == "tool_use":
                        pending.append(block)
                        raw_name = block.get("name")
                        raw_args = block.get("input")
                        raw_call_id = block.get("id")
                        name = raw_name if isinstance(raw_name, str) else ""
                        args = raw_args if isinstance(raw_args, dict) else {}
                        call_id = raw_call_id if isinstance(raw_call_id, str) else ""
                        yield ToolCall(name=name, input=args, call_id=call_id)

                if response.stop_reason not in {None, "end_turn", "stop_sequence", "tool_use"}:
                    guidance = (
                        "Request a smaller result or raise the deployment output limit."
                        if response.stop_reason == "max_tokens"
                        else "Review the managed-cloud safety and model policy."
                    )
                    yield TurnError(message=(
                        f"{self.PROVIDER_LABEL} stopped with {response.stop_reason!r}. "
                        f"{guidance} The incomplete turn was not committed."
                    ))
                    return

                assistant_message = {
                    "role": "assistant",
                    "content": [dict(block) for block in response.content],
                }
                turn_messages.append(assistant_message)
                if not pending:
                    self._history.extend(turn_messages)
                    yield TurnDone(
                        input_tokens=last_input,
                        output_tokens=last_output,
                        post_turn_tokens=last_input + last_output,
                    )
                    return

                tool_results: list[dict[str, Any]] = []
                for position, call in enumerate(pending):
                    raw_id = call.get("id")
                    call_id = (
                        raw_id
                        if isinstance(raw_id, str) and raw_id
                        else f"managed:r{round_index}:{position}"
                    )
                    raw_name = call.get("name")
                    name = raw_name if isinstance(raw_name, str) else ""
                    tool_args = call.get("input")
                    handler = HANDLERS.get(name)
                    is_error = False
                    if not call_id or not name or not isinstance(tool_args, dict):
                        out_text = json.dumps({
                            "status": "error",
                            "reason": "managed Claude returned a malformed tool call; regenerate it with an id, known name, and object arguments",
                        })
                        is_error = True
                    elif handler is None:
                        out_text = json.dumps({"status": "error", "reason": f"unknown tool: {name!r}"})
                        is_error = True
                    else:
                        try:
                            result = await handler(tool_args)
                            out_text = _mcp_payload_to_text(result)
                        except Exception as exc:  # noqa: BLE001 - tool boundary
                            if os.environ.get("SIFT_DEBUG_USAGE") == "1":
                                print(
                                    f"[sift.{self.PROVIDER}] tool {name!r} raised {type(exc).__name__}: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            out_text = json.dumps({
                                "status": "error",
                                "reason": f"tool handler failed with {type(exc).__name__}; retry with different arguments",
                            })
                            is_error = True
                    run_dir, language = _extract_hints(out_text)
                    yield ToolCallResult(
                        call_id=call_id,
                        text=out_text,
                        is_error=is_error,
                        run_dir=run_dir,
                        language=language,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": out_text,
                        "is_error": is_error,
                    })
                turn_messages.append({"role": "user", "content": tool_results})

            yield TurnError(message=(
                f"{self.PROVIDER_LABEL} tool loop did not converge within {rounds} rounds. "
                "No partial conversation state was committed."
            ))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - final provider boundary
            yield self._failure_event(exc)

    async def __aenter__(self):
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


__all__ = [
    "ManagedClaudeResponse",
    "ManagedClaudeSession",
    "build_claude_tools",
    "verify_claude_lockdown",
]
