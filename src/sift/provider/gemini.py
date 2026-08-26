"""Google Gemini-backed ``ProviderSession`` implementation.

Wraps the ``google-genai`` SDK's chat interface behind the same
``ProviderSession`` contract Anthropic and OpenAI implement. Tool
registration draws from ``provider.tool_schemas`` (the canonical,
provider-neutral list); tool dispatch routes back to the same
``sift.tools.HANDLERS`` map every provider uses, so the privacy
semantics -- sandbox-wrapped execution, sanitizer-clamped results --
are byte-for-byte identical regardless of which model authored the
call.

Lockdown discipline, same headline guarantee as the OpenAI provider:

- The ``tools`` list sent to Gemini is a single ``Tool`` object whose
  ONLY populated field is ``function_declarations`` -- built from
  EXACTLY the Sift tools in ``build_tool_specs()``. Every other field
  on ``types.Tool`` (``google_search``, ``code_execution``,
  ``url_context``, ``computer_use``, ``mcp_servers``, ...) is a
  Gemini-native built-in capability and is never set.
  ``test_gemini_lockdown.py`` asserts this on every request.
- The Gemini SDK's own "automatic function calling" convenience layer
  (which would execute Python callables the SDK is given, entirely
  outside Sift's sandbox/sanitizer) is explicitly disabled
  (``automatic_function_calling=AutomaticFunctionCallingConfig(
  disable=True)``) -- Sift always intercepts the function-call turn
  itself and dispatches through ``HANDLERS``, never through the SDK's
  own executor.

Conversation state: like OpenAI's local ``store=false`` replay but unlike the
Anthropic Agent SDK (which rebuilds context from its transcript), the ``google-genai``
SDK's ``AsyncChat`` object holds the full conversation IN THE CLIENT
PROCESS and appends to it automatically on every ``send_message()``
call. This session keeps one ``AsyncChat`` alive for its lifetime --
there is no remote chain-expiry failure mode to recover from.

Automated coverage validates request and response shapes against the pinned
``google-genai`` SDK. Live-provider qualification is a separate, credentialed
release activity and must not be inferred from mocked transport tests.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from typing_extensions import Self

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
from sift.provider.error_safety import provider_error_message
from sift.provider.tool_schemas import build_tool_specs
from sift.tools import HANDLERS

PROVIDER_ID = "gemini"


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------


def detect_auth() -> str:
    """Return ``'api_key'`` if a Gemini credential source is present,
    ``'unknown'`` otherwise. Gemini (the public API, not Vertex AI)
    has no Sift-supported subscription path -- only API keys."""
    return "api_key" if _resolve_api_key() else "unknown"


def _resolve_api_key() -> str | None:
    """Pick the Gemini API key from env first (researcher override),
    keyring second. Checks both ``GEMINI_API_KEY`` (Google's current
    documented name) and ``GOOGLE_API_KEY`` (the SDK's own fallback
    env var, still honored by ``google-genai`` internally, so a
    researcher with either already set gets picked up)."""
    from sift import auth as _auth

    return _auth.resolve_provider_credential(
        "gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------
#
# Module-level (not session methods) so the lockdown test can call
# these directly without spinning up a real session, mirroring the
# OpenAI provider's ``build_openai_tools`` / ``_verify_lockdown``
# split.


def build_gemini_tools() -> Any:
    """Build the single Gemini ``Tool`` carrying every Sift function
    declaration, and nothing else -- no ``google_search``,
    ``code_execution``, ``url_context``, or any other Gemini-native
    built-in capability field."""
    from google.genai import types

    declarations = []
    for spec in build_tool_specs():
        declarations.append(
            types.FunctionDeclaration(
                name=spec.name,
                description=spec.description,
                parameters=types.Schema.model_validate(spec.input_schema),
            )
        )
    return types.Tool(function_declarations=declarations)


def _verify_lockdown(tool: Any) -> None:
    """Last-line guard: raise if the Gemini tool carries anything
    besides ``function_declarations``, or if any declaration isn't a
    known Sift tool. Checked on every request, not just at startup --
    same defense-in-depth posture as the OpenAI provider.
    """
    expected_names = {s.name for s in build_tool_specs()}
    # ``model_fields`` covers every field the pinned SDK's ``Tool``
    # type knows about (retrieval, google_search, code_execution,
    # computer_use, mcp_servers, ...) -- enumerate them generically so
    # a future SDK version adding a new built-in capability field is
    # caught automatically rather than requiring a matching update to
    # a hand-maintained forbidden-list (the class of drift the OpenAI
    # provider's ``FORBIDDEN_BUILTIN_TYPES`` comment explicitly warns
    # about avoiding).
    for field_name in type(tool).model_fields:
        if field_name == "function_declarations":
            continue
        if getattr(tool, field_name, None) is not None:
            raise RuntimeError(
                f"Sift lockdown violation: Gemini tool has non-function "
                f"field {field_name!r} set"
            )
    declarations = tool.function_declarations or []
    for decl in declarations:
        if decl.name not in expected_names:
            raise RuntimeError(
                f"Sift lockdown violation: unknown function declaration {decl.name!r}"
            )
    if {d.name for d in declarations} != expected_names:
        missing = expected_names - {d.name for d in declarations}
        raise RuntimeError(
            f"Sift lockdown violation: Gemini tool is missing "
            f"declarations for {sorted(missing)}"
        )


# Effort → Gemini ``ThinkingLevel``. Gemini's ladder has no rung above
# HIGH (no analogue to OpenAI's orthogonal "pro" mode or Anthropic's
# "max" -- see ``catalog.PROVIDER_EFFORTS`` for the ladder this
# provider actually offers: low/medium/high only). ``xhigh`` clamps
# down to ``high`` before this mapping is even consulted (via
# ``clamp_effort`` at construction / ``set_effort``), so every key
# this dict needs to answer for is already one of the three rungs
# Gemini's own ladder offers.
_THINKING_LEVEL_BY_EFFORT: dict[str, str] = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


def _translate_send_exception(
    e: Exception,
    model_id: str,
) -> AuthFailure | TurnError:
    """Translate any exception ``chat.send_message()`` can raise into
    a provider-neutral event.

    Prefers STRUCTURED signals over string matching wherever the SDK
    provides them:

    - ``google.genai.errors.APIError`` (raised for any non-2xx HTTP
      response the SDK got back) carries a real numeric ``.code`` and
      a machine ``.status`` string (``"UNAUTHENTICATED"``,
      ``"PERMISSION_DENIED"``, ``"NOT_FOUND"``, ``"RESOURCE_EXHAUSTED"``,
      ...) -- branching on those is far more reliable than guessing
      from ``str(e)``, which is free-text and not a stable contract
      the SDK promises to keep wording-compatible across versions.
    - ``httpx.TimeoutException`` / ``httpx.ConnectError`` (the
      underlying transport library google-genai's async client uses)
      fire when no HTTP response was ever received at all -- a
      different failure class from an API error, and one an
      ``APIError``-only check would never catch. Distinct, actionable
      messages for each: a timeout suggests transient network
      latency (retry), a connect failure suggests something is
      actively blocking the connection (corporate proxy/firewall) --
      different enough that conflating them into one generic
      "request failed" message would send a researcher down the
      wrong troubleshooting path.
    - Anything else falls back to the substring heuristic this
      module originally shipped with, preserved as the last-resort
      case so a future SDK version raising something entirely
      unanticipated -- or a test double shaped as a bare
      ``RuntimeError`` -- still gets SOME translation rather than an
      unhandled exception reaching the caller.
    """
    import httpx
    from google.genai import errors as genai_errors

    safe_error = provider_error_message(e, secrets=(_resolve_api_key(),))
    if isinstance(e, genai_errors.APIError):
        code = e.code
        status = (e.status or "").upper()
        message = provider_error_message(
            e.message or safe_error,
            secrets=(_resolve_api_key(),),
        )
        if code in (401, 403) or status in (
            "UNAUTHENTICATED",
            "PERMISSION_DENIED",
        ):
            return AuthFailure(
                reason=f"Gemini auth failure ({code} {status}): {message}"
            )
        if code == 404 or status == "NOT_FOUND":
            return TurnError(
                message=(
                    f"Gemini returned 'not found' for model {model_id!r} "
                    f"({code} {status}). Confirm this model id is still "
                    f"valid -- check https://ai.google.dev/gemini-api/docs"
                    f"/models for the current catalog; a preview model "
                    f"can be retired without notice. Underlying error: "
                    f"{message}"
                )
            )
        if code == 429 or status == "RESOURCE_EXHAUSTED":
            return TurnError(
                message=(
                    f"Gemini rate-limited or quota-exhausted this request "
                    f"({code} {status}). Wait a moment and retry, or "
                    f"check your Google AI Studio / Cloud project's "
                    f"quota. Underlying error: {message}"
                )
            )
        if code is not None and code >= 500:
            return TurnError(
                message=(
                    f"Gemini's server returned an error ({code} {status}) "
                    f"-- likely transient. Retry the request. Underlying "
                    f"error: {message}"
                )
            )
        # Context-length overflow is reported as a 400
        # INVALID_ARGUMENT whose MESSAGE names token/context limits --
        # there's no distinct status code for it, so this still needs
        # a text check, but scoped to the structured ``.message``
        # field the SDK parsed out of the response body, not
        # ``str(e)`` at large (which also includes the code/status
        # prefix this function already extracted separately).
        lower_message = message.lower()
        if (
            "context" in lower_message
            and "token" in lower_message
            and (
                "limit" in lower_message
                or "exceed" in lower_message
                or "too long" in lower_message
            )
        ):
            return TurnError(
                message=(
                    "Conversation hit the model's context window. To "
                    "continue: start a new session, or reduce earlier "
                    "turns from this session by summarizing them via "
                    "``recall_conversation`` and starting fresh from the "
                    f"summary. Underlying error: {message}"
                )
            )
        return TurnError(
            message=(f"Gemini request failed ({code} {status}): {message}")
        )

    if isinstance(e, httpx.TimeoutException):
        return TurnError(
            message=(
                "Gemini request timed out before any response arrived. "
                "This is usually transient network latency -- retry. If "
                "it keeps happening, check network connectivity to "
                f"Google's API. Underlying error: {safe_error}"
            )
        )
    if isinstance(e, httpx.ConnectError):
        return TurnError(
            message=(
                "Could not connect to Gemini's API at all -- no HTTP "
                "response was received. Check network connectivity "
                "(including any corporate proxy or firewall that might "
                "block generativelanguage.googleapis.com). Underlying "
                f"error: {safe_error}"
            )
        )

    # Last-resort fallback for any exception shape not specifically
    # recognized above.
    msg = safe_error
    lower = msg.lower()
    if (
        "api key" in lower
        or "unauthorized" in lower
        or "401" in lower
        or "permission" in lower
        or "403" in lower
    ):
        return AuthFailure(reason=f"Gemini auth failure: {msg}")
    if (
        "context" in lower
        and "token" in lower
        and ("limit" in lower or "exceed" in lower or "too long" in lower)
    ):
        return TurnError(
            message=(
                "Conversation hit the model's context window. To "
                "continue: start a new session, or reduce earlier turns "
                "from this session by summarizing them via "
                "``recall_conversation`` and starting fresh from the "
                f"summary. Underlying error: {msg}"
            )
        )
    return TurnError(message=f"Gemini request failed: {msg}")


class GeminiSession:
    """Google Gemini chat session via ``google-genai``'s ``AsyncChat``.

    The ``AsyncChat`` object IS the conversation state for this
    session's lifetime -- unlike OpenAI's response-id chain or
    Anthropic's on-disk-transcript rebuild, there is no separate
    pointer to manage; ``send_message()`` appends to the chat's own
    history automatically, on both success and (partially) failure
    paths, so this session keeps exactly one ``AsyncChat`` alive for
    as long as the session itself is open.
    """

    PROVIDER = PROVIDER_ID
    PROVIDER_LABEL = "Gemini"

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
        self.effort: str = clamp_effort(effort, self.PROVIDER)
        # Interface symmetry with the other providers; Gemini has no
        # analogous resumable-session concept in this integration.
        del continue_conversation
        self._client: Any = None
        self._chat: Any = None
        self._tool: Any = None

    def _wire_model_id(self) -> str:
        return self.model

    def _missing_auth_reason(self) -> str:
        return (
            "no Gemini API key configured. Add one in the auth screen or "
            "set GEMINI_API_KEY in the environment."
        )

    def _translate_failure(self, error: Exception) -> Event:
        return _translate_send_exception(error, self._wire_model_id())

    def _error_secrets(self) -> tuple[str | None, ...]:
        return (_resolve_api_key(),)

    # ---- lifecycle -------------------------------------------------------

    async def open(self) -> None:
        """Lazy-build the genai client + chat if a key is available.

        Same "no-op on missing key, let send() yield AuthFailure"
        posture as the OpenAI provider -- see that method's docstring
        for why raising here would surface as the wrong event type.
        """
        if self._client is not None:
            return
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover — google-genai is a dep
            raise RuntimeError(
                f"google-genai SDK not installed: {e}. Reinstall Sift dependencies."
            )
        api_key = _resolve_api_key()
        if not api_key:
            return
        from google.genai import types

        from sift.integration_core import MODEL_REQUEST_TIMEOUT_SECONDS

        # google-genai otherwise defaults to five HTTP attempts. A model turn
        # is not safe to replay automatically: a timed-out request may already
        # have generated output or initiated a tool call. Keep one attempt and
        # let the researcher decide whether to retry the whole turn.
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(MODEL_REQUEST_TIMEOUT_SECONDS * 1_000),
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        self._tool = build_gemini_tools()
        _verify_lockdown(self._tool)
        self._chat = self._client.aio.chats.create(
            model=self._wire_model_id(),
            config=self._build_config(),
        )

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._chat = None
        self._tool = None
        if client is None:
            return
        # google-genai owns distinct synchronous and asynchronous transports.
        # Its documented lifecycle requires ``aio.aclose()`` for the async
        # HTTP client and ``Client.close()`` for the sync client; dropping the
        # reference alone can leak pooled sockets and schedule destructor work
        # after Sift's event loop has stopped.
        try:
            async_client = getattr(client, "aio", None)
            aclose = getattr(async_client, "aclose", None)
            if callable(aclose):
                await aclose()
        except Exception:  # noqa: BLE001 — continue closing the sync layer
            pass
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        except Exception:  # noqa: BLE001 — close-time errors aren't useful
            pass

    @staticmethod
    def _history_checkpoint(chat: Any) -> tuple[list[Any], list[Any]] | None:
        """Copy both SDK histories so a failed turn can be rolled back.

        Gemini's chat records each HTTP-successful round before returning it.
        Safety stops and incomplete tool loops therefore need an explicit
        rollback or the next turn inherits an invalid branch.
        """
        curated = getattr(chat, "_curated_history", None)
        comprehensive = getattr(chat, "_comprehensive_history", None)
        if not isinstance(curated, list) or not isinstance(comprehensive, list):
            return None
        return list(curated), list(comprehensive)

    @staticmethod
    def _restore_history(
        chat: Any,
        checkpoint: tuple[list[Any], list[Any]] | None,
    ) -> None:
        """Restore a turn checkpoint in place when the SDK exposes it."""
        if checkpoint is None:
            return
        curated = getattr(chat, "_curated_history", None)
        comprehensive = getattr(chat, "_comprehensive_history", None)
        if isinstance(curated, list) and isinstance(comprehensive, list):
            curated[:] = checkpoint[0]
            comprehensive[:] = checkpoint[1]

    def _build_config(self) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=self._system_prompt,
            tools=[self._tool],
            thinking_config=types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=types.ThinkingLevel(
                    _THINKING_LEVEL_BY_EFFORT.get(self.effort, "HIGH")
                ),
            ),
            # Belt-and-braces alongside ``_verify_lockdown``: Gemini's
            # own convenience layer for auto-executing Python
            # callables never runs here, because Sift never gives it
            # any -- ``tools`` above carries only declarations, not
            # callables -- but pinning this explicitly documents the
            # intent at the request boundary rather than relying on
            # "we never happened to pass callables" as the guarantee.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True,
            ),
        )

    # ---- model swap ------------------------------------------------------

    async def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch active Gemini model.

        Unlike OpenAI (model is a per-request field), Gemini's
        ``AsyncChat`` is bound to a model at construction. Rebuild it
        transactionally with curated local history so changing models
        never silently erases the research conversation.
        """
        from sift.provider.catalog import get_model

        try:
            info = get_model(model_id)
        except KeyError:
            return {"ok": False, "reason": f"unknown model: {model_id}"}
        if info.provider != self.PROVIDER:
            return {
                "ok": False,
                "reason": (
                    f"model {model_id!r} belongs to provider "
                    f"{info.provider!r}, not Gemini"
                ),
            }
        if model_id == self.model:
            return {
                "ok": True,
                "model": model_id,
                "label": info.label,
                "context_window": info.context_window,
                "unchanged": True,
            }
        if self._client is not None:
            old_chat = self._chat
            try:
                history = (
                    list(old_chat.get_history(curated=True))
                    if old_chat is not None
                    else []
                )
                new_chat = self._client.aio.chats.create(
                    model=(self._wire_model_id() if model_id == self.model else model_id),
                    config=self._build_config(),
                    history=history,
                )
            except Exception as e:  # noqa: BLE001 - SDK shape varies
                return {
                    "ok": False,
                    "reason": (
                        "Gemini model switch could not preserve the local "
                        f"conversation ({e.__class__.__name__})."
                    ),
                }
            self._chat = new_chat
        self.model = model_id
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
        }

    async def set_effort(self, effort: str) -> dict[str, Any]:
        """Switch reasoning effort.

        Gemini's thinking level is part of ``GenerateContentConfig``,
        passed per ``send_message()`` call in this implementation (see
        ``_build_config`` called fresh each turn in ``send``), so this
        is a field change that takes effect on the next message with
        no chat rebuild and no history loss.
        """
        from sift.provider.catalog import (
            effort_levels_for_provider,
            get_effort,
        )

        if effort not in effort_levels_for_provider(self.PROVIDER):
            return {
                "ok": False,
                "reason": (
                    f"{self.PROVIDER_LABEL} does not support effort level {effort!r}"
                ),
            }
        info = get_effort(effort)
        if effort == self.effort:
            return {
                "ok": True,
                "effort": effort,
                "label": info.label,
                "unchanged": True,
            }
        self.effort = effort
        return {"ok": True, "effort": effort, "label": info.label}

    # ---- send --------------------------------------------------------

    async def send(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one chat turn. Yields a flat stream of provider-neutral
        events terminated by ``TurnDone`` / ``TurnError`` /
        ``AuthFailure``.

        Internally drives Gemini's function-calling loop: each
        round-trip is one ``chat.send_message()`` call. If the
        response contains ``function_call`` parts, dispatch them via
        ``HANDLERS`` and send the results back as
        ``function_response`` parts in the next round. Loop exits
        when a response has no function-call parts.
        """
        from google.genai import types

        await self.open()
        chat = self._chat
        if chat is None:
            yield AuthFailure(
                reason=(
                    self._missing_auth_reason()
                )
            )
            return


        from sift.provider.attachments import (
            AttachmentValidationError,
            validate_explicit_images,
        )
        try:
            images = validate_explicit_images(images)
        except AttachmentValidationError as exc:
            yield TurnError(message=str(exc))
            return

        message_parts = _build_user_parts(prompt, images)

        try:
            MAX_TOOL_ROUNDS = int(os.environ.get("SIFT_GEMINI_MAX_TOOL_ROUNDS", "16"))
            if not 1 <= MAX_TOOL_ROUNDS <= 64:
                MAX_TOOL_ROUNDS = 16
        except (TypeError, ValueError):
            MAX_TOOL_ROUNDS = 16

        last_input_tokens = 0
        last_output_tokens = 0
        last_thoughts_tokens = 0
        config = self._build_config()
        history_checkpoint = self._history_checkpoint(chat)

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                _verify_lockdown(self._tool)
                try:
                    resp = await chat.send_message(
                        message_parts,
                        config=config,
                    )
                except Exception as e:  # noqa: BLE001 — translate to event
                    self._restore_history(chat, history_checkpoint)
                    yield self._translate_failure(e)
                    return

                usage = getattr(resp, "usage_metadata", None)
                if usage is not None:
                    last_input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    last_output_tokens = (
                        getattr(usage, "candidates_token_count", 0) or 0
                    )
                    last_thoughts_tokens = (
                        getattr(usage, "thoughts_token_count", 0) or 0
                    )
                if os.environ.get("SIFT_DEBUG_USAGE") == "1" and usage is not None:
                    from sift.provider.usage_log import append_usage_line

                    cached = getattr(usage, "cached_content_token_count", 0) or 0
                    line = (
                        f"[sift.usage.gemini] round model={self.model} "
                        f"prompt_tokens={last_input_tokens} "
                        f"candidates_tokens={last_output_tokens} "
                        f"thoughts_tokens={last_thoughts_tokens} "
                        f"cached_tokens={cached} "
                        f"(cached is a SUBSET of prompt_tokens, not additive)"
                    )
                    print(line, file=sys.stderr, flush=True)
                    append_usage_line(self.cwd, line)

                candidates = getattr(resp, "candidates", None) or []
                if not candidates:
                    self._restore_history(chat, history_checkpoint)
                    prompt_feedback = getattr(resp, "prompt_feedback", None)
                    blocked_reason = (
                        getattr(prompt_feedback, "block_reason_message", None)
                        or getattr(prompt_feedback, "block_reason", None)
                    )
                    safe_reason = (
                        provider_error_message(
                            getattr(blocked_reason, "value", blocked_reason),
                            secrets=self._error_secrets(),
                        )
                        if blocked_reason
                        else "no block reason was returned"
                    )
                    yield TurnError(
                        message=(
                            "Gemini returned a response with no candidates "
                            "(likely blocked by a safety filter or prompt "
                            f"feedback issue: {safe_reason}). The failed turn "
                            "was removed from local conversation history."
                        )
                    )
                    return
                candidate = candidates[0]
                content = getattr(candidate, "content", None)
                parts = list(getattr(content, "parts", None) or [])
                raw_finish_reason = getattr(candidate, "finish_reason", None)
                finish_reason = getattr(raw_finish_reason, "value", raw_finish_reason)
                if finish_reason not in (None, "STOP"):
                    self._restore_history(chat, history_checkpoint)
                    finish_message = getattr(candidate, "finish_message", None)
                    safe_message = (
                        provider_error_message(
                            finish_message,
                            secrets=self._error_secrets(),
                        )
                        if finish_message
                        else "the provider did not return further details"
                    )
                    if finish_reason == "MAX_TOKENS":
                        guidance = "Increase the output limit or request a smaller result."
                    elif finish_reason == "MALFORMED_FUNCTION_CALL":
                        guidance = "Retry so the model can regenerate the tool call."
                    else:
                        guidance = (
                            "Review the request and provider-project safety "
                            "policy, then retry if appropriate."
                        )
                    yield TurnError(
                        message=(
                            f"Gemini stopped with {finish_reason}: {safe_message}. "
                            f"{guidance} The failed turn was removed from local "
                            "conversation history."
                        )
                    )
                    return
                if not parts:
                    self._restore_history(chat, history_checkpoint)
                    yield TurnError(
                        message=(
                            "Gemini returned an empty candidate. The failed turn "
                            "was removed from local conversation history; retry."
                        )
                    )
                    return

                # Each ToolCall/ToolCallResult pair needs a call_id
                # that's unique WITHIN THIS TURN -- the frontend keys
                # a tool-result card off it (see app.js's
                # ``card.dataset.callId`` / the matching ``.find()``
                # by call_id) to route each result back to the card
                # its own call created. Reusing the bare tool NAME as
                # call_id (the previous behavior) is fine for a
                # round's only call to a given tool, but breaks the
                # instant the model issues two PARALLEL calls to the
                # SAME tool in one round (e.g. two ``get_schema``
                # calls for different datasets): both cards would
                # share one call_id, so the frontend's ``.find()``
                # always resolves to the FIRST card -- the first
                # result overwrites it twice and the second card is
                # left stuck pending forever. Prefer Gemini's own
                # ``FunctionCall.id`` when the SDK populates it (it's
                # documented as optional and frequently absent for
                # single-call rounds); otherwise synthesize a
                # deterministic, turn-unique id from the tool name,
                # round number, and position within the round.
                pending_calls: list[
                    tuple[str, str, dict[str, Any] | None]
                ] = []
                for part in parts:
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        name = getattr(fc, "name", "") or ""
                        args = getattr(fc, "args", None)
                        parsed_args = args if isinstance(args, dict) else None
                        raw_id = getattr(fc, "id", None)
                        call_id = (
                            raw_id
                            if isinstance(raw_id, str) and raw_id
                            else f"{name}:r{_round}:{len(pending_calls)}"
                        )
                        pending_calls.append((call_id, name, parsed_args))
                        yield ToolCall(
                            name=name,
                            input=parsed_args or {},
                            call_id=call_id,
                        )
                        continue
                    is_thought = bool(getattr(part, "thought", False))
                    text = getattr(part, "text", None)
                    if text:
                        if is_thought:
                            yield AssistantThinking(text=text)
                        else:
                            yield AssistantText(text=text)

                if not pending_calls:
                    yield TurnDone(
                        input_tokens=last_input_tokens,
                        output_tokens=last_output_tokens,
                        # Gemini's ``thoughts_token_count`` occupies
                        # the context window (it's billed output) but
                        # isn't a cache field on any other provider's
                        # taxonomy -- folded into post_turn_tokens
                        # directly rather than forced into one of the
                        # Anthropic-shaped cache fields, which would
                        # misrepresent what it actually is.
                        post_turn_tokens=(
                            last_input_tokens
                            + last_output_tokens
                            + last_thoughts_tokens
                        ),
                    )
                    return

                # Dispatch every function call in this round, in
                # order. On the WIRE back to Gemini, function_response
                # parts are matched to function_call parts
                # POSITIONALLY within the same turn -- this is
                # separate from (and unaffected by) the call_id fix
                # above, which is purely for Sift's own internal
                # event stream / frontend card routing.
                # ``from_function_response`` below is built from
                # ``name`` alone (matching the SDK's own documented
                # automatic-function-calling contract, which this
                # session deliberately doesn't use but whose matching
                # contract this dispatch loop mirrors); it never
                # carries our synthesized call_id.
                response_parts = []
                for call_id, name, args in pending_calls:
                    handler = HANDLERS.get(name)
                    if handler is None:
                        out_text = json.dumps(
                            {
                                "status": "error",
                                "reason": f"unknown tool: {name!r}",
                            }
                        )
                        yield ToolCallResult(
                            call_id=call_id,
                            text=out_text,
                            is_error=True,
                        )
                        response_parts.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": out_text},
                            )
                        )
                        continue
                    if args is None:
                        out_text = json.dumps(
                            {
                                "status": "error",
                                "reason": (
                                    "function arguments were not a JSON object; "
                                    "regenerate the call with named arguments"
                                ),
                            }
                        )
                        yield ToolCallResult(
                            call_id=call_id,
                            text=out_text,
                            is_error=True,
                        )
                        response_parts.append(
                            types.Part.from_function_response(
                                name=name,
                                response={"result": out_text},
                            )
                        )
                        continue
                    try:
                        result = await handler(args)
                        out_text = _mcp_payload_to_text(result)
                        is_error = False
                    except Exception as e:  # noqa: BLE001
                        # See openai.py's matching fix for the full
                        # rationale: full message + traceback (which
                        # can carry parser excerpts / file paths / raw
                        # data values) are gated behind
                        # SIFT_DEBUG_USAGE rather than persisted to
                        # .sift-usage.log unconditionally.
                        if os.environ.get("SIFT_DEBUG_USAGE") == "1":
                            import traceback

                            diag = (
                                f"[sift.gemini] tool {name!r} handler "
                                f"raised {e.__class__.__name__}: {e}\n"
                                + traceback.format_exc()
                            )
                        else:
                            diag = (
                                f"[sift.gemini] tool {name!r} handler "
                                f"raised {e.__class__.__name__} (set "
                                f"SIFT_DEBUG_USAGE=1 for the full "
                                f"message/traceback)"
                            )
                        print(diag, file=sys.stderr, flush=True)
                        try:
                            from sift.provider.usage_log import (
                                append_usage_line,
                            )

                            append_usage_line(self.cwd, diag.rstrip())
                        except Exception:  # noqa: BLE001 — logging must not block
                            pass
                        out_text = json.dumps(
                            {
                                "status": "error",
                                "reason": (
                                    f"tool handler failed with "
                                    f"{e.__class__.__name__}. Retry with "
                                    f"different arguments or fall back to "
                                    f"another tool."
                                ),
                            }
                        )
                        is_error = True
                    run_dir, language = _extract_hints(out_text)
                    yield ToolCallResult(
                        call_id=call_id,
                        text=out_text,
                        is_error=is_error,
                        run_dir=run_dir,
                        language=language,
                    )
                    # Gemini's function_response ``response`` field
                    # must be a dict, not a bare string -- wrap the
                    # JSON-text tool output under a "result" key
                    # (parallel to how the OpenAI path sends the same
                    # text as a bare string ``output`` field; the
                    # WIRE shape differs per provider, but the actual
                    # content crossing back to the model is identical
                    # either way).
                    response_parts.append(
                        types.Part.from_function_response(
                            name=name,
                            response={"result": out_text},
                        )
                    )
                message_parts = response_parts

            self._restore_history(chat, history_checkpoint)
            yield TurnError(
                message=(
                    f"Gemini tool loop did not converge within "
                    f"{MAX_TOOL_ROUNDS} rounds; last response still had "
                    f"pending function calls."
                )
            )
        except Exception as e:  # noqa: BLE001 — last-line catch
            self._restore_history(chat, history_checkpoint)
            msg = provider_error_message(e, secrets=self._error_secrets())
            yield TurnError(message=f"{self.PROVIDER_LABEL} session error: {msg}")

    # ---- async-context-manager sugar --------------------------------

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_user_parts(prompt: str, images: list[dict[str, Any]] | None) -> list[Any]:
    """Build Gemini message parts. Text is one text part; each image
    is one inline-data part built from raw bytes."""
    import base64

    from google.genai import types

    parts: list[Any] = []
    if prompt:
        parts.append(types.Part.from_text(text=prompt))
    if images:
        for img in images:
            mime = img.get("mime", "image/png")
            data = img.get("data", "")
            try:
                raw = base64.b64decode(data)
            except (ValueError, TypeError):
                continue
            parts.append(types.Part.from_bytes(data=raw, mime_type=mime))
    return parts


def _payload_has_image_block(payload: Any) -> bool:
    """Same check as the OpenAI provider's -- see that module for the
    rationale (``read_attached_file`` may carry an inline image block
    this provider can't forward as a function-response part)."""
    if not isinstance(payload, dict):
        return False
    content = payload.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            return True
    return False


def _rewrite_for_dropped_image(descriptor_text: str) -> str:
    """Gemini analogue of the OpenAI provider's function of the same
    name -- see there for the full rationale."""
    try:
        meta = json.loads(descriptor_text)
    except (ValueError, TypeError):
        return descriptor_text
    if not isinstance(meta, dict):
        return descriptor_text
    name = meta.get("name") or "the file"
    rewritten = {
        "status": "image_not_supported_on_provider",
        "name": meta.get("name"),
        "kind": meta.get("kind"),
        "ext": meta.get("ext"),
        "mime": meta.get("mime"),
        "size": meta.get("size"),
        "reason": (
            "Image tool results aren't supported on this provider. "
            f"Ask the researcher to re-@mention {name!r} in their "
            "next message — that routes the bytes through the user-"
            "message vision channel, which the model can see."
        ),
    }
    rewritten = {k: v for k, v in rewritten.items() if v is not None}
    return json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)


def _mcp_payload_to_text(payload: Any) -> str:
    """Sift handlers return MCP-shaped payloads:
    ``{"content": [{"type": "text", "text": "..."}]}``. Extract the
    JSON-text body for the Gemini function-response part."""
    if isinstance(payload, dict):
        content = payload.get("content")
        if isinstance(content, list):
            text_block = None
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_block = block
                    break
            if text_block is not None:
                if _payload_has_image_block(payload):
                    return _rewrite_for_dropped_image(
                        str(text_block.get("text", "")),
                    )
                return str(text_block.get("text", ""))
        return json.dumps(payload)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _extract_hints(text: str) -> tuple[str | None, str | None]:
    """Same hint-extraction as the other providers -- peel ``_run_dir``
    and ``_language`` from the tool-result JSON if present."""
    if not text.strip():
        return None, None
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    rd = parsed.get("_run_dir")
    lang = parsed.get("_language")
    return (
        rd if isinstance(rd, str) else None,
        lang if isinstance(lang, str) else None,
    )
