"""OpenAI-backed ``ProviderSession`` implementation.

Wraps the OpenAI Responses API behind the same ``ProviderSession``
contract Anthropic implements. Tool registration draws from
``provider.tool_schemas`` (the canonical, provider-neutral list);
tool dispatch routes back to the same handler functions in
``sift.tools.HANDLERS`` Anthropic uses, so the privacy semantics —
sandbox-wrapped execution, sanitizer-clamped results — are byte-for-
byte identical regardless of which model authored the call.

Lockdown discipline is the headline guarantee:

- The ``tools`` list sent to OpenAI contains EXACTLY the Sift
  function tools listed in ``build_tool_specs()`` and nothing else.
  No ``{"type": "web_search"}``, ``{"type": "code_interpreter"}``,
  ``{"type": "file_search"}``, no Agents-SDK built-ins.
  ``test_openai_lockdown.py`` mocks the client and asserts this on
  every request.
- ``parallel_tool_calls`` is on (the default) so the model can run
  ``get_schema`` and ``request_data`` in parallel during exploration,
  but every dispatch goes through the SAME ``HANDLERS`` map and the
  SAME sanitizer/sandbox boundary.
- The OpenAI client is constructed with an explicit ``api_key``
  pulled from the keyring (``sift.auth``). Env-var
  ``OPENAI_API_KEY`` is also honored as a fallback for power users
  who'd rather export their own.

Conversation state is local. Every request sets ``store=false``
and Sift carries returned output items (including opaque encrypted reasoning
state) forward in process memory. This avoids opting into the Responses API's
application-state retention. It does not claim zero retention for abuse-
monitoring logs; that is controlled by the researcher's OpenAI organization
or project agreement. There is no environment override that can silently
weaken this request-level storage boundary.
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

PROVIDER_ID = "openai"


def _translate_request_failure(
    error: Exception,
    *,
    model_id: str,
    safe_message: str,
) -> AuthFailure | TurnError:
    """Map SDK/HTTP failures to the common actionable event contract."""
    lower = safe_message.casefold()
    status = getattr(error, "status_code", None)
    class_name = type(error).__name__.casefold()
    if status in {401, 403} or any(
        token in lower for token in ("authentication", "api key", "unauthorized")
    ):
        return AuthFailure(reason=f"OpenAI auth failure: {safe_message}")
    if status == 429 or "ratelimit" in class_name or any(
        token in lower for token in ("rate limit", "rate_limit", "quota exceeded")
    ):
        return TurnError(message=(
            "OpenAI rate-limited or quota-exhausted this request. Wait a "
            "moment and retry, or check the API project's usage limits. "
            f"Underlying error: {safe_message}"
        ))
    if status == 404 or "notfound" in class_name or "model not found" in lower:
        return TurnError(message=(
            f"OpenAI model {model_id!r} is unavailable to this API project or "
            "has been retired. Run the model availability check and select its "
            f"documented replacement. Underlying error: {safe_message}"
        ))
    if (
        "context_length_exceeded" in lower
        or "maximum context length" in lower
        or (
            "input" in lower and "tokens" in lower
            and ("limit" in lower or "exceed" in lower)
        )
    ):
        return TurnError(message=(
            "Conversation hit the model's context window. Start a new session, "
            "or continue from a local conversation summary. Underlying error: "
            f"{safe_message}"
        ))
    if "timeout" in class_name or "timed out" in lower:
        return TurnError(message=(
            "OpenAI request timed out. The ambiguous turn was not retried "
            f"automatically; retry when ready. Underlying error: {safe_message}"
        ))
    if isinstance(status, int) and status >= 500:
        return TurnError(message=(
            f"OpenAI returned a transient server error ({status}). Retry the "
            f"turn when ready. Underlying error: {safe_message}"
        ))
    return TurnError(message=f"OpenAI request failed: {safe_message}")


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------


def detect_auth() -> str:
    """Return ``'api_key'`` if any OpenAI credential source is
    present, ``'unknown'`` otherwise. OpenAI has no Sift-supported
    subscription path — only API keys are recognised here."""
    return "api_key" if _resolve_api_key() else "unknown"


def _resolve_api_key() -> str | None:
    """Pick the OpenAI API key from env first (researcher override),
    keyring second. Returns ``None`` if neither is set."""
    from sift import auth as _auth

    return _auth.resolve_provider_credential("openai", ("OPENAI_API_KEY",))


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------
#
# These three helpers are kept module-level (not session methods) so
# the lockdown test can call them directly without spinning up a real
# session — the test imports ``build_openai_tools`` and asserts the
# returned list matches the canonical Sift tool set exactly, with no
# Responses-API built-ins mixed in.


def build_openai_tools() -> list[dict[str, Any]]:
    """Build the OpenAI Responses-API tool list from the canonical
    spec. Returns exactly the Sift tools defined in
    ``sift.provider.tool_schemas.TOOL_SPECS`` — never any Responses-API
    built-ins (``web_search``, ``code_interpreter``, ``file_search``,
    …)."""
    return [spec.as_openai_tool() for spec in build_tool_specs()]


# Names of OpenAI Responses-API built-in tools the lockdown forbids.
# Any future built-in additions should be added here AND named in the
# lockdown test so a regression is caught at the API surface, not at
# runtime.
FORBIDDEN_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "web_search",
        "web_search_preview",
        "computer_use_preview",
        "code_interpreter",
        "file_search",
        "image_generation",
        "mcp",  # remote MCP servers; Sift's MCP runs in-process only
    }
)


def _verify_lockdown(tools: list[dict[str, Any]]) -> None:
    """Last-line guard: raise if anything in ``tools`` is a built-in
    type or otherwise off-allowlist. This is checked on every
    request, not just at startup, because a tool list that round-trips
    through serialisation could in principle pick up extra entries.

    The non-function check has to enumerate ``FORBIDDEN_BUILTIN_TYPES``
    explicitly before falling through to the generic non-function
    error: every entry in that frozenset is itself a non-function
    type, so a single ``ttype != "function"`` branch would fire on
    them too — but with a generic message that hides which
    forbidden built-in was attempted. Test cases that pin the
    "forbidden built-in {name}" message would never see it.
    """
    expected_names = {s.name for s in build_tool_specs()}
    seen_names: list[str] = []
    for t in tools:
        ttype = t.get("type")
        # Forbidden built-ins first, so the error names the specific
        # built-in (web_search / code_interpreter / mcp / ...) rather
        # than the generic "non-function entry" message that would
        # otherwise cover the same input.
        if ttype in FORBIDDEN_BUILTIN_TYPES:
            raise RuntimeError(f"Sift lockdown violation: forbidden built-in {ttype!r}")
        if ttype != "function":
            raise RuntimeError(
                f"Sift lockdown violation: tools list contains "
                f"non-function entry of type {ttype!r}"
            )
        if t.get("name") not in expected_names:
            raise RuntimeError(
                f"Sift lockdown violation: unknown function tool {t.get('name')!r}"
            )
        seen_names.append(t["name"])
    if len(seen_names) != len(expected_names) or set(seen_names) != expected_names:
        raise RuntimeError(
            "Sift lockdown violation: function tool set is incomplete or duplicated"
        )


# ---------------------------------------------------------------------------
# Few-shot turn (structural, not prompt text)
# ---------------------------------------------------------------------------


# One demonstration exchange prepended to the very first round of every
# new session, BEFORE the real user message. The Responses API treats
# these items the same as real prior conversation: the model sees
# ``submit_script`` returning a payload whose ``markdown`` field is a
# pipe table, and an assistant reply that pastes that table verbatim
# followed by a single short sentence of interpretation. The point is
# behavioral demonstration of the desired output shape, parallel to the
# tool descriptions that say "Drop the markdown directly into your
# reply" for ``compose_results`` / ``expand_result``.
#
# Why few-shot rather than another system-prompt rule. Rules describe
# the desired behavior; demonstrations are the behavior. Demonstrations
# carry stronger pull because the model has now SEEN the pattern in its
# own conversation history. The same pull is unavailable here on the
# Anthropic provider because the Claude Agent SDK doesn't expose a
# point to seed prior assistant / tool_result turns; that path uses an
# embedded example in ``_STYLE_RIDER`` instead.
#
# Token cost. ~350 input tokens on round 1. It remains in the local replay
# history for subsequent rounds and turns, so it is inserted only once.
#
# Stability of call_id. The ``fewshot_call_1`` id is a literal string
# the model never produces (handler dispatch is name-keyed, not id-
# keyed). It cannot collide with a real call because real ids come
# from the OpenAI server side and use a different format.
#
# Opt out for A/B testing: ``SIFT_DISABLE_FEWSHOT=1``.
_FEWSHOT_USER_TEXT = "What's the breakdown of `treatment` in this sample?"

_FEWSHOT_TOOL_ARGS = json.dumps(
    {
        "language": "stata",
        "code": 'sift_result_tab treatment, label("Treatment assignment")',
        "label": "Treatment frequency",
    }
)

_FEWSHOT_TOOL_OUTPUT = json.dumps(
    {
        "results": [
            {
                "status": "ok",
                "result_id": "r_demo_treatment_freq",
                "label": "Treatment assignment",
                "type": "frequency_table",
                "markdown": (
                    "| treatment | n   | %    |\n"
                    "| --------- | --- | ---- |\n"
                    "| control   | 487 | 49.4 |\n"
                    "| treated   | 499 | 50.6 |"
                ),
            }
        ]
    }
)

_FEWSHOT_ASSISTANT_TEXT = (
    "| treatment | n   | %    |\n"
    "| --------- | --- | ---- |\n"
    "| control   | 487 | 49.4 |\n"
    "| treated   | 499 | 50.6 |\n"
    "\n"
    "Balanced 50/50 assignment, n=986."
)


def _build_fewshot_items() -> list[dict[str, Any]]:
    """Return the few-shot exchange as Responses-API input items.

    Order matches a real prior turn: user message, function_call,
    function_call_output, assistant message. Round-trips byte-for-byte
    with what the server would emit for the same exchange, which is
    why the demonstration registers as "real history" rather than a
    style instruction.
    """
    return [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": _FEWSHOT_USER_TEXT}],
        },
        {
            "type": "function_call",
            "call_id": "fewshot_call_1",
            "name": "submit_script",
            "arguments": _FEWSHOT_TOOL_ARGS,
        },
        {
            "type": "function_call_output",
            "call_id": "fewshot_call_1",
            "output": _FEWSHOT_TOOL_OUTPUT,
        },
        {
            "type": "message",
            "role": "assistant",
            "phase": "final_answer",
            "content": [{"type": "output_text", "text": _FEWSHOT_ASSISTANT_TEXT}],
        },
    ]


def _fewshot_enabled() -> bool:
    """``SIFT_DISABLE_FEWSHOT=1`` opts out for A/B testing."""
    return os.environ.get("SIFT_DISABLE_FEWSHOT") != "1"


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class OpenAISession:
    """OpenAI Responses-API session.

    Maintains an in-memory ``_input`` list of messages for the
    duration of the session — this is what makes multi-turn
    conversation work (each ``send()`` appends to it, the next
    ``send()`` sends the whole history). The bridge's session
    lifecycle therefore IS the conversation lifecycle on this
    provider.
    """

    PROVIDER = PROVIDER_ID
    PROVIDER_LABEL = "OpenAI"

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
        # Reasoning effort — sent per request in ``reasoning.effort``
        # (see the send loop), so a change applies on the next
        # message with no client rebuild and no conversation reset.
        self.effort: str = clamp_effort(effort, self.PROVIDER)
        # ``continue_conversation`` is Anthropic-specific (CLI session
        # store); accepted for interface symmetry but ignored here.
        del continue_conversation
        self._client: Any = None
        # Store=false conversation state. Complete Responses items
        # live only in this provider process and are cleared on close.
        self._history_items: list[dict[str, Any]] = []
        # Cached tool list — same function tools for every call.
        # Built once at open() rather than per-send so the lockdown
        # check has a stable reference.
        self._tools: list[dict[str, Any]] = []

    def _resolve_session_credential(self) -> str | None:
        return _resolve_api_key()

    def _client_options(self, credential: str) -> dict[str, Any]:
        """Arguments for ``AsyncOpenAI``; managed subclasses override this."""
        from sift.integration_core import (
            MODEL_REQUEST_TIMEOUT_SECONDS,
            MODEL_SDK_MAX_RETRIES,
        )

        return {
            "api_key": credential,
            "timeout": MODEL_REQUEST_TIMEOUT_SECONDS,
            "max_retries": MODEL_SDK_MAX_RETRIES,
        }

    def _wire_model_id(self) -> str:
        return self.model

    def _missing_auth_reason(self) -> str:
        return (
            "no OpenAI API key configured. Add one in the auth screen or "
            "set OPENAI_API_KEY in the environment."
        )

    def _translate_failure(self, error: Exception, safe_message: str) -> Event:
        return _translate_request_failure(
            error,
            model_id=self._wire_model_id(),
            safe_message=safe_message,
        )

    def _error_secrets(self) -> tuple[str | None, ...]:
        return (self._resolve_session_credential(),)

    # ---- lifecycle -------------------------------------------------------

    async def open(self) -> None:
        """Lazy-build the AsyncOpenAI client if a key is available.

        Missing-key is NOT raised here: the provider contract terminates
        a turn with an ``AuthFailure`` event, and ``open()`` runs from
        ``SessionRunner.ensure_session`` BEFORE any event stream
        exists, so a ``RuntimeError`` at this layer would surface as a
        generic ``turn_error`` instead. To stay parity with the
        Anthropic path (whose ``open()`` doesn't fail on missing key
        either), we no-op here on a missing key and let
        :meth:`send` yield ``AuthFailure`` on the first round.
        """
        if self._client is not None:
            return
        # Lazy import: keeps the openai SDK off Anthropic-only code
        # paths and means a missing dep manifests as a clean
        # AuthFailure instead of an import error at startup.
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover — openai is a dep
            raise RuntimeError(
                f"openai SDK not installed: {e}. Reinstall Sift dependencies."
            )
        api_key = self._resolve_session_credential()
        if not api_key:
            # Defer: ``send()`` checks ``self._client`` and emits
            # ``AuthFailure`` on the first round, matching how
            # Anthropic surfaces a missing key.
            return
        self._client = AsyncOpenAI(**self._client_options(api_key))
        self._tools = build_openai_tools()
        # Lockdown verified at session open AND at every send (defense
        # in depth — a future caller could in principle mutate
        # _tools).
        _verify_lockdown(self._tools)

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._history_items = []
        self._tools = []
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 — close-time errors aren't useful
                pass

    # ---- model swap ------------------------------------------------------

    async def set_model(self, model_id: str) -> dict[str, Any]:
        """Switch active OpenAI model. The Responses API accepts the
        model id per request, so swapping is just a field change —
        no client rebuild. Conversation state is preserved."""
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
                    f"{info.provider!r}, not OpenAI"
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
        self.model = model_id
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
        }

    def _reasoning_params(self) -> dict[str, Any]:
        """Translate the picker's effort level into the Responses
        API's reasoning parameters.

        The bar shows one ladder — low … xhigh … max … pro — but OpenAI
        splits that across two independent knobs, so the top rung
        needs unpacking:

        - ``low``/``medium``/``high``/``xhigh``/``max`` → ``reasoning.effort``
          verbatim, standard mode.
        - ``pro`` → ``reasoning.mode="pro"`` (more model work per
          turn) *plus* ``reasoning.effort="max"``. Mode and effort
          are independent in the API and effort would otherwise
          default to ``medium`` in pro mode — which would make the
          bar's top rung reason *less* than the rung below it. Pairing
          pro with the highest supported effort is what
          makes the ladder monotonic, which is the whole promise of
          rendering it as a bar.

        ``context="all_turns"`` asks GPT-5.6 to preserve reasoning
        items across the locally replayed, ``store=false`` conversation,
        improving multi-turn reasoning without exposing hidden chain of
        thought: Sift still surfaces only provider-generated summaries.
        """
        params: dict[str, Any] = {"summary": "auto", "context": "all_turns"}
        if self.effort == "pro":
            params["effort"] = "max"
            params["mode"] = "pro"
        else:
            params["effort"] = self.effort
        return params

    async def set_effort(self, effort: str) -> dict[str, Any]:
        """Switch reasoning effort. Sent per request, so this is a
        field change that takes effect on the next message — the
        locally replayed conversation is untouched; no reopen is needed."""
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

    # ---- send ------------------------------------------------------------

    async def send(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one chat turn. Yields a flat stream of provider-neutral
        events terminated by ``TurnDone`` / ``TurnError`` /
        ``AuthFailure``.

        Internally drives the OpenAI tool-loop: each round-trip is one
        ``responses.create()`` call. If the response contains
        ``function_call`` items, dispatch them via ``HANDLERS``,
        append outputs to ``_input``, and loop. Loop exits when the
        response has no further function calls.
        """
        await self.open()
        client = self._client
        if client is None:
            # ``open()`` declined to build a client because no API key
            # is configured. Yield the provider-neutral ``AuthFailure``
            # event so the runner emits ``auth_failure`` — same shape
            # the API-call path uses for 401s further down. Without
            # this branch a missing key crashed ``ensure_session`` and
            # surfaced as a generic ``turn_error``, breaking parity
            # with the Anthropic provider.
            yield AuthFailure(reason=self._missing_auth_reason())
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

        # Round-1 input is just the new user message. Subsequent
        # rounds inside this turn carry only the function-call outputs
        # we produce locally — the prior assistant / function_call /
        # reasoning items already live on the server, reachable via
        # the response-id chain.
        user_content = _build_user_content(prompt, images)
        pending_input: list[dict[str, Any]] = [
            {"role": "user", "content": user_content},
        ]

        # First-turn-only: prepend the demonstration to local history.
        if not self._history_items and _fewshot_enabled():
            pending_input = _build_fewshot_items() + pending_input

        # Provisional local items are committed only after a clean terminal
        # response, so failed/cancelled tool loops cannot poison history.
        turn_items: list[dict[str, Any]] = []

        # Bound the tool-loop iterations so a runaway model can't pin
        # the loop forever. 16 is generous — most analyses use 1–4.
        # Env-overridable via ``SIFT_OPENAI_MAX_TOOL_ROUNDS`` for
        # researchers running parameterised batches that legitimately
        # need >16 rounds (e.g. 24 specs each requiring a
        # submit_script + a few expand_result rounds + a
        # compose_results). Bounded to [1, 64]: lower than 1 makes no
        # sense, higher than 64 is past the point where the user
        # would rather see "stop and ask for guidance" than continue
        # autonomously. Invalid env values fall back to the default
        # rather than failing the turn — a typo in the env var
        # shouldn't block a researcher mid-analysis.
        try:
            MAX_TOOL_ROUNDS = int(os.environ.get("SIFT_OPENAI_MAX_TOOL_ROUNDS", "16"))
            if not 1 <= MAX_TOOL_ROUNDS <= 64:
                MAX_TOOL_ROUNDS = 16
        except (TypeError, ValueError):
            MAX_TOOL_ROUNDS = 16
        # We capture the LAST round's prompt size (not a cross-round
        # sum) because the Responses API reports ``input_tokens`` for
        # the FULL prompt at each round, including locally replayed history.
        # Within a tool-loop turn the prompt grows monotonically as
        # tool outputs join the chain, so the last round naturally
        # captures the peak prompt size for the turn. Accumulating
        # with ``+=`` (the previous behavior) multi-counted the cached
        # prefix once per round, inflating the chip into nonsense on
        # tool-heavy turns.
        last_input_tokens = 0
        last_output_tokens = 0

        try:
            for _round in range(MAX_TOOL_ROUNDS):
                _verify_lockdown(self._tools)
                request_kwargs: dict[str, Any] = {
                    "model": self._wire_model_id(),
                    "instructions": self._system_prompt,
                    "input": self._history_items + turn_items + pending_input,
                    "tools": self._tools,
                    "tool_choice": "auto",
                    # Pin to True explicitly. The Responses API has
                    # defaulted this to True historically — and the
                    # docstring above relies on that — but a silent
                    # SDK/API default change would otherwise break
                    # Sift's tool-loop ergonomics without warning.
                    # Pinning surfaces the dependency at the request
                    # boundary and the lockdown test asserts it.
                    "parallel_tool_calls": True,
                    # Reasoning controls — the OpenAI analogue of the
                    # Anthropic provider's effort +
                    # thinking.display="summarized" pinning. Built by
                    # ``_reasoning_params`` because the picker's top
                    # rung (``pro``) is not an effort value at all;
                    # see that method. ``summary="auto"`` rides along
                    # so the thinking trace below has something to
                    # surface ("concise" is NOT supported by the
                    # gpt-5 series; "auto" lets the server pick).
                    "reasoning": self._reasoning_params(),
                    # Privacy boundary: context is replayed from process
                    # memory and every eligible request receives store=false.
                    "store": False,
                    # Surface context-window overruns as errors instead
                    # of letting the server silently drop the oldest
                    # items in the chain. With ``truncation="auto"``
                    # (which is the API default in some SDK versions),
                    # ``usage.input_tokens`` reports the truncated
                    # prompt size — making the context chip read
                    # smaller than the actual conversation, then
                    # smaller still as more turns get truncated. The
                    # chip's whole point is honesty about how full the
                    # window is; silent truncation defeats it. If we
                    # ever want to allow truncation as a UX choice,
                    # surface it in settings rather than baking it in
                    # at the request boundary.
                    "truncation": "disabled",
                }
                request_kwargs["include"] = ["reasoning.encrypted_content"]
                try:
                    resp = await client.responses.create(**request_kwargs)
                except Exception as e:  # noqa: BLE001 — translate to event
                    msg = provider_error_message(e, secrets=self._error_secrets())
                    yield self._translate_failure(e, msg)
                    return

                # Track usage. ``=`` not ``+=`` (see the rationale
                # above where last_input_tokens is initialized).
                # ``input_tokens`` already includes the cached prefix
                # for this round, so the last round's value is the
                # full prompt size at end-of-turn — exactly what the
                # "context occupied" chip wants.
                usage = getattr(resp, "usage", None)
                if usage is not None:
                    last_input_tokens = getattr(usage, "input_tokens", 0) or 0
                    last_output_tokens = getattr(usage, "output_tokens", 0) or 0
                # Diagnostic: gated by SIFT_DEBUG_USAGE. Mirrors the
                # Anthropic provider's logging so a head-to-head
                # comparison of the two providers' token accounting is
                # possible from the same on-disk file. Writes to
                # stderr (visible if launched from a terminal) AND
                # appends to ``<cwd>/.sift-usage.log`` (always reachable
                # by the researcher regardless of launch method —
                # pywebview swallows stderr on a double-clicked app).
                if os.environ.get("SIFT_DEBUG_USAGE") == "1" and usage is not None:
                    from sift.provider.usage_log import append_usage_line

                    cached = (
                        getattr(
                            getattr(usage, "input_tokens_details", None),
                            "cached_tokens",
                            0,
                        )
                        or 0
                    )
                    line = (
                        f"[sift.usage.openai] round model={self.model} "
                        f"input_tokens={last_input_tokens} "
                        f"output_tokens={last_output_tokens} "
                        f"cached_tokens={cached} "
                        f"(cached is a SUBSET of input_tokens, not additive)"
                    )
                    print(line, file=sys.stderr, flush=True)
                    append_usage_line(self.cwd, line)

                output = list(getattr(resp, "output", []) or [])
                # Provisional until the turn ends cleanly. A failed or
                # exhausted tool loop leaves committed history unchanged.
                turn_items.extend(pending_input)
                turn_items.extend(_dump_response_item(item) for item in output)
                # Translate items and decide whether to keep looping.
                pending_calls: list[
                    tuple[str, str, str]
                ] = []  # (call_id, name, args_json)
                for item in output:
                    itype = getattr(item, "type", None)
                    if itype == "message":
                        text = _extract_message_text(item)
                        if text and text.strip():
                            yield AssistantText(text=text)
                    elif itype == "function_call":
                        name = getattr(item, "name", "")
                        call_id = getattr(item, "call_id", "") or getattr(
                            item, "id", ""
                        )
                        args_json = getattr(item, "arguments", "") or "{}"
                        pending_calls.append((call_id, name, args_json))
                        yield ToolCall(
                            name=name,
                            input=_safe_json(args_json),
                            call_id=call_id,
                        )
                    elif itype == "reasoning":
                        # Surface the model's reasoning SUMMARY (requested
                        # via reasoning.summary="auto" above) as a thinking
                        # trace, mirroring the Anthropic provider's
                        # ThinkingBlock -> AssistantThinking translation so
                        # the UI's thinking panel populates on both
                        # providers. We read ``summary`` (a list of
                        # ``{type:"summary_text", text:...}`` parts), NOT the
                        # raw ``content``/``encrypted_content`` — OpenAI's
                        # policy only sanctions the summary surface, and
                        # summaries aren't emitted every round, so the
                        # strip()-guard keeps empties out. The item itself is
                        # still carried forward server-side via the
                        # response-id chain; this is display-only.
                        summary = getattr(item, "summary", None) or []
                        trace = "".join(
                            getattr(part, "text", "") or ""
                            for part in summary
                            if getattr(part, "type", None) == "summary_text"
                        )
                        if trace.strip():
                            yield AssistantThinking(text=trace)
                    # Any other item types are held server-side and
                    # carried forward implicitly by the chain — no
                    # local tracking needed.

                # A Responses call can return normally at the HTTP layer
                # while the generated response itself is incomplete,
                # failed, or cancelled. Never dispatch tool calls or commit
                # conversation state from such a response. Partial text and
                # reasoning summaries above remain visible, followed by an
                # honest terminal error.
                response_status = getattr(resp, "status", None)
                if response_status in {"incomplete", "failed", "cancelled"}:
                    detail_obj = (
                        getattr(resp, "incomplete_details", None)
                        if response_status == "incomplete"
                        else getattr(resp, "error", None)
                    )
                    detail = (
                        getattr(detail_obj, "reason", None)
                        or getattr(detail_obj, "message", None)
                        or getattr(detail_obj, "code", None)
                    )
                    safe_detail = (
                        provider_error_message(
                            detail,
                            secrets=self._error_secrets(),
                        )
                        if detail
                        else "the provider did not return a reason"
                    )
                    yield TurnError(
                        message=(
                            f"OpenAI response {response_status}: {safe_detail}. "
                            "No tool calls or conversation state from this "
                            "response were committed; retry the turn or reduce "
                            "its requested output."
                        )
                    )
                    return

                if not pending_calls:
                    # Clean turn end. Commit local items and emit TurnDone —
                    # falling through to the post-loop branch would also
                    # emit TurnDone on the exhausted-rounds path, which
                    # silently truncates a model that's still requesting
                    # tools. That path is now treated as TurnError below.
                    self._history_items.extend(turn_items)
                    yield TurnDone(
                        input_tokens=last_input_tokens,
                        output_tokens=last_output_tokens,
                        # Cache fields stay None on this path. OpenAI's
                        # ``cached_tokens`` is a subset of ``input_tokens``
                        # (not additive), so emitting it through
                        # ``cache_read_input_tokens`` would double-count
                        # any consumer that reads the breakdown directly.
                        # ``input_tokens`` already represents the full
                        # prompt size on the OpenAI path; ``post_turn_tokens``
                        # adds output for the canonical post-turn snapshot.
                        # ``cost_usd`` is also None — Sift doesn't compute
                        # OpenAI costs locally.
                        post_turn_tokens=last_input_tokens + last_output_tokens,
                    )
                    return

                # Dispatch every function_call in this round (parallel-
                # friendly but executed serially here; the underlying
                # handlers aren't expected to be concurrency-safe yet).
                # The next round carries these outputs with local history.
                next_input: list[dict[str, Any]] = []
                for call_id, name, args_json in pending_calls:
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
                        next_input.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": out_text,
                            }
                        )
                        continue
                    args, parse_error = _parse_tool_args(args_json)
                    if parse_error is not None:
                        # Malformed args — surface as an explicit tool
                        # error so the model fixes its JSON instead of
                        # being told "missing required arg X" by the
                        # handler's schema layer (which is what
                        # happened when we silently coerced bad JSON
                        # to ``{}``). The handler is NOT invoked in
                        # this branch.
                        out_text = json.dumps(
                            {
                                "status": "error",
                                "reason": parse_error,
                            }
                        )
                        yield ToolCallResult(
                            call_id=call_id,
                            text=out_text,
                            is_error=True,
                        )
                        next_input.append(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": out_text,
                            }
                        )
                        continue
                    try:
                        result = await handler(args)
                        out_text = _mcp_payload_to_text(result)
                        is_error = False
                    except Exception as e:  # noqa: BLE001
                        # Catch-all fallback: a per-tool handler raised
                        # an unexpected exception. The exception
                        # message itself may include parser excerpts,
                        # file paths, or raw data values the tool's
                        # curated error-handling would have redacted —
                        # interpolating ``str(e)`` into the model-
                        # visible reason bypasses the per-tool
                        # disclosure contract. Log the full details
                        # locally for debugging and return only the
                        # exception CLASS (a bounded identifier) plus
                        # a generic recovery hint.
                        # Full message + traceback are gated behind
                        # SIFT_DEBUG_USAGE, matching every other
                        # diagnostic-logging path in this file: ``e``
                        # and the traceback can both carry the exact
                        # parser excerpts / file paths / raw data
                        # values the comment above says must not reach
                        # the model, and append_usage_line persists to
                        # a file on disk (``.sift-usage.log`` in the
                        # project directory) -- writing that content
                        # there unconditionally, for every researcher,
                        # on every unexpected tool error, was itself a
                        # local disclosure-boundary violation, not
                        # just a debug-verbosity choice.
                        if os.environ.get("SIFT_DEBUG_USAGE") == "1":
                            import traceback

                            diag = (
                                f"[sift.openai] tool {name!r} handler "
                                f"raised {e.__class__.__name__}: {e}\n"
                                + traceback.format_exc()
                            )
                        else:
                            diag = (
                                f"[sift.openai] tool {name!r} handler "
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
                    next_input.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": out_text,
                        }
                    )
                pending_input = next_input

            # Loop exhausted — the model kept requesting tools through
            # MAX_TOOL_ROUNDS without producing a final non-tool
            # response. Surface as TurnError so the caller sees the
            # truncation rather than a misleading "clean done".
            #
            # The final response carries an unsatisfied function call and its
            # result was never sent back. Provisional ``turn_items`` therefore
            # remain uncommitted; the next turn replays the last clean state.
            yield TurnError(
                message=(
                    f"OpenAI tool loop did not converge within "
                    f"{MAX_TOOL_ROUNDS} rounds; last response still had "
                    f"pending function calls."
                ),
            )
        except Exception as e:  # noqa: BLE001 — last-line catch
            msg = provider_error_message(e, secrets=self._error_secrets())
            yield TurnError(message=f"{self.PROVIDER_LABEL} session error: {msg}")

    # ---- async-context-manager sugar ------------------------------------

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dump_response_item(item: Any) -> dict[str, Any]:
    """Convert one SDK output item into a replayable Responses input item.

    Current Pydantic SDK objects accept ``exclude_none``. Test doubles and
    older compatible objects may expose a no-argument ``model_dump`` only,
    so the fallback is intentional.
    """
    dump = getattr(item, "model_dump", None)
    if not callable(dump):
        raise TypeError(
            f"OpenAI response item {type(item).__name__} is not serializable"
        )
    try:
        value = dump(exclude_none=True)
    except TypeError:
        value = dump()
    if not isinstance(value, dict):
        raise TypeError(
            f"OpenAI response item {type(item).__name__} did not serialize to an object"
        )
    return value


def _build_user_content(
    prompt: str, images: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    """Build a Responses-API user message content list. Text is one
    ``input_text`` block; each image is one ``input_image`` block
    with a base64 data URL."""
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "input_text", "text": prompt})
    if images:
        for img in images:
            mime = img.get("mime", "image/png")
            data = img.get("data", "")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{mime};base64,{data}",
                }
            )
    return content


def _extract_message_text(item: Any) -> str:
    """Pull the concatenated output text from a Responses-API message
    item. Items may carry multiple text blocks (e.g., when reasoning
    interleaves)."""
    parts: list[str] = []
    content = getattr(item, "content", None) or []
    for block in content:
        btype = getattr(block, "type", None)
        if btype in ("output_text", "text"):
            t = getattr(block, "text", None) or ""
            if t:
                parts.append(t)
    return "".join(parts)


def _safe_json(s: str) -> dict[str, Any]:
    """Best-effort decode used for the ``ToolCall`` audit event.

    Display-side only — the audit event renders whatever args the
    model thought it was sending, and a non-empty malformed string
    degrades to ``{}`` so the chat panel still shows the call. The
    actual handler dispatch goes through :func:`_parse_tool_args`,
    which surfaces malformed JSON as an explicit tool error so the
    model isn't told "missing required arg" when its real problem
    was bad JSON.
    """
    if not s:
        return {}
    try:
        out = json.loads(s)
    except (ValueError, TypeError):
        return {}
    if not isinstance(out, dict):
        return {}
    return out


def _parse_tool_args(s: str) -> tuple[dict[str, Any] | None, str | None]:
    """Decode a ``function_call.arguments`` string for handler dispatch.

    Returns ``(args, None)`` on success and ``(None, reason)`` when
    the string is non-empty but not a valid JSON object. The empty
    string maps to ``({}, None)`` because OpenAI emits ``""`` for
    zero-arg calls, and that's a legitimate shape for the small
    handful of Sift tools that take no arguments.

    The previous behaviour silently coerced malformed JSON to ``{}``
    and dispatched the handler anyway. The handler's required-arg
    validator then complained about missing fields, which told the
    model the wrong story: it thought it had forgotten ``code`` /
    ``language`` when the real failure was that its JSON didn't
    parse. The model retried with the same broken serialiser and
    burned a turn. Returning a parse error here lets the caller
    emit an explicit "tool arguments were not valid JSON" result so
    the model fixes the actual problem on the next round.

    Non-dict top-level values (a bare list, string, or number from
    the model's perspective is "I sent some args" without a key, so
    we still call out the shape mismatch.
    """
    if not s:
        return {}, None
    try:
        out = json.loads(s)
    except (ValueError, TypeError) as e:
        # Truncate the offending payload so a model that emitted a
        # multi-MB malformed blob doesn't blow up the error-message
        # context. ``json.JSONDecodeError`` carries position info
        # that helps the model self-correct on the retry.
        #
        # Strip non-printables BEFORE the 120-char cap so the
        # truncation cap actually bounds output size. ``repr()``
        # on bidi overrides or control chars expands each
        # character to ``\\u202e`` / ``\\x07`` (4–6 chars), so a
        # raw 120-char snippet of pathological input can render as
        # 700+ chars after ``!r`` — defeating the cap whose entire
        # job is bounding error-message size. Replacing
        # non-printables with ``?`` before the slice keeps the cap
        # honest and still lets the model see the rough shape of
        # the JSON it tried to send.
        cleaned = "".join(c if (c.isprintable() or c in "\t ") else "?" for c in s)
        snippet = cleaned[:120] + ("…" if len(cleaned) > 120 else "")
        return None, (f"tool arguments were not valid JSON: {e}; received {snippet!r}")
    if not isinstance(out, dict):
        return None, (f"tool arguments must be a JSON object, got {type(out).__name__}")
    return out, None


def _payload_has_image_block(payload: Any) -> bool:
    """Whether an MCP-shaped tool result carries an inline image block.

    ``read_attached_file`` returns ``{"type": "image", "data": ...}``
    alongside a text descriptor when the user recalls a PNG / PDF /
    EPS. The Anthropic dispatcher forwards both blocks; the OpenAI
    Responses API's ``function_call_output`` only takes a single
    string, so the image bytes can't ride with the tool result. We
    detect the image-block case so the descriptor we DO send tells
    the model definitively that pixel data was dropped on this path.
    """
    if not isinstance(payload, dict):
        return False
    content = payload.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if isinstance(block, dict) and block.get("type") == "image":
            return True
    return False


def _mcp_payload_to_text(payload: Any) -> str:
    """Sift handlers return MCP-shaped payloads:
    ``{"content": [{"type": "text", "text": "..."}]}``. Extract the
    JSON-text body for the OpenAI tool output.

    For payloads that carry an image content block alongside a text
    descriptor (``read_attached_file`` for PNG / PDF recall), rewrite
    the text descriptor so the model knows the pixels were dropped on
    this provider — without that, the descriptor's hedged "if your
    provider doesn't support images" hint is the only signal, and the
    model may still try to reason about the (absent) bytes. The
    Anthropic path is unaffected — its content list survives intact
    on its own dispatcher.
    """
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
        # Already-flat dict: serialise.
        return json.dumps(payload)
    if isinstance(payload, str):
        return payload
    return json.dumps(payload)


def _rewrite_for_dropped_image(descriptor_text: str) -> str:
    """Replace ``read_attached_file``'s hedged image descriptor with an
    OpenAI-specific reason telling the model the pixels weren't sent.

    The original descriptor (see ``read_attached_file`` in
    ``tools.py``) reads "If your provider doesn't support image tool
    results, ask the researcher to re-@mention the file …". On this
    provider it definitely doesn't, so we promote that conditional
    note to the primary status and keep the file metadata so the
    model can name the file precisely in its follow-up message to the
    researcher.

    Falls back to the original text if the descriptor isn't the JSON
    shape we expect — never raises.
    """
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
    # Drop None fields so the response stays tight.
    rewritten = {k: v for k, v in rewritten.items() if v is not None}
    return json.dumps(rewritten, separators=(",", ":"), ensure_ascii=False)


def _extract_hints(text: str) -> tuple[str | None, str | None]:
    """Same hint-extraction as the Anthropic side: peel ``_run_dir``
    and ``_language`` from the tool-result JSON if present so the UI
    can render raw R/Stata output and the right "Open in …" button."""
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
