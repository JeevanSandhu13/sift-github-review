"""OpenAI-compatible endpoint provider (Chat Completions API).

Covers local models and third-party gateways: Ollama, vLLM, LM
Studio, llama.cpp server, text-generation-webui, OpenRouter,
Together, Groq, and anything else that speaks the Chat Completions
wire protocol (``POST /v1/chat/completions``) -- the actual
"OpenAI-compatible" lingua franca. This is deliberately NOT built on
the Responses API that ``provider/openai.py`` uses: the Responses API
is OpenAI-proprietary and essentially no third-party server
implements it, so a provider meant to cover local/self-hosted models
has to speak Chat Completions instead.

Configuration is env-var-driven, mirroring the ``OPENAI_API_KEY``
fallback pattern the OpenAI provider already uses for power users:

- ``SIFT_OPENAI_COMPATIBLE_BASE_URL`` (required) -- e.g.
  ``http://localhost:11434/v1`` (Ollama), ``http://localhost:8000/v1``
  (vLLM), ``https://openrouter.ai/api/v1`` (OpenRouter).
- ``SIFT_OPENAI_COMPATIBLE_MODEL`` (required) -- the model name/id
  the target server expects (e.g. ``llama3.1``, ``qwen2.5:32b``,
  ``meta-llama/Llama-3.1-70B-Instruct``).
- ``SIFT_OPENAI_COMPATIBLE_API_KEY`` (optional) -- many local servers
  (Ollama, vLLM in default config) accept any string or no auth at
  all; gateway services (OpenRouter et al.) require a real key.
  Falls back to the keyring under provider id ``"openai_compatible"``
  if set via the auth screen (``sift.auth``).
- ``SIFT_OPENAI_COMPATIBLE_CONTEXT_WINDOW`` (optional, default
  32000) -- the catalog entry needs SOME context-window number for
  the UI's usage chip; there is no discovery endpoint in the Chat
  Completions spec, so this has to be told rather than queried.

Chat Completions is stateless server-side (no ``previous_response_id``
equivalent) -- this session therefore keeps the FULL message list in
memory (``self._messages``) and resends it on every call, growing
across turns exactly the way every Chat-Completions client works.
This is a real architectural difference from ``OpenAISession``, not
an oversight: there is no cheaper option against this API shape.

Tool dispatch routes through the SAME ``sift.tools.HANDLERS`` map
every other provider uses -- privacy semantics (sandbox execution,
sanitizer-clamped results) are identical regardless of which model or
server authored the call. Several wire-format-agnostic helpers
(argument parsing, MCP-payload-to-text, run-dir/language hint
extraction, dropped-image rewriting, best-effort JSON decode for the
audit event) are imported directly from ``provider.openai`` rather
than duplicated -- they operate on plain JSON strings/dicts and have
nothing Responses-API-specific about them.

Reasoning effort: most self-hosted/open-weight models have no
"effort" dial equivalent to Anthropic's ``output_config.effort`` or
OpenAI's ``reasoning.effort``. ``set_effort`` is therefore a no-op
that always reports success (never blocks the picker) but does not
send any effort-shaped field the target server almost certainly
doesn't understand. Reasoning-capable local models served behind vLLM
(DeepSeek-R1, Qwen3, ...) that DO respect a ``reasoning_effort``-style
field would need it forwarded as an ``extra_body`` kwarg -- left as a
documented gap rather than guessed at without a way to verify a
specific server's behaviour (see the final accounting).

Streaming: this provider does NOT stream tokens. Every provider in
Sift already yields complete text blocks per round rather than
token-by-token deltas (see ``AssistantText``'s docstring in
``base.py``), so a single non-streaming ``chat.completions.create()``
call per round costs nothing in UX and is far more robust across the
wildly heterogeneous SSE implementations different "OpenAI-compatible"
servers ship.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import AsyncIterator
from ipaddress import ip_address
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from typing_extensions import Self

from sift.provider.base import (
    AssistantText,
    AuthFailure,
    Event,
    ToolCall,
    ToolCallResult,
    TurnDone,
    TurnError,
)
from sift.provider.error_safety import provider_error_message
from sift.provider.openai import (
    _extract_hints,
    _mcp_payload_to_text,
    _parse_tool_args,
    _safe_json,
)
from sift.provider.response_limits import read_bounded_async_response
from sift.provider.tool_schemas import build_tool_specs
from sift.tools import HANDLERS

PROVIDER_ID = "openai_compatible"

ENV_BASE_URL = "SIFT_OPENAI_COMPATIBLE_BASE_URL"
ENV_MODEL = "SIFT_OPENAI_COMPATIBLE_MODEL"
ENV_API_KEY = "SIFT_OPENAI_COMPATIBLE_API_KEY"
ENV_CONTEXT_WINDOW = "SIFT_OPENAI_COMPATIBLE_CONTEXT_WINDOW"
ENV_MAX_TOOL_ROUNDS = "SIFT_OPENAI_COMPATIBLE_MAX_TOOL_ROUNDS"
ENV_ALLOW_INSECURE_REMOTE = "SIFT_OPENAI_COMPATIBLE_ALLOW_INSECURE_REMOTE"

DEFAULT_CONTEXT_WINDOW = 32_000

# The raw SDK response is bounded while arriving and the decoded fields are
# checked again before use. Keep a model turn small enough to inspect and
# persist safely, while still allowing the existing attachment contract (up
# to eight 5 MB source images) to fit in one bounded Chat-Completions history.
MAX_COMPATIBLE_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_COMPATIBLE_TOOL_ARGUMENT_BYTES = 512 * 1024
MAX_COMPATIBLE_HISTORY_BYTES = 80 * 1024 * 1024
MAX_COMPATIBLE_TOOL_CALLS = 64
MAX_COMPATIBLE_OUTPUT_TOKENS = 4_096


class _CompatiblePayloadLimitError(ValueError):
    """A safe, value-free failure for endpoint-controlled payloads."""


def _json_wire_size(value: Any, limit: int, *, depth: int = 0) -> int:
    """Measure compact UTF-8 JSON without first materializing the full JSON.

    The recursive structures here are Sift-created message payloads.  A depth
    limit nevertheless makes the helper fail closed if an SDK or test double
    supplies an adversarial object.
    """
    if depth > 64:
        raise _CompatiblePayloadLimitError("payload nesting exceeded safety limit")
    if isinstance(value, str):
        # UTF-8 JSON cannot be shorter than its character count.  This early
        # test avoids making a second enormous copy solely to measure it.
        if len(value) > limit:
            raise _CompatiblePayloadLimitError("payload exceeded safety limit")
        size = len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
    elif value is None:
        size = 4
    elif value is True:
        size = 4
    elif value is False:
        size = 5
    elif isinstance(value, (int, float)):
        size = len(json.dumps(value, allow_nan=False))
    elif isinstance(value, (list, tuple)):
        size = 2
        for index, item in enumerate(value):
            if index:
                size += 1
            if size > limit:
                raise _CompatiblePayloadLimitError("payload exceeded safety limit")
            size += _json_wire_size(item, limit - size, depth=depth + 1)
    elif isinstance(value, dict):
        size = 2
        for index, (key, item) in enumerate(value.items()):
            if not isinstance(key, str):
                raise _CompatiblePayloadLimitError("payload was not JSON-safe")
            if index:
                size += 1
            size += _json_wire_size(key, limit - size, depth=depth + 1) + 1
            if size > limit:
                raise _CompatiblePayloadLimitError("payload exceeded safety limit")
            size += _json_wire_size(item, limit - size, depth=depth + 1)
    else:
        raise _CompatiblePayloadLimitError("payload was not JSON-safe")
    if size > limit:
        raise _CompatiblePayloadLimitError("payload exceeded safety limit")
    return size


def _require_text_size(value: str, limit: int) -> int:
    """Return UTF-8 size or fail before copying an obviously huge string."""
    if len(value) > limit:
        raise _CompatiblePayloadLimitError("payload exceeded safety limit")
    size = len(value.encode("utf-8"))
    if size > limit:
        raise _CompatiblePayloadLimitError("payload exceeded safety limit")
    return size


def _require_history_size(messages: list[dict[str, Any]]) -> None:
    _json_wire_size(messages, MAX_COMPATIBLE_HISTORY_BYTES)


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from either an SDK model or bounded raw JSON object."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


async def _bounded_chat_completion_create(api: Any, **kwargs: Any) -> Any:
    """Perform a Chat Completions call without buffering an untrusted body.

    OpenAI's ``with_streaming_response`` is raw HTTP response streaming; this
    remains a normal non-SSE Chat Completion.  We incrementally cap decoded
    wire bytes, then parse only the bounded JSON.  The fallback exists solely
    for narrow SDK-compatible test doubles and is still protected by the
    decoded object checks in ``send``.
    """
    streaming = getattr(api, "with_streaming_response", None)
    create = getattr(streaming, "create", None)
    if not callable(create):
        return await api.create(**kwargs)
    async with create(**kwargs) as raw_response:
        body = await read_bounded_async_response(
            raw_response,
            max_bytes=MAX_COMPATIBLE_RESPONSE_BYTES,
            label="endpoint response",
        )
    value = json.loads(body)
    if not isinstance(value, dict):
        raise TypeError("endpoint response was not a JSON object")
    return value


# ---------------------------------------------------------------------------
# Configuration + auth
# ---------------------------------------------------------------------------


def detect_auth() -> str:
    """Return an auth/configuration mode only when the endpoint is usable.

    A key by itself is deliberately not enough: both the base URL and target
    model are required to open a session. ``'endpoint'`` distinguishes an
    auth-free local service from a saved/exported API key so trust surfaces do
    not falsely claim that a credential exists.
    """
    if not configuration_issues():
        return "api_key" if _resolve_api_key() else "endpoint"
    return "unknown"


def configuration_issues() -> list[str]:
    """Return missing non-secret settings without exposing their values."""
    issues: list[str] = []
    base_url = _resolve_base_url()
    if not base_url:
        issues.append("base_url_required")
    else:
        issues.extend(_base_url_issues(base_url))
    if not _resolve_model_name():
        issues.append("model_name_required")
    return issues


def _base_url_issues(base_url: str) -> list[str]:
    """Validate the data-destination boundary for a compatible endpoint.

    Remote plaintext HTTP would expose prompts, tool results, and API keys in
    transit. It is rejected by default while loopback HTTP remains available
    for Ollama/vLLM/LM Studio. An explicit environment opt-in covers advanced
    private-network deployments whose transport security is handled elsewhere.
    """
    if base_url != base_url.strip() or any(ord(ch) < 32 for ch in base_url):
        return ["base_url_invalid"]
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return ["base_url_invalid"]
    del port  # Access validates a malformed/out-of-range port.
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return ["base_url_invalid"]
    if parsed.username is not None or parsed.password is not None:
        return ["base_url_userinfo_forbidden"]
    if parsed.query or parsed.fragment:
        return ["base_url_query_or_fragment_forbidden"]
    host = parsed.hostname.rstrip(".").lower()
    is_loopback = host == "localhost" or host.endswith(".localhost")
    if not is_loopback:
        try:
            is_loopback = ip_address(host).is_loopback
        except ValueError:
            pass
    if (
        parsed.scheme == "http"
        and not is_loopback
        and os.environ.get(ENV_ALLOW_INSECURE_REMOTE) != "1"
    ):
        return ["insecure_remote_http_requires_explicit_opt_in"]
    return []


def validate_base_url(base_url: str) -> tuple[str, ...]:
    """Public, side-effect-free validation for compatible endpoints.

    Discovery and setup diagnostics must apply exactly the same destination
    rules as a real model turn.  Keeping this small public wrapper prevents
    probes and future setup surfaces from copying (and eventually drifting
    from) the provider's security boundary.
    """
    return tuple(_base_url_issues(base_url))


def _resolve_base_url() -> str | None:
    return os.environ.get(ENV_BASE_URL) or None


def _resolve_model_name() -> str | None:
    return os.environ.get(ENV_MODEL) or None


def _resolve_api_key() -> str | None:
    from sift import auth as _auth

    return _auth.resolve_provider_credential(PROVIDER_ID, (ENV_API_KEY,))


def resolve_context_window() -> int:
    """The catalog's context-window number for this provider's one
    entry. There is no discovery endpoint in the Chat Completions
    spec to query this from the server, so it is configured, with a
    conservative default sized for a typical local-model deployment
    rather than assuming a frontier-scale window."""
    raw = os.environ.get(ENV_CONTEXT_WINDOW)
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return DEFAULT_CONTEXT_WINDOW


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def build_chat_completion_tools() -> list[dict[str, Any]]:
    """Build the Chat Completions tool list from the canonical spec.
    Returns exactly the Sift tools defined in
    ``sift.provider.tool_schemas.TOOL_SPECS`` -- same lockdown
    discipline as the Responses-API builder in ``provider/openai.py``,
    just in the nested-under-``function`` shape Chat Completions
    expects."""
    return [spec.as_chat_completion_tool() for spec in build_tool_specs()]


def _verify_lockdown(tools: list[dict[str, Any]]) -> None:
    """Same lockdown discipline as
    ``provider.openai._verify_lockdown``, adapted for the Chat
    Completions tool shape (the function name lives under
    ``t["function"]["name"]``, not top-level ``t["name"]``)."""
    expected_names = {s.name for s in build_tool_specs()}
    seen_names: list[str] = []
    for t in tools:
        if t.get("type") != "function":
            raise RuntimeError(
                f"Sift lockdown violation: tools list contains "
                f"non-function entry of type {t.get('type')!r}"
            )
        fn = t.get("function")
        name = fn.get("name") if isinstance(fn, dict) else None
        if name not in expected_names:
            raise RuntimeError(
                f"Sift lockdown violation: unknown function tool {name!r}"
            )
        seen_names.append(name)
    if len(seen_names) != len(expected_names) or set(seen_names) != expected_names:
        raise RuntimeError(
            "Sift lockdown violation: function tool set is incomplete or duplicated"
        )


def _build_user_message(
    prompt: str,
    images: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build a Chat Completions user message. A text-only prompt uses
    the plain-string ``content`` form (maximally compatible across
    servers); a prompt WITH images uses the multi-part content-array
    form most vision-capable compatible servers follow. Not every
    compatible server supports vision -- Sift doesn't try to detect
    that up front, and a server that can't handle the image content
    block is expected to return a clear error rather than being
    silently second-guessed here."""
    if not images:
        return {"role": "user", "content": prompt}
    content: list[dict[str, Any]] = []
    if prompt:
        content.append({"type": "text", "text": prompt})
    for img in images:
        mime = img.get("mime", "image/png")
        data = img.get("data", "")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{data}"},
            }
        )
    return {"role": "user", "content": content}


class OpenAICompatibleSession:
    """Chat Completions session against an arbitrary OpenAI-compatible
    endpoint.

    Unlike the Responses-API-backed ``OpenAISession``, this provider
    is genuinely stateless server-side: the full message history is
    kept in ``self._messages`` and resent on every call. ``model`` is
    accepted for interface symmetry with the other providers, but the
    real model actually invoked is resolved from
    ``SIFT_OPENAI_COMPATIBLE_MODEL`` at ``open()`` time -- the catalog
    id this session is constructed with is always the fixed
    ``openai-compatible-custom`` selector, never a real model name
    (there is no way to enumerate "real" model ids across arbitrary
    servers up front).
    """

    PROVIDER = PROVIDER_ID

    def __init__(
        self,
        cwd: Path,
        model: str,
        system_prompt: str,
        continue_conversation: bool = False,
        effort: str | None = None,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self._system_prompt = system_prompt
        # Anthropic-specific (CLI session store); accepted for
        # interface symmetry, ignored here -- same posture as
        # OpenAISession.
        del continue_conversation
        # Advisory only -- see the module docstring's "Reasoning
        # effort" section. Never rejected, never forwarded.
        self.effort: str = effort or "medium"
        self._client: Any = None
        self._resolved_model: str | None = None
        self._messages: list[dict[str, Any]] = []
        self._tools: list[dict[str, Any]] = []

    @property
    def resolved_model_name(self) -> str | None:
        """The REAL model name actually being invoked against the
        target server, or ``None`` before ``open()`` has resolved it.

        ``self.model`` (set at construction, read by ``set_model``)
        is always the fixed catalog placeholder id
        (``openai-compatible-custom``) — that's what the model picker
        and ``swap_model`` deal in, since there's no way to enumerate
        real model ids across arbitrary servers up front. Usage
        accounting (``runner.py``'s per-turn ``usage_meter.record_turn``
        call) needs the ACTUAL model name instead: recording every
        openai_compatible session's tokens under the literal string
        "openai-compatible-custom" would collapse a researcher's
        distinct local/gateway models (a Llama build one day, a
        Mistral build the next) into one indistinguishable bucket in
        the usage summary's per-model breakdown — exact token counts,
        attributed to the wrong (meaningless) name. Read via
        ``getattr(session, "resolved_model_name", None)`` so runner.py
        doesn't need a provider-specific branch; every other session
        type simply doesn't define this attribute and the caller
        falls back to its own ``self.model``.
        """
        return self._resolved_model

    # ---- lifecycle ---------------------------------------------------

    async def open(self) -> None:
        """Lazy-build the client if a base URL and model are
        configured. Missing config is NOT raised here -- ``send()``
        yields ``AuthFailure`` on the first round instead, matching
        both other providers' posture (``open()`` runs before any
        event stream exists, from ``SessionRunner.ensure_session``)."""
        if self._client is not None:
            return
        base_url = _resolve_base_url()
        model_name = _resolve_model_name()
        if configuration_issues() or not base_url or not model_name:
            return
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover -- openai is a dep
            raise RuntimeError(
                f"openai SDK not installed: {e}. Reinstall Sift dependencies."
            )
        # Many local servers (Ollama, vLLM's default config) don't
        # check the key at all, but the SDK requires a non-empty
        # string to construct a client -- a harmless placeholder
        # rather than forcing every local-model researcher to
        # configure a fake credential explicitly.
        api_key = _resolve_api_key() or "not-needed"
        from sift.integration_core import (
            MODEL_REQUEST_TIMEOUT_SECONDS,
            MODEL_SDK_MAX_RETRIES,
        )

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
            max_retries=MODEL_SDK_MAX_RETRIES,
        )
        self._resolved_model = model_name
        self._messages = [{"role": "system", "content": self._system_prompt}]
        self._tools = build_chat_completion_tools()
        _verify_lockdown(self._tools)

    async def close(self) -> None:
        client = self._client
        self._client = None
        self._resolved_model = None
        self._messages = []
        self._tools = []
        if client is not None:
            try:
                await client.close()
            except Exception:  # noqa: BLE001 — close-time errors aren't useful
                pass

    # ---- model / effort ------------------------------------------------

    async def set_model(self, model_id: str) -> dict[str, Any]:
        """This provider exposes exactly one catalog entry -- the
        target server's real model name comes from
        ``SIFT_OPENAI_COMPATIBLE_MODEL``, not a per-session choice --
        so this only validates the id matches that one entry. It
        never actually changes what gets invoked."""
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
                    f"{info.provider!r}, not {self.PROVIDER!r}"
                ),
            }
        self.model = model_id
        return {
            "ok": True,
            "model": model_id,
            "label": info.label,
            "context_window": info.context_window,
            "unchanged": True,
        }

    async def set_effort(self, effort: str) -> dict[str, Any]:
        """Accepted but not forwarded to the endpoint -- see the
        module docstring's "Reasoning effort" section. Always reports
        success so the picker never blocks on a provider that can't
        meaningfully act on the dial; ``unsupported: True`` lets a
        caller that cares distinguish this from a real change."""
        self.effort = effort
        return {"ok": True, "effort": effort, "unsupported": True}

    # ---- send ------------------------------------------------------------

    async def send(
        self,
        prompt: str,
        images: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """Drive one chat turn against the configured endpoint.

        Internally drives the standard Chat Completions tool loop:
        each round-trip is one ``chat.completions.create()`` call. If
        the response's message carries ``tool_calls``, dispatch them
        via ``HANDLERS`` (the same dispatch table every provider
        uses), append the assistant + tool-result messages to
        ``self._messages``, and loop. Loop exits when a response
        carries no further tool calls.
        """
        await self.open()
        client = self._client
        if client is None:
            issues = configuration_issues()
            issue_guidance = {
                "base_url_required": f"set {ENV_BASE_URL}",
                "model_name_required": f"set {ENV_MODEL}",
                "base_url_invalid": "use an absolute http:// or https:// URL",
                "base_url_userinfo_forbidden": (
                    "remove embedded username/password data from the URL and "
                    "store any API key through Sift's credential store"
                ),
                "base_url_query_or_fragment_forbidden": (
                    "remove the query string or fragment from the base URL"
                ),
                "insecure_remote_http_requires_explicit_opt_in": (
                    "use HTTPS for remote endpoints, or explicitly set "
                    f"{ENV_ALLOW_INSECURE_REMOTE}=1 only when a trusted "
                    "private-network transport protects the connection"
                ),
            }
            guidance = "; ".join(
                issue_guidance.get(issue, issue) for issue in issues
            )
            if "base_url_required" in issues:
                example = (
                    f" Example local setup: {ENV_BASE_URL}="
                    f"http://localhost:11434/v1 and {ENV_MODEL}=llama3.1."
                )
            elif "model_name_required" in issues:
                example = f" Example: {ENV_MODEL}=llama3.1."
            else:
                example = ""
            yield AuthFailure(
                reason=(
                    "OpenAI-compatible endpoint is not safely configured: "
                    f"{guidance}.{example}"
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

        turn_start = len(self._messages)
        self._messages.append(_build_user_message(prompt, images))

        def rollback_turn() -> None:
            del self._messages[turn_start:]

        try:
            _require_history_size(self._messages)
        except _CompatiblePayloadLimitError:
            rollback_turn()
            yield TurnError(
                message=(
                    "the local conversation exceeds Sift's bounded request "
                    "history limit; start a new session or attach fewer/smaller "
                    "images"
                )
            )
            return

        try:
            MAX_TOOL_ROUNDS = int(os.environ.get(ENV_MAX_TOOL_ROUNDS, "16"))
            if not 1 <= MAX_TOOL_ROUNDS <= 64:
                MAX_TOOL_ROUNDS = 16
        except (TypeError, ValueError):
            MAX_TOOL_ROUNDS = 16

        last_input_tokens = 0
        last_output_tokens = 0

        try:
            import openai as _openai_sdk

            for _round in range(MAX_TOOL_ROUNDS):
                _verify_lockdown(self._tools)
                try:
                    _require_history_size(self._messages)
                except _CompatiblePayloadLimitError:
                    rollback_turn()
                    yield TurnError(
                        message=(
                            "the local conversation exceeds Sift's bounded "
                            "request history limit; start a new session"
                        )
                    )
                    return
                try:
                    resp = await _bounded_chat_completion_create(
                        client.chat.completions,
                        model=self._resolved_model,
                        messages=self._messages,
                        tools=self._tools,
                        tool_choice="auto",
                        max_tokens=MAX_COMPATIBLE_OUTPUT_TOKENS,
                    )
                except _openai_sdk.AuthenticationError as e:
                    msg = provider_error_message(e, secrets=(_resolve_api_key(),))
                    rollback_turn()
                    yield AuthFailure(
                        reason=f"the endpoint rejected the request: {msg}"
                    )
                    return
                except _openai_sdk.APITimeoutError as e:
                    endpoint = provider_error_message(_resolve_base_url() or "")
                    msg = provider_error_message(e, secrets=(_resolve_api_key(),))
                    rollback_turn()
                    # Subclasses APIConnectionError, so this branch
                    # MUST precede it -- a slow-to-respond local model
                    # (a large model on modest hardware, still loading
                    # weights into memory) times out rather than
                    # refusing the connection outright, and deserves
                    # a distinct, more specific message than "could
                    # not connect at all".
                    yield TurnError(
                        message=(
                            f"the endpoint at {endpoint!r} timed "
                            f"out. A local model may still be loading "
                            f"weights into memory (large models can take a "
                            f"while on first request) -- try again in a "
                            f"moment. If this keeps happening, confirm the "
                            f"server isn't stuck. Underlying error: {msg}"
                        )
                    )
                    return
                except _openai_sdk.APIConnectionError as e:
                    endpoint = provider_error_message(_resolve_base_url() or "")
                    msg = provider_error_message(e, secrets=(_resolve_api_key(),))
                    rollback_turn()
                    # The single most common first-use failure for
                    # this provider: the local server (Ollama, vLLM,
                    # ...) isn't running yet, is on a different port,
                    # or the base URL has a typo. A bare stringified
                    # "[Errno 111] Connection refused" tells a
                    # researcher nothing about what to check -- name
                    # the actual configured URL and the likely causes.
                    yield TurnError(
                        message=(
                            f"could not connect to the endpoint at "
                            f"{endpoint!r}. Confirm the server "
                            f"is running and that {ENV_BASE_URL} points at "
                            f"it -- common causes: the server hasn't been "
                            f"started yet, the port is wrong, or the URL is "
                            f"missing its ``/v1`` suffix. Underlying error: "
                            f"{msg}"
                        )
                    )
                    return
                except _openai_sdk.NotFoundError as e:
                    msg = provider_error_message(e, secrets=(_resolve_api_key(),))
                    rollback_turn()
                    # Most often a model-name mismatch: the server is
                    # reachable and authenticated fine, but doesn't
                    # recognize SIFT_OPENAI_COMPATIBLE_MODEL's value.
                    # Distinct from the generic 404 case other errors
                    # might produce, and worth naming explicitly since
                    # "not found" on an HTTP API otherwise reads as a
                    # routing problem rather than a config typo.
                    yield TurnError(
                        message=(
                            f"the endpoint returned 'not found' for model "
                            f"{self._resolved_model!r}. Confirm "
                            f"{ENV_MODEL} matches a model name the server "
                            f"actually has available (for Ollama: run "
                            f"``ollama list``; for vLLM, check the "
                            f"--served-model-name the server was started "
                            f"with). Underlying error: {msg}"
                        )
                    )
                    return
                except _openai_sdk.RateLimitError as e:
                    msg = provider_error_message(e, secrets=(_resolve_api_key(),))
                    rollback_turn()
                    # Gateway services (OpenRouter, Together, Groq)
                    # enforce real rate limits unlike most local
                    # servers -- worth a distinct message pointing at
                    # "wait and retry" rather than reading as a
                    # generic request failure.
                    yield TurnError(
                        message=(
                            f"the endpoint rate-limited this request. If "
                            f"this is a hosted gateway (OpenRouter, "
                            f"Together, Groq, ...), wait a moment and "
                            f"retry, or check your account's rate limits. "
                            f"Underlying error: {msg}"
                        )
                    )
                    return
                except Exception as e:  # noqa: BLE001 — translate to event
                    msg = provider_error_message(e, secrets=(_resolve_api_key(),))
                    lower = msg.lower()
                    rollback_turn()
                    # Non-conformant servers don't always raise the
                    # typed exception above for an auth failure — a
                    # substring check on the error body is the same
                    # fallback ``provider/openai.py`` uses for its own
                    # auth-failure detection.
                    if "auth" in lower or "api key" in lower or "401" in lower:
                        yield AuthFailure(reason=f"endpoint auth failure: {msg}")
                        return
                    if (
                        "context_length_exceeded" in lower
                        or "maximum context length" in lower
                        or ("context" in lower and "length" in lower)
                    ):
                        yield TurnError(
                            message=(
                                "Conversation hit the model's context "
                                "window. Start a new session, or reduce "
                                "earlier turns via ``recall_conversation`` "
                                f"and start fresh from the summary. "
                                f"Underlying error: {msg}"
                            )
                        )
                        return
                    yield TurnError(message=f"request to endpoint failed: {msg}")
                    return

                usage = _field(resp, "usage")
                if usage is not None:
                    last_input_tokens = _field(usage, "prompt_tokens") or 0
                    last_output_tokens = _field(usage, "completion_tokens") or 0

                choices = _field(resp, "choices") or []
                if not isinstance(choices, (list, tuple)) or len(choices) > 16:
                    rollback_turn()
                    yield TurnError(
                        message="endpoint returned a malformed or oversized choices list"
                    )
                    return
                if not choices:
                    rollback_turn()
                    yield TurnError(message="endpoint returned no choices")
                    return
                choice = choices[0]
                message = _field(choice, "message")
                if message is None:
                    rollback_turn()
                    yield TurnError(message="endpoint returned no message")
                    return

                text = _field(message, "content")
                if text is not None and not isinstance(text, str):
                    rollback_turn()
                    yield TurnError(
                        message=(
                            "endpoint returned assistant content in an unsupported "
                            "shape; expected text"
                        )
                    )
                    return
                try:
                    response_bytes = (
                        _require_text_size(text, MAX_COMPATIBLE_RESPONSE_BYTES)
                        if text is not None else 0
                    )
                except _CompatiblePayloadLimitError:
                    rollback_turn()
                    yield TurnError(
                        message=(
                            "endpoint response exceeded Sift's 2 MB decoded "
                            "response safety limit"
                        )
                    )
                    return
                raw_tool_calls = _field(message, "tool_calls") or []
                if (
                    not isinstance(raw_tool_calls, (list, tuple))
                    or len(raw_tool_calls) > MAX_COMPATIBLE_TOOL_CALLS
                ):
                    rollback_turn()
                    yield TurnError(
                        message="endpoint returned an oversized tool-call list"
                    )
                    return
                tool_calls: list[tuple[str, str, str]] = []
                used_call_ids: set[str] = set()
                for call_index, tool_call in enumerate(raw_tool_calls):
                    function = _field(tool_call, "function")
                    name = _field(function, "name", "") or ""
                    if not isinstance(name, str) or len(name) > 256:
                        rollback_turn()
                        yield TurnError(
                            message="endpoint returned a malformed tool name"
                        )
                        return
                    raw_args = _field(function, "arguments")
                    if isinstance(raw_args, str):
                        args_json = raw_args
                    elif isinstance(raw_args, dict):
                        try:
                            _json_wire_size(
                                raw_args, MAX_COMPATIBLE_TOOL_ARGUMENT_BYTES,
                            )
                        except _CompatiblePayloadLimitError:
                            rollback_turn()
                            yield TurnError(
                                message=(
                                    "endpoint tool arguments exceeded Sift's "
                                    "512 KB safety limit"
                                )
                            )
                            return
                        args_json = json.dumps(raw_args)
                    else:
                        args_json = ""
                    try:
                        argument_bytes = _require_text_size(
                            args_json, MAX_COMPATIBLE_TOOL_ARGUMENT_BYTES,
                        )
                    except _CompatiblePayloadLimitError:
                        rollback_turn()
                        yield TurnError(
                            message=(
                                "endpoint tool arguments exceeded Sift's "
                                "512 KB safety limit"
                            )
                        )
                        return
                    raw_call_id = _field(tool_call, "id")
                    if raw_call_id is not None and not isinstance(raw_call_id, str):
                        rollback_turn()
                        yield TurnError(
                            message="endpoint returned a malformed tool-call id"
                        )
                        return
                    if isinstance(raw_call_id, str) and len(raw_call_id) > 512:
                        rollback_turn()
                        yield TurnError(
                            message="endpoint returned an oversized tool-call id"
                        )
                        return
                    call_id = (
                        raw_call_id
                        if isinstance(raw_call_id, str) and raw_call_id
                        else f"compat:r{_round}:{call_index}"
                    )
                    if call_id in used_call_ids:
                        call_id = f"{call_id}:r{_round}:{call_index}"
                    used_call_ids.add(call_id)
                    response_bytes += (
                        argument_bytes
                        + _require_text_size(name, MAX_COMPATIBLE_RESPONSE_BYTES)
                        + _require_text_size(call_id, MAX_COMPATIBLE_RESPONSE_BYTES)
                    )
                    if response_bytes > MAX_COMPATIBLE_RESPONSE_BYTES:
                        rollback_turn()
                        yield TurnError(
                            message=(
                                "endpoint response exceeded Sift's 2 MB decoded "
                                "response safety limit"
                            )
                        )
                        return
                    tool_calls.append((call_id, name, args_json))

                raw_finish_reason = _field(choice, "finish_reason")
                finish_reason = _field(
                    raw_finish_reason, "value", raw_finish_reason,
                )
                if (
                    finish_reason is not None
                    and (
                        not isinstance(finish_reason, str)
                        or len(finish_reason) > 64
                    )
                ):
                    rollback_turn()
                    yield TurnError(
                        message="endpoint returned a malformed finish reason"
                    )
                    return
                if isinstance(finish_reason, str):
                    response_bytes += _require_text_size(
                        finish_reason, MAX_COMPATIBLE_RESPONSE_BYTES,
                    )
                    if response_bytes > MAX_COMPATIBLE_RESPONSE_BYTES:
                        rollback_turn()
                        yield TurnError(
                            message=(
                                "endpoint response exceeded Sift's 2 MB decoded "
                                "response safety limit"
                            )
                        )
                        return
                if finish_reason not in (None, "stop", "tool_calls", "function_call"):
                    if text:
                        yield AssistantText(text=text)
                    rollback_turn()
                    guidance = (
                        "Request a smaller result or raise the endpoint's output "
                        "limit."
                        if finish_reason == "length"
                        else "Review the endpoint policy and retry if appropriate."
                    )
                    yield TurnError(
                        message=(
                            f"endpoint stopped with finish_reason={finish_reason!r}. "
                            f"{guidance} The incomplete turn was not added to "
                            "conversation history."
                        )
                    )
                    return
                if finish_reason in {"tool_calls", "function_call"} and not tool_calls:
                    rollback_turn()
                    yield TurnError(
                        message=(
                            f"endpoint reported finish_reason={finish_reason!r} "
                            "without returning any callable tools"
                        )
                    )
                    return
                if not text and not tool_calls:
                    rollback_turn()
                    yield TurnError(message="endpoint returned an empty completion")
                    return

                # Append the assistant message to history EXACTLY as
                # received (including any tool_calls) before deciding
                # what to do next — Chat Completions requires the
                # assistant's tool_calls message to precede the
                # matching tool-role responses in the message list on
                # the next call.
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": text,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": args_json,
                            },
                        }
                        for call_id, name, args_json in tool_calls
                    ]
                try:
                    _require_history_size([*self._messages, assistant_msg])
                except _CompatiblePayloadLimitError:
                    rollback_turn()
                    yield TurnError(
                        message=(
                            "the local conversation exceeds Sift's bounded "
                            "request history limit; start a new session"
                        )
                    )
                    return
                self._messages.append(assistant_msg)

                if text:
                    yield AssistantText(text=text)

                if not tool_calls:
                    yield TurnDone(
                        input_tokens=last_input_tokens,
                        output_tokens=last_output_tokens,
                        # No provider-specific cache/cost concept for
                        # an arbitrary compatible endpoint — left
                        # None, same as OpenAISession's cache fields.
                        post_turn_tokens=(last_input_tokens + last_output_tokens),
                    )
                    return

                for call_id, name, args_json in tool_calls:
                    yield ToolCall(
                        name=name,
                        input=_safe_json(args_json),
                        call_id=call_id,
                    )

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
                        self._messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": out_text,
                            }
                        )
                        continue
                    args, parse_error = _parse_tool_args(args_json)
                    if parse_error is not None:
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
                        self._messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call_id,
                                "content": out_text,
                            }
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
                                f"[sift.openai_compatible] tool {name!r} "
                                f"handler raised {e.__class__.__name__}: "
                                f"{e}\n" + traceback.format_exc()
                            )
                        else:
                            diag = (
                                f"[sift.openai_compatible] tool {name!r} "
                                f"handler raised {e.__class__.__name__} "
                                f"(set SIFT_DEBUG_USAGE=1 for the full "
                                f"message/traceback)"
                            )
                        print(diag, file=sys.stderr, flush=True)
                        try:
                            from sift.provider.usage_log import (
                                append_usage_line,
                            )

                            append_usage_line(self.cwd, diag.rstrip())
                        except Exception:  # noqa: BLE001
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
                    try:
                        _require_text_size(
                            out_text, MAX_COMPATIBLE_RESPONSE_BYTES,
                        )
                    except _CompatiblePayloadLimitError:
                        out_text = json.dumps({
                            "status": "error",
                            "reason": (
                                "tool result exceeded Sift's 2 MB safety limit; "
                                "request a smaller or summarized result"
                            ),
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
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": out_text,
                    }
                    try:
                        _require_history_size([*self._messages, tool_message])
                    except _CompatiblePayloadLimitError:
                        rollback_turn()
                        yield TurnError(
                            message=(
                                "the local conversation exceeds Sift's bounded "
                                "request history limit; start a new session"
                            )
                        )
                        return
                    self._messages.append(tool_message)

            rollback_turn()
            yield TurnError(
                message=(
                    f"tool loop did not converge within {MAX_TOOL_ROUNDS} "
                    f"rounds; the endpoint kept requesting tools."
                )
            )
        except Exception as e:  # noqa: BLE001 — last-line catch
            rollback_turn()
            msg = provider_error_message(e, secrets=(_resolve_api_key(),))
            yield TurnError(message=f"OpenAI-compatible session error: {msg}")

    # ---- async-context-manager sugar -------------------------------------

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()
