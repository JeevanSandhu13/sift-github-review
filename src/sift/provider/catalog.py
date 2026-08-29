"""Cross-provider model catalog.

The model picker in both frontends reads from here. Each entry is a
``ModelInfo`` carrying everything the UI needs to render a row in the
picker (label, context window) plus the provider name needed to route
``set_model`` calls to the right session.

Adding a new model: extend the relevant per-provider tuple. The web
UI's ``list_models`` bridge call returns rows derived from this
catalog, filtered to providers the researcher has authenticated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelInfo:
    """One selectable model. ``provider`` matches the keys used by
    ``provider/__init__.py`` and ``auth.py`` (``"anthropic"``,
    ``"openai"``)."""

    id: str
    label: str
    context_window: int
    provider: str
    lifecycle: Literal["stable", "preview", "configured"] = "stable"
    max_output_tokens: int | None = None
    input_modalities: tuple[str, ...] = ("text", "image")

    @property
    def selectable(self) -> bool:
        return self.lifecycle in {"stable", "preview", "configured"}


# Pricing URLs surfaced as a "view pricing" link in the model picker.
# Per-provider — every model in a provider's catalog points at the
# same page; the page itself lists each variant. Centralised here so
# a future docs URL change is a one-line edit.
PROVIDER_PRICING_URLS: dict[str, str] = {
    "anthropic": "https://platform.claude.com/docs/en/about-claude/pricing",
    "openai": "https://openai.com/api/pricing/",
    "gemini": "https://ai.google.dev/gemini-api/docs/pricing",
    "azure_openai": "https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/",
    "vertex_gemini": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
    "bedrock_anthropic": "https://aws.amazon.com/bedrock/pricing/",
    "vertex_anthropic": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
}

# API-key creation pages surfaced as a "create / get a key" link on
# the auth screen's per-provider help text. A researcher arriving
# without a key clicks straight through to the provider's console.
PROVIDER_API_KEY_URLS: dict[str, str] = {
    "anthropic": "https://console.anthropic.com/settings/keys",
    "openai": "https://platform.openai.com/api-keys",
    "gemini": "https://aistudio.google.com/apikey",
    "azure_openai": "https://learn.microsoft.com/azure/ai-services/openai/how-to/managed-identity",
}


# ---------------------------------------------------------------------------
# Per-provider models
# ---------------------------------------------------------------------------

# Anthropic — the Claude 5 family, cheapest tier first. The ``[1m]``
# suffix is the Claude CLI / Agent SDK convention for the 1M-context
# opt-in (the CLI strips it and sets the beta header; ``fable[1m]``
# is a built-in CLI alias, so the suffix parses generically). On the
# Claude 5 models 1M is both the default and the maximum, so the
# suffix is redundant but retained so every
# Anthropic id in the catalog follows one convention and per-session
# model memory (which restores only catalog-known ids) stays stable.
# There's no pricing tier on context length: a 900k-token request
# costs the same per-token as a 9k-token one. Labels are clean —
# context-window numbers live in the picker's right-side column.
# Tiers (per Anthropic's pricing doc, Aug 21 2026): Sonnet 5 is on a
# temporary $2/$10 per-MTok rate through Aug 31 ($3/$15 afterward),
# Opus 5 is $5/$25, and Fable 5 is $10/$50. Opus 5 and Fable 5 are
# listed in the picker but deliberately NOT the default (see
# PROVIDER_DEFAULTS below); a researcher opts in per session and
# per-session model memory keeps the choice.
# Haiku is intentionally excluded for now: the Sift workload
# (multi-turn analysis with tool use) calls for the heavier models.
ANTHROPIC_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="claude-sonnet-5[1m]",
        label="Sonnet 5",
        context_window=1_000_000,
        provider="anthropic",
        max_output_tokens=128_000,
    ),
    ModelInfo(
        id="claude-opus-5[1m]",
        label="Opus 5",
        context_window=1_000_000,
        provider="anthropic",
        max_output_tokens=128_000,
    ),
    ModelInfo(
        id="claude-fable-5[1m]",
        label="Fable 5",
        context_window=1_000_000,
        provider="anthropic",
        max_output_tokens=128_000,
    ),
)

# OpenAI — the GPT-5.6 family, cheapest tier first. ``gpt-5.6-terra``
# is the balanced / cost-tier model (roughly the "mini" slot of
# earlier GPT-5 families; $2/$12 per MTok); ``gpt-5.6-sol`` is the
# flagship ($5/$30 — same price point as the gpt-5.5 it replaces;
# the bare ``gpt-5.6`` alias routes to Sol). Both accept the full
# none/low/medium/high/xhigh/max reasoning range, so the provider's
# pinned ``effort="xhigh"`` is valid on either. Ids match the OpenAI
# Models API exactly so a researcher can cross-reference pricing and
# limits in OpenAI's own docs / billing dashboard.
# Context window: 1.05M tokens for both per OpenAI's published spec
# (the Models API itself doesn't expose this — it has to be hard-coded
# from OpenAI's docs and updated when they publish new variants).
OPENAI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gpt-5.6-terra",
        label="GPT-5.6 Terra",
        context_window=1_050_000,
        provider="openai",
        max_output_tokens=128_000,
    ),
    ModelInfo(
        id="gpt-5.6-sol",
        label="GPT-5.6 Sol",
        context_window=1_050_000,
        provider="openai",
        max_output_tokens=128_000,
    ),
)


# Gemini — the Gemini 3 family, cheapest tier first. ``gemini-3.7-flash``
# is the stable, GA "workhorse" tier ($0.75/$3.75 per MTok introductory
# through 2026-12-31, $1.50/$7.50 standard afterward) — Sift defaults
# here for the same reason the OpenAI provider defaults to its
# same-price-tier model: a researcher who hasn't chosen shouldn't be
# opted into the pricier tier by default. ``gemini-3.1-pro-preview``
# is the flagship reasoning tier ("advanced intelligence, complex
# problem-solving", ~$2/$12 per MTok up to 200k context, higher above)
# — still Preview status per Google's own model page as of this
# writing, offered as the opt-in heavier tier the same way Claude's
# Opus/Fable and OpenAI's Sol sit above their respective cheaper
# defaults. Context window: 1,048,576 tokens for both per Google's
# published model specs (ids and specs verified against
# ai.google.dev/gemini-api/docs/models, Aug 2026 — update this comment
# alongside the tuple if Google's lineup moves again).
GEMINI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="gemini-3.7-flash",
        label="Gemini 3.7 Flash",
        context_window=1_048_576,
        provider="gemini",
        max_output_tokens=65_536,
    ),
    ModelInfo(
        id="gemini-3.1-pro-preview",
        label="Gemini 3.1 Pro",
        context_window=1_048_576,
        provider="gemini",
        lifecycle="preview",
        max_output_tokens=65_536,
    ),
)


# OpenAI-compatible endpoints — local models (Ollama, vLLM, LM
# Studio, ...) and third-party gateways (OpenRouter et al.) reached
# via the Chat Completions wire protocol. Unlike the two catalogs
# above, there is no fixed set of real model ids to list — the
# researcher's target server can expose anything. This is therefore
# always exactly ONE catalog entry: a fixed selector id the UI shows
# as "Custom (OpenAI-compatible)". The REAL model name actually
# invoked is resolved from ``SIFT_OPENAI_COMPATIBLE_MODEL`` at session
# -open time (see ``provider/openai_compatible.py``); this entry's id
# is never sent to the target server. Context window is similarly a
# configured number (``SIFT_OPENAI_COMPATIBLE_CONTEXT_WINDOW``, default
# 32k) rather than a hard-coded published spec, since it varies by
# whichever model the researcher has pointed Sift at — resolved lazily
# via a function (not a literal in this tuple) so an env var set after
# import still takes effect.
# Inlined rather than imported from ``provider.openai_compatible``:
# that module imports ``sift.tools`` (for the shared ``HANDLERS``
# dispatch table), and ``sift.tools`` itself imports
# ``provider.tool_schemas``, which pulls in this whole ``provider``
# package — importing ``openai_compatible`` from here at module load
# time closes that cycle. The env var name and default are kept in
# sync with ``openai_compatible.ENV_CONTEXT_WINDOW`` /
# ``DEFAULT_CONTEXT_WINDOW`` by ``test_openai_compatible.py``.
_OPENAI_COMPATIBLE_ENV_CONTEXT_WINDOW = "SIFT_OPENAI_COMPATIBLE_CONTEXT_WINDOW"
_OPENAI_COMPATIBLE_DEFAULT_CONTEXT_WINDOW = 32_000


def _openai_compatible_context_window() -> int:
    import os
    raw = os.environ.get(_OPENAI_COMPATIBLE_ENV_CONTEXT_WINDOW)
    if raw:
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass
    return _OPENAI_COMPATIBLE_DEFAULT_CONTEXT_WINDOW


OPENAI_COMPATIBLE_CUSTOM_MODEL_ID = "openai-compatible-custom"


def _build_openai_compatible_models() -> tuple[ModelInfo, ...]:
    return (
        ModelInfo(
            id=OPENAI_COMPATIBLE_CUSTOM_MODEL_ID,
            label="Custom (OpenAI-compatible)",
            context_window=_openai_compatible_context_window(),
            provider="openai_compatible",
            lifecycle="configured",
            max_output_tokens=None,
            input_modalities=("text",),
        ),
    )


OPENAI_COMPATIBLE_MODELS: tuple[ModelInfo, ...] = _build_openai_compatible_models()


# Managed cloud deployments are configured resources, not aliases for the
# direct model catalogs.  The selector id is stable in saved Sift sessions;
# the actual Azure deployment, Vertex model, or Bedrock inference-profile id
# is resolved and validated by that provider immediately before use.
AZURE_OPENAI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="azure-openai-deployment",
        label="Configured Azure OpenAI deployment",
        context_window=128_000,
        provider="azure_openai",
        lifecycle="configured",
        # Deployment model is operator-defined.  Reserve a conservative 16k
        # by default so a 128k deployment remains usable; administrators can
        # set the exact managed context/output limits in deployment config.
        max_output_tokens=16_384,
    ),
)

VERTEX_GEMINI_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="vertex-gemini-model",
        label="Configured Vertex AI Gemini model",
        context_window=1_048_576,
        provider="vertex_gemini",
        lifecycle="configured",
        max_output_tokens=65_536,
    ),
)

BEDROCK_ANTHROPIC_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="bedrock-anthropic-model",
        label="Configured Amazon Bedrock Claude model",
        context_window=200_000,
        provider="bedrock_anthropic",
        lifecycle="configured",
        max_output_tokens=65_536,
    ),
)

VERTEX_ANTHROPIC_MODELS: tuple[ModelInfo, ...] = (
    ModelInfo(
        id="vertex-anthropic-model",
        label="Configured Vertex AI Claude model",
        context_window=200_000,
        provider="vertex_anthropic",
        lifecycle="configured",
        max_output_tokens=65_536,
    ),
)


ALL_MODELS: tuple[ModelInfo, ...] = (
    ANTHROPIC_MODELS
    + OPENAI_MODELS
    + GEMINI_MODELS
    + OPENAI_COMPATIBLE_MODELS
    + AZURE_OPENAI_MODELS
    + VERTEX_GEMINI_MODELS
    + BEDROCK_ANTHROPIC_MODELS
    + VERTEX_ANTHROPIC_MODELS
)


# Retired/deprecated ids are not selectable, but a persisted session using one
# is migrated deliberately instead of silently falling back to an unrelated
# default. Keep this table small and evidence-backed by provider deprecation or
# migration guidance.
MODEL_REPLACEMENTS: dict[str, str] = {
    "claude-sonnet-4-6[1m]": "claude-sonnet-5[1m]",
    "claude-opus-4-8[1m]": "claude-opus-5[1m]",
    "gpt-5.5": "gpt-5.6-sol",
    "gpt-5.5-pro": "gpt-5.6-sol",
    "gemini-3.6-flash": "gemini-3.7-flash",
    "gemini-3.5-flash": "gemini-3.7-flash",
    "gemini-3-flash-preview": "gemini-3.7-flash",
    "gemini-3-pro-preview": "gemini-3.1-pro-preview",
}


def current_model_id(model_id: str) -> str:
    """Return a selectable id, migrating one known retired id if needed."""
    replacement = MODEL_REPLACEMENTS.get(model_id, model_id)
    get_model(replacement)
    return replacement


# Default model per provider — what an "open a session for provider X"
# call uses when the researcher hasn't picked something explicitly.
# Anthropic stays on Sonnet even though heavier tiers (Opus, Fable)
# are in the picker: the default is what a researcher gets without
# asking, and silently defaulting to a 2–3x-priced tier would change
# their bill, not just their model. OpenAI defaults to Sol — it is
# the direct successor to gpt-5.5 (the previous default) at the same
# $5/$30 price point, so a researcher's bill doesn't move on upgrade;
# Terra is the cheaper opt-in.
PROVIDER_DEFAULTS: dict[str, str] = {
    "anthropic": "claude-sonnet-5[1m]",
    "openai": "gpt-5.6-sol",
    "gemini": "gemini-3.7-flash",
    # The one and only catalog entry this provider has — see
    # OPENAI_COMPATIBLE_MODELS above.
    "openai_compatible": OPENAI_COMPATIBLE_CUSTOM_MODEL_ID,
    "azure_openai": AZURE_OPENAI_MODELS[0].id,
    "vertex_gemini": VERTEX_GEMINI_MODELS[0].id,
    "bedrock_anthropic": BEDROCK_ANTHROPIC_MODELS[0].id,
    "vertex_anthropic": VERTEX_ANTHROPIC_MODELS[0].id,
}


# Default provider when none is configured yet — only matters as a
# placeholder; the auth screen forces the researcher to pick before
# they reach the chat view.
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = PROVIDER_DEFAULTS[DEFAULT_PROVIDER]


# ---------------------------------------------------------------------------
# Effort levels
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EffortInfo:
    """One selectable reasoning-effort level. ``id`` is the wire value
    the provider accepts (Anthropic ``output_config.effort``, passed
    by the Agent SDK as the CLI's ``--effort`` flag; OpenAI
    ``reasoning.effort`` on the Responses request)."""

    id: str
    label: str


# Reasoning effort, cheapest first. **The ladders differ per provider**
# — the picker renders whichever one belongs to the selected model's
# provider, so a level that provider can't take is never offered:
#
#   Anthropic  low, medium, high, xhigh, max
#   OpenAI     low, medium, high, xhigh, max, pro
#   Gemini     low, medium, high
#
# The four lower rungs are the same dial on both sides
# (``output_config.effort`` on Anthropic, ``reasoning.effort`` on
# OpenAI). The top rung is where they part:
#
# - Anthropic's ceiling is ``max``. The Claude Agent SDK types
#   ``EffortLevel`` as low|medium|high|xhigh|max and the CLI lists
#   all five.
# - Gemini's ``ThinkingConfig.thinking_level`` enum tops out at
#   ``HIGH`` (LOW/MEDIUM/HIGH/MINIMAL are the only rungs the pinned
#   SDK exposes) — no analogue to Anthropic's ``max`` or OpenAI's
#   ``pro``. ``xhigh`` therefore clamps down to ``high`` via
#   ``clamp_effort`` before ``provider/gemini.py`` ever consults its
#   effort→thinking-level map; that map only needs to answer for the
#   three rungs Gemini's own ladder offers. ``minimal`` is excluded
#   for the same reason OpenAI's ``none``/``minimal`` are (see below)
#   — it can suppress the thinking trace Sift's UI surfaces.
# - OpenAI's GPT-5.6 models accept ``max`` effort. They also expose
#   ``pro`` as a *separate* ``reasoning.mode`` knob. Mode is genuinely
#   orthogonal to effort in the API, but for a researcher choosing
#   "how hard should this try", pro mode at max effort is the rung
#   above standard mode at max effort. ``provider/openai.py``
#   translates that final display rung back into both real parameters.
#
# ``none`` / ``minimal`` (OpenAI-only) are deliberately excluded:
# Sift surfaces the reasoning trace in the thinking panel, and those
# levels suppress it. Anthropic has no equivalent rung either, so
# offering them would make the two panels diverge for no gain.
_LOW = EffortInfo(id="low", label="Low")
_MEDIUM = EffortInfo(id="medium", label="Medium")
_HIGH = EffortInfo(id="high", label="High")
_XHIGH = EffortInfo(id="xhigh", label="Extra high")
_MAX = EffortInfo(id="max", label="Max")
_PRO = EffortInfo(id="pro", label="Pro")

PROVIDER_EFFORTS: dict[str, tuple[EffortInfo, ...]] = {
    "anthropic": (_LOW, _MEDIUM, _HIGH, _XHIGH, _MAX),
    "openai": (_LOW, _MEDIUM, _HIGH, _XHIGH, _MAX, _PRO),
    "gemini": (_LOW, _MEDIUM, _HIGH),
    "azure_openai": (_LOW, _MEDIUM, _HIGH, _XHIGH, _MAX, _PRO),
    "vertex_gemini": (_LOW, _MEDIUM, _HIGH),
    "bedrock_anthropic": (_LOW, _MEDIUM, _HIGH, _XHIGH, _MAX),
    "vertex_anthropic": (_LOW, _MEDIUM, _HIGH, _XHIGH, _MAX),
}

# Every level this build knows, in canonical cheapest-first order.
EFFORT_OPTIONS: tuple[EffortInfo, ...] = (
    _LOW, _MEDIUM, _HIGH, _XHIGH, _MAX, _PRO,
)
EFFORT_LEVELS: tuple[str, ...] = tuple(e.id for e in EFFORT_OPTIONS)

# Rank drives ``clamp_effort``. OpenAI ``pro`` sits above ``max``:
# both use max effort, but pro additionally enables the provider's
# heavier reasoning mode. This distinction prevents an Anthropic
# ``max`` choice from silently escalating to OpenAI pro during a
# provider switch.
_EFFORT_RANK: dict[str, int] = {
    "low": 0, "medium": 1, "high": 2, "xhigh": 3, "max": 4, "pro": 5,
}

# What a session runs at when nobody has chosen: exactly what both
# providers were hard-pinned to before effort became selectable, so
# turning the dial on changed no existing behaviour. Supported on
# both ladders.
DEFAULT_EFFORT = "xhigh"


def get_effort(effort_id: str) -> EffortInfo:
    """Look up an effort level by id. Raises ``KeyError`` if unknown."""
    for e in EFFORT_OPTIONS:
        if e.id == effort_id:
            return e
    raise KeyError(f"unknown effort level: {effort_id!r}")


def efforts_for_provider(provider: str) -> tuple[EffortInfo, ...]:
    """The ladder a given provider actually accepts. Unknown provider
    falls back to the canonical list — callers validate the provider
    elsewhere, and an empty picker would be worse than a wrong one."""
    return PROVIDER_EFFORTS.get(provider, EFFORT_OPTIONS)


def effort_levels_for_provider(provider: str) -> tuple[str, ...]:
    """Just the ids — the validation surface for ``set_effort``."""
    return tuple(e.id for e in efforts_for_provider(provider))


def clamp_effort(effort_id: str | None, provider: str) -> str:
    """Return the closest level ``provider`` supports at or below
    ``effort_id``, else the provider's default.

    Two callers need this, both crossing a boundary where the ladder
    can change out from under a recorded choice:

    - a cross-provider model swap (for example OpenAI ``pro`` to a
      provider whose ladder tops out at ``max``), and
    - the per-session restore path (a state file recording ``max``
      against a session whose model is now an OpenAI one).

    Stepping *down* rather than resetting keeps the researcher's
    intent: someone who asked for the ceiling gets the new provider's
    ceiling, not a silent drop to the middle of the ladder. Never
    steps up — that would raise spend nobody asked for.
    """
    supported = effort_levels_for_provider(provider)
    if effort_id in supported:
        return effort_id  # type: ignore[return-value]
    default = DEFAULT_EFFORT if DEFAULT_EFFORT in supported else supported[-1]
    if effort_id not in _EFFORT_RANK:
        return default
    want = _EFFORT_RANK[effort_id]
    at_or_below = [e for e in supported if _EFFORT_RANK[e] <= want]
    # Ladders are cheapest-first, so the last match is the highest
    # supported rung that doesn't exceed what was asked for.
    return at_or_below[-1] if at_or_below else supported[0]


def normalize_effort(effort_id: str | None, provider: str | None = None) -> str:
    """Return ``effort_id`` when this build knows it, else the default.

    With ``provider``, this is :func:`clamp_effort` — the level is
    additionally held to that provider's ladder. Without one it only
    checks the canonical list, which is what the runner does before
    it knows which session it will open.
    """
    if provider is not None:
        return clamp_effort(effort_id, provider)
    if effort_id in EFFORT_LEVELS:
        return effort_id  # type: ignore[return-value]
    return DEFAULT_EFFORT


def get_model(model_id: str) -> ModelInfo:
    """Look up a model by id. Raises ``KeyError`` if unknown."""
    for m in ALL_MODELS:
        if m.id == model_id:
            return m
    raise KeyError(f"unknown model id: {model_id!r}")


def models_for_provider(provider: str) -> tuple[ModelInfo, ...]:
    """All models exposed by a given provider."""
    return tuple(m for m in ALL_MODELS if m.provider == provider)


def provider_for_model(model_id: str) -> str:
    """Which provider owns a given model id."""
    return get_model(model_id).provider
