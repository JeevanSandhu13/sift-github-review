"""Stable integration identifiers shared by low-level registries.

This module deliberately imports nothing from the rest of Sift.  Credential,
provider, trust, policy, and UI code can all depend on it without creating an
import cycle or maintaining their own provider tuple.
"""

from __future__ import annotations

MODEL_PROVIDER_IDS: tuple[str, ...] = (
    "anthropic",
    "openai",
    "gemini",
    "openai_compatible",
    "azure_openai",
    "vertex_gemini",
    "bedrock_anthropic",
    "vertex_anthropic",
)

REMOTE_MODEL_PROVIDER_IDS: frozenset[str] = frozenset(
    set(MODEL_PROVIDER_IDS) - {"openai_compatible"}
)


__all__ = ["MODEL_PROVIDER_IDS", "REMOTE_MODEL_PROVIDER_IDS"]
