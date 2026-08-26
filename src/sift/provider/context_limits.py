"""Provider-neutral, conservative context-window preflight."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContextBudget:
    context_window: int
    occupied_tokens: int
    estimated_new_tokens: int
    reserved_output_tokens: int

    @property
    def projected_tokens(self) -> int:
        return (
            self.occupied_tokens
            + self.estimated_new_tokens
            + self.reserved_output_tokens
        )

    @property
    def fits(self) -> bool:
        return self.projected_tokens <= self.context_window


class ContextBudgetExceeded(ValueError):
    """Raised before a paid request that cannot fit the selected model."""


def conservative_text_tokens(text: str) -> int:
    """Return a tokenizer-independent safe upper estimate for plain text.

    Supported provider tokenizers never require more tokens than UTF-8 bytes
    for ordinary model input.  Byte count intentionally overestimates ASCII
    prose, making this suitable as a safety gate rather than a billing meter.
    A small envelope covers message framing and content-block structure.
    """
    return max(1, len(text.encode("utf-8"))) + 32


def model_context_limits(
    model_id: str,
    provider: str,
) -> tuple[int, int] | None:
    """Return ``(window, output reserve)`` for a configured model."""
    try:
        from sift.provider.catalog import get_model
        info = get_model(model_id)
    except KeyError:
        return None
    window = info.context_window
    if provider == "openai_compatible":
        # Unlike the static catalog families, this value is user-configured
        # and can change during a long-running process before a session opens.
        from sift.provider.openai_compatible import resolve_context_window
        window = resolve_context_window()
    reserve = info.max_output_tokens
    if reserve is None:
        reserve = min(8_192, max(1_024, window // 8))
    return window, min(reserve, max(1, window - 1))


def enforce_context_budget(
    *,
    model_id: str,
    provider: str,
    occupied_tokens: int | None,
    prompt: str,
) -> ContextBudget | None:
    """Reject a text turn that cannot leave the model's output reserve.

    ``None`` means the model is an injected/test model with no catalog limit;
    the provider remains the authority in that case.  Image accounting stays
    provider-side because providers tokenize pixels differently.
    """
    limits = model_context_limits(model_id, provider)
    if limits is None:
        return None
    window, reserve = limits
    budget = ContextBudget(
        context_window=window,
        occupied_tokens=max(0, occupied_tokens or 0),
        estimated_new_tokens=conservative_text_tokens(prompt),
        reserved_output_tokens=reserve,
    )
    if not budget.fits:
        raise ContextBudgetExceeded(
            "the message cannot fit the selected model's context window while "
            f"reserving space for its answer (projected {budget.projected_tokens:,} "
            f"of {budget.context_window:,} tokens). Start a new session, shorten "
            "the message, or use a model/configuration with a larger context window"
        )
    return budget


__all__ = [
    "ContextBudget", "ContextBudgetExceeded", "conservative_text_tokens",
    "enforce_context_budget", "model_context_limits",
]
