"""Small, shared wire-response limits for provider diagnostics.

Provider setup probes talk to endpoints that may be misconfigured or hostile.
They must therefore bound bytes while the body is arriving, rather than asking
an HTTP client to materialize the complete response before checking its size.
"""

from __future__ import annotations

import json
from typing import Any


class ProviderResponseTooLarge(ValueError):
    """Raised with a disclosure-safe message when a response crosses its cap."""


def read_bounded_response(
    response: Any,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read a sync HTTP response incrementally, enforcing decoded-byte limits.

    A valid Content-Length can reject an oversized body before any body bytes
    are consumed.  The incremental check remains authoritative because the
    header may be missing, false, or describe compressed rather than decoded
    content (``httpx.iter_bytes`` yields decoded bytes).
    """
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            declared_length = int(raw_length, 10)
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and declared_length > max_bytes:
            raise ProviderResponseTooLarge(
                f"{label} exceeded the {max_bytes // (1024 * 1024)} MB safety limit"
            )

    captured = bytearray()
    for chunk in response.iter_bytes(chunk_size=64 * 1024):
        if len(chunk) > max_bytes - len(captured):
            raise ProviderResponseTooLarge(
                f"{label} exceeded the {max_bytes // (1024 * 1024)} MB safety limit"
            )
        captured.extend(chunk)
    return bytes(captured)


async def read_bounded_async_response(
    response: Any,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Async equivalent for SDK raw-response wrappers such as OpenAI's."""
    raw_length = response.headers.get("content-length")
    if raw_length is not None:
        try:
            declared_length = int(raw_length, 10)
        except (TypeError, ValueError):
            declared_length = None
        if declared_length is not None and declared_length > max_bytes:
            raise ProviderResponseTooLarge(
                f"{label} exceeded the {max_bytes // (1024 * 1024)} MB safety limit"
            )

    captured = bytearray()
    async for chunk in response.iter_bytes(chunk_size=64 * 1024):
        if len(chunk) > max_bytes - len(captured):
            raise ProviderResponseTooLarge(
                f"{label} exceeded the {max_bytes // (1024 * 1024)} MB safety limit"
            )
        captured.extend(chunk)
    return bytes(captured)


def read_bounded_json_object(
    response: Any,
    *,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    """Read a bounded response and decode a top-level JSON object."""
    body = read_bounded_response(response, max_bytes=max_bytes, label=label)
    value = json.loads(body)
    if not isinstance(value, dict):
        raise TypeError(f"{label} was not an object")
    return value


__all__ = [
    "ProviderResponseTooLarge",
    "read_bounded_async_response",
    "read_bounded_json_object",
    "read_bounded_response",
]
