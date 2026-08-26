"""Credential-safe provider error rendering.

Provider SDK exceptions are untrusted text.  HTTP clients can include request
URLs, headers, or reflected server bodies in them, and compatible endpoints
vary widely in how much they echo.  Errors remain useful for diagnosis, but
known credential shapes and the exact credential used for the request must be
removed before an event is persisted or displayed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

_AUTHORIZATION_RE = re.compile(
    r"(?i)(\b(?:proxy-authorization|authorization)\s*[:=]\s*)[^\r\n]+"
)
_COOKIE_RE = re.compile(r"(?i)(\b(?:set-cookie|cookie)\s*[:=]\s*)[^\r\n]+")
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)(\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|"
    r"client[_-]?secret|password|passwd|pwd)\b\s*[:=]\s*[\"']?)"
    r"[^\"'\s,;&}]+"
)
_URL_USERINFO_RE = re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s/@:]+:)[^\s/@]+(@)")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_ERROR_CHARS = 2_000


def provider_error_message(
    error: Any,
    *,
    secrets: Iterable[str | None] = (),
) -> str:
    """Return bounded diagnostic text with credentials removed.

    Exact secret replacement is performed before pattern-based redaction so a
    novel provider key format is still covered.  Every non-empty exact secret
    is removed, including unusually short development credentials: once a
    caller identifies a value as secret, preserving diagnostic prose is less
    important than preventing that value from reaching a log or UI surface.
    """
    try:
        message = str(error)
    except Exception:  # noqa: BLE001 - an exception may have a broken __str__
        message = type(error).__name__
    message = _CONTROL_RE.sub("", message)
    for secret in secrets:
        if isinstance(secret, str) and secret:
            message = message.replace(secret, "***")
    message = _AUTHORIZATION_RE.sub(r"\1***", message)
    message = _COOKIE_RE.sub(r"\1***", message)
    message = _SECRET_ASSIGNMENT_RE.sub(r"\1***", message)
    message = _URL_USERINFO_RE.sub(r"\1***\2", message)
    if len(message) > _MAX_ERROR_CHARS:
        message = message[: _MAX_ERROR_CHARS - 15].rstrip() + " [TRUNCATED]"
    return message or type(error).__name__


__all__ = ["provider_error_message"]
