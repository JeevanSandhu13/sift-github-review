"""Portable, bounded names for files Sift creates from external labels."""

from __future__ import annotations

import re
from pathlib import Path


_WINDOWS_RESERVED_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_WINDOWS_ILLEGAL_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def portable_filename(value: str, *, max_chars: int = 160) -> str:
    """Return a single portable component while preserving its extension.

    Windows reserves device basenames even when an extension is present, and
    legacy path limits make a valid 255-character filesystem component unsafe
    inside Sift's nested session directories. Apply the same deterministic
    rule on every OS so a session or export keeps its names when moved.
    """
    if max_chars < 8:
        raise ValueError("portable filename limit must be at least 8 characters")
    name = _WINDOWS_ILLEGAL_CHARS.sub("_", Path(value).name).strip(" .")
    if not name:
        return "dataset"
    suffix = Path(name).suffix
    stem = name[:-len(suffix)] if suffix else name
    stem = stem.rstrip(" .") or "dataset"
    if stem.casefold() in _WINDOWS_RESERVED_STEMS:
        stem = f"_{stem}"
    suffix_budget = min(len(suffix), max_chars - 1)
    suffix = suffix[-suffix_budget:] if suffix_budget else ""
    stem = stem[: max(1, max_chars - len(suffix))]
    return f"{stem}{suffix}"


def portable_stem(value: str, *, max_chars: int = 60) -> str:
    """Return a portable extension-free component for generated outputs."""
    cleaned = portable_filename(value, max_chars=max_chars)
    cleaned = cleaned.strip(" .") or "dataset"
    if cleaned.casefold() in _WINDOWS_RESERVED_STEMS:
        cleaned = f"_{cleaned}"
    return cleaned[:max_chars]
