"""Provider-neutral validation for explicit model image attachments."""

from __future__ import annotations

import base64
import binascii
import struct
from typing import Any

from sift.limits import MODEL_IMAGE_MAX_BYTES

ALLOWED_MODEL_IMAGE_MIMES = frozenset({"image/png", "image/jpeg"})
MAX_MODEL_IMAGES_PER_TURN = 8
MAX_MODEL_IMAGE_PIXELS = 40_000_000


class AttachmentValidationError(ValueError):
    """Safe error for an attachment rejected before a provider request."""


def _matches_signature(mime: str, raw: bytes) -> bool:
    if mime == "image/png":
        return (
            raw.startswith(b"\x89PNG\r\n\x1a\n")
            and raw.endswith(b"\x00\x00\x00\x00IEND\xaeB`\x82")
        )
    if mime == "image/jpeg":
        return raw.startswith(b"\xff\xd8\xff") and raw.endswith(b"\xff\xd9")
    return False


def _image_dimensions(mime: str, raw: bytes) -> tuple[int, int] | None:
    """Read dimensions from bounded headers without decoding pixels."""
    if mime == "image/png":
        if len(raw) < 24 or raw[12:16] != b"IHDR":
            return None
        return struct.unpack(">II", raw[16:24])
    # JPEG dimensions live in a Start Of Frame marker.  Walk length-prefixed
    # segments only; malformed lengths fail closed and no pixel decoder runs.
    pos = 2
    sof = {
        0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
    }
    while pos + 4 <= len(raw):
        if raw[pos] != 0xFF:
            pos += 1
            continue
        while pos < len(raw) and raw[pos] == 0xFF:
            pos += 1
        if pos >= len(raw):
            return None
        marker = raw[pos]
        pos += 1
        if marker in {0x01, 0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if pos + 2 > len(raw):
            return None
        length = struct.unpack(">H", raw[pos:pos + 2])[0]
        if length < 2 or pos + length > len(raw):
            return None
        if marker in sof:
            if length < 7:
                return None
            height, width = struct.unpack(">HH", raw[pos + 3:pos + 7])
            return width, height
        pos += length
    return None


def validate_explicit_images(
    images: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Validate type, encoding, signature, count, and decoded size."""
    if not images:
        return []
    if len(images) > MAX_MODEL_IMAGES_PER_TURN:
        raise AttachmentValidationError(
            f"at most {MAX_MODEL_IMAGES_PER_TURN} images may be attached to one turn"
        )
    validated: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        if not isinstance(image, dict):
            raise AttachmentValidationError(f"image {index} is malformed")
        mime = image.get("mime")
        data = image.get("data")
        if mime not in ALLOWED_MODEL_IMAGE_MIMES:
            raise AttachmentValidationError(
                f"image {index} has an unsupported type; use PNG or JPEG"
            )
        if not isinstance(data, str) or not data:
            raise AttachmentValidationError(f"image {index} has no valid content")
        encoded_limit = ((MODEL_IMAGE_MAX_BYTES + 2) // 3) * 4
        if len(data) > encoded_limit + 4:
            raise AttachmentValidationError(
                f"image {index} exceeds the {MODEL_IMAGE_MAX_BYTES // (1024 * 1024)} MB limit"
            )
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise AttachmentValidationError(
                f"image {index} is not valid base64 content"
            ) from exc
        if not raw or len(raw) > MODEL_IMAGE_MAX_BYTES:
            raise AttachmentValidationError(
                f"image {index} exceeds the {MODEL_IMAGE_MAX_BYTES // (1024 * 1024)} MB limit"
            )
        if not _matches_signature(mime, raw):
            raise AttachmentValidationError(
                f"image {index} content is incomplete or does not match its declared type"
            )
        dimensions = _image_dimensions(mime, raw)
        if dimensions is None:
            raise AttachmentValidationError(
                f"image {index} is truncated or malformed"
            )
        width, height = dimensions
        if (
            width <= 0 or height <= 0
            or width * height > MAX_MODEL_IMAGE_PIXELS
        ):
            raise AttachmentValidationError(
                f"image {index} dimensions exceed the safe 40-megapixel limit"
            )
        normalized = dict(image)
        normalized["mime"] = mime
        normalized["data"] = base64.b64encode(raw).decode("ascii")
        validated.append(normalized)
    return validated


__all__ = [
    "ALLOWED_MODEL_IMAGE_MIMES",
    "MAX_MODEL_IMAGES_PER_TURN",
    "MAX_MODEL_IMAGE_PIXELS",
    "AttachmentValidationError",
    "validate_explicit_images",
]
