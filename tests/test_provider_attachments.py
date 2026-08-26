from __future__ import annotations

import base64

import pytest

from sift.provider.attachments import (
    MAX_MODEL_IMAGES_PER_TURN,
    AttachmentValidationError,
    validate_explicit_images,
)

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4"
    "//8/AAX+Av4N70a4AAAAAElFTkSuQmCC"
)
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsL"
    "DBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/"
    "2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QA"
    "HwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUF"
    "BAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkK"
    "FhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1"
    "dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXG"
    "x8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEB"
    "AQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAEC"
    "AxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRom"
    "JygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOE"
    "hYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU"
    "1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//"
    "2Q=="
)


def _image(mime: str, raw: bytes) -> dict[str, str]:
    return {"mime": mime, "data": base64.b64encode(raw).decode("ascii")}


def test_valid_png_and_jpeg_are_canonicalized() -> None:
    rows = validate_explicit_images([
        _image("image/png", PNG), _image("image/jpeg", JPEG),
    ])
    assert [row["mime"] for row in rows] == ["image/png", "image/jpeg"]
    assert base64.b64decode(rows[0]["data"]) == PNG


@pytest.mark.parametrize(
    "row, message",
    [
        ({"mime": "application/pdf", "data": "eA=="}, "unsupported type"),
        ({"mime": "image/png", "data": "not base64"}, "valid base64"),
        (_image("image/png", JPEG), "does not match"),
        ({"mime": "image/png", "data": ""}, "valid content"),
    ],
)
def test_unsupported_or_deceptive_attachments_fail_before_upload(
    row: dict[str, str], message: str,
) -> None:
    with pytest.raises(AttachmentValidationError, match=message):
        validate_explicit_images([row])


def test_attachment_count_is_bounded() -> None:
    with pytest.raises(AttachmentValidationError, match="at most"):
        validate_explicit_images(
            [_image("image/png", PNG)] * (MAX_MODEL_IMAGES_PER_TURN + 1)
        )


def test_truncated_container_and_decompression_bomb_dimensions_fail() -> None:
    with pytest.raises(AttachmentValidationError, match="incomplete"):
        validate_explicit_images([_image("image/png", PNG[:28])])

    bomb = bytearray(PNG)
    bomb[16:24] = (100_000).to_bytes(4, "big") * 2
    with pytest.raises(AttachmentValidationError, match="40-megapixel"):
        validate_explicit_images([_image("image/png", bytes(bomb))])
