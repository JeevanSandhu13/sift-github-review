from __future__ import annotations

from sift.provider.error_safety import provider_error_message


def test_exact_credentials_and_common_secret_shapes_are_redacted():
    secret = "novel-provider-key-format-123"
    raw = (
        "request failed api_key=visible Authorization: Bearer bearer-secret "
        "https://alice:url-password@example.test/v1 exact=" + secret
    )

    safe = provider_error_message(raw, secrets=(secret,))

    assert "visible" not in safe
    assert "bearer-secret" not in safe
    assert "url-password" not in safe
    assert secret not in safe
    assert safe.count("***") >= 2


def test_basic_auth_and_cookie_headers_are_redacted():
    safe = provider_error_message(
        "Authorization: Basic dXNlcjpwYXNzd29yZA==\n"
        "Set-Cookie: session=browser-secret; Path=/; Secure"
    )

    assert "dXNlcjpwYXNzd29yZA==" not in safe
    assert "browser-secret" not in safe
    assert safe.count("***") >= 2


def test_signed_authorization_headers_are_redacted_as_one_sensitive_field():
    safe = provider_error_message(
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE, "
        "SignedHeaders=host, Signature=highly-sensitive-signature\n"
        "ordinary diagnostic"
    )

    assert "AKIAEXAMPLE" not in safe
    assert "highly-sensitive-signature" not in safe
    assert "ordinary diagnostic" in safe


def test_exact_short_credentials_are_redacted_even_without_a_secret_label():
    for secret in ("x", "xy", "xyz"):
        safe = provider_error_message(
            f"upstream reflected [{secret}] verbatim",
            secrets=(secret,),
        )
        assert secret not in safe
        assert "***" in safe


def test_error_messages_are_bounded_and_control_char_free():
    safe = provider_error_message("prefix\x00" + "x" * 10_000)

    assert "\x00" not in safe
    assert len(safe) <= 2_000
    assert safe.endswith("[TRUNCATED]")


def test_broken_exception_stringification_is_safe():
    class Broken:
        def __str__(self) -> str:
            raise RuntimeError("no string")

    assert provider_error_message(Broken()) == "Broken"
