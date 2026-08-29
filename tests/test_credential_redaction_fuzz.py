from __future__ import annotations

from urllib.parse import quote

from hypothesis import HealthCheck, assume, given, settings, strategies as st

from sift.connectors import redact_connection
from sift.provider.error_safety import provider_error_message


_TOKEN = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=8,
    max_size=40,
)
_HOST = st.from_regex(r"[a-z][a-z0-9]{2,15}\.example", fullmatch=True)


@given(user=_TOKEN, password=_TOKEN, host=_HOST, database=_TOKEN)
@settings(suppress_health_check=[HealthCheck.too_slow])
def test_uri_credentials_are_redacted_for_generated_connection_strings(
    user: str,
    password: str,
    host: str,
    database: str,
) -> None:
    secret = f"sift-secret-{password}"
    assume(secret not in user)
    assume(secret not in host)
    assume(secret not in database)
    uri = (
        f"postgresql+psycopg://{quote(user, safe='')}:{quote(secret, safe='')}"
        f"@{host}/{quote(database, safe='')}?sslmode=verify-full"
    )
    shown = redact_connection(uri)
    assert secret not in shown
    assert quote(secret, safe="") not in shown
    assert "***" in shown


@given(password=_TOKEN, host=_HOST, database=_TOKEN)
def test_odbc_passwords_are_redacted_for_generated_connection_strings(
    password: str,
    host: str,
    database: str,
) -> None:
    assume(password not in host)
    assume(password not in database)
    connection = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={host};DATABASE={database};UID=researcher;PWD={password};"
        "Encrypt=yes;TrustServerCertificate=no"
    )
    shown = redact_connection(connection)
    assert password not in shown
    assert "PWD=***" in shown


@given(secret=_TOKEN, prefix=st.text(min_size=0, max_size=80))
def test_provider_error_redaction_never_returns_seeded_secret(
    secret: str,
    prefix: str,
) -> None:
    error = RuntimeError(f"{prefix} credential={secret} endpoint failed")
    shown = provider_error_message(error, secrets=(secret,))
    assert secret not in shown
