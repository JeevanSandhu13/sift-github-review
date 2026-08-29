"""Sift — data connectors: databases and warehouses as local datasets.

Most institutional data never exists as a file. It lives in a
hospital's SQL Server, a university's Postgres research warehouse, a
bank's Snowflake, a ministry's Oracle estate. A tool that only reads
CSVs is a tool those researchers cannot use for their real work.

The architectural problem
-------------------------
Sift's sandbox denies network access; this is a core security boundary:
it is why generated code cannot exfiltrate data. A connector needs a
network. Naively, the two requirements are in direct conflict, and the
tempting resolutions are both wrong — punching a hole in the sandbox
for "trusted" queries destroys the guarantee, and putting credentials
where generated code can read them destroys it faster.

The resolution
--------------
The query runs on the **host**, never inside the sandbox — the same
posture as ``install_packages``, which is the existing precedent for a
network-touching, researcher-approved operation:

1. The **researcher** supplies the connection (a URI or a DuckDB/
   SQLite file path) and approves each query through the same modal
   consent gate used for package installs.
2. Sift executes the query host-side with the researcher's
   credentials and **materializes the result to Parquet inside the
   session directory**.
3. From that point the extract is an ordinary Sift dataset: schema
   depth policy, disclosure control, profiling and the size guards all
   apply unchanged.

What this buys, precisely:

- The **sandbox keeps its network denial**. Generated code still
  cannot reach the database, or anything else.
- **Credentials never enter the sandbox** (the executor's env
  allowlist already excludes them) and are never shown to the model —
  connection strings are redacted before any model-visible surface.
- The **model never issues the query**. It can propose SQL in chat for
  the researcher to run; it cannot execute one. A connector the model
  could drive would be a data-exfiltration primitive wearing a
  helpful hat.
- Every materialization is recorded in the release ledger as a local
  ingestion event, so the provenance of an extract is auditable
  alongside everything else.

Backends: DuckDB (files, Parquet, out-of-core SQL), SQLite (files),
and Sift's explicitly reviewed SQLAlchemy integrations — PostgreSQL,
MySQL/MariaDB, SQL Server, Oracle, Snowflake, BigQuery, Redshift, and
Databricks — provided the researcher's environment has that backend's
DBAPI driver. Driver absence is reported as a clear, actionable message
rather than a traceback. Unreviewed SQLAlchemy dialects are refused so
they cannot silently inherit Sift's transport and read-only claims.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Literal
from urllib.parse import unquote

from sift.filename_safety import portable_stem
from sift.integration_core import (
    CancellationToken,
    IntegrationCancelled,
    IntegrationError,
)

# Hard cap on rows materialized in one extract. A researcher who
# genuinely needs more should aggregate in SQL — which is the point of
# having a database — rather than pull a warehouse onto a laptop.
DEFAULT_ROW_LIMIT = 5_000_000

# Row count alone is not a memory bound: a single JSON/text/blob column can
# make a few thousand rows larger than several million narrow numeric rows.
# Keep extraction host memory bounded independently of the row cap.  The
# limit is measured on each materialized pandas batch with ``deep=True``.
DEFAULT_BYTE_LIMIT = 256 * 1024 * 1024
_MIN_FREE_DISK_RESERVE = 512 * 1024 * 1024
MAX_CONNECTION_INPUT_BYTES = 64 * 1024

# A remote warehouse can otherwise hold the host bridge indefinitely. The
# value is runtime-configurable for institutionally approved long jobs, but
# malformed values never disable the guard.
DEFAULT_QUERY_TIMEOUT_SECONDS = 300
MAX_QUERY_TIMEOUT_SECONDS = 3_600

# Connection establishment has a deliberately shorter ceiling than a query.
# DNS, TCP and authentication all happen before Sift has a live connection it
# can interrupt.  The only safe portable strategy is therefore to configure
# the DBAPI/SDK *before* calling connect and keep that call synchronous: a
# worker-thread timeout would return to the UI while an abandoned credentialed
# connection attempt continued in the background.
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30
MAX_CONNECT_TIMEOUT_SECONDS = 120

# Fetch in bounded batches.  SQLAlchemy maps ``yield_per`` to server-side
# cursors on backends that support them and keeps a fixed client buffer on
# the rest.  This also avoids the old, non-portable ``SELECT ... LIMIT``
# wrapper (SQL Server uses TOP/OFFSET, for example).
FETCH_BATCH_ROWS = 10_000

# Database kinds Sift has deliberately reviewed and exposes in its integration
# contract. Accepting an arbitrary SQLAlchemy dialect would silently extend
# the product's security claims to a driver whose TLS and read-only semantics
# we have never checked. Common historical aliases are normalized before
# policy and transport validation so they cannot bypass those controls.
_BACKEND_ALIASES = {
    "postgres": "postgresql",
}
_SUPPORTED_REMOTE_BACKENDS = frozenset({
    "postgresql",
    "mysql",
    "mariadb",
    "mssql",
    "oracle",
    "snowflake",
    "bigquery",
    "redshift",
    "databricks",
})

# A connection URI must have one unambiguous, reviewable network target.
# Several DBAPIs accept query parameters that either replace the hostname in
# the URI or load a second configuration document which can do so.  That is
# more than a presentation problem: a URI displaying an allowed/loopback host
# could connect to a different server after Sift's TLS and enterprise-policy
# checks.  Keep the small, explicit URI form as the security boundary.
_OPAQUE_OR_OVERRIDE_QUERY_OPTIONS: Mapping[str, frozenset[str]] = {
    "postgresql": frozenset({"host", "hostaddr", "service", "servicefile"}),
    "redshift": frozenset({"host", "hostaddr", "service", "servicefile", "endpoint_url"}),
    "mysql": frozenset({"read_default_file", "read_default_group", "unix_socket"}),
    "mariadb": frozenset({"read_default_file", "read_default_group", "unix_socket"}),
    "oracle": frozenset({"dsn"}),
    # Snowflake SQLAlchemy 1.11 also rejects these sensitive URL parameters.
    # Reject them here to give a stable Sift error before dialect behavior or
    # the legacy compatibility environment switch can change the result.
    "snowflake": frozenset({
        "host", "protocol", "token_file_path", "private_key_file",
        "ocsp_response_cache_filename", "connection_diag_log_path",
        "crl_cache_dir", "unsafe_file_write",
        "unsafe_skip_file_permissions_check",
    }),
}

# Some drivers can perform work while opening the connection or turn a SELECT
# into a write through default job configuration.  These options would evade
# the SQL statement gate because their side effect happens outside the SQL
# string Sift reviewed.
_CONNECTION_SIDE_EFFECT_QUERY_OPTIONS: Mapping[str, frozenset[str]] = {
    "mysql": frozenset({"init_command"}),
    "mariadb": frozenset({"init_command"}),
    "oracle": frozenset({"newpassword"}),
    "bigquery": frozenset({
        "clustering_fields", "create_disposition", "destination",
        "destination_encryption_configuration", "dry_run",
        "schema_update_options", "write_disposition",
    }),
}

# Reflection is metadata-only but an enterprise warehouse can still contain
# tens of thousands of schemas and objects.  Keep bridge responses bounded so
# a catalog click cannot freeze the desktop webview or create an accidental
# metadata dump.  The researcher can select a schema to narrow the next call.
MAX_CATALOG_SCHEMAS = 200
MAX_CATALOG_OBJECTS = 1_000
MAX_CATALOG_COLUMNS = 500

# Statements that read. Anything else is refused: a connector that can
# DROP a table is a liability in a tool whose users are pointed at
# production research databases, and read-only is the only posture
# that needs no trust in the query text.
_READ_ONLY_PREFIXES = ("select", "with", "values")

_WRITE_TOKENS = (
    "insert",
    "update",
    "delete",
    "drop",
    "truncate",
    "alter",
    "create",
    "grant",
    "revoke",
    "attach",
    "copy",
    "merge",
    "replace",
    "vacuum",
    "call",
    "execute",
    "exec",
    # PostgreSQL SELECT ... INTO creates a table; MySQL SELECT ... INTO
    # OUTFILE/DUMPFILE writes a server-side file. Both otherwise begin with
    # the read-looking SELECT prefix.
    "into",
)

# ODBC / driver-manager style connection strings use semicolon-
# delimited ``KEY=value`` pairs (``PWD=secret;``, ``Password=secret;``)
# instead of a ``scheme://user:pass@host`` URI. These never contain
# "://", so the URI-only logic below used to let them straight
# through unredacted — a real, reachable path: some ODBC drivers
# (pyodbc's SQL Server driver among them) echo their own constructed
# connection string, credentials included, into a connection-failure
# exception's message, and that message is exactly what flows through
# ``redact_connection`` at the ``_run_sqlalchemy``/``_run_duckdb``
# error sites above. Matches ``pwd``/``password`` case-insensitively
# up to the next ``;`` (or end of string) and blanks the value —
# applied unconditionally, not just when "://" is absent, so a
# message containing BOTH a URI and a separately-echoed key=value
# string still gets both forms scrubbed.
_KV_PASSWORD_PATTERN = re.compile(
    # ODBC permits semicolons inside a braced value and escapes a literal
    # closing brace as ``}}``. Treat the complete braced value as the secret;
    # stopping at its first semicolon would expose the remainder.
    r"(?i)\b(pwd|password)\s*=\s*(?:\{(?:[^}]|}})*\}|[^;]*)"
)

# URI query strings and cloud-driver error messages commonly carry bearer
# credentials outside the user:password authority component.  Redact the
# whole value while preserving the separator and key for diagnosis.
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&;]\s*(?:(?:[a-z0-9_-]*_)?token|api_?key|secret|client_?secret|"
    r"private_?key(?:_?(?:file(?:_?pwd)?|pwd))?|passcode|password|pwd|"
    r"credentials_?(?:base64|info))\s*=\s*)"
    r"(?:\{(?:[^}]|}})*\}|[^&;\s]*)"
)


def redact_connection(uri: str) -> str:
    """Return a connection string safe to display or log.

    Passwords are removed; the username and host are kept, which is
    enough for a researcher to recognise the connection with nothing
    reusable if the string reaches a log, a report, or the model.

    Parsed by scanning ``@`` right-to-left rather than with a tidy
    single regex, because real pasted URIs contain passwords with
    ``/``, ``:`` and even unencoded ``@``. An earlier regex-based
    version leaked such passwords — found by fuzzing, not review.

    Where the structure is ambiguous the function **over-redacts**:
    hiding part of a host is a cosmetic problem, while showing a
    password is a credential disclosure.
    """
    if not isinstance(uri, str):
        return ""
    # SQLAlchemy's common ``odbc_connect=...`` form percent-encodes the
    # entire semicolon-delimited driver string. Redact a decoded display
    # representation so ``PWD%3Dhunter2`` cannot evade the normal PWD rule.
    # This function returns display text only; the real connection string is
    # never modified before it reaches the driver.
    # Scrub query credentials before *and* after percent decoding. Before is
    # essential for a value such as ``token=abc%26def``: decoding first turns
    # the encoded ampersand into a delimiter and leaks ``def``. The second
    # pass covers encoded keys and credentials nested in ``odbc_connect``.
    uri = _QUERY_SECRET_PATTERN.sub(r"\1***", uri)
    uri = unquote(uri)
    uri = _QUERY_SECRET_PATTERN.sub(r"\1***", uri)
    uri = _KV_PASSWORD_PATTERN.sub(lambda m: f"{m.group(1)}=***", uri)
    if "://" not in uri:
        return _redact_schemeless_credentials(uri)
    scheme, _, remainder = uri.partition("://")
    positions = [i for i, ch in enumerate(remainder) if ch == "@"]
    for pos in reversed(positions):
        candidate = remainder[:pos]
        if ":" not in candidate:
            # ``user@host`` — a username with no password to hide.
            continue
        user = candidate.split(":", 1)[0]
        return f"{scheme}://{user}:***@{remainder[pos + 1 :]}"
    return uri


def _redact_schemeless_credentials(uri: str) -> str:
    """Redact credential-bearing connection strings that have no
    ``scheme://`` prefix at all and no ``pwd=``/``password=`` key --
    the gap the two branches above (and ``_KV_PASSWORD_PATTERN``)
    don't cover. The motivating case is Oracle: its EZConnect syntax
    is ``user/password@host:port/service`` (``/`` between user and
    password, no scheme), and its JDBC form is
    ``jdbc:oracle:thin:user/password@host:port:sid`` -- neither
    contains "://", so both used to sail straight through
    ``redact_connection`` with the password fully intact.

    Same right-to-left, over-redact-when-ambiguous approach as the
    scheme branch: scan for an "@", and if a ":" or "/" appears
    before it, treat everything before that separator as the
    "user" and blank the rest. No "@" at all, or an "@" with
    neither separator before it (a bare ``user@host`` — nothing
    that looks like a password), is left untouched.
    """
    positions = [i for i, ch in enumerate(uri) if ch == "@"]
    for pos in reversed(positions):
        candidate = uri[:pos]
        sep = max(candidate.rfind(":"), candidate.rfind("/"))
        if sep == -1:
            # No ":" or "/" before this "@" -- nothing that looks
            # like a user<sep>password pairing to hide.
            continue
        user = candidate[:sep]
        return f"{user}:***@{uri[pos + 1 :]}"
    return uri


class ConnectorError(IntegrationError):
    """A connector problem stated in terms the researcher can act on."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "database_error",
        retryable: bool = False,
        action: str = "Review the connection, query, and database permissions.",
    ) -> None:
        super().__init__(
            code,
            message,
            integration_id="database",
            retryable=retryable,
            action=action,
        )


@dataclass(frozen=True, repr=False)
class ConnectionSpec:
    """Host-only structured database connection.

    Most database connections are fully represented by a URI.  A few secure
    authentication mechanisms deliberately cannot be: Snowflake private keys
    and Databricks OAuth credential material belong in driver ``connect_args``
    and must never be copied into a URL, log, model message, or browser bridge.

    Instances are constructed by the vault layer or trusted host code.  The
    public connector validates the small provider-specific argument allowlist
    again before an engine is created.
    """

    uri: str
    authentication: str = "uri"
    connect_args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``frozen=True`` does not freeze a caller-owned dict. Copy it into an
        # immutable mapping so credential mode and arguments cannot change
        # between validation and engine construction.
        object.__setattr__(
            self, "connect_args", MappingProxyType(dict(self.connect_args)),
        )

    def __repr__(self) -> str:
        return (
            "ConnectionSpec(uri="
            f"{redact_connection(self.uri)!r}, authentication={self.authentication!r})"
        )


ConnectionInput = str | ConnectionSpec


_STRUCTURED_CONNECT_ARGS: Mapping[tuple[str, str], frozenset[str]] = {
    ("snowflake", "key_pair"): frozenset({"private_key"}),
    ("databricks", "oauth_u2m"): frozenset({"auth_type"}),
    ("databricks", "oauth_m2m"): frozenset({"credentials_provider"}),
}


def _connection_uri(connection: ConnectionInput) -> str:
    return connection.uri if isinstance(connection, ConnectionSpec) else connection


def _validated_connection_text(connection: ConnectionInput) -> str:
    """Return a bounded, single-line connection value or fail closed.

    URL parsers may silently remove tabs and newlines. Reject them before any
    classification or TLS-policy decision so validation always applies to the
    exact endpoint the researcher supplied.
    """
    raw = _connection_uri(connection)
    if not isinstance(raw, str) or not raw.strip():
        raise ConnectorError("no connection given")
    try:
        encoded_size = len(raw.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise ConnectorError("the database connection contains invalid text") from exc
    if encoded_size > MAX_CONNECTION_INPUT_BYTES:
        raise ConnectorError("the database connection exceeds the 64 KiB safety limit")
    if any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise ConnectorError("the database connection contains control characters")
    return raw.strip()


def _validated_structured_connect_args(
    connection: ConnectionInput,
    backend: str,
) -> dict[str, Any]:
    if not isinstance(connection, ConnectionSpec):
        return {}
    allowed = _STRUCTURED_CONNECT_ARGS.get(
        (backend, connection.authentication), frozenset(),
    )
    supplied = set(connection.connect_args)
    if not allowed or not supplied or not supplied <= allowed:
        raise ConnectorError(
            "the structured database authentication configuration is invalid"
        )
    if connection.authentication == "snowflake_key_pair":  # legacy typo guard
        raise ConnectorError("invalid Snowflake authentication configuration")
    if (backend, connection.authentication) == ("snowflake", "key_pair"):
        key = connection.connect_args.get("private_key")
        if not isinstance(key, bytes) or not key:
            raise ConnectorError("Snowflake key-pair authentication needs a private key")
    elif (backend, connection.authentication) == ("databricks", "oauth_u2m"):
        if connection.connect_args.get("auth_type") != "databricks-oauth":
            raise ConnectorError("Databricks U2M authentication is not configured")
    elif (backend, connection.authentication) == ("databricks", "oauth_m2m"):
        if not callable(connection.connect_args.get("credentials_provider")):
            raise ConnectorError("Databricks M2M authentication is not configured")
    return dict(connection.connect_args)


def snowflake_key_pair_connection(
    uri: str,
    *,
    private_key_pem: str,
    passphrase: str | None = None,
) -> ConnectionSpec:
    """Build a Snowflake key-pair connection without placing key data in a URI."""
    try:
        from sqlalchemy.engine import make_url

        url = make_url(uri)
    except Exception as exc:
        raise ConnectorError("the Snowflake connection URI is invalid") from exc
    if describe_backend(uri) != "snowflake":
        raise ConnectorError("Snowflake key-pair authentication needs a Snowflake URI")
    authenticator = str(url.query.get("authenticator", "")).casefold()
    if url.password or any(
        any(marker in key.casefold() for marker in (
            "token", "private_key", "passcode", "password",
        ))
        for key in url.query
    ) or authenticator not in {"", "snowflake_jwt"}:
        raise ConnectorError(
            "Snowflake key-pair authentication cannot be combined with a "
            "password, token, or different authenticator"
        )
    try:
        from cryptography.hazmat.primitives import serialization

        key = serialization.load_pem_private_key(
            private_key_pem.encode("utf-8"),
            password=passphrase.encode("utf-8") if passphrase else None,
        )
        private_key = key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    except Exception as e:
        raise ConnectorError(
            f"could not load the Snowflake private key ({type(e).__name__})"
        ) from e
    return ConnectionSpec(
        uri, authentication="key_pair", connect_args={"private_key": private_key},
    )


def databricks_oauth_connection(
    uri: str,
    *,
    mode: Literal["oauth_u2m", "oauth_m2m"],
    client_id: str | None = None,
    client_secret: str | None = None,
) -> ConnectionSpec:
    """Build a Databricks OAuth connection using native driver arguments."""
    try:
        from sqlalchemy.engine import make_url

        url = make_url(uri)
    except Exception as exc:
        raise ConnectorError("the Databricks connection URI is invalid") from exc
    if describe_backend(uri) != "databricks":
        raise ConnectorError("Databricks OAuth authentication needs a Databricks URI")
    conflicting = {
        "token", "access_token", "auth_type", "oauth_client_id",
        "oauth_client_secret", "credentials_provider",
    }
    if url.password or any(key.casefold() in conflicting for key in url.query):
        raise ConnectorError(
            "Databricks OAuth cannot be combined with a token or a second "
            "authentication configuration in the URI"
        )
    args: dict[str, Any] = {"auth_type": "databricks-oauth"}
    if mode == "oauth_m2m":
        if not client_id or not client_secret:
            raise ConnectorError("Databricks M2M needs a client ID and client secret")
        host = url.host
        if not host:
            raise ConnectorError("the Databricks connection URI needs a host")
        if url.port and url.port != 443:
            host = f"{host}:{url.port}"

        @dataclass(frozen=True, repr=False)
        class _CredentialsProvider:
            hostname: str
            oauth_client_id: str
            oauth_client_secret: str
            timeout_seconds: int = 15

            def __repr__(self) -> str:
                return "DatabricksOAuthM2MCredentialsProvider(***)"

            def with_timeout(self, seconds: int) -> "_CredentialsProvider":
                return replace(self, timeout_seconds=max(1, int(seconds)))

            def __call__(self) -> Any:
                try:
                    from databricks.sdk.core import Config, oauth_service_principal
                except ImportError as exc:
                    raise ConnectorError(
                        "Databricks OAuth M2M needs the databricks-sdk package; "
                        "install Sift's databricks database extra"
                    ) from exc
                config = Config(
                    host=f"https://{self.hostname}",
                    client_id=self.oauth_client_id,
                    client_secret=self.oauth_client_secret,
                    http_timeout_seconds=float(self.timeout_seconds),
                    retry_timeout_seconds=self.timeout_seconds,
                )
                return oauth_service_principal(config)

        # The SQL connector's supported Thrift/SEA M2M path is an external
        # credentials provider. ``oauth_client_secret`` by itself is consumed
        # only by its optional kernel backend and is ignored by the default
        # connector, which could silently start an interactive U2M flow.
        args = {"credentials_provider": _CredentialsProvider(
            host, client_id, client_secret,
        )}
    elif mode != "oauth_u2m":  # pragma: no cover - static type plus runtime gate
        raise ConnectorError("unsupported Databricks OAuth mode")
    return ConnectionSpec(uri, authentication=mode, connect_args=args)


def _check_cancellation(cancellation: CancellationToken | None, action: str) -> None:
    if cancellation is None:
        return
    try:
        cancellation.raise_if_cancelled()
    except IntegrationCancelled as e:
        raise ConnectorError(
            f"database {action} cancelled",
            code="cancelled",
            action=f"Start a new {action} when ready.",
        ) from e


@dataclass(frozen=True)
class ExtractResult:
    dataset_path: Path
    rows: int
    columns: int
    truncated: bool
    backend: str
    connection_display: str
    query_sha256: str
    dataset_sha256: str
    # Position-aware rename records make duplicate database result names
    # auditable without changing the long-standing fields above.
    column_renames: tuple[dict[str, Any], ...] = ()
    canonical_fingerprint: str | None = None


@dataclass(frozen=True)
class ConnectionCheck:
    """Researcher-visible connection diagnostic with no sampled data."""

    backend: str
    connection_display: str
    latency_ms: int
    server_version: str | None
    read_only_enforcement: str


@dataclass(frozen=True)
class ExtractionProgress:
    """A bounded, value-free extraction progress event."""

    stage: Literal["starting", "querying", "materializing", "finalizing", "complete"]
    rows_materialized: int = 0
    bytes_buffered: int = 0


@dataclass(frozen=True)
class QueryTransportEvidence:
    """Value-free transport attestation for live connector qualification."""

    backend: str
    transport: Literal["inline", "cloudfetch"]


def _query_transport_evidence(cursor: Any, backend: str) -> QueryTransportEvidence:
    active = getattr(cursor, "active_result_set", None)
    results = getattr(active, "results", None)
    is_cloudfetch = (
        type(results).__name__ == "CloudFetchQueue"
        and type(results).__module__.startswith("databricks.sql")
    )
    return QueryTransportEvidence(
        backend=backend,
        transport="cloudfetch" if is_cloudfetch else "inline",
    )


@dataclass(frozen=True)
class QueryPreview:
    """Metadata-only query preflight; never contains SQL or sample values."""

    backend: str
    connection_display: str
    query_sha256: str
    read_only_enforcement: str
    dry_run_supported: bool
    estimate_source: str | None
    estimated_bytes: int | None
    estimated_rows: int | None
    metered_warehouse: bool
    warnings: tuple[str, ...]
    executes_query: bool = False


@dataclass(frozen=True)
class DatabaseCatalog:
    """A bounded, metadata-only view of a database namespace.

    Catalog names can themselves be confidential, so this structure is only
    returned by researcher-driven bridge methods. It is deliberately absent
    from the model tool registry.
    """

    backend: str
    connection_display: str
    default_schema: str | None
    schemas: tuple[str, ...]
    objects: tuple[dict[str, Any], ...]
    schemas_truncated: bool
    objects_truncated: bool
    warnings: tuple[str, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def database_query_timeout_seconds() -> int:
    """Return the bounded host/database query deadline."""
    raw = os.environ.get("SIFT_DATABASE_QUERY_TIMEOUT_SECONDS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_QUERY_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_QUERY_TIMEOUT_SECONDS
    return max(1, min(value, MAX_QUERY_TIMEOUT_SECONDS))


def database_connect_timeout_seconds() -> int:
    """Return the bounded DNS/TCP/authentication deadline.

    This setting contains no endpoint or credential material and is safe to
    include in diagnostics. Malformed values fall back to the secure default;
    they never disable the limit.
    """
    raw = os.environ.get("SIFT_DATABASE_CONNECT_TIMEOUT_SECONDS", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_CONNECT_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_CONNECT_TIMEOUT_SECONDS
    return max(1, min(value, MAX_CONNECT_TIMEOUT_SECONDS))


def _native_connect_timeout_seconds(outer_remaining: float | None = None) -> int:
    """Clamp a driver's whole-second timeout to the remaining operation."""
    timeout = database_connect_timeout_seconds()
    if outer_remaining is not None:
        timeout = min(timeout, max(1, int(outer_remaining)))
    return timeout


def database_cost_warning_bytes() -> int:
    """Return the minimum dry-run estimate that warrants a cost warning."""
    default = 1024 * 1024 * 1024
    raw = os.environ.get("SIFT_DATABASE_COST_WARNING_BYTES", "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(1, min(value, 10 * 1024**5))


def _emit_progress(
    callback: Callable[[ExtractionProgress], None] | None,
    stage: Literal["starting", "querying", "materializing", "finalizing", "complete"],
    *,
    rows: int = 0,
    bytes_buffered: int = 0,
) -> None:
    if callback is None:
        return
    try:
        callback(ExtractionProgress(stage, max(0, rows), max(0, bytes_buffered)))
    except Exception:  # noqa: BLE001, S110 - observer cannot break extraction
        pass


# A syntactically read-only SELECT can still reach a callable
# function that performs a write, a filesystem read, or a network
# call entirely inside the database server — none of that shows up
# as a top-level write KEYWORD in the SQL text, so ``_WRITE_TOKENS``
# alone never catches it (``SELECT pg_read_file('/etc/passwd')`` and
# ``SELECT lo_export(loid, '/tmp/x')`` are both, syntactically, plain
# SELECTs). This is a best-effort BLOCKLIST of well-known cross-
# backend functions with exactly this shape, not a claim of
# completeness — no text scan can ever enumerate every side-effecting
# function a target server might expose (a researcher's own
# extensions, custom stored procedures, etc. are fundamentally out of
# reach here). Matched the same way ``_WRITE_TOKENS`` is: a name
# immediately followed by ``(``, case-insensitively, as a standalone
# word so a column merely NAMED e.g. "sleep_duration" doesn't trip it.
_COMMON_DANGEROUS_FUNCTIONS = (
    # Generic resource-exhaustion / timing primitives.
    "sleep",
    "benchmark",
)

# Backend-specific names are kept separate both for reviewability and so the
# preview surface can explain which database rule rejected a query.  When no
# backend is supplied ``normalize_sql`` deliberately checks the union: callers
# that have not classified the connection receive the safest possible answer.
_DANGEROUS_FUNCTIONS_BY_BACKEND: dict[str, tuple[str, ...]] = {
    "postgresql": (
    # Server-side filesystem / network / admin escapes.
    "pg_read_file",
    "pg_read_binary_file",
    "pg_ls_dir",
    "pg_ls_logdir",
    "pg_ls_waldir",
    "pg_stat_file",
    "lo_export",
    "lo_import",
    "dblink",
    "dblink_exec",
    "dblink_connect",
    "pg_terminate_backend",
    "pg_cancel_backend",
    "pg_reload_conf",
    "pg_rotate_logfile",
    "pg_advisory_lock",
    "pg_advisory_lock_shared",
    "pg_try_advisory_lock",
    "pg_try_advisory_lock_shared",
    "pg_advisory_unlock",
    "pg_advisory_unlock_all",
    "pg_advisory_unlock_shared",
    "pg_notify",
    "pg_logical_emit_message",
    "nextval",
    "setval",
    "set_config",
    "pg_sleep",
    "pg_sleep_for",
    "pg_sleep_until",
    ),
    "redshift": (
        "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_sleep",
    ),
    "mysql": (
    "load_file",
    "sys_exec",
    "sys_eval",
    "master_pos_wait",
    "get_lock",
    ),
    "mariadb": (
        "load_file", "sys_exec", "sys_eval", "master_pos_wait", "get_lock",
    ),
    "mssql": (
    "xp_cmdshell",
    "sp_configure",
    "openrowset",
    "opendatasource",
    "openquery",
    "waitfor",
    ),
    "sqlite": ("load_extension", "readfile", "writefile"),
    # DuckDB can otherwise turn a SELECT into an arbitrary local-file or
    # remote-object reader.  Researcher-selected file sources are registered
    # as the fixed ``source`` view before the query is run.
    "duckdb": (
        "read_blob", "read_csv", "read_csv_auto", "read_json",
        "read_json_auto", "read_ndjson", "read_parquet", "parquet_scan",
        "sqlite_scan", "postgres_scan", "postgres_query", "mysql_scan",
        "httpfs", "glob",
    ),
    "duckdb-file": (
        "read_blob", "read_csv", "read_csv_auto", "read_json",
        "read_json_auto", "read_ndjson", "read_parquet", "parquet_scan",
        "sqlite_scan", "postgres_scan", "postgres_query", "mysql_scan",
        "httpfs", "glob",
    ),
    "oracle": (
        "utl_http.request", "utl_http.request_pieces",
        "utl_inaddr.get_host_address", "dbms_lock.sleep",
        "dbms_pipe.receive_message", "dbms_ldap.init",
    ),
    "snowflake": (
        "get_presigned_url", "get_stage_location", "build_scoped_file_url",
    ),
    "bigquery": ("external_query",),
    "databricks": ("java_method", "reflect"),
}

_SQLGLOT_DIALECTS = {
    "postgresql": "postgres", "redshift": "redshift", "mysql": "mysql",
    "mariadb": "mysql", "mssql": "tsql", "sqlite": "sqlite",
    "duckdb": "duckdb", "duckdb-file": "duckdb", "oracle": "oracle",
    "snowflake": "snowflake", "bigquery": "bigquery",
    "databricks": "databricks",
}


_DOLLAR_QUOTE = re.compile(r"\$(?:[A-Za-z_][A-Za-z0-9_]*)?\$")


def _sql_code_mask(sql: str) -> str | None:
    """Return a same-length view containing only executable SQL code.

    String literals, quoted identifiers, and comments are replaced with
    spaces. Newlines are retained so token boundaries cannot accidentally
    collapse. ``None`` means a quoted region or block comment was not closed.

    This is intentionally a small lexical scanner rather than a SQL parser:
    the latter would need to select a dialect before Sift has connected.  It
    covers the quoting forms used by the supported engines, including
    PostgreSQL dollar quotes, SQL Server brackets, and MySQL backticks.
    """
    out = list(sql)
    length = len(sql)
    index = 0

    def blank(start: int, end: int) -> None:
        for pos in range(start, end):
            if out[pos] not in "\r\n":
                out[pos] = " "

    while index < length:
        if sql.startswith("--", index):
            end = sql.find("\n", index + 2)
            if end < 0:
                end = length
            blank(index, end)
            index = end
            continue

        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            if end < 0:
                return None
            end += 2
            blank(index, end)
            index = end
            continue

        char = sql[index]
        # Oracle alternative quoting: q'[text]', q'{text}', q'(text)',
        # q'<text>', and q'!text!'. Without this branch, a semicolon or word
        # such as DROP inside legitimate Oracle text was mistaken for code.
        if (
            char in {"q", "Q"}
            and index + 2 < length
            and sql[index + 1] == "'"
        ):
            opener = sql[index + 2]
            closer = {"[": "]", "{": "}", "(": ")", "<": ">"}.get(
                opener, opener
            )
            closing = closer + "'"
            end = sql.find(closing, index + 3)
            if end < 0:
                return None
            end += len(closing)
            blank(index, end)
            index = end
            continue

        if char in {"'", '"', "`"}:
            quote = char
            start = index
            index += 1
            while index < length:
                if sql[index] == quote:
                    # SQL escapes quote characters by doubling them.
                    if index + 1 < length and sql[index + 1] == quote:
                        index += 2
                        continue
                    index += 1
                    blank(start, index)
                    break
                # MySQL and a number of drivers also accept backslash escapes.
                if sql[index] == "\\" and index + 1 < length:
                    index += 2
                else:
                    index += 1
            else:
                return None
            continue

        if char == "[":
            start = index
            index += 1
            while index < length:
                if sql[index] == "]":
                    if index + 1 < length and sql[index + 1] == "]":
                        index += 2
                        continue
                    index += 1
                    blank(start, index)
                    break
                index += 1
            else:
                return None
            continue

        if char == "$":
            match = _DOLLAR_QUOTE.match(sql, index)
            if match:
                delimiter = match.group(0)
                end = sql.find(delimiter, match.end())
                if end < 0:
                    return None
                end += len(delimiter)
                blank(index, end)
                index = end
                continue

        index += 1

    return "".join(out)


def _dialect_query_is_read_only(sql: str, backend: str) -> bool:
    """Use the selected backend grammar as defense in depth.

    SQLGlot intentionally does not replace the lexical scanner: vendor
    extensions evolve faster than a client parser.  A successful parse must
    describe one query expression and may not contain a mutating node.  A
    parser miss falls back to the stricter lexical policy, except for
    ``duckdb-file`` where the fixed source-view boundary must be proven.
    """
    try:
        from sqlglot import exp, parse
        from sqlglot.errors import ParseError
    except ImportError:  # pragma: no cover - declared runtime dependency
        return backend != "duckdb-file"
    try:
        statements = [
            item for item in parse(sql, read=_SQLGLOT_DIALECTS.get(backend))
            if item is not None
        ]
    except (ParseError, ValueError):
        return backend != "duckdb-file"
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return False
    tree = statements[0]
    mutating = (
        exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter,
        exp.Merge, exp.Command, exp.Transaction,
    )
    if any(tree.find(kind) is not None for kind in mutating):
        return False

    if backend in {"duckdb", "duckdb-file"}:
        cte_names = {
            str(cte.alias_or_name).casefold()
            for cte in tree.find_all(exp.CTE)
            if cte.alias_or_name
        }
        for table in tree.find_all(exp.Table):
            name = str(table.name or "")
            lowered = name.casefold()
            looks_external = (
                "/" in name or "\\" in name or "://" in name
                or lowered.endswith((
                    ".parquet", ".csv", ".tsv", ".json", ".jsonl",
                    ".sqlite", ".sqlite3", ".db", ".duckdb", ".ddb",
                ))
            )
            if looks_external:
                return False
            if (
                backend == "duckdb-file"
                and lowered != "source"
                and lowered not in cte_names
            ):
                return False
    return True


def normalize_sql(sql: str, backend: str | None = None) -> str | None:
    """Return the single read-only statement in ``sql``, or None.

    One function decides both *whether* the statement may run and
    *what text* runs, so the two can never disagree. That mattered
    immediately: an early version validated the raw string (accepting
    a trailing semicolon) and then wrapped it in a ``SELECT * FROM
    (...)`` subquery for the row cap, producing a syntax error at the
    database for a query the gate had approved.

    The normalized form preserves the researcher's exact query except for an
    optional final semicolon. A lexical mask ensures punctuation and words in
    literals, comments, and quoted identifiers are never mistaken for code.

    Deliberately strict rather than clever: exactly one statement, a
    read-only prefix, no write keyword anywhere as a code word, and no call
    to a known dangerous function (see
    ``_DANGEROUS_FUNCTIONS`` — a syntactically read-only SELECT can
    still invoke a function that writes a file, opens a remote
    connection, or blocks the server). A real parser would be more
    precise; strictness is worth more than precision when the
    downside is a dropped production table or a server-side file
    read.
    """
    if not isinstance(sql, str) or not sql.strip():
        return None
    body = sql.strip()
    code_mask = _sql_code_mask(body)
    if code_mask is None or not code_mask.strip():
        return None

    semicolons = [pos for pos, char in enumerate(code_mask) if char == ";"]
    if semicolons:
        final = semicolons[0]
        # Only one semicolon is allowed and it must be the last non-space
        # character in the submitted text. Refusing a comment after it keeps
        # the returned query byte-for-byte equivalent after removal.
        if len(semicolons) != 1 or body[final + 1 :].strip():
            return None
        body = body[:final].rstrip()
        code_mask = code_mask[:final].rstrip()

    lowered = code_mask.strip().lower()
    if not re.match(r"^(?:select|with|values)\b", lowered):
        return None
    if any(re.search(rf"\b{tok}\b", lowered) for tok in _WRITE_TOKENS):
        return None
    # Locking reads are not harmless on a shared production database: they can
    # block writers for the full bounded query timeout even though they return
    # rows and contain no write keyword. Read-only transaction modes reject
    # many of these, but the lexical gate keeps behavior consistent across
    # vendors and driver versions before a connection is opened.
    if re.search(r"\bfor\s+(?:no\s+key\s+)?update\b", lowered):
        return None
    if re.search(r"\bfor\s+(?:key\s+)?share\b", lowered):
        return None
    if re.search(r"\block\s+in\s+share\s+mode\b", lowered):
        return None
    if backend is None:
        dangerous = _COMMON_DANGEROUS_FUNCTIONS + tuple(
            name
            for names in _DANGEROUS_FUNCTIONS_BY_BACKEND.values()
            for name in names
        )
    else:
        dangerous = (
            _COMMON_DANGEROUS_FUNCTIONS
            + _DANGEROUS_FUNCTIONS_BY_BACKEND.get(backend, ())
        )
    if any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(fn)}\s*\(", lowered)
        for fn in dangerous
    ):
        return None
    if backend is not None and not _dialect_query_is_read_only(body, backend):
        return None
    return body


def _is_read_only(sql: str) -> bool:
    """Backwards-compatible predicate over :func:`normalize_sql`."""
    return normalize_sql(sql) is not None


def describe_backend(connection: ConnectionInput) -> str:
    """Classify a connection string into a backend label."""
    conn = _validated_connection_text(connection)
    if "://" in conn:
        raw_backend = conn.split("://", 1)[0].split("+", 1)[0].lower()
        backend = _BACKEND_ALIASES.get(raw_backend, raw_backend)
        if backend == "sqlite" or backend in _SUPPORTED_REMOTE_BACKENDS:
            return backend
        raise ConnectorError(
            f"database backend {raw_backend!r} is not supported. Use one of "
            "Sift's reviewed database adapters; arbitrary SQLAlchemy dialects "
            "are refused because their transport and read-only controls are "
            "unknown."
        )
    suffix = Path(conn).suffix.lower()
    if suffix in (".duckdb", ".ddb"):
        return "duckdb"
    if suffix in (".db", ".sqlite", ".sqlite3"):
        return "sqlite"
    if suffix in (".parquet", ".csv", ".tsv", ".json", ".jsonl"):
        # DuckDB queries files directly — the natural way to run SQL
        # over a Parquet extract too large for pandas.
        return "duckdb-file"
    raise ConnectorError(
        f"could not tell what kind of connection {redact_connection(conn)!r} "
        f"is. Use a SQLAlchemy URI (postgresql://…, mysql://…, "
        f"mssql+pyodbc://…, snowflake://…), or a path to a .duckdb, "
        f".sqlite or .parquet file."
    )


def _connection_secrets(connection: ConnectionInput) -> tuple[str, ...]:
    """Extract exact credential values for defensive error redaction."""
    values: set[str] = set()
    raw_connection = _connection_uri(connection)
    decoded = unquote(raw_connection) if isinstance(raw_connection, str) else ""
    try:
        from sqlalchemy.engine import make_url

        url = make_url(raw_connection)
        if url.password:
            values.add(str(url.password))
        for key, value in url.query.items():
            normalized_key = str(key).casefold().replace("-", "_")
            if normalized_key in {
                "access_token", "token", "api_key", "secret",
                "client_secret", "private_key", "password", "pwd",
                "odbc_connect", "credentials_base64", "credentials_info",
            } or any(marker in normalized_key for marker in (
                "token", "secret", "password", "private_key", "passcode",
            )):
                values.add(unquote(str(value)))
    except Exception:  # noqa: BLE001 - tolerate odd connection strings
        values.clear()
    for match in re.finditer(
        r"(?i)\b(?:pwd|password|token|access_?token|api_?key|"
        r"client_?secret|private_?key)\s*=\s*([^;&\s]+)",
        decoded,
    ):
        values.add(match.group(1))
    if isinstance(connection, ConnectionSpec):
        # Driver setup failures may echo native connect arguments even though
        # they were never placed in the URI. Treat every string captured by a
        # structured provider object as sensitive; over-redaction is safer
        # than guessing which vendor field a future driver might print.
        for value in connection.connect_args.values():
            if isinstance(value, str) and value:
                values.add(value)
            elif isinstance(value, bytes) and value:
                # A badly behaved DBAPI can echo a native private-key argument
                # using Python's bytes representation. It never belongs in a
                # researcher-facing diagnostic.
                values.add(str(value))
            for attribute in ("oauth_client_id", "oauth_client_secret"):
                nested = getattr(value, attribute, None)
                if isinstance(nested, str) and nested:
                    values.add(nested)
    return tuple(value for value in values if value)


def _safe_connector_error(error: Any, connection: ConnectionInput) -> str:
    """Return one bounded, credential-safe driver diagnostic line."""
    from sift.provider.error_safety import provider_error_message

    message = provider_error_message(
        error,
        secrets=_connection_secrets(connection),
    )
    return redact_connection(message).split("\n", 1)[0][:300]


def _driver_connector_error(
    prefix: str,
    error: Any,
    connection: ConnectionInput,
) -> ConnectorError:
    """Classify authentication failures without returning driver secrets."""
    safe = _safe_connector_error(error, connection)
    lowered = safe.casefold()
    code = str(
        getattr(error, "sqlstate", "")
        or getattr(getattr(error, "orig", None), "sqlstate", "")
        or ""
    ).upper()
    expired = code in {"28001", "28P01"} and "expired" in lowered
    expired = expired or any(phrase in lowered for phrase in (
        "token expired", "token has expired", "password expired",
        "password has expired", "authentication expired",
        "expired authentication", "oauth token is expired",
        "credential has expired", "credentials have expired",
    ))
    auth_failed = code.startswith("28") or any(phrase in lowered for phrase in (
        "authentication failed", "access denied for user", "login failed",
        "invalid credentials", "invalid username/password",
    ))
    if expired:
        return ConnectorError(
            f"{prefix}: authentication expired",
            code="authentication_expired",
            retryable=True,
            action=(
                "Refresh the managed identity/session or rotate the saved "
                "database credential, then retry."
            ),
        )
    if auth_failed:
        return ConnectorError(
            f"{prefix}: authentication failed",
            code="authentication_failed",
            action="Verify or rotate the database credential and permissions.",
        )
    return ConnectorError(f"{prefix}: {safe}")


def _driver_error_is_timeout(error: Any, connection: ConnectionInput) -> bool:
    """Recognize safe, common DBAPI/SDK timeout classifications."""
    name = type(error).__name__.casefold()
    safe = _safe_connector_error(error, connection).casefold()
    return "timeout" in name or any(phrase in safe for phrase in (
        "timed out", "timeout", "time-out", "deadline exceeded",
    ))


def _connect_deadline_error(
    action: str,
    seconds: int | None = None,
) -> ConnectorError:
    seconds = database_connect_timeout_seconds() if seconds is None else seconds
    return ConnectorError(
        f"database {action} could not establish a connection within "
        f"Sift's {seconds}-second connection deadline",
        code="deadline_exceeded",
        retryable=True,
        action=(
            "Check DNS, network reachability, TLS, and the authentication "
            "service, then retry. An administrator may raise the bounded "
            "connection timeout for an approved environment."
        ),
    )


def _database_driver_install_guidance(backend: str) -> str:
    """Return the reviewed Sift extra for a database backend, if any.

    Keep this lookup tied to the public integration catalog so an error shown
    to a researcher cannot drift from the driver packaged and advertised by
    Sift. The fallback is deliberately generic for future/custom dialects.
    """
    try:
        from sift.integrations import DATABASE_ADAPTERS

        adapter = next(item for item in DATABASE_ADAPTERS if item.id == backend)
    except (ImportError, StopIteration):
        return "Check the URI and install the database driver for this backend."
    if adapter.install_extra == "built-in":
        return "Check the URI and that this built-in database support is available."
    return (
        f"Install Sift's {adapter.label} support with "
        f'`pip install "sift[{adapter.install_extra}]"`, then retry.'
    )


def _engine_creation_error(error: Any, backend: str) -> ConnectorError:
    """Make SQLAlchemy engine-setup failures actionable without secrets.

    Driver imports can fail before a connection exists, so forwarding the
    exception would needlessly expose portions of a URI or environment. The
    exception class is enough to distinguish the failure while the canonical
    catalog supplies a safe, exact recovery step.
    """
    return ConnectorError(
        f"could not initialize the {backend} database driver "
        f"({type(error).__name__}). {_database_driver_install_guidance(backend)}"
    )


def _connection_timeout_args(
    connection: ConnectionInput,
    backend: str,
    seconds: int,
) -> dict[str, Any]:
    """Return reviewed DBAPI connect/read timeout arguments.

    The mapping is selected from the effective SQLAlchemy driver name, not
    solely from the dialect, because connect keyword spellings are DBAPI
    contracts.  It intentionally contains only numeric policy values: URI
    credentials are never copied into logs, exceptions, or thread state.
    """
    try:
        from sqlalchemy.engine import make_url

        driver = make_url(_connection_uri(connection)).drivername.casefold()
    except Exception:
        driver = backend.casefold()
    timeout = max(1, min(int(seconds), MAX_CONNECT_TIMEOUT_SECONDS))
    if backend == "postgresql":
        if driver == "postgresql+psycopg":
            return {"connect_timeout": timeout}
        raise ConnectorError(
            "PostgreSQL connections require Sift's reviewed psycopg driver "
            "so login and socket waits can be bounded"
        )
    if backend == "redshift":
        if "redshift_connector" in driver:
            return {"timeout": timeout}
        raise ConnectorError(
            "Redshift connections require Sift's reviewed "
            "redshift_connector driver so login and socket waits can be bounded"
        )
    if backend in {"mysql", "mariadb"}:
        # PyMySQL and the reviewed MySQL DBAPIs distinguish establishment
        # from subsequent socket reads/writes. Bound all three so read-only
        # session setup and the first fetch cannot inherit an infinite socket.
        if driver not in {"mysql+pymysql", "mariadb+pymysql"}:
            raise ConnectorError(
                f"{backend} connections require Sift's reviewed PyMySQL "
                "driver so connect, read, and write waits can be bounded"
            )
        return {
            "connect_timeout": timeout,
            "read_timeout": timeout,
            "write_timeout": timeout,
        }
    if backend == "mssql":
        if driver == "mssql+pyodbc":
            return {"timeout": timeout}
        raise ConnectorError(
            "SQL Server connections require Sift's reviewed pyodbc "
            "driver so DNS, login, and socket waits can be bounded"
        )
    if backend == "oracle":
        if "oracledb" in driver:
            return {"tcp_connect_timeout": float(timeout)}
        raise ConnectorError(
            "Oracle connections require Sift's reviewed oracledb driver so "
            "TCP establishment can be bounded"
        )
    if backend == "snowflake":
        return {
            "login_timeout": timeout,
            "network_timeout": timeout,
            "socket_timeout": timeout,
        }
    if backend == "databricks":
        if driver == "databricks":
            return {"_socket_timeout": float(timeout)}
        raise ConnectorError(
            "Databricks connections require Sift's reviewed "
            "databricks-sqlalchemy driver so socket waits can be bounded"
        )
    # SQLite is local and BigQuery uses its HTTP SDK timeout at the request
    # site. Unknown explicit drivers get SQLAlchemy's normal arguments rather
    # than an unreviewed keyword that could make a supported URI unusable.
    return {}


def _create_bounded_engine(
    connection: ConnectionInput,
    backend: str,
    *,
    timeout_cap_seconds: float | None = None,
) -> Any:
    """Create an engine whose first network operation is already bounded."""
    try:
        from sqlalchemy import create_engine
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ConnectorError("sqlalchemy is not installed") from e
    # Defense in depth for trusted host callers that invoke this internal
    # helper without first passing through `_prepare_connection`.
    _validate_reviewed_database_driver(connection, backend)
    # Reviewed DBAPIs accept whole seconds. Flooring (with their required
    # one-second minimum) ensures the native wait never exceeds a longer outer
    # operation budget. Production operation budgets are themselves clamped
    # to at least one second.
    timeout = _native_connect_timeout_seconds(timeout_cap_seconds)
    kwargs: dict[str, Any] = {}
    engine_connection = _connection_uri(connection)
    if backend == "bigquery":
        # sqlalchemy-bigquery exposes ``billing_project_id`` as a dialect
        # constructor argument, not as a URL option. Sift profiles keep it in
        # the URI, so move it to the documented argument before URL parsing.
        try:
            from sqlalchemy.engine import make_url

            bigquery_url = make_url(engine_connection)
            billing_project = bigquery_url.query.get("billing_project_id")
            if billing_project:
                if not isinstance(billing_project, str):
                    raise ConnectorError(
                        "BigQuery billing_project_id must be specified once"
                    )
                kwargs["billing_project_id"] = billing_project
                bigquery_url = bigquery_url.difference_update_query(
                    ["billing_project_id"]
                )
                engine_connection = bigquery_url.render_as_string(
                    hide_password=False
                )
        except ConnectorError:
            raise
        except Exception as exc:
            raise ConnectorError("the BigQuery connection URI is invalid") from exc
    connect_args = _connection_timeout_args(connection, backend, timeout)
    structured = _validated_structured_connect_args(connection, backend)
    if (backend, getattr(connection, "authentication", None)) == (
        "databricks", "oauth_m2m",
    ):
        # OAuth discovery/token acquisition and the SQL socket share the same
        # connection-establishment budget instead of each receiving the full
        # allowance sequentially.
        provider = structured["credentials_provider"]
        provider_timeout = max(1, timeout // 2)
        with_timeout = getattr(provider, "with_timeout", None)
        if callable(with_timeout):
            structured["credentials_provider"] = with_timeout(provider_timeout)
        if "_socket_timeout" in connect_args:
            connect_args["_socket_timeout"] = float(
                max(1, timeout - provider_timeout)
            )
    overlap = set(connect_args) & set(structured)
    if overlap:
        raise ConnectorError(
            "structured authentication attempted to override Sift's connection timeout"
        )
    connect_args.update(structured)
    if connect_args:
        kwargs["connect_args"] = connect_args
    # A saturated remote pool is another pre-connection wait. Sift currently
    # creates one short-lived engine per operation, but bounding this now keeps
    # that guarantee true if pooling is later reused.
    if backend not in {"sqlite", "bigquery"}:
        kwargs["pool_timeout"] = timeout
    try:
        return create_engine(engine_connection, **kwargs)
    except Exception as e:
        raise _engine_creation_error(e, backend) from e


class _DatabaseDeadline:
    """One monotonic operation budget, including connect and initial calls."""

    def __init__(
        self,
        cancellation: CancellationToken | None,
        action: str,
    ) -> None:
        self.timeout_seconds = database_query_timeout_seconds()
        self.ends_at = time.monotonic() + self.timeout_seconds
        self.cancellation = cancellation
        self.action = action

    @property
    def remaining(self) -> float:
        return max(0.0, self.ends_at - time.monotonic())

    def check(self) -> None:
        _check_cancellation(self.cancellation, self.action)
        if self.remaining <= 0:
            raise _database_deadline_error(
                self.action, self.timeout_seconds,
            )


def _database_deadline_error(action: str, seconds: int) -> ConnectorError:
    return ConnectorError(
        f"database {action} exceeded Sift's {seconds}-second deadline",
        code="deadline_exceeded",
        retryable=True,
        action=(
            "Check network reachability or narrow the operation, then retry. "
            "An administrator may raise the bounded database timeout for an "
            "approved long-running operation."
        ),
    )


@dataclass
class _OperationInterruptState:
    cancelled: threading.Event
    timed_out: threading.Event
    raw: Any
    active_result: Any | None = None
    interrupt_mechanism: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _interrupt_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bind_result(self, result: Any) -> None:
        with self._lock:
            self.active_result = result

    def interrupt(self) -> None:
        # Deadline and user-cancellation signals can race. Most DBAPI cancel
        # methods are not re-entrant, so exactly one thread owns interruption.
        with self._interrupt_lock:
            if self.interrupt_mechanism not in {None, "unavailable"}:
                return
            with self._lock:
                result = self.active_result
            self.interrupt_mechanism = _interrupt_driver_operation(
                result, self.raw,
            )


@contextmanager
def _interrupt_connected_operation(
    conn: Any,
    *,
    deadline: _DatabaseDeadline,
) -> Any:
    """Interrupt an in-flight operation once a DBAPI connection exists.

    No database operation is moved to a background thread. The only helper
    thread watches the immutable cancellation signal and can merely interrupt
    or close the already-owned connection; it never possesses credentials or
    starts work of its own.
    """
    raw = getattr(conn.connection, "driver_connection", conn.connection)
    stopped = threading.Event()
    cancelled = threading.Event()
    timed_out = threading.Event()

    state = _OperationInterruptState(cancelled, timed_out, raw)

    def expire() -> None:
        timed_out.set()
        state.interrupt()

    timer = threading.Timer(max(0.001, deadline.remaining), expire)
    timer.daemon = True
    timer.start()
    watcher: threading.Thread | None = None
    cancellation = deadline.cancellation
    if cancellation is not None:
        def watch_cancellation() -> None:
            while not stopped.is_set():
                if cancellation.wait(0.05):
                    cancelled.set()
                    state.interrupt()
                    return

        watcher = threading.Thread(target=watch_cancellation, daemon=True)
        watcher.start()
    try:
        yield state
        if cancelled.is_set():
            _check_cancellation(cancellation, deadline.action)
        if timed_out.is_set():
            raise _database_deadline_error(
                deadline.action, deadline.timeout_seconds,
            )
    except BaseException as exc:
        # Driver cancellation commonly surfaces as a driver-specific error,
        # and read-only/timeout setup may already have wrapped that error as a
        # ConnectorError. The interrupt cause remains authoritative.
        if cancelled.is_set():
            try:
                _check_cancellation(cancellation, deadline.action)
            except ConnectorError as interrupted:
                raise interrupted from exc
        if timed_out.is_set():
            raise _database_deadline_error(
                deadline.action, deadline.timeout_seconds,
            ) from exc
        raise
    finally:
        # Close while the deadline/cancellation interrupters are still armed;
        # a driver's rollback/close handshake is part of the operation, not
        # an unbounded cleanup epilogue. The outer SQLAlchemy context manager
        # then observes an already-closed connection.
        try:
            close = getattr(conn, "close", None)
            if callable(close):
                close()
        finally:
            stopped.set()
            timer.cancel()
            if watcher is not None:
                watcher.join(timeout=0.2)


def _raise_interrupted_operation(
    state: _OperationInterruptState | None,
    deadline: _DatabaseDeadline,
) -> None:
    """Convert an interruption-caused driver exception to a typed error."""
    if state is not None and state.cancelled.is_set():
        _check_cancellation(deadline.cancellation, deadline.action)
    if state is not None and state.timed_out.is_set():
        raise _database_deadline_error(
            deadline.action, deadline.timeout_seconds,
        )


def _canonicalize_result_columns(frame: Any) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """Give every result column a stable, unique Parquet-safe name."""
    used: set[str] = set()
    names: list[str] = []
    renames: list[dict[str, Any]] = []
    for position, raw in enumerate(frame.columns):
        original = str(raw) if raw is not None and str(raw) else "column"
        candidate = original
        suffix = 2
        while candidate in used:
            candidate = f"{original}__{suffix}"
            suffix += 1
        used.add(candidate)
        names.append(candidate)
        if candidate != original:
            renames.append({
                "position": position,
                "original": original,
                "materialized": candidate,
            })
    if names != list(frame.columns):
        frame = frame.copy(deep=False)
        frame.columns = names
    return frame, tuple(renames)


def _sqlite_connection_uri(connection: str, cwd: Path) -> str:
    """Normalize a promised plain SQLite path into a SQLAlchemy URI."""
    if "://" in connection:
        try:
            from sqlalchemy.engine import make_url

            url = make_url(connection)
            database = url.database
            if (
                database
                and database != ":memory:"
                and not database.startswith("file:")
            ):
                database_path = Path(database).expanduser()
                if not database_path.is_absolute():
                    database_path = cwd / database_path
                url = url.set(database=database_path.resolve().as_posix())
            return url.render_as_string(hide_password=False)
        except Exception as e:
            raise ConnectorError("the SQLite connection URI is invalid") from e
    target = Path(connection).expanduser()
    if not target.is_absolute():
        target = cwd / target
    return f"sqlite:///{target.resolve().as_posix()}"


_DECLARED_SQLALCHEMY_DRIVERS = {
    "postgresql": "psycopg",
    "mysql": "pymysql",
    "mariadb": "pymysql",
    "oracle": "oracledb",
    "redshift": "redshift_connector",
}

# The connector's TLS, timeout, cancellation, and native-type behavior is a
# joint property of a dialect *and its DBAPI*. Accepting an arbitrary
# ``dialect+driver`` silently extends Sift's security claim to code we do not
# package or test. Bare forms are retained only where `_prepare_connection`
# deterministically qualifies them with the declared driver below.
_REVIEWED_SQLALCHEMY_DRIVER_NAMES: Mapping[str, frozenset[str]] = {
    "postgresql": frozenset({"postgres", "postgresql", "postgresql+psycopg"}),
    "mysql": frozenset({"mysql", "mysql+pymysql"}),
    "mariadb": frozenset({"mariadb", "mariadb+pymysql"}),
    "mssql": frozenset({"mssql+pyodbc"}),
    "oracle": frozenset({"oracle", "oracle+oracledb"}),
    "snowflake": frozenset({"snowflake"}),
    "bigquery": frozenset({"bigquery"}),
    "redshift": frozenset({"redshift", "redshift+redshift_connector"}),
    "databricks": frozenset({"databricks"}),
}


def _validate_reviewed_database_driver(
    connection: ConnectionInput,
    backend: str,
) -> None:
    """Fail closed when a URI selects an unreviewed DBAPI implementation."""
    if backend not in _SUPPORTED_REMOTE_BACKENDS:
        return
    try:
        from sqlalchemy.engine import make_url

        selected = make_url(_connection_uri(connection)).drivername.casefold()
    except Exception as exc:
        raise ConnectorError(f"the {backend} connection URI is invalid") from exc
    if selected not in _REVIEWED_SQLALCHEMY_DRIVER_NAMES[backend]:
        expected = sorted(_REVIEWED_SQLALCHEMY_DRIVER_NAMES[backend])[-1]
        raise ConnectorError(
            f"the {selected!r} database driver is not supported for {backend}; "
            f"use Sift's reviewed {expected!r} connection scheme"
        )


def _declared_dbapi_connection_uri(connection: str, backend: str) -> str:
    """Select the declared DBAPI for an unqualified reviewed URI.

    SQLAlchemy's implicit PostgreSQL, MySQL/MariaDB, Oracle, and Redshift dialects map
    to historical DBAPIs that Sift does not declare.  Qualifying a common URI
    with Sift's reviewed extra keeps normal connection strings usable while
    preserving any explicit researcher driver choice.
    """
    try:
        from sqlalchemy.engine import make_url

        url = make_url(connection)
        driver = url.drivername.casefold()
        dialect = driver.split("+", 1)[0]
        if dialect == "postgres":
            # ``postgres://`` is an accepted compatibility alias in Sift's
            # contract but not a SQLAlchemy dialect name.
            driver = "postgresql" + url.drivername[len("postgres"):]
            url = url.set(drivername=driver)
            dialect = "postgresql"
        if dialect == backend and "+" not in driver:
            url = url.set(
                drivername=f"{backend}+{_DECLARED_SQLALCHEMY_DRIVERS[backend]}",
            )
        # SQLAlchemy moves ``ssl_ca`` into PyMySQL's nested ``ssl`` mapping,
        # but leaves ``ssl_verify_cert`` and ``ssl_verify_identity`` at the
        # top level.  PyMySQL treats the latter as a request to rebuild that
        # mapping and, in doing so, loses the CA path.  The resulting client
        # consults the system trust store instead of the researcher-selected
        # CA and rejects otherwise-valid private PKI.  Express the same strict
        # policy in SQLAlchemy's native spelling when an explicit CA is used.
        if (
            backend in {"mysql", "mariadb"}
            and url.drivername.casefold().endswith("+pymysql")
            and url.query.get("ssl_ca")
            and str(url.query.get("ssl_verify_cert", "")).casefold()
            in {"1", "true", "yes"}
            and str(url.query.get("ssl_verify_identity", "")).casefold()
            in {"1", "true", "yes"}
        ):
            url = url.difference_update_query(
                ["ssl_verify_cert", "ssl_verify_identity"]
            ).update_query_dict({"ssl_check_hostname": "true"})
        return url.render_as_string(hide_password=False)
    except Exception as exc:
        raise ConnectorError(f"the {backend} connection URI is invalid") from exc


def _validate_local_database_target(connection: str, backend: str) -> None:
    """Prevent a read-only SQLite open from creating a new database file."""
    if backend != "sqlite":
        return
    try:
        from sqlalchemy.engine import make_url

        url = make_url(connection)
    except Exception as e:
        raise ConnectorError("the SQLite connection URI is invalid") from e
    database = url.database
    if not database or database == ":memory:":
        return
    if database.startswith("file:"):
        mode = str(url.query.get("mode", "")).casefold()
        if mode == "memory" or database.startswith("file::memory:"):
            return
        database = database.removeprefix("file:")
    target = Path(database).expanduser()
    if not target.is_file():
        raise ConnectorError(
            f"SQLite database does not exist: {target}. Sift will not create "
            "a database while opening a read-only extract."
        )


def _parse_odbc_connection_options(value: str) -> dict[str, tuple[str, ...]]:
    """Parse ODBC ``KEY=value`` pairs without splitting braced values.

    ODBC permits semicolons inside ``{...}`` and represents a literal closing
    brace as ``}}``.  A security decision based on a normal ``split(';')``
    can therefore be made about different text than Driver Manager receives.
    Values remain strings and duplicate keys remain visible to the caller.
    """
    options: dict[str, list[str]] = {}
    position = 0
    length = len(value)
    while position < length:
        while position < length and value[position] in "; \t\r\n":
            position += 1
        if position >= length:
            break
        equals = value.find("=", position)
        if equals < 0:
            raise ConnectorError("the ODBC connection string has an invalid option")
        key = value[position:equals].strip().casefold()
        if not key or ";" in key:
            raise ConnectorError("the ODBC connection string has an invalid option")
        position = equals + 1
        if position < length and value[position] == "{":
            position += 1
            chars: list[str] = []
            closed = False
            while position < length:
                char = value[position]
                if char == "}":
                    if position + 1 < length and value[position + 1] == "}":
                        chars.append("}")
                        position += 2
                        continue
                    position += 1
                    closed = True
                    break
                chars.append(char)
                position += 1
            if not closed:
                raise ConnectorError("the ODBC connection string has an unclosed value")
            while position < length and value[position].isspace():
                position += 1
            if position < length and value[position] != ";":
                raise ConnectorError("the ODBC connection string has an invalid value")
            item = "".join(chars)
        else:
            end = value.find(";", position)
            if end < 0:
                end = length
            item = value[position:end].strip()
            position = end
        options.setdefault(key, []).append(item)
        if position < length and value[position] == ";":
            position += 1
    return {key: tuple(items) for key, items in options.items()}


def _single_odbc_option(
    options: Mapping[str, tuple[str, ...]], *names: str,
) -> str | None:
    found: list[str] = []
    for name in names:
        found.extend(options.get(name.casefold(), ()))
    if len(found) > 1:
        raise ConnectorError(
            f"the ODBC connection string repeats the {names[0]!r} option"
        )
    return found[0] if found else None


def _odbc_server_policy_endpoint(value: str) -> str | None:
    """Return a URL-shaped endpoint for an explicit SQL Server target."""
    options = _parse_odbc_connection_options(value)
    server = _single_odbc_option(
        options, "server", "data source", "address", "addr", "network address",
    )
    if not server:
        return None
    target = server.strip()
    if target.casefold().startswith("tcp:"):
        target = target[4:]
    # Named pipes and local shared-memory names are intentionally opaque to a
    # hostname allowlist.  A deployment using them can omit that allowlist.
    if target.casefold().startswith(("np:", "lpc:")):
        return None
    if target.startswith("["):
        close = target.find("]")
        host = target[:close + 1] if close > 0 else ""
    else:
        host = target.split(",", 1)[0].split("\\", 1)[0].strip()
    return f"https://{host}" if host else None


def _database_policy_endpoint(connection: ConnectionInput, backend: str) -> str | None:
    """Expose the actual reviewable endpoint to enterprise host policy."""
    uri = _connection_uri(connection)
    try:
        from sqlalchemy.engine import make_url

        url = make_url(uri)
    except Exception:
        return None
    if backend == "mssql" and url.query.get("odbc_connect"):
        return _odbc_server_policy_endpoint(str(url.query["odbc_connect"]))
    if backend == "bigquery":
        # Project is the URI authority for this dialect, not a network host.
        return "https://bigquery.googleapis.com"
    if backend == "oracle" and (
        not url.port or not str(url.query.get("service_name", "")).strip()
    ):
        # A bare Oracle name is a TNS alias whose real endpoints live in an
        # external file. It cannot honestly satisfy a hostname allowlist.
        return None
    return uri


def validate_connection_security(connection: ConnectionInput, backend: str) -> None:
    """Reject known plaintext/certificate-bypass configurations.

    Cloud drivers have inconsistent defaults.  PostgreSQL's ``prefer`` can
    fall back to plaintext, while SQL Server examples frequently include
    ``TrustServerCertificate=yes``. Sift accepts those only for a literal
    loopback endpoint; remote certificate verification cannot be disabled by
    process environment because that would make the saved profile's security
    dependent on invisible ambient state.
    """
    raw_connection = _validated_connection_text(connection)
    if backend in {"sqlite", "duckdb", "duckdb-file"}:
        return
    _validate_reviewed_database_driver(connection, backend)
    try:
        from sqlalchemy.engine import make_url

        url = make_url(raw_connection)
        host = url.host
        duplicate_keys = {
            str(k).casefold() for k, v in url.query.items()
            if not isinstance(v, str)
        }
        if duplicate_keys:
            raise ConnectorError(
                "database connection options must not be repeated: "
                + ", ".join(sorted(duplicate_keys))
            )
        query = {str(k).casefold(): str(v).casefold() for k, v in url.query.items()}
    except ConnectorError:
        raise
    except Exception as e:
        raise ConnectorError(
            "the remote database connection URI could not be validated for "
            "transport security"
        ) from e

    # Credential documents are never safe in a URI, even for a loopback test
    # endpoint: URIs routinely reach shell history, diagnostics, and process
    # listings. Keep this invariant independent of transport exceptions.
    forbidden = _OPAQUE_OR_OVERRIDE_QUERY_OPTIONS.get(backend, frozenset())
    hidden = forbidden & query.keys()
    if hidden:
        raise ConnectorError(
            f"{backend} connection options must not hide or replace the visible "
            "server: " + ", ".join(sorted(hidden))
        )
    side_effects = (
        _CONNECTION_SIDE_EFFECT_QUERY_OPTIONS.get(backend, frozenset())
        & query.keys()
    )
    if side_effects:
        raise ConnectorError(
            f"{backend} connection options must not execute setup work or turn "
            "a read into a write: " + ", ".join(sorted(side_effects))
        )

    if backend == "bigquery" and any(
        key in query
        for key in ("credentials_base64", "credentials_info", "credentials_path")
    ):
        raise ConnectorError(
            "BigQuery credentials must use Application Default Credentials "
            "or workload identity; embedded credential JSON is not accepted "
            "in a connection URI."
        )
    if backend == "bigquery" and query.get("user_supplied_client") in {
        "1", "true", "yes",
    }:
        raise ConnectorError(
            "BigQuery user_supplied_client is not accepted in a Sift URI; "
            "use Application Default Credentials or workload identity."
        )

    # pyodbc itself is only the Python bridge. Selecting Microsoft's current
    # native driver is required for consistent TLS defaults and cross-platform
    # authentication behavior, including when a developer targets loopback.
    if backend == "mssql":
        driver = query.get("driver", "").strip(" {}").casefold()
        odbc = str(url.query.get("odbc_connect", ""))
        odbc_options = _parse_odbc_connection_options(odbc) if odbc else {}
        odbc_driver = _single_odbc_option(odbc_options, "driver")
        if odbc:
            # ``odbc_connect`` is the exact string Driver Manager receives;
            # ordinary URL parameters are ignored by the dialect in this mode.
            driver_18 = (
                bool(odbc_driver)
                and str(odbc_driver).strip(" {}").casefold()
                == "odbc driver 18 for sql server"
            )
        else:
            driver_18 = driver == "odbc driver 18 for sql server"
        if not driver_18:
            raise ConnectorError(
                "SQL Server connections must explicitly select Microsoft "
                "ODBC Driver 18 for SQL Server."
            )

    if host:
        from sift.integrations import endpoint_is_local

        if endpoint_is_local(f"http://{host}"):
            return

    if backend in {"postgresql", "redshift"}:
        if query.get("sslmode") != "verify-full":
            raise ConnectorError(
                f"{backend} connections must set sslmode=verify-full so both "
                "the certificate chain and server hostname are verified."
            )
        if backend == "redshift":
            ssl_disabled = query.get("ssl") in {"0", "false", "no"}
            idp_insecure = query.get("ssl_insecure") in {"1", "true", "yes"}
            automatic_user_creation = query.get("auto_create") in {
                "1", "true", "yes",
            }
            if ssl_disabled or idp_insecure or automatic_user_creation:
                raise ConnectorError(
                    "Redshift connections must enable SSL and must not disable "
                    "identity-provider certificate verification or create a "
                    "database user during authentication."
                )
    elif backend in {"mysql", "mariadb"}:
        ssl_disabled = query.get("ssl_disabled") in {"1", "true", "yes"}
        local_file_upload = query.get("local_infile") in {"1", "true", "yes"}
        explicit_verification = (
            query.get("ssl_verify_cert") in {"1", "true", "yes"}
            and query.get("ssl_verify_identity") in {"1", "true", "yes"}
        )
        sqlalchemy_verified_ca = (
            bool(query.get("ssl_ca"))
            and query.get("ssl_check_hostname") in {"1", "true", "yes"}
        )
        if ssl_disabled or not (explicit_verification or sqlalchemy_verified_ca):
            raise ConnectorError(
                f"{backend} connections must set ssl_verify_cert=true and "
                "ssl_verify_identity=true, or set ssl_ca together with "
                "ssl_check_hostname=true."
            )
        if local_file_upload:
            raise ConnectorError(
                f"{backend} local_infile is not allowed because a database "
                "server could request a host file during a read-only query."
            )
    elif backend == "mssql":
        # ``odbc_connect`` may contain a percent-decoded semicolon string;
        # checking the decoded SQLAlchemy query values covers both normal
        # URI parameters and that common pyodbc form.
        encrypt = query.get("encrypt") in {"yes", "true", "mandatory", "strict"}
        trust = query.get("trustservercertificate") in {"yes", "true", "1"}
        if odbc_options:
            odbc_encrypt = _single_odbc_option(odbc_options, "encrypt")
            odbc_trust = _single_odbc_option(
                odbc_options, "trustservercertificate",
            )
            encrypt = str(odbc_encrypt or "").casefold() in {
                "yes", "true", "mandatory", "strict",
            }
            trust = str(odbc_trust or "").casefold() in {"yes", "true", "1"}
        if not encrypt or trust:
            raise ConnectorError(
                "SQL Server connections must set Encrypt=yes (or strict) "
                "and TrustServerCertificate=no."
            )
    elif backend == "oracle":
        protocol = query.get("protocol", "")
        dn_match = query.get("ssl_server_dn_match") in {"1", "true", "yes", "on"}
        explicit_service = bool(url.port and query.get("service_name", "").strip())
        if protocol != "tcps" or not dn_match or not explicit_service:
            raise ConnectorError(
                "Oracle connections must use an explicit host, port, and "
                "service_name with protocol=tcps and ssl_server_dn_match=true "
                "so a TNS alias cannot hide the target or bypass verification."
            )
    elif backend == "snowflake":
        insecure = query.get("insecure_mode") in {"1", "true", "yes"}
        fail_open = query.get("ocsp_fail_open", "true") not in {"0", "false", "no"}
        if insecure or fail_open:
            raise ConnectorError(
                "Snowflake connections must set ocsp_fail_open=false, and "
                "insecure_mode is not allowed. This makes certificate "
                "revocation checks fail closed."
            )
    elif backend == "databricks":
        tls_disabled = query.get("_enable_ssl") in {"0", "false", "no"}
        no_verify = query.get("_tls_no_verify") in {"1", "true", "yes"}
        no_hostname = query.get("_tls_verify_hostname") in {"0", "false", "no"}
        http = query.get("http_scheme") == "http"
        if tls_disabled or no_verify or no_hostname or http:
            raise ConnectorError(
                "Databricks connections must use HTTPS with certificate and "
                "hostname verification enabled."
            )


def _prepare_connection(
    cwd: Path,
    connection: ConnectionInput,
) -> tuple[str, ConnectionInput]:
    """Validate one connection and return ``(backend, effective_value)``.

    Plain local paths are resolved against the active session rather than the
    process launch directory.  That keeps behaviour identical on macOS,
    Windows, and Linux and fixes a particularly confusing relative DuckDB
    path failure in packaged applications.
    """
    raw_value = _connection_uri(connection)
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ConnectorError("no connection given")
    raw = raw_value.strip()
    backend = describe_backend(raw)
    effective: ConnectionInput
    if backend == "sqlite":
        effective = _sqlite_connection_uri(raw, cwd)
    elif backend in _DECLARED_SQLALCHEMY_DRIVERS:
        effective = _declared_dbapi_connection_uri(raw, backend)
    elif backend in {"duckdb", "duckdb-file"} and "://" not in raw:
        target = Path(raw).expanduser()
        if not target.is_absolute():
            target = Path(cwd) / target
        effective = str(target.resolve())
    else:
        effective = raw

    if isinstance(connection, ConnectionSpec):
        effective = replace(connection, uri=str(effective))

    _validate_local_database_target(_connection_uri(effective), backend)
    if (
        backend in {"duckdb", "duckdb-file"}
        and not Path(_connection_uri(effective)).is_file()
    ):
        raise ConnectorError(
            "local data source does not exist: "
            f"{redact_connection(_connection_uri(effective))}"
        )
    validate_connection_security(effective, backend)

    from sift import enterprise_policy

    ent = enterprise_policy.load_enterprise_policy()
    if not enterprise_policy.database_backend_allowed(backend, ent):
        raise ConnectorError(
            f"database backend {backend!r} is blocked by enterprise policy"
        )
    if not enterprise_policy.integration_endpoint_allowed(
        _database_policy_endpoint(effective, backend),
        ent,
        local_hint=backend in {"sqlite", "duckdb", "duckdb-file"},
    ):
        raise ConnectorError("database endpoint is blocked by enterprise policy")
    return backend, effective


def _register_duckdb_source(con: Any, path: str) -> None:
    """Expose a researcher-selected data file as the read-only ``source`` view."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".parquet":
            relation = con.read_parquet(path)
        elif suffix in {".csv", ".tsv"}:
            relation = con.read_csv(path, delimiter="\t" if suffix == ".tsv" else ",")
        elif suffix in {".json", ".jsonl"}:
            relation = con.read_json(path)
        else:  # pragma: no cover - describe_backend is the outer allowlist
            raise ConnectorError(f"unsupported DuckDB file source: {suffix}")
        relation.create_view("source", replace=True)
    except ConnectorError:
        raise
    except Exception as e:
        raise ConnectorError(
            f"could not inspect the local data source ({type(e).__name__})"
        ) from e


def _open_duckdb(connection: str, backend: str) -> Any:
    try:
        import duckdb
    except ImportError as e:  # pragma: no cover - dependency is declared
        raise ConnectorError("duckdb is not installed in this environment") from e
    target = ":memory:" if backend == "duckdb-file" else connection
    try:
        con = duckdb.connect(target, read_only=(backend == "duckdb"))
        if backend == "duckdb-file":
            _register_duckdb_source(con, connection)
        return con
    except ConnectorError:
        raise
    except Exception as e:
        raise ConnectorError(
            f"could not open {redact_connection(str(target))}: {type(e).__name__}"
        ) from e


def _read_only_enforcement(dialect: str) -> str:
    return (
        "database_session_and_query_gate"
        if (dialect or "").lower() in {
            "sqlite", "postgresql", "redshift", "oracle", "mysql", "mariadb",
        }
        else "query_gate_only_use_select_only_principal"
    )


def check_connection(
    cwd: Path,
    *,
    connection: ConnectionInput,
    cancellation: CancellationToken | None = None,
) -> ConnectionCheck:
    """Verify connectivity without reading a table or returning any values."""
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise ConnectorError("no active session directory")
    _check_cancellation(cancellation, "connection test")
    backend, effective = _prepare_connection(cwd, connection)
    deadline = _DatabaseDeadline(cancellation, "connection test")
    started = time.monotonic()
    version: str | None = None
    enforcement = "read_only_file_and_query_gate"
    if backend in {"duckdb", "duckdb-file"}:
        con = _open_duckdb(_connection_uri(effective), backend)
        try:
            con.execute("SELECT 1").fetchone()
            version_row = con.execute("SELECT version()").fetchone()
            if version_row:
                version = str(version_row[0])[:120]
        finally:
            con.close()
    else:
        try:
            from sqlalchemy import text
        except ImportError as e:  # pragma: no cover
            raise ConnectorError("sqlalchemy is not installed") from e
        connect_timeout = _native_connect_timeout_seconds(deadline.remaining)
        engine = _create_bounded_engine(
            effective, backend, timeout_cap_seconds=deadline.remaining,
        )
        interrupt_state: _OperationInterruptState | None = None
        connected = False
        try:
            deadline.check()
            with engine.connect() as conn:
                connected = True
                deadline.check()
                _configure_sdk_request_deadline(
                    conn, engine.dialect.name, deadline.remaining,
                )
                with _interrupt_connected_operation(
                    conn, deadline=deadline,
                ) as interrupt_state:
                    dialect = engine.dialect.name
                    _configure_connection_read_only(conn, dialect)
                    probe = (
                        "SELECT 1 FROM DUAL" if dialect == "oracle" else "SELECT 1"
                    )
                    conn.execute(text(probe)).fetchone()
                    raw_version = getattr(
                        engine.dialect, "server_version_info", None,
                    )
                    if raw_version:
                        version = ".".join(str(v) for v in raw_version)[:120]
                    enforcement = _read_only_enforcement(dialect)
        except ConnectorError:
            raise
        except Exception as e:
            _raise_interrupted_operation(interrupt_state, deadline)
            deadline.check()
            if not connected and _driver_error_is_timeout(e, effective):
                raise _connect_deadline_error(
                    "connection test", connect_timeout,
                ) from e
            raise _driver_connector_error(
                "connection check failed", e, effective,
            ) from e
        finally:
            engine.dispose()
    return ConnectionCheck(
        backend=backend,
        connection_display=redact_connection(_connection_uri(connection)),
        latency_ms=max(0, round((time.monotonic() - started) * 1000)),
        server_version=version,
        read_only_enforcement=enforcement,
    )


def _catalog_name(value: str | None, *, label: str) -> str | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str) or len(value) > 255 or any(
        ord(char) < 32 for char in value
    ):
        raise ConnectorError(f"invalid {label} name")
    return value


def _duckdb_catalog(
    connection: str,
    backend: str,
    *,
    schema: str | None,
    object_name: str | None,
) -> tuple[str | None, list[str], list[dict[str, Any]], list[str]]:
    con = _open_duckdb(connection, backend)
    warnings: list[str] = []
    try:
        if backend == "duckdb-file":
            if object_name not in {None, "source"}:
                raise ConnectorError(f"no database object named {object_name!r}")
            source_columns: list[dict[str, Any]] = []
            if object_name in {None, "source"} and object_name is not None:
                described = con.execute("DESCRIBE source").fetchmany(
                    MAX_CATALOG_COLUMNS + 1,
                )
                for row in described[:MAX_CATALOG_COLUMNS]:
                    source_columns.append({
                        "name": str(row[0]),
                        "type": str(row[1]),
                        "nullable": str(row[2]).casefold() != "no",
                    })
                if len(described) > MAX_CATALOG_COLUMNS:
                    warnings.append("column list truncated")
            return "main", ["main"], [{
                "schema": "main", "name": "source", "kind": "view",
                "columns": source_columns,
            }], warnings

        default_row = con.execute("SELECT current_schema()").fetchone()
        default_schema = str(default_row[0]) if default_row and default_row[0] else None
        schemas = [str(row[0]) for row in con.execute(
            "SELECT schema_name FROM information_schema.schemata "
            "ORDER BY schema_name LIMIT ?",
            [MAX_CATALOG_SCHEMAS + 1],
        ).fetchall()]
        selected = schema or default_schema
        if object_name is None:
            rows = con.execute(
                "SELECT table_schema, table_name, table_type "
                "FROM information_schema.tables "
                "WHERE (? IS NULL OR table_schema = ?) "
                "ORDER BY table_schema, table_name LIMIT ?",
                [selected, selected, MAX_CATALOG_OBJECTS + 1],
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT table_schema, table_name, table_type "
                "FROM information_schema.tables "
                "WHERE (? IS NULL OR table_schema = ?) AND table_name = ? "
                "ORDER BY table_schema, table_name LIMIT 2",
                [selected, selected, object_name],
            ).fetchall()
            if not rows:
                raise ConnectorError(f"no database object named {object_name!r}")
        objects: list[dict[str, Any]] = []
        for obj_schema, name, kind in rows:
            columns: list[dict[str, Any]] = []
            if object_name is not None and str(name) == object_name:
                col_rows = con.execute(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = ? AND table_name = ? "
                    "ORDER BY ordinal_position LIMIT ?",
                    [str(obj_schema), str(name), MAX_CATALOG_COLUMNS + 1],
                ).fetchall()
                columns = [{
                    "name": str(c[0]), "type": str(c[1]),
                    "nullable": str(c[2]).casefold() == "yes",
                } for c in col_rows[:MAX_CATALOG_COLUMNS]]
                if len(col_rows) > MAX_CATALOG_COLUMNS:
                    warnings.append("column list truncated")
            objects.append({
                "schema": str(obj_schema), "name": str(name),
                "kind": "view" if "VIEW" in str(kind).upper() else "table",
                "columns": columns,
            })
        return default_schema, schemas, objects, warnings
    except ConnectorError:
        raise
    except Exception as e:
        raise ConnectorError(
            "database catalog inspection failed: "
            f"{_safe_connector_error(e, connection)}"
        ) from e
    finally:
        con.close()


def _bounded_remote_catalog(
    conn: Any,
    backend: str,
    *,
    default_schema: str | None,
    selected_schema: str | None,
) -> tuple[list[str], list[tuple[str, str, str]], list[str]]:
    """Read a shallow remote catalog with limits enforced by the server.

    SQLAlchemy's reflection helpers return complete Python lists. Applying a
    slice afterwards protects the bridge response but does not protect the
    database, driver, or desktop process from a very large institutional
    catalog. These deliberately small, code-owned metadata queries put the
    ceiling in the statement executed by the provider.
    """
    from sqlalchemy import text

    warnings: list[str] = []
    limit = MAX_CATALOG_OBJECTS + 1
    schema_limit = MAX_CATALOG_SCHEMAS + 1

    # BigQuery's INFORMATION_SCHEMA namespace requires a dataset or region
    # identifier in the SQL text, and identifiers cannot be bound safely.
    # Do not interpolate researcher input or fall back to the unbounded SDK
    # list operation. A detailed object request still uses targeted
    # reflection in ``inspect_database`` below.
    if backend == "bigquery":
        bigquery_schemas = [selected_schema or default_schema] if (
            selected_schema or default_schema
        ) else []
        warnings.append(
            "broad BigQuery catalog browsing is disabled; select a dataset "
            "and object for bounded metadata discovery"
        )
        return [value for value in bigquery_schemas if value], [], warnings

    if backend == "sqlite":
        selected = selected_schema or default_schema or "main"
        # SQLite exposes attached databases and objects through bounded local
        # pragmas/catalog tables rather than INFORMATION_SCHEMA.
        sqlite_schemas = [
            str(row[1]) for row in conn.exec_driver_sql(
                "PRAGMA database_list"
            ).fetchmany(schema_limit)
        ]
        rows = conn.execute(
            text(
                "SELECT :selected AS table_schema, name, type "
                "FROM sqlite_master WHERE type IN ('table', 'view') "
                "ORDER BY name, type LIMIT :limit"
            ),
            {"selected": selected, "limit": limit},
        ).fetchall()
        return sqlite_schemas, [
            (str(row[0]), str(row[1]), str(row[2])) for row in rows
        ], warnings

    if backend == "oracle":
        schema_sql = (
            "SELECT username FROM (SELECT username FROM all_users "
            "ORDER BY username) WHERE ROWNUM <= :limit"
        )
        object_sql = (
            "SELECT owner, object_name, object_type FROM ("
            "SELECT owner, object_name, object_type FROM all_objects "
            "WHERE object_type IN ('TABLE', 'VIEW', 'MATERIALIZED VIEW') "
            "AND (:selected IS NULL OR owner = :selected) "
            "ORDER BY owner, object_name, object_type) "
            "WHERE ROWNUM <= :limit"
        )
    elif backend == "mssql":
        schema_sql = (
            "SELECT TOP (:limit) schema_name FROM information_schema.schemata "
            "ORDER BY schema_name"
        )
        object_sql = (
            "SELECT TOP (:limit) table_schema, table_name, table_type "
            "FROM information_schema.tables "
            "WHERE (:selected IS NULL OR table_schema = :selected) "
            "ORDER BY table_schema, table_name, table_type"
        )
    else:
        schema_sql = (
            "SELECT schema_name FROM information_schema.schemata "
            "ORDER BY schema_name LIMIT :limit"
        )
        object_sql = (
            "SELECT table_schema, table_name, table_type "
            "FROM information_schema.tables "
            "WHERE (:selected IS NULL OR table_schema = :selected) "
            "ORDER BY table_schema, table_name, table_type LIMIT :limit"
        )

    remote_schemas = [
        str(row[0]) for row in conn.execute(
            text(schema_sql), {"limit": schema_limit},
        ).fetchall()
    ]
    rows = conn.execute(
        text(object_sql), {"selected": selected_schema, "limit": limit},
    ).fetchall()
    objects: list[tuple[str, str, str]] = []
    for row in rows:
        raw_kind = str(row[2]).casefold().replace(" ", "_")
        if "materialized" in raw_kind:
            kind = "materialized_view"
        elif "view" in raw_kind:
            kind = "view"
        else:
            kind = "table"
        objects.append((str(row[0]), str(row[1]), kind))
    return remote_schemas, objects, warnings


def inspect_database(
    cwd: Path,
    *,
    connection: ConnectionInput,
    schema: str | None = None,
    object_name: str | None = None,
    cancellation: CancellationToken | None = None,
) -> DatabaseCatalog:
    """Return bounded schemas/objects and optional columns, never row data.

    Pass ``object_name`` to fetch columns for one object. Without it the
    response is intentionally shallow, which keeps very large warehouses
    fast and prevents a single click from copying their entire dictionary.
    """
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise ConnectorError("no active session directory")
    _check_cancellation(cancellation, "catalog discovery")
    schema = _catalog_name(schema, label="schema")
    object_name = _catalog_name(object_name, label="object")
    backend, effective = _prepare_connection(cwd, connection)
    deadline = _DatabaseDeadline(cancellation, "catalog discovery")
    warnings: list[str] = []

    if backend in {"duckdb", "duckdb-file"}:
        default_schema, schemas, objects, warnings = _duckdb_catalog(
            _connection_uri(effective), backend,
            schema=schema, object_name=object_name,
        )
    else:
        listener: Any | None = None
        conn: Any | None = None
        try:
            from sqlalchemy import event, inspect
        except ImportError as e:  # pragma: no cover
            raise ConnectorError("sqlalchemy is not installed") from e
        connect_timeout = _native_connect_timeout_seconds(deadline.remaining)
        engine = _create_bounded_engine(
            effective, backend, timeout_cap_seconds=deadline.remaining,
        )
        interrupt_state: _OperationInterruptState | None = None
        connected = False
        try:
            deadline.check()
            with engine.connect() as conn:
                connected = True
                deadline.check()
                _configure_sdk_request_deadline(
                    conn, engine.dialect.name, deadline.remaining,
                )
                with _interrupt_connected_operation(
                    conn, deadline=deadline,
                ) as interrupt_state:
                    def capture_cursor(
                        _conn: Any,
                        cursor: Any,
                        _statement: Any,
                        _parameters: Any,
                        _context: Any,
                        _executemany: Any,
                    ) -> None:
                        if interrupt_state is not None:
                            interrupt_state.bind_result(cursor)

                    listener = capture_cursor
                    event.listen(conn, "before_cursor_execute", listener)
                    _configure_connection_read_only(conn, engine.dialect.name)
                    inspector = inspect(conn)
                    default_schema = inspector.default_schema_name
                    selected = schema or default_schema
                    if object_name is not None:
                        # A selected-object request must not enumerate every
                        # object merely to prove one name exists. ``has_table``
                        # and ``get_columns`` are targeted provider operations.
                        schemas = [str(selected)] if selected is not None else []
                        if not inspector.has_table(object_name, schema=selected):
                            raise ConnectorError(
                                f"no database object named {object_name!r}"
                            )
                        reflected = inspector.get_columns(
                            object_name, schema=selected,
                        )
                        deadline.check()
                        columns = [{
                            "name": str(c.get("name", "")),
                            "type": str(c.get("type", "unknown")),
                            "nullable": bool(c.get("nullable", True)),
                        } for c in reflected[:MAX_CATALOG_COLUMNS]]
                        if len(reflected) > MAX_CATALOG_COLUMNS:
                            warnings.append("column list truncated")
                        objects = [{
                            "schema": (
                                str(selected) if selected is not None else None
                            ),
                            "name": object_name,
                            "kind": "object",
                            "columns": columns,
                        }]
                    else:
                        schemas, combined, catalog_warnings = (
                            _bounded_remote_catalog(
                                conn,
                                backend,
                                default_schema=(
                                    str(default_schema)
                                    if default_schema is not None else None
                                ),
                                selected_schema=(
                                    str(selected) if selected is not None else None
                                ),
                            )
                        )
                        warnings.extend(catalog_warnings)
                        deadline.check()
                        objects = [{
                            "schema": obj_schema,
                            "name": name,
                            "kind": kind,
                            "columns": [],
                        } for obj_schema, name, kind in combined]
        except ConnectorError:
            raise
        except Exception as e:
            _raise_interrupted_operation(interrupt_state, deadline)
            deadline.check()
            if not connected and _driver_error_is_timeout(e, effective):
                raise _connect_deadline_error(
                    "catalog discovery", connect_timeout,
                ) from e
            raise _driver_connector_error(
                "database catalog inspection failed", e, effective,
            ) from e
        finally:
            if listener is not None and conn is not None:
                try:
                    event.remove(conn, "before_cursor_execute", listener)
                except Exception:  # noqa: BLE001 - connection may be closed
                    pass
            engine.dispose()

    schemas_truncated = len(schemas) > MAX_CATALOG_SCHEMAS
    objects_truncated = len(objects) > MAX_CATALOG_OBJECTS
    return DatabaseCatalog(
        backend=backend,
        connection_display=redact_connection(_connection_uri(connection)),
        default_schema=default_schema,
        schemas=tuple(schemas[:MAX_CATALOG_SCHEMAS]),
        objects=tuple(objects[:MAX_CATALOG_OBJECTS]),
        schemas_truncated=schemas_truncated,
        objects_truncated=objects_truncated,
        warnings=tuple(warnings),
    )


_METERED_WAREHOUSES = frozenset({
    "snowflake", "bigquery", "redshift", "databricks",
})


def _interrupt_driver_operation(result: Any, raw: Any) -> str:
    """Best-effort cross-driver interruption, ending with connection close.

    Returns a value-free mechanism label suitable for live-test attestation.
    """
    bigquery_client = getattr(raw, "_client", None)
    cancel_bigquery = getattr(bigquery_client, "cancel_active_job", None)
    if callable(cancel_bigquery) and cancel_bigquery():
        return "bigquery_job_cancel"

    cursor = getattr(result, "cursor", result)
    targets = [cursor, raw]
    for target in targets:
        if target is None:
            continue
        sfqid = getattr(target, "sfqid", None)
        abort_query = getattr(target, "abort_query", None)
        if sfqid and callable(abort_query):
            try:
                if abort_query(sfqid):
                    return "snowflake_query_abort"
            except Exception:  # noqa: BLE001, S110
                pass
        for method in ("cancel", "interrupt"):
            operation = getattr(target, method, None)
            if callable(operation):
                try:
                    operation()
                    return f"driver_{method}"
                except Exception:  # noqa: BLE001, S110
                    pass
    # Drivers without a cancellation method must not be allowed to hold the
    # host forever. Closing an in-flight connection is intentionally the final
    # fallback; the connection is disposed and never returned to the pool.
    close = getattr(raw, "close", None)
    if callable(close):
        try:
            close()
            return "connection_close"
        except Exception:  # noqa: BLE001, S110
            pass
    return "unavailable"


class _BoundedBigQueryClient:
    """Delegate BigQuery calls with finite HTTP/wait budgets and no retries."""

    _STANDARD_TIMEOUT_METHODS = frozenset({
        "query", "list_datasets", "list_tables", "get_table",
    })

    def __init__(self, client: Any, seconds: float) -> None:
        self._client = client
        self._seconds = max(0.001, float(seconds))
        self._active_job: Any | None = None
        self._active_lock = threading.Lock()

    def cancel_active_job(self) -> bool:
        """Request provider-side cancellation for the current BigQuery job."""
        with self._active_lock:
            job = self._active_job
        if job is None:
            return False
        try:
            return bool(job.cancel(retry=None, timeout=min(10.0, self._seconds)))
        except Exception:  # noqa: BLE001 - interruption remains best effort
            return False

    def __getattr__(self, name: str) -> Any:
        if name == "query_and_wait":
            def query_and_wait(*args: Any, **kwargs: Any) -> Any:
                # The SDK convenience method hides its QueryJob until waiting
                # is over, which makes real cancellation impossible. Start the
                # same job explicitly, retain the handle while waiting, and
                # use the same finite/no-retry policy for both HTTP phases.
                query = args[0] if args else kwargs.pop("query")
                page_size = kwargs.pop("page_size", None)
                max_results = kwargs.pop("max_results", None)
                api_timeout = self._bounded(kwargs.pop("api_timeout", None))
                wait_timeout = self._bounded(kwargs.pop("wait_timeout", None))
                kwargs.pop("retry", None)
                kwargs.pop("job_retry", None)
                job = self._client.query(
                    query,
                    timeout=api_timeout,
                    retry=None,
                    job_retry=None,
                    **kwargs,
                )
                with self._active_lock:
                    self._active_job = job
                try:
                    return job.result(
                        page_size=page_size,
                        max_results=max_results,
                        timeout=wait_timeout,
                        retry=None,
                        job_retry=None,
                    )
                finally:
                    with self._active_lock:
                        self._active_job = None

            return query_and_wait
        target = getattr(self._client, name)
        if not callable(target):
            return target
        if name in self._STANDARD_TIMEOUT_METHODS:
            def bounded(*args: Any, **kwargs: Any) -> Any:
                kwargs["timeout"] = self._bounded(kwargs.get("timeout"))
                kwargs["retry"] = None
                if name == "query":
                    kwargs["job_retry"] = None
                return target(*args, **kwargs)

            return bounded
        return target

    def _bounded(self, supplied: Any) -> float:
        try:
            value = float(supplied)
        except (TypeError, ValueError):
            return self._seconds
        return max(0.001, min(value, self._seconds))


def _configure_sdk_request_deadline(
    conn: Any,
    dialect: str,
    seconds: float,
) -> None:
    """Bound SDK calls whose SQLAlchemy dialect has no DBAPI socket option."""
    if (dialect or "").casefold() != "bigquery":
        return
    wrapper = getattr(conn, "connection", conn)
    raw = getattr(wrapper, "driver_connection", wrapper)
    client = getattr(raw, "_client", None)
    if client is None:
        raise ConnectorError(
            "could not establish a bounded BigQuery client request deadline"
        )
    if not isinstance(client, _BoundedBigQueryClient):
        raw._client = _BoundedBigQueryClient(client, seconds)  # noqa: SLF001
    # The BigQuery Storage client uses a separate RPC stack. The bounded REST
    # row iterator is retained so extraction cannot silently switch to an
    # independently configured, potentially unbounded transport.
    if hasattr(raw, "_bqstorage_client"):
        storage = raw._bqstorage_client  # noqa: SLF001
        close_storage = getattr(storage, "close", None)
        if callable(close_storage):
            close_storage()
        raw._bqstorage_client = None  # noqa: SLF001
        raw._owns_bqstorage_client = False  # noqa: SLF001


def _sum_json_key(value: Any, key: str) -> int:
    total = 0
    if isinstance(value, dict):
        for item_key, item_value in value.items():
            if str(item_key).casefold() == key.casefold():
                try:
                    total += max(0, int(item_value))
                except (TypeError, ValueError):
                    pass
            else:
                total += _sum_json_key(item_value, key)
    elif isinstance(value, list):
        total += sum(_sum_json_key(item, key) for item in value)
    return total


def _bigquery_dry_run(
    connection: ConnectionInput,
    sql: str,
    cancellation: CancellationToken | None = None,
    deadline: _DatabaseDeadline | None = None,
) -> tuple[int | None, int | None]:
    """Compile a BigQuery query with ``dry_run=True`` and no cache use."""
    # Validate the researcher-controlled connection options before checking
    # the optional SDK.  Duplicate security-sensitive parameters are invalid
    # regardless of local driver readiness, and surfacing "driver missing"
    # first would hide a malformed connection until deployment.
    from sqlalchemy.engine import make_url

    url = make_url(_connection_uri(connection))
    project = url.host or url.database or None
    billing_project = url.query.get("billing_project_id")
    location = url.query.get("location")
    if billing_project is not None and not isinstance(billing_project, str):
        raise ConnectorError("BigQuery billing_project_id must be specified once")
    if location is not None and not isinstance(location, str):
        raise ConnectorError("BigQuery location must be specified once")
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise ConnectorError(
            "BigQuery preview requires Sift's bigquery integration extra"
        ) from e
    operation_deadline = deadline or _DatabaseDeadline(
        cancellation, "query preview",
    )
    try:
        operation_deadline.check()
        client = bigquery.Client(project=str(billing_project or project or "") or None)
        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        # The BigQuery SDK performs the dry-run request synchronously. Pass
        # the remaining outer budget to its HTTP layer and disable retries so
        # the SDK cannot outlive that budget after an ambiguous network loss.
        # The runtime API documents None as the retry-disable value even though
        # its current annotation omits Optional.
        job = client.query(
            sql,
            job_config=config,
            location=location,
            timeout=max(0.001, operation_deadline.remaining),
            retry=None,  # type: ignore[arg-type]
        )
        operation_deadline.check()
        estimated = getattr(job, "total_bytes_processed", None)
        return (int(estimated) if estimated is not None else None, None)
    except Exception as e:
        operation_deadline.check()
        if _driver_error_is_timeout(e, connection):
            raise _database_deadline_error(
                "query preview", operation_deadline.timeout_seconds,
            ) from e
        raise ConnectorError(
            "BigQuery dry run failed: " + _safe_connector_error(e, connection)
        ) from e
    finally:
        close = getattr(locals().get("client"), "close", None)
        if callable(close):
            close()


def _snowflake_dry_run(
    connection: ConnectionInput,
    sql: str,
    cancellation: CancellationToken | None = None,
    deadline: _DatabaseDeadline | None = None,
) -> tuple[int | None, int | None]:
    """Compile a Snowflake plan without starting a warehouse or query."""
    try:
        import sqlalchemy  # noqa: F401 - dependency readiness check
    except ImportError as e:  # pragma: no cover
        raise ConnectorError("sqlalchemy is not installed") from e
    operation_deadline = deadline or _DatabaseDeadline(
        cancellation, "query preview",
    )
    connect_timeout = _native_connect_timeout_seconds(
        operation_deadline.remaining,
    )
    engine = _create_bounded_engine(
        connection,
        "snowflake",
        timeout_cap_seconds=operation_deadline.remaining,
    )
    interrupt_state: _OperationInterruptState | None = None
    connected = False
    try:
        operation_deadline.check()
        with engine.connect() as conn:
            connected = True
            operation_deadline.check()
            _configure_sdk_request_deadline(
                conn, engine.dialect.name, operation_deadline.remaining,
            )
            with _interrupt_connected_operation(
                conn, deadline=operation_deadline,
            ) as interrupt_state:
                cleanup_timeout = _configure_query_timeout(
                    conn,
                    engine.dialect.name,
                    max(1, int(operation_deadline.remaining)),
                )
                try:
                    # ``sql`` is the normalized single read-only statement.
                    # Driver SQL avoids SQLAlchemy treating vendor colon
                    # syntax or JSON text as application bind parameters.
                    result = conn.exec_driver_sql(f"EXPLAIN USING JSON {sql}")
                    try:
                        raw = result.scalar()
                    finally:
                        result.close()
                finally:
                    cleanup_timeout()
        plan = json.loads(str(raw))
        estimated = _sum_json_key(plan, "bytesAssigned")
        rows = _sum_json_key(plan, "rows")
        return (estimated or None, rows or None)
    except ConnectorError:
        raise
    except Exception as e:
        _raise_interrupted_operation(interrupt_state, operation_deadline)
        operation_deadline.check()
        if not connected and _driver_error_is_timeout(e, connection):
            raise _connect_deadline_error(
                "query preview", connect_timeout,
            ) from e
        raise ConnectorError(
            "Snowflake query preview failed: "
            + _safe_connector_error(e, connection)
        ) from e
    finally:
        engine.dispose()


def preview_query(
    cwd: Path,
    *,
    connection: ConnectionInput,
    sql: str,
    cancellation: CancellationToken | None = None,
) -> QueryPreview:
    """Validate a query and return safe metadata before materialization.

    Only documented compile/dry-run mechanisms are used.  In particular,
    this function never issues ``EXPLAIN ANALYZE`` and never fetches sample
    rows. BigQuery and Snowflake expose byte estimates; other adapters return
    an honest "estimate unavailable" result instead of executing the query.
    """
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise ConnectorError("no active session directory")
    _check_cancellation(cancellation, "query preview")
    backend, effective = _prepare_connection(cwd, connection)
    deadline = _DatabaseDeadline(cancellation, "query preview")
    normalized = normalize_sql(sql, backend=backend)
    if normalized is None:
        raise ConnectorError("query preview accepts one read-only query only")
    if re.search(r"(?i)\bexplain\s+(?:\([^)]*\)\s*)?analyze\b", normalized):
        raise ConnectorError("EXPLAIN ANALYZE is never allowed as a preview")

    estimated_bytes: int | None = None
    estimated_rows: int | None = None
    source: str | None = None
    if backend == "bigquery":
        estimated_bytes, estimated_rows = _bigquery_dry_run(
            effective, normalized, cancellation, deadline,
        )
        source = "bigquery_dry_run"
    elif backend == "snowflake":
        estimated_bytes, estimated_rows = _snowflake_dry_run(
            effective, normalized, cancellation, deadline,
        )
        source = "snowflake_explain_upper_bound"

    metered = backend in _METERED_WAREHOUSES
    warnings: list[str] = []
    if metered and estimated_bytes is None:
        warnings.append(
            "This warehouse can incur usage charges; Sift could not obtain a "
            "reliable byte estimate without executing the query."
        )
    elif estimated_bytes is not None and estimated_bytes >= database_cost_warning_bytes():
        warnings.append(
            f"The provider estimates this query may scan {estimated_bytes:,} "
            "bytes, above Sift's configured cost-warning threshold."
        )
    return QueryPreview(
        backend=backend,
        connection_display=redact_connection(_connection_uri(connection)),
        query_sha256=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        read_only_enforcement=_read_only_enforcement(backend),
        dry_run_supported=backend in {"bigquery", "snowflake"},
        estimate_source=source,
        estimated_bytes=estimated_bytes,
        estimated_rows=estimated_rows,
        metered_warehouse=metered,
        warnings=tuple(warnings),
    )


def _run_duckdb(
    connection: str,
    sql: str,
    backend: str,
    limit: int,
    cancellation: CancellationToken | None = None,
):
    con = _open_duckdb(connection, backend)
    timed_out = threading.Event()
    stopped = threading.Event()
    cancelled = threading.Event()

    def interrupt() -> None:
        timed_out.set()
        try:
            con.interrupt()
        except Exception:  # noqa: BLE001, S110 - connection may be closing
            pass

    timer = threading.Timer(database_query_timeout_seconds(), interrupt)
    timer.daemon = True
    timer.start()
    watcher: threading.Thread | None = None
    if cancellation is not None:
        def watch_cancellation() -> None:
            while not stopped.is_set():
                if cancellation.wait(0.1):
                    cancelled.set()
                    try:
                        con.interrupt()
                    except Exception:  # noqa: BLE001, S110
                        pass
                    return

        watcher = threading.Thread(target=watch_cancellation, daemon=True)
        watcher.start()
    try:
        # ``sql`` is the output of normalize_sql's lexical + SQLGlot
        # single-read-only-statement gate, and ``limit`` is a clamped integer.
        # DuckDB cannot parameterize a subquery expression; interpolation is
        # necessary here and is safe only because the caller enforces both
        # invariants immediately before dispatch.
        cursor = con.execute(
            f"SELECT * FROM ({sql}) LIMIT {limit + 1}"  # nosec B608
        )
        yielded = False
        while True:
            # A one-vector chunk is small enough to stop wide values before
            # the whole answer is resident in memory.
            frame = cursor.fetch_df_chunk(1)
            if frame.empty:
                if not yielded:
                    yield frame
                break
            yielded = True
            yield frame
    except ConnectorError:
        raise
    except Exception as e:
        if cancelled.is_set():
            raise ConnectorError(
                "database extraction cancelled",
                code="cancelled",
                action="Start a new extraction when ready.",
            ) from e
        if timed_out.is_set():
            raise ConnectorError(
                "the query exceeded Sift's database timeout; narrow or "
                "aggregate it, or raise SIFT_DATABASE_QUERY_TIMEOUT_SECONDS "
                "for an approved long-running job"
            ) from e
        raise _driver_connector_error("the query failed", e, connection) from e
    finally:
        stopped.set()
        timer.cancel()
        con.close()
        if watcher is not None:
            watcher.join(timeout=0.2)


def _run_sqlalchemy(
    connection: ConnectionInput,
    sql: str,
    backend: str,
    limit: int,
    cancellation: CancellationToken | None = None,
    transport_probe: Callable[[QueryTransportEvidence], None] | None = None,
):
    try:
        import pandas as pd
        import sqlalchemy  # noqa: F401 - dependency readiness check
    except ImportError as e:  # pragma: no cover
        raise ConnectorError("sqlalchemy is not installed") from e
    deadline = _DatabaseDeadline(cancellation, "extraction")
    connect_timeout = _native_connect_timeout_seconds(deadline.remaining)
    engine = _create_bounded_engine(
        connection, backend, timeout_cap_seconds=deadline.remaining,
    )
    connected = False
    interrupt_state: _OperationInterruptState | None = None
    try:
        deadline.check()
        with engine.connect() as conn:
            connected = True
            # Native connect/login timeouts bound the synchronous call above.
            # Cancellation is observed here before Sift can configure or run
            # any statement; no abandoned worker retains the connection.
            deadline.check()
            with _interrupt_connected_operation(
                conn, deadline=deadline,
            ) as interrupt_state:
                _configure_sdk_request_deadline(
                    conn, engine.dialect.name, deadline.remaining,
                )
                _configure_connection_read_only(conn, engine.dialect.name)
                deadline.check()
                timeout_seconds = max(1, int(deadline.remaining))
                cleanup_timeout = _configure_query_timeout(
                    conn,
                    engine.dialect.name,
                    timeout_seconds,
                )
                result: Any | None = None
                capture_listener: Any | None = None
                try:
                    streamed = conn.execution_options(
                        stream_results=True,
                        max_row_buffer=FETCH_BATCH_ROWS,
                        yield_per=FETCH_BATCH_ROWS,
                    )
                    # SQLAlchemy normally does not return CursorResult until a
                    # blocking execute has completed. Capture the DBAPI cursor
                    # immediately before execution so cancellation can reach
                    # the active provider request instead of merely closing an
                    # otherwise opaque connection.
                    from sqlalchemy import event

                    def capture_cursor(
                        _conn: Any,
                        cursor: Any,
                        _statement: Any,
                        _parameters: Any,
                        _context: Any,
                        _executemany: Any,
                    ) -> None:
                        if interrupt_state is not None:
                            interrupt_state.bind_result(cursor)

                    capture_listener = capture_cursor
                    event.listen(conn, "before_cursor_execute", capture_listener)
                    # ``sql`` has already passed Sift's single-statement,
                    # read-only validator and contains no application-supplied
                    # bind values. Execute it as driver SQL so SQLAlchemy's
                    # ``text()`` template parser cannot reinterpret colons inside
                    # JSON, timestamps, casts, or vendor syntax as bind markers.
                    result = streamed.exec_driver_sql(sql)
                    if transport_probe is not None and interrupt_state is not None:
                        with interrupt_state._lock:
                            active_cursor = interrupt_state.active_result
                        transport_probe(_query_transport_evidence(
                            active_cursor, backend,
                        ))
                    columns = list(result.keys())
                    rows_seen = 0
                    yielded = False
                    while rows_seen < limit + 1:
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        wanted = min(FETCH_BATCH_ROWS, limit + 1 - rows_seen)
                        batch = result.fetchmany(wanted)
                        if not batch:
                            break
                        frame = pd.DataFrame.from_records(batch, columns=columns)
                        rows_seen += len(frame)
                        yielded = True
                        yield frame
                    if not yielded:
                        yield pd.DataFrame(columns=columns)
                finally:
                    if capture_listener is not None:
                        try:
                            from sqlalchemy import event

                            event.remove(conn, "before_cursor_execute", capture_listener)
                        except Exception:  # noqa: BLE001, S110
                            pass
                    if result is not None:
                        result.close()
                    cleanup_timeout()
    except ConnectorError:
        raise
    except IntegrationCancelled as e:
        raise ConnectorError(
            "database extraction cancelled",
            code="cancelled",
            action="Start a new extraction when ready.",
        ) from e
    except Exception as e:
        _raise_interrupted_operation(interrupt_state, deadline)
        # This includes DNS/TCP/authentication and read-only-session setup.
        # Driver-native limits make the blocking connect synchronous and
        # finite; a late cancellation/deadline is checked before classifying
        # the safe driver diagnostic.
        deadline.check()
        if not connected and _driver_error_is_timeout(e, connection):
            raise _connect_deadline_error("extraction", connect_timeout) from e
        raise _driver_connector_error("the query failed", e, connection) from e
    finally:
        engine.dispose()


@dataclass(frozen=True)
class _MaterializedExtract:
    path: Path
    rows: int
    columns: int
    truncated: bool
    column_renames: tuple[dict[str, Any], ...]


def _materialize_query_batches(
    batches: Any,
    cwd: Path,
    safe_stem: str,
    limit: int,
    cancellation: CancellationToken | None = None,
    progress: Callable[[ExtractionProgress], None] | None = None,
) -> _MaterializedExtract:
    """Spool bounded query batches and atomically assemble one Parquet file.

    Parts are deliberately written before the final merge.  This keeps peak
    memory proportional to ``FETCH_BATCH_ROWS`` and also lets Arrow inspect
    every part's schema before creating the final writer, so a null-only
    first batch cannot lock a later numeric/string column to Arrow ``null``.
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as e:  # pragma: no cover - declared dependency
        raise ConnectorError("pyarrow is not installed") from e

    rows = 0
    bytes_seen = 0
    truncated = False
    canonical_names: list[str] | None = None
    column_renames: tuple[dict[str, Any], ...] = ()
    iterator = iter(batches)
    with tempfile.TemporaryDirectory(prefix=".sift-query-parts-", dir=cwd) as td:
        parts: list[Path] = []
        try:
            for index, frame in enumerate(iterator):
                if cancellation is not None:
                    cancellation.raise_if_cancelled()
                frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
                bytes_seen += frame_bytes
                if bytes_seen > DEFAULT_BYTE_LIMIT:
                    raise ConnectorError(
                        "the extract exceeded Sift's 256 MiB in-memory safety "
                        "limit. Aggregate or select fewer/wider columns in "
                        "SQL, then try again."
                    )
                if canonical_names is None:
                    frame, column_renames = _canonicalize_result_columns(frame)
                    canonical_names = list(frame.columns)
                else:
                    if len(frame.columns) != len(canonical_names):
                        raise ConnectorError(
                            "the database changed the result column count "
                            "while the extract was streaming"
                        )
                    frame = frame.copy(deep=False)
                    frame.columns = canonical_names

                remaining = limit - rows
                if len(frame) > remaining:
                    truncated = True
                    frame = frame.iloc[:max(0, remaining)]
                if len(frame):
                    if shutil.disk_usage(cwd).free < (
                        frame_bytes + _MIN_FREE_DISK_RESERVE
                    ):
                        raise ConnectorError(
                            "database extraction stopped before exhausting "
                            "disk space; Sift's free-space safety reserve "
                            "would be crossed"
                        )
                    part = Path(td) / f"part-{index:06d}.parquet"
                    try:
                        table = pa.Table.from_pandas(frame, preserve_index=False)
                        pq.write_table(table, part)
                    except Exception as e:
                        raise ConnectorError(
                            "the database returned a value type that cannot be "
                            "represented safely in Parquet; cast that column to "
                            "text, JSON, binary, decimal, date, or timestamp in "
                            f"the query ({type(e).__name__})"
                        ) from e
                    parts.append(part)
                    rows += len(frame)
                    _emit_progress(
                        progress,
                        "materializing",
                        rows=rows,
                        bytes_buffered=bytes_seen,
                    )
                if truncated or rows >= limit:
                    # We queried limit+1 rows; reaching exactly the limit is
                    # only known to be truncation once one extra row exists.
                    if not truncated:
                        try:
                            extra = next(iterator)
                        except StopIteration:
                            pass
                        else:
                            bytes_seen += int(
                                extra.memory_usage(index=True, deep=True).sum()
                            )
                            if bytes_seen > DEFAULT_BYTE_LIMIT:
                                raise ConnectorError(
                                    "the extract exceeded Sift's 256 MiB "
                                    "in-memory safety limit. Aggregate or "
                                    "select fewer/wider columns in SQL, then "
                                    "try again."
                                )
                            truncated = len(extra) > 0
                    break
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()

        if canonical_names is None:
            # A zero-row SQL result still has useful column metadata, but the
            # current DBAPI/DuckDB dataframe iterators yield no batch. Preserve
            # a valid zero-column Parquet rather than fabricating types.
            canonical_names = []

        fd, tmp_name = tempfile.mkstemp(
            prefix=".sift-extract-", suffix=".parquet", dir=cwd,
        )
        os.close(fd)
        assembled = Path(tmp_name)
        target: Path | None = None
        try:
            _emit_progress(
                progress, "finalizing", rows=rows, bytes_buffered=bytes_seen,
            )
            spooled_bytes = sum(part.stat().st_size for part in parts)
            if shutil.disk_usage(cwd).free < (
                spooled_bytes + _MIN_FREE_DISK_RESERVE
            ):
                raise ConnectorError(
                    "database extraction cannot assemble the final dataset "
                    "without crossing Sift's free-space safety reserve"
                )
            if not parts:
                import pandas as pd
                pd.DataFrame(columns=canonical_names).to_parquet(
                    assembled, index=False,
                )
            else:
                schemas = []
                for part in parts:
                    parquet_file = pq.ParquetFile(part)
                    try:
                        schemas.append(parquet_file.schema_arrow)
                    finally:
                        # POSIX permits deleting an open file and hid this
                        # leak. Windows correctly keeps the part locked until
                        # every Arrow reader is closed.
                        parquet_file.close()
                try:
                    schema = pa.unify_schemas(schemas, promote_options="permissive")
                except TypeError:  # pyarrow 15 compatibility
                    schema = pa.unify_schemas(schemas)
                with pq.ParquetWriter(assembled, schema) as writer:
                    for part in parts:
                        if cancellation is not None:
                            cancellation.raise_if_cancelled()
                        parquet_file = pq.ParquetFile(part)
                        try:
                            for batch in parquet_file.iter_batches(
                                batch_size=FETCH_BATCH_ROWS,
                            ):
                                table = pa.Table.from_batches([batch])
                                if table.schema != schema:
                                    table = table.cast(schema)
                                writer.write_table(table)
                        finally:
                            parquet_file.close()

            counter = 0
            while True:
                suffix = "" if counter == 0 else f"_{counter}"
                candidate = cwd / f"{safe_stem[:60]}{suffix}.parquet"
                try:
                    reserve_fd = os.open(
                        candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600,
                    )
                except FileExistsError:
                    counter += 1
                    continue
                os.close(reserve_fd)
                target = candidate
                break
            os.replace(assembled, target)
        except Exception:
            assembled.unlink(missing_ok=True)
            if target is not None:
                target.unlink(missing_ok=True)
            raise

    return _MaterializedExtract(
        path=target,
        rows=rows,
        columns=len(canonical_names),
        truncated=truncated,
        column_renames=column_renames,
    )


def _configure_connection_read_only(conn: Any, dialect: str) -> None:
    """Ask the database itself to reject writes where it has a portable
    session/transaction control.

    SQL text validation remains defense in depth.  The authoritative safety
    boundary is still a database account granted SELECT-only privileges,
    because no client-side parser can know whether an institution-specific
    function has side effects.  These controls protect the common engines
    even when an over-privileged account is supplied.
    """
    name = (dialect or "").lower()
    try:
        if name == "sqlite":
            conn.exec_driver_sql("PRAGMA query_only = ON")
        elif name in {"postgresql", "redshift", "oracle"}:
            # Both engines apply this to the transaction implicitly opened
            # by SQLAlchemy on the first statement.
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
        elif name in {"mysql", "mariadb"}:
            # Applies to the next transaction.  SQLAlchemy's first real data
            # statement then starts that read-only transaction.
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
    except Exception as e:
        # If Sift knows an engine has a read-only control, failure to apply it
        # is a safety failure, not a warning to ignore.
        raise ConnectorError(
            f"could not establish a read-only {name or 'database'} "
            f"session ({type(e).__name__}). Use a database account with "
            "SELECT-only permissions or correct the server configuration."
        ) from e


def _configure_query_timeout(conn: Any, dialect: str, seconds: int):
    """Apply a driver/server deadline where the adapter supports it."""
    name = (dialect or "").lower()

    def noop() -> None:
        return None

    milliseconds = int(seconds * 1000)
    try:
        raw = getattr(conn.connection, "driver_connection", conn.connection)
        if name == "sqlite":
            deadline = time.monotonic() + seconds
            raw.set_progress_handler(
                lambda: 1 if time.monotonic() >= deadline else 0,
                10_000,
            )
            return lambda: raw.set_progress_handler(None, 0)
        if name in {"postgresql", "redshift"}:
            conn.exec_driver_sql(f"SET LOCAL statement_timeout = {milliseconds}")
        elif name == "mysql":
            conn.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME = {milliseconds}")
        elif name == "mariadb":
            conn.exec_driver_sql(f"SET SESSION max_statement_time = {seconds}")
        elif name == "snowflake":
            conn.exec_driver_sql(
                f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {seconds}"
            )
        elif name == "mssql" and hasattr(raw, "timeout"):
            raw.timeout = seconds
        elif name == "oracle" and hasattr(raw, "call_timeout"):
            raw.call_timeout = milliseconds
        return noop
    except Exception as e:
        raise ConnectorError(
            f"could not establish a bounded {name or 'database'} query "
            f"deadline ({type(e).__name__})"
        ) from e


def run_extract(
    cwd: Path,
    *,
    connection: ConnectionInput,
    sql: str,
    dataset_name: str,
    row_limit: int = DEFAULT_ROW_LIMIT,
    cancellation: CancellationToken | None = None,
    progress: Callable[[ExtractionProgress], None] | None = None,
    transport_probe: Callable[[QueryTransportEvidence], None] | None = None,
) -> ExtractResult:
    """Run a read-only query and materialize it as a session dataset.

    Host-side by design; see the module docstring. Raises
    :class:`ConnectorError` with an actionable message on any failure.
    """
    cwd = Path(cwd)
    if not cwd.is_dir():
        raise ConnectorError("no active session directory")
    _check_cancellation(cancellation, "extraction")
    raw_connection = _connection_uri(connection)
    if not isinstance(raw_connection, str) or not raw_connection.strip():
        raise ConnectorError("no connection given")
    if not isinstance(sql, str):
        raise ConnectorError("no query given")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        dataset_name = "extract"
    _emit_progress(progress, "starting")
    backend, effective_connection = _prepare_connection(cwd, connection)
    normalized = normalize_sql(sql, backend=backend)
    if normalized is None:
        raise ConnectorError(
            "only a single row-returning read-only statement is allowed "
            "(SELECT / WITH / VALUES). Sift will not run statements that "
            "can modify a database, and will not run multiple statements "
            "in one call."
        )
    try:
        limit = max(1, min(int(row_limit), DEFAULT_ROW_LIMIT))
    except (TypeError, ValueError):
        limit = DEFAULT_ROW_LIMIT

    requested_name = (dataset_name or "extract").strip()
    # The source hub labels this value as a filename because that is how a
    # non-technical researcher thinks about the result. The materializer
    # always writes Parquet, so accept either ``cohort`` or
    # ``cohort.parquet`` without producing ``cohort_parquet.parquet``.
    if requested_name.casefold().endswith(".parquet"):
        requested_name = requested_name[:-len(".parquet")]
    safe_stem = portable_stem(
        re.sub(r"[^A-Za-z0-9_-]+", "_", requested_name) or "extract"
    )
    try:
        _emit_progress(progress, "querying")
        batches = (
            _run_duckdb(
                _connection_uri(effective_connection), normalized, backend,
                limit, cancellation,
            )
            if backend in ("duckdb", "duckdb-file")
            else _run_sqlalchemy(
                effective_connection, normalized, backend, limit, cancellation,
                transport_probe,
            )
        )
        materialized = _materialize_query_batches(
            batches, cwd, safe_stem, limit, cancellation, progress,
        )
    except Exception as e:
        if isinstance(e, ConnectorError):
            raise
        if isinstance(e, IntegrationCancelled):
            raise ConnectorError(
                "database extraction cancelled",
                code="cancelled",
                action="Start a new extraction when ready.",
            ) from e
        raise ConnectorError(
            f"could not write the extract to disk: {type(e).__name__}"
        ) from e
    target = materialized.path
    row_count = materialized.rows
    column_count = materialized.columns
    truncated = materialized.truncated
    column_renames = materialized.column_renames

    display = redact_connection(raw_connection)
    query_sha256 = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    metadata_sidecar = target.with_suffix(target.suffix + ".metadata.json")
    try:
        dataset_sha256 = _sha256_file(target)
    except OSError as e:
        target.unlink(missing_ok=True)
        raise ConnectorError(
            f"could not verify the written extract: {type(e).__name__}"
        ) from e
    try:
        metadata_body = {
            "format": "database_extract",
            "extraction_scope": "bounded_read_only_query",
            "query_sha256": query_sha256,
            "backend": backend,
            "row_limit": limit,
            "column_renames": list(column_renames),
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=".extract-metadata-", suffix=".tmp", dir=target.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(metadata_body, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, metadata_sidecar)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        from sift.canonical_dataset import ensure_manifest
        canonical_manifest = ensure_manifest(
            cwd,
            target,
            selection={
                "extraction_scope": "bounded_read_only_query",
                "query_sha256": query_sha256,
                "backend": backend,
                "row_limit": limit,
                "column_renames": list(column_renames),
            },
            transformations=({
                "operation": "database_extract",
                "query_sha256": query_sha256,
                "runtime": backend,
            },),
        )
    except Exception as e:
        target.unlink(missing_ok=True)
        metadata_sidecar.unlink(missing_ok=True)
        raise ConnectorError(
            f"could not establish canonical extract identity: {type(e).__name__}"
        ) from e
    try:
        _record_ingestion(
            cwd, target, backend, display, row_count, column_count,
            truncated, query_sha256, dataset_sha256, column_renames,
            str(canonical_manifest["fingerprint"]),
        )
    except _IngestionRecordError as e:
        if not e.release_recorded:
            # No durable provenance names this dataset. Roll back the
            # canonical copy as well as the visible file so a failed import
            # does not silently retain confidential bytes.
            rollback_complete = False
            try:
                from sift.canonical_dataset import discard_uncommitted_manifest

                rollback_complete = discard_uncommitted_manifest(
                    cwd, target, str(canonical_manifest["fingerprint"]),
                )
            except Exception:
                rollback_complete = False
            try:
                target.unlink(missing_ok=True)
                metadata_sidecar.unlink(missing_ok=True)
            except OSError:
                rollback_complete = False
            if not rollback_complete:
                raise ConnectorError(
                    "could not record extract provenance and confidential "
                    "local cleanup was incomplete; close any program using "
                    "the extract and retry removal from .sift/datasets"
                ) from e
        raise ConnectorError(
            f"could not record extract provenance: {type(e).__name__}"
        ) from e
    except Exception as e:
        # An unexpected failure may have happened after the append committed.
        # Preserve the file rather than create a dangling immutable ledger row.
        raise ConnectorError(
            f"could not record extract provenance: {type(e).__name__}"
        ) from e
    _emit_progress(
        progress,
        "complete",
        rows=row_count,
        bytes_buffered=target.stat().st_size,
    )
    return ExtractResult(
        dataset_path=target,
        rows=row_count,
        columns=column_count,
        truncated=truncated,
        backend=backend,
        connection_display=display,
        query_sha256=query_sha256,
        dataset_sha256=dataset_sha256,
        column_renames=column_renames,
        canonical_fingerprint=canonical_manifest["fingerprint"],
    )


class _IngestionRecordError(OSError):
    """A required provenance append failed, with commit state preserved."""

    def __init__(self, message: str, *, release_recorded: bool) -> None:
        super().__init__(message)
        self.release_recorded = release_recorded


def _record_ingestion(
    cwd: Path,
    target: Path,
    backend: str,
    connection_display: str,
    rows: int,
    columns: int,
    truncated: bool,
    query_sha256: str,
    dataset_sha256: str,
    column_renames: tuple[dict[str, Any], ...],
    canonical_fingerprint: str,
) -> None:
    """Record the materialization in the release ledger.

    This is an *ingestion*, not a disclosure — nothing left the
    machine. It is recorded anyway so an auditor reading the ledger
    can see where a dataset came from; an extract whose provenance is
    invisible is exactly the thing a data-governance reviewer asks
    about.
    """
    from sift import release_ledger

    recorded = release_ledger.record_release(
        cwd,
        kind="local_ingestion",
        tool="(database extract)",
        extra={
            "dataset": target.name,
            "backend": backend,
            "connection": connection_display,
            "rows": rows,
            "columns": columns,
            "truncated": truncated,
            # Audit provenance without persisting SQL text or identifiers.
            "query_sha256": query_sha256,
            "dataset_sha256": dataset_sha256,
            "canonical_fingerprint": canonical_fingerprint,
            "column_renames": list(column_renames),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    )
    if not recorded:
        raise _IngestionRecordError(
            "release ledger append failed", release_recorded=False,
        )
    from sift.integration_audit import record_integration_event

    if not record_integration_event(
        cwd,
        integration_id=backend,
        kind="database",
        action="materialize",
        outcome="success",
        metadata={"rows": rows, "columns": columns, "truncated": truncated},
    ):
        raise _IngestionRecordError(
            "integration audit append failed", release_recorded=True,
        )
