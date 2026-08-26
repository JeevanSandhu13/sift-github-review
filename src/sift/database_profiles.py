"""Named database connections backed by the operating-system secret store."""

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sift.connectors import (
    ConnectionInput,
    ConnectorError,
    describe_backend,
    redact_connection,
    validate_connection_security,
)
from sift.file_lock import exclusive_file_lock
from sift.integration_core import keyring_backend_is_secure as _keyring_backend_is_secure

KEYRING_SERVICE = "sift-database"
_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")
MAX_CONNECTION_BYTES = 64 * 1024
MAX_VAULT_PAYLOAD_BYTES = 1024 * 1024


class DatabaseProfileError(Exception):
    pass


def profile_index_path() -> Path:
    override = os.environ.get("SIFT_CONFIG_DIR")
    if override:
        root = Path(override).expanduser()
    elif platform.system() == "Windows":
        root = Path(os.environ.get("APPDATA", Path.home() / "AppData/Roaming")) / "Sift"
    elif platform.system() == "Darwin":
        root = Path.home() / "Library/Application Support/Sift"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "sift"
    return root / "database_profiles.json"


def _profile_id(name: str) -> str:
    if not isinstance(name, str) or not _PROFILE_RE.fullmatch(name.strip()):
        raise DatabaseProfileError(
            "profile name must be 1-64 characters using letters, numbers, "
            "spaces, dot, underscore, or hyphen"
        )
    return name.strip().casefold()


def _load_index() -> dict[str, dict[str, Any]]:
    path = profile_index_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise DatabaseProfileError(
            "the database profile index is unreadable or corrupt; refusing "
            "to overwrite it because doing so could orphan keychain secrets"
        ) from e
    if not isinstance(data, dict) or data.get("version") != 1:
        raise DatabaseProfileError(
            "the database profile index has an unsupported or invalid format"
        )
    rows = data.get("profiles")
    if not isinstance(rows, dict):
        raise DatabaseProfileError("the database profile index is malformed")
    if not all(isinstance(k, str) and isinstance(v, dict) for k, v in rows.items()):
        raise DatabaseProfileError("the database profile index contains invalid entries")
    return dict(rows)


def _save_index(rows: dict[str, dict[str, Any]]) -> None:
    path = profile_index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path.parent, 0o700)
    except OSError:
        pass
    fd, tmp_name = tempfile.mkstemp(
        prefix=".database_profiles.", suffix=".tmp", dir=path.parent
    )
    try:
        try:
            os.chmod(tmp_name, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"version": 1, "profiles": rows}, indent=2) + "\n"
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _keyring() -> Any:
    try:
        import keyring
    except ImportError as e:  # pragma: no cover - declared dependency
        raise DatabaseProfileError("OS credential store is unavailable") from e
    try:
        backend = keyring.get_keyring()
    except Exception as e:
        raise DatabaseProfileError(
            f"OS credential store could not be initialized ({type(e).__name__})"
        ) from e
    if not _keyring_backend_is_secure(backend):
        raise DatabaseProfileError(
            "a secure OS credential store is unavailable; Sift refuses null, "
            "failed, and plaintext keyring backends"
        )
    return keyring


def _profile_mutation_lock():
    return exclusive_file_lock(profile_index_path().with_suffix(".lock"))


def _save_profile_payload(
    name: str,
    connection: str,
    *,
    vault_secret: str,
    authentication: str,
) -> dict[str, Any]:
    """Validate a public URI and atomically store its host-only vault payload."""
    pid = _profile_id(name)
    if not isinstance(connection, str) or not connection.strip():
        raise DatabaseProfileError("connection is empty")
    if len(connection.encode("utf-8")) > MAX_CONNECTION_BYTES:
        raise DatabaseProfileError("connection exceeds the 64 KiB safety limit")
    if (
        not isinstance(vault_secret, str)
        or not vault_secret
        or len(vault_secret.encode("utf-8")) > MAX_VAULT_PAYLOAD_BYTES
        or "\x00" in vault_secret
    ):
        raise DatabaseProfileError("credential payload is invalid or too large")
    try:
        backend = describe_backend(connection)
        validate_connection_security(connection, backend)
    except ConnectorError as e:
        raise DatabaseProfileError(str(e)) from e
    from sift import enterprise_policy

    enterprise = enterprise_policy.load_enterprise_policy()
    if not enterprise_policy.database_backend_allowed(
        backend,
        enterprise,
    ):
        raise DatabaseProfileError(
            f"database backend {backend!r} is blocked by enterprise policy"
        )
    if not enterprise_policy.integration_endpoint_allowed(
        connection,
        enterprise,
        local_hint=backend in {"sqlite", "duckdb", "duckdb-file"},
    ):
        raise DatabaseProfileError("database endpoint is blocked by enterprise policy")

    with _profile_mutation_lock():
        # Load-modify-save and the matching keyring mutation are one critical
        # section. Otherwise concurrent saves can lose an index entry while
        # leaving its now-undiscoverable secret in the OS vault.
        rows = _load_index()
        previous_row = rows.get(pid, {})
        ring = _keyring()
        try:
            previous = ring.get_password(KEYRING_SERVICE, pid)
            ring.set_password(KEYRING_SERVICE, pid, vault_secret)
        except Exception as e:
            raise DatabaseProfileError(
                "could not save profile in the OS credential store "
                f"({type(e).__name__})"
            ) from e

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row = {
            "name": name.strip(),
            "backend": backend,
            "authentication": authentication,
            "connection": redact_connection(connection.strip()),
            "updated_at": now,
            "credential_updated_at": now,
            "credential_generation": int(
                previous_row.get("credential_generation", 0)
            ) + 1,
            "health_status": "unknown",
            "last_checked_at": None,
            "last_error_code": None,
        }
        rows[pid] = row
        try:
            _save_index(rows)
        except Exception as e:
            # Keep keyring and index aligned. An update restores the old
            # secret; a new profile removes its orphan secret.
            try:
                if previous is None:
                    ring.delete_password(KEYRING_SERVICE, pid)
                else:
                    ring.set_password(KEYRING_SERVICE, pid, previous)
            except Exception as restore_error:
                raise DatabaseProfileError(
                    "profile index update and credential rollback both "
                    "failed; the OS credential store needs manual inspection"
                ) from restore_error
            raise DatabaseProfileError(
                f"could not save the non-secret profile index ({type(e).__name__})"
            ) from e
    return dict(row)


def save_profile(name: str, connection: str) -> dict[str, Any]:
    """Validate and store a connection URI without writing it to disk."""
    if not isinstance(connection, str) or not connection.strip():
        raise DatabaseProfileError("connection is empty")
    return _save_profile_payload(
        name,
        connection,
        vault_secret=connection.strip(),
        authentication="uri",
    )


def save_snowflake_key_pair_profile(
    name: str,
    connection: str,
    *,
    private_key_pem: str,
    passphrase: str | None = None,
) -> dict[str, Any]:
    """Store a Snowflake private key only in the OS credential vault."""
    try:
        backend = describe_backend(connection)
    except ConnectorError as exc:
        raise DatabaseProfileError(str(exc)) from exc
    if backend != "snowflake":
        raise DatabaseProfileError("Snowflake key-pair authentication needs a Snowflake URI")
    if not isinstance(private_key_pem, str) or "PRIVATE KEY" not in private_key_pem:
        raise DatabaseProfileError("a PEM private key is required")
    if len(private_key_pem.encode("utf-8")) > MAX_VAULT_PAYLOAD_BYTES:
        raise DatabaseProfileError("private key exceeds the 1 MiB safety limit")
    # Reject an invalid/encrypted-with-wrong-passphrase key before persisting
    # it. Otherwise the profile appears healthy until its first connection and
    # the researcher can only repair it by replacing the vault entry.
    from sift.connectors import snowflake_key_pair_connection

    try:
        snowflake_key_pair_connection(
            connection,
            private_key_pem=private_key_pem,
            passphrase=passphrase,
        )
    except ConnectorError as exc:
        raise DatabaseProfileError(str(exc)) from exc
    payload = json.dumps({
        "version": 2,
        "connection": connection.strip(),
        "authentication": "snowflake_key_pair",
        "private_key_pem": private_key_pem,
        "passphrase": passphrase,
    }, separators=(",", ":"))
    return _save_profile_payload(
        name,
        connection,
        vault_secret=payload,
        authentication="key_pair",
    )


def save_databricks_oauth_profile(
    name: str,
    connection: str,
    *,
    mode: str,
    client_id: str | None = None,
    client_secret: str | None = None,
) -> dict[str, Any]:
    """Store Databricks U2M or M2M configuration in the credential vault."""
    try:
        backend = describe_backend(connection)
    except ConnectorError as exc:
        raise DatabaseProfileError(str(exc)) from exc
    if backend != "databricks":
        raise DatabaseProfileError("Databricks OAuth needs a Databricks URI")
    if mode not in {"oauth_u2m", "oauth_m2m"}:
        raise DatabaseProfileError("Databricks OAuth mode must be oauth_u2m or oauth_m2m")
    if mode == "oauth_m2m" and (
        not isinstance(client_id, str) or not client_id.strip()
        or not isinstance(client_secret, str) or not client_secret.strip()
    ):
        raise DatabaseProfileError("Databricks M2M needs a client ID and client secret")
    # Validate the complete structured configuration before persisting it.
    # This rejects a URI that also contains a PAT/password/auth_type now,
    # rather than letting the profile appear valid until first resolution.
    from sift.connectors import databricks_oauth_connection

    normalized_client_id = client_id.strip() if isinstance(client_id, str) else None
    normalized_client_secret = (
        client_secret.strip() if isinstance(client_secret, str) else None
    )
    try:
        databricks_oauth_connection(
            connection,
            mode=mode,  # type: ignore[arg-type]
            client_id=normalized_client_id,
            client_secret=normalized_client_secret,
        )
    except ConnectorError as exc:
        raise DatabaseProfileError(str(exc)) from exc
    payload = json.dumps({
        "version": 2,
        "connection": connection.strip(),
        "authentication": mode,
        "client_id": normalized_client_id,
        "client_secret": normalized_client_secret,
    }, separators=(",", ":"))
    return _save_profile_payload(
        name,
        connection,
        vault_secret=payload,
        authentication=mode,
    )


def rotate_profile_credential(name: str, connection: str) -> dict[str, Any]:
    """Atomically replace a profile credential and reset stale health state."""
    return save_profile(name, connection)


def record_profile_health(
    name: str,
    *,
    healthy: bool,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Persist a value-free connection health result in the public index."""
    pid = _profile_id(name)
    with _profile_mutation_lock():
        rows = _load_index()
        if pid not in rows:
            raise DatabaseProfileError(f"no database profile named {name!r}")
        row = dict(rows[pid])
        row["health_status"] = "healthy" if healthy else (
            "authentication_expired"
            if error_code == "authentication_expired"
            else "unhealthy"
        )
        row["last_checked_at"] = datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        )
        row["last_error_code"] = None if healthy else (
            str(error_code or "database_error")[:64]
        )
        rows[pid] = row
        try:
            _save_index(rows)
        except Exception as e:
            raise DatabaseProfileError(
                f"could not record profile health ({type(e).__name__})"
            ) from e
    return dict(row)


def profile_health(name: str) -> dict[str, Any]:
    """Return saved health plus current credential-presence state."""
    pid = _profile_id(name)
    rows = _load_index()
    if pid not in rows:
        raise DatabaseProfileError(f"no database profile named {name!r}")
    row = dict(rows[pid])
    row.setdefault("health_status", "unknown")
    try:
        present = bool(_keyring().get_password(KEYRING_SERVICE, pid))
    except Exception as e:
        raise DatabaseProfileError(
            f"OS credential store could not be read ({type(e).__name__})"
        ) from e
    if not present:
        row["health_status"] = "credential_missing"
    row["credential_present"] = present
    return row


def list_profiles() -> list[dict[str, Any]]:
    rows = _load_index()
    result: list[dict[str, Any]] = []
    for _, value in sorted(
        rows.items(),
        key=lambda item: str(item[1].get("name", "")).casefold(),
    ):
        row = dict(value)
        row.setdefault("credential_generation", 1)
        row.setdefault("credential_updated_at", row.get("updated_at"))
        row.setdefault("health_status", "unknown")
        row.setdefault("last_checked_at", None)
        row.setdefault("last_error_code", None)
        result.append(row)
    return result


def resolve_profile(name: str) -> ConnectionInput:
    """Resolve a secret for host-side use. Never return this via a bridge."""
    pid = _profile_id(name)
    if pid not in _load_index():
        raise DatabaseProfileError(f"no database profile named {name!r}")
    try:
        secret = _keyring().get_password(KEYRING_SERVICE, pid)
    except Exception as e:
        raise DatabaseProfileError(
            f"OS credential store could not be read ({type(e).__name__})"
        ) from e
    if not secret:
        raise DatabaseProfileError(
            f"credential for database profile {name!r} is missing"
        )
    raw = str(secret)
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if not isinstance(payload, dict) or payload.get("version") != 2:
        # A normal connection URI could itself be valid JSON text only in a
        # malformed profile. Preserve backwards compatibility for all v1 data.
        return raw
    connection = payload.get("connection")
    authentication = payload.get("authentication")
    if not isinstance(connection, str) or not connection.strip():
        raise DatabaseProfileError("the structured database profile is malformed")
    from sift.connectors import (
        ConnectorError,
        databricks_oauth_connection,
        snowflake_key_pair_connection,
    )

    if authentication == "snowflake_key_pair":
        pem = payload.get("private_key_pem")
        passphrase = payload.get("passphrase")
        if not isinstance(pem, str):
            raise DatabaseProfileError("the Snowflake private key is missing")
        try:
            return snowflake_key_pair_connection(
                connection, private_key_pem=pem, passphrase=passphrase,
            )
        except ConnectorError as e:
            raise DatabaseProfileError(str(e)) from e
    try:
        if authentication == "oauth_u2m":
            return databricks_oauth_connection(connection, mode="oauth_u2m")
        if authentication == "oauth_m2m":
            return databricks_oauth_connection(
                connection,
                mode="oauth_m2m",
                client_id=str(payload.get("client_id") or ""),
                client_secret=str(payload.get("client_secret") or ""),
            )
    except ConnectorError as e:
        raise DatabaseProfileError(str(e)) from e
    raise DatabaseProfileError("the structured database authentication mode is invalid")


def delete_profile(name: str) -> None:
    pid = _profile_id(name)
    with _profile_mutation_lock():
        rows = _load_index()
        if pid not in rows:
            return
        ring = _keyring()
        try:
            # A missing vault entry is repairable: no secret remains.
            existing = ring.get_password(KEYRING_SERVICE, pid)
            if existing is not None:
                ring.delete_password(KEYRING_SERVICE, pid)
        except Exception as e:
            raise DatabaseProfileError(
                "OS credential store could not delete the profile "
                f"({type(e).__name__})"
            ) from e
        rows.pop(pid, None)
        try:
            _save_index(rows)
        except Exception as e:
            if existing is not None:
                try:
                    ring.set_password(KEYRING_SERVICE, pid, existing)
                except Exception as restore_error:
                    raise DatabaseProfileError(
                        "credential deletion and rollback both failed; the OS "
                        "credential store needs manual inspection"
                    ) from restore_error
            raise DatabaseProfileError(
                "the local profile index could not be updated; credential "
                f"deletion was rolled back ({type(e).__name__})"
            ) from e
