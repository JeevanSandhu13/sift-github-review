from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from sift import database_profiles as profiles


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, value: str) -> None:
        self.values[(service, username)] = value

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


@pytest.fixture()
def vault(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeKeyring:
    ring = FakeKeyring()
    monkeypatch.setattr(profiles, "_keyring", lambda: ring)
    monkeypatch.setattr(
        profiles,
        "profile_index_path",
        lambda: tmp_path / "profiles.json",
    )
    return ring


def test_profile_secret_lives_only_in_keyring(vault: FakeKeyring) -> None:
    secret = "postgresql://alice:hunter2@localhost/research"
    row = profiles.save_profile("Hospital", secret)
    assert row["connection"] == "postgresql://alice:***@localhost/research"
    assert profiles.resolve_profile("hospital") == secret
    disk = profiles.profile_index_path().read_text(encoding="utf-8")
    assert "hunter2" not in disk
    assert "***" in disk
    assert vault.values[(profiles.KEYRING_SERVICE, "hospital")] == secret


def test_snowflake_key_pair_profile_resolves_host_only_connect_args(
    vault: FakeKeyring,
) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from sift.connectors import ConnectionSpec

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    uri = "snowflake://researcher@account/db/schema?ocsp_fail_open=false"
    profiles.save_snowflake_key_pair_profile(
        "Snowflake Key", uri, private_key_pem=pem,
    )
    resolved = profiles.resolve_profile("snowflake key")
    assert isinstance(resolved, ConnectionSpec)
    assert resolved.authentication == "key_pair"
    assert isinstance(resolved.connect_args["private_key"], bytes)
    disk = profiles.profile_index_path().read_text(encoding="utf-8")
    assert "BEGIN PRIVATE KEY" not in disk
    assert "BEGIN PRIVATE KEY" in vault.values[
        (profiles.KEYRING_SERVICE, "snowflake key")
    ]
    assert "PRIVATE KEY" not in repr(resolved)


@pytest.mark.parametrize("mode", ["oauth_u2m", "oauth_m2m"])
def test_databricks_oauth_profile_builds_driver_connect_args(
    mode: str,
    vault: FakeKeyring,
) -> None:
    from sift.connectors import ConnectionSpec

    uri = "databricks://user@workspace:443?http_path=/sql/1.0/warehouses/test"
    profiles.save_databricks_oauth_profile(
        f"Databricks {mode}",
        uri,
        mode=mode,
        client_id="client-id" if mode == "oauth_m2m" else None,
        client_secret="client-secret" if mode == "oauth_m2m" else None,
    )
    resolved = profiles.resolve_profile(f"databricks {mode}")
    assert isinstance(resolved, ConnectionSpec)
    if mode == "oauth_m2m":
        provider = resolved.connect_args["credentials_provider"]
        assert provider.oauth_client_id == "client-id"
        assert provider.oauth_client_secret == "client-secret"
        assert "client-secret" not in repr(provider)
    else:
        assert resolved.connect_args["auth_type"] == "databricks-oauth"
    assert "client-secret" not in profiles.profile_index_path().read_text(encoding="utf-8")
    assert "client-secret" not in repr(resolved)


def test_databricks_oauth_profile_rejects_conflicting_uri_before_vault_write(
    vault: FakeKeyring,
) -> None:
    with pytest.raises(profiles.DatabaseProfileError, match="cannot be combined"):
        profiles.save_databricks_oauth_profile(
            "Databricks OAuth",
            "databricks://user:pat-secret@workspace:443?http_path=/sql/test",
            mode="oauth_u2m",
        )
    assert vault.values == {}
    assert not profiles.profile_index_path().exists()


def test_databricks_m2m_profile_normalizes_credential_whitespace(
    vault: FakeKeyring,
) -> None:
    profiles.save_databricks_oauth_profile(
        "Databricks M2M",
        "databricks://user@workspace:443?http_path=/sql/test",
        mode="oauth_m2m",
        client_id="  client-id  ",
        client_secret="  client-secret  ",
    )
    resolved = profiles.resolve_profile("Databricks M2M")
    provider = resolved.connect_args["credentials_provider"]
    assert provider.oauth_client_id == "client-id"
    assert provider.oauth_client_secret == "client-secret"


def test_profile_names_are_case_insensitive_and_updates_replace(
    vault: FakeKeyring,
) -> None:
    profiles.save_profile("Lab DB", "sqlite:////tmp/one.db")
    profiles.save_profile("lab db", "sqlite:////tmp/two.db")
    rows = profiles.list_profiles()
    assert len(rows) == 1
    assert rows[0]["name"] == "lab db"
    assert profiles.resolve_profile("LAB DB") == "sqlite:////tmp/two.db"


def test_delete_removes_vault_secret_and_index(vault: FakeKeyring) -> None:
    profiles.save_profile("Local", "sqlite:////tmp/research.db")
    profiles.delete_profile("local")
    assert profiles.list_profiles() == []
    assert vault.values == {}
    with pytest.raises(profiles.DatabaseProfileError, match="no database profile"):
        profiles.resolve_profile("local")


def test_delete_repairs_index_when_vault_secret_is_already_missing(
    vault: FakeKeyring,
) -> None:
    profiles.save_profile("Local", "sqlite:////tmp/research.db")
    vault.values.clear()
    profiles.delete_profile("local")
    assert profiles.list_profiles() == []


def test_failed_profile_update_restores_previous_vault_secret(
    monkeypatch: pytest.MonkeyPatch,
    vault: FakeKeyring,
) -> None:
    original = "sqlite:////tmp/original.db"
    profiles.save_profile("Local", original)
    monkeypatch.setattr(
        profiles,
        "_save_index",
        lambda _rows: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(profiles.DatabaseProfileError, match="profile index"):
        profiles.save_profile("Local", "sqlite:////tmp/replacement.db")
    assert vault.values[(profiles.KEYRING_SERVICE, "local")] == original


def test_non_os_profile_update_failure_restores_previous_vault_secret(
    monkeypatch: pytest.MonkeyPatch,
    vault: FakeKeyring,
) -> None:
    original = "sqlite:////tmp/original.db"
    profiles.save_profile("Local", original)
    monkeypatch.setattr(
        profiles,
        "_save_index",
        lambda _rows: (_ for _ in ()).throw(RuntimeError("serialization failed")),
    )
    with pytest.raises(profiles.DatabaseProfileError, match="profile index"):
        profiles.save_profile("Local", "sqlite:////tmp/replacement.db")
    assert vault.values[(profiles.KEYRING_SERVICE, "local")] == original


def test_profile_reports_when_index_and_vault_rollback_both_fail(
    monkeypatch: pytest.MonkeyPatch,
    vault: FakeKeyring,
) -> None:
    original = "sqlite:////tmp/original.db"
    profiles.save_profile("Local", original)
    real_set_password = vault.set_password

    def fail_restore(service: str, username: str, value: str) -> None:
        if value == original:
            raise RuntimeError("vault unavailable during rollback")
        real_set_password(service, username, value)

    monkeypatch.setattr(vault, "set_password", fail_restore)
    monkeypatch.setattr(
        profiles,
        "_save_index",
        lambda _rows: (_ for _ in ()).throw(RuntimeError("serialization failed")),
    )
    with pytest.raises(profiles.DatabaseProfileError, match="manual inspection"):
        profiles.save_profile("Local", "sqlite:////tmp/replacement.db")


def test_failed_profile_delete_restores_vault_secret(
    monkeypatch: pytest.MonkeyPatch,
    vault: FakeKeyring,
) -> None:
    original = "sqlite:////tmp/original.db"
    profiles.save_profile("Local", original)
    monkeypatch.setattr(
        profiles,
        "_save_index",
        lambda _rows: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(profiles.DatabaseProfileError, match="rolled back"):
        profiles.delete_profile("Local")
    assert vault.values[(profiles.KEYRING_SERVICE, "local")] == original


def test_non_os_profile_delete_failure_restores_vault_secret(
    monkeypatch: pytest.MonkeyPatch,
    vault: FakeKeyring,
) -> None:
    original = "sqlite:////tmp/original.db"
    profiles.save_profile("Local", original)
    monkeypatch.setattr(
        profiles,
        "_save_index",
        lambda _rows: (_ for _ in ()).throw(RuntimeError("serialization failed")),
    )
    with pytest.raises(profiles.DatabaseProfileError, match="rolled back"):
        profiles.delete_profile("Local")
    assert vault.values[(profiles.KEYRING_SERVICE, "local")] == original


def test_insecure_keyring_backends_are_refused() -> None:
    class SecureBackend:
        priority = 5

    class PlaintextBackend:
        priority = 5

    PlaintextBackend.__module__ = "keyrings.alt.file"

    class FailedBackend:
        priority = 0

    FailedBackend.__module__ = "keyring.backends.fail"

    class ChainedBackend:
        priority = 10

        def __init__(self, backends) -> None:
            self.backends = backends

    assert profiles._keyring_backend_is_secure(SecureBackend()) is True
    assert profiles._keyring_backend_is_secure(PlaintextBackend()) is False
    assert profiles._keyring_backend_is_secure(FailedBackend()) is False
    assert profiles._keyring_backend_is_secure(
        ChainedBackend((SecureBackend(), PlaintextBackend())),
    ) is False


@pytest.mark.parametrize("name", ["", "../escape", "x" * 65, "emoji-🔑"])
def test_profile_name_is_bounded(name: str, vault: FakeKeyring) -> None:
    with pytest.raises(profiles.DatabaseProfileError):
        profiles.save_profile(name, "sqlite:////tmp/research.db")


def test_database_profile_payloads_are_bounded_before_vault_write(
    vault: FakeKeyring,
) -> None:
    with pytest.raises(profiles.DatabaseProfileError, match="64 KiB"):
        profiles.save_profile(
            "Oversized", "sqlite:///" + "x" * profiles.MAX_CONNECTION_BYTES,
        )
    with pytest.raises(profiles.DatabaseProfileError, match="invalid or too large"):
        profiles._save_profile_payload(
            "Oversized",
            "sqlite:////tmp/research.db",
            vault_secret="x" * (profiles.MAX_VAULT_PAYLOAD_BYTES + 1),
            authentication="test",
        )
    assert vault.values == {}


def test_index_never_contains_secret_fields(vault: FakeKeyring) -> None:
    profiles.save_profile(
        "Warehouse",
        "snowflake://user:password-value@acct/db?token=token-value&ocsp_fail_open=false",
    )
    data = json.loads(profiles.profile_index_path().read_text(encoding="utf-8"))
    blob = json.dumps(data)
    assert "password-value" not in blob
    assert "token-value" not in blob
    assert data["profiles"]["warehouse"]["backend"] == "snowflake"


def test_corrupt_index_fails_closed_without_mutating_keyring(
    vault: FakeKeyring,
) -> None:
    profiles.profile_index_path().write_text("{broken", encoding="utf-8")
    vault.values[(profiles.KEYRING_SERVICE, "hospital")] = "original-secret"

    with pytest.raises(profiles.DatabaseProfileError, match="corrupt"):
        profiles.save_profile("Hospital", "sqlite:////tmp/replacement.db")

    assert vault.values[(profiles.KEYRING_SERVICE, "hospital")] == "original-secret"
    assert profiles.profile_index_path().read_text(encoding="utf-8") == "{broken"


def test_concurrent_profile_saves_do_not_lose_index_entries(
    vault: FakeKeyring,
) -> None:
    def save(index: int) -> None:
        profiles.save_profile(
            f"Database {index}",
            f"sqlite:////tmp/research-{index}.db",
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(save, range(24)))

    rows = profiles.list_profiles()
    assert len(rows) == 24
    assert len(vault.values) == 24


def test_profile_health_and_credential_rotation_are_auditable(
    vault: FakeKeyring,
) -> None:
    original = profiles.save_profile("Warehouse", "sqlite:////tmp/one.db")
    assert original["credential_generation"] == 1
    assert original["health_status"] == "unknown"

    healthy = profiles.record_profile_health("warehouse", healthy=True)
    assert healthy["health_status"] == "healthy"
    assert healthy["last_checked_at"]
    assert profiles.profile_health("WAREHOUSE")["credential_present"] is True

    rotated = profiles.rotate_profile_credential(
        "Warehouse", "sqlite:////tmp/two.db",
    )
    assert rotated["credential_generation"] == 2
    assert rotated["health_status"] == "unknown"
    assert profiles.resolve_profile("warehouse").endswith("two.db")


def test_profile_health_detects_missing_and_expired_credentials(
    vault: FakeKeyring,
) -> None:
    profiles.save_profile("Clinical", "sqlite:////tmp/clinical.db")
    expired = profiles.record_profile_health(
        "Clinical", healthy=False, error_code="authentication_expired",
    )
    assert expired["health_status"] == "authentication_expired"
    assert expired["last_error_code"] == "authentication_expired"

    vault.values.clear()
    health = profiles.profile_health("Clinical")
    assert health["health_status"] == "credential_missing"
    assert health["credential_present"] is False
