from __future__ import annotations

import sys

import pytest

from sift import remote_credentials as credentials


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.priority = 5

    def get_keyring(self):
        return self

    def set_password(self, service: str, key: str, value: str) -> None:
        self.values[(service, key)] = value

    def get_password(self, service: str, key: str) -> str | None:
        return self.values.get((service, key))

    def delete_password(self, service: str, key: str) -> None:
        self.values.pop((service, key), None)


@pytest.fixture()
def ring(monkeypatch: pytest.MonkeyPatch) -> FakeKeyring:
    value = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", value)
    return value


@pytest.mark.parametrize("kind", sorted(credentials.KINDS))
def test_remote_credentials_live_only_in_os_vault(
    kind: str,
    ring: FakeKeyring,
) -> None:
    credentials.save_remote_credential("Research Source", kind, "secret-value")
    assert credentials.resolve_remote_credential("research source", kind) == (
        "secret-value"
    )
    assert list(ring.values.values()) == ["secret-value"]
    credentials.delete_remote_credential("RESEARCH SOURCE", kind)
    with pytest.raises(credentials.RemoteCredentialError, match="no .* profile"):
        credentials.resolve_remote_credential("Research Source", kind)


@pytest.mark.parametrize(
    ("name", "kind", "secret"),
    [
        ("../escape", "https_bearer", "secret"),
        ("valid", "unknown", "secret"),
        ("valid", "azure_sas", ""),
    ],
)
def test_invalid_remote_credential_profiles_fail_closed(
    name: str,
    kind: str,
    secret: str,
    ring: FakeKeyring,
) -> None:
    with pytest.raises(credentials.RemoteCredentialError):
        credentials.save_remote_credential(name, kind, secret)
    assert ring.values == {}


def test_header_credentials_reject_controls_but_sftp_keys_allow_pem_newlines(
    ring: FakeKeyring,
) -> None:
    with pytest.raises(credentials.RemoteCredentialError, match="control"):
        credentials.save_remote_credential(
            "Header", "https_bearer", "safe\r\ninjected",
        )
    pem = "-----BEGIN " + "PRIVATE KEY-----\nmaterial\n-----END PRIVATE KEY-----"
    credentials.save_remote_credential("SSH", "sftp_key", pem)
    assert credentials.resolve_remote_credential("SSH", "sftp_key") == pem
    with pytest.raises(credentials.RemoteCredentialError, match="control"):
        credentials.save_remote_credential("Bad SSH", "sftp_key", "key\x01data")


def test_manually_inserted_unsafe_header_credential_is_rejected(
    ring: FakeKeyring,
) -> None:
    ring.values[(credentials.KEYRING_SERVICE, "https_bearer:header")] = (
        "safe\r\ninjected"
    )
    with pytest.raises(credentials.RemoteCredentialError, match="invalid"):
        credentials.resolve_remote_credential("Header", "https_bearer")


def test_plaintext_keyring_backend_is_refused_but_can_be_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlaintextBackend(FakeKeyring):
        pass

    PlaintextBackend.__module__ = "keyrings.alt.file"
    plaintext = PlaintextBackend()
    plaintext.values[(
        credentials.KEYRING_SERVICE, "https_bearer:legacy",
    )] = "legacy-secret"
    monkeypatch.setitem(sys.modules, "keyring", plaintext)

    with pytest.raises(credentials.RemoteCredentialError, match="secure OS"):
        credentials.resolve_remote_credential("Legacy", "https_bearer")
    with pytest.raises(credentials.RemoteCredentialError, match="secure OS"):
        credentials.save_remote_credential("New", "https_bearer", "secret")
    credentials.delete_remote_credential("Legacy", "https_bearer")
    assert plaintext.values == {}
