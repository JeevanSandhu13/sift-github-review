"""Vault-backed credentials for remote object integrations.

Only opaque profile names cross the desktop bridge. Secrets are resolved in
the host integration immediately before opening a download and never enter a
session directory, generated-code environment, model message, URI, or audit
record.
"""

from __future__ import annotations

import re
from typing import Any


KEYRING_SERVICE = "sift-remote-source"
KINDS = frozenset({
    "azure_sas", "https_bearer", "research_token", "sftp_key",
})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$")


class RemoteCredentialError(Exception):
    pass


def _credential_has_unsupported_controls(secret: str, kind: str) -> bool:
    allowed = {10, 13} if kind == "sftp_key" else set()
    return any(
        (ord(character) < 32 and ord(character) not in allowed)
        or ord(character) == 127
        for character in secret
    )


def _keyring(*, require_secure: bool = True) -> Any:
    try:
        import keyring
    except ImportError as e:  # pragma: no cover - declared dependency
        raise RemoteCredentialError("OS credential store is unavailable") from e
    if require_secure:
        from sift.integration_core import keyring_module_is_secure

        if not keyring_module_is_secure(keyring):
            raise RemoteCredentialError(
                "a secure OS credential store is unavailable; plaintext, null, "
                "and failed keyring backends are refused"
            )
    return keyring


def _key(name: str, kind: str) -> str:
    cleaned = name.strip() if isinstance(name, str) else ""
    if not _NAME.fullmatch(cleaned):
        raise RemoteCredentialError("credential profile name is invalid")
    if kind not in KINDS:
        raise RemoteCredentialError(f"unsupported remote credential kind: {kind!r}")
    return f"{kind}:{cleaned.casefold()}"


def save_remote_credential(name: str, kind: str, secret: str) -> None:
    key = _key(name, kind)
    if not isinstance(secret, str) or not secret.strip():
        raise RemoteCredentialError("credential is empty")
    if len(secret.encode("utf-8")) > 65_536:
        raise RemoteCredentialError("credential profile exceeds 64 KiB")
    cleaned = secret.strip()
    if _credential_has_unsupported_controls(cleaned, kind):
        raise RemoteCredentialError(
            "credential contains unsupported control characters"
        )
    try:
        _keyring().set_password(KEYRING_SERVICE, key, cleaned)
    except RemoteCredentialError:
        raise
    except Exception as e:
        raise RemoteCredentialError(
            f"OS credential store could not save the profile ({type(e).__name__})"
        ) from e


def resolve_remote_credential(name: str, kind: str) -> str:
    key = _key(name, kind)
    try:
        secret = _keyring().get_password(KEYRING_SERVICE, key)
    except RemoteCredentialError:
        raise
    except Exception as e:
        raise RemoteCredentialError(
            f"OS credential store could not read the profile ({type(e).__name__})"
        ) from e
    if not secret:
        raise RemoteCredentialError(
            f"no {kind.replace('_', ' ')} credential profile named {name!r}"
        )
    resolved = str(secret).strip()
    if (
        len(resolved.encode("utf-8")) > 65_536
        or _credential_has_unsupported_controls(resolved, kind)
    ):
        raise RemoteCredentialError("stored credential profile is invalid")
    return resolved


def delete_remote_credential(name: str, kind: str) -> None:
    key = _key(name, kind)
    # Deletion remains available even if the active backend has become
    # insecure, so a researcher can remove a legacy/plaintext entry rather
    # than Sift trapping it in place.
    ring = _keyring(require_secure=False)
    try:
        if ring.get_password(KEYRING_SERVICE, key) is not None:
            ring.delete_password(KEYRING_SERVICE, key)
    except Exception as e:
        raise RemoteCredentialError(
            f"OS credential store could not delete the profile ({type(e).__name__})"
        ) from e
