"""Fail-closed, user-initiated desktop update preparation.

Sift never replaces a running application.  This module downloads one native
installer and its SBOM into a versioned staging directory only after a signed
manifest passes the release trust, channel, minimum-version, and rollback
checks.  The UI can then ask the researcher to launch the ordinary native
installer.  No check runs at startup and no dataset/session path is read.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import ssl
import stat
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from sift.release_manifest import (
    ReleaseManifestError,
    _strict_json_loads,
    _version_key,
    canonical_json,
    load_trusted_json,
    sha256_file,
    verify_release_policy,
)


MAX_MANIFEST_BYTES = 1024 * 1024
MAX_SBOM_BYTES = 32 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
FREE_SPACE_RESERVE_BYTES = 128 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 20.0
UPDATE_STATE_SERVICE = "org.sapieninstitute.sift.update"


class UpdateError(RuntimeError):
    """An update could not be proven safe and was not prepared."""


def load_highest_seen_version(channel: str, keyring_module: Any | None = None) -> str | None:
    """Read rollback state from the OS credential vault on explicit request."""
    if channel not in {"stable", "beta"}:
        raise UpdateError("update channel is invalid")
    try:
        backend = keyring_module
        if backend is None:
            import keyring
            backend = keyring
        value = backend.get_password(UPDATE_STATE_SERVICE, channel)
    except Exception as exc:  # noqa: BLE001 - every vault failure is fail-closed
        raise UpdateError(
            "the protected credential store is unavailable; rollback protection cannot be verified"
        ) from exc
    if value is None:
        return None
    try:
        _version_key(value)
    except ReleaseManifestError as exc:
        raise UpdateError("protected update rollback state is invalid") from exc
    return value


def persist_highest_seen_version(
    channel: str, version: str, keyring_module: Any | None = None,
) -> None:
    """Monotonically persist signed release state in the OS credential vault."""
    current = load_highest_seen_version(channel, keyring_module)
    try:
        _version_key(version)
    except ReleaseManifestError as exc:
        raise UpdateError("verified update version is invalid") from exc
    if current is not None and _version_key(current) >= _version_key(version):
        return
    try:
        backend = keyring_module
        if backend is None:
            import keyring
            backend = keyring
        backend.set_password(UPDATE_STATE_SERVICE, channel, version)
        confirmed = backend.get_password(UPDATE_STATE_SERVICE, channel)
    except Exception as exc:  # noqa: BLE001
        raise UpdateError("update rollback state could not be stored securely") from exc
    if confirmed != version:
        raise UpdateError("update rollback state could not be confirmed")


class Response(Protocol):
    headers: Any

    def read(self, amount: int = -1) -> bytes: ...
    def __enter__(self) -> "Response": ...
    def __exit__(self, *args: object) -> object: ...


class Transport(Protocol):
    def open(self, url: str, timeout: float) -> Response: ...


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise UpdateError("update locations must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise UpdateError("update locations cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise UpdateError("update locations cannot contain a query or fragment")
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise UpdateError("update location has an invalid port") from exc
    return parsed.scheme, parsed.hostname.casefold(), port


class _SameOriginRedirects(urllib.request.HTTPRedirectHandler):
    def __init__(self, origin: tuple[str, str, int]) -> None:
        self._origin = origin

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> Any:
        if _origin(newurl) != self._origin:
            raise UpdateError("update download redirected to a different origin")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HTTPSUpdateTransport:
    """TLS-verifying transport with same-origin redirect enforcement."""

    def __init__(self, manifest_url: str) -> None:
        origin = _origin(manifest_url)
        context = ssl.create_default_context()
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=context),
            _SameOriginRedirects(origin),
        )

    def open(self, url: str, timeout: float) -> Response:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json, application/octet-stream;q=0.9",
                "User-Agent": "Sift-Desktop-Updater/1",
            },
            method="GET",
        )
        try:
            return self._opener.open(request, timeout=timeout)  # type: ignore[return-value]
        except UpdateError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            raise UpdateError("the update service could not be reached securely") from exc


@dataclass(frozen=True)
class UpdateCandidate:
    manifest_url: str
    manifest: Mapping[str, Any]
    artifact: Mapping[str, Any]
    policy: Mapping[str, Any]
    installed_version: str

    @property
    def available(self) -> bool:
        return _version_key(str(self.manifest["version"])) > _version_key(
            self.installed_version
        )


def native_target(
    system: str | None = None, machine: str | None = None,
) -> tuple[str, str]:
    os_name = (system or platform.system()).casefold()
    architecture = (machine or platform.machine()).casefold()
    platform_name = {
        "darwin": "macos", "windows": "windows", "linux": "linux",
    }.get(os_name)
    arch = {
        "arm64": "arm64", "aarch64": "aarch64" if os_name == "linux" else "arm64",
        "x86_64": "x86_64", "amd64": "x86_64",
    }.get(architecture)
    if platform_name is None or arch is None:
        raise UpdateError("this operating system or architecture is not supported")
    return platform_name, arch


def default_update_root(system: str | None = None) -> Path:
    """Return an OS-conventional per-user cache location for installers."""
    os_name = (system or platform.system()).casefold()
    if os_name == "darwin":
        return Path.home() / "Library" / "Caches" / "Sift" / "updates"
    if os_name == "windows":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            raise UpdateError("Windows local application-data directory is unavailable")
        return Path(base) / "Sift" / "updates"
    if os_name == "linux":
        configured = os.environ.get("XDG_CACHE_HOME")
        linux_base = Path(configured) if configured else Path.home() / ".cache"
        return linux_base / "sift" / "updates"
    raise UpdateError("this operating system is not supported")


def _read_response(response: Response, *, limit: int, expected: int | None) -> bytes:
    raw_length = response.headers.get("Content-Length")
    if raw_length is not None:
        try:
            declared = int(raw_length)
        except (TypeError, ValueError) as exc:
            raise UpdateError("update response has an invalid length") from exc
        if declared < 0 or declared > limit or (expected is not None and declared != expected):
            raise UpdateError("update response length does not match its signed metadata")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, limit + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise UpdateError("update response exceeded its allowed size")
        chunks.append(chunk)
    if expected is not None and total != expected:
        raise UpdateError("update response length does not match its signed metadata")
    return b"".join(chunks)


def _fetch_bytes(
    transport: Transport, url: str, *, limit: int, expected: int | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> bytes:
    with transport.open(url, timeout) as response:
        return _read_response(response, limit=limit, expected=expected)


def _artifact_url(manifest_url: str, filename: str) -> str:
    if Path(filename).name != filename or filename in {".", ".."}:
        raise UpdateError("signed update filename is unsafe")
    base = manifest_url.rsplit("/", 1)[0] + "/"
    result = urllib.parse.urljoin(base, urllib.parse.quote(filename))
    if _origin(result) != _origin(manifest_url):
        raise UpdateError("signed update filename crossed the release origin")
    return result


def check_for_update(
    manifest_url: str,
    *,
    trust_store_path: Path,
    trust_store_sha256: str,
    channel: str,
    installed_version: str,
    highest_seen_version: str | None = None,
    transport: Transport | None = None,
    system: str | None = None,
    machine: str | None = None,
) -> UpdateCandidate:
    """Fetch and verify a manifest, then select exactly one native artifact."""
    _origin(manifest_url)
    client = transport or HTTPSUpdateTransport(manifest_url)
    raw = _fetch_bytes(client, manifest_url, limit=MAX_MANIFEST_BYTES)
    value = _strict_json_loads(raw, "update manifest")
    if not isinstance(value, Mapping):
        raise UpdateError("update manifest must be a JSON object")
    if raw not in {canonical_json(value), canonical_json(value) + b"\n"}:
        raise UpdateError("update manifest is not canonical JSON")
    try:
        trust = load_trusted_json(trust_store_path, trust_store_sha256)
        policy = verify_release_policy(
            value,
            trust,
            expected_channel=channel,
            installed_version=installed_version,
            highest_seen_version=highest_seen_version,
        )
    except ReleaseManifestError as exc:
        raise UpdateError(str(exc)) from exc
    target = native_target(system, machine)
    matches = [
        row for row in value["artifacts"]
        if (row["platform"], row["architecture"]) == target
    ]
    if len(matches) != 1:
        raise UpdateError("signed manifest does not contain exactly one native update")
    return UpdateCandidate(
        manifest_url=manifest_url,
        manifest=value,
        artifact=matches[0],
        policy=policy,
        installed_version=installed_version,
    )


def _write_verified_download(
    transport: Transport, url: str, destination: Path,
    descriptor: Mapping[str, Any], *, maximum: int,
) -> None:
    expected_size = int(descriptor["size"])
    if expected_size <= 0 or expected_size > maximum:
        raise UpdateError("signed update size is outside the supported limit")
    # Low-level Windows descriptors otherwise default to text mode, which
    # translates LF bytes during ``os.write``. That makes a correctly
    # downloaded JSON SBOM differ from its signed size and digest when it is
    # read back in binary mode for the final verification.
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
    )
    fd = os.open(destination, flags, 0o600)
    digest = hashlib.sha256()
    total = 0
    try:
        with transport.open(url, DEFAULT_TIMEOUT_SECONDS) as response:
            raw_length = response.headers.get("Content-Length")
            if raw_length is not None:
                try:
                    declared = int(raw_length)
                except (TypeError, ValueError) as exc:
                    raise UpdateError("update response has an invalid length") from exc
                if declared != expected_size:
                    raise UpdateError(
                        "update response length does not match its signed metadata"
                    )
            while True:
                chunk = response.read(
                    min(DOWNLOAD_CHUNK_BYTES, expected_size + 1 - total)
                )
                if not chunk:
                    break
                total += len(chunk)
                if total > expected_size:
                    raise UpdateError("update response exceeded its signed size")
                digest.update(chunk)
                remaining = memoryview(chunk)
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("short update write")
                    remaining = remaining[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    if total != expected_size:
        raise UpdateError("update response length does not match its signed metadata")
    if digest.hexdigest() != descriptor["sha256"]:
        raise UpdateError("downloaded update failed its signed SHA-256 check")


def prepare_update(
    candidate: UpdateCandidate,
    destination_root: Path,
    *,
    transport: Transport | None = None,
) -> dict[str, Any]:
    """Download and verify the selected installer and SBOM atomically."""
    if not candidate.available:
        return {
            "ok": True, "status": "current",
            "version": candidate.manifest["version"],
        }
    root = Path(destination_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    metadata = root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise UpdateError("update destination must be a private local directory")
    if os.name != "nt":
        # A pre-existing cache directory may have been created with a permissive
        # umask.  Installers are public artifacts, but keeping the staging root
        # owner-only also prevents another local account from swapping files
        # between verification and the researcher's explicit install action.
        if metadata.st_uid != os.getuid():
            raise UpdateError("update destination is not owned by the current user")
        try:
            os.chmod(root, 0o700)
        except OSError as exc:
            raise UpdateError("update destination permissions could not be secured") from exc
        secured = root.lstat()
        if stat.S_IMODE(secured.st_mode) != 0o700:
            raise UpdateError("update destination permissions are not private")
    artifact = candidate.artifact
    sbom = artifact["sbom"]
    required = int(artifact["size"]) + int(sbom["size"]) + FREE_SPACE_RESERVE_BYTES
    if shutil.disk_usage(root).free < required:
        raise UpdateError("there is not enough free disk space to prepare this update")
    client = transport or HTTPSUpdateTransport(candidate.manifest_url)
    version = str(candidate.manifest["version"])
    final = root / version
    if final.exists():
        raise UpdateError("an update staging directory already exists for this version")
    staging = Path(tempfile.mkdtemp(prefix=".sift-update-", dir=root))
    try:
        artifact_path = staging / str(artifact["filename"])
        sbom_path = staging / str(sbom["filename"])
        _write_verified_download(
            client,
            _artifact_url(candidate.manifest_url, artifact_path.name),
            artifact_path,
            artifact,
            maximum=int(artifact["size"]),
        )
        _write_verified_download(
            client,
            _artifact_url(candidate.manifest_url, sbom_path.name),
            sbom_path,
            sbom,
            maximum=MAX_SBOM_BYTES,
        )
        if sha256_file(artifact_path) != (artifact["sha256"], artifact["size"]):
            raise UpdateError("prepared installer changed during verification")
        if sha256_file(sbom_path) != (sbom["sha256"], sbom["size"]):
            raise UpdateError("prepared SBOM changed during verification")
        manifest_path = staging / "release-manifest.json"
        manifest_path.write_bytes(canonical_json(candidate.manifest) + b"\n")
        os.chmod(manifest_path, 0o600)
        os.replace(staging, final)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "ok": True,
        "status": "ready",
        "version": version,
        "installer": str(final / str(artifact["filename"])),
        "sbom": str(final / str(sbom["filename"])),
        "signing_key_id": candidate.policy["signing_key_id"],
        "highest_seen_version_to_persist": candidate.policy[
            "highest_seen_version_to_persist"
        ],
    }


__all__ = [
    "HTTPSUpdateTransport", "UpdateCandidate", "UpdateError",
    "check_for_update", "default_update_root", "load_highest_seen_version",
    "native_target", "persist_highest_seen_version", "prepare_update",
]
