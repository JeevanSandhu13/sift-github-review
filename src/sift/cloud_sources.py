"""Researcher-driven cloud object imports with a local privacy boundary.

Cloud credentials and network access live only in this host-side module. The
model tool registry cannot call it. A successful import is an immutable local
dataset with a content hash and required provenance record; from then on it
passes through the same schema, sandbox, and disclosure controls as any file
the researcher selected from disk.
"""

from __future__ import annotations

import hashlib
import base64
import http.client
import hmac
import ipaddress
import inspect
import json
import os
import re
import shutil
import socket
import stat
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import parse_qsl, unquote, urlencode, urljoin, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    getproxies,
    proxy_bypass,
)

from sift.integration_core import (
    CancellationToken,
    Deadline,
    IntegrationCancelled,
    IntegrationDeadlineExceeded,
    IntegrationError,
)
from sift.filename_safety import portable_filename
from sift.provider.error_safety import provider_error_message
from sift.schema import DATA_EXTENSIONS
from sift.text_safety import safe_text

DEFAULT_CLOUD_IMPORT_MAX_BYTES = 10 * 1024 * 1024 * 1024
MAX_CLOUD_IMPORT_MAX_BYTES = 1024 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
MAX_RESUME_ATTEMPTS = 2
_MIN_FREE_DISK_RESERVE = 512 * 1024 * 1024
_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
_SOCKET_DEFAULT_TIMEOUT: object = getattr(
    socket, "_GLOBAL_DEFAULT_TIMEOUT", object(),
)


class _TransferGuard:
    """Enforce one wall-clock deadline across blocking cloud SDK calls.

    Provider socket timeouts bound individual connects and idle reads, but do
    not impose a total deadline when a peer drip-feeds bytes.  This small
    watchdog performs no network work: it only closes resources registered by
    the foreground transfer when cancellation or the deadline fires.  The
    stop event makes the watcher bounded and joinable; unlike running an SDK
    operation in an executor thread, it cannot leave an unkillable request
    mutating state in the background.
    """

    def __init__(
        self,
        deadline: Deadline,
        cancellation: CancellationToken | None,
    ) -> None:
        self.deadline = deadline
        self.cancellation = cancellation
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._closers: list[Callable[[], Any]] = []
        self._reason: str | None = None
        self._thread = threading.Thread(
            target=self._watch,
            name="sift-cloud-transfer-deadline",
            daemon=True,
        )
        self._thread.start()

    def register(self, closer: Callable[[], Any] | None) -> None:
        if not callable(closer):
            return
        close_now = False
        with self._lock:
            if self._reason is None and not self._stop.is_set():
                self._closers.append(closer)
            else:
                close_now = self._reason is not None
        if close_now:
            try:
                closer()
            except Exception:
                pass

    def _watch(self) -> None:
        while not self._stop.is_set():
            if self.cancellation is not None and self.cancellation.cancelled:
                self._trip("cancelled")
                return
            remaining = self.deadline.remaining
            if remaining <= 0:
                self._trip("deadline")
                return
            self._stop.wait(min(0.05, remaining))

    def _trip(self, reason: str) -> None:
        with self._lock:
            if self._reason is not None or self._stop.is_set():
                return
            self._reason = reason
            closers = tuple(reversed(self._closers))
        for closer in closers:
            try:
                closer()
            except Exception:
                # Closing one SDK layer must not prevent the underlying stream
                # or client from being closed too.
                pass

    def check(self) -> None:
        # If the caller reaches the boundary just before the watcher thread,
        # trip synchronously so every already-registered SDK resource is
        # closed before the timeout/cancellation escapes.
        if self.cancellation is not None and self.cancellation.cancelled:
            self._trip("cancelled")
        elif self.deadline.remaining <= 0:
            self._trip("deadline")
        with self._lock:
            reason = self._reason
        if reason == "cancelled":
            raise IntegrationCancelled()
        if reason == "deadline":
            raise IntegrationDeadlineExceeded(self.deadline.timeout_seconds)
        self.deadline.check(self.cancellation)

    def finish(self) -> None:
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=0.25)

    def abort(self) -> None:
        """Close every registered provider resource after an open failure.

        ``finish`` deliberately leaves resources open because the successful
        download path still owns them.  Before a download object exists,
        however, an ordinary SDK error (authentication, missing object,
        metadata failure, and so on) has no later owner that can close the
        client registered by the provider adapter.  Trip the guard first so
        those resources are closed exactly once, then stop its watcher.
        """
        self._trip("aborted")
        self.finish()


def _guarded_timeout(timeout_seconds: float, guard: _TransferGuard | None) -> float:
    """Return a positive SDK connect/read timeout within the outer deadline."""
    remaining = guard.deadline.remaining if guard is not None else timeout_seconds
    configured = (
        guard.deadline.timeout_seconds if guard is not None else timeout_seconds
    )
    return max(0.05, min(60.0, configured, timeout_seconds, remaining))


def _call_with_guard(
    factory: Callable[..., _Download],
    *args: Any,
    guard: _TransferGuard,
) -> _Download:
    """Pass the private guard to built-ins without breaking adapter shims."""
    try:
        supports_guard = "_guard" in inspect.signature(factory).parameters
    except (TypeError, ValueError):
        supports_guard = False
    if supports_guard:
        return factory(*args, _guard=guard)
    return factory(*args)


def _call_supported_kwargs(
    operation: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Pass SDK timeout controls while tolerating minimal adapter shims."""
    try:
        parameters = inspect.signature(operation).parameters
    except (TypeError, ValueError):
        return operation(*args, **kwargs)
    accepts_arbitrary = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported = kwargs if accepts_arbitrary else {
        key: value for key, value in kwargs.items() if key in parameters
    }
    return operation(*args, **supported)


def _close_resources(*resources: Any) -> None:
    """Best-effort close every layer, even when an outer SDK close fails."""
    first_error: Exception | None = None
    for resource in resources:
        close = resource if callable(resource) else getattr(resource, "close", None)
        if not callable(close):
            continue
        try:
            close()
        except Exception as error:
            if first_error is None:
                first_error = error
    if first_error is not None:
        raise first_error


class CloudSourceError(IntegrationError):
    """An import problem stated without credentials or raw response data."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "cloud_source_error",
        retryable: bool = False,
        action: str = "Review the object selection and integration configuration.",
    ) -> None:
        super().__init__(
            code,
            message,
            integration_id="cloud_source",
            retryable=retryable,
            action=action,
        )


def _cloud_source_install_guidance(kind: str) -> str:
    """Return catalog-backed, copyable install guidance for an adapter."""
    try:
        from sift.integrations import CLOUD_SOURCE_ADAPTERS

        adapter = next(item for item in CLOUD_SOURCE_ADAPTERS if item.id == kind)
    except (ImportError, StopIteration):
        return "Install the integration required for this cloud source, then retry."
    if adapter.install_extra == "built-in":
        return "This cloud source is built in; check the URI and retry."
    return (
        f"Install Sift's {adapter.label} integration with "
        f'`pip install "sift[{adapter.install_extra}]"`, then retry.'
    )


@dataclass(frozen=True)
class CloudImportResult:
    dataset_path: Path
    source_kind: str
    source_display: str
    bytes_downloaded: int
    dataset_sha256: str
    remote_version: str | None
    content_type: str | None
    remote_checksum: str | None = None
    checksum_verified: bool = False
    remote_identifiers: dict[str, str] | None = None
    canonical_fingerprint: str | None = None
    integrity_checksum: str | None = None


def _verify_expected_checksum(
    path: Path,
    expected: str | None,
    *,
    guard: _TransferGuard | None = None,
) -> str | None:
    """Verify a repository checksum before an import is provenance-committed."""
    if not expected:
        return None
    value = expected.strip().casefold()
    if ":" in value:
        algorithm, wanted = value.split(":", 1)
    elif len(value) == 32:
        algorithm, wanted = "md5", value
    elif len(value) == 40:
        algorithm, wanted = "sha1", value
    elif len(value) == 64:
        algorithm, wanted = "sha256", value
    else:
        raise CloudSourceError("repository checksum format is unsupported")
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)  # noqa: S324 - integrity
    elif algorithm == "sha1":
        digest = hashlib.sha1(usedforsecurity=False)  # noqa: S324 - integrity
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise CloudSourceError("repository checksum algorithm is unsupported")
    if not re.fullmatch(r"[0-9a-f]+", wanted) or len(wanted) != digest.digest_size * 2:
        raise CloudSourceError("repository checksum format is unsupported")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if guard is not None:
                guard.check()
            digest.update(chunk)
    actual = digest.hexdigest()
    if not hmac.compare_digest(actual, wanted):
        raise CloudSourceError("repository checksum did not match the selected file")
    return f"{algorithm}:{actual}"


def cloud_import_max_bytes() -> int:
    raw = os.environ.get("SIFT_CLOUD_IMPORT_MAX_BYTES", "").strip()
    try:
        value = int(raw) if raw else DEFAULT_CLOUD_IMPORT_MAX_BYTES
    except ValueError:
        value = DEFAULT_CLOUD_IMPORT_MAX_BYTES
    return max(1, min(value, MAX_CLOUD_IMPORT_MAX_BYTES))


def describe_cloud_source(uri: str) -> str:
    if not isinstance(uri, str) or not uri.strip():
        raise CloudSourceError("no cloud object URI given")
    try:
        parsed = urlsplit(uri.strip())
    except ValueError as e:
        raise CloudSourceError("the cloud object URI is invalid") from e
    scheme = parsed.scheme.casefold()
    aliases = {"gs": "gcs", "az": "azure_blob", "azure": "azure_blob"}
    kind = aliases.get(scheme, scheme)
    if kind not in {"s3", "gcs", "azure_blob", "https", "sftp"}:
        raise CloudSourceError(
            "unsupported cloud object URI. Use s3://, gs://, az://, sftp://, "
            "or an https:// signed/download URL."
        )
    if not parsed.hostname:
        raise CloudSourceError("the cloud object URI has no bucket, account, or host")
    if parsed.password or (parsed.username and kind != "sftp"):
        raise CloudSourceError("credentials are not allowed in cloud URI authority fields")
    return kind


def redact_source_uri(uri: str) -> str:
    """Show origin/path while removing every query parameter and fragment."""
    try:
        parsed = urlsplit(uri)
    except (TypeError, ValueError):
        return "[invalid cloud URI]"
    if not parsed.scheme or not parsed.netloc:
        return "[invalid cloud URI]"
    return f"{parsed.scheme.casefold()}://{parsed.netloc}{unquote(parsed.path)}"


def _uri_secrets(uri: str) -> tuple[str, ...]:
    try:
        values = [value for _key, value in parse_qsl(urlsplit(uri).query)]
    except (TypeError, ValueError):
        return ()
    return tuple(value for value in values if len(value) >= 4)


def _safe_error(error: Any, uri: str, *secrets: str | None) -> str:
    values = _uri_secrets(uri) + tuple(
        value for value in secrets if isinstance(value, str) and value
    )
    return provider_error_message(error, secrets=values).split("\n", 1)[0]


def _object_parts(uri: str, kind: str) -> tuple[str, str, dict[str, str]]:
    parsed = urlsplit(uri)
    root = parsed.hostname or ""
    path = unquote(parsed.path).lstrip("/")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    allowed_query_keys = {
        "s3": {"versionId", "requestPayer"},
        "gcs": {"generation"},
        "azure_blob": {"versionid", "snapshot"},
    }[kind]
    unexpected = set(query) - allowed_query_keys
    if unexpected:
        raise CloudSourceError(
            f"{kind} URIs do not accept embedded credentials or arbitrary "
            "query parameters; use the SDK's external identity chain"
        )
    if kind == "azure_blob":
        pieces = path.split("/", 1)
        if len(pieces) != 2 or not all(pieces):
            raise CloudSourceError(
                "Azure Blob URIs must be az://account/container/blob-name"
            )
    elif not path:
        raise CloudSourceError("the cloud object URI has no object key")
    return root, path, query


def _remote_metadata(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = safe_text(str(value), max_len=200)
    return cleaned or None


def _validated_filename(uri: str, dataset_name: str | None) -> str:
    parsed = urlsplit(uri)
    remote_name = Path(unquote(parsed.path)).name
    requested = (dataset_name or "").strip()
    if requested:
        requested_path = Path(requested)
        if requested_path.suffix:
            name = requested_path.name
        else:
            name = requested + Path(remote_name).suffix
    else:
        name = remote_name
    name = _NAME_RE.sub("_", Path(name).name).strip(" .")
    if not name:
        raise CloudSourceError("the cloud object has no usable filename")
    if Path(name).suffix.casefold() not in DATA_EXTENSIONS:
        raise CloudSourceError(
            f"unsupported cloud dataset format {Path(name).suffix!r}; "
            "downloaded objects must use a Sift data-file extension"
        )
    # Bound the stem rather than slicing the full component: slicing here used
    # to remove a long object's already-validated extension. The portable rule
    # also neutralizes Windows device-name collisions before a download begins.
    return portable_filename(name)


_ARCHIVE_MAGIC = (
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08",  # zip/jar
    b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00", b"Rar!\x1a\x07", b"7z\xbc\xaf\x27\x1c",
)


def _reject_archive_payload(path: Path, *, limit: int) -> None:
    with path.open("rb") as handle:
        header = handle.read(512)
    # Office/OpenDocument files are ZIP containers but are bounded and parsed
    # by format-specific readers. Generic/nested archives are never extracted
    # by remote ingestion, eliminating traversal, member-count, nesting, and
    # decompression-bomb classes at this boundary.
    office = path.suffix.casefold() in {".xlsx", ".xlsm", ".ods"}
    archive_magic = any(header.startswith(magic) for magic in _ARCHIVE_MAGIC)
    if office and archive_magic:
        try:
            with zipfile.ZipFile(path) as archive:
                members = archive.infolist()
                if len(members) > 10_000:
                    raise CloudSourceError(
                        "remote document archive has too many members"
                    )
                expanded = 0
                for member in members:
                    member_path = Path(member.filename.replace("\\", "/"))
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise CloudSourceError(
                            "remote document archive contains an unsafe path"
                        )
                    mode = member.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise CloudSourceError(
                            "remote document archive contains a symbolic link"
                        )
                    if member_path.suffix.casefold() in {
                        ".zip", ".tar", ".tgz", ".gz", ".bz2", ".xz", ".rar", ".7z",
                    }:
                        raise CloudSourceError(
                            "remote document archive contains a nested archive"
                        )
                    expanded += max(0, member.file_size)
                    if member.file_size > max(1, member.compress_size) * 1_000:
                        raise CloudSourceError(
                            "remote document archive has an unsafe compression ratio"
                        )
                if expanded > min(limit, 2 * 1024**3):
                    raise CloudSourceError(
                        "remote document archive expands beyond the safe limit"
                    )
        except (zipfile.BadZipFile, OSError) as e:
            raise CloudSourceError("remote document archive is malformed") from e
        return
    if archive_magic:
        raise CloudSourceError(
            "remote archives are not accepted or extracted; provide the "
            "individual dataset object instead"
        )
    if len(header) >= 265 and header[257:262] == b"ustar":
        raise CloudSourceError("remote tar archives are not accepted or extracted")


def _validate_content_type(filename: str, content_type: str | None) -> None:
    if not content_type:
        return
    media = content_type.split(";", 1)[0].strip().casefold()
    # Generic binary is normal for Parquet, Stata, SAS, Excel, and signed
    # institutional endpoints. Strongly contradictory active/document media
    # types are rejected before any parser sees their bytes.
    forbidden_prefixes = ("image/", "audio/", "video/")
    forbidden = {
        "text/html", "application/xhtml+xml", "application/pdf",
        "application/javascript", "text/javascript",
        "application/x-dosexec", "application/x-executable",
    }
    if media in forbidden or media.startswith(forbidden_prefixes):
        raise CloudSourceError(
            f"remote object content type {media!r} is not a dataset type for "
            f"{Path(filename).suffix.casefold() or 'this file'}"
        )


def _safe_session_directory(cwd: Path) -> Path:
    if cwd.is_symlink():
        raise CloudSourceError("cloud import session directory cannot be a symlink")
    try:
        resolved = cwd.resolve(strict=True)
    except OSError as e:
        raise CloudSourceError("cloud import session directory is unavailable") from e
    if not resolved.is_dir():
        raise CloudSourceError("no active session directory")
    return resolved


def _reserve_target(cwd: Path, filename: str) -> Path:
    stem, suffix = Path(filename).stem, Path(filename).suffix
    for counter in range(10_000):
        label = "" if counter == 0 else f" ({counter})"
        candidate = cwd / f"{stem[:200]}{label}{suffix}"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate
    raise CloudSourceError("could not reserve a unique local dataset name")


def _guard_expected_size(cwd: Path, size: int | None, limit: int) -> None:
    if size is not None and (size < 0 or size > limit):
        raise CloudSourceError(
            f"cloud object is larger than the configured {limit} byte import limit"
        )
    free = shutil.disk_usage(cwd).free
    needed = (size if size is not None else 0) + _MIN_FREE_DISK_RESERVE
    if free < needed:
        raise CloudSourceError(
            "there is not enough free disk space for the cloud import plus "
            "Sift's safety reserve"
        )


def _copy_chunks(
    chunks: Iterable[bytes],
    handle: BinaryIO,
    *,
    limit: int,
    disk_directory: Path,
    cancellation: CancellationToken | None = None,
    deadline: Deadline | None = None,
) -> tuple[int, str, dict[str, str]]:
    digest = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)  # noqa: S324 - provider integrity only
    try:
        import google_crc32c

        crc32c: Any = google_crc32c.Checksum()
    except ImportError:
        crc32c = None
    total = 0
    for chunk in chunks:
        if deadline is not None:
            deadline.check(cancellation)
        elif cancellation is not None:
            cancellation.raise_if_cancelled()
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise CloudSourceError("cloud driver returned a non-byte download chunk")
        if not chunk:
            continue
        total += len(chunk)
        if total > limit:
            raise CloudSourceError(
                f"cloud object exceeded the configured {limit} byte import limit"
            )
        if shutil.disk_usage(disk_directory).free < (
            len(chunk) + _MIN_FREE_DISK_RESERVE
        ):
            raise CloudSourceError(
                "cloud import stopped before exhausting disk space; Sift's "
                "free-space safety reserve was reached"
            )
        handle.write(chunk)
        digest.update(chunk)
        md5.update(chunk)
        if crc32c is not None:
            crc32c.update(bytes(chunk))
    handle.flush()
    os.fsync(handle.fileno())
    checksums = {
        "sha256": base64.b64encode(digest.digest()).decode("ascii"),
        "md5": base64.b64encode(md5.digest()).decode("ascii"),
    }
    if crc32c is not None:
        checksums["crc32c"] = base64.b64encode(crc32c.digest()).decode("ascii")
    return total, digest.hexdigest(), checksums


@dataclass(frozen=True)
class _DownloadInfo:
    chunks: Iterator[bytes]
    expected_size: int | None
    version: str | None
    content_type: str | None
    close: Callable[[], Any]
    checksum_algorithm: str | None = None
    checksum_base64: str | None = None
    identifiers: dict[str, str] | None = None
    resumable: bool = False


_Download = _DownloadInfo | tuple[
    Iterator[bytes], int | None, str | None, str | None, Callable[[], Any]
]


def _download_info(value: _Download) -> _DownloadInfo:
    """Accept the historical five-tuple used by third-party adapters/tests."""
    if isinstance(value, _DownloadInfo):
        return value
    chunks, size, version, content_type, close = value
    return _DownloadInfo(chunks, size, version, content_type, close)


def _resolved_addresses(host: str, port: int) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        rows = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as e:
        raise CloudSourceError("HTTPS download host could not be resolved") from e
    addresses = {
        ipaddress.ip_address(str(row[4][0]).split("%", 1)[0]) for row in rows
    }
    if not addresses:
        raise CloudSourceError("HTTPS download host resolved to no addresses")
    return tuple(sorted(addresses, key=str))


def _https_origin(uri: str) -> tuple[str, str, int]:
    """Return the normalized HTTPS origin used for credential forwarding."""
    try:
        parsed = urlsplit(uri)
        port = parsed.port
    except (TypeError, ValueError) as e:
        raise CloudSourceError("remote HTTPS endpoint is invalid") from e
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise CloudSourceError("remote downloads and redirects must use HTTPS")
    if parsed.username or parsed.password:
        raise CloudSourceError("credentials are not allowed in HTTPS authority fields")
    return (
        "https",
        parsed.hostname.rstrip(".").casefold(),
        port or 443,
    )


def _validated_https_addresses(
    uri: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    """Resolve once, reject unsafe targets, and return the approved IP set.

    The caller must connect only to an address in this returned set. Merely
    resolving here and allowing the HTTP stack to resolve the hostname again
    would leave a DNS-rebinding window between validation and connection.
    """
    _scheme, host, port = _https_origin(uri)
    addresses = _resolved_addresses(host, port)
    if os.environ.get("SIFT_ALLOW_PRIVATE_HTTPS_IMPORT") == "1":
        return addresses
    for address in addresses:
        if not address.is_global:
            raise CloudSourceError(
                "HTTPS download resolved to a loopback, private, link-local, "
                "reserved, or otherwise non-public address. An administrator "
                "may opt in to an approved institutional endpoint with "
                "SIFT_ALLOW_PRIVATE_HTTPS_IMPORT=1."
            )
    return addresses


def _validate_https_endpoint(uri: str) -> None:
    _validated_https_addresses(uri)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection whose TCP peer is one prevalidated numeric address.

    ``HTTPSConnection`` retains the original hostname in ``self.host`` and
    therefore still performs certificate hostname verification and sends that
    hostname as TLS SNI. Only its TCP connection factory is replaced, closing
    the otherwise unavoidable second DNS lookup in the standard client.
    """

    def __init__(
        self,
        host: str,
        *,
        validated_addresses: tuple[
            ipaddress.IPv4Address | ipaddress.IPv6Address, ...
        ],
        expected_origin: tuple[str, str, int],
        transfer_guard: _TransferGuard | None = None,
        **kwargs: Any,
    ) -> None:
        request_origin = _https_origin(f"https://{host}")
        if request_origin != expected_origin:
            raise CloudSourceError(
                "HTTPS request origin changed after endpoint validation"
            )
        if not validated_addresses:
            raise CloudSourceError("HTTPS endpoint has no validated address")
        self._validated_addresses = validated_addresses
        self._transfer_guard = transfer_guard
        super().__init__(host, **kwargs)
        if transfer_guard is not None:
            transfer_guard.register(self.close)
        # HTTPConnection.connect calls this hook before HTTPSConnection wraps
        # the resulting socket with TLS using ``self.host`` as server_hostname.
        self._create_connection = self._connect_validated

    def _connect_validated(
        self,
        _address: tuple[str, int],
        timeout: float | object = _SOCKET_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        numeric_timeout = (
            float(timeout)
            if isinstance(timeout, (int, float)) and timeout is not None
            else None
        )
        ends_at = (
            time.monotonic() + numeric_timeout
            if numeric_timeout is not None
            else None
        )
        last_error: OSError | None = None
        for address in self._validated_addresses:
            if self._transfer_guard is not None:
                self._transfer_guard.check()
            family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            sock = socket.socket(family, socket.SOCK_STREAM)
            if self._transfer_guard is not None:
                self._transfer_guard.register(sock.close)
            try:
                if ends_at is not None:
                    remaining = ends_at - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("validated HTTPS connection timed out")
                    sock.settimeout(remaining)
                elif timeout is None:
                    sock.settimeout(None)
                if source_address is not None:
                    sock.bind(source_address)
                peer: tuple[Any, ...] = (
                    (str(address), self.port, 0, 0)
                    if address.version == 6
                    else (str(address), self.port)
                )
                sock.connect(peer)
                if self._transfer_guard is not None:
                    self._transfer_guard.check()
                return sock
            except IntegrationError:
                sock.close()
                raise
            except OSError as e:
                last_error = e
                sock.close()
        raise OSError("could not connect to the validated HTTPS endpoint") from last_error


class _PinnedHTTPSHandler(HTTPSHandler):
    """Resolve and pin each initial, redirected, and resumed HTTPS request."""

    def __init__(self, transfer_guard: _TransferGuard | None = None) -> None:
        super().__init__()
        self._transfer_guard = transfer_guard

    def https_open(self, req):
        target = req.get_full_url()
        origin = _https_origin(target)
        addresses = _validated_https_addresses(target)

        def connection_factory(host: str, **kwargs: Any):
            return _PinnedHTTPSConnection(
                host,
                validated_addresses=addresses,
                expected_origin=origin,
                transfer_guard=self._transfer_guard,
                **kwargs,
            )

        return self.do_open(
            connection_factory,
            req,
            context=self._context,
        )


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Validate every redirect and never carry secrets across origins."""

    def __init__(self, *, require_proxy: bool = False) -> None:
        super().__init__()
        self._require_proxy = require_proxy

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urljoin(req.full_url, newurl)
        _validate_https_endpoint(target)
        if self._require_proxy:
            target_host = _https_origin(target)[1]
            if proxy_bypass(target_host):
                raise CloudSourceError(
                    "trusted HTTPS proxy mode would bypass the proxy for a "
                    "redirect target; remove the bypass or disable trusted proxy mode"
                )
        cross_origin = _https_origin(target) != _https_origin(req.full_url)
        if cross_origin and req.data is not None:
            # redcap_form carries its token in the POST body, where removing
            # headers is insufficient. Dropbox also uses an authenticated POST
            # with a body marker. Never reinterpret or replay either request at
            # a different origin; the researcher must select the final trusted
            # endpoint explicitly.
            raise CloudSourceError(
                "authenticated HTTPS POST redirects may not cross origins"
            )
        redirected = super().redirect_request(
            req, fp, code, msg, headers, target,
        )
        if redirected is not None and cross_origin:
            sensitive = {
                "authorization",
                "x-api-token",
                "dropbox-api-arg",
                "zotero-api-key",
            }
            # urllib's Request.add_header uses ``str.capitalize()`` while
            # remove_header is case-sensitive, so calling remove_header with
            # the conventional all-caps spelling leaves X-API-TOKEN behind.
            # Remove by a case-insensitive scan of both header stores.
            for store in (
                redirected.headers,
                redirected.unredirected_hdrs,
            ):
                for header in tuple(store):
                    if header.casefold() in sensitive:
                        store.pop(header, None)
        return redirected


def _http_chunks(
    uri: str,
    timeout_seconds: float = 60,
    credential: str | None = None,
    auth_mode: str = "bearer",
    form_fields: dict[str, str] | None = None,
    *,
    _guard: _TransferGuard | None = None,
) -> _Download:
    from urllib.request import Request, build_opener

    _validate_https_endpoint(uri)

    headers = {"User-Agent": "Sift/0.1 cloud-import"}
    request_data: bytes | None = None
    if credential:
        if auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {credential}"
        elif auth_mode == "qualtrics":
            headers["X-API-TOKEN"] = credential
        elif auth_mode == "kobo":
            headers["Authorization"] = f"Token {credential}"
        elif auth_mode == "redcap_form":
            fields = dict(form_fields or {})
            fields["token"] = credential
            request_data = urlencode(fields).encode("utf-8")
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        elif auth_mode == "dropbox":
            headers["Authorization"] = f"Bearer {credential}"
            argument = dict(form_fields or {})
            if set(argument) != {"path"} or not argument.get("path"):
                raise CloudSourceError("Dropbox download requires one selected file id")
            headers["Dropbox-API-Arg"] = json.dumps(
                argument, separators=(",", ":"), ensure_ascii=True,
            )
            request_data = b""
        else:
            raise CloudSourceError("unsupported HTTPS authentication mode")
    # Environment proxy variables must not silently move this confidentiality
    # boundary to an intermediary or reintroduce proxy-side DNS rebinding.
    # An administrator can explicitly trust the configured proxy, in which
    # case its routing is the administrator's declared network boundary.
    if os.environ.get("SIFT_TRUST_HTTPS_IMPORT_PROXY") == "1":
        proxy_url = getproxies().get("https")
        if not proxy_url:
            raise CloudSourceError(
                "trusted HTTPS proxy mode requires an explicitly configured "
                "HTTPS proxy; disable SIFT_TRUST_HTTPS_IMPORT_PROXY or configure one"
            )
        target_host = _https_origin(uri)[1]
        if proxy_bypass(target_host):
            raise CloudSourceError(
                "trusted HTTPS proxy mode would bypass the proxy for this host; "
                "remove the bypass or disable trusted proxy mode"
            )
        opener = build_opener(
            ProxyHandler({"https": proxy_url}),
            _SafeRedirectHandler(require_proxy=True),
        )
    else:
        opener = build_opener(
            ProxyHandler({}),
            _PinnedHTTPSHandler(_guard),
            _SafeRedirectHandler(),
        )
    if _guard is not None:
        _guard.register(getattr(opener, "close", None))
        _guard.check()
    timeout = _guarded_timeout(timeout_seconds, _guard)
    response = opener.open(
        Request(uri, data=request_data, headers=headers), timeout=timeout,
    )
    if _guard is not None:
        _guard.register(response.close)
        _guard.check()
    _validate_https_endpoint(response.geturl())
    length = response.headers.get("Content-Length")
    size = int(length) if length and length.isdecimal() else None
    content_type = response.headers.get_content_type()
    resume_validator = response.headers.get("ETag") or response.headers.get(
        "Last-Modified"
    )
    version = _remote_metadata(resume_validator)
    resumable = request_data is None and (
        str(response.headers.get("Accept-Ranges", "")).casefold() == "bytes"
        and bool(resume_validator)
    )
    active_response = [response]

    def chunks() -> Iterator[bytes]:
        offset = 0
        attempts = 0
        try:
            while True:
                try:
                    while True:
                        if _guard is not None:
                            _guard.check()
                        # HTTPResponse.read(n) may wait for all n bytes while a
                        # peer drip-feeds indefinitely. read1 performs at most
                        # one underlying buffered read, allowing the outer
                        # wall-clock deadline to be checked after every arrival.
                        read = getattr(active_response[0], "read1", None)
                        data = (
                            read(DOWNLOAD_CHUNK_BYTES)
                            if callable(read)
                            else active_response[0].read(DOWNLOAD_CHUNK_BYTES)
                        )
                        if not data:
                            if _guard is not None:
                                _guard.check()
                            return
                        if _guard is not None:
                            _guard.check()
                        offset += len(data)
                        yield data
                except Exception:
                    active_response[0].close()
                    if _guard is not None:
                        _guard.check()
                    if (
                        not resumable
                        or offset <= 0
                        or attempts >= MAX_RESUME_ATTEMPTS
                    ):
                        raise
                    attempts += 1
                    resumed_headers = dict(headers)
                    resumed_headers["Range"] = f"bytes={offset}-"
                    resumed_headers["If-Range"] = str(resume_validator)
                    if _guard is not None:
                        _guard.check()
                    resumed = opener.open(
                        Request(uri, headers=resumed_headers),
                        timeout=_guarded_timeout(timeout_seconds, _guard),
                    )
                    if _guard is not None:
                        _guard.register(resumed.close)
                        _guard.check()
                    _validate_https_endpoint(resumed.geturl())
                    content_range = str(resumed.headers.get("Content-Range") or "")
                    if resumed.getcode() != 206 or not content_range.startswith(
                        f"bytes {offset}-"
                    ):
                        resumed.close()
                        raise CloudSourceError(
                            "HTTPS server did not honor the requested safe resume offset"
                        )
                    active_response[0] = resumed
        finally:
            active_response[0].close()

    return _DownloadInfo(
        chunks(), size, version, _remote_metadata(content_type),
        lambda: active_response[0].close(),
        identifiers={"url": redact_source_uri(response.geturl())},
        resumable=resumable,
    )


def _s3_chunks(
    uri: str,
    timeout_seconds: float = 60,
    credential: str | None = None,
    *,
    _guard: _TransferGuard | None = None,
) -> _Download:
    del credential  # AWS's default credential chain/roles are the boundary.
    try:
        import boto3
    except ImportError as e:
        raise CloudSourceError(
            "Amazon S3 support is not installed. "
            + _cloud_source_install_guidance("s3")
        ) from e
    bucket, key, query = _object_parts(uri, "s3")
    if "versionId" in query and not query["versionId"]:
        raise CloudSourceError("S3 versionId cannot be empty")
    kwargs: dict[str, Any] = {"Bucket": bucket, "Key": key}
    if query.get("versionId"):
        kwargs["VersionId"] = query["versionId"]
    if query.get("requestPayer"):
        if query["requestPayer"] != "requester":
            raise CloudSourceError("S3 requestPayer must be exactly 'requester'")
        kwargs["RequestPayer"] = "requester"
    kwargs["ChecksumMode"] = "ENABLED"
    timeout = _guarded_timeout(timeout_seconds, _guard)
    try:
        from botocore.config import Config

        config = Config(
            connect_timeout=timeout,
            read_timeout=timeout,
            retries={"max_attempts": 0, "mode": "standard"},
        )
        try:
            client = boto3.client("s3", config=config)
        except TypeError:
            # Lightweight third-party adapters written against the historical
            # one-argument factory remain usable; the official boto3 factory
            # accepts ``config``.
            client = boto3.client("s3")
    except ImportError:
        client = boto3.client("s3")
    if _guard is not None:
        _guard.register(getattr(client, "close", None))
        _guard.check()
    response = client.get_object(**kwargs)
    body = response["Body"]
    if _guard is not None:
        _guard.register(body.close)
        _guard.check()
    active_body = [body]
    etag = response.get("ETag")

    def chunks() -> Iterator[bytes]:
        offset = 0
        attempts = 0
        try:
            while True:
                try:
                    for chunk in active_body[0].iter_chunks(
                        chunk_size=DOWNLOAD_CHUNK_BYTES,
                    ):
                        if _guard is not None:
                            _guard.check()
                        if chunk:
                            offset += len(chunk)
                            yield chunk
                    return
                except Exception:
                    active_body[0].close()
                    if _guard is not None:
                        _guard.check()
                    if offset <= 0 or attempts >= MAX_RESUME_ATTEMPTS:
                        raise
                    attempts += 1
                    resumed_kwargs = dict(kwargs)
                    resumed_kwargs["Range"] = f"bytes={offset}-"
                    if etag and "VersionId" not in resumed_kwargs:
                        resumed_kwargs["IfMatch"] = etag
                    if _guard is not None:
                        _guard.check()
                    resumed = client.get_object(**resumed_kwargs)
                    if _guard is not None:
                        _guard.register(resumed["Body"].close)
                        _guard.check()
                    content_range = str(resumed.get("ContentRange") or "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        resumed["Body"].close()
                        raise CloudSourceError(
                            "S3 did not honor the requested safe resume offset"
                        )
                    active_body[0] = resumed["Body"]
        finally:
            active_body[0].close()

    checksum = (
        response.get("ChecksumSHA256")
        if response.get("ChecksumType") in {None, "FULL_OBJECT"}
        else None
    )
    return _DownloadInfo(
        chunks=chunks(),
        expected_size=(
            int(response["ContentLength"])
            if response.get("ContentLength") is not None else None
        ),
        version=_remote_metadata(response.get("VersionId") or response.get("ETag")),
        content_type=_remote_metadata(response.get("ContentType")),
        close=lambda: _close_resources(active_body[0], client),
        checksum_algorithm="sha256" if checksum else None,
        checksum_base64=_remote_metadata(checksum),
        identifiers={
            "bucket": bucket,
            "object": key,
            **({"version_id": str(response["VersionId"])} if response.get("VersionId") else {}),
            **({"etag": str(response["ETag"])} if response.get("ETag") else {}),
        },
        resumable=True,
    )


def _gcs_chunks(
    uri: str,
    timeout_seconds: float = 60,
    credential: str | None = None,
    *,
    _guard: _TransferGuard | None = None,
) -> _Download:
    del credential  # ADC/workload identity only.
    try:
        import google.cloud.storage as storage
    except ImportError as e:
        raise CloudSourceError(
            "Google Cloud Storage support is not installed. "
            + _cloud_source_install_guidance("gcs")
        ) from e
    bucket_name, key, query = _object_parts(uri, "gcs")
    if (
        "generation" in query
        and (
            not query["generation"].isdigit()
            or int(query["generation"]) <= 0
        )
    ):
        raise CloudSourceError("GCS generation must be a positive integer")
    client = storage.Client()
    if _guard is not None:
        _guard.register(getattr(client, "close", None))
        _guard.check()
    blob = client.bucket(bucket_name).blob(
        key,
        generation=int(query["generation"]) if query.get("generation", "").isdigit() else None,
    )
    timeout = _guarded_timeout(timeout_seconds, _guard)
    _call_supported_kwargs(blob.reload, timeout=timeout, retry=None)
    if _guard is not None:
        _guard.check()
    reader = _call_supported_kwargs(
        blob.open,
        "rb", chunk_size=DOWNLOAD_CHUNK_BYTES, timeout=timeout, retry=None,
    )
    if _guard is not None:
        _guard.register(reader.close)
        _guard.check()

    def chunks() -> Iterator[bytes]:
        try:
            while True:
                if _guard is not None:
                    _guard.check()
                data = reader.read(DOWNLOAD_CHUNK_BYTES)
                if not data:
                    return
                yield data
        finally:
            reader.close()

    checksum_algorithm = "crc32c" if blob.crc32c else "md5" if blob.md5_hash else None
    checksum = blob.crc32c or blob.md5_hash
    return _DownloadInfo(
        chunks(), blob.size,
        _remote_metadata(blob.generation or blob.etag),
        _remote_metadata(blob.content_type),
        lambda: _close_resources(reader, client),
        checksum_algorithm=checksum_algorithm,
        checksum_base64=_remote_metadata(checksum),
        identifiers={
            "bucket": bucket_name,
            "object": key,
            **({"generation": str(blob.generation)} if blob.generation else {}),
            **({"etag": str(blob.etag)} if blob.etag else {}),
        },
        # The GCS Blob reader issues bounded range requests with SDK retries.
        resumable=True,
    )


def _azure_chunks(
    uri: str,
    timeout_seconds: float = 60,
    credential: str | None = None,
    *,
    _guard: _TransferGuard | None = None,
) -> _Download:
    try:
        from azure.identity import DefaultAzureCredential
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise CloudSourceError(
            "Azure Blob support is not installed. "
            + _cloud_source_install_guidance("azure_blob")
        ) from e
    account, path, query = _object_parts(uri, "azure_blob")
    container, blob_name = path.split("/", 1)
    if query.get("versionid") and query.get("snapshot"):
        raise CloudSourceError("select either an Azure blob version or snapshot, not both")
    azure_credential: Any = credential or DefaultAzureCredential()
    timeout = _guarded_timeout(timeout_seconds, _guard)
    server_timeout = max(1, int(timeout))
    service = BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=azure_credential,
        connection_timeout=timeout,
        read_timeout=timeout,
        retry_total=0,
    )
    if _guard is not None:
        _guard.register(getattr(azure_credential, "close", None))
        _guard.register(getattr(service, "close", None))
        _guard.check()
    blob = service.get_blob_client(
        container=container,
        blob=blob_name,
        version_id=query.get("versionid") or None,
        snapshot=query.get("snapshot") or None,
    )
    properties = _call_supported_kwargs(
        blob.get_blob_properties, timeout=server_timeout,
    )
    if _guard is not None:
        _guard.check()
    downloader = _call_supported_kwargs(
        blob.download_blob,
        max_concurrency=1, validate_content=True, timeout=server_timeout,
    )
    if _guard is not None:
        _guard.register(getattr(downloader, "close", None))
        _guard.check()

    def chunks() -> Iterator[bytes]:
        try:
            for chunk in downloader.chunks():
                if _guard is not None:
                    _guard.check()
                yield chunk
        finally:
            close_downloader = getattr(downloader, "close", None)
            if callable(close_downloader):
                close_downloader()
            close_service = getattr(service, "close", None)
            if callable(close_service):
                close_service()
            close = getattr(azure_credential, "close", None)
            if callable(close):
                close()

    def close() -> None:
        close_downloader = getattr(downloader, "close", None)
        if callable(close_downloader):
            close_downloader()
        close_service = getattr(service, "close", None)
        if callable(close_service):
            close_service()
        close_credential = getattr(azure_credential, "close", None)
        if callable(close_credential):
            close_credential()
    md5 = getattr(properties.content_settings, "content_md5", None)
    return _DownloadInfo(
        chunks(),
        int(properties.size) if properties.size is not None else None,
        _remote_metadata(properties.version_id or properties.etag),
        _remote_metadata(getattr(properties.content_settings, "content_type", None)),
        close,
        checksum_algorithm="md5" if md5 else None,
        checksum_base64=(base64.b64encode(md5).decode("ascii") if md5 else None),
        identifiers={
            "account": account,
            "container": container,
            "object": blob_name,
            **({"version_id": str(properties.version_id)} if properties.version_id else {}),
            **({"etag": str(properties.etag)} if properties.etag else {}),
            **({"snapshot": str(query["snapshot"])} if query.get("snapshot") else {}),
        },
        resumable=True,
    )


def _sftp_chunks(
    uri: str,
    timeout_seconds: float = 60,
    credential: str | None = None,
    *,
    _guard: _TransferGuard | None = None,
) -> _Download:
    try:
        import paramiko
    except ImportError as e:
        raise CloudSourceError(
            "SFTP support is not installed. "
            + _cloud_source_install_guidance("sftp")
        ) from e
    if not credential:
        raise CloudSourceError(
            "SFTP requires a vault-backed key profile with private_key and known_hosts"
        )
    try:
        config = json.loads(credential)
    except (TypeError, ValueError) as e:
        raise CloudSourceError("the SFTP key profile is invalid") from e
    if not isinstance(config, dict):
        raise CloudSourceError("the SFTP key profile is invalid")
    parsed = urlsplit(uri)
    if parsed.query or parsed.fragment or parsed.password:
        raise CloudSourceError("SFTP URIs cannot contain passwords, query, or fragments")
    username = parsed.username or str(config.get("username") or "")
    if not username:
        raise CloudSourceError("SFTP requires a username in the URI or key profile")
    remote_path = unquote(parsed.path)
    if not remote_path.startswith("/") or not Path(remote_path).name:
        raise CloudSourceError("SFTP requires an absolute remote object path")

    private_key = Path(str(config.get("private_key") or "")).expanduser()
    known_hosts = Path(str(config.get("known_hosts") or "")).expanduser()
    for label, path in (("private key", private_key), ("known_hosts", known_hosts)):
        if not path.is_file() or path.is_symlink():
            raise CloudSourceError(f"SFTP {label} must be a regular non-symlink file")
    if os.name != "nt" and private_key.stat().st_mode & 0o077:
        raise CloudSourceError("SFTP private key permissions must not allow group/other access")

    client = paramiko.SSHClient()
    if _guard is not None:
        _guard.register(client.close)
        _guard.check()
    client.load_host_keys(str(known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    timeout = _guarded_timeout(timeout_seconds, _guard)
    try:
        client.connect(
            hostname=parsed.hostname,
            port=parsed.port or 22,
            username=username,
            key_filename=str(private_key),
            passphrase=str(config.get("passphrase") or "") or None,
            allow_agent=False,
            look_for_keys=False,
            timeout=timeout,
            banner_timeout=timeout,
            auth_timeout=timeout,
        )
        if _guard is not None:
            transport = client.get_transport()
            _guard.register(getattr(transport, "close", None))
            _guard.check()
        sftp = client.open_sftp()
        if _guard is not None:
            _guard.register(sftp.close)
            _guard.check()
        stat = sftp.stat(remote_path)
        if _guard is not None:
            _guard.check()
        remote = sftp.open(remote_path, "rb")
        settimeout = getattr(remote, "settimeout", None)
        if callable(settimeout):
            settimeout(timeout)
        if _guard is not None:
            _guard.register(remote.close)
            _guard.check()
    except Exception:
        client.close()
        if _guard is not None:
            _guard.check()
        raise

    def close() -> None:
        try:
            remote.close()
        finally:
            try:
                sftp.close()
            finally:
                client.close()

    def chunks() -> Iterator[bytes]:
        try:
            while True:
                if _guard is not None:
                    _guard.check()
                data = remote.read(DOWNLOAD_CHUNK_BYTES)
                if _guard is not None:
                    _guard.check()
                if not data:
                    return
                yield data
        finally:
            close()

    return _DownloadInfo(
        chunks=chunks(),
        expected_size=int(stat.st_size),
        version=_remote_metadata(f"mtime:{int(stat.st_mtime)};size:{int(stat.st_size)}"),
        content_type=None,
        close=close,
        identifiers={
            "host": str(parsed.hostname),
            "object": remote_path,
            "host_key_policy": "reject_unknown",
        },
        resumable=False,
    )


def import_cloud_dataset(
    cwd: Path,
    *,
    uri: str,
    dataset_name: str = "",
    validate_dataset: bool = True,
    timeout_seconds: float = 300,
    cancellation: CancellationToken | None = None,
    credential_profile: str | None = None,
    _credential_kind: str | None = None,
    _https_auth_mode: str = "bearer",
    _https_form_fields: dict[str, str] | None = None,
    _expected_checksum: str | None = None,
) -> CloudImportResult:
    """Download one researcher-selected cloud object into the session."""
    cwd = _safe_session_directory(Path(cwd))
    uri = uri.strip() if isinstance(uri, str) else ""
    try:
        deadline = Deadline(timeout_seconds)
    except ValueError as e:
        raise CloudSourceError(
            "cloud import timeout must be between 0 and 86400 seconds",
            code="invalid_timeout",
        ) from e
    try:
        deadline.check(cancellation)
    except IntegrationCancelled as e:
        raise CloudSourceError(
            "cloud import cancelled", code="cancelled",
            action="Start a new import when ready.",
        ) from e
    kind = describe_cloud_source(uri)
    from sift import enterprise_policy

    enterprise = enterprise_policy.load_enterprise_policy()
    if not enterprise_policy.cloud_source_allowed(
        kind,
        enterprise,
    ):
        raise CloudSourceError(
            f"cloud source {kind!r} is blocked by enterprise policy"
        )
    if not enterprise_policy.integration_endpoint_allowed(uri, enterprise):
        raise CloudSourceError("cloud source endpoint is blocked by enterprise policy")
    filename = _validated_filename(uri, dataset_name)
    display = redact_source_uri(uri)
    limit = cloud_import_max_bytes()
    factories: dict[str, Callable[..., _Download]] = {
        "https": _http_chunks,
        "s3": _s3_chunks,
        "gcs": _gcs_chunks,
        "azure_blob": _azure_chunks,
        "sftp": _sftp_chunks,
    }
    credential: str | None = None
    if credential_profile:
        credential_kind = _credential_kind or {
            "https": "https_bearer",
            "azure_blob": "azure_sas",
            "sftp": "sftp_key",
        }.get(kind)
        if _credential_kind and not (
            kind == "https" and _credential_kind == "research_token"
        ):
            raise CloudSourceError("credential kind is not valid for this source")
        if credential_kind is None:
            raise CloudSourceError(
                f"{kind} uses its managed/default identity chain and does not "
                "accept a Sift credential profile"
            )
        try:
            from sift.remote_credentials import (
                RemoteCredentialError,
                resolve_remote_credential,
            )

            credential = resolve_remote_credential(
                credential_profile, credential_kind,
            )
        except RemoteCredentialError as e:
            raise CloudSourceError(str(e), code="credential_unavailable") from e
    guard = _TransferGuard(deadline, cancellation)
    info: _DownloadInfo | None = None
    try:
        if kind == "https" and credential is not None:
            if _https_auth_mode == "bearer" and _https_form_fields is None:
                opened = _call_with_guard(
                    _http_chunks, uri, deadline.remaining, credential,
                    guard=guard,
                )
            else:
                opened = _call_with_guard(
                    _http_chunks,
                    uri, deadline.remaining, credential, _https_auth_mode,
                    _https_form_fields, guard=guard,
                )
        else:
            opened = (
                _call_with_guard(
                    factories[kind], uri, deadline.remaining, credential,
                    guard=guard,
                )
                if credential is not None or kind == "sftp"
                else _call_with_guard(
                    factories[kind], uri, deadline.remaining, guard=guard,
                )
            )
        info = _download_info(opened)
        guard.register(info.close)
        guard.check()
        try:
            _guard_expected_size(cwd, info.expected_size, limit)
            _validate_content_type(filename, info.content_type)
        except Exception:
            info.close()
            raise
    except CloudSourceError:
        if info is None:
            guard.abort()
        else:
            guard.finish()
        raise
    except IntegrationCancelled as e:
        if info is None:
            guard.abort()
        else:
            guard.finish()
        raise CloudSourceError(
            "cloud import cancelled", code="cancelled",
            action="Start a new import when ready.",
        ) from e
    except IntegrationDeadlineExceeded as e:
        if info is None:
            guard.abort()
        else:
            guard.finish()
        raise CloudSourceError(
            f"cloud import exceeded its {timeout_seconds:g}-second timeout",
            code="deadline_exceeded", retryable=True,
            action="Increase the bounded timeout or select a smaller object.",
        ) from e
    except Exception as e:
        try:
            guard.check()
        except IntegrationCancelled as cancelled:
            guard.finish()
            raise CloudSourceError(
                "cloud import cancelled", code="cancelled",
                action="Start a new import when ready.",
            ) from cancelled
        except IntegrationDeadlineExceeded as exceeded:
            guard.finish()
            raise CloudSourceError(
                f"cloud import exceeded its {timeout_seconds:g}-second timeout",
                code="deadline_exceeded", retryable=True,
                action="Increase the bounded timeout or select a smaller object.",
            ) from exceeded
        if info is None:
            guard.abort()
        else:
            guard.finish()
        raise CloudSourceError(
            f"could not open {display}: {_safe_error(e, uri, credential)}"
        ) from e

    fd, tmp_name = tempfile.mkstemp(
        prefix=".sift-cloud-", suffix=Path(filename).suffix, dir=cwd,
    )
    target: Path | None = None
    metadata_sidecar: Path | None = None
    canonical_manifest: dict[str, Any] | None = None
    release_recorded = False
    succeeded = False
    cleanup_complete = True
    try:
        try:
            with os.fdopen(fd, "wb") as handle:
                downloaded, digest, computed_checksums = _copy_chunks(
                    info.chunks,
                    handle,
                    limit=limit,
                    disk_directory=cwd,
                    cancellation=cancellation,
                    deadline=deadline,
                )
        finally:
            info.close()
        guard.check()
        if info.expected_size is not None and downloaded != info.expected_size:
            raise CloudSourceError(
                f"cloud object changed or ended early: expected {info.expected_size} bytes, "
                f"received {downloaded}"
            )
        temporary = Path(tmp_name)
        integrity_checksum = _verify_expected_checksum(
            temporary, _expected_checksum, guard=guard,
        )
        checksum_verified = False
        remote_checksum: str | None = None
        if info.checksum_algorithm and info.checksum_base64:
            remote_checksum = (
                f"{info.checksum_algorithm}:{info.checksum_base64}"
            )
            computed = computed_checksums.get(info.checksum_algorithm)
            if computed is None:
                raise CloudSourceError(
                    f"cannot verify the provider's {info.checksum_algorithm} checksum"
                )
            if not hmac.compare_digest(computed, info.checksum_base64):
                raise CloudSourceError(
                    "remote object checksum did not match the downloaded bytes"
                )
            checksum_verified = True
        _reject_archive_payload(temporary, limit=limit)
        guard.check()
        if validate_dataset:
            from sift.schema import extract

            extract(temporary, "names_only")
            guard.check()
        target = _reserve_target(cwd, filename)
        if target.parent.resolve() != cwd:
            target.unlink(missing_ok=True)
            raise CloudSourceError("cloud import target escaped the session directory")
        os.replace(tmp_name, target)
        metadata_sidecar = target.with_suffix(target.suffix + ".metadata.json")
        from sift.reliability import atomic_write_json
        atomic_write_json(metadata_sidecar, {
            "format": "cloud_import",
            "source_kind": kind,
            "remote_version": info.version,
            "remote_identifiers": info.identifiers or {},
        })
        if validate_dataset:
            try:
                from sift.canonical_dataset import ensure_manifest
                canonical_manifest = ensure_manifest(
                    cwd,
                    target,
                    selection={
                        "source_kind": kind,
                        "remote_version": info.version,
                        "remote_identifiers": info.identifiers or {},
                    },
                    transformations=({
                        "operation": "cloud_import",
                        "runtime": kind,
                    },),
                )
            except Exception as e:
                raise CloudSourceError(
                    f"could not establish canonical cloud dataset identity: {type(e).__name__}"
                ) from e
        from sift import release_ledger

        recorded = release_ledger.record_release(
            cwd,
            kind="local_ingestion",
            tool="(cloud object import)",
            extra={
                "dataset": target.name,
                "source_kind": kind,
                "source": display,
                "bytes": downloaded,
                "dataset_sha256": digest,
                "remote_version": info.version,
                "content_type": info.content_type,
                "remote_checksum": remote_checksum,
                "checksum_verified": checksum_verified,
                "integrity_checksum": integrity_checksum,
                "remote_identifiers": info.identifiers or {},
                "resumable_protocol": info.resumable,
                "canonical_fingerprint": (
                    canonical_manifest["fingerprint"]
                    if canonical_manifest is not None else None
                ),
            },
        )
        if not recorded:
            raise CloudSourceError("could not record cloud import provenance")
        release_recorded = True
        from sift.integration_audit import record_integration_event

        if not record_integration_event(
            cwd,
            integration_id=kind,
            kind="object_storage",
            action="materialize",
            outcome="success",
            metadata={"bytes": downloaded},
        ):
            raise CloudSourceError("could not record cloud import audit event")
        succeeded = True
        return CloudImportResult(
            dataset_path=target,
            source_kind=kind,
            source_display=display,
            bytes_downloaded=downloaded,
            dataset_sha256=digest,
            remote_version=info.version,
            content_type=info.content_type,
            remote_checksum=remote_checksum,
            checksum_verified=checksum_verified,
            remote_identifiers=dict(info.identifiers or {}),
            canonical_fingerprint=(
                canonical_manifest["fingerprint"]
                if canonical_manifest is not None else None
            ),
            integrity_checksum=integrity_checksum,
        )
    except CloudSourceError:
        raise
    except IntegrationCancelled as e:
        raise CloudSourceError(
            "cloud import cancelled", code="cancelled",
            action="Start a new import when ready.",
        ) from e
    except IntegrationDeadlineExceeded as e:
        raise CloudSourceError(
            f"cloud import exceeded its {timeout_seconds:g}-second timeout",
            code="deadline_exceeded", retryable=True,
            action="Increase the bounded timeout or select a smaller object.",
        ) from e
    except Exception as e:
        try:
            guard.check()
        except IntegrationCancelled as cancelled:
            raise CloudSourceError(
                "cloud import cancelled", code="cancelled",
                action="Start a new import when ready.",
            ) from cancelled
        except IntegrationDeadlineExceeded as exceeded:
            raise CloudSourceError(
                f"cloud import exceeded its {timeout_seconds:g}-second timeout",
                code="deadline_exceeded", retryable=True,
                action="Increase the bounded timeout or select a smaller object.",
            ) from exceeded
        raise CloudSourceError(
            f"cloud import failed: {_safe_error(e, uri, credential)}"
        ) from e
    finally:
        guard.finish()
        Path(tmp_name).unlink(missing_ok=True)
        if target is not None and not succeeded and not release_recorded:
            if canonical_manifest is not None:
                try:
                    from sift.canonical_dataset import discard_uncommitted_manifest

                    cleanup_complete = discard_uncommitted_manifest(
                        cwd, target, str(canonical_manifest["fingerprint"]),
                    )
                except Exception:
                    cleanup_complete = False
            try:
                target.unlink(missing_ok=True)
            except OSError:
                cleanup_complete = False
        if metadata_sidecar is not None and not succeeded and not release_recorded:
            try:
                metadata_sidecar.unlink(missing_ok=True)
            except OSError:
                cleanup_complete = False
        if not cleanup_complete:
            raise CloudSourceError(
                "cloud import failed and confidential local cleanup was "
                "incomplete; close any program using the dataset and retry "
                "removal from .sift/datasets"
            )


__all__ = [
    "DEFAULT_CLOUD_IMPORT_MAX_BYTES",
    "CloudImportResult",
    "CloudSourceError",
    "cloud_import_max_bytes",
    "describe_cloud_source",
    "import_cloud_dataset",
    "redact_source_uri",
]
