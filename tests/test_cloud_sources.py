from __future__ import annotations

import hashlib
import base64
import io
import ipaddress
import json
import socket
import sys
import threading
import time
import zipfile
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request

import pandas as pd
import pytest

from sift import cloud_sources
from sift.cloud_sources import CloudSourceError
from sift.integrations import CLOUD_SOURCE_ADAPTERS


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://bucket/path/data.parquet", "s3"),
        ("gs://bucket/path/data.csv", "gcs"),
        ("az://account/container/data.dta", "azure_blob"),
        ("azure://account/container/data.dta", "azure_blob"),
        ("https://files.example/data.csv?sig=secret", "https"),
        ("sftp://alice@files.example/data.csv", "sftp"),
    ],
)
def test_cloud_source_detection(uri: str, expected: str) -> None:
    assert cloud_sources.describe_cloud_source(uri) == expected


def test_cloud_filename_limit_keeps_validated_extension() -> None:
    name = cloud_sources._validated_filename(
        f"https://files.example/{'x' * 400}.parquet", None,
    )
    assert len(name) == 160
    assert name.endswith(".parquet")


def test_cloud_filename_neutralizes_windows_device_name() -> None:
    assert cloud_sources._validated_filename(
        "https://files.example/CON.csv", None,
    ) == "_CON.csv"


@pytest.mark.parametrize(
    "kind",
    tuple(
        adapter.id
        for adapter in CLOUD_SOURCE_ADAPTERS
        if adapter.install_extra != "built-in"
    ),
)
def test_missing_cloud_adapter_guidance_uses_catalog_extra(kind: str) -> None:
    adapter = next(item for item in CLOUD_SOURCE_ADAPTERS if item.id == kind)
    message = cloud_sources._cloud_source_install_guidance(kind)
    assert adapter.label in message
    assert f'sift[{adapter.install_extra}]' in message


@pytest.mark.parametrize(
    ("kind", "modules", "operation"),
    (
        (
            "s3",
            ("boto3",),
            lambda: cloud_sources._s3_chunks("s3://bucket/research.csv"),
        ),
        (
            "gcs",
            ("google.cloud.storage",),
            lambda: cloud_sources._gcs_chunks("gs://bucket/research.csv"),
        ),
        (
            "azure_blob",
            ("azure.identity", "azure.storage.blob"),
            lambda: cloud_sources._azure_chunks(
                "az://account/container/research.csv",
            ),
        ),
        (
            "sftp",
            ("paramiko",),
            lambda: cloud_sources._sftp_chunks("sftp://user@host/research.csv"),
        ),
    ),
)
def test_missing_cloud_driver_paths_expose_exact_safe_install_command(
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    modules: tuple[str, ...],
    operation,
) -> None:
    """The real import failures must retain the catalog recovery contract."""
    for module in modules:
        monkeypatch.setitem(sys.modules, module, None)
    adapter = next(item for item in CLOUD_SOURCE_ADAPTERS if item.id == kind)
    with pytest.raises(CloudSourceError) as raised:
        operation()
    message = str(raised.value)
    assert f'sift[{adapter.install_extra}]' in message
    assert adapter.label in message
    assert "research.csv" not in message


@pytest.mark.parametrize(
    "uri",
    [
        "http://files.example/data.csv",
        "ftp://files.example/data.csv",
        "file:///tmp/data.csv",
        "s3:///data.csv",
        "https://user:secret@files.example/data.csv",
        "sftp://user:secret@files.example/data.csv",
    ],
)
def test_unsupported_or_credentialed_sources_fail_closed(uri: str) -> None:
    with pytest.raises(CloudSourceError):
        cloud_sources.describe_cloud_source(uri)


def test_signed_url_display_removes_all_query_credentials() -> None:
    uri = (
        "https://files.example/research/data.csv?X-Amz-Credential=alice"
        "&X-Amz-Signature=highly-secret&X-Amz-Security-Token=token-secret"
    )
    shown = cloud_sources.redact_source_uri(uri)
    assert shown == "https://files.example/research/data.csv"
    assert "secret" not in shown and "alice" not in shown


@pytest.mark.parametrize(
    "uri,kind",
    [
        ("s3://bucket/data.csv?access_key=secret", "s3"),
        ("gs://bucket/data.csv?token=secret", "gcs"),
        ("az://account/container/data.csv?sig=secret", "azure_blob"),
    ],
)
def test_native_cloud_uris_reject_embedded_credentials(
    uri: str,
    kind: str,
) -> None:
    with pytest.raises(CloudSourceError, match="external identity"):
        cloud_sources._object_parts(uri, kind)


def _parquet_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "source.parquet"
    pd.DataFrame({"id": [1, 2], "value": [3.0, 4.0]}).to_parquet(path)
    return path.read_bytes()


def test_https_import_streams_validates_hashes_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _parquet_bytes(tmp_path)
    (tmp_path / "source.parquet").unlink()
    closed: list[bool] = []

    def fake(_uri: str, _timeout: float):
        return (
            iter((payload[:17], payload[17:])),
            len(payload),
            '"etag-1"',
            "application/octet-stream",
            lambda: closed.append(True),
        )

    monkeypatch.setattr(cloud_sources, "_http_chunks", fake)
    result = cloud_sources.import_cloud_dataset(
        tmp_path,
        uri="https://files.example/source.parquet?sig=do-not-record",
    )
    assert result.dataset_path.read_bytes() == payload
    assert result.dataset_sha256 == hashlib.sha256(payload).hexdigest()
    assert result.source_display == "https://files.example/source.parquet"
    assert closed == [True]

    from sift.tools import _canonicalize_analysis_sources
    sources, error = _canonicalize_analysis_sources(
        tmp_path, (result.dataset_path.name,),
    )
    assert error is None
    assert sources[0]["selection"]["source_kind"] == "https"
    assert sources[0]["selection"]["remote_version"] == '"etag-1"'

    from sift.release_ledger import read_ledger

    record = read_ledger(tmp_path)[0]
    assert record["kind"] == "local_ingestion"
    assert record["extra"]["dataset_sha256"] == result.dataset_sha256
    assert record["extra"]["canonical_fingerprint"] == result.canonical_fingerprint
    assert "do-not-record" not in str(record)


def test_declared_oversize_closes_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(cloud_sources, "cloud_import_max_bytes", lambda: 10)
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (iter((b"x",)), 11, None, None, lambda: closed.append(True)),
    )
    with pytest.raises(CloudSourceError, match="larger"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv?sig=secret",
        )
    assert closed == [True]
    assert not list(tmp_path.glob(".sift-cloud-*"))


def test_actual_oversize_removes_partial_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(cloud_sources, "cloud_import_max_bytes", lambda: 3)
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (iter((b"ab", b"cd")), None, None, None, lambda: None),
    )
    with pytest.raises(CloudSourceError, match="exceeded"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
        )
    assert not list(tmp_path.glob(".sift-cloud-*"))
    assert not (tmp_path / "data.csv").exists()


def test_parse_failure_and_provenance_failure_leave_no_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (iter((b"not parquet",)), 11, None, None, lambda: None),
    )
    with pytest.raises(CloudSourceError, match="cloud import failed"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/broken.parquet",
        )
    assert not (tmp_path / "broken.parquet").exists()

    payload = _parquet_bytes(tmp_path)
    (tmp_path / "source.parquet").unlink()
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (iter((payload,)), len(payload), None, None, lambda: None),
    )
    from sift import release_ledger

    monkeypatch.setattr(release_ledger, "record_release", lambda *a, **k: False)
    with pytest.raises(CloudSourceError, match="provenance"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/source.parquet",
        )
    assert not (tmp_path / "source.parquet").exists()
    canonical = tmp_path / ".sift" / "datasets"
    assert not list((canonical / "manifests").glob("*.json"))
    assert not list((canonical / "paths").glob("*.json"))
    assert not list((canonical / "snapshots").rglob("*.parquet"))


def test_cloud_import_surfaces_incomplete_confidential_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _parquet_bytes(tmp_path)
    (tmp_path / "source.parquet").unlink()
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            iter((payload,)), len(payload), None, None, lambda: None,
        ),
    )
    from sift import canonical_dataset, release_ledger

    monkeypatch.setattr(release_ledger, "record_release", lambda *a, **k: False)
    monkeypatch.setattr(
        canonical_dataset, "discard_uncommitted_manifest", lambda *a, **k: False,
    )
    with pytest.raises(CloudSourceError, match="cleanup was incomplete"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/source.parquet",
        )


def test_audit_failure_does_not_delete_provenance_committed_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An append-only release must never be left pointing at a deleted file."""
    payload = _parquet_bytes(tmp_path)
    (tmp_path / "source.parquet").unlink()
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            iter((payload,)), len(payload), None, None, lambda: None,
        ),
    )
    from sift import integration_audit

    monkeypatch.setattr(
        integration_audit, "record_integration_event", lambda *a, **k: False,
    )
    with pytest.raises(CloudSourceError, match="audit event"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/source.parquet",
        )
    assert (tmp_path / "source.parquet").is_file()
    assert (tmp_path / "source.parquet.metadata.json").is_file()
    assert list((tmp_path / ".sift" / "datasets" / "manifests").glob("*.json"))


def test_repository_checksum_is_verified_before_provenance_commit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = _parquet_bytes(tmp_path)
    (tmp_path / "source.parquet").unlink()
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            iter((payload,)), len(payload), None, None, lambda: None,
        ),
    )
    wanted = hashlib.sha256(payload).hexdigest()
    result = cloud_sources.import_cloud_dataset(
        tmp_path,
        uri="https://files.example/source.parquet",
        _expected_checksum=f"sha256:{wanted}",
    )
    assert result.integrity_checksum == f"sha256:{wanted}"

    result.dataset_path.unlink()
    with pytest.raises(CloudSourceError, match="checksum did not match"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/mismatch.parquet",
            _expected_checksum="sha256:" + "0" * 64,
        )
    assert not (tmp_path / "mismatch.parquet").exists()


def test_cloud_import_is_not_a_model_tool() -> None:
    from sift.tools import ALLOWED_TOOL_NAMES, HANDLERS

    joined = " ".join((*ALLOWED_TOOL_NAMES, *HANDLERS)).casefold()
    assert "cloud" not in joined
    assert "s3" not in joined


def test_enterprise_cloud_source_allowlist_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sift import enterprise_policy

    monkeypatch.setattr(
        enterprise_policy,
        "load_enterprise_policy",
        lambda: enterprise_policy.EnterprisePolicy(
            allowed_cloud_sources=frozenset({"s3"}),
        ),
    )
    with pytest.raises(CloudSourceError, match="enterprise policy"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
        )


def test_cloud_import_cancels_before_opening_remote_source(tmp_path: Path) -> None:
    from sift.integration_core import CancellationToken

    token = CancellationToken()
    token.cancel()
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
            cancellation=token,
        )
    assert raised.value.code == "cancelled"
    assert not list(tmp_path.glob(".sift-cloud-*"))


def test_cloud_import_midstream_cancel_closes_and_cleans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sift.integration_core import CancellationToken

    token = CancellationToken()
    closed: list[bool] = []

    def chunks():
        yield b"a,b\n1,2\n"
        token.cancel()
        yield b"3,4\n"

    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            chunks(), None, None, "text/csv", lambda: closed.append(True)
        ),
    )
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
            cancellation=token,
        )
    assert raised.value.code == "cancelled"
    assert closed == [True]
    assert not list(tmp_path.glob(".sift-cloud-*"))


@pytest.mark.parametrize("mode", ["deadline", "cancelled"])
def test_https_initial_open_is_closed_on_outer_deadline_or_cancellation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    """A stuck initial request is interrupted without an SDK worker thread."""
    from sift.integration_core import CancellationToken

    released = threading.Event()
    opened = threading.Event()

    class Opener:
        def open(self, _request, timeout):
            assert 0 < timeout <= 1
            opened.set()
            assert released.wait(1)
            raise OSError("closed by transfer guard")

        def close(self):
            released.set()

    monkeypatch.setattr("urllib.request.build_opener", lambda *_args: Opener())
    monkeypatch.setattr(cloud_sources, "_validate_https_endpoint", lambda _uri: None)
    token = CancellationToken()
    canceller = None
    if mode == "cancelled":
        def cancel_after_open() -> None:
            assert opened.wait(1)
            token.cancel()

        canceller = threading.Thread(target=cancel_after_open)
        canceller.start()
    started = time.monotonic()
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
            validate_dataset=False,
            timeout_seconds=0.12 if mode == "deadline" else 1,
            cancellation=token,
        )
    if canceller is not None:
        canceller.join(1)
    # A heavily loaded runner may not schedule opener.open before the 120 ms
    # deadline has already expired. Both safe outcomes are valid: the request
    # never opens, or every opened resource is synchronously closed.
    assert not opened.is_set() or released.is_set()
    assert time.monotonic() - started < 0.5
    assert raised.value.code == (
        "deadline_exceeded" if mode == "deadline" else "cancelled"
    )
    assert not list(tmp_path.glob(".sift-cloud-*"))


def test_s3_blocking_initial_metadata_is_closed_at_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The S3 client is registered before get_object can block."""
    released = threading.Event()
    observed_config: dict[str, object] = {}

    class Client:
        def get_object(self, **_kwargs):
            assert released.wait(1)
            raise OSError("client closed")

        def close(self):
            released.set()

    def make_client(_name, **kwargs):
        observed_config.update(kwargs)
        return Client()

    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=make_client))
    started = time.monotonic()
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="s3://bucket/data.csv",
            validate_dataset=False,
            timeout_seconds=0.1,
        )
    assert time.monotonic() - started < 0.5
    assert released.is_set()
    assert raised.value.code == "deadline_exceeded"
    # Official botocore receives explicit bounded connect/read timeouts and
    # no hidden retries underneath Sift's verified resume protocol.
    assert "config" in observed_config
    assert observed_config["config"].connect_timeout <= 0.1
    assert observed_config["config"].read_timeout <= 0.1


def test_s3_client_is_closed_when_initial_metadata_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Ordinary pre-download SDK failures must not leak the client."""
    closed = threading.Event()

    class Client:
        def get_object(self, **_kwargs):
            raise OSError("ordinary metadata failure")

        def close(self):
            closed.set()

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda *_args, **_kwargs: Client()),
    )

    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="s3://bucket/data.csv",
            validate_dataset=False,
            timeout_seconds=1,
        )

    assert closed.is_set()
    assert "ordinary metadata failure" in str(raised.value)
    assert not list(tmp_path.glob(".sift-cloud-*"))


def test_https_drip_feed_cannot_extend_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """read1 yields promptly, then the wall clock is checked every arrival."""
    response_headers = Message()
    response_headers["Content-Type"] = "text/csv"
    closed = threading.Event()
    opened = threading.Event()

    class Response:
        headers = response_headers

        def read1(self, _size):
            if closed.wait(0.025):
                return b""
            return b"x"

        def geturl(self):
            return "https://files.example/data.csv"

        def close(self):
            closed.set()

    class Opener:
        def open(self, _request, timeout):
            assert timeout <= 0.15
            opened.set()
            return Response()

    monkeypatch.setattr("urllib.request.build_opener", lambda *_args: Opener())
    monkeypatch.setattr(cloud_sources, "_validate_https_endpoint", lambda _uri: None)
    started = time.monotonic()
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
            validate_dataset=False,
            timeout_seconds=0.12,
        )
    assert time.monotonic() - started < 0.5
    assert raised.value.code == "deadline_exceeded"
    assert not opened.is_set() or closed.is_set()
    assert not list(tmp_path.glob(".sift-cloud-*"))
    assert not (tmp_path / "data.csv").exists()


def test_cloud_import_timeout_is_bounded(tmp_path: Path) -> None:
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
            timeout_seconds=0,
        )
    assert raised.value.code == "invalid_timeout"


def test_https_endpoint_rejects_private_and_accepts_public_addresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("127.0.0.1"),),
    )
    with pytest.raises(CloudSourceError, match="loopback"):
        cloud_sources._validate_https_endpoint("https://files.example/data.csv")

    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("8.8.8.8"),),
    )
    cloud_sources._validate_https_endpoint("https://files.example/data.csv")
    with pytest.raises(CloudSourceError, match="HTTPS"):
        cloud_sources._validate_https_endpoint("http://files.example/data.csv")


def test_private_https_requires_explicit_administrator_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("10.0.0.8"),),
    )
    monkeypatch.setenv("SIFT_ALLOW_PRIVATE_HTTPS_IMPORT", "1")
    cloud_sources._validate_https_endpoint("https://intranet.example/data.csv")


def test_https_redirect_strips_credentials_when_only_port_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("8.8.8.8"),),
    )
    request = Request(
        "https://files.example/data.csv",
        headers={
            "Authorization": "Bearer port-secret",
            "X-API-TOKEN": "qualtrics-secret",
            "Dropbox-API-Arg": '{"path":"selected"}',
            "Zotero-API-Key": "zotero-secret",
        },
    )
    redirected = cloud_sources._SafeRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        Message(),
        "https://files.example:8443/other.csv",
    )
    assert redirected is not None
    assert redirected.get_header("Authorization") is None
    assert redirected.get_header("X-api-token") is None
    assert redirected.get_header("Dropbox-api-arg") is None
    assert redirected.get_header("Zotero-api-key") is None


def test_https_redirect_rejects_cross_origin_form_credential_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("8.8.8.8"),),
    )
    request = Request(
        "https://redcap.example/export.csv",
        data=b"token=body-secret&content=record",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with pytest.raises(CloudSourceError, match="POST redirects"):
        cloud_sources._SafeRedirectHandler().redirect_request(
            request,
            None,
            307,
            "Temporary Redirect",
            Message(),
            "https://downloads.example/export.csv",
        )


def test_https_connection_uses_only_the_validated_numeric_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later DNS answer cannot rebind the already-approved connection."""
    lookups = 0

    def rebinding_answers(_host: str, _port: int):
        nonlocal lookups
        lookups += 1
        return (
            (ipaddress.ip_address("8.8.8.8"),)
            if lookups == 1
            else (ipaddress.ip_address("127.0.0.1"),)
        )

    connected: list[tuple[object, ...]] = []

    class FakeSocket:
        def settimeout(self, _timeout):
            return None

        def setsockopt(self, *_args):
            return None

        def bind(self, _address):
            return None

        def connect(self, peer):
            connected.append(peer)

        def close(self):
            return None

    monkeypatch.setattr(cloud_sources, "_resolved_addresses", rebinding_answers)
    monkeypatch.setattr(socket, "socket", lambda *_args, **_kwargs: FakeSocket())
    uri = "https://files.example/data.csv"
    approved = cloud_sources._validated_https_addresses(uri)
    connection = cloud_sources._PinnedHTTPSConnection(
        "files.example",
        validated_addresses=approved,
        expected_origin=cloud_sources._https_origin(uri),
    )
    server_names: list[str] = []

    class FakeTLSContext:
        def wrap_socket(self, sock, *, server_hostname):
            server_names.append(server_hostname)
            return sock

    connection._context = FakeTLSContext()
    connection.connect()
    assert isinstance(connection.sock, FakeSocket)
    assert connected == [("8.8.8.8", 443)]
    assert lookups == 1
    assert server_names == ["files.example"]
    assert connection.host == "files.example"  # TLS SNI/hostname remains original


def test_pinned_https_connect_socket_is_closed_at_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation can interrupt the numeric TCP connect itself."""
    from sift.integration_core import Deadline, IntegrationDeadlineExceeded

    released = threading.Event()
    created = threading.Event()

    class BlockingSocket:
        def settimeout(self, _timeout):
            return None

        def bind(self, _address):
            return None

        def connect(self, _peer):
            assert released.wait(1)

        def close(self):
            released.set()

    def make_socket(*_args, **_kwargs):
        created.set()
        return BlockingSocket()

    monkeypatch.setattr(socket, "socket", make_socket)
    guard = cloud_sources._TransferGuard(Deadline(0.1), None)
    try:
        connection = cloud_sources._PinnedHTTPSConnection(
            "files.example",
            validated_addresses=(ipaddress.ip_address("8.8.8.8"),),
            expected_origin=("https", "files.example", 443),
            transfer_guard=guard,
        )
        with pytest.raises(IntegrationDeadlineExceeded):
            connection._connect_validated(("files.example", 443), timeout=10)
    finally:
        guard.finish()
    assert not created.is_set() or released.is_set()


def test_https_handler_rejects_rebinding_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter((
        (ipaddress.ip_address("8.8.8.8"),),
        (ipaddress.ip_address("127.0.0.1"),),
    ))
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: next(answers),
    )
    # The outer preflight accepts the first public answer. The per-request
    # handler resolves again and fails closed before its `do_open` can connect.
    cloud_sources._validate_https_endpoint("https://files.example/data.csv")
    handler = cloud_sources._PinnedHTTPSHandler()
    monkeypatch.setattr(
        handler,
        "do_open",
        lambda *_args, **_kwargs: pytest.fail("rebound endpoint was connected"),
    )
    with pytest.raises(CloudSourceError, match="loopback"):
        handler.https_open(Request("https://files.example/data.csv"))


def test_https_import_bypasses_environment_proxy_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_handlers: list[object] = []
    headers = Message()
    headers["Content-Length"] = "0"
    headers["Content-Type"] = "text/csv"

    class Response:
        def __init__(self):
            self.headers = headers

        def geturl(self):
            return "https://files.example/data.csv"

        def close(self):
            return None

    class Opener:
        def open(self, _request, timeout):
            assert timeout > 0
            return Response()

    def build(*handlers):
        captured_handlers.extend(handlers)
        return Opener()

    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9999")
    monkeypatch.delenv("SIFT_TRUST_HTTPS_IMPORT_PROXY", raising=False)
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("8.8.8.8"),),
    )
    monkeypatch.setattr("urllib.request.build_opener", build)
    info = cloud_sources._download_info(
        cloud_sources._http_chunks("https://files.example/data.csv")
    )
    info.close()
    proxy_handlers = [
        handler for handler in captured_handlers
        if isinstance(handler, cloud_sources.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
    assert any(
        isinstance(handler, cloud_sources._PinnedHTTPSHandler)
        for handler in captured_handlers
    )


def test_https_import_uses_environment_proxy_only_after_explicit_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_handlers: list[object] = []
    headers = Message()
    headers["Content-Length"] = "0"
    headers["Content-Type"] = "text/csv"

    class Response:
        def __init__(self):
            self.headers = headers

        def geturl(self):
            return "https://files.example/data.csv"

        def close(self):
            return None

    class Opener:
        def open(self, _request, timeout):
            assert timeout > 0
            return Response()

    def build(*handlers):
        captured_handlers.extend(handlers)
        return Opener()

    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8443")
    monkeypatch.setenv("SIFT_TRUST_HTTPS_IMPORT_PROXY", "1")
    monkeypatch.setattr(cloud_sources, "proxy_bypass", lambda _host: False)
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("8.8.8.8"),),
    )
    monkeypatch.setattr("urllib.request.build_opener", build)
    info = cloud_sources._download_info(
        cloud_sources._http_chunks("https://files.example/data.csv")
    )
    info.close()
    # Trusted mode pins the exact configured proxy and rejects any bypass;
    # it never falls back to a direct, separately resolved connection.
    assert len(captured_handlers) == 2
    assert isinstance(captured_handlers[0], cloud_sources.ProxyHandler)
    assert captured_handlers[0].proxies == {
        "https": "http://proxy.example:8443",
    }
    assert isinstance(captured_handlers[1], cloud_sources._SafeRedirectHandler)


@pytest.mark.parametrize("case", ["missing", "bypassed"])
def test_trusted_https_proxy_fails_closed_without_guaranteed_proxy_route(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    """The trust flag can never silently reopen direct DNS resolution."""
    monkeypatch.setenv("SIFT_TRUST_HTTPS_IMPORT_PROXY", "1")
    monkeypatch.setattr(cloud_sources, "_validate_https_endpoint", lambda _uri: None)
    monkeypatch.setattr(
        cloud_sources,
        "getproxies",
        lambda: {} if case == "missing" else {"https": "http://proxy.example:8443"},
    )
    monkeypatch.setattr(
        cloud_sources, "proxy_bypass", lambda _host: case == "bypassed",
    )
    with pytest.raises(CloudSourceError, match="proxy"):
        cloud_sources._http_chunks("https://files.example/data.csv")


def test_unexpected_active_content_type_is_rejected_before_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[bool] = []
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            iter((b"<html>login</html>",)), 18, None, "text/html",
            lambda: closed.append(True),
        ),
    )
    with pytest.raises(CloudSourceError, match="content type"):
        cloud_sources.import_cloud_dataset(
            tmp_path, uri="https://files.example/data.csv",
        )
    assert closed == [True]
    assert not list(tmp_path.glob(".sift-cloud-*"))


def test_provider_checksum_is_verified_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"id,value\n1,2\n"
    expected = base64.b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: cloud_sources._DownloadInfo(
            iter((payload,)), len(payload), '"etag"', "text/csv", lambda: None,
            checksum_algorithm="sha256",
            checksum_base64=expected,
            identifiers={"host": "files.example", "object": "data.csv"},
        ),
    )
    result = cloud_sources.import_cloud_dataset(
        tmp_path, uri="https://files.example/data.csv",
    )
    assert result.checksum_verified is True
    assert result.remote_checksum == f"sha256:{expected}"
    assert result.remote_identifiers == {
        "host": "files.example", "object": "data.csv",
    }


def test_provider_checksum_mismatch_removes_partial_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"id,value\n1,2\n"
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: cloud_sources._DownloadInfo(
            iter((payload,)), len(payload), None, "text/csv", lambda: None,
            checksum_algorithm="sha256",
            checksum_base64=base64.b64encode(b"x" * 32).decode("ascii"),
        ),
    )
    with pytest.raises(CloudSourceError, match="checksum"):
        cloud_sources.import_cloud_dataset(
            tmp_path, uri="https://files.example/data.csv",
        )
    assert not (tmp_path / "data.csv").exists()
    assert not list(tmp_path.glob(".sift-cloud-*"))


def test_generic_and_nested_archives_are_never_ingested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generic = io.BytesIO()
    with zipfile.ZipFile(generic, "w") as archive:
        archive.writestr("data.csv", "id\n1\n")
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            iter((generic.getvalue(),)), len(generic.getvalue()), None,
            "application/octet-stream", lambda: None,
        ),
    )
    with pytest.raises(CloudSourceError, match="archives are not accepted"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/disguised.csv",
            validate_dataset=False,
        )

    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as archive:
        archive.writestr("xl/workbook.xml", "<workbook />")
        archive.writestr("xl/nested.zip", b"PK\x03\x04")
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout: (
            iter((nested.getvalue(),)), len(nested.getvalue()), None,
            "application/octet-stream", lambda: None,
        ),
    )
    with pytest.raises(CloudSourceError, match="nested archive"):
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/disguised.xlsx",
            validate_dataset=False,
        )


def test_symlink_session_destination_is_rejected(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(CloudSourceError, match="symlink"):
        cloud_sources.import_cloud_dataset(
            link, uri="https://files.example/data.csv",
        )


def test_s3_explicit_version_requester_pays_and_checksum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class Body:
        def iter_chunks(self, *, chunk_size: int):
            assert chunk_size == cloud_sources.DOWNLOAD_CHUNK_BYTES
            yield b"id\n1\n"

        def close(self) -> None:
            return None

    class Client:
        def get_object(self, **kwargs):
            seen.update(kwargs)
            payload = b"id\n1\n"
            return {
                "Body": Body(), "ContentLength": len(payload),
                "VersionId": "v1", "ETag": '"etag"', "ContentType": "text/csv",
                "ChecksumSHA256": base64.b64encode(
                    hashlib.sha256(payload).digest()
                ).decode("ascii"),
                "ChecksumType": "FULL_OBJECT",
            }

    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda _name: Client()),
    )
    info = cloud_sources._s3_chunks(
        "s3://research/data.csv?versionId=v1&requestPayer=requester"
    )
    assert seen["VersionId"] == "v1"
    assert seen["RequestPayer"] == "requester"
    assert seen["ChecksumMode"] == "ENABLED"
    assert info.checksum_algorithm == "sha256"
    assert info.identifiers["bucket"] == "research"


def test_s3_stream_resumes_from_verified_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class FirstBody:
        def iter_chunks(self, *, chunk_size: int):
            yield b"abc"
            raise OSError("transient disconnect")

        def close(self):
            return None

    class SecondBody:
        def iter_chunks(self, *, chunk_size: int):
            yield b"def"

        def close(self):
            return None

    class Client:
        def get_object(self, **kwargs):
            calls.append(dict(kwargs))
            if len(calls) == 1:
                return {
                    "Body": FirstBody(), "ContentLength": 6,
                    "ETag": '"stable"', "ContentType": "text/csv",
                }
            return {
                "Body": SecondBody(), "ContentLength": 3,
                "ETag": '"stable"', "ContentRange": "bytes 3-5/6",
            }

    monkeypatch.setitem(
        sys.modules, "boto3", SimpleNamespace(client=lambda _name: Client()),
    )
    info = cloud_sources._s3_chunks("s3://bucket/data.csv")
    assert b"".join(info.chunks) == b"abcdef"
    assert calls[1]["Range"] == "bytes=3-"
    assert calls[1]["IfMatch"] == '"stable"'
    assert info.resumable is True


def test_https_stream_resumes_only_with_validator_and_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_headers = Message()
    first_headers["Content-Length"] = "6"
    first_headers["Content-Type"] = "text/csv"
    first_headers["ETag"] = '"stable"'
    first_headers["Accept-Ranges"] = "bytes"
    second_headers = Message()
    second_headers["Content-Range"] = "bytes 3-5/6"
    second_headers["Content-Type"] = "text/csv"

    class Response:
        def __init__(self, headers, chunks, code):
            self.headers = headers
            self._chunks = iter(chunks)
            self._code = code

        def read(self, _size):
            value = next(self._chunks, b"")
            if isinstance(value, Exception):
                raise value
            return value

        def geturl(self):
            return "https://files.example/data.csv"

        def getcode(self):
            return self._code

        def close(self):
            return None

    responses = iter((
        Response(first_headers, (b"abc", OSError("disconnect")), 200),
        Response(second_headers, (b"def",), 206),
    ))
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return next(responses)

    monkeypatch.setattr(
        "urllib.request.build_opener", lambda *_handlers: Opener(),
    )
    monkeypatch.setattr(
        cloud_sources,
        "_resolved_addresses",
        lambda _host, _port: (ipaddress.ip_address("8.8.8.8"),),
    )
    info = cloud_sources._http_chunks("https://files.example/data.csv")
    assert b"".join(info.chunks) == b"abcdef"
    assert requests[1].get_header("Range") == "bytes=3-"
    assert requests[1].get_header("If-range") == '"stable"'
    assert info.resumable is True


def test_sftp_requires_pinned_hosts_and_key_only_authentication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    key = tmp_path / "id_ed25519"
    key.write_text("private", encoding="utf-8")
    key.chmod(0o600)
    known = tmp_path / "known_hosts"
    known.write_text("files.example ssh-ed25519 AAAA", encoding="utf-8")
    calls: dict[str, object] = {}

    class Remote(io.BytesIO):
        def __init__(self):
            super().__init__(b"id\n1\n")

    class SFTP:
        def stat(self, path):
            calls["path"] = path
            return SimpleNamespace(st_size=5, st_mtime=123)

        def open(self, path, mode):
            assert mode == "rb"
            return Remote()

        def close(self):
            return None

    class Client:
        def load_host_keys(self, path):
            calls["known_hosts"] = path

        def set_missing_host_key_policy(self, policy):
            calls["policy"] = type(policy).__name__

        def connect(self, **kwargs):
            calls.update(kwargs)

        def open_sftp(self):
            return SFTP()

        def close(self):
            return None

    class RejectPolicy:
        pass

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=Client, RejectPolicy=RejectPolicy),
    )
    profile = json.dumps({
        "private_key": str(key), "known_hosts": str(known), "passphrase": "vault",
    })
    info = cloud_sources._sftp_chunks(
        "sftp://alice@files.example/research/data.csv",
        credential=profile,
    )
    assert b"".join(info.chunks) == b"id\n1\n"
    assert calls["policy"] == "RejectPolicy"
    assert calls["allow_agent"] is False
    assert calls["look_for_keys"] is False
    assert calls["username"] == "alice"


def test_sftp_blocking_connect_is_closed_at_outer_deadline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from sift.integration_core import Deadline, IntegrationDeadlineExceeded

    key = tmp_path / "id_ed25519"
    key.write_text("private", encoding="utf-8")
    key.chmod(0o600)
    known = tmp_path / "known_hosts"
    known.write_text("files.example ssh-ed25519 AAAA", encoding="utf-8")
    released = threading.Event()

    class Client:
        def load_host_keys(self, _path):
            return None

        def set_missing_host_key_policy(self, _policy):
            return None

        def connect(self, **kwargs):
            assert kwargs["timeout"] <= 0.1
            assert released.wait(1)
            raise OSError("closed")

        def close(self):
            released.set()

    monkeypatch.setitem(
        sys.modules,
        "paramiko",
        SimpleNamespace(SSHClient=Client, RejectPolicy=type("RejectPolicy", (), {})),
    )
    profile = json.dumps({
        "private_key": str(key), "known_hosts": str(known),
    })
    guard = cloud_sources._TransferGuard(Deadline(0.1), None)
    try:
        with pytest.raises(IntegrationDeadlineExceeded):
            cloud_sources._sftp_chunks(
                "sftp://alice@files.example/data.csv",
                timeout_seconds=0.1,
                credential=profile,
                _guard=guard,
            )
    finally:
        guard.finish()
    assert released.is_set()


def test_gcs_generation_adc_metadata_checksum_and_resumable_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"id\n1\n"
    crc = pytest.importorskip("google_crc32c").Checksum(payload)

    class Reader(io.BytesIO):
        pass

    class Blob:
        size = len(payload)
        generation = 42
        etag = "etag-gcs"
        content_type = "text/csv"
        crc32c = base64.b64encode(crc.digest()).decode("ascii")
        md5_hash = None

        def reload(self):
            return None

        def open(self, mode):
            assert mode == "rb"
            return Reader(payload)

    class Bucket:
        def blob(self, key, generation=None):
            assert key == "research/data.csv"
            assert generation == 42
            return Blob()

    class Client:
        def bucket(self, name):
            assert name == "study-bucket"
            return Bucket()

    storage = SimpleNamespace(Client=Client)
    monkeypatch.setitem(sys.modules, "google.cloud", SimpleNamespace(storage=storage))
    monkeypatch.setitem(sys.modules, "google.cloud.storage", storage)
    info = cloud_sources._gcs_chunks(
        "gs://study-bucket/research/data.csv?generation=42"
    )
    assert b"".join(info.chunks) == payload
    assert info.checksum_algorithm == "crc32c"
    assert info.identifiers["generation"] == "42"
    assert info.resumable is True


def test_azure_sas_snapshot_checksum_and_resumable_downloader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"id\n1\n"
    calls: dict[str, object] = {}

    class Credential:
        def close(self):
            return None

    class Downloader:
        def chunks(self):
            yield payload

    class Blob:
        def get_blob_properties(self):
            return SimpleNamespace(
                size=len(payload), version_id=None, etag="etag-azure",
                content_settings=SimpleNamespace(
                    content_type="text/csv",
                    content_md5=hashlib.md5(
                        payload, usedforsecurity=False,
                    ).digest(),
                ),
            )

        def download_blob(self, **kwargs):
            calls.update(kwargs)
            return Downloader()

    class Service:
        def __init__(self, **kwargs):
            calls.update(kwargs)

        def get_blob_client(self, **kwargs):
            calls.update(kwargs)
            return Blob()

    identity = SimpleNamespace(DefaultAzureCredential=Credential)
    blob_module = SimpleNamespace(BlobServiceClient=Service)
    monkeypatch.setitem(sys.modules, "azure.identity", identity)
    monkeypatch.setitem(sys.modules, "azure.storage.blob", blob_module)
    info = cloud_sources._azure_chunks(
        "az://account/container/data.csv?snapshot=snapshot-id",
        credential="?sv=secret-sas",
    )
    assert b"".join(info.chunks) == payload
    assert calls["credential"] == "?sv=secret-sas"
    assert calls["snapshot"] == "snapshot-id"
    assert calls["validate_content"] is True
    assert info.checksum_algorithm == "md5"
    assert info.identifiers["snapshot"] == "snapshot-id"
    assert info.resumable is True


def test_vault_credential_is_resolved_host_side_and_never_recorded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"id\n1\n"
    secret = "PRIVATE-BEARER-TOKEN"
    monkeypatch.setattr(
        "sift.remote_credentials.resolve_remote_credential",
        lambda name, kind: secret if (name, kind) == ("Institution", "https_bearer") else None,
    )

    def open_source(_uri: str, _timeout: float, credential: str):
        assert credential == secret
        return cloud_sources._DownloadInfo(
            iter((payload,)), len(payload), None, "text/csv", lambda: None,
        )

    monkeypatch.setattr(cloud_sources, "_http_chunks", open_source)
    result = cloud_sources.import_cloud_dataset(
        tmp_path,
        uri="https://files.example/data.csv",
        credential_profile="Institution",
    )
    assert result.dataset_path.is_file()
    from sift.release_ledger import read_ledger

    assert secret not in json.dumps(read_ledger(tmp_path))


def test_vault_credential_is_redacted_from_driver_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "PRIVATE-BEARER-TOKEN"
    monkeypatch.setattr(
        "sift.remote_credentials.resolve_remote_credential",
        lambda _name, _kind: secret,
    )
    monkeypatch.setattr(
        cloud_sources,
        "_http_chunks",
        lambda _uri, _timeout, _credential: (_ for _ in ()).throw(
            RuntimeError(f"authorization failed for {secret}")
        ),
    )
    with pytest.raises(CloudSourceError) as raised:
        cloud_sources.import_cloud_dataset(
            tmp_path,
            uri="https://files.example/data.csv",
            credential_profile="Institution",
        )
    assert secret not in str(raised.value)
