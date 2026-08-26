"""Explicit-selection research artifact integrations.

No adapter exposes search/list/account APIs to the model. A researcher selects
one remote file (or a bounded set of local Zotero items), Sift materializes it
locally, and only then can ordinary analysis begin.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urljoin, urlsplit
from urllib.request import ProxyHandler, Request, build_opener, getproxies, proxy_bypass

from sift.cloud_sources import (
    CloudImportResult,
    CloudSourceError,
    _PinnedHTTPSHandler,
    _SafeRedirectHandler,
    _https_origin,
    _validate_https_endpoint,
    cloud_import_max_bytes,
    import_cloud_dataset,
)
from sift.provider.error_safety import provider_error_message
from sift.text_safety import safe_text


class ResearchServiceError(Exception):
    pass


@dataclass(frozen=True)
class ResearchArtifactResult:
    dataset_path: Path
    service: str
    artifact_id: str
    revision: str | None
    dataset_sha256: str
    metadata: dict[str, str]
    canonical_fingerprint: str | None = None


@dataclass(frozen=True)
class ServiceContract:
    id: str
    label: str
    host_suffixes: tuple[str, ...]
    preserves: tuple[str, ...]
    authentication: str


SERVICE_CONTRACTS: dict[str, ServiceContract] = {
    "zotero": ServiceContract("zotero", "Zotero", ("zotero.org",), ("citation_key", "item", "version", "attachment"), "local API or scoped API key"),
    "osf": ServiceContract("osf", "OSF", ("osf.io",), ("project", "version"), "optional bearer token"),
    "dataverse": ServiceContract("dataverse", "Dataverse", (), ("doi", "version", "license", "file"), "optional scoped bearer token"),
    "zenodo": ServiceContract("zenodo", "Zenodo", ("zenodo.org",), ("doi", "version", "license", "file"), "optional bearer token"),
    "figshare": ServiceContract("figshare", "Figshare", ("figshare.com",), ("doi", "version", "license", "file"), "optional bearer token"),
    "dryad": ServiceContract("dryad", "Dryad", ("datadryad.org",), ("doi", "version", "license", "file"), "optional bearer token"),
    "google_drive": ServiceContract("google_drive", "Google Drive", ("googleapis.com",), ("file", "revision"), "OAuth bearer token"),
    "onedrive": ServiceContract("onedrive", "OneDrive", ("microsoft.com",), ("drive", "file", "revision"), "Microsoft Graph OAuth token"),
    "sharepoint": ServiceContract("sharepoint", "SharePoint", ("microsoft.com",), ("site", "drive", "file", "revision"), "Microsoft Graph OAuth token"),
    "box": ServiceContract("box", "Box", ("box.com",), ("file", "version"), "OAuth bearer token"),
    "dropbox": ServiceContract("dropbox", "Dropbox", ("dropbox.com", "dropboxapi.com", "dropboxusercontent.com"), ("file", "revision"), "selected shared link or scoped token"),
    "redcap": ServiceContract("redcap", "REDCap", (), ("project", "report", "revision"), "project-scoped API token"),
    "qualtrics": ServiceContract("qualtrics", "Qualtrics", ("qualtrics.com",), ("survey", "export"), "user API token"),
    "kobotoolbox": ServiceContract("kobotoolbox", "KoboToolbox", (), ("project", "export_settings", "revision"), "scoped API token"),
    "openclinica": ServiceContract("openclinica", "OpenClinica", (), ("study", "extract_job", "format"), "study-scoped bearer token"),
}

_MAX_API_JSON_BYTES = 16 * 1024 * 1024
_MAX_ZOTERO_EXPORT_BYTES = 128 * 1024 * 1024
_MAX_ZOTERO_ATTACHMENTS = 100


def _research_token(profile: str | None) -> str | None:
    if not profile:
        return None
    try:
        from sift.remote_credentials import resolve_remote_credential

        return resolve_remote_credential(profile, "research_token")
    except Exception as e:
        raise ResearchServiceError(str(e)) from e


def _auth_headers(service: str, token: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json", "User-Agent": "Sift/0.1 research-import"}
    if not token:
        return headers
    if service == "zotero":
        headers["Zotero-API-Key"] = token
        headers["Zotero-API-Version"] = "3"
    elif service == "qualtrics":
        headers["X-API-TOKEN"] = token
    elif service == "kobotoolbox":
        headers["Authorization"] = f"Token {token}"
    else:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _api_json(
    url: str,
    *,
    service: str,
    token: str | None = None,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read one bounded service record; no search or pagination surface."""
    _validate_https_endpoint(url)
    parsed_request = urlsplit(url)
    contract = SERVICE_CONTRACTS.get(service)
    if contract is None:
        raise ResearchServiceError(f"unknown research service: {service!r}")
    if contract.host_suffixes and not any(
        parsed_request.hostname == suffix
        or str(parsed_request.hostname).endswith(f".{suffix}")
        for suffix in contract.host_suffixes
    ):
        raise ResearchServiceError(
            f"metadata URL host is not an approved {contract.label} endpoint"
        )

    class SafeRedirect(_SafeRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            target = urljoin(req.full_url, newurl)
            _validate_https_endpoint(target)
            parsed_target = urlsplit(target)
            if contract.host_suffixes and not any(
                parsed_target.hostname == suffix
                or str(parsed_target.hostname).endswith(f".{suffix}")
                for suffix in contract.host_suffixes
            ):
                raise ResearchServiceError(
                    f"metadata redirect escaped the approved {contract.label} endpoint"
                )
            return super().redirect_request(
                req, fp, code, msg, headers, target,
            )

    response = None
    try:
        headers = _auth_headers(service, token)
        body = None
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        # Match dataset-download confidentiality: ambient proxy variables do
        # not silently receive scoped tokens, and the TCP connection uses the
        # exact public addresses approved during endpoint validation. An
        # administrator may deliberately opt into a configured HTTPS proxy.
        if os.environ.get("SIFT_TRUST_HTTPS_IMPORT_PROXY") == "1":
            proxy_url = getproxies().get("https")
            if not proxy_url:
                raise ResearchServiceError(
                    "trusted HTTPS proxy mode requires an explicitly configured proxy"
                )
            if proxy_bypass(_https_origin(url)[1]):
                raise ResearchServiceError(
                    "trusted HTTPS proxy mode would bypass the configured proxy"
                )
            opener = build_opener(
                ProxyHandler({"https": proxy_url}),
                SafeRedirect(require_proxy=True),
            )
        else:
            opener = build_opener(
                ProxyHandler({}), _PinnedHTTPSHandler(), SafeRedirect(),
            )
        response = opener.open(
            Request(url, data=body, headers=headers, method=method), timeout=30,
        )
        _validate_https_endpoint(response.geturl())
        parsed_response = urlsplit(response.geturl())
        if contract.host_suffixes and not any(
            parsed_response.hostname == suffix
            or str(parsed_response.hostname).endswith(f".{suffix}")
            for suffix in contract.host_suffixes
        ):
            raise ResearchServiceError(
                f"metadata redirect escaped the approved {contract.label} endpoint"
            )
        length = response.headers.get("Content-Length")
        if length and length.isdecimal() and int(length) > _MAX_API_JSON_BYTES:
            raise ResearchServiceError("research service metadata response is too large")
        raw = response.read(_MAX_API_JSON_BYTES + 1)
        if len(raw) > _MAX_API_JSON_BYTES:
            raise ResearchServiceError("research service metadata response is too large")
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ResearchServiceError("research service returned malformed metadata")
        return value
    except ResearchServiceError:
        raise
    except Exception as e:
        message = provider_error_message(e, secrets=(token,) if token else ())
        raise ResearchServiceError(
            f"{service} metadata request failed: {message.splitlines()[0][:300]}"
        ) from e
    finally:
        if response is not None:
            response.close()


_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,511}$")
_OPAQUE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,511}$")


def _identifier(value: str, label: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not _IDENTIFIER.fullmatch(cleaned):
        raise ResearchServiceError(f"invalid {label}")
    return cleaned


def _opaque_identifier(value: str, label: str) -> str:
    cleaned = value.strip() if isinstance(value, str) else ""
    if not _OPAQUE_IDENTIFIER.fullmatch(cleaned):
        raise ResearchServiceError(f"invalid {label}")
    return cleaned


def selected_download_url(
    service: str,
    *,
    artifact_id: str,
    file_id: str | None = None,
    base_url: str | None = None,
    download_url: str | None = None,
    drive_id: str | None = None,
) -> str:
    """Build/validate a URL for one already-selected artifact; never list."""
    if service not in SERVICE_CONTRACTS:
        raise ResearchServiceError(f"unknown research service: {service!r}")
    artifact = _identifier(artifact_id, "artifact id")
    selected_file = _opaque_identifier(file_id, "file id") if file_id else (
        _opaque_identifier(artifact_id, "file id")
        if service in {"google_drive", "box"} else artifact
    )
    if download_url:
        url = download_url.strip()
    elif service == "figshare":
        url = f"https://api.figshare.com/v2/file/download/{quote(selected_file)}"
    elif service == "dryad":
        url = f"https://datadryad.org/api/v2/files/{quote(selected_file)}/download"
    elif service == "google_drive":
        url = f"https://www.googleapis.com/drive/v3/files/{quote(selected_file)}?alt=media"
    elif service in {"onedrive", "sharepoint"}:
        drive = _opaque_identifier(drive_id or "", "drive id")
        url = (
            f"https://graph.microsoft.com/v1.0/drives/{quote(drive)}"
            f"/items/{quote(selected_file)}/content"
        )
    elif service == "box":
        url = f"https://api.box.com/2.0/files/{quote(selected_file)}/content"
    elif service == "dataverse":
        if not base_url:
            raise ResearchServiceError("Dataverse requires its installation base URL")
        root = base_url.rstrip("/")
        url = f"{root}/api/access/datafile/{quote(selected_file)}"
    else:
        raise ResearchServiceError(
            f"{service} requires the selected file's API-provided download URL"
        )

    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ResearchServiceError("research artifact download URL must be credential-free HTTPS")
    contract = SERVICE_CONTRACTS[service]
    if contract.host_suffixes and not any(
        parsed.hostname == suffix or parsed.hostname.endswith(f".{suffix}")
        for suffix in contract.host_suffixes
    ):
        raise ResearchServiceError(
            f"selected URL host is not an approved {contract.label} endpoint"
        )
    if service == "dataverse":
        base = urlsplit(base_url or "")
        if base.scheme != "https" or base.hostname != parsed.hostname:
            raise ResearchServiceError("Dataverse download must stay on its selected installation")
    return url


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    allowed = {
        "title", "doi", "version", "license", "project", "record",
        "file", "file_id", "revision", "citation_key", "library", "site", "drive",
        "survey", "report", "project_id", "export", "format", "checksum",
    }
    result: dict[str, str] = {}
    for key, value in (metadata or {}).items():
        if str(key) not in allowed or value is None:
            continue
        cleaned = safe_text(str(value), max_len=500)
        if cleaned:
            result[str(key)] = cleaned
    return result


def import_selected_artifact(
    cwd: Path,
    *,
    service: str,
    artifact_id: str,
    filename: str,
    file_id: str | None = None,
    revision: str | None = None,
    metadata: dict[str, Any] | None = None,
    base_url: str | None = None,
    download_url: str | None = None,
    drive_id: str | None = None,
    credential_profile: str | None = None,
    expected_checksum: str | None = None,
    auth_mode: str = "bearer",
    form_fields: dict[str, str] | None = None,
) -> ResearchArtifactResult:
    artifact = _identifier(artifact_id, "artifact id")
    url = selected_download_url(
        service,
        artifact_id=artifact,
        file_id=file_id,
        base_url=base_url,
        download_url=download_url,
        drive_id=drive_id,
    )
    clean = _clean_metadata(metadata)
    if revision:
        clean["revision"] = safe_text(revision, max_len=200)
    try:
        imported: CloudImportResult = import_cloud_dataset(
            Path(cwd),
            uri=url,
            dataset_name=filename,
            credential_profile=credential_profile,
            _credential_kind="research_token" if credential_profile else None,
            _https_auth_mode=auth_mode,
            _https_form_fields=form_fields,
            _expected_checksum=expected_checksum,
        )
    except CloudSourceError as e:
        raise ResearchServiceError(str(e)) from e

    repository_checksum = imported.integrity_checksum
    if expected_checksum and not repository_checksum:
        # A real cloud import always returns the pre-commit proof. Refuse an
        # adapter/test double that cannot prove it enforced the checksum.
        raise ResearchServiceError(
            "research artifact import did not return checksum verification proof"
        )
    if repository_checksum:
        clean["checksum"] = repository_checksum

    try:
        from sift.canonical_dataset import ensure_manifest
        canonical_manifest = ensure_manifest(
            Path(cwd),
            imported.dataset_path,
            selection={
                "research_service": service,
                "artifact_id": artifact,
                "file_id": file_id,
                "revision": revision,
            },
            transformations=({
                "operation": "research_artifact_import",
                "runtime": service,
            },),
        )
    except Exception as e:
        raise ResearchServiceError(
            f"could not establish canonical research dataset identity: {type(e).__name__}"
        ) from e

    from sift import release_ledger

    recorded = release_ledger.record_release(
        Path(cwd),
        kind="research_artifact_import",
        tool=f"({service} selected artifact)",
        extra={
            "dataset": imported.dataset_path.name,
            "service": service,
            "artifact_id": artifact,
            "file_id": file_id,
            "revision": revision,
            "dataset_sha256": imported.dataset_sha256,
            "remote_version": imported.remote_version,
            "repository_checksum": repository_checksum,
            "canonical_fingerprint": canonical_manifest["fingerprint"],
            "metadata": clean,
        },
    )
    if not recorded:
        raise ResearchServiceError("could not record research artifact provenance")
    return ResearchArtifactResult(
        imported.dataset_path,
        service,
        artifact,
        safe_text(revision, max_len=200) if revision else None,
        imported.dataset_sha256,
        clean,
        canonical_manifest["fingerprint"],
    )


def _data(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("data")
    return nested if isinstance(nested, dict) else value


def _attributes(value: dict[str, Any]) -> dict[str, Any]:
    nested = value.get("attributes")
    return nested if isinstance(nested, dict) else value


def _nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _selected_file(files: Any, file_id: str) -> dict[str, Any]:
    wanted = _opaque_identifier(file_id, "file id")
    if not isinstance(files, list) or len(files) > 100_000:
        raise ResearchServiceError("repository returned an invalid file manifest")
    matches = []
    for row in files:
        if not isinstance(row, dict):
            continue
        data = _data(row)
        candidate = str(
            data.get("id") or data.get("key") or data.get("file_id")
            or _nested(data, "dataFile", "id") or ""
        )
        if candidate == wanted:
            matches.append(data)
    if len(matches) != 1:
        raise ResearchServiceError("the selected repository file was not found uniquely")
    return matches[0]


def import_osf_file(
    cwd: Path, *, node_id: str, file_id: str, filename: str,
    version_id: str | None = None, credential_profile: str | None = None,
) -> ResearchArtifactResult:
    """Import one OSF file/version without listing a project or account."""
    node = _opaque_identifier(node_id, "OSF node id")
    selected = _opaque_identifier(file_id, "OSF file id")
    version = _opaque_identifier(version_id, "OSF version id") if version_id else None
    token = _research_token(credential_profile)
    endpoint = f"https://api.osf.io/v2/files/{quote(selected)}/"
    if version:
        endpoint += f"versions/{quote(version)}/"
    record = _data(_api_json(endpoint, service="osf", token=token))
    attrs = _attributes(record)
    related_node = str(_nested(record, "relationships", "node", "data", "id") or node)
    if related_node != node:
        raise ResearchServiceError("selected OSF file does not belong to the selected project")
    download = str(_nested(record, "links", "download") or attrs.get("download_url") or "")
    checksum = _nested(attrs, "extra", "hashes", "sha256") or attrs.get("sha256")
    return import_selected_artifact(
        cwd, service="osf", artifact_id=node, file_id=selected,
        filename=filename, revision=version or str(attrs.get("version") or "") or None,
        metadata={"project": node, "file": attrs.get("name") or filename},
        download_url=download, credential_profile=credential_profile,
        expected_checksum=f"sha256:{checksum}" if checksum else None,
    )


def import_dataverse_file(
    cwd: Path, *, base_url: str, persistent_id: str, dataset_version: str,
    file_id: str, filename: str, credential_profile: str | None = None,
) -> ResearchArtifactResult:
    """Import one file from one exact Dataverse dataset version."""
    doi = _identifier(persistent_id, "Dataverse persistent id")
    version = _identifier(dataset_version, "Dataverse dataset version")
    selected = _opaque_identifier(file_id, "Dataverse file id")
    root = base_url.rstrip("/")
    parsed = urlsplit(root)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ResearchServiceError("Dataverse base URL must be HTTPS")
    token = _research_token(credential_profile)
    url = (
        f"{root}/api/datasets/:persistentId/versions/{quote(version, safe='')}"
        f"?persistentId={quote(doi, safe='')}"
    )
    record = _data(_api_json(url, service="dataverse", token=token))
    row = _selected_file(record.get("files"), selected)
    raw_data_file = row.get("dataFile")
    data_file = raw_data_file if isinstance(raw_data_file, dict) else row
    raw_checksum = data_file.get("checksum")
    checksum = raw_checksum if isinstance(raw_checksum, dict) else {}
    checksum_value = checksum.get("value")
    checksum_type = str(checksum.get("type") or "").casefold().replace("-", "")
    expected = f"{checksum_type}:{checksum_value}" if checksum_type in {"md5", "sha1", "sha256"} and checksum_value else None
    license_value = record.get("license")
    if isinstance(license_value, dict):
        license_value = license_value.get("name") or license_value.get("uri")
    return import_selected_artifact(
        cwd, service="dataverse", artifact_id=doi, file_id=selected,
        filename=filename, revision=str(record.get("versionNumber") or version),
        metadata={"doi": doi, "version": version, "license": license_value,
                  "file": data_file.get("filename") or filename, "file_id": selected},
        base_url=root, credential_profile=credential_profile,
        expected_checksum=expected,
    )


def import_zenodo_file(
    cwd: Path, *, record_id: str, file_id: str, filename: str,
    credential_profile: str | None = None,
) -> ResearchArtifactResult:
    record_key = _opaque_identifier(record_id, "Zenodo record id")
    selected = _opaque_identifier(file_id, "Zenodo file id")
    token = _research_token(credential_profile)
    record = _api_json(
        f"https://zenodo.org/api/records/{quote(record_key)}",
        service="zenodo", token=token,
    )
    row = _selected_file(record.get("files"), selected)
    raw_links = row.get("links")
    links = raw_links if isinstance(raw_links, dict) else {}
    raw_metadata = record.get("metadata")
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    license_value = metadata.get("license")
    if isinstance(license_value, dict):
        license_value = license_value.get("id") or license_value.get("title")
    return import_selected_artifact(
        cwd, service="zenodo", artifact_id=record_key, file_id=selected,
        filename=filename, revision=str(record.get("revision") or record.get("updated") or "") or None,
        metadata={"doi": record.get("doi") or metadata.get("doi"),
                  "version": metadata.get("version"), "license": license_value,
                  "file": row.get("key") or filename, "file_id": selected},
        download_url=str(links.get("content") or links.get("self") or ""),
        credential_profile=credential_profile,
        expected_checksum=str(row.get("checksum") or "") or None,
    )


def import_figshare_file(
    cwd: Path, *, article_id: str, file_id: str, filename: str,
    article_version: str | None = None, credential_profile: str | None = None,
) -> ResearchArtifactResult:
    article = _opaque_identifier(article_id, "Figshare article id")
    selected = _opaque_identifier(file_id, "Figshare file id")
    version = _opaque_identifier(article_version, "Figshare version") if article_version else None
    token = _research_token(credential_profile)
    endpoint = f"https://api.figshare.com/v2/articles/{quote(article)}"
    if version:
        endpoint += f"/versions/{quote(version)}"
    record = _api_json(endpoint, service="figshare", token=token)
    row = _selected_file(record.get("files"), selected)
    license_value = record.get("license")
    if isinstance(license_value, dict):
        license_value = license_value.get("name") or license_value.get("url")
    return import_selected_artifact(
        cwd, service="figshare", artifact_id=article, file_id=selected,
        filename=filename, revision=version or str(record.get("version") or "") or None,
        metadata={"doi": record.get("doi"), "version": version or record.get("version"),
                  "license": license_value, "file": row.get("name") or filename,
                  "file_id": selected},
        download_url=str(row.get("download_url") or ""),
        credential_profile=credential_profile,
        expected_checksum=f"md5:{row['supplied_md5']}" if row.get("supplied_md5") else None,
    )


def import_repository_file(
    cwd: Path, *, service: str, artifact_id: str, file_id: str,
    filename: str, metadata_url: str, download_url: str,
    revision: str | None = None, metadata: dict[str, Any] | None = None,
    expected_checksum: str | None = None,
    credential_profile: str | None = None,
) -> ResearchArtifactResult:
    """Exact-file adapter for Dryad or institutional repository variants.

    Both URLs come from the researcher's already-selected record. Sift reads
    that single metadata record to validate identity but never exposes search.
    """
    token = _research_token(credential_profile)
    record = _api_json(metadata_url, service=service, token=token)
    remote_id = str(record.get("id") or _nested(record, "data", "id") or artifact_id)
    if remote_id != artifact_id:
        raise ResearchServiceError("repository metadata did not match the selected artifact")
    combined = dict(metadata or {})
    combined.setdefault("doi", record.get("doi"))
    combined.setdefault("version", record.get("version"))
    combined.setdefault("license", record.get("license"))
    combined.setdefault("file_id", file_id)
    return import_selected_artifact(
        cwd, service=service, artifact_id=artifact_id, file_id=file_id,
        filename=filename, revision=revision or str(record.get("version") or "") or None,
        metadata=combined, download_url=download_url,
        credential_profile=credential_profile, expected_checksum=expected_checksum,
    )


def import_google_drive_file(
    cwd: Path, *, file_id: str, filename: str,
    credential_profile: str,
) -> ResearchArtifactResult:
    selected = _opaque_identifier(file_id, "Google Drive file id")
    token = _research_token(credential_profile)
    fields = quote("id,name,mimeType,md5Checksum,version,modifiedTime,size", safe=",")
    record = _api_json(
        f"https://www.googleapis.com/drive/v3/files/{quote(selected)}?fields={fields}",
        service="google_drive", token=token,
    )
    if str(record.get("id")) != selected:
        raise ResearchServiceError("Google Drive metadata did not match the selected file")
    revision = str(record.get("version") or record.get("modifiedTime") or "") or None
    return import_selected_artifact(
        cwd, service="google_drive", artifact_id=selected, file_id=selected,
        filename=filename, revision=revision,
        metadata={"file": record.get("name") or filename, "file_id": selected,
                  "revision": revision}, credential_profile=credential_profile,
        expected_checksum=f"md5:{record['md5Checksum']}" if record.get("md5Checksum") else None,
    )


def import_microsoft_drive_file(
    cwd: Path, *, service: str, drive_id: str, file_id: str, filename: str,
    credential_profile: str,
) -> ResearchArtifactResult:
    if service not in {"onedrive", "sharepoint"}:
        raise ResearchServiceError("Microsoft drive service must be OneDrive or SharePoint")
    drive = _opaque_identifier(drive_id, "drive id")
    selected = _opaque_identifier(file_id, "file id")
    token = _research_token(credential_profile)
    endpoint = (
        f"https://graph.microsoft.com/v1.0/drives/{quote(drive)}"
        f"/items/{quote(selected)}?$select=id,name,eTag,cTag,lastModifiedDateTime,size,file"
    )
    record = _api_json(endpoint, service=service, token=token)
    if str(record.get("id")) != selected:
        raise ResearchServiceError("Microsoft Graph metadata did not match the selected file")
    revision = str(record.get("eTag") or record.get("cTag") or record.get("lastModifiedDateTime") or "") or None
    hashes = _nested(record, "file", "hashes") or {}
    expected = None
    if isinstance(hashes, dict):
        if hashes.get("sha256Hash"):
            expected = f"sha256:{hashes['sha256Hash']}"
        elif hashes.get("sha1Hash"):
            expected = f"sha1:{hashes['sha1Hash']}"
    return import_selected_artifact(
        cwd, service=service, artifact_id=selected, file_id=selected,
        drive_id=drive, filename=filename, revision=revision,
        metadata={"drive": drive, "file": record.get("name") or filename,
                  "file_id": selected, "revision": revision},
        credential_profile=credential_profile, expected_checksum=expected,
    )


def import_box_file(
    cwd: Path, *, file_id: str, filename: str, credential_profile: str,
) -> ResearchArtifactResult:
    selected = _opaque_identifier(file_id, "Box file id")
    token = _research_token(credential_profile)
    record = _api_json(
        f"https://api.box.com/2.0/files/{quote(selected)}?fields=id,name,sha1,file_version,modified_at,size",
        service="box", token=token,
    )
    if str(record.get("id")) != selected:
        raise ResearchServiceError("Box metadata did not match the selected file")
    revision = str(_nested(record, "file_version", "id") or record.get("modified_at") or "") or None
    return import_selected_artifact(
        cwd, service="box", artifact_id=selected, file_id=selected,
        filename=filename, revision=revision,
        metadata={"file": record.get("name") or filename, "file_id": selected,
                  "version": revision}, credential_profile=credential_profile,
        expected_checksum=f"sha1:{record['sha1']}" if record.get("sha1") else None,
    )


def import_dropbox_file(
    cwd: Path, *, file_id: str, filename: str, revision: str,
    credential_profile: str,
) -> ResearchArtifactResult:
    """Download one selected Dropbox file id and preserve its known revision."""
    selected = _opaque_identifier(file_id, "Dropbox file id")
    rev = _opaque_identifier(revision, "Dropbox revision")
    return import_selected_artifact(
        cwd, service="dropbox", artifact_id=selected, file_id=selected,
        filename=filename, revision=rev,
        metadata={"file_id": selected, "revision": rev},
        download_url="https://content.dropboxapi.com/2/files/download",
        credential_profile=credential_profile, auth_mode="dropbox",
        form_fields={"path": selected},
    )


def import_redcap_export(
    cwd: Path, *, api_url: str, project_id: str, filename: str,
    credential_profile: str, report_id: str | None = None,
) -> ResearchArtifactResult:
    """Export records from exactly one token-scoped REDCap project/report."""
    project = _opaque_identifier(project_id, "REDCap project id")
    report = _opaque_identifier(report_id, "REDCap report id") if report_id else None
    parsed = urlsplit(api_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise ResearchServiceError("REDCap API URL must be credential-free HTTPS")
    fields = {
        "content": "report" if report else "record",
        "format": "csv", "type": "flat", "rawOrLabel": "raw",
        "rawOrLabelHeaders": "raw", "exportCheckboxLabel": "false",
        "returnFormat": "json",
    }
    if report:
        fields["report_id"] = report
    return import_selected_artifact(
        cwd, service="redcap", artifact_id=project,
        file_id=report or project, filename=filename,
        revision=report, metadata={"project_id": project, "report": report,
                                   "format": "csv"},
        download_url=api_url, credential_profile=credential_profile,
        auth_mode="redcap_form", form_fields=fields,
    )


def import_qualtrics_export(
    cwd: Path, *, datacenter: str, survey_id: str, filename: str,
    credential_profile: str, timeout_seconds: float = 300,
) -> ResearchArtifactResult:
    """Create, await, and import a CSV export for one selected survey."""
    survey = _opaque_identifier(survey_id, "Qualtrics survey id")
    dc = datacenter.strip().casefold()
    if not re.fullmatch(r"[a-z0-9-]{1,63}(?:\.[a-z0-9-]{1,63})*", dc):
        raise ResearchServiceError("invalid Qualtrics data-center host")
    host = dc if dc.endswith(".qualtrics.com") else f"{dc}.qualtrics.com"
    root = f"https://{host}/API/v3/responseexports"
    token = _research_token(credential_profile)
    created = _api_json(
        root, service="qualtrics", token=token, method="POST",
        payload={"surveyId": survey, "format": "csv"},
    )
    export_id = str(_nested(created, "result", "id") or "")
    export_id = _opaque_identifier(export_id, "Qualtrics export id")
    deadline = time.monotonic() + max(1.0, min(3600.0, timeout_seconds))
    while True:
        progress = _api_json(
            f"{root}/{quote(export_id)}", service="qualtrics", token=token,
        )
        raw_result = progress.get("result")
        result = raw_result if isinstance(raw_result, dict) else progress
        status = str(result.get("status") or "").casefold()
        percent = float(result.get("percentComplete") or 0)
        if status in {"failed", "error"}:
            raise ResearchServiceError("Qualtrics response export failed")
        if status == "complete" or percent >= 100:
            break
        if time.monotonic() >= deadline:
            raise ResearchServiceError("Qualtrics response export timed out")
        time.sleep(0.25)
    return import_selected_artifact(
        cwd, service="qualtrics", artifact_id=survey, file_id=export_id,
        filename=filename, revision=export_id,
        metadata={"survey": survey, "export": export_id, "format": "csv"},
        download_url=f"{root}/{quote(export_id)}/file",
        credential_profile=credential_profile, auth_mode="qualtrics",
    )


def import_kobo_export(
    cwd: Path, *, server_url: str, asset_uid: str, export_settings_uid: str,
    filename: str, credential_profile: str,
) -> ResearchArtifactResult:
    """Import one preconfigured synchronous export from one Kobo project."""
    asset = _opaque_identifier(asset_uid, "Kobo asset uid")
    setting = _opaque_identifier(export_settings_uid, "Kobo export settings uid")
    root = server_url.rstrip("/")
    parsed = urlsplit(root)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise ResearchServiceError("Kobo server URL must be credential-free HTTPS")
    url = (
        f"{root}/api/v2/assets/{quote(asset)}/export-settings/"
        f"{quote(setting)}/data.csv"
    )
    return import_selected_artifact(
        cwd, service="kobotoolbox", artifact_id=asset, file_id=setting,
        filename=filename, revision=setting,
        metadata={"project_id": asset, "export": setting, "format": "csv"},
        download_url=url, credential_profile=credential_profile, auth_mode="kobo",
    )


def import_openclinica_extract(
    cwd: Path, *, server_url: str, study_id: str, job_execution_id: str,
    filename: str, credential_profile: str,
) -> ResearchArtifactResult:
    """Download one already-created, permission-scoped OpenClinica extract."""
    study = _opaque_identifier(study_id, "OpenClinica study id")
    job = _opaque_identifier(job_execution_id, "OpenClinica extract job id")
    root = server_url.rstrip("/")
    parsed = urlsplit(root)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username:
        raise ResearchServiceError("OpenClinica server URL must be credential-free HTTPS")
    url = f"{root}/pages/auth/api/extractJobs/jobExecutions/{quote(job)}/dataset"
    return import_selected_artifact(
        cwd, service="openclinica", artifact_id=study, file_id=job,
        filename=filename, revision=job,
        metadata={"project_id": study, "export": job,
                  "format": Path(filename).suffix.lstrip(".")},
        download_url=url, credential_profile=credential_profile,
    )


def import_local_zotero_selection(
    cwd: Path,
    *,
    exported_items: Path,
    item_keys: list[str],
    attachment_paths: list[Path] | None = None,
) -> ResearchArtifactResult:
    """Import explicitly selected Zotero JSON and local attachments only."""
    cwd = Path(cwd).resolve(strict=True)
    source = Path(exported_items).expanduser()
    from sift.secure_file import copy_regular_no_follow, read_bytes_no_follow
    try:
        raw_export = read_bytes_no_follow(
            source, max_bytes=_MAX_ZOTERO_EXPORT_BYTES,
        )
    except OSError as e:
        message = (
            "Zotero export exceeds the 128 MiB safety limit"
            if "size limit" in str(e)
            else "Zotero export must be a regular, readable local JSON file"
        )
        raise ResearchServiceError(message) from e
    keys = {_identifier(key, "Zotero item key") for key in item_keys}
    if not keys or len(keys) > 1_000:
        raise ResearchServiceError("select between 1 and 1000 Zotero items")
    try:
        payload = json.loads(raw_export.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise ResearchServiceError("Zotero JSON export is unreadable") from e
    rows = payload if isinstance(payload, list) else payload.get("items", []) if isinstance(payload, dict) else []
    selected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_data = row.get("data")
        data = raw_data if isinstance(raw_data, dict) else row
        key = str(row.get("key") or data.get("key") or "")
        if key in keys:
            selected.append({
                "key": key,
                "version": row.get("version") or data.get("version"),
                "citationKey": data.get("citationKey"),
                "itemType": data.get("itemType"),
                "title": data.get("title"),
                "creators": data.get("creators", []),
                "date": data.get("date"),
                "DOI": data.get("DOI"),
                "ISBN": data.get("ISBN"),
                "ISSN": data.get("ISSN"),
                "url": data.get("url"),
                "tags": data.get("tags", []),
                "collections": data.get("collections", []),
                "relations": data.get("relations", {}),
            })
    found = {str(row["key"]) for row in selected}
    if found != keys:
        raise ResearchServiceError(f"selected Zotero keys were not found: {sorted(keys - found)}")

    target = cwd / "zotero_selection.json"
    if target.exists():
        target = cwd / f"zotero_selection_{hashlib.sha256(os.urandom(16)).hexdigest()[:8]}.json"
    fd, temporary_name = tempfile.mkstemp(prefix=".zotero-selection-", suffix=".json", dir=cwd)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(selected, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        Path(temporary_name).unlink(missing_ok=True)

    attachments = attachment_paths or []
    if len(attachments) > _MAX_ZOTERO_ATTACHMENTS:
        target.unlink(missing_ok=True)
        raise ResearchServiceError("select no more than 100 Zotero attachments")
    copied: list[str] = []
    attachment_hashes: dict[str, str] = {}
    try:
        total_attachment_bytes = 0
        for attachment in attachments:
            path = Path(attachment).expanduser()
            destination = cwd / Path(path.name).name
            if destination.exists():
                raise ResearchServiceError(
                    f"Zotero attachment name already exists: {destination.name}"
                )
            remaining = cloud_import_max_bytes() - total_attachment_bytes
            tmp_attachment = cwd / (
                ".zotero-attachment-"
                + hashlib.sha256(os.urandom(16)).hexdigest()[:16]
            )
            try:
                size, attachment_hash = copy_regular_no_follow(
                    path, tmp_attachment, max_bytes=max(0, remaining),
                )
                os.replace(tmp_attachment, destination)
            except OSError as e:
                message = (
                    "selected Zotero attachments exceed the configured import limit"
                    if "size limit" in str(e)
                    else "Zotero attachment must be a stable, regular local file"
                )
                raise ResearchServiceError(message) from e
            finally:
                Path(tmp_attachment).unlink(missing_ok=True)
            total_attachment_bytes += size
            copied.append(destination.name)
            attachment_hashes[destination.name] = attachment_hash
    except Exception:
        target.unlink(missing_ok=True)
        for name in copied:
            (cwd / name).unlink(missing_ok=True)
        raise

    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    from sift import release_ledger
    if not release_ledger.record_release(
        cwd,
        kind="research_artifact_import",
        tool="(local Zotero selection)",
        extra={
            "dataset": target.name,
            "service": "zotero_local",
            "item_keys": sorted(keys),
            "item_versions": {row["key"]: row["version"] for row in selected},
            "citation_keys": {row["key"]: row["citationKey"] for row in selected},
            "attachments": copied,
            "attachment_sha256": attachment_hashes,
            "dataset_sha256": digest,
            "network_used": False,
        },
    ):
        target.unlink(missing_ok=True)
        for name in copied:
            (cwd / name).unlink(missing_ok=True)
        raise ResearchServiceError("could not record Zotero import provenance")
    return ResearchArtifactResult(
        target, "zotero_local", ",".join(sorted(keys)), None, digest,
        {"citation_keys_preserved": "true", "network_used": "false"},
    )
