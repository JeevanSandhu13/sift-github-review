from __future__ import annotations

import hashlib
import json
from email.message import Message
from pathlib import Path
from urllib.request import ProxyHandler, Request

import pytest

from sift import research_services as services
from sift.cloud_sources import CloudImportResult, CloudSourceError


def test_metadata_transport_ignores_ambient_proxy_and_pins_validated_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Response:
        headers = {}

        def geturl(self):
            return "https://api.osf.io/v2/files/file-1/"

        def read(self, _limit):
            return b'{"data":{"id":"file-1"}}'

        def close(self):
            return None

    class Opener:
        def open(self, request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return Response()

    def opener(*handlers):
        captured["handlers"] = handlers
        return Opener()

    monkeypatch.setenv("HTTPS_PROXY", "https://ambient-proxy.invalid:8443")
    monkeypatch.delenv("SIFT_TRUST_HTTPS_IMPORT_PROXY", raising=False)
    monkeypatch.setattr(services, "_validate_https_endpoint", lambda _url: None)
    monkeypatch.setattr(services, "build_opener", opener)

    assert services._api_json(
        "https://api.osf.io/v2/files/file-1/",
        service="osf", token="scoped-secret",
    )["data"]["id"] == "file-1"
    handlers = captured["handlers"]
    proxy = next(item for item in handlers if isinstance(item, ProxyHandler))
    assert proxy.proxies == {}
    assert any(
        isinstance(item, services._PinnedHTTPSHandler) for item in handlers
    )
    assert captured["request"].get_header("Authorization") == (
        "Bearer scoped-secret"
    )

    redirect = next(
        item for item in handlers
        if isinstance(item, services._SafeRedirectHandler)
    )
    with pytest.raises(services.ResearchServiceError, match="escaped"):
        redirect.redirect_request(
            Request("https://api.osf.io/v2/files/file-1/"),
            None, 302, "Found", Message(), "https://evil.example/file-1",
        )


@pytest.mark.parametrize(
    ("service", "kwargs", "fragment"),
    [
        ("figshare", {"artifact_id": "article-1", "file_id": "22"}, "/22"),
        ("dryad", {"artifact_id": "doi:10.1/x", "file_id": "33"}, "/33/download"),
        ("google_drive", {"artifact_id": "file-44"}, "file-44?alt=media"),
        ("onedrive", {"artifact_id": "item-1", "drive_id": "drive-1"}, "/items/item-1/content"),
        ("box", {"artifact_id": "55"}, "/55/content"),
        (
            "dataverse",
            {"artifact_id": "doi:10.1/x", "file_id": "66", "base_url": "https://data.example.edu"},
            "/api/access/datafile/66",
        ),
    ],
)
def test_selected_download_urls_never_list_accounts(
    service: str,
    kwargs: dict[str, str],
    fragment: str,
) -> None:
    url = services.selected_download_url(service, **kwargs)
    assert fragment in url
    assert not url.endswith("/files")


def test_service_download_url_cannot_escape_approved_operator() -> None:
    with pytest.raises(services.ResearchServiceError, match="approved OSF"):
        services.selected_download_url(
            "osf",
            artifact_id="node-1",
            download_url="https://evil.example/data.csv",
        )
    with pytest.raises(services.ResearchServiceError, match="HTTPS"):
        services.selected_download_url(
            "dataverse",
            artifact_id="doi:10.1/x",
            file_id="1",
            base_url="http://data.example.edu",
        )


def test_selected_artifact_is_local_before_research_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "selected.csv"
    dataset.write_text("id,value\n1,2\n", encoding="utf-8")
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    seen: dict[str, object] = {}

    def fake_import(cwd: Path, **kwargs):
        seen.update(kwargs)
        assert dataset.is_file()
        return CloudImportResult(
            dataset, "https", "https://api.figshare.com/file", dataset.stat().st_size,
            digest, "revision-7", "text/csv",
        )

    monkeypatch.setattr(services, "import_cloud_dataset", fake_import)
    result = services.import_selected_artifact(
        tmp_path,
        service="figshare",
        artifact_id="article-1",
        file_id="7",
        filename="selected.csv",
        revision="v3",
        metadata={"doi": "10.6084/example", "license": "CC-BY", "ignored": "x"},
    )
    assert result.dataset_path == dataset
    assert result.metadata == {"doi": "10.6084/example", "license": "CC-BY", "revision": "v3"}
    assert seen["dataset_name"] == "selected.csv"
    from sift.release_ledger import read_ledger

    records = read_ledger(tmp_path)
    research_record = next(
        record for record in records
        if record["kind"] == "research_artifact_import"
    )
    assert research_record["extra"]["canonical_fingerprint"] == result.canonical_fingerprint
    assert "ignored" not in json.dumps(records)


def test_local_zotero_selection_preserves_metadata_without_network(
    tmp_path: Path,
) -> None:
    export = tmp_path / "library-export.json"
    export.write_text(json.dumps([
        {
            "key": "AAAA1111", "version": 4,
            "data": {
                "itemType": "journalArticle", "title": "Selected study",
                "citationKey": "Sandhu2026", "DOI": "10.1/selected",
                "creators": [{"creatorType": "author", "name": "Researcher"}],
            },
        },
        {
            "key": "BBBB2222", "version": 9,
            "data": {"title": "Private unselected item", "citationKey": "Private2026"},
        },
    ]), encoding="utf-8")
    attachment = tmp_path / "selected-note.txt"
    attachment.write_text("local attachment", encoding="utf-8")
    session = tmp_path / "session"
    session.mkdir()

    result = services.import_local_zotero_selection(
        session,
        exported_items=export,
        item_keys=["AAAA1111"],
        attachment_paths=[attachment],
    )
    written = json.loads(result.dataset_path.read_text(encoding="utf-8"))
    assert written[0]["citationKey"] == "Sandhu2026"
    assert written[0]["DOI"] == "10.1/selected"
    assert "Private unselected item" not in result.dataset_path.read_text(encoding="utf-8")
    assert (session / "selected-note.txt").read_text(encoding="utf-8") == "local attachment"
    from sift.release_ledger import read_ledger

    record = read_ledger(session)[0]
    assert record["extra"]["network_used"] is False
    assert record["extra"]["citation_keys"] == {"AAAA1111": "Sandhu2026"}
    assert record["extra"]["attachment_sha256"] == {
        "selected-note.txt": hashlib.sha256(b"local attachment").hexdigest(),
    }


def test_zotero_requires_explicit_bounded_selection(tmp_path: Path) -> None:
    export = tmp_path / "library.json"
    export.write_text("[]", encoding="utf-8")
    session = tmp_path / "session"
    session.mkdir()
    with pytest.raises(services.ResearchServiceError, match="select between"):
        services.import_local_zotero_selection(
            session, exported_items=export, item_keys=[],
        )


def test_repository_checksum_is_enforced_before_cloud_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen: dict[str, object] = {}

    def reject_before_commit(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen.update(kwargs)
        raise CloudSourceError("repository checksum did not match the selected file")

    monkeypatch.setattr(services, "import_cloud_dataset", reject_before_commit)
    with pytest.raises(services.ResearchServiceError, match="checksum"):
        services.import_selected_artifact(
            tmp_path, service="figshare", artifact_id="1", file_id="2",
            filename="selected.csv", expected_checksum="md5:" + "0" * 32,
        )
    assert seen["_expected_checksum"] == "md5:" + "0" * 32
    assert not (tmp_path / "selected.csv").exists()


def test_osf_import_reads_only_selected_file_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen: list[str] = []
    monkeypatch.setattr(services, "_research_token", lambda profile: "secret")

    def api(url: str, **kwargs):
        seen.append(url)
        assert kwargs["token"] == "secret"
        return {"data": {"id": "file1", "attributes": {
            "name": "study.csv", "version": "3",
        }, "relationships": {"node": {"data": {"id": "node1"}}},
            "links": {"download": "https://files.osf.io/v1/resources/node1/providers/osfstorage/file1"}}}

    captured: dict[str, object] = {}
    monkeypatch.setattr(services, "_api_json", api)
    monkeypatch.setattr(services, "import_selected_artifact", lambda cwd, **kwargs: (
        captured.update(kwargs) or services.ResearchArtifactResult(
            tmp_path / "study.csv", "osf", "node1", "v3", "a" * 64, {},
        )
    ))
    services.import_osf_file(
        tmp_path, node_id="node1", file_id="file1", version_id="v3",
        filename="study.csv", credential_profile="OSF",
    )
    assert seen == ["https://api.osf.io/v2/files/file1/versions/v3/"]
    assert captured["download_url"].startswith("https://files.osf.io/")
    assert "/nodes" not in seen[0]


def test_dataverse_preserves_exact_version_license_and_checksum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(services, "_research_token", lambda profile: None)
    monkeypatch.setattr(services, "_api_json", lambda url, **kwargs: {"data": {
        "versionNumber": 4, "license": {"name": "CC BY 4.0"},
        "files": [{"dataFile": {"id": 9, "filename": "data.csv",
            "checksum": {"type": "MD5", "value": "a" * 32}}}],
    }})
    captured: dict[str, object] = {}
    monkeypatch.setattr(services, "import_selected_artifact", lambda cwd, **kwargs: (
        captured.update(kwargs) or services.ResearchArtifactResult(
            tmp_path / "data.csv", "dataverse", "doi:10.1/x", "4", "b" * 64, {},
        )
    ))
    services.import_dataverse_file(
        tmp_path, base_url="https://data.example.edu", persistent_id="doi:10.1/x",
        dataset_version="4.0", file_id="9", filename="data.csv",
    )
    assert captured["revision"] == "4"
    assert captured["metadata"]["license"] == "CC BY 4.0"
    assert captured["expected_checksum"] == "md5:" + "a" * 32


@pytest.mark.parametrize(
    ("service", "call", "metadata_url_fragment"),
    [
        ("google_drive", lambda p: services.import_google_drive_file(
            p, file_id="g1", filename="g.csv", credential_profile="drive"),
         "/drive/v3/files/g1?fields="),
        ("box", lambda p: services.import_box_file(
            p, file_id="b1", filename="b.csv", credential_profile="box"),
         "/2.0/files/b1?fields="),
    ],
)
def test_drive_adapters_fetch_only_selected_file_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, service: str, call,
    metadata_url_fragment: str,
) -> None:
    monkeypatch.setattr(services, "_research_token", lambda profile: "secret")
    seen: list[str] = []

    def api(url: str, **kwargs):
        seen.append(url)
        if service == "google_drive":
            return {"id": "g1", "name": "g.csv", "version": "11"}
        return {"id": "b1", "name": "b.csv", "sha1": "a" * 40,
                "file_version": {"id": "v2"}}

    monkeypatch.setattr(services, "_api_json", api)
    monkeypatch.setattr(services, "import_selected_artifact", lambda cwd, **kwargs:
        services.ResearchArtifactResult(tmp_path / kwargs["filename"], service,
                                        kwargs["artifact_id"], kwargs["revision"],
                                        "c" * 64, {}))
    call(tmp_path)
    assert len(seen) == 1 and metadata_url_fragment in seen[0]
    assert not seen[0].rstrip("/").endswith("files")


def test_scoped_exports_never_call_account_listing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    captured: list[dict[str, object]] = []

    def imported(cwd, **kwargs):
        captured.append(kwargs)
        return services.ResearchArtifactResult(
            tmp_path / kwargs["filename"], kwargs["service"], kwargs["artifact_id"],
            kwargs.get("revision"), "d" * 64, {},
        )

    monkeypatch.setattr(services, "import_selected_artifact", imported)
    services.import_redcap_export(
        tmp_path, api_url="https://redcap.example.edu/api/", project_id="42",
        report_id="7", filename="report.csv", credential_profile="redcap",
    )
    services.import_kobo_export(
        tmp_path, server_url="https://kf.kobotoolbox.org", asset_uid="a1",
        export_settings_uid="e1", filename="kobo.csv", credential_profile="kobo",
    )
    services.import_openclinica_extract(
        tmp_path, server_url="https://study.openclinica.io/OpenClinica",
        study_id="S1", job_execution_id="J1", filename="study.xml",
        credential_profile="oc",
    )
    services.import_dropbox_file(
        tmp_path, file_id="id:file1", revision="rev1", filename="dropbox.csv",
        credential_profile="dropbox",
    )
    assert captured[0]["auth_mode"] == "redcap_form"
    assert captured[0]["form_fields"]["report_id"] == "7"
    assert "/assets/a1/export-settings/e1/data.csv" in captured[1]["download_url"]
    assert "/jobExecutions/J1/dataset" in captured[2]["download_url"]
    assert captured[3]["download_url"] == "https://content.dropboxapi.com/2/files/download"
    assert captured[3]["form_fields"] == {"path": "id:file1"}
    assert captured[3]["auth_mode"] == "dropbox"


def test_qualtrics_three_step_export_is_one_selected_survey(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(services, "_research_token", lambda profile: "secret")
    calls: list[tuple[str, str]] = []

    def api(url: str, **kwargs):
        calls.append((kwargs.get("method", "GET"), url))
        if kwargs.get("method") == "POST":
            assert kwargs["payload"] == {"surveyId": "SV_1", "format": "csv"}
            return {"result": {"id": "EX_1"}}
        return {"result": {"status": "complete", "percentComplete": 100}}

    monkeypatch.setattr(services, "_api_json", api)
    captured: dict[str, object] = {}
    monkeypatch.setattr(services, "import_selected_artifact", lambda cwd, **kwargs: (
        captured.update(kwargs) or services.ResearchArtifactResult(
            tmp_path / "q.csv", "qualtrics", "SV_1", "EX_1", "e" * 64, {},
        )
    ))
    services.import_qualtrics_export(
        tmp_path, datacenter="ca1", survey_id="SV_1", filename="q.csv",
        credential_profile="qualtrics",
    )
    assert calls == [
        ("POST", "https://ca1.qualtrics.com/API/v3/responseexports"),
        ("GET", "https://ca1.qualtrics.com/API/v3/responseexports/EX_1"),
    ]
    assert captured["download_url"].endswith("/EX_1/file")
    assert captured["auth_mode"] == "qualtrics"
