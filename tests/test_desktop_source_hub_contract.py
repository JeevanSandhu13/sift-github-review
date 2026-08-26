"""Regression contract for the desktop source hub and privacy walkthrough."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sift.ui import SiftBridge


WEB = Path(__file__).parents[1] / "src" / "sift" / "web"


def test_source_hub_assets_are_loaded_after_the_existing_shell() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    assert html.index('href="desktop-shell.css"') < html.index('href="sources.css"')
    assert html.index('src="app.js"') < html.index('src="sources.js"')
    assert (WEB / "sources.css").is_file()
    assert (WEB / "sources.js").is_file()


def test_source_hub_exposes_each_supported_source_family() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "sources.js").read_text(encoding="utf-8")
    for tab in ("local", "database", "cloud", "research"):
        assert f'data-source-tab="{tab}"' in html
    for integration in (
        "postgresql", "mssql", "snowflake", "bigquery", "databricks",
        "s3", "gcs", "azure_blob", "sftp", "zotero", "osf",
        "google_drive", "dropbox", "redcap", "qualtrics",
    ):
        assert integration in javascript


def test_landing_use_cases_prepare_real_analysis_prompts() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "sources.js").read_text(encoding="utf-8")
    for use_case in ("understand", "verify", "compare", "report"):
        assert f'data-use-case="{use_case}"' in html
        assert f"{use_case}:" in javascript
    assert "applyPendingUseCase" in javascript
    assert "You can change it before sending" in html


def test_connector_catalog_has_search_groups_and_specific_copy() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    javascript = (WEB / "sources.js").read_text(encoding="utf-8")
    assert 'id="source-search-input"' in html
    assert "CONNECTOR_GROUPS" in javascript
    assert "Dry-run byte cost before importing a query" in javascript
    assert "Import one completed extract from a study" in javascript
    assert "You choose the exact data to copy locally" not in javascript


def test_connector_setup_is_guided_and_recovers_from_render_failures() -> None:
    javascript = (WEB / "sources.js").read_text(encoding="utf-8")
    css = (WEB / "sources.css").read_text(encoding="utf-8")
    assert "if (!options.textarea) input.type" in javascript
    assert "source-config-back" in javascript
    assert "Could not open ${manifest.label || fallback}" in javascript
    assert "Choose database file" in javascript
    assert "1 · Test connection" in javascript
    assert "2 · Check query" in javascript
    assert "3 · Import data" in javascript
    assert ".source-setup-help" in css


def test_walkthrough_covers_the_actual_product_boundaries() -> None:
    javascript = (WEB / "sources.js").read_text(encoding="utf-8").casefold()
    for concept in (
        "Raw observations remain on this computer",
        "Generated analysis code has no network access",
        "Bring your own model",
        "read-only database extraction",
        "Verify and export",
    ):
        assert concept.casefold() in javascript


def test_feedback_and_note_bridges_are_not_public_product_functions() -> None:
    html = (WEB / "index.html").read_text(encoding="utf-8").casefold()
    javascript = (WEB / "app.js").read_text(encoding="utf-8")
    assert "feedback-btn" not in html
    assert "leave a note" not in html
    assert "send_feedback" not in javascript
    assert "feedback_available" not in javascript
    assert not hasattr(SiftBridge, "send_feedback")
    assert not hasattr(SiftBridge, "feedback_available")
    assert not hasattr(SiftBridge, "add_research_note")
    assert not hasattr(SiftBridge, "list_research_notes")


def test_empty_session_is_available_for_connector_first_workflows(tmp_path, monkeypatch) -> None:
    created = tmp_path / "connector-session"
    created.mkdir()
    monkeypatch.setattr("sift.ui._new_session_dir", lambda: created)
    bridge = SiftBridge()
    result = bridge.start_empty_session()
    assert result["ok"] is True
    assert bridge.cwd == created.resolve()


def test_source_hub_database_bridge_materializes_a_real_read_only_query(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    database = tmp_path / "research.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE observations (id INTEGER, score REAL)")
        connection.executemany(
            "INSERT INTO observations VALUES (?, ?)",
            [(1, 2.5), (2, 4.0), (3, 8.25)],
        )

    bridge = SiftBridge(workspace)
    result = bridge.run_database_extract(
        f"sqlite:///{database}",
        "SELECT id, score FROM observations ORDER BY id",
        "database_extract.parquet",
    )

    assert result["ok"] is True
    assert result["rows"] == 3
    assert result["columns"] == 2
    assert result["dataset"] == "database_extract.parquet"
    assert (workspace / "database_extract.parquet").is_file()
    assert len(result["dataset_sha256"]) == 64


def test_local_database_picker_builds_uri_without_manual_syntax(tmp_path) -> None:
    database = tmp_path / "study.sqlite"
    database.touch()

    class Window:
        def create_file_dialog(self, *_args, **_kwargs):
            return (str(database),)

    bridge = SiftBridge()
    bridge._window = Window()  # type: ignore[attr-defined]
    result = bridge.choose_database_file("sqlite")

    assert result["ok"] is True
    assert result["display"] == "study.sqlite"
    from sqlalchemy.engine import make_url

    connection = result["connection"]
    assert make_url(connection).database == database.resolve().as_posix()
    assert "\\" not in connection
