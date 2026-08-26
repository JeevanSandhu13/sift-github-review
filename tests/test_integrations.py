from __future__ import annotations

from pathlib import Path

import pytest

from sift import enterprise_policy as ep
from sift.integration_ids import MODEL_PROVIDER_IDS
from sift.integrations import (
    CLOUD_SOURCE_ADAPTERS,
    DATABASE_ADAPTERS,
    database_adapter_status,
    database_integration,
    endpoint_is_local,
    list_integration_manifests,
    model_integration,
    integration_contracts,
    integration_readiness_diagnostics,
)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "https://[::1]:9443/v1",
    ],
)
def test_only_literal_loopback_endpoints_are_local(url: str) -> None:
    assert endpoint_is_local(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://openrouter.ai/api/v1",
        "http://model.internal/v1",
        "file:///tmp/socket",
        "not a url",
        "",
    ],
)
def test_remote_or_ambiguous_endpoints_fail_closed(url: str) -> None:
    assert endpoint_is_local(url) is False


def test_compatible_manifest_tracks_actual_endpoint(monkeypatch) -> None:
    monkeypatch.setenv(
        "SIFT_OPENAI_COMPATIBLE_BASE_URL",
        "http://127.0.0.1:11434/v1",
    )
    assert model_integration("openai_compatible").location == "local"
    monkeypatch.setenv(
        "SIFT_OPENAI_COMPATIBLE_BASE_URL",
        "https://gateway.example/v1",
    )
    assert model_integration("openai_compatible").location == "remote_or_unverified"


def test_manifests_state_remote_data_flow_honestly() -> None:
    openai = model_integration("openai")
    assert openai.raw_dataset_access is False
    assert openai.sends_prompts is True
    assert openai.sends_sanitized_results is True
    assert openai.sends_user_attachments is True
    remote_db = database_integration("snowflake")
    assert remote_db.location == "remote"
    assert remote_db.raw_dataset_access is True
    assert any("log" in c.lower() for c in remote_db.cautions)
    anthropic = model_integration("anthropic")
    assert any("Fable 5" in caution and "30-day" in caution for caution in anthropic.cautions)
    gemini = model_integration("gemini")
    assert any("Unpaid" in caution and "confidential" in caution for caution in gemini.cautions)
    openai = model_integration("openai")
    assert "30 days" in openai.retention


def test_catalog_has_all_model_and_database_families() -> None:
    catalog = list_integration_manifests()
    assert tuple(m["id"] for m in catalog["models"]) == MODEL_PROVIDER_IDS
    assert {"postgresql", "mssql", "snowflake", "bigquery", "databricks"} <= {
        d["id"] for d in catalog["databases"]
    }


def test_redshift_adapter_uses_installed_driver_and_verified_tls() -> None:
    adapters = {
        row["id"]: row for row in list_integration_manifests()["database_adapters"]
    }
    redshift = adapters["redshift"]
    assert redshift["driver_module"] == "redshift_connector"
    assert "redshift+redshift_connector://" in redshift["uri_example"]
    assert "sslmode=verify-full" in redshift["uri_example"]


def test_plain_uri_driver_normalization_matches_advertised_adapter_drivers() -> None:
    from sift.connectors import _DECLARED_SQLALCHEMY_DRIVERS

    adapters = {adapter.id: adapter for adapter in DATABASE_ADAPTERS}
    assert _DECLARED_SQLALCHEMY_DRIVERS == {
        "postgresql": adapters["postgresql"].driver_module,
        "mysql": adapters["mysql"].driver_module,
        "mariadb": adapters["mariadb"].driver_module,
        "oracle": adapters["oracle"].driver_module,
        "redshift": adapters["redshift"].driver_module,
    }


def test_adapter_manifests_expose_trustworthy_setup_capabilities() -> None:
    catalog = list_integration_manifests()
    adapters = {row["id"]: row for row in catalog["database_adapters"]}
    assert "metadata_only_catalog" in adapters["postgresql"]["capabilities"]
    assert "protocol=tcps" in adapters["oracle"]["uri_example"]
    assert "ssl_server_dn_match=true" in adapters["oracle"]["uri_example"]
    assert "ocsp_fail_open=false" in adapters["snowflake"]["uri_example"]
    assert "Application Default Credentials" in adapters["bigquery"]["authentication"]

    models = {row["id"]: row for row in catalog["model_adapters"]}
    assert models["openai"]["conversation_state"].endswith("store=false")
    assert "grounding" in models["gemini"]["tools"]
    assert models["openai_compatible"]["vision"] == "endpoint_dependent"
    assert all(row["account_privacy_verifiable"] is False for row in models.values())
    assert all(row["model_access_included"] is False for row in models.values())
    assert all(row["credential_owner"] == "researcher" for row in models.values())
    assert all(row["lifecycle"][0] == "unconfigured" for row in models.values())


def test_adapter_readiness_checks_exact_dialect_module(monkeypatch) -> None:
    import importlib.util

    checked: list[str] = []

    def fake_find_spec(name: str):
        checked.append(name)
        return object()

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    assert all(row["installed"] for row in database_adapter_status())
    assert "snowflake.sqlalchemy" in checked
    assert "databricks.sql" in checked


def test_enterprise_local_only_rule_checks_endpoint(monkeypatch) -> None:
    policy = ep.EnterprisePolicy(require_local_model=True)
    monkeypatch.setenv(
        "SIFT_OPENAI_COMPATIBLE_BASE_URL",
        "https://gateway.example/v1",
    )
    assert ep.model_provider_allowed("openai_compatible", policy) is False
    monkeypatch.setenv(
        "SIFT_OPENAI_COMPATIBLE_BASE_URL",
        "http://localhost:11434/v1",
    )
    assert ep.model_provider_allowed("openai_compatible", policy) is True
    assert ep.model_provider_allowed("openai", policy) is False


def test_provider_factory_enforces_enterprise_policy(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        ep,
        "load_enterprise_policy",
        lambda: ep.EnterprisePolicy(
            allowed_model_providers=frozenset({"gemini"}),
        ),
    )
    from sift.provider import open_session

    with pytest.raises(PermissionError, match="enterprise policy"):
        open_session("openai", tmp_path, "gpt-5.6-sol", "system")


def test_every_integration_uses_one_security_and_lifecycle_contract() -> None:
    from sift.research_services import SERVICE_CONTRACTS

    contracts = integration_contracts()
    expected = (
        len(MODEL_PROVIDER_IDS) + len(DATABASE_ADAPTERS) + len(CLOUD_SOURCE_ADAPTERS)
        + len(SERVICE_CONTRACTS)
    )
    assert len(contracts) == expected
    assert {(row.kind, row.id) for row in contracts} == (
        {("model", provider) for provider in MODEL_PROVIDER_IDS}
        | {("database", row.id) for row in DATABASE_ADAPTERS}
        | {("object_storage", row.id) for row in CLOUD_SOURCE_ADAPTERS}
        | {("research_service", service) for service in SERVICE_CONTRACTS}
    )
    lifecycle = contracts[0].lifecycle
    for row in contracts:
        assert row.lifecycle == lifecycle
        assert row.authentication
        assert row.capabilities
        assert row.operations
        assert row.policy_boundary
        assert row.credential_boundary
        assert row.retention.disclosure
        assert row.residency.disclosure
        assert all(policy.timeout_seconds > 0 for policy in row.operations.values())
        # No current network operation automatically retries; this locks out
        # duplicate queries/downloads until an idempotent policy is explicit.
        assert all(policy.automatic_retries == 0 for policy in row.operations.values())


def test_research_service_manifests_expose_only_explicit_selection() -> None:
    rows = list_integration_manifests()["research_service_adapters"]
    assert {row["id"] for row in rows} >= {
        "zotero", "osf", "dataverse", "zenodo", "figshare", "dryad",
        "redcap", "qualtrics", "kobotoolbox", "openclinica",
        "google_drive", "onedrive", "sharepoint", "box", "dropbox",
    }
    assert all(row["explicit_selection_only"] for row in rows)
    assert all(row["account_listing_exposed"] is False for row in rows)
    assert all(row["materializes_locally_before_analysis"] for row in rows)


def test_integration_contracts_are_secret_free_json_metadata() -> None:
    import json

    payload = list_integration_manifests()["contracts"]
    encoded = json.dumps(payload, sort_keys=True)
    assert encoded
    assert "api_key" in encoded  # authentication METHOD is disclosed
    assert "secret_value" not in encoded
    assert "password_value" not in encoded
    assert all("data_flow" in row and "residency" in row for row in payload)


def test_model_contract_never_grants_connector_or_raw_upload_access() -> None:
    for row in integration_contracts():
        if row.kind != "model":
            continue
        assert row.data_flow.remote_receives_raw_dataset is False
        assert row.data_flow.generated_code_network_access is False
        assert "database" not in row.capabilities
        assert "object_storage" not in row.capabilities


def test_actual_model_tool_registry_has_no_data_connector_entrypoint() -> None:
    """Lock the executable boundary, not only the metadata contract."""
    import inspect

    import sift.tools as model_tools

    names = set(model_tools.HANDLERS)
    assert not any(
        token in name
        for name in names
        for token in ("database", "connector", "cloud", "bucket", "object_store")
    )
    source = inspect.getsource(model_tools)
    assert "from sift import connectors" not in source
    assert "from sift import cloud_sources" not in source
    assert "from sift.connectors" not in source
    assert "from sift.cloud_sources" not in source


def test_readiness_is_bounded_structured_and_never_connects(monkeypatch) -> None:
    from sift.research_services import SERVICE_CONTRACTS

    monkeypatch.setattr(
        "sift.provider.all_provider_readiness",
        lambda: {
            provider: {
                "state": "needs_credentials",
                "ready": False,
                "issues": ["credential_required"],
                "auth_mode": "unknown",
            }
            for provider in MODEL_PROVIDER_IDS
        },
    )
    rows = integration_readiness_diagnostics()
    expected = (
        len(MODEL_PROVIDER_IDS) + len(DATABASE_ADAPTERS) + len(CLOUD_SOURCE_ADAPTERS)
        + len(SERVICE_CONTRACTS)
    )
    assert len(rows) == expected
    assert len({(row["kind"], row["integration_id"]) for row in rows}) == (
        expected
    )
    for row in rows:
        assert row["state"] in {
            "needs_credentials", "needs_configuration", "unavailable",
        }
        assert isinstance(row["ready"], bool)
        assert isinstance(row["issues"], list)
        assert isinstance(row["diagnostics"], dict)
        serialized = str(row).casefold()
        assert "password=" not in serialized
        assert "api_key=" not in serialized
