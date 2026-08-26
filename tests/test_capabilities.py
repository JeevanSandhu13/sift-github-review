"""Product-contract drift checks.

These tests make a product statement fail at build time when the enforcing
registry changes without the public support contract changing with it.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from sift.capabilities import (
    CAPABILITY_COMPATIBILITY_VERSION,
    DATABASE_SUPPLY,
    MODEL_SUPPLY,
    capabilities,
    product_contract,
)
from sift.data_request import SUPPORTED_REQUEST_TYPES
from sift.integrations import CLOUD_SOURCE_ADAPTERS, DATABASE_ADAPTERS
from sift.provider import SUPPORTED_PROVIDERS
from sift.sanitizer import supported_types
from sift.schema import DATA_EXTENSIONS
from sift.tools import friendly_tool_names


def _ids() -> set[str]:
    return {row.id for row in capabilities()}


def test_contract_is_json_serializable_and_has_unique_ids():
    contract = product_contract()
    encoded = json.dumps(contract, sort_keys=True)

    assert encoded
    ids = [row["id"] for row in contract["capabilities"]]
    assert len(ids) == len(set(ids))
    assert sum(contract["capability_counts"].values()) == len(ids)
    assert contract["compatibility_version"] == CAPABILITY_COMPATIBILITY_VERSION
    assert sum(contract["capability_status_counts"].values()) == len(ids)


def test_model_supply_is_explicitly_researcher_owned():
    assert MODEL_SUPPLY["models_included"] is False
    assert MODEL_SUPPLY["model_proxy_operated_by_sift"] is False
    assert MODEL_SUPPLY["credentials"] == "researcher_supplied"
    assert MODEL_SUPPLY["billing_relationship"].startswith("researcher_to_")

    model_rows = [row for row in capabilities() if row.category == "model_provider"]
    assert {row.id for row in model_rows} == {
        f"model-provider.{provider}" for provider in SUPPORTED_PROVIDERS
    }
    assert all("not included" in " ".join(row.limitations) for row in model_rows)


def test_database_supply_is_explicitly_researcher_owned():
    assert DATABASE_SUPPLY["databases_included"] is False
    assert DATABASE_SUPPLY["database_proxy_operated_by_sift"] is False
    assert DATABASE_SUPPLY["credentials"] == "researcher_supplied"
    assert DATABASE_SUPPLY["billing_relationship"].startswith("researcher_to_")
    assert DATABASE_SUPPLY["release_blocking"] is False

    contract = product_contract()
    assert contract["database_supply"] == DATABASE_SUPPLY
    database_rows = [
        row for row in capabilities() if row.category == "database"
    ]
    assert database_rows
    assert all(row.status == "supported" for row in database_rows)
    assert all(
        "researcher supplies" in " ".join(row.limitations).casefold()
        for row in database_rows
    )


def test_contract_covers_every_enforcement_registry():
    ids = _ids()

    assert {f"database.{row.id}" for row in DATABASE_ADAPTERS} <= ids
    assert {f"cloud-source.{row.id}" for row in CLOUD_SOURCE_ADAPTERS} <= ids
    assert {
        f"data-format.{extension.removeprefix('.')}" for extension in DATA_EXTENSIONS
    } <= ids
    assert {
        f"data-request.{request_type}" for request_type in SUPPORTED_REQUEST_TYPES
    } <= ids
    assert {
        f"analysis-result.{result_type}" for result_type in supported_types()
    } <= ids
    assert {
        f"research-tool.{name}" for name in friendly_tool_names(prefixed=False)
    } <= ids


def test_every_capability_has_an_importable_owner_and_verification():
    root = Path(__file__).resolve().parents[1]
    for row in capabilities():
        assert importlib.util.find_spec(row.implementation) is not None, row.id
        assert row.verification, row.id
        assert all((root / item).is_file() for item in row.verification), row.id
        assert row.claim.strip(), row.id
        assert all(item.strip() for item in row.limitations), row.id
        assert row.surfaces and all(item.strip() for item in row.surfaces), row.id
        assert row.acceptance_criteria, row.id
        assert all(item.strip() for item in row.acceptance_criteria), row.id
        assert row.compatibility_since <= CAPABILITY_COMPATIBILITY_VERSION


def test_contract_limits_are_positive_and_integrations_are_embedded():
    contract = product_contract()

    assert all(value > 0 for value in contract["limits"].values())
    assert {row["id"] for row in contract["integrations"]["model_adapters"]} == set(
        SUPPORTED_PROVIDERS
    )
    assert {row["id"] for row in contract["integrations"]["database_adapters"]} == {
        row.id for row in DATABASE_ADAPTERS
    }
    assert {
        row["id"] for row in contract["integrations"]["cloud_source_adapters"]
    } == {row.id for row in CLOUD_SOURCE_ADAPTERS}


def test_claims_are_scoped_enforced_evidenced_and_qualified():
    claims = product_contract()["claims"]

    assert claims
    assert len({claim["id"] for claim in claims}) == len(claims)
    for claim in claims:
        assert claim["statement"].strip()
        assert claim["scope"].strip()
        assert claim["enforcement"]
        assert claim["evidence"]
        assert claim["caveats"]
        assert claim["acceptance_criteria"]


def test_contract_defines_privacy_quality_scale_and_risks() -> None:
    contract = product_contract()
    privacy = contract["privacy_contract"]
    assert privacy["raw_data_guarantee"]["sift_direct_upload"] is False
    assert privacy["raw_data_guarantee"]["provider_raw_dataset_access"] is False
    assert privacy["explicit_attachments"]["researcher_confirmation_required"] is True
    assert privacy["operator_observability"]["database_operator"]
    assert privacy["control_responsibility"]["sift_enforced"]
    assert privacy["control_responsibility"]["provider_or_operator_controlled"]

    assert contract["quality_standards"]["statistical_correctness"]["human_review_required"] is True
    assert contract["quality_standards"]["reproducibility"]["minimum"]
    assert contract["quality_standards"]["provenance"]["minimum"]
    assert contract["supported_scale"]["files"]["drag_drop_file_bytes"] > 0
    assert contract["supported_scale"]["archives"]["spreadsheet_container_members"] > 0
    assert contract["supported_scale"]["outputs"]["result_payloads_per_run"] > 0

    risks = contract["risk_register"]
    assert {row["domain"] for row in risks} == {
        "privacy", "statistical", "integration", "reliability",
    }
    assert all(row["controls"] and row["evidence"] and row["residual_risk"] for row in risks)


def test_byo_database_support_is_independent_of_optional_live_certification() -> None:
    rows = {row.id: row for row in capabilities()}
    database_ids = {f"database.{row.id}" for row in DATABASE_ADAPTERS}
    assert all(rows[identifier].status == "supported" for identifier in database_ids)
    for identifier in (
        "cloud-source.s3",
        "cloud-source.gcs",
        "cloud-source.azure_blob",
        "cloud-source.https",
        "model-provider.openai",
        "model-provider.anthropic",
        "model-provider.gemini",
    ):
        assert rows[identifier].status in {"preview", "experimental"}, identifier

    contract = product_contract()
    incomplete = set(contract["advertised_but_not_fully_certified"])
    assert not database_ids & incomplete
    assert "model-provider.openai" in incomplete
    assert contract["implemented_but_not_surfaced"] == []


def test_external_release_evidence_capabilities_are_explicit_and_internal() -> None:
    rows = {row.id: row for row in capabilities()}
    assert rows["operations.live-database-certification"].status == "internal"
    assert rows["operations.independent-pentest-intake"].status == "internal"
    assert "optional program" in " ".join(
        rows["operations.live-database-certification"].limitations
    )
    assert "does not operate, fund, proxy, or require" in " ".join(
        rows["operations.live-database-certification"].limitations
    )
    assert "independent assessor" in " ".join(
        rows["operations.independent-pentest-intake"].limitations
    )
    assert "preflight" in rows["operations.live-database-certification"].claim
    assert "preflight" in rows["operations.independent-pentest-intake"].claim
    assert "no values" in " ".join(
        rows["operations.live-database-certification"].acceptance_criteria
    )
    assert "without creating assessor evidence" in " ".join(
        rows["operations.independent-pentest-intake"].acceptance_criteria
    )
