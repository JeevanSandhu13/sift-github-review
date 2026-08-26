"""Machine-readable trust contracts for Sift integrations.

An integration is not trustworthy merely because it uses TLS.  Researchers
need to know which process receives which class of data, whether the target is
local, where credentials live, and which controls Sift can actually enforce.
This module is the single source for those answers.  It intentionally avoids
claims about a vendor account's retention configuration, which Sift cannot
inspect or prove from an API key.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from sift.integration_core import (
    AuthenticationMethod,
    DataFlowContract,
    IntegrationContract,
    IntegrationState,
    OperationPolicy,
    ResidencyContract,
    RetentionContract,
)
from sift.integration_ids import MODEL_PROVIDER_IDS


@dataclass(frozen=True)
class IntegrationTrust:
    id: str
    kind: str
    label: str
    location: str
    raw_dataset_access: bool
    sends_prompts: bool
    sends_sanitized_results: bool
    sends_user_attachments: bool
    credentials: str
    retention: str
    guarantees: tuple[str, ...]
    cautions: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_REMOTE_MODEL_COMMON = (
    "Generated analysis code cannot use the network",
    "Model tools cannot open raw dataset files directly",
    "Statistical tool results pass through Sift disclosure controls",
)

MODEL_TRUST: dict[str, IntegrationTrust] = {
    "anthropic": IntegrationTrust(
        id="anthropic",
        kind="model",
        label="Anthropic Claude",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="Researcher's Claude Console API key in OS keyring or environment",
        retention="Controlled by the Anthropic account/product agreement",
        guarantees=_REMOTE_MODEL_COMMON,
        cautions=(
            "Prompts, sanitized results, and explicitly attached images are sent to Anthropic",
            "Sift does not reuse Claude consumer subscription OAuth; it accepts a Claude API credential only",
            "Anthropic ZDR is an organization-level commercial API arrangement that Sift cannot infer from a credential",
            "Claude Fable 5 is a Covered Model requiring 30-day retention and is not ZDR eligible",
        ),
    ),
    "openai": IntegrationTrust(
        id="openai",
        kind="model",
        label="OpenAI",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="OS keyring or environment",
        retention="Sift requests store=false by default; OpenAI may still retain abuse-monitoring logs for up to 30 days unless approved project controls apply",
        guarantees=_REMOTE_MODEL_COMMON,
        cautions=(
            "Prompts, sanitized results, and explicitly attached images are sent to OpenAI",
            "Zero Data Retention or Modified Abuse Monitoring requires provider approval and project configuration; Sift cannot infer it from the key",
            "Image inputs may be retained for manual review when provider safety classifiers flag them, including under ZDR",
        ),
    ),
    "gemini": IntegrationTrust(
        id="gemini",
        kind="model",
        label="Google Gemini",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="OS keyring or environment",
        retention="Paid Gemini services do not use prompts or responses to improve Google products, but limited abuse-monitoring logs apply unless the project is approved for ZDR",
        guarantees=_REMOTE_MODEL_COMMON,
        cautions=(
            "Prompts, sanitized results, and explicitly attached images are sent to Google",
            "Unpaid Gemini services may use submitted content and responses to improve products and may involve human review; do not use them for confidential data",
            "Paid-service status and project ZDR approval cannot be inferred from an API key",
            "Sift does not use the File API, explicit context caching, grounding, or the stateful Interactions API",
        ),
    ),
    "azure_openai": IntegrationTrust(
        id="azure_openai",
        kind="model",
        label="Azure OpenAI",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="Azure API key or Microsoft Entra managed/workload identity",
        retention="Controlled by the Azure resource, deployment type, abuse-monitoring configuration, and Microsoft agreement",
        guarantees=_REMOTE_MODEL_COMMON + (
            "Direct OpenAI credentials are never consulted",
            "Sift sets store=false and replays conversation state locally",
        ),
        cautions=(
            "Prompts, sanitized results, and explicit images are processed by Microsoft Azure",
            "Global and DataZone deployments have wider processing geographies than regional deployments",
            "Modified abuse monitoring and content-logging state cannot be inferred by Sift",
        ),
    ),
    "vertex_gemini": IntegrationTrust(
        id="vertex_gemini",
        kind="model",
        label="Vertex AI Gemini",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="Google Application Default Credentials or workload identity",
        retention="Controlled by the Google Cloud project, enabled Vertex features, and its data-processing agreement",
        guarantees=_REMOTE_MODEL_COMMON + (
            "Direct Gemini API keys are never consulted",
            "Grounding, File API, explicit caching, and session resumption are disabled",
        ),
        cautions=(
            "Prompts, sanitized results, and explicit images are processed by Google Cloud",
            "Abuse-monitoring/ZDR status cannot be inferred from ADC",
        ),
    ),
    "bedrock_anthropic": IntegrationTrust(
        id="bedrock_anthropic",
        kind="model",
        label="Amazon Bedrock Claude",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="AWS default credential chain, IAM role, or temporary credentials",
        retention="Amazon Bedrock documents that Converse prompts and completions are not stored or logged",
        guarantees=_REMOTE_MODEL_COMMON + (
            "Direct Anthropic credentials are never consulted",
            "Conversation history is retained locally by Sift",
        ),
        cautions=(
            "Prompts, sanitized results, and explicit images are processed by Amazon Bedrock",
            "Effective IAM permissions and network routing cannot be inferred without cloud calls",
        ),
    ),
    "vertex_anthropic": IntegrationTrust(
        id="vertex_anthropic",
        kind="model",
        label="Vertex AI Claude",
        location="remote",
        raw_dataset_access=False,
        sends_prompts=True,
        sends_sanitized_results=True,
        sends_user_attachments=True,
        credentials="Google Application Default Credentials or workload identity",
        retention="Controlled by the Google Cloud Vertex AI partner-model processing contract and project logging settings",
        guarantees=_REMOTE_MODEL_COMMON + (
            "Direct Anthropic credentials are never consulted",
            "Conversation history is retained locally by Sift",
        ),
        cautions=(
            "Prompts, sanitized results, and explicit images are processed through Google Cloud's Claude partner-model service",
            "Vertex Claude model availability and retirement differ from the direct Anthropic API",
        ),
    ),
}


def endpoint_is_local(base_url: str | None) -> bool:
    """Return True only for an unambiguously loopback HTTP endpoint.

    Hostnames other than literal ``localhost`` are not trusted as local:
    DNS and hosts-file mappings can change.  Unparseable URLs fail closed.
    """
    if not base_url:
        return False
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
    except (TypeError, ValueError):
        return False
    if parsed.scheme not in {"http", "https"} or not host:
        return False
    if host.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def model_integration(provider: str) -> IntegrationTrust:
    if provider != "openai_compatible":
        try:
            profile = MODEL_TRUST[provider]
        except KeyError as e:
            raise ValueError(f"unknown model provider: {provider!r}") from e
        return profile

    base_url = os.environ.get("SIFT_OPENAI_COMPATIBLE_BASE_URL")
    local = endpoint_is_local(base_url)
    return IntegrationTrust(
        id="openai_compatible",
        kind="model",
        label="Local / OpenAI-compatible",
        location="local" if local else "remote_or_unverified",
        raw_dataset_access=False,
        sends_prompts=not local,
        sends_sanitized_results=not local,
        sends_user_attachments=not local,
        credentials="OS keyring or environment",
        retention=(
            "Local process memory and Sift's local session history"
            if local
            else "Determined entirely by the configured endpoint operator"
        ),
        guarantees=_REMOTE_MODEL_COMMON,
        cautions=(
            "Only localhost or a literal loopback IP is classified as local",
            "A gateway such as OpenRouter is a remote processor even though it uses an OpenAI-compatible API",
        ),
    )


def provider_is_local(provider: str) -> bool:
    return model_integration(provider).location == "local"


DATABASE_LABELS: dict[str, str] = {
    "sqlite": "SQLite",
    "duckdb": "DuckDB",
    "duckdb-file": "DuckDB files",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL / MariaDB",
    "mariadb": "MariaDB",
    "mssql": "Microsoft SQL Server",
    "oracle": "Oracle",
    "snowflake": "Snowflake",
    "bigquery": "Google BigQuery",
    "redshift": "Amazon Redshift",
    "databricks": "Databricks SQL",
}


@dataclass(frozen=True)
class DatabaseAdapter:
    id: str
    label: str
    install_extra: str
    uri_example: str
    driver_module: str
    os_notes: str = "Available on macOS, Windows, and Linux"
    authentication: str = "Database-driver credentials or managed identity"
    tls_requirement: str = "Verified TLS required for remote connections"
    capabilities: tuple[str, ...] = (
        "connection_test",
        "metadata_only_catalog",
        "read_only_extract",
        "local_parquet_materialization",
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DATABASE_ADAPTERS: tuple[DatabaseAdapter, ...] = (
    DatabaseAdapter(
        "sqlite",
        "SQLite",
        "built-in",
        "sqlite:////path/research.db",
        "sqlite3",
        authentication="None; local file permissions apply",
        tls_requirement="Not applicable; local file only",
    ),
    DatabaseAdapter(
        "duckdb",
        "DuckDB",
        "built-in",
        "/path/research.duckdb",
        "duckdb",
        authentication="None for local databases and files",
        tls_requirement="Not applicable to local DuckDB files",
    ),
    DatabaseAdapter(
        "postgresql",
        "PostgreSQL",
        "postgres",
        "postgresql+psycopg://user@host/db?sslmode=verify-full",
        "psycopg",
    ),
    DatabaseAdapter(
        "mysql",
        "MySQL",
        "mysql",
        "mysql+pymysql://user@host/db?ssl_verify_cert=true&ssl_verify_identity=true",
        "pymysql",
    ),
    DatabaseAdapter(
        "mariadb",
        "MariaDB",
        "mysql",
        "mariadb+pymysql://user@host/db?ssl_verify_cert=true&ssl_verify_identity=true",
        "pymysql",
        authentication="Password, certificate, or driver-managed identity",
    ),
    DatabaseAdapter(
        "mssql",
        "Microsoft SQL Server",
        "sqlserver",
        "mssql+pyodbc://user@host/db?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no",
        "pyodbc",
        "Python support is cross-platform; Microsoft's ODBC Driver 18 must also be installed on the host",
    ),
    DatabaseAdapter(
        "oracle",
        "Oracle",
        "oracle",
        "oracle+oracledb://user@host:1522/?service_name=research&protocol=tcps&ssl_server_dn_match=true",
        "oracledb",
        authentication="Password, wallet/mTLS, or driver-managed Oracle identity",
    ),
    DatabaseAdapter(
        "snowflake",
        "Snowflake",
        "snowflake",
        "snowflake://user@account/database/schema?warehouse=research&ocsp_fail_open=false",
        "snowflake.sqlalchemy",
        authentication="Password, SSO/OAuth, key pair, or driver-managed identity",
        tls_requirement="Verified TLS with fail-closed OCSP is required",
    ),
    DatabaseAdapter(
        "bigquery",
        "Google BigQuery",
        "bigquery",
        "bigquery://project-id",
        "sqlalchemy_bigquery",
        authentication="Google Application Default Credentials or workload identity",
        tls_requirement="Google client library HTTPS transport",
    ),
    DatabaseAdapter(
        "redshift",
        "Amazon Redshift",
        "redshift",
        "redshift+redshift_connector://user@host:5439/db?ssl=true&sslmode=verify-full",
        "redshift_connector",
    ),
    DatabaseAdapter(
        "databricks",
        "Databricks SQL",
        "databricks",
        "databricks://token:<access-token>@host:443?http_path=/sql/1.0/warehouses/id&enable_telemetry=0",
        "databricks.sql",
        authentication="OAuth U2M/M2M or token; OAuth/federation is recommended",
        tls_requirement="Databricks SQL connector HTTPS transport",
    ),
)


@dataclass(frozen=True)
class CloudSourceAdapter:
    id: str
    label: str
    uri_scheme: str
    install_extra: str
    driver_module: str
    authentication: str
    privacy: str = (
        "Host-only researcher-triggered download; credentials never enter "
        "the model or generated-code sandbox"
    )
    capabilities: tuple[str, ...] = (
        "streamed_download",
        "byte_and_disk_limits",
        "content_hash",
        "required_local_provenance",
        "dataset_parse_validation",
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


CLOUD_SOURCE_ADAPTERS: tuple[CloudSourceAdapter, ...] = (
    CloudSourceAdapter(
        "s3", "Amazon S3", "s3://bucket/key", "s3", "boto3",
        "AWS default credential chain, SSO, role, or workload identity",
    ),
    CloudSourceAdapter(
        "gcs", "Google Cloud Storage", "gs://bucket/key", "gcs",
        "google.cloud.storage",
        "Google Application Default Credentials or workload identity",
    ),
    CloudSourceAdapter(
        "azure_blob", "Azure Blob Storage", "az://account/container/blob",
        "azure", "azure.storage.blob",
        "Azure DefaultAzureCredential (managed identity, CLI, or approved login)",
    ),
    CloudSourceAdapter(
        "https", "Signed HTTPS download", "https://host/path", "built-in",
        "urllib.request",
        "Short-lived signed URL supplied by the researcher",
    ),
    CloudSourceAdapter(
        "sftp", "SFTP", "sftp://user@host/path", "sftp", "paramiko",
        "Vault-backed private-key profile and pinned known_hosts file",
    ),
)


def database_adapter_status() -> list[dict[str, Any]]:
    """Return install readiness without importing or connecting drivers."""
    import importlib.util

    out: list[dict[str, Any]] = []
    for adapter in DATABASE_ADAPTERS:
        try:
            # Check the exact import Sift needs, not only its namespace root.
            # ``snowflake-connector`` can provide ``snowflake`` while the
            # separately packaged ``snowflake.sqlalchemy`` dialect is absent;
            # a root-only check would let that broken release pass.
            installed = importlib.util.find_spec(adapter.driver_module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            installed = False
        out.append({**adapter.as_dict(), "installed": installed})
    return out


def cloud_source_adapter_status() -> list[dict[str, Any]]:
    """Return cloud-source readiness without importing SDKs or credentials."""
    import importlib.util

    out: list[dict[str, Any]] = []
    for adapter in CLOUD_SOURCE_ADAPTERS:
        if adapter.install_extra == "built-in":
            installed = True
        else:
            try:
                installed = importlib.util.find_spec(adapter.driver_module) is not None
            except (ImportError, ModuleNotFoundError, ValueError):
                installed = False
        out.append({**adapter.as_dict(), "installed": installed})
    return out


MODEL_CAPABILITIES: dict[str, dict[str, Any]] = {
    "anthropic": {
        "wire_api": "Anthropic API / Claude Agent SDK",
        "authentication": (
            "OS keyring API key",
            "environment API key",
        ),
        "tools": "Sift built-ins only; each sensitive tool is permission-gated",
        "vision": True,
        "conversation_state": "Sift session history is local; vendor processing terms still apply per request",
        "account_privacy_verifiable": False,
    },
    "openai": {
        "wire_api": "OpenAI Responses API",
        "authentication": ("OS keyring API key", "environment API key"),
        "tools": "Sift function tools only; hosted tools are disabled",
        "vision": True,
        "conversation_state": "Local by default; every request sets store=false",
        "account_privacy_verifiable": False,
    },
    "gemini": {
        "wire_api": "Gemini Generate Content API",
        "authentication": ("OS keyring API key", "environment API key"),
        "tools": "Sift function declarations only; grounding and hosted retrieval are disabled",
        "vision": True,
        "conversation_state": "Sift session history is local; no File API or explicit cache is used",
        "account_privacy_verifiable": False,
    },
    "openai_compatible": {
        "wire_api": "OpenAI-compatible Chat Completions API",
        "authentication": (
            "OS keyring API key",
            "environment API key",
            "no key for localhost",
        ),
        "tools": "Required; exact support depends on the configured model server",
        "vision": "endpoint_dependent",
        "conversation_state": "Sift session history is local; endpoint retention is operator-defined",
        "account_privacy_verifiable": False,
    },
    "azure_openai": {
        "wire_api": "Azure OpenAI Responses API",
        "authentication": ("Azure API key", "Microsoft Entra managed identity"),
        "tools": "Sift function tools only; hosted tools are disabled",
        "vision": True,
        "conversation_state": "Local replay with store=false",
        "account_privacy_verifiable": False,
    },
    "vertex_gemini": {
        "wire_api": "Gemini API on Vertex AI",
        "authentication": ("Google ADC", "workload identity federation"),
        "tools": "Sift function declarations only; grounding and hosted retrieval disabled",
        "vision": True,
        "conversation_state": "Local SDK chat state; no File API or explicit cache",
        "account_privacy_verifiable": False,
    },
    "bedrock_anthropic": {
        "wire_api": "Amazon Bedrock Converse API",
        "authentication": ("AWS credential chain", "IAM role", "temporary credentials"),
        "tools": "Sift tool specifications only; no Agents or Knowledge Bases",
        "vision": True,
        "conversation_state": "Complete local replay",
        "account_privacy_verifiable": False,
    },
    "vertex_anthropic": {
        "wire_api": "Anthropic Messages API through Vertex AI",
        "authentication": ("Google ADC", "workload identity federation"),
        "tools": "Sift function tools only",
        "vision": True,
        "conversation_state": "Complete local replay",
        "account_privacy_verifiable": False,
    },
}

for _provider_id in MODEL_PROVIDER_IDS:
    MODEL_CAPABILITIES[_provider_id].update(
        {
            "model_access_included": False,
            "credential_owner": "researcher",
            "provider_account_owner": "researcher_or_researcher_organization",
            "billing_relationship": "direct_with_provider_or_endpoint_operator",
            "lifecycle": (
                "unconfigured",
                "needs_configuration_or_credentials",
                "ready",
                "active_session",
                "credential_removed_or_configuration_changed",
                "session_closed",
            ),
        }
    )

if set(MODEL_CAPABILITIES) != set(MODEL_PROVIDER_IDS):
    raise RuntimeError("model capability registry is out of sync with provider ids")
if set(MODEL_TRUST) != set(MODEL_PROVIDER_IDS) - {"openai_compatible"}:
    raise RuntimeError("model trust registry is out of sync with provider ids")


def database_integration(backend: str) -> IntegrationTrust:
    local = backend in {"sqlite", "duckdb", "duckdb-file"}
    label = DATABASE_LABELS.get(backend, backend or "Database")
    return IntegrationTrust(
        id=backend,
        kind="database",
        label=label,
        location="local" if local else "remote",
        raw_dataset_access=True,
        sends_prompts=False,
        sends_sanitized_results=False,
        sends_user_attachments=False,
        credentials=(
            "No network credential"
            if local
            else "Supplied to the database driver on the host; never exposed to model tools"
        ),
        retention=(
            "Local files only"
            if local
            else "The database server may retain query/audit logs under its own policy"
        ),
        guarantees=(
            "Queries execute outside the generated-code sandbox",
            "Only SELECT/WITH/VALUES statements are accepted",
            "Results are fetched with row and memory ceilings and materialized locally",
            "Credentials are redacted before logs, provenance, or UI responses",
        ),
        cautions=(
            "Use a SELECT-only database principal; client-side SQL validation cannot classify institution-specific functions",
            "Remote database operators can observe queries and may log them",
        ),
    )


_COMMON_LIFECYCLE: tuple[IntegrationState, ...] = (
    "unconfigured",
    "needs_credentials",
    "needs_configuration",
    "blocked_by_policy",
    "ready",
    "active",
    "degraded",
    "unavailable",
)


def _api_key_auth(provider: str) -> tuple[AuthenticationMethod, ...]:
    if provider == "azure_openai":
        return (
            AuthenticationMethod(
                "entra_identity", "managed_identity", False,
                "Azure DefaultAzureCredential chain",
                "Microsoft Entra managed or workload identity scoped to the Azure resource",
            ),
            AuthenticationMethod(
                "azure_api_key", "api_key", True,
                "OS credential store or AZURE_OPENAI_API_KEY",
                "Azure-resource API key; never a direct OpenAI key",
            ),
        )
    if provider in {"vertex_gemini", "vertex_anthropic"}:
        return (AuthenticationMethod(
            "google_adc", "workload_identity", False,
            "Google Application Default Credentials chain",
            "User ADC, service account, metadata identity, or workload identity federation",
        ),)
    if provider == "bedrock_anthropic":
        return (AuthenticationMethod(
            "aws_chain", "workload_identity", False,
            "AWS SDK default credential chain",
            "AWS profile, SSO, IAM role, web identity, or temporary credentials",
        ),)
    methods = [AuthenticationMethod(
        "api_key", "api_key", True, "OS credential store or approved environment input",
        f"Researcher-controlled {provider} API credential",
    )]
    if provider == "openai_compatible":
        methods.append(AuthenticationMethod(
            "localhost_none", "none", False, "not applicable",
            "Authentication-free literal loopback endpoint",
        ))
    return tuple(methods)


def _database_auth(adapter: DatabaseAdapter) -> tuple[AuthenticationMethod, ...]:
    if adapter.id in {"sqlite", "duckdb"}:
        return (AuthenticationMethod(
            "local_permissions", "local_permissions", False, "operating-system file permissions",
            "Local database file access",
        ),)
    return (
        AuthenticationMethod(
            "driver_secret", "password", True, "OS credential store or driver-managed secure input",
            adapter.authentication,
        ),
        AuthenticationMethod(
            "managed_identity", "managed_identity", False, "platform identity chain",
            "Managed/workload identity when supported by the database driver",
        ),
    )


def _cloud_auth(adapter: CloudSourceAdapter) -> tuple[AuthenticationMethod, ...]:
    if adapter.id == "https":
        return (
            AuthenticationMethod(
                "signed_url", "oauth", True,
                "researcher-supplied transient URL; never persisted",
                adapter.authentication,
            ),
            AuthenticationMethod(
                "bearer_profile", "oauth", True, "OS credential store",
                "Vault-backed bearer token for an institutional HTTPS endpoint",
            ),
        )
    if adapter.id == "sftp":
        return (AuthenticationMethod(
            "ssh_key", "certificate", True,
            "private key file plus OS-vault passphrase profile",
            "Pinned known_hosts and key-only SSH authentication",
        ),)
    if adapter.id == "azure_blob":
        return (
            AuthenticationMethod(
                "identity_chain", "workload_identity", False,
                "Azure DefaultAzureCredential chain", adapter.authentication,
            ),
            AuthenticationMethod(
                "sas_profile", "api_key", True, "OS credential store",
                "Vault-backed Azure SAS token",
            ),
        )
    return (
        AuthenticationMethod(
            "identity_chain", "workload_identity", False, "cloud SDK default identity chain",
            adapter.authentication,
        ),
    )


def integration_contracts() -> tuple[IntegrationContract, ...]:
    """Return every integration through one security/lifecycle schema."""
    rows: list[IntegrationContract] = []
    for provider in MODEL_PROVIDER_IDS:
        trust = model_integration(provider)
        managed = provider in {
            "azure_openai", "vertex_gemini", "bedrock_anthropic", "vertex_anthropic",
        }
        region = {
            "azure_openai": os.environ.get("SIFT_AZURE_OPENAI_REGION"),
            "vertex_gemini": os.environ.get("SIFT_VERTEX_GEMINI_LOCATION"),
            "bedrock_anthropic": os.environ.get("SIFT_BEDROCK_REGION"),
            "vertex_anthropic": os.environ.get("SIFT_VERTEX_ANTHROPIC_LOCATION"),
        }.get(provider)
        rows.append(IntegrationContract(
            id=provider,
            kind="model",
            label=trust.label,
            maturity=("experimental" if provider == "openai_compatible" else "preview"),
            authentication=_api_key_auth(provider),
            data_flow=DataFlowContract(
                host_reads_raw_data=False,
                generated_code_reads_raw_data=True,
                generated_code_network_access=False,
                remote_receives_prompts=trust.sends_prompts,
                remote_receives_sanitized_results=trust.sends_sanitized_results,
                remote_receives_explicit_attachments=trust.sends_user_attachments,
                remote_receives_raw_dataset=False,
            ),
            retention=RetentionContract(
                sift_persists_remote_content=False,
                controlled_by="endpoint operator",
                account_setting_verifiable_by_sift=False,
                disclosure=trust.retention,
            ),
            residency=ResidencyContract(
                mode=(
                    "local" if trust.location == "local"
                    else "region_configurable" if managed
                    else "unverified"
                ),
                region=region,
                enforced_by=(
                    "Sift loopback validation" if trust.location == "local"
                    else "managed cloud endpoint and enterprise allowlist" if managed
                    else "provider account/contract"
                ),
                disclosure=(
                    "Local host only" if trust.location == "local"
                    else "The configured managed-cloud location is enforced before session open" if managed
                    else "Sift cannot infer processing region from an API key"
                ),
            ),
            capabilities=(
                "conversation",
                "sift_function_tools",
                "explicit_image_attachments",
                "usage_accounting",
            ),
            lifecycle=_COMMON_LIFECYCLE,
            operations={
                "connection_test": OperationPolicy(30, True),
                "conversation_turn": OperationPolicy(300, True),
            },
            policy_boundary="sift.provider.open_session",
            credential_boundary="host provider client; never generated code or model context",
        ))
    for adapter in DATABASE_ADAPTERS:
        local = adapter.id in {"sqlite", "duckdb"}
        rows.append(IntegrationContract(
            id=adapter.id,
            kind="database",
            label=adapter.label,
            maturity="supported" if local else "preview",
            authentication=_database_auth(adapter),
            data_flow=DataFlowContract(
                host_reads_raw_data=True,
                generated_code_reads_raw_data=True,
                generated_code_network_access=False,
                remote_receives_prompts=False,
                remote_receives_sanitized_results=False,
                remote_receives_explicit_attachments=False,
                remote_receives_raw_dataset=False,
            ),
            retention=RetentionContract(
                sift_persists_remote_content=False,
                controlled_by="database operator",
                account_setting_verifiable_by_sift=False,
                disclosure=("Local file only" if local else "The database may retain query and audit logs"),
            ),
            residency=ResidencyContract(
                mode="local" if local else "provider_defined",
                region=None,
                enforced_by="local filesystem" if local else "database deployment",
                disclosure="Local host only" if local else "Configured and enforced by the database operator",
            ),
            capabilities=adapter.capabilities,
            lifecycle=_COMMON_LIFECYCLE,
            operations={
                "connection_test": OperationPolicy(30, True),
                "catalog_discovery": OperationPolicy(30, True),
                "materialize": OperationPolicy(300, True),
            },
            policy_boundary="sift.connectors host factory",
            credential_boundary="host database driver; never generated code or model tools",
        ))
    for cloud_adapter in CLOUD_SOURCE_ADAPTERS:
        rows.append(IntegrationContract(
            id=cloud_adapter.id,
            kind="object_storage",
            label=cloud_adapter.label,
            maturity="preview",
            authentication=_cloud_auth(cloud_adapter),
            data_flow=DataFlowContract(
                host_reads_raw_data=True,
                generated_code_reads_raw_data=True,
                generated_code_network_access=False,
                remote_receives_prompts=False,
                remote_receives_sanitized_results=False,
                remote_receives_explicit_attachments=False,
                remote_receives_raw_dataset=False,
            ),
            retention=RetentionContract(
                sift_persists_remote_content=False,
                controlled_by="storage operator",
                account_setting_verifiable_by_sift=False,
                disclosure="The storage service may retain access and transfer audit logs",
            ),
            residency=ResidencyContract(
                mode="provider_defined", region=None, enforced_by="bucket/container configuration",
                disclosure="Configured and enforced by the storage operator",
            ),
            capabilities=cloud_adapter.capabilities,
            lifecycle=_COMMON_LIFECYCLE,
            operations={
                "connection_test": OperationPolicy(30, True),
                "materialize": OperationPolicy(300, True),
            },
            policy_boundary="sift.cloud_sources host factory",
            credential_boundary="host SDK identity chain; never generated code or model tools",
        ))
    from sift.research_services import SERVICE_CONTRACTS

    for service in SERVICE_CONTRACTS.values():
        local = service.id == "zotero"
        rows.append(IntegrationContract(
            id=service.id,
            kind="research_service",
            label=service.label,
            maturity="preview",
            authentication=(AuthenticationMethod(
                "local_selection" if local else "scoped_token",
                "local_permissions" if local else "oauth",
                not local,
                "local Zotero export/API" if local else "OS credential store",
                service.authentication,
            ),),
            data_flow=DataFlowContract(
                host_reads_raw_data=True,
                generated_code_reads_raw_data=True,
                generated_code_network_access=False,
                remote_receives_prompts=False,
                remote_receives_sanitized_results=False,
                remote_receives_explicit_attachments=False,
                remote_receives_raw_dataset=False,
            ),
            retention=RetentionContract(
                sift_persists_remote_content=False,
                controlled_by="local filesystem" if local else "research service operator",
                account_setting_verifiable_by_sift=False,
                disclosure=(
                    "Selected local records and attachments only"
                    if local else
                    "The service may retain API access logs under its account policy"
                ),
            ),
            residency=ResidencyContract(
                mode="local" if local else "provider_defined", region=None,
                enforced_by="local filesystem" if local else "selected service account/project",
                disclosure=("Local host only" if local else
                            "Configured and enforced by the research service operator"),
            ),
            capabilities=("explicit_selection", "local_materialization", *service.preserves),
            lifecycle=_COMMON_LIFECYCLE,
            operations={
                "selected_metadata": OperationPolicy(30, True),
                "materialize": OperationPolicy(300, True),
            },
            policy_boundary="sift.research_services explicit selection",
            credential_boundary=(
                "local OS permissions" if local else
                "host research-service client; never generated code or model tools"
            ),
        ))
    ids = [(row.kind, row.id) for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("integration contract identifiers must be unique per kind")
    return tuple(rows)


def integration_readiness_diagnostics() -> list[dict[str, Any]]:
    """Return secret-free, non-network readiness for every integration."""
    from sift.provider import all_provider_readiness

    provider_states = all_provider_readiness()
    database_states = {row["id"]: row for row in database_adapter_status()}
    cloud_states = {row["id"]: row for row in cloud_source_adapter_status()}
    out: list[dict[str, Any]] = []
    for contract in integration_contracts():
        if contract.kind == "model":
            state = dict(provider_states[contract.id])
            out.append({
                "integration_id": contract.id,
                "kind": contract.kind,
                "state": state["state"],
                "ready": state["ready"],
                "issues": list(state["issues"]),
                "diagnostics": {"auth_method": state["auth_mode"]},
            })
            continue
        if contract.kind == "research_service":
            out.append({
                "integration_id": contract.id,
                "kind": contract.kind,
                "state": "needs_configuration",
                "ready": False,
                "issues": ["explicit_object_selection_required"],
                "diagnostics": {
                    "driver_installed": True,
                    "metadata_only": True,
                    "account_listing_exposed": False,
                },
            })
            continue
        source = database_states[contract.id] if contract.kind == "database" else cloud_states[contract.id]
        installed = bool(source["installed"])
        out.append({
            "integration_id": contract.id,
            "kind": contract.kind,
            "state": "needs_configuration" if installed else "unavailable",
            "ready": False,
            "issues": ["connection_or_object_selection_required"] if installed else [
                "driver_not_installed"
            ],
            "diagnostics": {
                "driver_installed": installed,
                "metadata_only": True,
            },
        })
    return out


def list_integration_manifests() -> dict[str, Any]:
    from sift.provider.enterprise_common import PRIVACY_MANIFESTS

    models = [model_integration(p).as_dict() for p in MODEL_PROVIDER_IDS]
    databases = [database_integration(p).as_dict() for p in DATABASE_LABELS]
    from sift.research_services import SERVICE_CONTRACTS

    return {
        "models": models,
        "model_adapters": [
            {"id": provider, **MODEL_CAPABILITIES[provider]}
            for provider in MODEL_PROVIDER_IDS
        ],
        "managed_model_privacy": {
            provider: manifest.as_dict()
            for provider, manifest in PRIVACY_MANIFESTS.items()
        },
        "databases": databases,
        "database_adapters": database_adapter_status(),
        "cloud_source_adapters": cloud_source_adapter_status(),
        "research_service_adapters": [
            {
                "id": row.id,
                "label": row.label,
                "preserves": list(row.preserves),
                "authentication": row.authentication,
                "explicit_selection_only": True,
                "account_listing_exposed": False,
                "materializes_locally_before_analysis": True,
            }
            for row in SERVICE_CONTRACTS.values()
        ],
        "contracts": [row.as_dict() for row in integration_contracts()],
        "readiness": integration_readiness_diagnostics(),
    }
