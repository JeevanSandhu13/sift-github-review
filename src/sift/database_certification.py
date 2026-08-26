"""Exact live-certification contract for Sift's remote database adapters.

Unit tests can prove parsing, policy, redaction, and driver orchestration. They
cannot honestly certify vendor services. This registry maps every remote-
database requirement to a concrete live scenario and to the environment
inputs that scenario needs. The live suite and optional compatibility reporter consume
this registry, so adding a claim without adding a real test is a build failure.

Connection URIs remain in the process environment and are never serialized.
All fixtures must be disposable and use synthetic data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping, TypedDict

ScenarioMode = Literal[
    "core", "tls_hostname", "secure_policy", "auth", "types", "read_only",
    "cancellation", "driver_18", "result", "preview", "metadata",
    "warehouse_cancellation", "cloudfetch",
]
AuthenticationAssurance = Literal["mechanism", "session_principal"]


@dataclass(frozen=True)
class AuthenticationProofContract:
    """Backend-owned query and the exact assurance its result can provide.

    ``mechanism`` means the backend exposes evidence of the authentication
    mechanism for the current connection. ``session_principal`` means the
    backend exposes the principal which opened the session, but its SQL API
    does not portably expose whether OAuth, a token, ADC, or a key was used.
    Those variants therefore require separate disposable identities and prove
    successful use of the configured fixture, without overstating that the
    server attested the client credential type.
    """

    backend: str
    query: str
    mechanism_variants: tuple[str, ...]
    context_semantics: str

    def assurance_for(self, variant: str) -> AuthenticationAssurance:
        return "mechanism" if variant in self.mechanism_variants else "session_principal"


# Queries are deliberately owned by Sift rather than supplied by the fixture
# operator. Each returns exactly the two canonical proof columns. The choices
# use unprivileged current-session interfaces so certification never needs
# broad monitoring or account-history permissions.
AUTHENTICATION_PROOF_CONTRACTS: Mapping[str, AuthenticationProofContract] = MappingProxyType({
    "postgresql": AuthenticationProofContract(
        "postgresql",
        """SELECT CAST(session_user AS TEXT) AS authenticated_identity,
CASE
  WHEN COALESCE(ssl.client_dn, '') <> ''
    THEN 'tls-client-certificate:' || ssl.client_dn
  WHEN COALESCE(ssl.ssl, FALSE) THEN 'tls:no-client-certificate'
  ELSE 'transport:no-tls'
END AS authentication_context
FROM (SELECT 1) AS sift_auth_probe
LEFT JOIN pg_stat_ssl AS ssl ON ssl.pid = pg_backend_pid()""",
        ("certificate",),
        "Session user plus current-connection TLS client-certificate state; "
        "PostgreSQL does not distinguish a managed/IAM token passed through "
        "the password protocol from an ordinary password.",
    ),
    "mssql": AuthenticationProofContract(
        "mssql",
        """SELECT CAST(ORIGINAL_LOGIN() AS nvarchar(128)) AS authenticated_identity,
CAST(connection.auth_scheme AS nvarchar(128)) AS authentication_context
FROM sys.dm_exec_connections AS connection
WHERE connection.session_id = @@SPID""",
        ("entra", "windows"),
        "Original login and the current connection's server-reported "
        "authentication scheme.",
    ),
    "oracle": AuthenticationProofContract(
        "oracle",
        """SELECT SYS_CONTEXT('USERENV', 'AUTHENTICATED_IDENTITY') AS \"authenticated_identity\",
SYS_CONTEXT('USERENV', 'AUTHENTICATION_METHOD') AS \"authentication_context\"
FROM DUAL""",
        ("wallet_mtls",),
        "Oracle authenticated identity and authentication method from the "
        "current USERENV context.",
    ),
    "snowflake": AuthenticationProofContract(
        "snowflake",
        """SELECT SYS_CONTEXT('SNOWFLAKE$SESSION', 'PRINCIPAL_NAME') AS \"authenticated_identity\",
'principal_type:' || SYS_CONTEXT('SNOWFLAKE$SESSION', 'PRINCIPAL_TYPE') AS \"authentication_context\"""",
        (),
        "Principal name and type which started the Snowflake session; "
        "Snowflake SQL does not portably attest the client credential type.",
    ),
    "bigquery": AuthenticationProofContract(
        "bigquery",
        """SELECT SESSION_USER() AS authenticated_identity,
CONCAT('query_project:', @@project_id) AS authentication_context""",
        (),
        "Query principal and executing project; BigQuery SQL does not expose "
        "whether ADC resolved to user, service-account, or federated credentials.",
    ),
    "redshift": AuthenticationProofContract(
        "redshift",
        """SELECT CAST(current_user AS VARCHAR) AS authenticated_identity,
CASE
  WHEN current_session_arn() IS NOT NULL THEN current_session_arn()
  WHEN CAST(current_user AS VARCHAR) LIKE 'IAM:%'
    OR CAST(current_user AS VARCHAR) LIKE 'IAMA:%'
    THEN 'temporary-iam-user:' || CAST(current_user AS VARCHAR)
  ELSE 'local-user:' || CAST(current_user AS VARCHAR)
END AS authentication_context""",
        ("iam",),
        "Database user plus either the globally authenticated IAM session ARN "
        "or Redshift's IAM:/IAMA: temporary-credential user marker.",
    ),
    "databricks": AuthenticationProofContract(
        "databricks",
        """SELECT current_user() AS authenticated_identity,
CONCAT('session_principal:', current_user()) AS authentication_context""",
        (),
        "Databricks current/session user (a service-principal UUID for service "
        "principals); SQL does not portably expose OAuth versus token use.",
    ),
})


def authentication_proof_contract(backend: str) -> AuthenticationProofContract:
    """Return the immutable, code-owned live authentication proof contract."""
    try:
        return AUTHENTICATION_PROOF_CONTRACTS[backend]
    except KeyError as exc:  # pragma: no cover - registry integrity guard
        raise ValueError(f"no authentication proof contract for {backend!r}") from exc


def validate_authentication_context(
    backend: str,
    variant: str,
    context: str,
) -> None:
    """Reject a successful query whose provider evidence contradicts its fixture."""
    normalized = context.strip()
    if backend == "postgresql" and variant == "certificate":
        if not normalized.startswith("tls-client-certificate:"):
            raise ValueError("PostgreSQL certificate proof lacks a client certificate")
    elif backend == "mssql":
        if variant == "windows" and normalized.upper() not in {"KERBEROS", "NTLM"}:
            raise ValueError("SQL Server Windows proof is not KERBEROS or NTLM")
        if variant == "entra" and normalized.upper() not in {
            "FEDAUTH", "AZURE ACTIVE DIRECTORY", "MICROSOFT ENTRA ID",
        }:
            raise ValueError("SQL Server Entra proof is not federated authentication")
    elif backend == "oracle" and variant == "wallet_mtls":
        if normalized.upper() != "SSL":
            raise ValueError("Oracle wallet/mTLS proof is not SSL-authenticated")
    elif backend == "redshift" and variant == "iam":
        if not (
            normalized.startswith("arn:")
            or normalized.startswith("temporary-iam-user:IAM:")
            or normalized.startswith("temporary-iam-user:IAMA:")
        ):
            raise ValueError("Redshift proof lacks an IAM session identity")


class CertificationEnvironmentStatus(TypedDict):
    step_id: str
    backend: str
    mode: ScenarioMode
    claim: str
    authentication_variants: list[str]
    authentication_assurance: dict[str, AuthenticationAssurance]
    authentication_context_semantics: str
    required_fixture_fields: list[str]
    configured: bool
    missing_environment: list[str]


@dataclass(frozen=True)
class DatabaseCertificationScenario:
    step_id: str
    backend: str
    mode: ScenarioMode
    claim: str
    variants: tuple[str, ...] = ()
    required_fields: tuple[str, ...] = ()
    proof_query: str | None = None

    @property
    def env_prefix(self) -> str:
        return f"SIFT_LIVE_{self.step_id.replace('-', '_')}"

    def required_authentication_environment(self, variant: str) -> tuple[str, ...]:
        if self.mode != "auth" or variant not in self.variants:
            raise ValueError("unknown authentication variant")
        prefix = f"{self.env_prefix}_{variant.upper()}"
        suffixes = (
            "URI", "EXPECTED_ROW_JSON",
            *_AUTH_VARIANT_EXTRA_SUFFIXES.get((self.step_id, variant), ()),
        )
        return tuple(f"{prefix}_{suffix}" for suffix in suffixes)

    def required_environment(self) -> tuple[str, ...]:
        prefix = self.env_prefix
        if self.mode == "auth":
            return tuple(
                name
                for variant in self.variants
                for name in self.required_authentication_environment(variant)
            )
        suffixes = {
            "core": ("URI",),
            "tls_hostname": ("URI", "NEGATIVE_HOSTNAME_URI"),
            "secure_policy": ("URI", "INSECURE_URI"),
            "types": (
                "URI", "EXPECTED_SCHEMA_JSON", "EXPECTED_ROW_JSON",
                "FIXTURE_MANIFEST_JSON",
            ),
            "read_only": ("URI",),
            "cancellation": (
                "URI", "CANCEL_QUERY",
            ),
            "driver_18": ("URI",),
            "result": (
                ("URI", "EXPECTED_ROW_JSON")
                if self.proof_query else ("URI", "QUERY", "EXPECTED_ROW_JSON")
            ),
            "preview": ("URI", "QUERY"),
            "metadata": (
                "URI", "EXPECTED_METADATA_JSON", "DENIED_METADATA_JSON",
            ),
            "warehouse_cancellation": (
                "URI", "CANCEL_QUERY",
            ),
            "cloudfetch": (
                "URI", "QUERY", "EXPECTED_MIN_ROWS", "CANCEL_QUERY",
            ),
        }[self.mode]
        if self.step_id == "S04-081":
            suffixes = (*suffixes, "OCSP_FAILURE_URI")
        return tuple(f"{prefix}_{suffix}" for suffix in suffixes)


_AUTH_VARIANT_EXTRA_SUFFIXES: Mapping[tuple[str, str], tuple[str, ...]] = MappingProxyType({
    ("S04-082", "key_pair"): ("PRIVATE_KEY_PEM",),
    ("S04-100", "oauth_m2m"): ("CLIENT_ID", "CLIENT_SECRET"),
})


DATABASE_CERTIFICATION_SCENARIOS: tuple[DatabaseCertificationScenario, ...] = (
    DatabaseCertificationScenario("S04-058", "postgresql", "core", "Certify PostgreSQL."),
    DatabaseCertificationScenario("S04-059", "postgresql", "tls_hostname", "PostgreSQL verified TLS and hostname rejection."),
    DatabaseCertificationScenario("S04-060", "postgresql", "auth", "PostgreSQL password, certificate, and managed authentication.", ("password", "certificate", "managed")),
    DatabaseCertificationScenario("S04-061", "postgresql", "types", "PostgreSQL-specific type fidelity.", required_fields=("integer_value", "bigint_value", "numeric_value", "uuid_value", "jsonb_value", "array_value", "timestamptz_value", "interval_value", "bytea_value")),
    DatabaseCertificationScenario("S04-062", "postgresql", "read_only", "PostgreSQL server-enforced read-only transaction."),
    DatabaseCertificationScenario("S04-063", "mysql", "core", "Certify MySQL."),
    DatabaseCertificationScenario("S04-064", "mariadb", "core", "Certify MariaDB separately."),
    DatabaseCertificationScenario("S04-065", "mysql", "tls_hostname", "MySQL certificate and hostname verification."),
    DatabaseCertificationScenario("S04-066", "mysql", "types", "MySQL unsigned integer, enum, set, JSON, and zero-date fidelity.", required_fields=("unsigned_integer", "enum_value", "set_value", "json_value", "zero_date")),
    DatabaseCertificationScenario("S04-067", "mysql", "read_only", "MySQL server-enforced read-only transaction."),
    DatabaseCertificationScenario("S04-068", "mssql", "core", "Certify SQL Server."),
    DatabaseCertificationScenario("S04-069", "mssql", "driver_18", "Microsoft ODBC Driver 18 runtime behavior."),
    DatabaseCertificationScenario("S04-070", "mssql", "auth", "SQL Server Entra authentication.", ("entra",)),
    DatabaseCertificationScenario("S04-071", "mssql", "auth", "SQL Server Windows authentication.", ("windows",)),
    DatabaseCertificationScenario("S04-072", "mssql", "types", "SQL Server datetimeoffset, money, GUID, XML, and spatial fidelity.", required_fields=("datetimeoffset_value", "money_value", "guid_value", "xml_value", "spatial_value")),
    DatabaseCertificationScenario("S04-073", "mssql", "cancellation", "SQL Server query cancellation."),
    DatabaseCertificationScenario("S04-074", "oracle", "core", "Certify Oracle."),
    DatabaseCertificationScenario("S04-075", "oracle", "secure_policy", "Oracle TCPS enforcement."),
    DatabaseCertificationScenario("S04-076", "oracle", "tls_hostname", "Oracle TLS hostname verification."),
    DatabaseCertificationScenario("S04-077", "oracle", "auth", "Oracle wallet and mutual-TLS connection.", ("wallet_mtls",)),
    DatabaseCertificationScenario("S04-078", "oracle", "types", "Oracle decimal, timestamp, interval, CLOB, and BLOB fidelity.", required_fields=("decimal_value", "timestamp_value", "interval_value", "clob_value", "blob_value")),
    DatabaseCertificationScenario("S04-079", "oracle", "read_only", "Oracle server-enforced read-only transaction."),
    DatabaseCertificationScenario("S04-080", "snowflake", "core", "Certify Snowflake."),
    DatabaseCertificationScenario("S04-081", "snowflake", "secure_policy", "Snowflake fail-closed OCSP enforcement."),
    DatabaseCertificationScenario("S04-082", "snowflake", "auth", "Snowflake OAuth, SSO, key-pair, and password authentication.", ("oauth", "sso", "key_pair", "password")),
    DatabaseCertificationScenario(
        "S04-083", "snowflake", "result",
        "Snowflake warehouse, database, schema, and role selection.",
        required_fields=("warehouse", "database", "schema", "role"),
        proof_query=(
            'SELECT CURRENT_WAREHOUSE() AS "warehouse", '
            'CURRENT_DATABASE() AS "database", CURRENT_SCHEMA() AS "schema", '
            'CURRENT_ROLE() AS "role"'
        ),
    ),
    DatabaseCertificationScenario("S04-084", "snowflake", "types", "Snowflake variant, array, object, geography, and timestamp fidelity.", required_fields=("variant_value", "array_value", "object_value", "geography_value", "timestamp_value")),
    DatabaseCertificationScenario("S04-085", "snowflake", "warehouse_cancellation", "Snowflake cancellation followed by verified warehouse suspension."),
    DatabaseCertificationScenario("S04-086", "bigquery", "core", "Certify BigQuery."),
    DatabaseCertificationScenario("S04-087", "bigquery", "auth", "BigQuery Application Default Credentials.", ("adc",)),
    DatabaseCertificationScenario("S04-088", "bigquery", "auth", "BigQuery user credentials and workload identity.", ("user", "workload_identity")),
    DatabaseCertificationScenario(
        "S04-089", "bigquery", "result",
        "BigQuery project and billing-project behavior.",
        required_fields=("project", "billing_project"),
        proof_query=(
            "SELECT @@project_id AS billing_project"
        ),
    ),
    DatabaseCertificationScenario("S04-090", "bigquery", "preview", "BigQuery dry-run byte estimates."),
    DatabaseCertificationScenario("S04-091", "bigquery", "types", "BigQuery record, repeated, geography, numeric, and bignumeric fidelity.", required_fields=("record_value", "repeated_value", "geography_value", "numeric_value", "bignumeric_value")),
    DatabaseCertificationScenario("S04-092", "bigquery", "cancellation", "BigQuery job cancellation."),
    DatabaseCertificationScenario("S04-093", "redshift", "core", "Certify Redshift."),
    DatabaseCertificationScenario("S04-094", "redshift", "tls_hostname", "Redshift verified TLS and hostname rejection."),
    DatabaseCertificationScenario("S04-095", "redshift", "auth", "Redshift IAM authentication.", ("iam",)),
    DatabaseCertificationScenario("S04-096", "redshift", "types", "Redshift decimal and SUPER/semi-structured fidelity.", required_fields=("decimal_value", "super_value")),
    DatabaseCertificationScenario("S04-097", "redshift", "cancellation", "Redshift query cancellation."),
    DatabaseCertificationScenario("S04-098", "databricks", "core", "Certify Databricks SQL."),
    DatabaseCertificationScenario("S04-099", "databricks", "auth", "Databricks OAuth user-to-machine authentication.", ("oauth_u2m",)),
    DatabaseCertificationScenario("S04-100", "databricks", "auth", "Databricks OAuth machine-to-machine authentication.", ("oauth_m2m",)),
    DatabaseCertificationScenario("S04-101", "databricks", "auth", "Databricks token authentication.", ("token",)),
    DatabaseCertificationScenario("S04-102", "databricks", "metadata", "Databricks catalogs, schemas, and Unity Catalog permissions."),
    DatabaseCertificationScenario("S04-103", "databricks", "types", "Databricks array, map, struct, decimal, and timestamp fidelity.", required_fields=("array_value", "map_value", "struct_value", "decimal_value", "timestamp_value")),
    DatabaseCertificationScenario("S04-104", "databricks", "cloudfetch", "Databricks CloudFetch transfer and cancellation."),
)


def certification_environment_status(
    environment: dict[str, str],
) -> tuple[CertificationEnvironmentStatus, ...]:
    """Return content-free configuration state for release reporting."""
    rows: list[CertificationEnvironmentStatus] = []
    for scenario in DATABASE_CERTIFICATION_SCENARIOS:
        required = scenario.required_environment()
        missing = [name for name in required if not environment.get(name)]
        auth_contract = (
            authentication_proof_contract(scenario.backend)
            if scenario.mode == "auth"
            else None
        )
        rows.append({
            "step_id": scenario.step_id,
            "backend": scenario.backend,
            "mode": scenario.mode,
            "claim": scenario.claim,
            "authentication_variants": list(scenario.variants),
            "authentication_assurance": (
                {
                    variant: auth_contract.assurance_for(variant)
                    for variant in scenario.variants
                }
                if auth_contract is not None else {}
            ),
            "authentication_context_semantics": (
                auth_contract.context_semantics if auth_contract is not None else ""
            ),
            "required_fixture_fields": list(scenario.required_fields),
            "configured": not missing,
            "missing_environment": missing,
        })
    return tuple(rows)


def certification_preflight(environment: dict[str, str]) -> dict[str, object]:
    """Return a secret-free provisioning checklist for the live gate.

    Operators need to know which disposable fixture inputs remain before
    running certification, but a release preflight must never print URI,
    token, certificate, or expected-result values. This reports names and
    boolean state only and is safe to use in CI logs or support tickets.
    """
    scenarios = certification_environment_status(environment)
    missing_count = sum(len(row["missing_environment"]) for row in scenarios)
    return {
        "schema_version": 1,
        "scope": "Optional disposable live-database compatibility preflight",
        "program": "optional_external_compatibility",
        "product_release_blocking": False,
        "provisioning_responsibility": "researcher_or_database_operator",
        "disposable_scope_confirmed": (
            environment.get("SIFT_LIVE_DATABASES_DISPOSABLE") == "1"
        ),
        "write_probe_acknowledged": (
            environment.get("SIFT_LIVE_DATABASE_WRITE_PROBE_ACK") == "1"
        ),
        "configured_scenarios": sum(1 for row in scenarios if row["configured"]),
        "total_scenarios": len(scenarios),
        "missing_environment_variable_count": missing_count,
        "remote_scenarios": scenarios,
        "security_note": (
            "Only environment-variable names and configuration state are "
            "reported; connection values are never serialized. Authentication "
            "proof SQL is fixed in Sift and cannot be supplied by an operator."
        ),
    }


__all__ = [
    "AUTHENTICATION_PROOF_CONTRACTS", "AuthenticationAssurance",
    "AuthenticationProofContract", "CertificationEnvironmentStatus",
    "DATABASE_CERTIFICATION_SCENARIOS", "DatabaseCertificationScenario",
    "ScenarioMode", "certification_environment_status", "certification_preflight",
    "authentication_proof_contract", "validate_authentication_context",
]
