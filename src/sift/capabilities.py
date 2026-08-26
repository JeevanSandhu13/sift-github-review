"""Machine-readable product contract for Sift's backend.

The contract is intentionally assembled from the registries that enforce the
behaviour.  Frontends, qualification checks, and future documentation can ask
one place what Sift supports without maintaining another handwritten list.

This module also makes the commercial and privacy boundary unambiguous: Sift
does not include, proxy, resell, or bill for a model.  A researcher connects a
provider account or endpoint they control and remains responsible for that
provider's terms and charges.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from sift.connectors import (
    DEFAULT_BYTE_LIMIT,
    DEFAULT_QUERY_TIMEOUT_SECONDS,
    DEFAULT_ROW_LIMIT,
    FETCH_BATCH_ROWS,
    MAX_CATALOG_COLUMNS,
    MAX_CATALOG_OBJECTS,
    MAX_CATALOG_SCHEMAS,
)
from sift.cloud_sources import cloud_import_max_bytes
from sift.data_request import SUPPORTED_REQUEST_TYPES
from sift.executor import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_RESULT_FILE_BYTES,
    MAX_RESULT_PAYLOADS,
    script_cpu_limit_seconds,
    script_file_size_limit_bytes,
    script_memory_limit_bytes,
    script_min_free_disk_bytes,
    script_process_limit,
)
from sift.integrations import (
    CLOUD_SOURCE_ADAPTERS,
    DATABASE_ADAPTERS,
    list_integration_manifests,
)
from sift.limits import (
    DRAG_DROP_AGGREGATE_MAX_BYTES,
    DRAG_DROP_FILE_MAX_BYTES,
    INLINE_SCRIPT_MAX_BYTES,
    INLINE_SCRIPT_TOTAL_MAX_BYTES,
    MODEL_IMAGE_MAX_BYTES,
    ZIP_CONTAINER_MAX_MEMBERS,
)
from sift.provider import SUPPORTED_PROVIDERS
from sift.sanitizer import DEFAULT_CONFIG, supported_types
from sift.schema import DATA_EXTENSIONS, full_load_max_bytes
from sift.tools import friendly_tool_names

ContractStatus = Literal["supported", "preview", "experimental", "internal"]
_VALID_STATUSES: frozenset[str] = frozenset(
    {"supported", "preview", "experimental", "internal"}
)


@dataclass(frozen=True)
class Capability:
    """One supportable product statement tied to code and verification."""

    id: str
    category: str
    label: str
    status: ContractStatus
    implementation: str
    verification: tuple[str, ...]
    claim: str
    limitations: tuple[str, ...]
    surfaces: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    compatibility_since: int = 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductClaim:
    """A scoped promise with its enforcement and honest qualification."""

    id: str
    statement: str
    scope: str
    enforcement: tuple[str, ...]
    evidence: tuple[str, ...]
    caveats: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProductRisk:
    """One tracked residual risk in the shipped product contract."""

    id: str
    domain: Literal["privacy", "statistical", "integration", "reliability"]
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    controls: tuple[str, ...]
    evidence: tuple[str, ...]
    residual_risk: str
    disposition: Literal["accepted", "mitigated", "blocked"]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRODUCT_CONTRACT_SCHEMA_VERSION = 3
FRONTEND_CONTRACT_SCHEMA_VERSION = 1
CAPABILITY_COMPATIBILITY_VERSION = 2

MODEL_SUPPLY: dict[str, Any] = {
    "models_included": False,
    "model_proxy_operated_by_sift": False,
    "credentials": "researcher_supplied",
    "provider_account": "researcher_owned_or_researcher_authorized",
    "billing_relationship": "researcher_to_provider_or_endpoint_operator",
    "supported_connection_modes": (
        "provider_api_key",
        "researcher_configured_openai_compatible_endpoint",
    ),
}


DATABASE_SUPPLY: dict[str, Any] = {
    "databases_included": False,
    "database_proxy_operated_by_sift": False,
    "credentials": "researcher_supplied",
    "database_account": "researcher_owned_or_researcher_authorized",
    "billing_relationship": "researcher_to_database_operator",
    "connection_execution": "direct_from_researcher_workstation",
    "credential_storage": "os_credential_vault_or_researcher_environment",
    "live_vendor_certification": "optional_external_compatibility_program",
    "release_blocking": False,
}


# Generated analysis code and the runtime helpers share an interpreter.  The
# per-run token therefore detects framing mistakes and trivial direct writes,
# but cannot attest semantics against code that deliberately reads the token
# or fabricates aggregate-shaped values.  Keep this machine-readable so a
# frontend or release report cannot accidentally overstate the guarantee.
GENERATED_CODE_TRUST: dict[str, Any] = {
    "semantics_cryptographically_attested": False,
    "runtime_token_detects": (
        "missing_runtime_framing",
        "stale_runtime_library",
        "trivial_direct_result_writes",
    ),
    "runtime_token_does_not_detect": (
        "intentional_token_introspection",
        "fabricated_aggregate_semantics",
        "mislabelled_source_variables",
    ),
    "required_operating_assumption": (
        "The selected model/provider and generated analysis code are not "
        "adversarial; researchers review generated code and results."
    ),
}


PRIVACY_CONTRACT: dict[str, Any] = {
    "raw_data_guarantee": {
        "sift_direct_upload": False,
        "provider_raw_dataset_access": False,
        "generated_code_network_access": False,
        "scope": (
            "Datasets opened through Sift's file, database, and remote-object "
            "flows; free-form researcher prompts remain researcher-authored content."
        ),
        "assumption": GENERATED_CODE_TRUST["required_operating_assumption"],
    },
    "model_visible_sanitized_information": (
        "policy-bounded schema names, logical types, labels, and coarse summaries",
        "allowlisted statistical result fields after suppression and precision controls",
        "bounded aggregate request answers or structured denials",
        "sanitized filenames, provenance identifiers, verification findings, and tool errors",
        "researcher-authored prompts, approved analysis-plan metadata, and generated script text",
    ),
    "explicit_attachments": {
        "automatic_dataset_attachment": False,
        "researcher_confirmation_required": True,
        "supported_model_media": ("image/png", "image/jpeg", "image/webp", "image/gif"),
        "scripts_and_documents": (
            "Locally extracted or bounded text may be included only after an explicit "
            "researcher attachment/mention action and text-safety processing."
        ),
        "unsupported_media_action": "reject_before_provider_request",
    },
    "operator_observability": {
        "remote_model_provider": (
            "prompts, sanitized tool results, and explicit supported attachments; "
            "provider account retention and abuse-monitoring controls apply"
        ),
        "database_operator": (
            "authentication identity, connection metadata, query text, timing, scanned "
            "bytes, and returned-row volume according to the database audit configuration"
        ),
        "storage_operator": (
            "identity, bucket/container/object identifiers, versions, request timing, "
            "and transferred bytes according to the storage audit configuration"
        ),
        "sift_project": "No Sift-operated model, database, storage, or telemetry service exists.",
    },
    "control_responsibility": {
        "sift_enforced": (
            "tool allowlists and permission gates",
            "generated-code filesystem/network confinement when qualification passes",
            "credential/environment filtering",
            "result schemas, disclosure controls, limits, local audit records, and export policy",
        ),
        "provider_or_operator_controlled": (
            "retention, training/abuse monitoring, data residency, contractual ZDR terms",
            "account access logs, regional processing, encryption key ownership, and billing",
        ),
        "researcher_controlled": (
            "provider/account choice, prompt content, explicit attachments, credentials, "
            "dataset classification, research design, and final interpretation"
        ),
    },
}


QUALITY_STANDARDS: dict[str, Any] = {
    "statistical_correctness": {
        "minimum": (
            "Every supported result must satisfy its typed schema, finite/range checks, "
            "cross-field invariants, disclosure thresholds, and deterministic verifier."
        ),
        "method_requirement": (
            "A method is not certified solely because its result shape is accepted; "
            "assumptions, diagnostics, and a reference or differential test are required."
        ),
        "human_review_required": True,
    },
    "reproducibility": {
        "minimum": (
            "Retain exact script source, source-dataset identity/hash where available, "
            "runtime/package environment, sanitized outputs, transformations, and seeds "
            "when a stochastic operation declares them."
        ),
        "model_recontact_required_for_replay": False,
        "bundle_verification": (
            "Complete file manifest, source/script hashes, exact parser/runtime/model/privacy "
            "metadata, environment-drift reporting, and local numerical replay."
        ),
    },
    "provenance": {
        "minimum": (
            "Every result identifies its session, script run, source dataset(s), creation "
            "time, sanitizer transformations, and append-only release-ledger event."
        ),
        "integrity": "Hash-chained and locally verifiable; tamper-evident, not tamper-proof.",
    },
}


SUPPORTED_SCALE: dict[str, Any] = {
    "dataset_rows": {
        "universal_local_row_cap": None,
        "constraint": "file bytes, parser behavior, projected columns, and available local resources",
        "minimum_analysis_n": min(
            DEFAULT_CONFIG.min_n_regression,
            DEFAULT_CONFIG.min_n_descriptive,
            DEFAULT_CONFIG.min_n_ttest_group,
        ),
    },
    "dataset_columns": {
        "universal_local_column_cap": None,
        "constraint": "method-specific result caps and available local resources",
    },
    "database_extract": {
        "maximum_rows": DEFAULT_ROW_LIMIT,
        "maximum_bytes": DEFAULT_BYTE_LIMIT,
    },
    "full_memory_load_bytes": full_load_max_bytes(),
    "files": {
        "drag_drop_file_bytes": DRAG_DROP_FILE_MAX_BYTES,
        "drag_drop_aggregate_bytes": DRAG_DROP_AGGREGATE_MAX_BYTES,
        "native_picker_file_bytes": None,
        "remote_object_bytes": cloud_import_max_bytes(),
        "model_image_bytes": MODEL_IMAGE_MAX_BYTES,
        "inline_script_bytes": INLINE_SCRIPT_MAX_BYTES,
        "inline_script_total_bytes": INLINE_SCRIPT_TOTAL_MAX_BYTES,
    },
    "archives": {
        "general_zip_ingestion": False,
        "spreadsheet_container_members": ZIP_CONTAINER_MAX_MEMBERS,
        "spreadsheet_uncompressed_bytes": full_load_max_bytes(),
        "nested_archives": False,
    },
    "outputs": {
        "result_file_bytes": MAX_RESULT_FILE_BYTES,
        "result_payloads_per_run": MAX_RESULT_PAYLOADS,
    },
    "statement": (
        "Sift does not claim a universal row/column maximum for local files. "
        "Operations that cannot stay inside the declared byte, memory, time, or output "
        "limits fail explicitly rather than silently sampling or truncating, except where "
        "a response explicitly reports bounded sampling/truncation."
    ),
}


RISK_REGISTER: tuple[ProductRisk, ...] = (
    ProductRisk(
        "generated-code-semantic-forgery", "privacy", "high",
        "Provider-authored code can fabricate aggregate-shaped values in the helper interpreter.",
        ("OS confinement", "typed sanitization", "runtime framing token", "researcher code review"),
        ("sift.executor", "sift.sanitizer", "tests/test_executor_token.py"),
        "The framing token is not semantic attestation; adversarial generated code is out of scope.",
        "blocked",
    ),
    ProductRisk(
        "cumulative-disclosure", "privacy", "high",
        "Individually safe releases may disclose more when differenced or combined across sessions.",
        ("release ledger", "query fingerprints", "privacy budgets", "per-result suppression"),
        ("sift.release_ledger", "sift.query_fingerprint", "sift.privacy_budget"),
        "Global formal composition is not yet enforced for every result family.",
        "mitigated",
    ),
    ProductRisk(
        "method-misspecification", "statistical", "high",
        "A structurally valid result can still answer the wrong estimand or violate design assumptions.",
        ("analysis plans", "deterministic verification", "challenge pass", "claim caveats"),
        ("sift.verification", "sift.system_prompt", "tests/test_verification_all_shapes.py"),
        "Qualified researcher review remains required.",
        "accepted",
    ),
    ProductRisk(
        "external-integration-drift", "integration", "high",
        "Provider APIs, models, drivers, authentication, and service behavior change independently.",
        ("typed adapters", "readiness diagnostics", "pinned dependency ranges", "error translation"),
        ("sift.integrations", "sift.provider", "tests/test_integrations.py"),
        "Remote integrations without live certification are classified preview.",
        "mitigated",
    ),
    ProductRisk(
        "platform-confinement-drift", "reliability", "high",
        "Kernel or runtime changes may invalidate a confinement backend or its resource controls.",
        ("fail-closed live probe", "platform-specific tests", "runtime qualification report"),
        ("sift.env_detect", "sift.executor", "sift.qualification"),
        "Each release still requires unrestricted native-kernel qualification.",
        "mitigated",
    ),
    ProductRisk(
        "local-state-corruption", "reliability", "medium",
        "Interrupted writes or disk faults can damage session metadata or evidence.",
        ("atomic writes", "file locks", "SQLite integrity checks", "hash-chain verification"),
        ("sift.file_lock", "sift.store", "sift.qualification"),
        "Underlying storage hardware and administrator-controlled backups remain outside Sift's recovery boundary.",
        "mitigated",
    ),
)


PRODUCT_CLAIMS: tuple[ProductClaim, ...] = (
    ProductClaim(
        id="model-supply",
        statement=(
            "Sift connects to models selected and authorized by the "
            "researcher; Sift does not provide or resell model access."
        ),
        scope="All model providers and endpoints",
        enforcement=(
            (
                "Provider sessions require researcher authentication or a "
                "researcher-configured endpoint"
            ),
            "No Sift-operated model proxy exists",
        ),
        evidence=("sift.provider", "sift.auth", "tests/test_auth.py"),
        caveats=(
            (
                "Provider availability, price, retention, and account controls "
                "are controlled by the provider or endpoint operator"
            ),
        ),
        acceptance_criteria=(
            "Every provider factory path requires researcher-owned authentication or an explicitly configured endpoint.",
            "The shipped package contains no Sift-operated model proxy or bundled model entitlement.",
            "tests/test_auth.py and provider-factory contract tests pass.",
        ),
    ),
    ProductClaim(
        id="raw-dataset-boundary",
        statement=(
            "Sift does not upload raw dataset files or rows directly; under "
            "the documented non-adversarial generated-code assumption, model-"
            "visible analysis outputs pass through disclosure controls."
        ),
        scope="Data files and database extracts analyzed through Sift tools",
        enforcement=(
            (
                "Generated analysis runs inside an OS confinement backend with "
                "network access denied"
            ),
            (
                "Model-visible statistical payloads pass through allowlist-based "
                "disclosure controls"
            ),
            (
                "Database credentials and queries remain in researcher-approved "
                "host-side connector flows"
            ),
        ),
        evidence=(
            "sift.executor",
            "sift.sanitizer",
            "sift.connectors",
            "tests/test_executor_sandbox.py",
            "tests/test_sanitizer.py",
        ),
        caveats=(
            (
                "Prompts, sanitized results, and explicitly attached supported "
                "media are sent to a remote provider"
            ),
            (
                "A researcher can intentionally paste data into a prompt; Sift "
                "cannot classify every piece of free-form authored text"
            ),
            (
                "The result sanitizer validates structure and disclosure rules "
                "but cannot cryptographically attest the semantics of generated "
                "code running in the same interpreter as the runtime helpers"
            ),
        ),
        acceptance_criteria=(
            "No model tool can open a raw dataset or invoke a host-side connector.",
            "Generated-code qualification proves network denial and confined file access before execution.",
            "Every model-visible result is produced by an allowlisted sanitizer path or an explicit researcher attachment path.",
            "Adversarial generated-code certification remains false until execution becomes host-owned and declarative.",
        ),
    ),
    ProductClaim(
        id="fail-closed-execution",
        statement=(
            "Generated analysis is refused when the platform confinement "
            "backend is absent or fails its live security probe."
        ),
        scope="macOS, Windows, and Linux generated-code execution",
        enforcement=(
            (
                "macOS sandbox-exec, Windows AppContainer and Job Objects, or "
                "Linux bubblewrap is selected by platform"
            ),
            "Filesystem and network denial are positively probed before use",
        ),
        evidence=(
            "sift.executor",
            "sift.env_detect",
            "sift.win_appcontainer",
            "tests/test_bwrap_sandbox.py",
            "tests/test_win_appcontainer.py",
        ),
        caveats=(
            "The required OS facility must be present and functional",
            (
                "Resource-limit support differs by operating system and is "
                "reported separately from filesystem/network confinement"
            ),
        ),
        acceptance_criteria=(
            "Execution refuses to start when the platform backend or its live read/network probes fail.",
            "Native macOS, Windows, and Linux release lanes exercise the actual kernel boundary.",
            "Process-tree termination and declared resource controls pass platform-specific tests.",
        ),
    ),
    ProductClaim(
        id="credential-isolation",
        statement=(
            "Provider and database credentials are not exposed to generated "
            "analysis code or model tools."
        ),
        scope="Credentials handled by supported integrations",
        enforcement=(
            ("Provider credentials use OS keyring or explicit environment input"),
            "The executor constructs a minimal environment allowlist",
            "Connection details are redacted from model-visible errors and provenance",
        ),
        evidence=(
            "sift.auth",
            "sift.executor",
            "sift.connectors",
            "tests/test_executor_env_allowlist.py",
            "tests/test_connectors.py",
        ),
        caveats=(
            (
                "Environment variables remain visible to other processes with "
                "sufficient host privileges"
            ),
            "Researchers should prefer least-privilege, read-only database identities",
        ),
        acceptance_criteria=(
            "Executor environment tests demonstrate that provider and connector secrets are absent.",
            "Credential redaction fuzz tests cover supported URI, ODBC, query-string, and driver-error forms.",
            "No serialized log, provenance event, result, or export contains seeded canary credentials.",
        ),
    ),
    ProductClaim(
        id="scientific-assistance",
        statement=(
            "Sift records, validates, and challenges supported analysis "
            "results; it does not guarantee that a chosen method is valid for "
            "a particular research question."
        ),
        scope="Supported statistical result contracts and verification rules",
        enforcement=(
            "Typed result contracts validate structure and statistical ranges",
            "Verification and challenge checks attach machine-readable findings",
            (
                "Scripts, source lineage, results, and disclosure transformations "
                "remain locally auditable"
            ),
        ),
        evidence=(
            "sift.sanitizer",
            "sift.verification",
            "sift.release_ledger",
            "tests/test_verification_all_shapes.py",
        ),
        caveats=(
            "Model output and generated code require researcher review",
            (
                "Passing structural checks is not proof of causal identification, "
                "replicability, or domain validity"
            ),
        ),
        acceptance_criteria=(
            "Every supported result type has structural, disclosure, invariant, and verifier tests.",
            "Method-specific certification requires diagnostics plus differential or reference-result evidence.",
            "Narrative surfaces label limitations and never treat a passing schema as proof of causal validity.",
        ),
    ),
)


_CORE_CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "execution.generated-code-confinement", "execution", "Generated-code confinement",
        "supported", "sift.executor",
        ("tests/test_executor_sandbox.py", "tests/test_executor_profile.py"),
        "Runs researcher-approved analysis code only after a fail-closed platform probe.",
        ("Availability and resource-limit strength are reported per host.",),
        ("application_service",),
        ("A failed filesystem or network probe prevents execution.",),
    ),
    Capability(
        "research.approval-bound-workflow", "research_workflow", "Approval-bound research workflow",
        "supported", "sift.research_workflow",
        ("tests/test_research_workflow.py", "tests/test_analysis_templates.py"),
        (
            "Persists research intent, estimand, assumptions, unresolved quality issues, "
            "primary/sensitivity analyses, seeds, and approval-bound consequential choices."
        ),
        (
            "Researcher approval attests the declared plan, not that its scientific assumptions are true; "
            "researcher-only notes are excluded from model memory.",
        ),
        ("model_tool", "application_service", "ui_bridge", "export"),
        (
            "Method-backed execution fails closed without exact-revision approval; material revisions "
            "invalidate approval; session resume restores metadata without raw observations.",
        ),
    ),
    Capability(
        "research.dataset-profile", "research_workflow", "Dataset profile",
        "supported", "sift.dataset_profile", ("tests/test_dataset_profile.py",),
        "Produces a local, bounded dataset overview and semantic-type hints.",
        ("Large-file paths may use an explicitly reported sample.",),
        ("ui_bridge",),
        ("Shape, missingness, semantic types, and cache invalidation tests pass.",),
    ),
    Capability(
        "research.canonical-dataset", "research_workflow", "Canonical dataset contract",
        "supported", "sift.canonical_dataset", ("tests/test_canonical_dataset.py",),
        (
            "Content-addresses exact source selections, preserves structural metadata, "
            "and supplies one verified session-local cache to trusted analyses."
        ),
        (
            "Very large formats without a safe partial reader retain a source-byte "
            "content identity and explicitly mark unavailable inferred metadata.",
        ),
        ("application_service", "ui_bridge"),
        (
            "Cross-format semantic fixtures, immutable snapshots, cache invalidation, "
            "selection, lineage, collections, and cross-session isolation all pass.",
        ),
    ),
    Capability(
        "data.lazy-arrow-scan", "data", "Lazy Arrow dataset scan",
        "supported", "sift.schema",
        ("tests/test_performance.py", "tests/test_parquet_projection.py", "tests/test_arrow_formats.py"),
        "Scans Parquet in bounded Arrow batches with column projection and validated predicate pushdown.",
        ("Predicate pushdown is currently exposed for Parquet; other formats use their reviewed bounded readers.",),
        ("application_service", "analysis_runtime"),
        ("Projection, predicates, batch bounds, invalid operators, and Arrow format paths are tested.",),
    ),
    Capability(
        "research.dataset-health", "research_workflow", "Dataset health",
        "supported", "sift.data_quality",
        ("tests/test_data_quality.py", "tests/test_dataset_health.py"),
        (
            "Runs aggregate-only, context-aware quality checks before analysis and "
            "creates only explicitly approved, provenance-linked corrected copies."
        ),
        (
            "Heuristic findings remain advisory; formats without a bounded large-file "
            "reader explicitly defer observation-level checks above the memory ceiling.",
        ),
        ("application_service", "ui_bridge", "model_tool_result"),
        (
            "Detector families, value-free preflight, blocking critical findings, "
            "source immutability, explicit approval, and correction lineage all pass.",
        ),
    ),
    Capability(
        "research.methodology-registry", "research_workflow", "Research methodology engine",
        "supported", "sift.methodology",
        ("tests/test_methodology.py", "tests/test_method_result.py"),
        (
            "Validates research specifications and binds supported methods to "
            "assumptions, diagnostics, reference implementations, typed outputs, and claim rules."
        ),
        (
            "Fitting occurs in researcher-controlled local R/Python/Stata libraries; "
            "conditional methods remain unavailable until their support condition is affirmed.",
        ),
        ("application_service", "model_tool", "model_tool_result", "ui_bridge"),
        (
            "Every registered method has a complete contract; incomplete specifications "
            "and missing diagnostics are rejected before fitting or release.",
        ),
    ),
    Capability(
        "research.deterministic-verification", "research_workflow", "Deterministic verification",
        "supported", "sift.verification",
        ("tests/test_verification_all_shapes.py", "tests/test_verification_diagnostics.py"),
        "Checks sanitized results for numerical, diagnostic, and claim-language problems.",
        ("Operates on sanitized outputs and cannot re-fit every model independently.",),
        ("model_tool_result", "ui_bridge", "export"),
        ("Every supported result shape has a non-raising verifier contract test.",),
    ),
    Capability(
        "research.challenge-pass", "research_workflow", "Challenge finding",
        "supported", "sift.verification", ("tests/test_challenge_finding.py",),
        "Independently verifies stored aggregates, compares approved alternatives, and reports contradictions.",
        (
            "The pass uses sanitized aggregates and declared diagnostics rather than independently refitting raw data; "
            "comparison quality still depends on the alternatives actually run.",
        ),
        ("model_tool_result", "ui_bridge"),
        ("Contradictory and majority-agreement fixtures produce deterministic findings.",),
    ),
    Capability(
        "privacy.dataset-policy", "privacy", "Dataset privacy policy",
        "supported", "sift.policy",
        ("tests/test_policy.py", "tests/test_privacy_profiles.py"),
        "Applies per-dataset disclosure ceilings, variable restrictions, and export controls.",
        ("Generated-code semantic relabelling is outside the current enforcement boundary.",),
        ("ui_bridge", "application_service"),
        ("Invalid security-sensitive policy values fail closed and stricter rules never loosen.",),
    ),
    Capability(
        "privacy.enterprise-policy", "privacy", "Enterprise policy",
        "preview", "sift.enterprise_policy", ("tests/test_enterprise_policy.py",),
        "Applies organization allowlists and stricter privacy/export requirements.",
        ("No centralized policy distribution or signed-policy authority is included.",),
        ("deployment_configuration", "application_service"),
        ("Provider, integration, dataset, and export boundaries enforce the same loaded policy.",),
    ),
    Capability(
        "privacy.pre-provider-review", "privacy", "Pre-provider disclosure review",
        "supported", "sift.security_assurance", ("tests/test_security_assurance.py",),
        "Warns locally about likely credentials, identifiers, PHI context, and organization-sensitive fields before a provider request.",
        (
            "Pattern review is advisory and cannot prove de-identification; researchers can disable this optional warning layer.",
        ),
        ("ui_bridge", "application_service"),
        ("Warnings contain categories and counts but never matched content, and disabling them does not weaken mandatory policy controls.",),
    ),
    Capability(
        "privacy.retention", "privacy", "Retention and secure cleanup",
        "supported", "sift.security_assurance", ("tests/test_security_assurance.py", "tests/test_store.py"),
        "Previews and explicitly applies age-based execution-artifact retention with SQLite secure deletion and best-effort file overwrites.",
        (
            "SSD wear levelling, copy-on-write filesystems, snapshots, and backups prevent a universal physical-erasure guarantee.",
        ),
        ("ui_bridge", "application_service"),
        ("Nothing is deleted before an exact preview and explicit confirmation; symlinked targets are ignored.",),
    ),
    Capability(
        "privacy.encrypted-session", "privacy", "Encrypted session bundle",
        "supported", "sift.security_assurance", ("tests/test_security_assurance.py",),
        "Stores a portable complete session as a chunk-authenticated AES-256-GCM bundle derived from a researcher passphrase with scrypt.",
        (
            "The active session remains plaintext under OS filesystem protections while open; this format protects the portable at-rest bundle.",
        ),
        ("ui_bridge", "export"),
        ("Wrong passwords, tampering, traversal, links, device files, and non-empty restore destinations fail closed.",),
    ),
    Capability(
        "provenance.release-ledger", "provenance", "Tamper-evident release ledger",
        "supported", "sift.release_ledger", ("tests/test_release_ledger.py",),
        "Records model-visible releases in a locally verifiable append-only hash chain.",
        ("Tamper-evident is not tamper-proof and local administrators remain trusted.",),
        ("ui_bridge", "export", "qualification"),
        ("Mutation, deletion, and reordering break chain verification.",),
    ),
    Capability(
        "provenance.signed-export", "provenance", "Signed provenance export",
        "supported", "sift.security_assurance", ("tests/test_security_assurance.py",),
        "Signs the complete SHA-256 export manifest with Ed25519 and embeds the verification key.",
        (
            "Software-backed keys in the OS credential service do not replace organization-managed hardware signing, rotation, or revocation.",
        ),
        ("ui_bridge", "export", "qualification"),
        ("Any file addition, removal, or mutation makes verification fail without accessing the private key.",),
    ),
    Capability(
        "security.release-qualification", "reliability", "Release security qualification",
        "supported", "sift.security_assurance", ("tests/test_security_assurance.py",),
        "Generates a CycloneDX SBOM and runs secret, static, and known-dependency-vulnerability checks.",
        (
            "Known-vulnerability databases cannot detect unknown flaws, and an independent third-party penetration test remains an external release requirement.",
        ),
        ("qualification", "build_pipeline"),
        ("The lock-bound SBOM is generated, critical/high findings block release, and external assurance is never self-certified.",),
    ),
    Capability(
        "operations.signed-native-updates", "operations", "Signed native updates",
        "preview", "sift.update_service", (
            "tests/test_update_service.py",
            "tests/test_release_manifest.py",
        ),
        "Checks and stages the one matching native installer only after its canonical release manifest, release key, channel, version policy, artifact, and SBOM are verified.",
        (
            "A production build must embed an operator-reviewed HTTPS release location and public trust store; Sift never checks at startup, silently installs, or replaces the running application.",
        ),
        ("command_line", "build_pipeline", "os_credential_store"),
        (
            "Insecure locations, cross-origin redirects, signature failures, revoked keys, unauthorized rollback, platform mismatch, low disk space, truncation, and content tampering all fail closed before an installer is exposed.",
        ),
    ),
    Capability(
        "export.replication-package", "export", "Replication package",
        "supported", "sift.research_export",
        ("tests/test_research_export.py", "tests/test_reproducibility.py"),
        "Exports an integrity-manifested, model-free rerunnable package of scripts, environments, sanitized results, and provenance.",
        ("Original datasets are excluded unless separately supplied by the researcher.",),
        ("ui_bridge",),
        ("Partial builds are never published, restricted datasets remain excluded, and mutation blocks replay.",),
    ),
    Capability(
        "reproducibility.offline-rerun", "reproducibility", "Offline numerical rerun",
        "supported", "sift.reproducibility", ("tests/test_reproducibility.py",),
        "Verifies every bundle file and source hash, reruns exact local scripts without a model, and compares sanitized results numerically.",
        ("The researcher must separately supply source files with the recorded exact hashes and compatible licensed runtimes.",),
        ("ui_bridge", "export"),
        ("Tampering, missing sources, source drift, and script drift block execution; dependency drift is reported explicitly.",),
    ),
    Capability(
        "export.analysis-report", "export", "Analysis report",
        "supported", "sift.research_export",
        ("tests/test_analysis_report.py", "tests/test_analysis_report_pdf_pptx.py"),
        "Builds local HTML, PDF, and presentation reports from stored evidence.",
        ("Narrative is evidence-derived and intentionally narrower than a journal manuscript.",),
        ("ui_bridge",),
        ("Reports contain only exportable stored evidence and survive empty/corrupt optional state.",),
    ),
    Capability(
        "export.codebook", "export", "Codebook",
        "supported", "sift.research_export", ("tests/test_codebook.py",),
        "Builds a local codebook from canonical schema metadata.",
        ("Source-parser metadata fidelity varies by format.",),
        ("ui_bridge",),
        ("Non-exportable datasets are excluded and hostile labels are text-sanitized.",),
    ),
    Capability(
        "reliability.crash-safe-state", "reliability", "Crash-safe local state",
        "supported", "sift.reliability", ("tests/test_reliability.py",),
        "Uses locked atomic replacement, transactional SQLite migrations, integrity checks, capacity preflights, and clock-skew-safe ordering for durable state.",
        ("Atomic replacement cannot repair failing storage hardware or external backup corruption.",),
        ("application_service", "ui_bridge", "qualification"),
        ("Concurrent writes, disk faults, migration faults, corrupt stores, and backward clocks are injected in tests.",),
    ),
    Capability(
        "reliability.session-recovery", "reliability", "Session recovery assessment",
        "supported", "sift.reliability", ("tests/test_reliability.py",),
        "Reports store, transcript, audit-chain, free-space, stale-staging, and optional-index health and performs only confirmed bounded cleanup.",
        ("Primary evidence corruption is reported, never silently rewritten; recovery may still require restoring an external backup.",),
        ("ui_bridge", "application_service"),
        ("Optional indexes are quarantined and rebuildable; unrecognized or recent files are never deleted.",),
    ),
    Capability(
        "workflow.checkpoints", "research_workflow", "Analysis checkpoints",
        "preview", "sift.checkpoints", ("tests/test_checkpoints.py",),
        "Snapshots methodological/session state for comparison and rewind.",
        ("This is not a full content-addressed version-control system.",),
        ("ui_bridge",),
        ("Concurrent creation, restore, comparison, pruning, and failure atomicity tests pass.",),
    ),
    Capability(
        "workflow.reusable-skills", "research_workflow", "Reusable research skills",
        "preview", "sift.skills", ("tests/test_skills.py",),
        "Loads bounded, structurally validated local methodology guidance for supported workflows.",
        ("Skills are inert guidance and do not replace method-specific empirical validation.",),
        ("model_tool", "local_configuration"),
        ("Malformed, traversal, oversized, and duplicate skill definitions are refused.",),
    ),
    Capability(
        "integration.database-profiles", "integration", "Secure database profiles",
        "preview", "sift.database_profiles", ("tests/test_database_profiles.py",),
        "Stores named connection secrets in the OS credential vault, outside profile metadata.",
        ("Managed-identity lifecycle and live credential rotation are not fully certified.",),
        ("ui_bridge", "os_credential_store"),
        ("Profile metadata never contains the URI and failed updates restore the prior vault state.",),
    ),
    Capability(
        "operations.usage-meter", "operations", "Provider usage and cost meter",
        "preview", "sift.usage_meter", ("tests/test_usage_meter.py",),
        "Records provider-reported usage and clearly labelled local cost estimates.",
        ("Pricing can become stale and unknown models intentionally report unavailable cost.",),
        ("ui_bridge", "export"),
        ("Reported and estimated costs remain distinct and incomplete pricing is explicit.",),
    ),
    Capability(
        "operations.backend-qualification", "operations", "Backend qualification",
        "internal", "sift.qualification", ("tests/test_qualification.py",),
        "Produces a machine-readable verdict over contract, host runtime, and session integrity.",
        ("Does not contact live providers, databases, storage, or native hosts not currently running.",),
        ("ui_bridge", "release_validation"),
        ("A warning or untested external dependency cannot be represented as a pass.",),
    ),
    Capability(
        "operations.performance-qualification", "operations", "Performance qualification",
        "supported", "sift.performance", ("tests/test_performance.py",),
        "Measures startup, schema, profiling, linkage, extraction, conversion, memory, tokens, and local workflow latency against release budgets.",
        ("Python allocation peaks exclude native Arrow allocations, and live provider latency requires credentialed external qualification.",),
        ("qualification", "build_pipeline"),
        ("Representative deterministic fixtures pass on modest hardware and an over-budget metric fails the release verdict.",),
    ),
    Capability(
        "operations.scientific-evaluation", "operations", "Scientific evaluation gate",
        "supported", "sift.evaluation", ("tests/test_evaluation.py",),
        "Continuously measures fixed-answer statistical correctness, privacy invariants, reproducibility, method and claim contracts, and optional provider comparisons.",
        ("R, Stata, and live-provider differentials run only when researcher-managed runtimes, licenses, credentials, and billing are available.",),
        ("qualification", "build_pipeline", "application_service"),
        ("Every credential-free check meets its method/domain threshold; correctness or privacy regression fails the release gate; unavailable external systems are explicitly skipped.",),
    ),
    Capability(
        "operations.backend-api-v1", "operations", "Frozen GUI backend API v1",
        "supported", "sift.backend_api", ("tests/test_backend_api.py",),
        "Provides a UI-neutral, versioned application-service contract with structured requests, responses, progress events, cancellation, integrations, evidence, privacy warnings, and view-ready capabilities.",
        ("This is the backend contract for a future GUI; it does not itself provide a network server or build the GUI.",),
        ("application_service", "ui_bridge", "future_gui"),
        ("The v1 schema hash is pinned; every endpoint has a contract test; all responses pass a final secret-redaction boundary and all errors use one structured envelope.",),
    ),
    Capability(
        "operations.live-database-certification", "operations",
        "Optional live-database compatibility qualification", "internal",
        "sift.database_certification", (
            "tests/test_database_certification.py",
            "tests/live/test_database_live.py",
        ),
        "Maps remote-database compatibility requirements to opt-in, fail-closed disposable live scenarios, provides a read-only content-free provisioning preflight, and emits content-free compatibility evidence.",
        (
            "The optional program requires operator-provisioned synthetic fixtures and credentials; Sift does not operate, fund, proxy, or require a database account for release.",
        ),
        ("qualification", "external_compatibility"),
        (
            "When an operator opts into strict certification, all selected scenarios execute; the preflight exposes required variable names, authentication variants, and fixture fields but no values; absent inputs fail strict mode; credentials and query results are never written to the qualification report.",
        ),
    ),
    Capability(
        "operations.independent-pentest-intake", "operations",
        "Independent penetration-test evidence verification", "internal",
        "sift.pentest_assurance", ("tests/test_pentest_assurance.py",),
        "Produces a non-evidentiary artifact-bound assessor preflight and verifies an assessor-signed penetration-test attestation before the confidential-production release gate can pass.",
        (
            "Sift cannot assess itself; an approved independent assessor, an administrator-controlled Ed25519 assessor trust store, and a retained confidential report are still required.",
        ),
        ("qualification", "build_pipeline"),
        (
            "The preflight hashes the exact dist artifact and schema without creating assessor evidence; signature, report hash, scope, platform coverage, freshness, independence, finding disposition, and retest state all validate.",
        ),
    ),
)


def _provider_capabilities() -> list[Capability]:
    statuses: dict[str, ContractStatus] = {
        # Remote providers have complete adapter/unit contracts but have not
        # passed credentialed, live, cross-provider qualification. Adapter and
        # unit-test evidence alone is not production certification.
        "anthropic": "preview",
        "openai": "preview",
        "gemini": "preview",
        "openai_compatible": "experimental",
        "azure_openai": "preview",
        "vertex_gemini": "preview",
        "bedrock_anthropic": "preview",
        "vertex_anthropic": "preview",
    }
    verification: dict[str, tuple[str, ...]] = {
        "anthropic": (
            "tests/test_anthropic_result_error.py",
            "tests/test_provider_reconcile.py",
            "tests/test_tool_schema_consistency.py",
        ),
        "openai": (
            "tests/test_openai_lockdown.py",
            "tests/test_provider_reconcile.py",
            "tests/test_tool_schema_consistency.py",
        ),
        "gemini": (
            "tests/test_gemini.py",
            "tests/test_gemini_lockdown.py",
            "tests/test_tool_schema_consistency.py",
        ),
        "openai_compatible": (
            "tests/test_openai_compatible.py",
            "tests/test_tool_schema_consistency.py",
        ),
        "azure_openai": (
            "tests/test_enterprise_model_providers.py",
            "tests/test_openai_lockdown.py",
        ),
        "vertex_gemini": (
            "tests/test_enterprise_model_providers.py",
            "tests/test_gemini_lockdown.py",
        ),
        "bedrock_anthropic": (
            "tests/test_enterprise_model_providers.py",
            "tests/test_tool_schema_consistency.py",
        ),
        "vertex_anthropic": (
            "tests/test_enterprise_model_providers.py",
            "tests/test_tool_schema_consistency.py",
        ),
    }
    return [
        Capability(
            id=f"model-provider.{provider}",
            category="model_provider",
            label=provider.replace("_", " ").title(),
            status=statuses[provider],
            implementation=f"sift.provider.{provider}",
            verification=verification[provider],
            claim=(
                "Connects Sift's provider-neutral research workflow to a "
                "researcher-authorized model service or endpoint."
            ),
            limitations=(
                (
                    "Model access is not included; the researcher supplies the "
                    "account, credentials, endpoint, and provider billing."
                ),
                (
                    "Remote-provider retention and training controls are governed "
                    "by that provider account, not inferred by Sift."
                ),
            ),
            surfaces=("authentication", "model_picker", "conversation_service"),
            acceptance_criteria=(
                "Only canonical Sift function tools are present on every request.",
                "Timeout, cancellation, malformed tool calls, and interrupted streams fail safely.",
                "A credentialed live scenario matrix is required before status can become supported.",
            ),
        )
        for provider in SUPPORTED_PROVIDERS
    ]


def _database_capabilities() -> list[Capability]:
    return [
        Capability(
            id=f"database.{adapter.id}",
            category="database",
            label=adapter.label,
            status="supported",
            implementation="sift.connectors",
            verification=(
                "tests/test_connectors.py",
                "tests/test_database_driver_packaging.py",
            ),
            claim=(
                "Performs researcher-approved, read-only extraction into a "
                "bounded local Parquet dataset."
            ),
            limitations=(
                (
                    f"Requires the {adapter.install_extra!r} integration extra "
                    "when its driver is not built in."
                ),
                (
                    "The researcher supplies and authorizes the database account, "
                    "endpoint, credentials, operator terms, and any provider billing."
                ),
                (
                    "Client-side SQL classification is defense in depth; use a "
                    "server-enforced read-only identity."
                ),
            ),
            surfaces=("ui_bridge", "host_integration"),
            acceptance_criteria=(
                "Connection tests and discovery read metadata only.",
                "Extraction is read-only, bounded, streamed, hashed, and atomically materialized.",
                "Optional live conformance can be run against an operator-owned disposable backend without becoming a Sift release prerequisite.",
            ),
            compatibility_since=(
                1 if adapter.id in {"sqlite", "duckdb"} else 2
            ),
        )
        for adapter in DATABASE_ADAPTERS
    ]


def _cloud_source_capabilities() -> list[Capability]:
    return [
        Capability(
            id=f"cloud-source.{adapter.id}",
            category="cloud_source",
            label=adapter.label,
            status="preview",
            implementation="sift.cloud_sources",
            verification=("tests/test_cloud_sources.py",),
            claim=(
                "Streams a researcher-selected cloud object into a hashed, "
                "validated local dataset without model network access."
            ),
            limitations=(
                (
                    f"Requires the {adapter.install_extra!r} integration "
                    "extra when its SDK is not built in."
                ),
                "The remote storage service can observe and audit the download.",
            ),
            surfaces=("ui_bridge", "host_integration"),
            acceptance_criteria=(
                "The model and generated-code sandbox cannot invoke the source adapter.",
                "Downloads are bounded, hashed, validated, locally materialized, and provenance-recorded.",
                "A credentialed live object/version/checksum scenario is required for supported status.",
            ),
        )
        for adapter in CLOUD_SOURCE_ADAPTERS
    ]


def _format_capabilities() -> list[Capability]:
    selection_extensions = {
        ".zip", ".avro", ".xml", ".dbf", ".h5", ".hdf5", ".nc", ".netcdf",
        ".mat", ".fits", ".fit", ".fts", ".geojson", ".gpkg", ".shp",
        ".tif", ".tiff", ".vrt", ".vcf", ".bcf", ".bed",
        ".nii", ".dcm", ".fhir",
    }
    return [
        Capability(
            id=f"data-format.{extension.removeprefix('.')}",
            category="data_format",
            label=extension,
            status="preview" if extension in selection_extensions else "supported",
            implementation=(
                "sift.format_selection" if extension in selection_extensions else "sift.schema"
            ),
            verification=(
                ("tests/test_format_selection.py",)
                if extension in selection_extensions else
                ("tests/test_new_schema_formats.py", "tests/test_new_formats.py")
            ),
            claim=(
                "Materializes an explicitly selected object through an isolated parser into the canonical data layer."
                if extension in selection_extensions else
                "Extracts schema and loads the format through the canonical data layer."
            ),
            limitations=(
                "Full loads are subject to the configured local-memory ceiling.",
                (
                    "Format-specific metadata and type fidelity depend on the "
                    "underlying parser library."
                ),
            ),
            surfaces=("file_picker", "drop_zone", "schema_service", "analysis_runtime"),
            acceptance_criteria=(
                "Schema extraction, profiling, controlled requests, and analysis loading agree.",
                "Malformed input fails without exposing parser-owned row content.",
                "Type/metadata limitations are explicit and ambiguous multi-object inputs are refused.",
            ),
        )
        for extension in DATA_EXTENSIONS
    ]


def _request_capabilities() -> list[Capability]:
    return [
        Capability(
            id=f"data-request.{request_type}",
            category="controlled_data_request",
            label=request_type,
            status="supported",
            implementation="sift.data_request",
            verification=("tests/test_data_request.py",),
            claim="Returns a narrowly typed, policy-checked aggregate request.",
            limitations=(
                "Requests can be denied by dataset or enterprise policy.",
                (
                    "Disclosure thresholds and privacy budgets can reduce or "
                    "withhold the answer."
                ),
            ),
            surfaces=("model_tool",),
            acceptance_criteria=(
                "The request resolves an unambiguous real column through Sift-owned code.",
                "Policy, disclosure, privacy-budget, and query-fingerprint controls run before release.",
                "Errors and denials do not echo raw values.",
            ),
        )
        for request_type in SUPPORTED_REQUEST_TYPES
    ]


def _result_capabilities() -> list[Capability]:
    return [
        Capability(
            id=f"analysis-result.{result_type}",
            category="analysis_result",
            label=result_type,
            status="supported",
            implementation="sift.sanitizer",
            verification=(
                "tests/test_sanitizer.py",
                "tests/test_verification_all_shapes.py",
            ),
            claim="Validates and disclosure-controls this typed result shape.",
            limitations=(
                (
                    "Acceptance verifies the result contract, not the substantive "
                    "correctness of the research design."
                ),
                (
                    "Unsafe, undersized, malformed, or excessive fields are "
                    "suppressed, transformed, or rejected."
                ),
            ),
            surfaces=("runtime_helper", "sanitized_store", "model_tool_result", "export"),
            acceptance_criteria=(
                "Malformed, undersized, non-finite, inconsistent, and forbidden fields are rejected or removed.",
                "The result has deterministic verifier and renderer coverage.",
                "Acceptance never implies substantive research-design validity.",
            ),
        )
        for result_type in supported_types()
    ]


def _tool_capabilities() -> list[Capability]:
    return [
        Capability(
            id=f"research-tool.{tool_name}",
            category="research_tool",
            label=tool_name,
            status="supported",
            implementation="sift.tools",
            verification=("tests/test_tool_schema_consistency.py",),
            claim="Exposes this bounded Sift-owned operation to model sessions.",
            limitations=(
                (
                    "The tool is subject to Sift permission, path, size, policy, "
                    "and lifecycle checks."
                ),
            ),
            surfaces=("model_tool",),
            acceptance_criteria=(
                "The canonical provider schemas and handler registry contain the same tool exactly once.",
                "Permission, path, size, policy, cancellation, and error-redaction contracts pass.",
            ),
        )
        for tool_name in friendly_tool_names(prefixed=False)
    ]


def capabilities() -> tuple[Capability, ...]:
    """Return the current capabilities, derived from enforcement registries."""
    rows = [
        *_CORE_CAPABILITIES,
        *_provider_capabilities(),
        *_database_capabilities(),
        *_cloud_source_capabilities(),
        *_format_capabilities(),
        *_request_capabilities(),
        *_result_capabilities(),
        *_tool_capabilities(),
    ]
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("product capability ids must be unique")
    if any(row.status not in _VALID_STATUSES for row in rows):
        raise RuntimeError("product capability has an invalid support status")
    if any(not row.verification for row in rows):
        raise RuntimeError("every product capability must have verification evidence")
    if any(not row.acceptance_criteria for row in rows):
        raise RuntimeError("every product capability must have acceptance criteria")
    if any(not row.surfaces for row in rows):
        raise RuntimeError("every product capability must declare its product surfaces")
    if any(row.compatibility_since > CAPABILITY_COMPATIBILITY_VERSION for row in rows):
        raise RuntimeError("capability requires a newer compatibility contract")
    return tuple(rows)


def product_contract() -> dict[str, Any]:
    """Return a JSON-serializable, runtime-derived backend contract."""
    rows = capabilities()
    return {
        "schema_version": PRODUCT_CONTRACT_SCHEMA_VERSION,
        "compatibility_version": CAPABILITY_COMPATIBILITY_VERSION,
        "product": "Sift",
        "frontend_contract": {
            "schema_version": FRONTEND_CONTRACT_SCHEMA_VERSION,
            "session_states": ("needs_auth", "needs_session", "ready"),
            "terminal_turn_events": ("turn_done", "turn_error", "auth_failure"),
            "response_envelope": "JSON objects with explicit ok/status/state fields",
            "qualification_schema_version": 1,
        },
        "model_supply": dict(MODEL_SUPPLY),
        "database_supply": dict(DATABASE_SUPPLY),
        "generated_code_trust": dict(GENERATED_CODE_TRUST),
        "privacy_contract": dict(PRIVACY_CONTRACT),
        "quality_standards": dict(QUALITY_STANDARDS),
        "supported_scale": dict(SUPPORTED_SCALE),
        "risk_register": [risk.as_dict() for risk in RISK_REGISTER],
        "claims": [claim.as_dict() for claim in PRODUCT_CLAIMS],
        "limits": {
            "dataset_full_load_bytes": full_load_max_bytes(),
            "drag_drop_file_bytes": DRAG_DROP_FILE_MAX_BYTES,
            "drag_drop_aggregate_bytes": DRAG_DROP_AGGREGATE_MAX_BYTES,
            "model_image_bytes": MODEL_IMAGE_MAX_BYTES,
            "inline_script_bytes": INLINE_SCRIPT_MAX_BYTES,
            "inline_script_total_bytes": INLINE_SCRIPT_TOTAL_MAX_BYTES,
            "spreadsheet_archive_members": ZIP_CONTAINER_MAX_MEMBERS,
            "spreadsheet_archive_uncompressed_bytes": full_load_max_bytes(),
            "cloud_import_bytes": cloud_import_max_bytes(),
            "database_extract_rows": DEFAULT_ROW_LIMIT,
            "database_extract_bytes": DEFAULT_BYTE_LIMIT,
            "database_fetch_batch_rows": FETCH_BATCH_ROWS,
            "database_query_timeout_seconds": DEFAULT_QUERY_TIMEOUT_SECONDS,
            "catalog_schemas": MAX_CATALOG_SCHEMAS,
            "catalog_objects": MAX_CATALOG_OBJECTS,
            "catalog_columns_per_object": MAX_CATALOG_COLUMNS,
            "script_wall_seconds": DEFAULT_TIMEOUT_SECONDS,
            "script_cpu_seconds": script_cpu_limit_seconds(),
            "script_memory_bytes": script_memory_limit_bytes(),
            "script_processes": script_process_limit(),
            "script_single_file_bytes": script_file_size_limit_bytes(),
            "script_min_free_disk_bytes": script_min_free_disk_bytes(),
            "result_file_bytes": MAX_RESULT_FILE_BYTES,
            "result_payloads": MAX_RESULT_PAYLOADS,
        },
        "capability_counts": {
            category: sum(row.category == category for row in rows)
            for category in sorted({row.category for row in rows})
        },
        "capability_status_counts": {
            status: sum(row.status == status for row in rows)
            for status in sorted(_VALID_STATUSES)
        },
        "advertised_but_not_fully_certified": [
            row.id for row in rows if row.status in {"preview", "experimental"}
        ],
        "implemented_but_not_surfaced": [row.id for row in rows if not row.surfaces],
        "capabilities": [row.as_dict() for row in rows],
        "integrations": list_integration_manifests(),
    }


__all__ = [
    "FRONTEND_CONTRACT_SCHEMA_VERSION",
    "CAPABILITY_COMPATIBILITY_VERSION",
    "DATABASE_SUPPLY",
    "GENERATED_CODE_TRUST",
    "MODEL_SUPPLY",
    "PRIVACY_CONTRACT",
    "PRODUCT_CLAIMS",
    "PRODUCT_CONTRACT_SCHEMA_VERSION",
    "QUALITY_STANDARDS",
    "RISK_REGISTER",
    "SUPPORTED_SCALE",
    "Capability",
    "ContractStatus",
    "ProductClaim",
    "ProductRisk",
    "capabilities",
    "product_contract",
]
