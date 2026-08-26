"""Shared, fail-closed contracts for managed model deployments.

Azure OpenAI, Vertex AI and Amazon Bedrock are distinct data processors, not
aliases for the vendors' direct APIs.  This module centralises the parts that
must remain identical across their adapters: strict configuration parsing,
enterprise allowlists, privacy manifests, and secret-safe cloud error
translation.  It deliberately performs no network I/O.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

from sift.provider.base import AuthFailure, TurnError
from sift.provider.error_safety import provider_error_message

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,254}$")
_REGION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,62}$")
_AWS_ACCOUNT_RE = re.compile(r"^[0-9]{12}$")


class ManagedDeploymentConfigError(ValueError):
    """Safe, actionable configuration or policy error."""


@dataclass(frozen=True)
class ManagedPrivacyManifest:
    provider: str
    processor: str
    authentication_boundary: str
    processing_location: str
    retention: str
    training: str
    sift_guarantees: tuple[str, ...]
    customer_controls: tuple[str, ...]
    cautions: tuple[str, ...]
    verified_by_sift: tuple[str, ...]
    not_verified_by_sift: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIVACY_MANIFESTS: dict[str, ManagedPrivacyManifest] = {
    "azure_openai": ManagedPrivacyManifest(
        provider="azure_openai",
        processor="Microsoft Azure OpenAI / Foundry",
        authentication_boundary=(
            "Azure API key or Microsoft Entra identity scoped to the configured "
            "Azure resource; direct OpenAI credentials are never consulted"
        ),
        processing_location=(
            "Configured Azure deployment geography; Global and DataZone deployment "
            "types can process across their documented wider geography"
        ),
        retention=(
            "Sift sends store=false. Azure safety and abuse-monitoring processing "
            "still follows the customer's Azure configuration and agreement"
        ),
        training="Azure states prompts and completions are not used to train base models",
        sift_guarantees=(
            "Only Sift function tools are sent",
            "Conversation items are replayed locally",
            "Raw datasets and cloud credentials never enter model context",
        ),
        customer_controls=(
            "Regional/DataZone deployment selection",
            "Azure RBAC and managed identity",
            "Modified abuse monitoring when approved by Microsoft",
        ),
        cautions=(
            "Content filters run synchronously",
            "Default abuse monitoring can permit flagged-content review",
        ),
        verified_by_sift=("endpoint", "deployment name", "declared region", "auth mode"),
        not_verified_by_sift=("Azure RBAC grants", "deployment SKU", "content-logging status"),
    ),
    "vertex_gemini": ManagedPrivacyManifest(
        provider="vertex_gemini",
        processor="Google Cloud Vertex AI (Google Gemini model)",
        authentication_boundary=(
            "Google Application Default Credentials, including workload identity; "
            "Gemini Developer API keys are never consulted"
        ),
        processing_location="Configured Vertex project and location",
        retention=(
            "Vertex retention depends on enabled features and abuse-monitoring status; "
            "Sift disables grounding, explicit caching, and session resumption"
        ),
        training=(
            "Google Cloud states customer data is not used to train or fine-tune "
            "models without permission or instruction"
        ),
        sift_guarantees=(
            "Only Sift function declarations are sent",
            "No Gemini File API, grounding, or explicit cache",
            "Raw datasets and Google credentials never enter model context",
        ),
        customer_controls=("IAM", "VPC Service Controls", "regional endpoint", "ZDR controls"),
        cautions=("In-memory project-isolated caching may apply", "Abuse logging may require an exception for ZDR"),
        verified_by_sift=("project", "location", "model id", "ADC configuration attempt"),
        not_verified_by_sift=("IAM grants", "VPC-SC perimeter", "ZDR approval"),
    ),
    "bedrock_anthropic": ManagedPrivacyManifest(
        provider="bedrock_anthropic",
        processor="Amazon Bedrock (Anthropic Claude model)",
        authentication_boundary=(
            "AWS default credential chain, IAM role, or temporary credentials; "
            "direct Anthropic API keys are never consulted"
        ),
        processing_location="Configured AWS region",
        retention=(
            "AWS documents that Bedrock does not store or log Converse prompts and completions"
        ),
        training="AWS states content is not used to train models or shared with model providers",
        sift_guarantees=(
            "Only Sift tool specifications are sent through Converse",
            "Conversation state is retained locally by Sift",
            "Raw datasets and AWS credentials never enter model context",
        ),
        customer_controls=("IAM", "regional endpoint", "VPC endpoints", "CloudTrail metadata auditing"),
        cautions=("Model access and inference-profile availability vary by region",),
        verified_by_sift=("region", "model/inference-profile id", "declared AWS account when policy requires it"),
        not_verified_by_sift=("effective IAM permissions", "VPC endpoint routing", "actual caller account without an STS request"),
    ),
    "vertex_anthropic": ManagedPrivacyManifest(
        provider="vertex_anthropic",
        processor="Google Cloud Vertex AI (Anthropic Claude partner model)",
        authentication_boundary=(
            "Google Application Default Credentials, including workload identity; "
            "direct Anthropic API keys are never consulted"
        ),
        processing_location="Configured Vertex project and location",
        retention=(
            "Google Cloud's Vertex data-processing contract applies; optional request-response "
            "logging is controlled by the customer and is not enabled by Sift"
        ),
        training="Google Cloud and Anthropic partner-model terms apply; Sift makes no account-specific inference",
        sift_guarantees=(
            "Only Sift function tools are sent",
            "Conversation state is retained locally by Sift",
            "Raw datasets and Google credentials never enter model context",
        ),
        customer_controls=("IAM", "regional or multi-region endpoint", "VPC Service Controls", "request-response logging"),
        cautions=("Claude model availability and retirement dates differ from the direct Anthropic API",),
        verified_by_sift=("project", "location", "partner-model id", "ADC configuration attempt"),
        not_verified_by_sift=("IAM grants", "VPC-SC perimeter", "optional logging configuration"),
    ),
}


def privacy_manifest(provider: str) -> ManagedPrivacyManifest:
    try:
        return PRIVACY_MANIFESTS[provider]
    except KeyError as exc:
        raise ValueError(f"unknown managed provider: {provider!r}") from exc


def required_env(name: str, *, label: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ManagedDeploymentConfigError(f"{label} is required ({name})")
    return value


def validate_identifier(value: str, *, label: str) -> str:
    value = value.strip()
    if not _ID_RE.fullmatch(value) or ".." in value or value.startswith(("/", "-")):
        raise ManagedDeploymentConfigError(f"{label} has an invalid value")
    return value


def validate_region(value: str, *, label: str = "cloud region") -> str:
    value = value.strip().casefold()
    if not _REGION_RE.fullmatch(value):
        raise ManagedDeploymentConfigError(f"{label} has an invalid value")
    return value


def validate_aws_account(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    value = value.strip()
    if not _AWS_ACCOUNT_RE.fullmatch(value):
        raise ManagedDeploymentConfigError("AWS account id must contain exactly 12 digits")
    return value


def validate_https_endpoint(
    value: str,
    *,
    allowed_suffixes: tuple[str, ...],
    label: str,
) -> str:
    """Return a canonical origin after strict endpoint validation."""
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ManagedDeploymentConfigError(f"{label} is not a valid URL") from exc
    host = (parsed.hostname or "").rstrip(".").casefold()
    if (
        parsed.scheme != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or port not in (None, 443)
        or not any(host == suffix or host.endswith(f".{suffix}") for suffix in allowed_suffixes)
    ):
        raise ManagedDeploymentConfigError(
            f"{label} must be a credential-free HTTPS endpoint on an approved cloud domain"
        )
    path = parsed.path.rstrip("/")
    if path and path != "/openai/v1":
        raise ManagedDeploymentConfigError(f"{label} contains an unsupported path")
    return f"https://{host}"


def enforce_managed_policy(
    provider: str,
    *,
    endpoint: str | None,
    region: str,
    project: str | None = None,
    account: str | None = None,
) -> None:
    """Apply provider, endpoint, region, project, and account restrictions."""
    from sift import enterprise_policy

    policy = enterprise_policy.load_enterprise_policy()
    if not enterprise_policy.model_provider_allowed(provider, policy):
        raise ManagedDeploymentConfigError(
            f"managed provider {provider!r} is blocked by enterprise policy"
        )
    if not enterprise_policy.integration_endpoint_allowed(endpoint, policy):
        raise ManagedDeploymentConfigError("managed endpoint is blocked by enterprise policy")
    if not enterprise_policy.integration_region_allowed(region, policy):
        raise ManagedDeploymentConfigError(
            f"region {region!r} is not approved by enterprise policy"
        )
    if not enterprise_policy.managed_project_allowed(project, policy):
        raise ManagedDeploymentConfigError(
            f"project/resource {project!r} is not approved by enterprise policy"
        )
    if not enterprise_policy.managed_account_allowed(account, policy):
        raise ManagedDeploymentConfigError(
            f"cloud account {account!r} is not approved by enterprise policy"
        )


def translate_managed_error(
    error: Exception,
    *,
    provider_label: str,
    resource_label: str,
    secrets: tuple[str | None, ...] = (),
) -> AuthFailure | TurnError:
    """Map cloud/SDK failures into the common actionable event contract."""
    safe = provider_error_message(error, secrets=secrets)
    lower = safe.casefold()
    status = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    code = ""
    if isinstance(response, dict):
        detail = response.get("Error")
        if isinstance(detail, dict):
            code = str(detail.get("Code", ""))
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict) and status is None:
            status = metadata.get("HTTPStatusCode")
    code_lower = code.casefold()
    kind = type(error).__name__.casefold()
    if status in {401, 403} or any(
        token in f"{kind} {code_lower} {lower}"
        for token in (
            "unauthorized", "unauthenticated", "accessdenied", "forbidden",
            "credential", "invalidauthentication", "permissiondenied",
        )
    ):
        return AuthFailure(
            reason=(
                f"{provider_label} rejected the managed identity or credential for "
                f"{resource_label}. Verify the cloud IAM/RBAC assignment. Underlying error: {safe}"
            )
        )
    if status == 429 or any(
        token in f"{kind} {code_lower} {lower}"
        for token in ("throttl", "ratelimit", "resourceexhausted", "quota")
    ):
        return TurnError(message=(
            f"{provider_label} throttled or quota-limited {resource_label}. No ambiguous "
            f"model turn was retried automatically. Wait or inspect cloud quota. Underlying error: {safe}"
        ))
    if status == 404 or any(
        token in f"{kind} {code_lower} {lower}"
        for token in ("notfound", "resourcenotfound", "modelnotready")
    ):
        return TurnError(message=(
            f"{provider_label} could not find or serve {resource_label}. Check deployment/model "
            f"access in the configured project and region. Underlying error: {safe}"
        ))
    if "context" in lower and "token" in lower and any(
        token in lower for token in ("limit", "exceed", "too long")
    ):
        return TurnError(message=(
            f"{provider_label} rejected the turn because it exceeded the managed model's "
            f"context limit. Start a new summarized session. Underlying error: {safe}"
        ))
    if any(token in f"{kind} {lower}" for token in ("timeout", "timed out")):
        return TurnError(message=(
            f"{provider_label} request timed out. The ambiguous turn was not retried "
            f"automatically. Underlying error: {safe}"
        ))
    if isinstance(status, int) and status >= 500:
        return TurnError(message=(
            f"{provider_label} returned a transient service error ({status}). Retry when "
            f"ready. Underlying error: {safe}"
        ))
    return TurnError(message=f"{provider_label} request failed: {safe}")


def configured_state(*required_values: str | None) -> Literal["ready", "needs_configuration"]:
    return "ready" if all(value and value.strip() for value in required_values) else "needs_configuration"


__all__ = [
    "PRIVACY_MANIFESTS",
    "ManagedDeploymentConfigError",
    "ManagedPrivacyManifest",
    "configured_state",
    "enforce_managed_policy",
    "privacy_manifest",
    "required_env",
    "translate_managed_error",
    "validate_aws_account",
    "validate_https_endpoint",
    "validate_identifier",
    "validate_region",
]
