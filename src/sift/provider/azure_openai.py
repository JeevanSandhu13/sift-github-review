"""Azure OpenAI managed-deployment adapter.

This is intentionally a separate provider from direct OpenAI.  It uses an
Azure deployment name, an Azure resource endpoint, and either an Azure-scoped
API key or Microsoft Entra identity.  ``OPENAI_API_KEY`` is never read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sift.provider.base import Event
from sift.provider.enterprise_common import (
    ManagedDeploymentConfigError,
    enforce_managed_policy,
    required_env,
    translate_managed_error,
    validate_https_endpoint,
    validate_identifier,
    validate_region,
)
from sift.provider.openai import OpenAISession

PROVIDER_ID = "azure_openai"
ENV_ENDPOINT = "SIFT_AZURE_OPENAI_ENDPOINT"
ENV_DEPLOYMENT = "SIFT_AZURE_OPENAI_DEPLOYMENT"
ENV_REGION = "SIFT_AZURE_OPENAI_REGION"
ENV_RESOURCE = "SIFT_AZURE_OPENAI_RESOURCE"
ENV_AUTH_MODE = "SIFT_AZURE_OPENAI_AUTH"
_MANAGED_IDENTITY_SENTINEL = "__sift_entra_managed_identity__"


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    deployment: str
    region: str
    resource: str
    auth_mode: str

    @property
    def base_url(self) -> str:
        return f"{self.endpoint}/openai/v1/"


def load_config() -> AzureOpenAIConfig:
    endpoint = validate_https_endpoint(
        required_env(ENV_ENDPOINT, label="Azure OpenAI endpoint"),
        allowed_suffixes=("openai.azure.com", "services.ai.azure.com"),
        label="Azure OpenAI endpoint",
    )
    deployment = validate_identifier(
        required_env(ENV_DEPLOYMENT, label="Azure OpenAI deployment name"),
        label="Azure OpenAI deployment name",
    )
    region = validate_region(
        required_env(ENV_REGION, label="Azure OpenAI deployment region"),
        label="Azure OpenAI deployment region",
    )
    resource = validate_identifier(
        os.environ.get(ENV_RESOURCE, "").strip()
        or endpoint.removeprefix("https://").split(".", 1)[0],
        label="Azure OpenAI resource id",
    )
    auth_mode = os.environ.get(ENV_AUTH_MODE, "managed_identity").strip().casefold()
    if auth_mode not in {"managed_identity", "api_key"}:
        raise ManagedDeploymentConfigError(
            f"{ENV_AUTH_MODE} must be 'managed_identity' or 'api_key'"
        )
    enforce_managed_policy(
        PROVIDER_ID,
        endpoint=endpoint,
        region=region,
        project=resource,
    )
    return AzureOpenAIConfig(endpoint, deployment, region, resource, auth_mode)


def _resolve_azure_key() -> str | None:
    from sift import auth

    # Distinct keyring slot and environment name.  In particular, never pass
    # OPENAI_API_KEY through to Azure by accident.
    return auth.resolve_provider_credential(PROVIDER_ID, ("AZURE_OPENAI_API_KEY",))


def detect_auth() -> str:
    try:
        config = load_config()
    except ManagedDeploymentConfigError:
        return "unknown"
    if config.auth_mode == "api_key":
        return "api_key" if _resolve_azure_key() else "unknown"
    return "managed_identity"


class AzureOpenAISession(OpenAISession):
    """Responses-API session routed through an Azure deployment."""

    PROVIDER = PROVIDER_ID
    PROVIDER_LABEL = "Azure OpenAI"

    def __init__(
        self,
        cwd: Path,
        model: str,
        system_prompt: str,
        continue_conversation: bool = False,
        effort: str | None = None,
    ) -> None:
        super().__init__(
            cwd, model, system_prompt,
            continue_conversation=continue_conversation,
            effort=effort,
        )
        self._config: AzureOpenAIConfig | None = None
        self._configuration_error: str | None = None
        self._identity_credential: Any = None

    def _resolve_session_credential(self) -> str | None:
        try:
            self._config = load_config()
        except ManagedDeploymentConfigError as exc:
            self._configuration_error = str(exc)
            return None
        if self._config.auth_mode == "api_key":
            return _resolve_azure_key()
        return _MANAGED_IDENTITY_SENTINEL

    def _client_options(self, credential: str) -> dict[str, Any]:
        if self._config is None:
            raise ManagedDeploymentConfigError("Azure OpenAI configuration was not loaded")
        from sift.integration_core import (
            MODEL_REQUEST_TIMEOUT_SECONDS,
            MODEL_SDK_MAX_RETRIES,
        )

        api_key: Any = credential
        if credential == _MANAGED_IDENTITY_SENTINEL:
            try:
                from azure.identity import (
                    DefaultAzureCredential,
                    get_bearer_token_provider,
                )
            except ImportError as exc:
                raise ManagedDeploymentConfigError(
                    "Microsoft Entra authentication requires the azure-identity package"
                ) from exc
            self._identity_credential = DefaultAzureCredential()
            api_key = get_bearer_token_provider(
                self._identity_credential,
                "https://ai.azure.com/.default",
            )
        return {
            "api_key": api_key,
            "base_url": self._config.base_url,
            "timeout": MODEL_REQUEST_TIMEOUT_SECONDS,
            "max_retries": MODEL_SDK_MAX_RETRIES,
        }

    async def open(self) -> None:
        try:
            await super().open()
        except ManagedDeploymentConfigError as exc:
            self._configuration_error = str(exc)
            await self.close()

    async def close(self) -> None:
        await super().close()
        credential = self._identity_credential
        self._identity_credential = None
        if credential is not None:
            try:
                credential.close()
            except Exception:  # noqa: BLE001, S110 - best-effort SDK close
                pass

    def _wire_model_id(self) -> str:
        if self._config is None:
            try:
                self._config = load_config()
            except ManagedDeploymentConfigError:
                return "unconfigured-azure-deployment"
        return self._config.deployment

    def _missing_auth_reason(self) -> str:
        if self._configuration_error:
            return f"Azure OpenAI is not ready: {self._configuration_error}"
        config = self._config
        if config is not None and config.auth_mode == "api_key":
            return (
                "no Azure OpenAI API key is configured. Store one for the "
                "azure_openai provider or set AZURE_OPENAI_API_KEY."
            )
        return "Azure OpenAI managed identity could not be initialized."

    def _translate_failure(self, error: Exception, safe_message: str) -> Event:
        del safe_message
        return translate_managed_error(
            error,
            provider_label=self.PROVIDER_LABEL,
            resource_label=f"deployment {self._wire_model_id()!r}",
            secrets=self._error_secrets(),
        )

    def _error_secrets(self) -> tuple[str | None, ...]:
        config = self._config
        return (_resolve_azure_key(),) if config and config.auth_mode == "api_key" else ()


__all__ = [
    "AzureOpenAIConfig",
    "AzureOpenAISession",
    "detect_auth",
    "load_config",
]
