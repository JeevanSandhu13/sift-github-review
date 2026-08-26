"""Gemini through Google Cloud Vertex AI as a distinct provider."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sift.provider.base import Event
from sift.provider.enterprise_common import (
    ManagedDeploymentConfigError,
    enforce_managed_policy,
    required_env,
    translate_managed_error,
    validate_identifier,
    validate_region,
)
from sift.provider.error_safety import provider_error_message
from sift.provider.gemini import GeminiSession, _verify_lockdown, build_gemini_tools

PROVIDER_ID = "vertex_gemini"
ENV_PROJECT = "SIFT_VERTEX_GEMINI_PROJECT"
ENV_LOCATION = "SIFT_VERTEX_GEMINI_LOCATION"
ENV_MODEL = "SIFT_VERTEX_GEMINI_MODEL"


@dataclass(frozen=True)
class VertexGeminiConfig:
    project: str
    location: str
    model: str

    @property
    def endpoint(self) -> str:
        if self.location == "global":
            return "https://aiplatform.googleapis.com"
        return f"https://{self.location}-aiplatform.googleapis.com"


def load_config() -> VertexGeminiConfig:
    project = validate_identifier(
        required_env(ENV_PROJECT, label="Vertex AI project"),
        label="Vertex AI project",
    )
    location = validate_region(
        required_env(ENV_LOCATION, label="Vertex AI location"),
        label="Vertex AI location",
    )
    model = validate_identifier(
        required_env(ENV_MODEL, label="Vertex AI Gemini model"),
        label="Vertex AI Gemini model",
    )
    config = VertexGeminiConfig(project, location, model)
    enforce_managed_policy(
        PROVIDER_ID,
        endpoint=config.endpoint,
        region=location,
        project=project,
    )
    return config


def detect_auth() -> str:
    try:
        load_config()
        import google.auth  # noqa: F401 - importability is readiness evidence
    except (ManagedDeploymentConfigError, ImportError):
        return "unknown"
    # ADC covers user ADC, service accounts, metadata identities, and external
    # account/workload identity federation.  We do not perform a network token
    # exchange merely to paint the readiness screen.
    return "workload_identity"


class VertexGeminiSession(GeminiSession):
    PROVIDER = PROVIDER_ID
    PROVIDER_LABEL = "Vertex AI Gemini"

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
        self._config: VertexGeminiConfig | None = None
        self._configuration_error: str | None = None

    def _wire_model_id(self) -> str:
        if self._config is not None:
            return self._config.model
        try:
            self._config = load_config()
        except ManagedDeploymentConfigError:
            return "unconfigured-vertex-gemini-model"
        return self._config.model

    async def open(self) -> None:
        if self._client is not None:
            return
        try:
            self._config = load_config()
            from google import genai
            from google.genai import types

            from sift.integration_core import MODEL_REQUEST_TIMEOUT_SECONDS

            self._client = genai.Client(
                vertexai=True,
                project=self._config.project,
                location=self._config.location,
                http_options=types.HttpOptions(
                    api_version="v1",
                    timeout=int(MODEL_REQUEST_TIMEOUT_SECONDS * 1_000),
                    retry_options=types.HttpRetryOptions(attempts=1),
                ),
            )
            self._tool = build_gemini_tools()
            _verify_lockdown(self._tool)
            self._chat = self._client.aio.chats.create(
                model=self._config.model,
                config=self._build_config(),
            )
        except ManagedDeploymentConfigError as exc:
            self._configuration_error = str(exc)
            await self.close()
        except Exception as exc:  # noqa: BLE001 - cloud identity/SDK families vary
            self._configuration_error = provider_error_message(exc)
            await self.close()

    def _missing_auth_reason(self) -> str:
        detail = self._configuration_error or "Application Default Credentials were unavailable"
        return (
            "Vertex AI Gemini is not ready. Configure its project, location, "
            f"model, and Google ADC/workload identity. Detail: {detail}"
        )

    def _translate_failure(self, error: Exception) -> Event:
        resource = (
            f"model {self._wire_model_id()!r} in project "
            f"{self._config.project!r}/{self._config.location!r}"
            if self._config is not None
            else "the configured Vertex AI Gemini deployment"
        )
        return translate_managed_error(
            error,
            provider_label=self.PROVIDER_LABEL,
            resource_label=resource,
        )

    def _error_secrets(self) -> tuple[str | None, ...]:
        # ADC tokens stay inside google-auth/google-genai and are never read by
        # Sift, so there is no secret string for the event renderer to carry.
        return ()


__all__ = ["VertexGeminiConfig", "VertexGeminiSession", "detect_auth", "load_config"]
