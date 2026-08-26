"""Anthropic Claude partner models through Google Cloud Vertex AI."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from sift.provider.managed_claude import (
    ManagedClaudeResponse,
    ManagedClaudeSession,
    verify_claude_lockdown,
)

PROVIDER_ID = "vertex_anthropic"
ENV_PROJECT = "SIFT_VERTEX_ANTHROPIC_PROJECT"
ENV_LOCATION = "SIFT_VERTEX_ANTHROPIC_LOCATION"
ENV_MODEL = "SIFT_VERTEX_ANTHROPIC_MODEL"


@dataclass(frozen=True)
class VertexAnthropicConfig:
    project: str
    location: str
    model: str

    @property
    def endpoint(self) -> str:
        if self.location == "global":
            return "https://aiplatform.googleapis.com"
        if self.location in {"us", "eu"}:
            return f"https://aiplatform.{self.location}.rep.googleapis.com"
        return f"https://{self.location}-aiplatform.googleapis.com"


def load_config() -> VertexAnthropicConfig:
    project = validate_identifier(
        required_env(ENV_PROJECT, label="Vertex AI Claude project"),
        label="Vertex AI Claude project",
    )
    location = validate_region(
        required_env(ENV_LOCATION, label="Vertex AI Claude location"),
        label="Vertex AI Claude location",
    )
    model = validate_identifier(
        required_env(ENV_MODEL, label="Vertex AI Claude model"),
        label="Vertex AI Claude model",
    )
    config = VertexAnthropicConfig(project, location, model)
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
        import google.auth  # noqa: F401
        from anthropic import AnthropicVertex  # noqa: F401
    except (ManagedDeploymentConfigError, ImportError):
        return "unknown"
    return "workload_identity"


def _block_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        value = dump(exclude_none=True)
        if isinstance(value, dict):
            return value
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if kind == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")}
    if kind == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(block, "id", None),
            "name": getattr(block, "name", None),
            "input": getattr(block, "input", None),
        }
    return {"type": str(kind or "unknown")}


class VertexAnthropicSession(ManagedClaudeSession):
    PROVIDER = PROVIDER_ID
    PROVIDER_LABEL = "Vertex AI Claude"

    def __init__(
        self,
        cwd: Path,
        model: str,
        system_prompt: str,
        continue_conversation: bool = False,
        effort: str | None = None,
    ) -> None:
        super().__init__(cwd, model, system_prompt, continue_conversation, effort)
        self._config: VertexAnthropicConfig | None = None

    async def open(self) -> None:
        if self._client is not None:
            return
        try:
            self._config = load_config()
            from anthropic import AnthropicVertex

            from sift.integration_core import (
                MODEL_REQUEST_TIMEOUT_SECONDS,
                MODEL_SDK_MAX_RETRIES,
            )

            # AnthropicVertex obtains Google ADC itself.  That ADC may be a
            # workload-identity external account; no direct Anthropic key is
            # accepted or copied into the environment.
            self._client = AnthropicVertex(
                project_id=self._config.project,
                region=self._config.location,
                timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
                max_retries=MODEL_SDK_MAX_RETRIES,
            )
            verify_claude_lockdown(self._tools)
        except ManagedDeploymentConfigError as exc:
            self._configuration_error = str(exc)
            await self.close()
        except ImportError:
            self._configuration_error = (
                "Vertex AI Claude support requires the anthropic[vertex] package"
            )
            await self.close()
        except Exception as exc:  # noqa: BLE001 - cloud identity/SDK families vary
            self._configuration_error = provider_error_message(exc)
            await self.close()

    async def _request(self, messages: list[dict[str, Any]]) -> ManagedClaudeResponse:
        if self._client is None or self._config is None:
            raise RuntimeError("Vertex Anthropic client is not open")
        response = await asyncio.to_thread(
            self._client.messages.create,
            model=self._config.model,
            max_tokens=65_536,
            system=self._system_prompt,
            messages=messages,
            tools=self._tools,
            tool_choice={"type": "auto"},
            output_config={"effort": self.effort},
        )
        usage = getattr(response, "usage", None)
        return ManagedClaudeResponse(
            content=tuple(_block_dict(block) for block in (getattr(response, "content", None) or [])),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            stop_reason=getattr(response, "stop_reason", None),
        )

    def _failure_event(self, error: Exception) -> Event:
        resource = (
            f"partner model {self._config.model!r} in project "
            f"{self._config.project!r}/{self._config.location!r}"
            if self._config is not None
            else "the configured Vertex AI Claude deployment"
        )
        return translate_managed_error(
            error,
            provider_label=self.PROVIDER_LABEL,
            resource_label=resource,
        )


__all__ = [
    "VertexAnthropicConfig",
    "VertexAnthropicSession",
    "detect_auth",
    "load_config",
]
