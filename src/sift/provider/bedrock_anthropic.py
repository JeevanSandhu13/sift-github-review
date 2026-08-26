"""Anthropic Claude through Amazon Bedrock Converse."""

from __future__ import annotations

import asyncio
import base64
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
    validate_aws_account,
    validate_identifier,
    validate_region,
)
from sift.provider.error_safety import provider_error_message
from sift.provider.managed_claude import (
    ManagedClaudeResponse,
    ManagedClaudeSession,
    verify_claude_lockdown,
)

PROVIDER_ID = "bedrock_anthropic"
ENV_REGION = "SIFT_BEDROCK_REGION"
ENV_MODEL = "SIFT_BEDROCK_ANTHROPIC_MODEL"
ENV_ACCOUNT = "SIFT_AWS_ACCOUNT_ID"
ENV_PROFILE = "SIFT_AWS_PROFILE"


@dataclass(frozen=True)
class BedrockAnthropicConfig:
    region: str
    model: str
    account: str | None
    profile: str | None

    @property
    def endpoint(self) -> str:
        suffix = "amazonaws.com.cn" if self.region.startswith("cn-") else "amazonaws.com"
        return f"https://bedrock-runtime.{self.region}.{suffix}"


def load_config() -> BedrockAnthropicConfig:
    region = validate_region(required_env(ENV_REGION, label="Amazon Bedrock region"))
    model = validate_identifier(
        required_env(ENV_MODEL, label="Amazon Bedrock model or inference-profile id"),
        label="Amazon Bedrock model or inference-profile id",
    )
    account = validate_aws_account(os.environ.get(ENV_ACCOUNT))
    raw_profile = os.environ.get(ENV_PROFILE, "").strip()
    profile = validate_identifier(raw_profile, label="AWS profile") if raw_profile else None
    config = BedrockAnthropicConfig(region, model, account, profile)
    enforce_managed_policy(
        PROVIDER_ID,
        endpoint=config.endpoint,
        region=region,
        account=account,
    )
    return config


def detect_auth() -> str:
    try:
        load_config()
        import boto3  # noqa: F401 - optional dependency readiness check
    except (ManagedDeploymentConfigError, ImportError):
        return "unknown"
    # The actual chain can resolve an SSO cache, web identity, ECS/EC2 role,
    # process provider, or temporary session.  Do not force a network fetch on
    # the readiness screen.
    return "workload_identity"


def _bedrock_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for block in content:
        kind = block.get("type")
        if kind == "text":
            converted.append({"text": str(block.get("text", ""))})
        elif kind == "image":
            raw_source = block.get("source")
            source = raw_source if isinstance(raw_source, dict) else {}
            mime = source.get("media_type")
            fmt = "png" if mime == "image/png" else "jpeg"
            converted.append({
                "image": {
                    "format": fmt,
                    "source": {"bytes": base64.b64decode(str(source.get("data", "")))},
                }
            })
        elif kind == "tool_use":
            raw_input = block.get("input")
            converted.append({
                "toolUse": {
                    "toolUseId": str(block.get("id", "")),
                    "name": str(block.get("name", "")),
                    "input": raw_input if isinstance(raw_input, dict) else {},
                }
            })
        elif kind == "tool_result":
            converted.append({
                "toolResult": {
                    "toolUseId": str(block.get("tool_use_id", "")),
                    "content": [{"text": str(block.get("content", ""))}],
                    "status": "error" if block.get("is_error") else "success",
                }
            })
    return converted


def _canonical_bedrock_content(content: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    out: list[dict[str, Any]] = []
    for block in content:
        if isinstance(block.get("text"), str):
            out.append({"type": "text", "text": block["text"]})
        elif isinstance(block.get("toolUse"), dict):
            tool = block["toolUse"]
            out.append({
                "type": "tool_use",
                "id": tool.get("toolUseId"),
                "name": tool.get("name"),
                "input": tool.get("input"),
            })
        elif isinstance(block.get("reasoningContent"), dict):
            reasoning = block["reasoningContent"]
            text_block = reasoning.get("reasoningText")
            if isinstance(text_block, dict) and isinstance(text_block.get("text"), str):
                out.append({"type": "thinking", "thinking": text_block["text"]})
    return tuple(out)


class BedrockAnthropicSession(ManagedClaudeSession):
    PROVIDER = PROVIDER_ID
    PROVIDER_LABEL = "Amazon Bedrock Claude"

    def __init__(
        self,
        cwd: Path,
        model: str,
        system_prompt: str,
        continue_conversation: bool = False,
        effort: str | None = None,
    ) -> None:
        super().__init__(cwd, model, system_prompt, continue_conversation, effort)
        self._config: BedrockAnthropicConfig | None = None

    async def open(self) -> None:
        if self._client is not None:
            return
        try:
            self._config = load_config()
            import boto3
            from botocore.config import Config

            from sift.integration_core import MODEL_REQUEST_TIMEOUT_SECONDS

            session = boto3.Session(profile_name=self._config.profile)
            self._client = session.client(
                "bedrock-runtime",
                region_name=self._config.region,
                config=Config(
                    connect_timeout=30,
                    read_timeout=MODEL_REQUEST_TIMEOUT_SECONDS,
                    retries={"max_attempts": 0, "mode": "standard"},
                    user_agent_extra="sift-research-assistant",
                ),
            )
            verify_claude_lockdown(self._tools)
        except ManagedDeploymentConfigError as exc:
            self._configuration_error = str(exc)
            await self.close()
        except ImportError:
            self._configuration_error = "Amazon Bedrock support requires the boto3 package"
            await self.close()
        except Exception as exc:  # noqa: BLE001 - cloud identity/SDK families vary
            self._configuration_error = provider_error_message(exc)
            await self.close()

    async def _request(self, messages: list[dict[str, Any]]) -> ManagedClaudeResponse:
        if self._client is None or self._config is None:
            raise RuntimeError("Bedrock client is not open")
        tools = [
            {"toolSpec": {
                "name": tool["name"],
                "description": tool["description"],
                "inputSchema": {"json": tool["input_schema"]},
            }}
            for tool in self._tools
        ]
        request = {
            "modelId": self._config.model,
            "messages": [
                {"role": message["role"], "content": _bedrock_content(message["content"])}
                for message in messages
            ],
            "system": [{"text": self._system_prompt}],
            "toolConfig": {"tools": tools, "toolChoice": {"auto": {}}},
            "inferenceConfig": {"maxTokens": 65_536},
            # Converse passes model-specific request fields through to Claude.
            "additionalModelRequestFields": {"output_config": {"effort": self.effort}},
        }
        response = await asyncio.to_thread(self._client.converse, **request)
        output = response.get("output", {})
        message = output.get("message", {}) if isinstance(output, dict) else {}
        content = message.get("content", []) if isinstance(message, dict) else []
        usage = response.get("usage", {})
        return ManagedClaudeResponse(
            content=_canonical_bedrock_content(content if isinstance(content, list) else []),
            input_tokens=int(usage.get("inputTokens", 0) or 0) if isinstance(usage, dict) else 0,
            output_tokens=int(usage.get("outputTokens", 0) or 0) if isinstance(usage, dict) else 0,
            stop_reason=response.get("stopReason") if isinstance(response.get("stopReason"), str) else None,
        )

    def _failure_event(self, error: Exception) -> Event:
        resource = (
            f"model/inference profile {self._config.model!r} in {self._config.region!r}"
            if self._config is not None
            else "the configured Bedrock deployment"
        )
        return translate_managed_error(
            error,
            provider_label=self.PROVIDER_LABEL,
            resource_label=resource,
        )


__all__ = [
    "BedrockAnthropicConfig",
    "BedrockAnthropicSession",
    "detect_auth",
    "load_config",
]
