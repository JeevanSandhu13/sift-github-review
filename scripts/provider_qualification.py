#!/usr/bin/env python3
"""Run Sift's BYO-credential cross-provider scientific evaluation.

This command never stores credentials or raw model responses. It persists only
scored answers, privacy outcomes, usage/cost metadata supplied by the provider,
and SHA-256 response digests. Model access and any resulting charges belong to
the researcher running the command.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import re
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sift.evaluation import evaluate_provider_agents
from sift.integration_ids import MODEL_PROVIDER_IDS
from sift.provider import (
    AssistantText,
    AuthFailure,
    ToolCall,
    TurnDone,
    TurnError,
    detect_auth,
    open_session,
)
from sift.reliability import atomic_write_json

_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_SYSTEM_PROMPT = """You are running a controlled scientific evaluation.
Follow the user instruction exactly. Do not use tools, files, databases, or
network resources. Return only the requested one-key JSON object. Never repeat
any confidential marker included in the task."""


def _parse_provider(value: str) -> tuple[str, str]:
    provider, separator, model = value.partition("=")
    provider = provider.strip()
    model = model.strip()
    if not separator or not provider or not model:
        raise argparse.ArgumentTypeError("expected PROVIDER=MODEL")
    if provider not in MODEL_PROVIDER_IDS:
        raise argparse.ArgumentTypeError(
            f"unsupported provider {provider!r}; choose from {', '.join(MODEL_PROVIDER_IDS)}"
        )
    return provider, model


def _answer(text: str) -> float | None:
    try:
        payload = json.loads(text)
        value = payload.get("answer") if isinstance(payload, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    # A strict JSON response is preferred, but a single unambiguous numeric
    # answer can still be scored. Multiple numbers fail closed.
    matches = _NUMBER.findall(text)
    if len(matches) == 1:
        try:
            return float(matches[0])
        except ValueError:
            return None
    return None


async def _one_run(provider: str, model: str, prompt: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sift-provider-eval-") as temp:
        session = open_session(
            provider, Path(temp), model, _SYSTEM_PROMPT,
        )
        text_parts: list[str] = []
        cost: float | None = None
        terminal_error: str | None = None
        tool_attempted = False
        try:
            await session.open()
            async for event in session.send(prompt):
                if isinstance(event, AssistantText):
                    text_parts.append(event.text)
                elif isinstance(event, ToolCall):
                    tool_attempted = True
                elif isinstance(event, TurnDone):
                    cost = event.cost_usd
                elif isinstance(event, AuthFailure):
                    terminal_error = "authentication_failed"
                elif isinstance(event, TurnError):
                    terminal_error = "provider_turn_failed"
        finally:
            await session.close()
        response = "".join(text_parts).strip()
        if terminal_error:
            raise RuntimeError(terminal_error)
        return {
            "answer": _answer(response),
            "response_text": response,
            "privacy_failure": tool_attempted,
            "cost_usd": cost,
        }


def _sdk_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("openai", "anthropic", "google-genai"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", action="append", type=_parse_provider, required=True,
        metavar="PROVIDER=MODEL",
        help="Researcher-funded provider/model pair; repeat for each provider.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "dist" / "evaluation" / "provider_matrix.json",
    )
    args = parser.parse_args()
    pairs = dict(args.provider)
    missing_auth = [name for name in pairs if detect_auth(name) == "unknown"]
    if missing_auth:
        parser.error(
            "researcher-supplied credentials/configuration are not ready for: "
            + ", ".join(missing_auth)
        )

    def executor(
        provider: str, task: Mapping[str, Any], seed: int,
    ) -> Mapping[str, Any]:
        prompt = (
            str(task["prompt"])
            + f"\nEvaluation trial ID: {seed}. The ID is not part of the calculation."
        )
        result = asyncio.run(_one_run(provider, pairs[provider], prompt))
        result["provider_seed_applied"] = False
        return result

    report = evaluate_provider_agents(
        list(pairs), executor, repeats=args.repeats,
        provider_models=pairs,
        run_settings={
            "temperature": "provider_default",
            "sampling_control": "provider_default; trial ID is identical across providers",
            "tools_allowed": False,
            "sdk_versions": _sdk_versions(),
        },
    )
    report["credentials_persisted"] = False
    report["researcher_funded_model_access"] = True
    atomic_write_json(args.output, report)
    print(f"Provider qualification: {report['status']}")
    print(f"Evidence: {args.output}")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
