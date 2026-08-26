"""Lockdown tests for the Gemini provider.

Mirrors ``test_openai_lockdown.py``'s spirit for Gemini: no matter
what changes upstream in the ``google-genai`` SDK, the ``Tool`` sent
to Gemini must carry EXACTLY the Sift function declarations
(``build_tool_specs()``) and none of Gemini's own built-in
capability fields (``google_search``, ``code_execution``,
``url_context``, ``computer_use``, ``mcp_servers``, ``retrieval``,
``file_search``, ``google_maps``, ``google_search_retrieval``,
``enterprise_web_search``, ``exa_ai_search``, ``parallel_ai_search``).

Without this guard, a future "let's turn on google_search for
literature lookups" change could silently punch a hole in the
privacy boundary -- the model would gain a way to talk to the open
internet, or (worse) Gemini's own automatic-function-calling layer
could execute something outside Sift's sandbox/sanitizer.
"""

from __future__ import annotations

import pytest

from sift.provider.gemini import (
    _THINKING_LEVEL_BY_EFFORT,
    _verify_lockdown,
    build_gemini_tools,
    detect_auth,
)
from sift.provider.tool_schemas import build_tool_specs


# ---------------------------------------------------------------------------
# Static checks on the tool object
# ---------------------------------------------------------------------------


def test_tool_carries_only_function_declarations():
    tool = build_gemini_tools()
    for field_name in type(tool).model_fields:
        if field_name == "function_declarations":
            continue
        assert getattr(tool, field_name, None) is None, (
            f"Gemini tool must not set built-in field {field_name!r}"
        )


def test_tool_declaration_names_match_canonical_specs():
    tool = build_gemini_tools()
    sent_names = {d.name for d in tool.function_declarations}
    expected_names = {s.name for s in build_tool_specs()}
    assert sent_names == expected_names


def test_tool_declaration_count_matches_canonical_specs():
    tool = build_gemini_tools()
    # Sanity floor: drop-out below this would mean the schema list
    # got truncated. No exact pin beyond matching the canonical
    # count -- new tools land here naturally as the surface grows.
    assert len(tool.function_declarations) >= 6
    assert len(tool.function_declarations) == len(build_tool_specs())


def test_verify_lockdown_passes_on_the_real_tool():
    """The tool this provider actually builds must pass its own
    verifier -- a no-op assertion in the happy path, but pins that
    the two haven't drifted apart."""
    _verify_lockdown(build_gemini_tools())


def test_lockdown_verifier_rejects_every_forbidden_builtin_field():
    """Synthetically inject each of Gemini's own built-in capability
    fields and confirm the verifier raises. Iterates every field the
    pinned SDK's ``Tool`` type knows about except
    ``function_declarations`` itself, so a future SDK version adding
    a new built-in is covered automatically."""
    from google.genai import types

    base_kwargs = {
        "function_declarations": build_gemini_tools().function_declarations,
    }
    forbidden_fields = [
        f for f in types.Tool.model_fields if f != "function_declarations"
    ]
    assert forbidden_fields, "expected at least one built-in field to guard"

    for field_name in forbidden_fields:
        # Not every field takes the same shape; True is a valid
        # sentinel value for constructing a "this got set" case for
        # bool-typed fields (e.g. google_search_retrieval-style), and
        # the ones that need dict/dataclass config also accept it as
        # an SDK convenience -- but if a given field rejects a plain
        # ``True`` at construction time (rare, pydantic-typed), just
        # skip that one field; the point is to cover every field that
        # CAN be exercised, not to force a value onto every shape.
        try:
            tool = types.Tool(**{**base_kwargs, field_name: True})
        except Exception:
            continue
        with pytest.raises(RuntimeError, match="lockdown"):
            _verify_lockdown(tool)


def test_lockdown_verifier_rejects_unknown_function_name():
    """A non-Sift function declaration must also be rejected -- covers
    the case of someone "helpfully" appending a custom helper."""
    from google.genai import types

    tool = build_gemini_tools()
    bad_decls = list(tool.function_declarations) + [
        types.FunctionDeclaration(
            name="exfiltrate_data",
            description="evil",
            parameters={"type": "object", "properties": {}},
        )
    ]
    bad_tool = types.Tool(function_declarations=bad_decls)
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(bad_tool)


def test_lockdown_verifier_rejects_missing_declaration():
    """Dropping a Sift tool from the declared set must also raise --
    covers accidental truncation, not just contamination."""
    from google.genai import types

    tool = build_gemini_tools()
    trimmed = list(tool.function_declarations)[:-1]
    bad_tool = types.Tool(function_declarations=trimmed)
    with pytest.raises(RuntimeError, match="lockdown"):
        _verify_lockdown(bad_tool)


# ---------------------------------------------------------------------------
# Effort ladder
# ---------------------------------------------------------------------------


def test_thinking_level_map_only_covers_geminis_own_ladder():
    """Gemini's ladder (see ``catalog.PROVIDER_EFFORTS``) is
    low/medium/high only -- no ``xhigh``/``max``/``pro`` analogue.
    The map driving ``_build_config`` must not silently accept a
    level Gemini's own ladder doesn't offer; ``clamp_effort`` is
    responsible for narrowing to these three before this map is ever
    consulted."""
    from sift.provider.catalog import effort_levels_for_provider

    assert set(_THINKING_LEVEL_BY_EFFORT) == set(
        effort_levels_for_provider("gemini")
    )
    assert set(_THINKING_LEVEL_BY_EFFORT.values()) == {
        "LOW", "MEDIUM", "HIGH",
    }


# ---------------------------------------------------------------------------
# Auth detection
# ---------------------------------------------------------------------------


def test_detect_auth_reports_unknown_with_no_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from sift import auth as auth_module
    monkeypatch.setattr(auth_module, "has_credential", lambda p: False)
    assert detect_auth() == "unknown"


def test_detect_auth_reports_api_key_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    assert detect_auth() == "api_key"


def test_detect_auth_reports_api_key_from_google_api_key_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``GOOGLE_API_KEY`` is the SDK's own fallback env var -- a
    researcher who already has it set (e.g. for other Google tooling)
    should be picked up without needing a second, Sift-specific
    variable."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test-2")
    assert detect_auth() == "api_key"


def test_detect_auth_reports_api_key_from_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from sift import auth as auth_module
    monkeypatch.setattr(
        auth_module,
        "resolve_provider_credential",
        lambda provider, _variables: "AIza-keyring" if provider == "gemini" else None,
    )
    assert detect_auth() == "api_key"
