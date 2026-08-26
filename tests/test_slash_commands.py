"""Slash command palette: ``/analyze /profile /verify
/challenge /chart /report /privacy /connect``.

Resolution logic lives in slash_commands.js, deliberately DOM-free so
it's testable through node the same way test_markdown_renderer.py /
test_finding_cards.py exercise markdown.js. The 'ui' outcomes are
just names here -- app.js (not covered by this harness) maps them to
the real panel-opening functions.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODULE = _REPO_ROOT / "src" / "sift" / "web" / "slash_commands.js"

pytestmark = pytest.mark.skipif(
    NODE is None or not _MODULE.is_file(),
    reason="node not installed or slash_commands.js missing",
)


def _resolve(raw_text) -> dict | None:
    js = (
        f"const fs = require('fs');"
        f"const code = fs.readFileSync({json.dumps(str(_MODULE))}, 'utf8');"
        f"const window = {{}};"
        f"eval(code);"
        f"const input = JSON.parse(process.argv[1]);"
        f"const out = window.SiftSlashCommands.resolveSlashCommand(input);"
        f"process.stdout.write(JSON.stringify(out === undefined ? null : out));"
    )
    proc = subprocess.run(
        [NODE, "-e", js, "--", json.dumps(raw_text)],
        capture_output=True, check=False, text=True, timeout=10,
    )
    if proc.returncode != 0:
        raise AssertionError(f"crashed: stderr={proc.stderr!r}")
    return json.loads(proc.stdout)


def _names() -> list[str]:
    js = (
        f"const fs = require('fs');"
        f"const code = fs.readFileSync({json.dumps(str(_MODULE))}, 'utf8');"
        f"const window = {{}};"
        f"eval(code);"
        f"process.stdout.write(JSON.stringify(window.SiftSlashCommands.ALL_COMMAND_NAMES));"
    )
    proc = subprocess.run(
        [NODE, "-e", js], capture_output=True, check=False, text=True, timeout=10,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# All supported commands are present and resolve
# ---------------------------------------------------------------------------


def test_all_supported_commands_are_registered():
    names = set(_names())
    expected = {"analyze", "profile", "verify", "challenge", "chart",
               "report", "privacy", "connect"}
    assert expected <= names, f"missing: {expected - names}"


@pytest.mark.parametrize("name", [
    "analyze", "profile", "verify", "challenge", "chart",
    "report", "privacy", "connect",
])
def test_each_command_resolves_to_something(name):
    result = _resolve("/" + name)
    assert result is not None
    assert result["kind"] in ("chat", "ui")


@pytest.mark.parametrize("name", ["profile", "report", "privacy"])
def test_ui_commands_resolve_to_ui_kind_with_matching_name(name):
    result = _resolve("/" + name)
    assert result == {"kind": "ui", "name": name}


@pytest.mark.parametrize("name", [
    "analyze", "verify", "challenge", "chart", "connect",
])
def test_chat_commands_resolve_to_nonempty_text(name):
    result = _resolve("/" + name)
    assert result["kind"] == "chat"
    assert isinstance(result["text"], str) and len(result["text"]) > 10


# ---------------------------------------------------------------------------
# Non-command text passes through untouched
# ---------------------------------------------------------------------------


def test_plain_text_is_not_a_command():
    assert _resolve("What is the mean of income?") is None


def test_text_with_embedded_slash_not_at_start_is_not_a_command():
    assert _resolve("the path is /usr/local/bin") is None


def test_unrecognised_slash_word_is_not_a_command():
    assert _resolve("/frobnicate") is None
    assert _resolve("/analyz") is None  # typo, no fuzzy matching


def test_empty_string_is_not_a_command():
    assert _resolve("") is None


def test_bare_slash_alone_is_not_a_command():
    assert _resolve("/") is None


# ---------------------------------------------------------------------------
# Case insensitivity + trailing argument handling
# ---------------------------------------------------------------------------


def test_command_name_is_case_insensitive():
    assert _resolve("/ANALYZE")["kind"] == "chat"
    assert _resolve("/Privacy") == {"kind": "ui", "name": "privacy"}


def test_chat_command_with_trailing_argument_appends_it():
    result = _resolve("/chart the top 5 regions by revenue")
    assert result["kind"] == "chat"
    assert "top 5 regions by revenue" in result["text"]
    # The canned phrase is still there too, not replaced.
    assert "Create a chart" in result["text"]


def test_ui_command_ignores_trailing_argument_but_still_resolves():
    """A researcher typing '/privacy show me everything' still opens
    the panel -- the extra words are simply irrelevant to a UI
    action, not an error."""
    result = _resolve("/privacy show me everything")
    assert result == {"kind": "ui", "name": "privacy"}


def test_command_with_only_whitespace_argument_uses_canned_text_alone():
    result = _resolve("/analyze   ")
    assert result["kind"] == "chat"
    assert result["text"] == (
        "Analyze this dataset end to end: profile it, find what stands "
        "out, verify and stress-test the findings that matter, and "
        "tell me the things most worth knowing."
    )
