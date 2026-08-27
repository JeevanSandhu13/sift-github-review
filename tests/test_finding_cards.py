"""Finding-card directive blocks in the web-UI markdown renderer.

``:::finding`` ... ``:::`` renders as a structured card, not a paragraph.

The renderer is JS-only (web UI), exercised through ``node`` the same
way test_markdown_renderer.py does. Skips rather than wedges the
suite if node or the renderer file is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from tests.runtime_probes import node_process_timeout_seconds

NODE = shutil.which("node")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RENDERER = _REPO_ROOT / "src" / "sift" / "web" / "markdown.js"

pytestmark = pytest.mark.skipif(
    NODE is None or not _RENDERER.is_file(),
    reason="node not installed or markdown.js missing",
)


def _render(text: str) -> str:
    js = (
        f"const fs = require('fs');"
        f"const code = fs.readFileSync({json.dumps(str(_RENDERER))}, 'utf8');"
        f"const window = {{}};"
        f"eval(code);"
        f"const input = fs.readFileSync(process.argv[1], 'utf8');"
        f"process.stdout.write(window.SiftMarkdown.render(input));"
    )
    # A temporary file avoids Windows command-line quoting of multiline model
    # output and avoids relying on Node's platform-specific stdin handle.
    with tempfile.TemporaryDirectory(prefix="sift render ") as temp_dir:
        input_path = Path(temp_dir) / "input.md"
        # Write bytes so Windows does not translate ``\n`` to ``\r\n``.
        # The renderer's directive fences are intentionally line-ending
        # sensitive, so text-mode translation would change the test input.
        input_path.write_bytes(text.encode("utf-8"))
        proc = subprocess.run(
            [NODE, "-e", js, str(input_path)],
            capture_output=True,
            check=False,
            text=True,
            # Keep the synchronous renderer bounded even on a loaded CI host.
            timeout=node_process_timeout_seconds(),
        )
    if proc.returncode != 0:
        raise AssertionError(
            f"renderer crashed:\nstderr={proc.stderr!r}"
            f"\nstdout={proc.stdout!r}"
        )
    return proc.stdout


def test_full_finding_card_renders_all_fields():
    out = _render(
        ":::finding\n"
        "claim: Tenure is associated with higher churn.\n"
        "result: [[result:M12|18%]]\n"
        "confidence: strong\n"
        "causality: associational\n"
        "caveat: correlational only.\n"
        ":::\n"
    )
    assert '<div class="finding-card">' in out
    assert '<div class="finding-claim">Tenure is associated with higher churn.</div>' in out
    assert 'finding-confidence-strong' in out
    assert 'strong confidence' in out
    assert 'finding-badge-causal' in out
    assert 'associational' in out
    assert 'data-result-id="M12"' in out  # evidence-cite reused
    assert '>18%</button>' in out
    assert '<div class="finding-caveat">correlational only.</div>' in out
    # Never rendered as a plain paragraph.
    assert '<p>:::finding' not in out


def test_finding_card_is_not_a_paragraph():
    """The literal directive syntax must never leak through as
    visible text -- that would mean the fence wasn't recognised."""
    out = _render(":::finding\nclaim: test\n:::\n")
    assert ":::finding" not in out
    assert ":::" not in out


@pytest.mark.parametrize("level,cls", [
    ("strong", "finding-confidence-strong"),
    ("moderate", "finding-confidence-moderate"),
    ("weak", "finding-confidence-weak"),
])
def test_confidence_level_maps_to_correct_css_class(level, cls):
    out = _render(f":::finding\nclaim: x\nconfidence: {level}\n:::\n")
    assert cls in out


def test_unrecognised_confidence_value_gets_no_color_class():
    out = _render(":::finding\nclaim: x\nconfidence: extremely_sure\n:::\n")
    assert "finding-confidence-strong" not in out
    assert "finding-confidence-moderate" not in out
    assert "finding-confidence-weak" not in out
    # Still rendered as a badge, just uncoloured.
    assert "finding-badge" in out


def test_sparse_card_omits_missing_field_divs():
    out = _render(":::finding\nclaim: just a claim\n:::\n")
    assert "finding-claim" in out
    assert "finding-badges" not in out
    assert "finding-evidence" not in out
    assert "finding-caveat" not in out


def test_empty_card_still_renders_the_wrapper():
    out = _render(":::finding\n:::\n")
    assert '<div class="finding-card">' in out


def test_unclosed_fence_still_renders_at_end_of_input():
    out = _render(":::finding\nclaim: truncated mid-stream\nconfidence: weak")
    assert "finding-claim" in out
    assert "truncated mid-stream" in out
    assert "finding-confidence-weak" in out


def test_unknown_key_is_ignored_not_rendered():
    out = _render(":::finding\nclaim: x\nbogus_field: should not appear\n:::\n")
    assert "should not appear" not in out


def test_finding_card_coexists_with_surrounding_prose():
    out = _render(
        "Some context first.\n\n"
        ":::finding\nclaim: the headline\n:::\n\n"
        "Some context after.\n"
    )
    assert "<p>Some context first.</p>" in out
    assert '<div class="finding-card">' in out
    assert "<p>Some context after.</p>" in out


def test_claim_and_caveat_run_through_inline_rendering():
    """Bold/code/emphasis inside a finding field should render the
    same way it would in a normal paragraph -- the card fields aren't
    a separate, dumber text path."""
    out = _render(
        ":::finding\nclaim: The **effect** is driven by `income`.\n:::\n"
    )
    assert "<strong>effect</strong>" in out
    assert "<code>income</code>" in out
