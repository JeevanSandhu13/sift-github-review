"""Sift Skills loader, sanitization, and
``get_skill`` tool.

A Sift Skill is TEXT: judgment/workflow guidance, never code that
runs. The properties worth testing are (1) the loader parses the
real bundled skills correctly and degrades gracefully on malformed
files rather than crashing a session, (2) every string that reaches
the model goes through the appropriate text-safety sanitizer with
newlines preserved in the body (not flattened like a single-line
label), and (3) the ``get_skill`` tool and the system prompt's
always-visible index both wire up to the same loader correctly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from sift import skills


# ---------------------------------------------------------------------------
# Builtin library — the two real shipped skills
# ---------------------------------------------------------------------------

def test_builtin_skills_load_and_have_expected_slugs():
    b = skills.list_builtin_skills()
    slugs = {s.slug for s in b}
    assert "staggered_did" in slugs
    assert "survey_weighted_analysis" in slugs
    for s in b:
        assert s.source == "builtin"
        assert s.name
        assert s.description
        assert s.body
        assert "\n" in s.body  # multi-paragraph structure preserved


def test_shipped_domain_skills_pass_structural_validation():
    domain_slugs = {
        "clinical_observational_study", "longitudinal_panel_analysis",
        "predictive_model_validation", "geospatial_research",
    }
    loaded = {skill.slug: skill for skill in skills.list_builtin_skills()}
    assert domain_slugs <= set(loaded)
    for slug in domain_slugs:
        assert skills.validate_skill(loaded[slug]) == ()


def test_builtin_skill_bodies_are_not_flattened_to_one_line():
    b = skills.list_builtin_skills()
    did = skills.get_skill_body(b, "staggered_did")
    assert did is not None
    # A real markdown header must survive intact.
    assert "## " in did.body


# ---------------------------------------------------------------------------
# User skills — loading, malformed-file handling
# ---------------------------------------------------------------------------

def test_user_skills_dir_absent_returns_empty(tmp_path: Path):
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_well_formed_loads(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "my_skill.md").write_text("""---
name: My Custom Skill
description: A researcher-authored skill for this project.
when_to_use: When the researcher asks about widgets.
---
# My Custom Skill

Line one.

Line two, a new paragraph.
""", encoding="utf-8")
    user = skills.list_user_skills(tmp_path)
    assert len(user) == 1
    s = user[0]
    assert s.slug == "my_skill"
    assert s.source == "user"
    assert "Line one." in s.body
    assert "Line two, a new paragraph." in s.body


def test_user_skill_missing_frontmatter_delimiter_skipped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("no frontmatter here at all\n", encoding="utf-8")
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_missing_name_skipped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("""---
description: has a description but no name
---
body
""", encoding="utf-8")
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_missing_description_skipped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("""---
name: has a name but no description
---
body
""", encoding="utf-8")
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_unclosed_frontmatter_skipped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("""---
name: unclosed
description: the closing --- is missing
body without a second delimiter
""", encoding="utf-8")
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_garbage_yaml_frontmatter_skipped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("""---
{{{ not valid yaml ][
---
body
""", encoding="utf-8")
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_frontmatter_not_a_mapping_skipped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "broken.md").write_text("""---
- just
- a
- list
---
body
""", encoding="utf-8")
    assert skills.list_user_skills(tmp_path) == []


def test_user_skill_missing_when_to_use_defaults_empty(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "no_trigger.md").write_text("""---
name: No Trigger Field
description: description only, no when_to_use key
---
body text
""", encoding="utf-8")
    user = skills.list_user_skills(tmp_path)
    assert len(user) == 1
    assert user[0].when_to_use == ""


# ---------------------------------------------------------------------------
# Sanitization — control chars stripped, structure preserved, truncation
# ---------------------------------------------------------------------------

def test_control_characters_stripped_from_body(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    evil = "line one\x07\n\nline two​ after zero-width"
    (d / "evil.md").write_text(
        "---\nname: Evil\ndescription: has control chars\n---\n" + evil,
        encoding="utf-8",
    )
    s = skills.list_user_skills(tmp_path)[0]
    assert "\x07" not in s.body
    assert "​" not in s.body
    assert "line one" in s.body
    assert "line two" in s.body


def test_body_truncates_over_cap_with_marker(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    long_body = "word " * 5000  # well over _BODY_MAX_LEN, under hard-reject
    (d / "long.md").write_text(
        "---\nname: Long\ndescription: very long body\n---\n" + long_body,
        encoding="utf-8",
    )
    s = skills.list_user_skills(tmp_path)[0]
    assert len(s.body) <= skills._BODY_MAX_LEN
    assert "[TRUNCATED]" in s.body


def test_name_and_description_capped(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "capped.md").write_text(
        "---\nname: " + ("x" * 500) + "\ndescription: " + ("y" * 500)
        + "\n---\nbody\n",
        encoding="utf-8",
    )
    s = skills.list_user_skills(tmp_path)[0]
    assert len(s.name) <= skills._NAME_MAX_LEN
    assert len(s.description) <= skills._DESCRIPTION_MAX_LEN


# ---------------------------------------------------------------------------
# load_all_skills — builtin + user merge, user overrides builtin by slug
# ---------------------------------------------------------------------------

def test_load_all_skills_includes_builtin_and_user(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "extra.md").write_text(
        "---\nname: Extra\ndescription: an extra user skill\n---\nbody\n",
        encoding="utf-8",
    )
    all_skills = skills.load_all_skills(tmp_path)
    slugs = {s.slug for s in all_skills}
    assert "staggered_did" in slugs
    assert "survey_weighted_analysis" in slugs
    assert "extra" in slugs


def test_user_skill_overrides_builtin_with_same_slug(tmp_path: Path):
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "staggered_did.md").write_text(
        "---\nname: My Own DiD Notes\ndescription: overrides the builtin\n---\n"
        "researcher's own version\n",
        encoding="utf-8",
    )
    all_skills = skills.load_all_skills(tmp_path)
    did_entries = [s for s in all_skills if s.slug == "staggered_did"]
    assert len(did_entries) == 1
    assert did_entries[0].source == "user"
    assert did_entries[0].name == "My Own DiD Notes"


# ---------------------------------------------------------------------------
# render_skills_index
# ---------------------------------------------------------------------------

def test_render_skills_index_empty():
    text = skills.render_skills_index([])
    assert "No Sift Skills installed" in text


def test_render_skills_index_lists_slug_and_description():
    b = skills.list_builtin_skills()
    text = skills.render_skills_index(b)
    assert "`staggered_did`" in text
    assert "`survey_weighted_analysis`" in text


# ---------------------------------------------------------------------------
# get_skill_body lookup
# ---------------------------------------------------------------------------

def test_get_skill_body_found_and_not_found():
    b = skills.list_builtin_skills()
    assert skills.get_skill_body(b, "staggered_did") is not None
    assert skills.get_skill_body(b, "does_not_exist") is None


# ---------------------------------------------------------------------------
# get_skill tool integration
# ---------------------------------------------------------------------------

def _call_get_skill(args: dict) -> dict:
    from sift.tools import get_skill
    envelope = asyncio.run(get_skill.handler(args))
    return json.loads(envelope["content"][0]["text"])


def test_get_skill_tool_returns_builtin_skill(tmp_path: Path):
    from sift.config import set_cwd
    set_cwd(tmp_path)
    body = _call_get_skill({"slug": "staggered_did"})
    assert body["status"] == "ok"
    assert body["slug"] == "staggered_did"
    assert body["source"] == "builtin"
    assert "Callaway" in body["body"]


def test_get_skill_tool_returns_user_skill(tmp_path: Path):
    from sift.config import set_cwd
    set_cwd(tmp_path)
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "custom.md").write_text(
        "---\nname: Custom\ndescription: a custom skill\n---\ncustom body text\n",
        encoding="utf-8",
    )
    body = _call_get_skill({"slug": "custom"})
    assert body["status"] == "ok"
    assert body["source"] == "user"
    assert "custom body text" in body["body"]


def test_get_skill_tool_not_found_lists_available(tmp_path: Path):
    from sift.config import set_cwd
    set_cwd(tmp_path)
    body = _call_get_skill({"slug": "totally_made_up_slug"})
    assert body["status"] == "not_found"
    assert "staggered_did" in body["reason"]


def test_get_skill_tool_rejects_empty_slug(tmp_path: Path):
    from sift.config import set_cwd
    set_cwd(tmp_path)
    body = _call_get_skill({"slug": ""})
    assert body["status"] == "error"


def test_get_skill_tool_rejects_missing_slug(tmp_path: Path):
    from sift.config import set_cwd
    set_cwd(tmp_path)
    body = _call_get_skill({})
    assert body["status"] == "error"


# ---------------------------------------------------------------------------
# system_prompt.py wiring
# ---------------------------------------------------------------------------

def test_system_prompt_includes_skills_index(tmp_path: Path):
    from sift.system_prompt import build_system_prompt
    prompt = build_system_prompt(tmp_path, "sift")
    assert "Sift Skills available this session" in prompt
    assert "staggered_did" in prompt
    assert "survey_weighted_analysis" in prompt


def test_system_prompt_includes_user_skill_too(tmp_path: Path):
    from sift.system_prompt import build_system_prompt
    d = tmp_path / ".sift" / "skills"
    d.mkdir(parents=True)
    (d / "custom.md").write_text(
        "---\nname: Custom\ndescription: shows up in the prompt\n---\nbody\n",
        encoding="utf-8",
    )
    prompt = build_system_prompt(tmp_path, "sift")
    assert "custom" in prompt
    assert "shows up in the prompt" in prompt


def test_get_skill_is_a_registered_tool():
    from sift.tools import ALLOWED_TOOL_NAMES
    assert any(name.endswith("__get_skill") for name in ALLOWED_TOOL_NAMES)
