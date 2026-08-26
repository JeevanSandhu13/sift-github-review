"""Curated, inert research guidance for Sift.

A Sift skill is text: methodological judgment, workflow guidance,
decision checklists, and examples that use existing Sift tools. Skills do not
register executable code, add tools, read datasets, or change the privacy
boundary. The selected model reads guidance and continues to use the same
sanitized execution path.

Two sources, both markdown-with-frontmatter files in this exact
shape:

    ---
    name: staggered-did
    description: One line, shown in the always-visible skills index.
    when_to_use: One line, a triggering heuristic for the model.
    ---
    Full guidance body (markdown). Loaded on demand via the
    ``get_skill`` tool, not injected into every turn.

- **Builtin** (``skills_library/*.md``, shipped with the package):
  maintained and reviewed with the application.
- **User** (``<cwd>/.sift/skills/*.md``): a researcher or an admin
  deploying Sift to a lab can drop in their own. Same format, same
  parsing, same sanitization — no special trust given to either
  source. A user skill with the same slug replaces the builtin guidance.

Every model-visible string passes through the text-safety layer with an
explicit size cap. This bounds both the always-visible index and individual
skill responses.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sift.text_safety import safe_key, safe_multiline_text, safe_text

# Caps allow substantive guidance while bounding the prompt-injection and
# context-growth surface. Bodies are loaded on demand but remain bounded.
_NAME_MAX_LEN = 80
_DESCRIPTION_MAX_LEN = 300
_WHEN_TO_USE_MAX_LEN = 300
_BODY_MAX_LEN = 12_000
_MAX_SKILLS_PER_SOURCE = 100  # a runaway skills/ directory degrades, not hangs

SKILLS_LIBRARY_DIR = Path(__file__).parent / "skills_library"
USER_SKILLS_DIR = Path(".sift") / "skills"


@dataclass(frozen=True)
class Skill:
    """A single parsed, sanitized skill."""
    slug: str
    name: str
    description: str
    when_to_use: str
    body: str
    source: str  # "builtin" or "user"
    path: str


def validate_skill(skill: Skill) -> tuple[str, ...]:
    """Return structural errors for a reusable guidance package.

    Skills remain inert text, but a valid one must still be actionable and
    bounded: identity, trigger, description, substantive body, and at least one
    explicit validation/check section. This prevents vague domain prompts from
    being presented as reviewed methodological guidance.
    """
    errors: list[str] = []
    if not skill.slug or not skill.name or not skill.description:
        errors.append("missing identity metadata")
    if not skill.when_to_use:
        errors.append("missing when_to_use trigger")
    if len(skill.body.strip()) < 200:
        errors.append("guidance body is too short")
    body = skill.body.casefold()
    if not any(token in body for token in (
        "diagnostic", "validation", "check", "assumption", "failure mode",
    )):
        errors.append("guidance has no validation, diagnostic, or assumption section")
    return tuple(errors)


def _split_frontmatter(text: str) -> tuple[dict[str, Any] | None, str]:
    """Split a ``---\\nYAML\\n---\\nbody`` file. Returns (frontmatter
    dict or None if malformed/absent, body text).
    """
    if not text.startswith("---"):
        return None, text
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None, text
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return None, text
    fm_text = "\n".join(lines[1:end_idx])
    body = "\n".join(lines[end_idx + 1:]).lstrip("\n")
    try:
        import yaml
        data = yaml.safe_load(fm_text)
    except Exception:  # noqa: BLE001 — malformed frontmatter, not a crash
        return None, body
    if not isinstance(data, dict):
        return None, body
    return data, body


def _parse_skill_file(path: Path, source: str) -> Skill | None:
    """Parse one skill file. Returns ``None`` (skip, never raise) for
    anything malformed — a broken skill file degrades to "not
    offered", the same posture every other researcher-editable
    config file in this project takes for corruption.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    frontmatter, body = _split_frontmatter(text)
    if frontmatter is None:
        return None
    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(name, str) or not name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None
    when_to_use = frontmatter.get("when_to_use")
    if not isinstance(when_to_use, str):
        when_to_use = ""

    slug = safe_key(path.stem)
    skill = Skill(
        slug=slug,
        name=safe_text(name, max_len=_NAME_MAX_LEN),
        description=safe_text(description, max_len=_DESCRIPTION_MAX_LEN),
        when_to_use=safe_text(when_to_use, max_len=_WHEN_TO_USE_MAX_LEN),
        # The body is genuinely multi-paragraph markdown -- use the
        # newline-preserving sanitizer, not the label-oriented one
        # (which would flatten every paragraph break to a space).
        body=safe_multiline_text(body, max_len=_BODY_MAX_LEN),
        source=source,
        path=str(path),
    )
    # Loading remains backwards-compatible and tolerant: validation is an
    # explicit review surface, not a reason to make an older/user-authored
    # guidance file disappear. Builtins are asserted valid in tests/release
    # qualification; the UI can display validation errors for local skills.
    return skill


def _load_dir(dir_path: Path, source: str) -> list[Skill]:
    if not dir_path.is_dir():
        return []
    try:
        files = sorted(dir_path.glob("*.md"))
    except OSError:
        return []
    out: list[Skill] = []
    for f in files[:_MAX_SKILLS_PER_SOURCE]:
        skill = _parse_skill_file(f, source)
        if skill is not None:
            out.append(skill)
    return out


def list_builtin_skills() -> list[Skill]:
    """Skills shipped with the Sift package itself."""
    return _load_dir(SKILLS_LIBRARY_DIR, "builtin")


def list_user_skills(cwd: Path) -> list[Skill]:
    """Skills a researcher or admin dropped into this session's
    ``.sift/skills/`` directory.
    """
    return _load_dir(Path(cwd) / USER_SKILLS_DIR, "user")


def load_all_skills(cwd: Path) -> list[Skill]:
    """Builtin skills plus user skills, deduplicated by slug.

    A user skill REPLACES a builtin one with the same slug — a
    researcher's own domain judgment outranks Sift's shipped default
    for the same topic. Order: builtin skills first, then user
    skills, so a listing that presents "in order" shows Sift's own
    library before session-local additions, matching how
    ``dataset_listing`` and other enumerations in this codebase order
    "ours" before "researcher's own".
    """
    by_slug: dict[str, Skill] = {}
    for skill in list_builtin_skills():
        by_slug[skill.slug] = skill
    for skill in list_user_skills(cwd):
        by_slug[skill.slug] = skill
    # Preserve builtin-first, then user-only ordering rather than
    # dict insertion order (which would put a user OVERRIDE of a
    # builtin slug in the builtin position but a NEW user slug at
    # the end — both true statements, but let's be explicit and not
    # rely on dict semantics an unfamiliar reader would have to
    # verify).
    builtin_slugs = [s.slug for s in list_builtin_skills()]
    ordered: list[Skill] = []
    seen: set[str] = set()
    for slug in builtin_slugs:
        ordered.append(by_slug[slug])
        seen.add(slug)
    for skill in list_user_skills(cwd):
        if skill.slug not in seen:
            ordered.append(by_slug[skill.slug])
            seen.add(skill.slug)
    return ordered


def render_skills_index(skills: list[Skill]) -> str:
    """A short, always-visible index: one line per skill. Full
    guidance is fetched on demand via the ``get_skill`` tool, not
    inlined here — same "cheap index, load full content on demand"
    posture as everything else in this project that can grow
    unboundedly (results, dataset schemas).
    """
    if not skills:
        return (
            "No Sift Skills installed for this session. Skills are "
            "optional guidance packages (judgment/workflow help on "
            "top of the tools you already have) — none are required "
            "for any analysis."
        )
    lines = []
    for s in skills:
        trigger = f" — use when: {s.when_to_use}" if s.when_to_use else ""
        lines.append(f"  - `{s.slug}` ({s.source}): {s.description}{trigger}")
    return "\n".join(lines)


def get_skill_body(skills: list[Skill], slug: str) -> Skill | None:
    """Look up a skill by slug (already-sanitized comparison)."""
    target = safe_key(slug)
    for s in skills:
        if s.slug == target:
            return s
    return None


__all__ = [
    "Skill", "get_skill_body", "list_builtin_skills", "list_user_skills",
    "load_all_skills", "render_skills_index", "validate_skill",
]
