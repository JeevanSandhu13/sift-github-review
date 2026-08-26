from __future__ import annotations

import tomllib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    "docs",
    "packaging",
    "scripts",
    "siftbench",
    "src",
    "tests",
)


def test_release_sources_contain_no_macos_metadata_files() -> None:
    """AppleDouble/resource-fork files must never become Windows inputs.

    Some cross-filesystem copy tools materialize macOS extended attributes as
    ``._*`` files. Those files can be mistaken for runtime helpers and can
    also leak workstation metadata into archives or installers.
    """
    offenders = sorted(
        path.relative_to(PROJECT_ROOT).as_posix()
        for root_name in SOURCE_ROOTS
        for path in (PROJECT_ROOT / root_name).rglob("*")
        if path.is_file()
        and (path.name.startswith("._") or path.name == ".DS_Store")
    )
    assert offenders == [], f"macOS metadata files in release sources: {offenders}"


def test_build_configuration_excludes_workstation_state() -> None:
    configuration = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    excluded = set(configuration["tool"]["hatch"]["build"]["exclude"])
    assert {
        "/.coverage",
        "/.hypothesis",
        "/.mypy_cache",
        "/.pytest_cache",
        "/.qualification-r-library",
        "/.r-libs",
        "/.uv-cache",
        "/.venv",
        "/.appenv",
        "/.uv-python",
        "/.sift",
        "/build",
        "/dist",
        "/packaging/vendor/python",
        "/**/.DS_Store",
        "/**/__pycache__",
        "/**/*.pyc",
        "/**/*.pyo",
    } <= excluded
