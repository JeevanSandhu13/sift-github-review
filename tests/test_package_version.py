"""Release metadata and the importable package must report one version."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import sift


def test_package_version_matches_installed_distribution() -> None:
    """Prevent wheels from exposing a version different from their metadata."""

    project_version = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    assert sift.__version__ == project_version

    try:
        distribution_version = version("sift")
    except PackageNotFoundError:
        return  # Source-only environments may not install distribution metadata.

    assert sift.__version__ == distribution_version
