from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_python_310_toml_backport_is_declared_and_used() -> None:
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"tomli>=2.0; python_version < \'3.11\'"' in project
    for relative in (
        "src/sift/security_assurance.py",
        "packaging/finalize_release.py",
        "packaging/build_linux.sh",
        "packaging/windows_build_probe.py",
        "packaging/release.sh",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "import tomli as tomllib" in source, relative

    windows_build = (ROOT / "packaging" / "build_windows.ps1").read_text(
        encoding="utf-8"
    )
    assert "packaging/windows_build_probe.py project-version" in windows_build


def test_macos_build_uses_the_uv_managed_project_python() -> None:
    source = (ROOT / "packaging" / "build_app.sh").read_text(encoding="utf-8")
    assert 'APP_VERSION="$("$UV_BIN" run python -c' in source
    assert "/usr/bin/python3" not in source
    assert "import tomllib" in source
