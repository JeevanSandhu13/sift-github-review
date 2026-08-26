from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "packaging" / "windows_build_probe.py"


def _run(probe: str) -> str:
    completed = subprocess.run(
        [sys.executable, str(PROBE), probe],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def test_python_target_probe_is_machine_readable() -> None:
    payload = json.loads(_run("python-target"))
    assert payload["pointer_bits"] in {32, 64}
    assert isinstance(payload["platform"], str) and payload["platform"]


def test_project_version_probe_matches_project_metadata() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        expected = tomllib.load(handle)["project"]["version"]
    assert _run("project-version") == expected


def test_windows_build_uses_file_backed_probes() -> None:
    source = (ROOT / "packaging" / "build_windows.ps1").read_text(encoding="utf-8")
    assert "python -c" not in source
    assert source.count("packaging/windows_build_probe.py") == 3
