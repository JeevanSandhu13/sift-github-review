"""Packaging coverage for the bundled Python runtime.

``packaging/sift.spec`` runs inside PyInstaller and
``packaging/vendor_python.sh`` requires a supported target with network
access. Platform-neutral tests verify the spec's data-
collection logic for the vendored runtime behaves correctly as plain
Python when the vendor directory is absent (every dev build, and the
overwhelmingly common case even for CI), and both packaging artifacts
are structurally present and self-consistent.
"""

from __future__ import annotations

import ast
import platform
import re
import runpy
import shutil
import sys
from importlib.metadata import version as distribution_version
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "packaging" / "sift.spec"
VENDOR_SCRIPT_PATH = REPO_ROOT / "packaging" / "vendor_python.sh"
VENDOR_PY_PATH = REPO_ROOT / "packaging" / "vendor_python.py"


def _working_bash() -> str | None:
    """Return a real Bash executable, excluding Windows' WSL launcher stub."""
    executable = shutil.which("bash")
    if executable is None:
        return None
    import subprocess
    try:
        probe = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0 or "GNU bash" not in probe.stdout:
        return None
    return executable


WORKING_BASH = _working_bash()


def test_vendor_python_script_exists_and_is_executable():
    assert VENDOR_SCRIPT_PATH.is_file()
    import os
    assert os.access(VENDOR_SCRIPT_PATH, os.X_OK), (
        "packaging/vendor_python.sh must be chmod +x so a maintainer "
        "can run it directly"
    )


@pytest.mark.skipif(WORKING_BASH is None, reason="a working GNU Bash is not installed")
def test_vendor_python_script_has_valid_bash_syntax():
    import subprocess
    result = subprocess.run(
        [WORKING_BASH or "bash", "-n", str(VENDOR_SCRIPT_PATH)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    WORKING_BASH is None
    or (sys.platform == "darwin" and platform.machine() == "arm64"),
    reason=(
        "requires bash on an unsupported host; macOS arm64 is the supported "
        "vendoring host"
    ),
)
def test_vendor_python_script_refuses_non_macos_arm64():
    """The script's own platform guard must fire before it attempts
    any network fetch -- verified by actually running it in this
    genuinely unsupported host and confirming it exits non-zero with a clear
    message rather than silently no-op'ing or crashing deep inside a
    ``uv python install`` invocation."""
    import subprocess
    result = subprocess.run(
        [WORKING_BASH or "bash", str(VENDOR_SCRIPT_PATH)],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert "arm64" in result.stderr or "arm64" in result.stdout


def test_sift_spec_references_vendor_python_datas():
    """Structural check: the spec must define VENDOR_PYTHON_DIR /
    VENDOR_PYTHON_DATAS and fold the latter into the Analysis(...)
    datas list. Doesn't execute the spec (that requires PyInstaller's
    own exec environment with SPECPATH injected) -- just confirms the
    wiring exists in source, catching the class of regression where
    someone edits ``datas=`` and drops the addition."""
    text = SPEC_PATH.read_text(encoding="utf-8")
    assert "VENDOR_PYTHON_DIR" in text
    assert "VENDOR_PYTHON_DATAS" in text
    assert "packaging" in text and "vendor" in text and "python" in text
    # The actual Analysis() datas= line must include it.
    import re
    m = re.search(r"datas\s*=\s*([^\n]+)", text)
    assert m is not None, "could not find datas= in Analysis(...)"
    assert "VENDOR_PYTHON_DATAS" in m.group(1)


def test_spreadsheet_engines_are_bundled_and_collected():
    """All spreadsheet formats promised by the app must ship in releases."""
    python_builder = VENDOR_PY_PATH.read_text(encoding="utf-8")
    shell_builder = VENDOR_SCRIPT_PATH.read_text(encoding="utf-8")
    spec = SPEC_PATH.read_text(encoding="utf-8")
    for distribution in ("openpyxl", "xlrd", "odfpy"):
        assert distribution in python_builder
        assert distribution in shell_builder
    for module in ("openpyxl", "xlrd", "odf"):
        assert f'runtime_submodules("{module}")' in spec


def test_frozen_runtime_excludes_upstream_tests_and_benchmarks(
    tmp_path: Path,
) -> None:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    assert "def is_vendored_runtime_file(path, root)" in spec
    assert '"test", "tests", "testing", "benchmark", "benchmarks"' in spec
    assert 'part.startswith("test_")' not in spec
    assert 'filename.startswith("test_")' not in spec
    assert "is_vendored_runtime_file(p, VENDOR_PYTHON_DIR)" in spec
    assert "def is_runtime_bundle_destination(destination)" in spec
    assert "if is_runtime_bundle_destination(entry[0])" in spec

    # Every broad dynamic-import collector must use the same runtime-only
    # boundary. Platform framework discovery is the sole exception because
    # those namespaces are generated by PyObjC rather than upstream QA trees.
    hidden = spec.split("HIDDEN_IMPORTS = [", 1)[1].split("block_cipher", 1)[0]
    outside_framework_guard = hidden.split(
        "# PyObjC framework packages", 1,
    )[0] + hidden.split("if sys.platform == \"darwin\" else []", 1)[1]
    assert "*collect_submodules(" not in outside_framework_guard

    tree = ast.parse(spec, filename=str(SPEC_PATH))
    definition = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "is_vendored_runtime_file"
    )
    namespace: dict[str, object] = {}
    ast.fix_missing_locations(definition)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(SPEC_PATH), "exec"), namespace)
    keep = namespace["is_vendored_runtime_file"]
    root = tmp_path / "runtime"
    assert keep(root / "lib/site-packages/pandas/core/frame.py", root)
    assert not keep(root / "lib/site-packages/pandas/tests/test_frame.py", root)
    # Runtime packages can legitimately use this prefix (PyMC does); only
    # unambiguous QA directory boundaries may be stripped.
    assert keep(root / "lib/site-packages/pymc/variational/test_functions.py", root)
    assert keep(root / "lib/site-packages/dbfread/test_parser.py", root)
    assert not keep(root / "lib/site-packages/pkg/benchmarks/speed.py", root)
    assert not keep(root / "lib/site-packages/pkg/__pycache__/core.pyc", root)

    bundle_definition = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "is_runtime_bundle_destination"
    )
    bundle_namespace = {"Path": Path}
    ast.fix_missing_locations(bundle_definition)
    exec(
        compile(
            ast.Module(body=[bundle_definition], type_ignores=[]),
            str(SPEC_PATH),
            "exec",
        ),
        bundle_namespace,
    )
    keep_destination = bundle_namespace["is_runtime_bundle_destination"]
    assert keep_destination("pymc/variational/test_functions.py")
    assert not keep_destination("astropy/io/tests/data.fits")
    assert not keep_destination("jsonschema/benchmarks/cases.json")


def test_all_maintained_python_method_engines_are_vendored() -> None:
    python_builder = VENDOR_PY_PATH.read_text(encoding="utf-8")
    shell_builder = VENDOR_SCRIPT_PATH.read_text(encoding="utf-8")
    for distribution in (
        "scikit-learn", "factor-analyzer", "pyfixest", "rdrobust",
        "differences", "geopandas", "arviz", "pymc",
    ):
        assert distribution in python_builder
        assert distribution in shell_builder


def test_uv_managed_runtime_is_intentionally_populated() -> None:
    """Modern uv standalone Pythons are PEP 668 externally managed."""
    assert "--break-system-packages" in VENDOR_PY_PATH.read_text(encoding="utf-8")
    assert "--break-system-packages" in VENDOR_SCRIPT_PATH.read_text(encoding="utf-8")


def test_windows_vendor_target_follows_x64_interpreter_not_arm_host() -> None:
    builder = runpy.run_path(str(VENDOR_PY_PATH))
    python_id = builder["_uv_python_id"](
        system_platform="win32",
        machine="ARM64",
        python_platform="win-amd64",
    )
    assert python_id == "cpython-3.12.11-windows-x86_64-none"
    with pytest.raises(SystemExit, match="require an x64 Python target"):
        builder["_uv_python_id"](
            system_platform="win32",
            machine="ARM64",
            python_platform="win-arm64",
        )


def test_windows_vendor_prefers_distribution_interpreter_over_scripts_launcher(
    tmp_path: Path,
) -> None:
    builder = runpy.run_path(str(VENDOR_PY_PATH))
    distribution = tmp_path / "cpython-3.12.11-windows-x86_64-none"
    scripts = distribution / "Scripts"
    scripts.mkdir(parents=True)
    top_level = distribution / "python.exe"
    nested = scripts / "python.exe"
    top_level.touch()
    nested.touch()

    interpreter = builder["_find_interpreter"](
        tmp_path,
        system_platform="win32",
    )
    assert interpreter == top_level
    assert builder["_distribution_root"](interpreter) == distribution
    # Even if a future runtime only exposes a Scripts launcher, copying must
    # still start from the complete distribution rather than Scripts itself.
    assert builder["_distribution_root"](nested) == distribution


def test_windows_vendor_uses_reviewed_polars_compatibility_runtime() -> None:
    builder = runpy.run_path(str(VENDOR_PY_PATH))
    assert "polars==1.43.2" in builder["ANALYSIS_PACKAGES"]
    assert "polars-runtime-compat==1.43.2" in builder["ANALYSIS_PACKAGES"]
    assert "polars" in builder["ANALYSIS_IMPORTS"]
    source = VENDOR_PY_PATH.read_text(encoding="utf-8")
    assert 'verification_env["POLARS_SKIP_CPU_CHECK"] = "1"' in source


def test_vendored_import_manifest_matches_runtime_release_gate() -> None:
    from sift.env_detect import _BUNDLED_ANALYSIS_PACKAGES

    builder = runpy.run_path(str(VENDOR_PY_PATH))
    assert set(builder["ANALYSIS_IMPORTS"]) == set(_BUNDLED_ANALYSIS_PACKAGES)


def test_vendored_analysis_versions_are_exact_and_present_in_uv_lock() -> None:
    """The bundled numerical stack must match exact versions in ``uv.lock``."""
    builder = runpy.run_path(str(VENDOR_PY_PATH))
    requirements = builder["ANALYSIS_PACKAGES"]
    lock = (REPO_ROOT / "uv.lock").read_text(encoding="utf-8")

    assert requirements
    for requirement in requirements:
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^=<>!~]+)", requirement)
        assert match is not None, f"analysis dependency is not exact: {requirement}"
        name, version = match.groups()
        expected = f'[[package]]\nname = "{name}"\nversion = "{version}"'
        assert expected in lock, f"{requirement} is absent from uv.lock"


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 12),
    reason="release analysis runtime is pinned to Python 3.12",
)
def test_vendored_versions_match_qualified_python_312_environment() -> None:
    """The release runtime and the environment running real-fit tests agree."""
    requirements = runpy.run_path(str(VENDOR_PY_PATH))["ANALYSIS_PACKAGES"]
    for requirement in requirements:
        name, expected = requirement.split("==", 1)
        assert distribution_version(name) == expected, requirement


def test_every_os_release_build_vendors_the_analysis_runtime():
    """A clean release must not silently omit the local analysis runtime."""
    for relative in (
        "packaging/build_app.sh",
        "packaging/build_linux.sh",
        "packaging/build_windows.ps1",
    ):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "packaging/vendor_python.py" in text, relative


def test_sift_spec_vendor_datas_is_empty_list_when_dir_absent(tmp_path: Path):
    """Executes just the vendor-python data-collection block from the
    spec (extracted as plain Python, not through PyInstaller) against
    a REPO_ROOT that has no packaging/vendor/python -- the normal dev-
    build case. Must produce an empty list, not raise."""
    ns: dict = {}
    # REPO_ROOT stands in for what the real spec computes from
    # SPECPATH; point it at a throwaway directory with no vendored
    # runtime so the "absent" branch is what actually executes.
    ns["REPO_ROOT"] = tmp_path
    ns["Path"] = Path
    (tmp_path / "packaging").mkdir()

    text = SPEC_PATH.read_text(encoding="utf-8")
    # Extract the VENDOR_PYTHON_DIR / VENDOR_PYTHON_DATAS block
    # verbatim so this test breaks (loudly) if the block's shape
    # changes, rather than duplicating separately-maintained logic
    # that could silently drift from the real spec.
    start = text.index("VENDOR_PYTHON_DIR = REPO_ROOT")
    end = text.index("HIDDEN_IMPORTS = [")
    block = text[start:end]

    exec(compile(block, str(SPEC_PATH), "exec"), ns)
    assert ns["VENDOR_PYTHON_DATAS"] == []
