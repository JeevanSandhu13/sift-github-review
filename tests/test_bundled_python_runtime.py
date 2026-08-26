"""Bundled Python analysis runtime.

Covers ``env_detect._bundled_python_root`` / ``_bundled_python_binary``
/ ``find_bundled_python``, and ``find_python``'s fallback to the
bundled runtime when no usable PATH candidate exists.

None of this exercises a REAL vendored distribution. Tests here stand in the
currently-running interpreter at the platform-native vendor location
(``python.exe`` on Windows, ``bin/python3`` elsewhere) so path resolution,
layout validation, probe reuse, and fallback ordering remain portable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from sift import env_detect
from sift.env_detect import (
    _BUNDLED_PYTHON_ENV_VAR,
    _bundled_python_binary,
    _bundled_python_root,
    find_bundled_python,
    find_python,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep detection tests independent of the host's live sandbox.

    These tests exercise runtime discovery and fallback ordering; sandbox
    enforcement has its own integration suite. A nested CI/app sandbox can
    legitimately reject a second sandbox, which must not make a fake,
    otherwise-working interpreter look undiscoverable here.
    """
    monkeypatch.delenv(_BUNDLED_PYTHON_ENV_VAR, raising=False)
    env_detect._SANDBOX_PROBE_CACHE.clear()
    monkeypatch.setattr(
        env_detect, "_probe_sandbox_health", lambda _path: (True, ""),
    )


def _make_fake_vendor_root(tmp_path: Path, *, executable: bool = True) -> Path:
    """Build a throwaway directory shaped like ``vendor_python.py`` output.

    Symlinks the real interpreter running this test rather than
    faking one, so version/package probes exercise real subprocess
    behavior end to end."""
    root = tmp_path / "vendor_root"
    target = _expected_vendor_binary(root)
    target.parent.mkdir(parents=True)
    os.symlink(sys.executable, target)
    if not executable:
        target.chmod(0o644)
    return root


def _expected_vendor_binary(root: Path) -> Path:
    if sys.platform.startswith("win"):
        return root / "python.exe"
    return root / "bin" / "python3"


def _stub_working_interpreter_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep layout tests independent of relocatability of the host venv.

    A Windows venv launcher cannot be moved to a temporary vendor root and
    still locate its base runtime. The release build exercises the real
    relocated interpreter; these unit tests isolate discovery and ordering.
    """
    monkeypatch.setattr(
        env_detect, "_python_version", lambda _path: "Python 3.12.11",
    )
    monkeypatch.setattr(
        env_detect, "_python_missing_packages", lambda _path, _packages: (),
    )
    monkeypatch.setattr(env_detect, "_python_prefixes", lambda _path: ())
    monkeypatch.setattr(env_detect, "_binary_read_roots", lambda _path: ())


# ---------------------------------------------------------------------------
# _bundled_python_root
# ---------------------------------------------------------------------------


def test_root_is_none_with_no_env_var_and_not_frozen() -> None:
    assert _bundled_python_root() is None


def test_root_from_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    d = tmp_path / "custom_vendor"
    d.mkdir()
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(d))
    assert _bundled_python_root() == d


def test_root_env_var_pointing_at_nonexistent_dir_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(tmp_path / "does-not-exist"))
    assert _bundled_python_root() is None


def test_root_from_frozen_meipass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates the PyInstaller-frozen path: ``sys.frozen=True`` and
    ``sys._MEIPASS`` pointing at the bundle's extracted data root,
    with ``sift/vendor_python`` present underneath it."""
    meipass = tmp_path / "meipass"
    vendor = meipass / "sift" / "vendor_python"
    vendor.mkdir(parents=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    try:
        assert _bundled_python_root() == vendor
    finally:
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_root_frozen_but_no_vendor_dir_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    meipass = tmp_path / "meipass_empty"
    meipass.mkdir()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    try:
        assert _bundled_python_root() is None
    finally:
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)


def test_env_var_override_wins_over_frozen_meipass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Env var is checked first -- a packager/researcher override
    must win over whatever the frozen bundle itself carries."""
    env_dir = tmp_path / "env_override"
    env_dir.mkdir()
    meipass = tmp_path / "meipass"
    (meipass / "sift" / "vendor_python").mkdir(parents=True)
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(env_dir))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(meipass), raising=False)
    try:
        assert _bundled_python_root() == env_dir
    finally:
        monkeypatch.delattr(sys, "frozen", raising=False)
        monkeypatch.delattr(sys, "_MEIPASS", raising=False)


# ---------------------------------------------------------------------------
# _bundled_python_binary
# ---------------------------------------------------------------------------


def test_binary_found_at_expected_layout(tmp_path: Path) -> None:
    root = _make_fake_vendor_root(tmp_path)
    assert _bundled_python_binary(root) == str(_expected_vendor_binary(root))


def test_binary_none_when_layout_missing(tmp_path: Path) -> None:
    root = tmp_path / "empty_root"
    root.mkdir()
    assert _bundled_python_binary(root) is None


def test_binary_none_when_not_executable(tmp_path: Path) -> None:
    # A symlink's own mode bits are irrelevant on most platforms
    # (permissions come from the target, which we can't chmod here --
    # it's the real interpreter running this test); build a real
    # non-executable regular file instead to exercise the rejection
    # path reliably.
    f = _expected_vendor_binary(tmp_path / "root2")
    f.parent.mkdir(parents=True)
    f.write_text("not actually a binary")
    f.chmod(0o644)
    if sys.platform.startswith("win"):
        pytest.skip("Windows does not expose POSIX execute permission bits")
    assert _bundled_python_binary(tmp_path / "root2") is None


# ---------------------------------------------------------------------------
# find_bundled_python
# ---------------------------------------------------------------------------


def test_find_bundled_python_none_without_root(monkeypatch: pytest.MonkeyPatch) -> None:
    assert find_bundled_python() is None


def test_find_bundled_python_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_fake_vendor_root(tmp_path)
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(root))
    _stub_working_interpreter_probes(monkeypatch)
    tool = find_bundled_python()
    assert tool is not None
    assert tool.bundled is True
    # name stays "Python" -- NOT "Python (bundled)" -- so every
    # existing ``tool.name == "Python"`` call site in executor.py
    # (package-summary view, sys_prefix surfacing) keeps working
    # unchanged for a bundled Tool.
    assert tool.name == "Python"
    assert tool.binary == str(_expected_vendor_binary(root))
    assert tool.version is not None and tool.version.startswith("Python 3")


def test_find_bundled_python_none_when_layout_wrong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "wrong_layout"
    root.mkdir()
    (root / "python3").touch()  # not under bin/
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(root))
    assert find_bundled_python() is None


# ---------------------------------------------------------------------------
# find_python's fallback ordering
# ---------------------------------------------------------------------------


def test_find_python_prefers_path_candidate_over_bundled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A researcher's own PATH-discovered interpreter must win over
    the bundled fallback when both are usable -- the bundled runtime
    exists to give a researcher with NO setup a floor, not to
    override a deliberate existing environment."""
    root = _make_fake_vendor_root(tmp_path)
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(root))
    path_candidate = (
        r"C:\ResearcherPython\python.exe"
        if sys.platform.startswith("win")
        else "/researcher/bin/python3"
    )
    monkeypatch.setattr(
        env_detect.shutil,
        "which",
        lambda command: path_candidate if command == "python3" else None,
    )
    _stub_working_interpreter_probes(monkeypatch)
    tool = find_python()
    assert tool is not None
    assert tool.binary == path_candidate
    assert tool.bundled is False


def test_find_python_falls_back_to_bundled_when_path_has_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _make_fake_vendor_root(tmp_path)
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(root))
    monkeypatch.setattr(env_detect.shutil, "which", lambda cmd: None)
    _stub_working_interpreter_probes(monkeypatch)
    tool = env_detect.find_python()
    assert tool is not None
    assert tool.bundled is True
    assert tool.binary == str(_expected_vendor_binary(root))


def test_incomplete_path_python_does_not_shadow_complete_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial user Python is common and must not disable the complete
    scientific runtime shipped with Sift."""
    root = _make_fake_vendor_root(tmp_path)
    bundled_binary = str(_expected_vendor_binary(root))
    monkeypatch.setenv(_BUNDLED_PYTHON_ENV_VAR, str(root))
    monkeypatch.setattr(
        env_detect.shutil,
        "which",
        lambda cmd: "/partial/python3" if cmd == "python3" else None,
    )
    monkeypatch.setattr(
        env_detect,
        "_python_version",
        lambda path: "Python 3.12.11",
    )
    monkeypatch.setattr(
        env_detect,
        "_python_missing_packages",
        lambda path, packages: (
            ("pandas", "statsmodels") if path == "/partial/python3" else ()
        ),
    )
    monkeypatch.setattr(env_detect, "_python_prefixes", lambda _path: ())
    monkeypatch.setattr(env_detect, "_binary_read_roots", lambda _path: ())

    tool = env_detect.find_python()
    assert tool is not None
    assert tool.binary == bundled_binary
    assert tool.bundled is True


def test_find_python_returns_none_when_neither_path_nor_bundled_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_detect.shutil, "which", lambda cmd: None)
    assert env_detect.find_python() is None
