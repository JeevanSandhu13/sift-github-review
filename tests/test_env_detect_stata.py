"""``env_detect.find_stata()`` -- PATH lookup plus native fallback installs.

No live Stata is required or assumed here. Tests select the platform they
exercise explicitly, so a host never has to pretend another platform's
binary is runnable.

Audit pass 2 finding: before this fix, the ONLY fallback locations
checked (after PATH) were macOS ``.app`` bundle paths
(``_STATA_APP_LOCATIONS``) -- a licensed, working Stata install on
Windows or Linux that simply wasn't on PATH (Stata's own Linux
installer explicitly does NOT add itself to PATH; see
``env_detect.py``'s ``_STATA_LINUX_LOCATIONS`` comment) was silently
reported as "not found" with no fallback checked at all.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from sift import env_detect


@pytest.fixture(autouse=True)
def _no_stata_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the PATH-based lookup (``shutil.which``) to always miss,
    so every test here exercises the fallback-location logic instead
    of whatever may or may not be on the real test machine's PATH."""
    monkeypatch.setattr(env_detect.shutil, "which", lambda cmd: None)


def _make_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho stata\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def test_no_stata_anywhere_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: with PATH missing and none of the fallback
    locations present on disk (the overwhelming common case on a
    machine without Stata), ``find_stata`` must cleanly return
    ``None`` rather than raising or false-positiving."""
    assert env_detect.find_stata() is None


def test_macos_app_bundle_location_is_still_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Regression guard: the pre-existing macOS fallback must keep
    working exactly as before -- this fix only ADDS locations, it
    must not disturb the macOS list or its ordering."""
    fake = tmp_path / "Applications" / "Stata" / "StataMP.app" / "Contents" / "MacOS" / "stata-mp"
    _make_executable(fake)
    monkeypatch.setattr(
        env_detect, "_STATA_APP_LOCATIONS", (str(fake),),
    )
    monkeypatch.setattr(env_detect, "_STATA_WINDOWS_LOCATIONS", ())
    monkeypatch.setattr(env_detect, "_STATA_LINUX_LOCATIONS", ())
    monkeypatch.setattr(env_detect.sys, "platform", "darwin")

    tool = env_detect.find_stata()
    assert tool is not None
    assert tool.name == "Stata"
    assert tool.binary == str(fake)


def test_windows_program_files_location_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The gap this closes: a real Stata install under Windows'
    ``Program Files\\StataNN\\`` must be discoverable even though
    it's never on PATH by default."""
    fake = tmp_path / "Stata18" / "StataMP-64.exe"
    _make_executable(fake)
    monkeypatch.setattr(env_detect, "_STATA_APP_LOCATIONS", ())
    monkeypatch.setattr(
        env_detect, "_STATA_WINDOWS_LOCATIONS", (str(fake),),
    )
    monkeypatch.setattr(env_detect, "_STATA_LINUX_LOCATIONS", ())
    monkeypatch.setattr(env_detect.sys, "platform", "win32")

    tool = env_detect.find_stata()
    assert tool is not None
    assert tool.name == "Stata"
    assert tool.binary == str(fake)


def test_linux_usr_local_location_is_found(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The gap this closes: StataCorp's own Linux tarball installer
    unpacks to ``/usr/local/stataNN/`` and does NOT add itself to
    PATH -- a real, working, licensed install was previously
    misreported as "not found" with no fallback checked at all."""
    fake = tmp_path / "stata18" / "stata-mp"
    _make_executable(fake)
    monkeypatch.setattr(env_detect, "_STATA_APP_LOCATIONS", ())
    monkeypatch.setattr(env_detect, "_STATA_WINDOWS_LOCATIONS", ())
    monkeypatch.setattr(
        env_detect, "_STATA_LINUX_LOCATIONS", (str(fake),),
    )
    monkeypatch.setattr(env_detect.sys, "platform", "linux")

    tool = env_detect.find_stata()
    assert tool is not None
    assert tool.name == "Stata"
    assert tool.binary == str(fake)


def test_path_lookup_still_takes_priority_over_every_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A PATH-configured Stata (the researcher set this up
    themselves) must win over any fallback location, on every
    platform's list -- unchanged priority order."""
    monkeypatch.setattr(
        env_detect.shutil, "which",
        lambda cmd: "/usr/local/bin/stata-mp" if cmd == "stata-mp" else None,
    )
    fake = tmp_path / "stata18" / "stata-mp"
    _make_executable(fake)
    monkeypatch.setattr(env_detect, "_STATA_LINUX_LOCATIONS", (str(fake),))

    tool = env_detect.find_stata()
    assert tool is not None
    assert tool.binary == "/usr/local/bin/stata-mp"


def test_non_executable_file_at_a_fallback_location_is_not_matched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file that merely exists at a candidate path but isn't
    executable (e.g. a stray non-executable placeholder, or wrong
    permissions after a partial install) must not be reported as a
    working Stata binary."""
    fake = tmp_path / "stata18" / "stata-mp"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text("not actually executable")
    fake.chmod(0o644)  # explicitly non-executable
    monkeypatch.setattr(env_detect, "_STATA_APP_LOCATIONS", ())
    monkeypatch.setattr(env_detect, "_STATA_WINDOWS_LOCATIONS", ())
    monkeypatch.setattr(env_detect, "_STATA_LINUX_LOCATIONS", (str(fake),))
    monkeypatch.setattr(env_detect.sys, "platform", "linux")
    real_access = env_detect.os.access
    monkeypatch.setattr(
        env_detect.os,
        "access",
        lambda path, mode: False if Path(path) == fake else real_access(path, mode),
    )

    assert env_detect.find_stata() is None


def test_only_native_fallback_locations_are_considered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(env_detect, "_STATA_APP_LOCATIONS", ("mac",))
    monkeypatch.setattr(env_detect, "_STATA_WINDOWS_LOCATIONS", ("windows",))
    monkeypatch.setattr(env_detect, "_STATA_LINUX_LOCATIONS", ("linux",))

    assert env_detect._stata_fallback_locations("darwin") == ("mac",)
    assert env_detect._stata_fallback_locations("win32") == ("windows",)
    assert env_detect._stata_fallback_locations("linux") == ("linux",)
    assert env_detect._stata_fallback_locations("freebsd13") == ()


def test_windows_and_linux_location_tuples_are_well_formed() -> None:
    """Sanity check on the actual (non-monkeypatched) module-level
    tuples: non-empty, every entry references a plausible Stata
    executable name, and the version span covers several recent
    major releases rather than just one guess."""
    assert len(env_detect._STATA_WINDOWS_LOCATIONS) > 0
    assert len(env_detect._STATA_LINUX_LOCATIONS) > 0
    assert all(
        "Stata" in p and p.endswith(".exe")
        for p in env_detect._STATA_WINDOWS_LOCATIONS
    )
    assert all(
        "/stata" in p.lower() for p in env_detect._STATA_LINUX_LOCATIONS
    )
    # Several distinct major versions represented, not just one.
    versions_seen = {
        p.split("Stata")[1].split("\\")[0]
        for p in env_detect._STATA_WINDOWS_LOCATIONS
    }
    assert len(versions_seen) >= 4
