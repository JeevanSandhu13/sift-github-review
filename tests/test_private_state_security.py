from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from sift.config import PrivateStateError, ensure_private_sift_dir
from sift.windows_private_state import WindowsAclError, _private_dacl_sddl


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX permission-bit repair; Windows privacy is enforced by DACL",
)
def test_private_state_repairs_and_verifies_posix_mode(tmp_path: Path) -> None:
    state = tmp_path / ".sift"
    state.mkdir(mode=0o755)
    ensure_private_sift_dir(tmp_path)
    assert stat.S_IMODE(state.stat().st_mode) == 0o700


def test_private_state_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "redirected"
    target.mkdir()
    (tmp_path / ".sift").symlink_to(target, target_is_directory=True)
    with pytest.raises(PrivateStateError, match="symlink or junction"):
        ensure_private_sift_dir(tmp_path)


def test_windows_private_dacl_replaces_inheritance_and_other_users() -> None:
    sddl = _private_dacl_sddl("S-1-5-21-1-2-3-1001")
    assert sddl.startswith("D:P")
    assert ";;;SY)" in sddl
    assert ";;;S-1-5-21-1-2-3-1001)" in sddl
    assert "WD" not in sddl  # Everyone
    assert "AU" not in sddl  # Authenticated Users


def test_windows_private_dacl_rejects_untrusted_sid_text() -> None:
    with pytest.raises(WindowsAclError):
        _private_dacl_sddl("S-1-5-21);(A;;FA;;;WD")
