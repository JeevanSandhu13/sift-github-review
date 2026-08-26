from __future__ import annotations

from pathlib import Path

import pytest

from sift.config import (
    WorkspaceScopeError,
    dangerous_workspace_reason,
    set_cwd,
    use_cwd,
    validate_workspace,
)


def test_backend_rejects_filesystem_root() -> None:
    root = Path(Path.cwd().anchor)
    assert dangerous_workspace_reason(root) is not None
    with pytest.raises(WorkspaceScopeError):
        validate_workspace(root)


def test_backend_rejects_home_directory() -> None:
    with pytest.raises(WorkspaceScopeError):
        validate_workspace(Path.home())


@pytest.mark.parametrize("name", ["Users", "Windows", "ProgramData", "etc"])
def test_backend_rejects_cross_platform_top_level_roots(name: str) -> None:
    anchor = Path(Path.cwd().anchor)
    assert dangerous_workspace_reason(anchor / name) is not None


def test_all_backend_cwd_entry_points_share_scope_policy() -> None:
    root = Path(Path.cwd().anchor)
    with pytest.raises(WorkspaceScopeError):
        set_cwd(root)
    with pytest.raises(WorkspaceScopeError):
        with use_cwd(root):
            pass


def test_specific_project_directory_is_accepted(tmp_path: Path) -> None:
    project = tmp_path / "research-project"
    project.mkdir()
    assert validate_workspace(project) == project.resolve()
