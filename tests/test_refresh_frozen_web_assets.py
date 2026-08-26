from __future__ import annotations

import io
import importlib.util
from pathlib import Path
import tarfile
import zipfile

import pytest


SCRIPT = Path(__file__).parents[1] / "packaging" / "refresh_frozen_web_assets.py"
SPEC = importlib.util.spec_from_file_location("sift_refresh_frozen_web_assets", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
refresh = MODULE.refresh


ASSET_ROOT = "Sift/app/_internal/sift/web"


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    web = repo / "src" / "sift" / "web"
    web.mkdir(parents=True)
    (web / "app.js").write_bytes(b"new app")
    (web / "desktop-shell.css").write_bytes(b"new css")
    return repo


def test_refresh_tar_is_atomic_and_limited_to_external_web_assets(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    archive = tmp_path / "Sift-Linux-x86_64.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        for name, payload, mode in (
            (f"{ASSET_ROOT}/app.js", b"old app", 0o640),
            (f"{ASSET_ROOT}/desktop-shell.css", b"old css", 0o600),
            ("Sift/app/sift", b"executable", 0o755),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = mode
            info.mtime = 123
            output.addfile(info, io.BytesIO(payload))

    refresh(archive, repo)

    with tarfile.open(archive, "r:gz") as source:
        assert source.extractfile(f"{ASSET_ROOT}/app.js").read() == b"new app"
        assert source.extractfile(f"{ASSET_ROOT}/desktop-shell.css").read() == b"new css"
        executable = source.getmember("Sift/app/sift")
        assert source.extractfile(executable).read() == b"executable"
        assert executable.mode == 0o755
        assert executable.mtime == 123


def test_refresh_zip_preserves_other_members_and_attributes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    archive = tmp_path / "Sift-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"Sift/_internal/sift/web/app.js", b"old app")
        output.writestr(f"Sift/_internal/sift/web/desktop-shell.css", b"old css")
        executable = zipfile.ZipInfo("Sift/Sift.exe")
        executable.external_attr = 0o755 << 16
        executable.comment = b"keep"
        output.writestr(executable, b"executable")

    refresh(archive, repo)

    with zipfile.ZipFile(archive, "r") as source:
        assert source.read("Sift/_internal/sift/web/app.js") == b"new app"
        assert source.read("Sift/_internal/sift/web/desktop-shell.css") == b"new css"
        executable = source.getinfo("Sift/Sift.exe")
        assert source.read(executable) == b"executable"
        assert executable.external_attr == 0o755 << 16
        assert executable.comment == b"keep"


def test_refresh_rejects_an_incomplete_archive_without_replacing_it(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    archive = tmp_path / "Sift-Windows-x64.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("Sift/_internal/sift/web/app.js", b"old app")
    original = archive.read_bytes()

    with pytest.raises(RuntimeError, match="incomplete or ambiguous"):
        refresh(archive, repo)

    assert archive.read_bytes() == original
    assert not archive.with_name(f".{archive.name}.refreshing").exists()
