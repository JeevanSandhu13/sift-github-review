from __future__ import annotations

import tempfile
from pathlib import Path

from sift.ui import _materialize_cache_busted_index


def test_cache_bust_never_writes_inside_macos_app_bundle(
    tmp_path: Path, monkeypatch
) -> None:
    web_dir = (
        tmp_path
        / "Sift.app"
        / "Contents"
        / "Resources"
        / "src"
        / "sift"
        / "web"
    )
    web_dir.mkdir(parents=True)
    index = web_dir / "index.html"
    index.write_text(
        '<html><head><script src="app.js"></script></head></html>',
        encoding="utf-8",
    )
    (web_dir / "app.js").write_text("// test", encoding="utf-8")
    temp_root = tmp_path / "temp"
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_root))

    materialized = _materialize_cache_busted_index(web_dir, index)

    assert materialized.parent.parent == temp_root / "sift-cache-bust"
    assert not list(web_dir.glob(".index.bust-*.html"))
    assert (materialized.parent / "app.js").read_text(encoding="utf-8") == (
        "// test"
    )
    generated_html = materialized.read_text(encoding="utf-8")
    assert "<base " not in generated_html
    assert 'src="app.js?v=' in generated_html


def test_generated_cache_file_does_not_change_its_own_build_id(
    tmp_path: Path,
) -> None:
    web_dir = tmp_path / "web"
    web_dir.mkdir()
    index = web_dir / "index.html"
    index.write_text(
        '<html><head><link href="style.css" rel="stylesheet"></head></html>',
        encoding="utf-8",
    )
    (web_dir / "style.css").write_text("body {}", encoding="utf-8")

    first = _materialize_cache_busted_index(web_dir, index)
    second = _materialize_cache_busted_index(web_dir, index)

    assert second == first
    assert list(web_dir.glob(".index.bust-*.html")) == [first]
