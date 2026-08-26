"""Generate and verify every native Sift application/installer image.

``icon-source.png`` is the sole artwork master.  Generated files are kept in
the source tree so native release builders never depend on an image editor or
on assets downloaded during a release.  ``--check`` is deliberately cheap and
is run by every platform build before PyInstaller starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
SOURCE = PACKAGING / "icon-source.png"
MANIFEST = PACKAGING / "brand-assets.json"

WINDOWS_SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
LINUX_SIZES = (16, 24, 32, 48, 64, 128, 256, 512)

# These are not a parallel installer palette.  They are the exact canonical
# tokens in web/desktop-shell.css, reused by every native brand surface.
UI_TOKENS = {
    "shell-bg": "#f4f5f2",
    "shell-panel-raised": "#ffffff",
    "shell-border": "#cfd5ce",
    "shell-text": "#19211d",
    "shell-text-secondary": "#536059",
    "shell-accent": "#276f50",
    "shell-code-bg": "#202723",
    "shell-code-text": "#e7eee9",
    "shell-dark-border": "#3b473f",
    "shell-dark-accent": "#79be96",
    "shell-dark-text-secondary": "#aeb9b2",
}


def _check_ui_tokens() -> None:
    css = (ROOT / "src" / "sift" / "web" / "desktop-shell.css").read_text(
        encoding="utf-8"
    )
    css_names = {
        "shell-bg": "shell-bg",
        "shell-panel-raised": "shell-panel-raised",
        "shell-border": "shell-border",
        "shell-text": "shell-text",
        "shell-text-secondary": "shell-text-secondary",
        "shell-accent": "shell-accent",
        "shell-code-bg": "shell-code-bg",
        "shell-code-text": "shell-code-text",
    }
    for token, css_name in css_names.items():
        match = re.search(rf"--{re.escape(css_name)}:\s*(#[0-9a-fA-F]{{6}})\s*;", css)
        if not match or match.group(1).lower() != UI_TOKENS[token]:
            raise SystemExit(
                f"native brand token {token} no longer matches desktop-shell.css"
            )
    # Dark-mode values occur later in the file; require their exact reviewed
    # forms as well, without mistaking the initial light declarations.
    for css_name, token in (
        ("shell-border", "shell-dark-border"),
        ("shell-accent", "shell-dark-accent"),
        ("shell-text-secondary", "shell-dark-text-secondary"),
    ):
        values = re.findall(
            rf"--{re.escape(css_name)}:\s*(#[0-9a-fA-F]{{6}})\s*;", css
        )
        if UI_TOKENS[token] not in {value.lower() for value in values}:
            raise SystemExit(
                f"native brand token {token} no longer matches desktop-shell.css"
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    # Pillow ships a deterministic fallback font.  Prefer common native fonts
    # only for the installer background's human-facing copy; font choice never
    # changes the logo itself.
    candidates = (
        [Path("/System/Library/Fonts/SFNS.ttf")]
        if platform.system() == "Darwin"
        else [
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    )
    if bold:
        candidates = (
            [Path("/System/Library/Fonts/SFNS.ttf")]
            if platform.system() == "Darwin"
            else [
                Path("C:/Windows/Fonts/segoeuib.ttf"),
                Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def _logo_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        (Path("/System/Library/Fonts/Menlo.ttc"), 1),
        (Path("C:/Windows/Fonts/consolab.ttf"), 0),
        (Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"), 0),
    ]
    for candidate, index in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size, index=index)
    raise SystemExit("a bold native monospace font is required to create the logo master")


def create_terminal_master() -> None:
    """Create the deliberately minimal, terminal-inspired ``>S`` master."""
    _check_ui_tokens()
    image = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (64, 64, 960, 960),
        radius=215,
        fill=UI_TOKENS["shell-code-bg"],
        outline=UI_TOKENS["shell-dark-border"],
        width=10,
    )
    prompt_font = _logo_font(280)
    letter_font = _logo_font(390)
    prompt_width = draw.textlength(">", font=prompt_font)
    letter_width = draw.textlength("S", font=letter_font)
    gap = 34
    x = (1024 - (prompt_width + gap + letter_width)) / 2
    baseline = 660
    draw.text(
        (x, baseline), ">", font=prompt_font,
        fill=UI_TOKENS["shell-dark-accent"], anchor="ls",
    )
    draw.text(
        (x + prompt_width + gap, baseline), "S", font=letter_font,
        fill=UI_TOKENS["shell-code-text"], anchor="ls",
    )
    image.save(SOURCE, format="PNG", optimize=True)
    print(f"Created terminal-inspired logo master: {SOURCE}")


def _open_master() -> Image.Image:
    _check_ui_tokens()
    if not SOURCE.is_file():
        raise SystemExit(f"missing canonical brand artwork: {SOURCE}")
    image = Image.open(SOURCE)
    if image.format != "PNG" or image.mode != "RGBA" or image.size != (1024, 1024):
        raise SystemExit("icon-source.png must be a 1024x1024 RGBA PNG")
    alpha = image.getchannel("A")
    bounds = alpha.getbbox()
    if bounds is None:
        raise SystemExit("icon-source.png is completely transparent")
    if any(alpha.getpixel(point) != 0 for point in ((0, 0), (1023, 0), (0, 1023), (1023, 1023))):
        raise SystemExit("icon-source.png corners must be transparent")
    # Native launchers mask icons differently.  A 6.25% safe area prevents
    # Windows tiles and Linux docks from clipping the mark.
    if bounds[0] < 64 or bounds[1] < 64 or bounds[2] > 961 or bounds[3] > 961:
        raise SystemExit(f"icon artwork exceeds the native safe area: {bounds}")
    return image.convert("RGBA")


def _render_png(master: Image.Image, size: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    master.resize((size, size), Image.Resampling.LANCZOS).save(
        destination, format="PNG", optimize=True
    )


def _generate_icns(master: Image.Image) -> None:
    # Pillow writes Apple's modern PNG-backed ICNS members (32 through 1024
    # px) and works identically on every release host.  This avoids an
    # iconutil regression on newer macOS versions that rejects otherwise valid
    # iconsets while still producing an ICNS Finder and LaunchServices read.
    variants = [
        master.resize((size, size), Image.Resampling.LANCZOS)
        for size in (32, 64, 128, 256, 512)
    ]
    master.save(PACKAGING / "Sift.icns", format="ICNS", append_images=variants)


def _generate_windows(master: Image.Image) -> None:
    windows = PACKAGING / "windows"
    windows.mkdir(parents=True, exist_ok=True)
    master.save(
        windows / "Sift.ico",
        format="ICO",
        sizes=[(size, size) for size in WINDOWS_SIZES],
        bitmap_format="png",
    )

    # Inno Setup's welcome/final panel and page header are bitmap-only.  Keep
    # these text-light and branded; all instructions remain native controls.
    wizard = Image.new("RGB", (164, 314), UI_TOKENS["shell-code-bg"])
    logo = master.resize((132, 132), Image.Resampling.LANCZOS)
    wizard.paste(logo, (16, 54), logo)
    draw = ImageDraw.Draw(wizard)
    draw.rounded_rectangle(
        (22, 280, 142, 284), radius=2, fill=UI_TOKENS["shell-dark-accent"]
    )
    wizard.save(windows / "installer-wizard.bmp", format="BMP")

    small = Image.new("RGB", (55, 55), "white")
    small_logo = master.resize((51, 51), Image.Resampling.LANCZOS)
    small.paste(small_logo, (2, 2), small_logo)
    small.save(windows / "installer-small.bmp", format="BMP")

    # Microsoft Store MSIX packages use PNG visual assets rather than ICO.
    # Keep the mark and safe area identical to the EXE, installer, macOS, and
    # Linux surfaces; Windows applies its own Start-menu/tile mask at runtime.
    msix_assets = windows / "msix" / "Assets"
    for filename, size in (
        ("StoreLogo.png", 50),
        ("Square44x44Logo.png", 44),
        ("Square150x150Logo.png", 150),
    ):
        _render_png(master, size, msix_assets / filename)


def _generate_linux(master: Image.Image) -> None:
    base = PACKAGING / "linux" / "icons" / "hicolor"
    for size in LINUX_SIZES:
        _render_png(
            master,
            size,
            base / f"{size}x{size}" / "apps" / "org.sapieninstitute.sift.png",
        )

    # pywebview's Qt backend uses this for the live window/task switcher icon;
    # the desktop entry alone only brands the launcher before the process runs.
    _render_png(master, 64, ROOT / "src" / "sift" / "web" / "app-icon.png")


def _generate_dmg_background(master: Image.Image) -> None:
    destination = PACKAGING / "macos" / "installer-background.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (660, 400), UI_TOKENS["shell-bg"])
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        (24, 24, 636, 376), radius=10,
        fill=UI_TOKENS["shell-panel-raised"],
        outline=UI_TOKENS["shell-border"], width=1,
    )
    # Finder draws the real Sift and Applications icons at either side.  Keep
    # those areas empty so the background never creates a doubled icon, and
    # keep the content light because Finder controls label text color.
    draw.text(
        (330, 64), "Install Sift", anchor="mm",
        fill=UI_TOKENS["shell-text"], font=_font(27, bold=True),
    )
    draw.line((48, 108, 612, 108), fill=UI_TOKENS["shell-border"], width=1)
    draw.line((278, 205, 382, 205), fill=UI_TOKENS["shell-accent"], width=6)
    draw.polygon(((382, 205), (362, 190), (362, 220)), fill=UI_TOKENS["shell-accent"])
    draw.text(
        (330, 347),
        "Drag Sift into Applications",
        anchor="mm",
        fill=UI_TOKENS["shell-text-secondary"],
        font=_font(16),
    )
    canvas.save(destination, format="PNG", optimize=True)


def _asset_paths() -> list[Path]:
    paths = [
        PACKAGING / "Sift.icns",
        PACKAGING / "windows" / "Sift.ico",
        PACKAGING / "windows" / "installer-wizard.bmp",
        PACKAGING / "windows" / "installer-small.bmp",
        PACKAGING / "windows" / "msix" / "Assets" / "StoreLogo.png",
        PACKAGING / "windows" / "msix" / "Assets" / "Square44x44Logo.png",
        PACKAGING / "windows" / "msix" / "Assets" / "Square150x150Logo.png",
        PACKAGING / "macos" / "installer-background.png",
        ROOT / "src" / "sift" / "web" / "app-icon.png",
    ]
    paths.extend(
        PACKAGING
        / "linux"
        / "icons"
        / "hicolor"
        / f"{size}x{size}"
        / "apps"
        / "org.sapieninstitute.sift.png"
        for size in LINUX_SIZES
    )
    return paths


def _write_manifest() -> None:
    document = {
        "format": "sift-brand-assets",
        "schema_version": 1,
        "canonical_source": SOURCE.relative_to(ROOT).as_posix(),
        "canonical_source_sha256": _sha256(SOURCE),
        "ui_tokens": UI_TOKENS,
        "assets": {
            path.relative_to(ROOT).as_posix(): {
                "sha256": _sha256(path),
                "size": path.stat().st_size,
            }
            for path in _asset_paths()
        },
    }
    MANIFEST.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def generate() -> None:
    master = _open_master()
    _generate_icns(master)
    _generate_windows(master)
    _generate_linux(master)
    _generate_dmg_background(master)
    _write_manifest()
    print(f"Generated {len(_asset_paths())} native brand assets from {SOURCE.name}")


def check() -> None:
    _open_master()
    try:
        document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"brand asset manifest is missing or invalid: {exc}") from exc
    if document.get("format") != "sift-brand-assets" or document.get("schema_version") != 1:
        raise SystemExit("brand asset manifest has an unsupported format")
    if document.get("canonical_source_sha256") != _sha256(SOURCE):
        raise SystemExit("brand artwork changed; run packaging/make_icons.sh")
    if document.get("ui_tokens") != UI_TOKENS:
        raise SystemExit("native brand tokens changed; run packaging/make_icons.sh")
    # Brand manifests are source artifacts shared by macOS, Windows, and
    # Linux. Always serialize POSIX-style relative keys; using ``str(Path)``
    # made a manifest generated on Unix fail every native Windows build even
    # when every file and digest was correct.
    expected = {path.relative_to(ROOT).as_posix() for path in _asset_paths()}
    recorded = document.get("assets")
    if not isinstance(recorded, dict) or set(recorded) != expected:
        raise SystemExit("brand asset inventory is incomplete; run packaging/make_icons.sh")
    for relative, metadata in recorded.items():
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"missing generated brand asset: {relative}")
        if path.stat().st_size != metadata.get("size") or _sha256(path) != metadata.get("sha256"):
            raise SystemExit(f"stale or modified brand asset: {relative}")

    with Image.open(PACKAGING / "windows" / "Sift.ico") as icon:
        sizes = set(icon.info.get("sizes", ()))
        required = {(size, size) for size in WINDOWS_SIZES}
        if not required.issubset(sizes):
            raise SystemExit(f"Windows icon is missing sizes: {sorted(required - sizes)}")
    for filename, size in (
        ("StoreLogo.png", 50),
        ("Square44x44Logo.png", 44),
        ("Square150x150Logo.png", 150),
    ):
        with Image.open(PACKAGING / "windows" / "msix" / "Assets" / filename) as icon:
            if icon.size != (size, size) or icon.mode != "RGBA":
                raise SystemExit(f"invalid Windows MSIX asset: {filename}")
    for size in LINUX_SIZES:
        path = (
            PACKAGING
            / "linux"
            / "icons"
            / "hicolor"
            / f"{size}x{size}"
            / "apps"
            / "org.sapieninstitute.sift.png"
        )
        with Image.open(path) as icon:
            if icon.size != (size, size) or icon.mode != "RGBA":
                raise SystemExit(f"invalid Linux icon: {path.relative_to(ROOT)}")
    print(f"Verified {len(expected)} native brand assets")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify committed assets only")
    parser.add_argument(
        "--create-master",
        action="store_true",
        help="replace icon-source.png with the reviewed >S terminal mark",
    )
    args = parser.parse_args()
    if args.check and args.create_master:
        parser.error("--check and --create-master are mutually exclusive")
    if args.create_master:
        create_terminal_master()
    check() if args.check else generate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
