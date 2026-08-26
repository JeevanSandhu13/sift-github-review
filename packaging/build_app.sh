#!/usr/bin/env bash
#
# Build Sift.app from the PyInstaller output.
#
# The bundled binary is the pywebview UI shell (entry point:
# src/sift/__main__.py, which calls sift.ui:main). Double-click
# opens the chat window directly with no Terminal popup.
#
# Pipeline:
#   1. `uv run pyinstaller packaging/sift.spec --clean --noconfirm`
#      produces `dist/sift/` (one-dir bundle).
#   2. Assemble Sift.app manually (no osacompile) so we control
#      the launcher. Contents/MacOS/Sift is the shell script from
#      packaging/launcher.sh — it just execs the bundled binary;
#      pywebview opens its own native window.
#   3. Copy the PyInstaller bundle into
#      Sift.app/Contents/Resources/sift/.
#   4. Write an Info.plist marking it a normal GUI app (LSUIElement
#      false → dock icon visible, Cmd-Q works as expected).
#
# Gatekeeper / signing:
#   - If $SIFT_SIGN_IDENTITY is set (e.g. "Developer ID Application: ...
#     (TEAMID)"), every Mach-O inside the bundle is signed with the
#     hardened runtime, and the bundle itself is signed with the
#     entitlements in packaging/entitlements.plist. This is the
#     prerequisite for build_dmg.sh's notarization step.
#   - If unset, the .app is unsigned and researchers need the right-click
#     → Open workaround documented in docs/install.md.
#
# Usage (from repo root):
#   bash packaging/build_app.sh
#   SIFT_SIGN_IDENTITY="Developer ID Application: ..." bash packaging/build_app.sh
#
# Produces:
#   dist/Sift.app      — the macOS application bundle (web UI)
#   dist/sift/         — the raw PyInstaller output (kept for
#                            direct invocation and debugging)

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="$REPO_ROOT/dist"
PYINSTALLER_OUT="$DIST_DIR/sift"
APP_BUNDLE="$DIST_DIR/Sift.app"
STAGED_APP="$DIST_DIR/.Sift.app.staging.$$"

cleanup_staged_app() {
    if [[ -d "$STAGED_APP" ]]; then
        rm -rf -- "$STAGED_APP"
    fi
}
trap cleanup_staged_app EXIT INT TERM

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
elif [[ -x "$REPO_ROOT/.appenv/bin/uv" ]]; then
    # The local bootstrap environment is intentionally outside the release
    # artifacts, but it is a valid maintainer-side location for the build tool.
    UV_BIN="$REPO_ROOT/.appenv/bin/uv"
else
    echo "uv is required to build Sift.app." >&2
    exit 1
fi
# vendor_python.py intentionally discovers uv as an executable so the same
# code works when called directly.  When this wrapper selected the repo-local
# fallback above, that binary was not necessarily on PATH and vendoring failed
# even though every other build step could use it.  Export its exact directory
# for child processes as well.
export PATH="$(dirname -- "$UV_BIN"):${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"

RELEASE_MODE="${SIFT_RELEASE_MODE:-development}"
if [[ "$RELEASE_MODE" != "development" && "$RELEASE_MODE" != "production" ]]; then
    echo "SIFT_RELEASE_MODE must be development or production." >&2
    exit 1
fi
if [[ "$RELEASE_MODE" == "production" && "${SIFT_SKIP_VENDOR:-0}" == "1" ]]; then
    echo "Production builds cannot skip the bundled analysis runtime." >&2
    exit 1
fi
if [[ "$RELEASE_MODE" == "production" ]]; then
    PYTHONPATH=src "$UV_BIN" run python -c '
from sift.update_config import load_update_policy
p = load_update_policy()
raise SystemExit(0 if p.get("configured") is True else "production update policy is not configured")
'
fi

# Derive the bundle version from pyproject.toml so the .app's
# Info.plist (CFBundleVersion / CFBundleShortVersionString) can't
# drift from the package version. Previously the heredoc below
# hardcoded the same "0.0.1" string and a bump of pyproject's version
# without touching this script would ship a .app whose Apple-side
# version disagreed with what the dependency manifest claimed —
# silent skew that's invisible until someone diff's the two later.
# ``tomllib`` is stdlib in the project's pinned Python.  The build already
# requires uv above, so version parsing uses that exact interpreter and never
# falls back to an older, machine-dependent system Python.
APP_VERSION="$("$UV_BIN" run python -c '
import tomllib
with open("pyproject.toml", "rb") as f:
    print(tomllib.load(f)["project"]["version"])
')"
if [[ -z "$APP_VERSION" ]]; then
    echo "Could not read project.version from pyproject.toml" >&2
    exit 1
fi
echo "==> Bundle version: $APP_VERSION"

echo "==> Verifying native brand assets"
"$UV_BIN" run python packaging/generate_brand_assets.py --check

if [[ "${SIFT_SKIP_VENDOR:-0}" != "1" ]]; then
    echo "==> Building the bundled Python analysis runtime"
    "$UV_BIN" run python packaging/vendor_python.py
fi

echo "==> Installing the complete connector set"
"$UV_BIN" sync --locked --all-extras
"$UV_BIN" run python packaging/verify_database_drivers.py
PYTHONPATH=src "$UV_BIN" run python -m sift --platform-check
PYTHONPATH=src "$UV_BIN" run python -m sift --integration-check >/dev/null
PYTHONPATH=src "$UV_BIN" run python -m sift --format-check >/dev/null

echo "==> Running PyInstaller"
if [[ "${SIFT_SKIP_TESTS:-0}" != "1" ]]; then
    echo "==> Running macOS release tests"
    "$UV_BIN" run pytest -q
fi
"$UV_BIN" run pyinstaller packaging/sift.spec --clean --noconfirm >/dev/null

if [[ ! -x "$PYINSTALLER_OUT/sift" ]]; then
    echo "PyInstaller did not produce $PYINSTALLER_OUT/sift" >&2
    exit 1
fi
"$PYINSTALLER_OUT/sift" --platform-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$PYINSTALLER_OUT/sift" --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$PYINSTALLER_OUT/sift" --format-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$PYINSTALLER_OUT/sift" --analysis-check >/dev/null
# Self-tests must not mutate the artifact after its surface was accepted.
"$UV_BIN" run python packaging/verify_frozen_bundle.py "$PYINSTALLER_OUT"

echo "==> Assembling Sift.app"
rm -rf -- "$STAGED_APP"
mkdir -p "$STAGED_APP/Contents/MacOS"
mkdir -p "$STAGED_APP/Contents/Resources/sift"

# Launcher script — the .app's executable per Info.plist.
cp "$REPO_ROOT/packaging/launcher.sh" "$STAGED_APP/Contents/MacOS/Sift"
chmod +x "$STAGED_APP/Contents/MacOS/Sift"

# PyInstaller bundle — lives under Resources/sift/.
# Use ``ditto`` so extended attributes (most importantly the ones a
# subsequent ``codesign`` will rely on) survive the copy. ``cp -R``
# can drop xattrs in certain configurations; the breakage is
# silent — spctl still accepts the bundle but the signature is
# corrupt enough to fail at launch on a stricter machine.
/usr/bin/ditto "$PYINSTALLER_OUT/" "$STAGED_APP/Contents/Resources/sift/"

# App icon — `Sift.icns` lives at the bundle's Resources root and is
# referenced by CFBundleIconFile in Info.plist below. Finder, Dock,
# Cmd-Tab, and the About window all pick it up from there. Generate
# the .icns from packaging/icon-source.png with:
#   ./packaging/make_icons.sh  (regenerates every native icon)
ICON_SRC="$REPO_ROOT/packaging/Sift.icns"
if [[ ! -f "$ICON_SRC" ]]; then
    echo "Missing $ICON_SRC — run packaging/make_icns.sh to regenerate." >&2
    exit 1
fi
cp "$ICON_SRC" "$STAGED_APP/Contents/Resources/Sift.icns"

echo "==> Writing Info.plist"
# Heredoc delimiter is unquoted (``PLIST``, not ``'PLIST'``) so
# ``$APP_VERSION`` expands. The rest of the heredoc contains no shell
# metacharacters, so the unquoted form is safe.
cat > "$STAGED_APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Sift</string>
    <key>CFBundleDisplayName</key>
    <string>Sift</string>
    <key>CFBundleIdentifier</key>
    <string>org.sapieninstitute.sift</string>
    <key>CFBundleVersion</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleShortVersionString</key>
    <string>${APP_VERSION}</string>
    <key>CFBundleExecutable</key>
    <string>Sift</string>
    <key>CFBundleIconFile</key>
    <string>Sift.icns</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
    <!-- LSUIElement=false: standard GUI app — dock icon present,    -->
    <!-- shows up in Cmd-Tab, Cmd-Q quits cleanly. Earlier comment   -->
    <!-- here referenced spawning Terminal; the launcher no longer   -->
    <!-- does that, so the only window the user sees is pywebview's. -->
    <key>LSUIElement</key>
    <false/>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSApplicationCategoryType</key>
    <string>public.app-category.productivity</string>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2026 Sapien Institute</string>
</dict>
</plist>
PLIST

if [[ -n "${SIFT_SIGN_IDENTITY:-}" ]]; then
    ENTITLEMENTS="$REPO_ROOT/packaging/entitlements.plist"
    if [[ ! -f "$ENTITLEMENTS" ]]; then
        echo "Missing $ENTITLEMENTS — required when SIFT_SIGN_IDENTITY is set." >&2
        exit 1
    fi

    echo "==> Signing nested Mach-O binaries"
    # The .app's CFBundleExecutable is the shell launcher at
    # Contents/MacOS/Sift; it ``exec``s the nested PyInstaller
    # binary at Contents/Resources/sift/sift. The nested binary is
    # what becomes the running process — and exec does NOT propagate
    # entitlements from the parent. So the hardened-runtime
    # allowances (allow-unsigned-executable-memory,
    # disable-library-validation, allow-dyld-environment-variables)
    # have to be embedded into THAT binary's signature too, not just
    # the outer bundle's. Without this, codesign verification still
    # passes but the running image launches under hardened-runtime
    # without the exemptions and aborts on the first ctypes / cffi /
    # unsigned-dylib path PyInstaller exercises at startup.
    NESTED_MAIN="$STAGED_APP/Contents/Resources/sift/sift"

    # `find -depth` walks deepest-first so each binary is signed before
    # its enclosing bundle. We use `file` to skip shell scripts and other
    # non-Mach-O executables that would otherwise trip codesign.
    while IFS= read -r -d '' candidate; do
        if ! /usr/bin/file -b "$candidate" | grep -q "Mach-O"; then
            continue
        fi
        if [[ "$candidate" == "$NESTED_MAIN" ]]; then
            # Sign the running process image with entitlements so
            # hardened-runtime exemptions actually apply at launch.
            /usr/bin/codesign --force --options runtime --timestamp \
                --entitlements "$ENTITLEMENTS" \
                --sign "$SIFT_SIGN_IDENTITY" "$candidate"
        else
            # Dylibs / extension modules: hardened runtime + timestamp,
            # but no entitlements (they don't become processes).
            /usr/bin/codesign --force --options runtime --timestamp \
                --sign "$SIFT_SIGN_IDENTITY" "$candidate"
        fi
    done < <(/usr/bin/find "$STAGED_APP" -depth -type f \
                \( -name "*.dylib" -o -name "*.so" -o -perm -u+x \) -print0)

    echo "==> Signing app bundle"
    # Outer bundle still gets --entitlements so codesign metadata is
    # consistent at every level a verifier might inspect (the bundle,
    # the CFBundleExecutable, and the nested running binary).
    /usr/bin/codesign --force --options runtime --timestamp \
        --entitlements "$ENTITLEMENTS" \
        --sign "$SIFT_SIGN_IDENTITY" \
        "$STAGED_APP"

    echo "==> Verifying signature"
    /usr/bin/codesign --verify --deep --strict --verbose=2 "$STAGED_APP"
    # Confirm the entitlements actually landed on the nested running
    # binary — this is the guard against future refactors that move
    # the entitlements step around and silently drop them from the
    # process image. ``codesign -d --entitlements -`` prints the
    # embedded plist; we grep for one of the keys we expect to see.
    if ! /usr/bin/codesign -d --entitlements - "$NESTED_MAIN" 2>/dev/null \
            | grep -q "com.apple.security.cs.disable-library-validation"; then
        echo "ERROR: nested binary $NESTED_MAIN is missing hardened-runtime entitlements." >&2
        echo "       This means the running process won't have the exemptions and" >&2
        echo "       will abort at launch despite passing codesign --verify." >&2
        exit 1
    fi
    echo "Signed with: $SIFT_SIGN_IDENTITY"
else
    echo "==> Skipping codesign (SIFT_SIGN_IDENTITY unset)"
fi

# Keep a previously usable app recoverable until the replacement succeeds.
PREVIOUS_APP="$DIST_DIR/.Sift.app.previous.$$"
rm -rf -- "$PREVIOUS_APP"
if [[ -e "$APP_BUNDLE" ]]; then
    mv "$APP_BUNDLE" "$PREVIOUS_APP"
fi
if ! mv "$STAGED_APP" "$APP_BUNDLE"; then
    if [[ -e "$PREVIOUS_APP" ]]; then
        mv "$PREVIOUS_APP" "$APP_BUNDLE"
    fi
    echo "Could not install the completed Sift.app bundle." >&2
    exit 1
fi
rm -rf -- "$PREVIOUS_APP"
trap - EXIT INT TERM

echo
echo "Built: $APP_BUNDLE"
echo "Size:  $(du -sh "$APP_BUNDLE" | cut -f1)"
echo
echo "Next: bash packaging/build_dmg.sh to produce a distributable .dmg."
