#!/usr/bin/env bash
# Build a small, launchable macOS development app around the source tree.
#
# This intentionally does not create a distributable release bundle. It reuses
# the project-local .appenv runtime so UI work can be launched from Finder while
# release signing, notarization, and the self-contained PyInstaller build remain
# separate concerns.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BUNDLE="$REPO_ROOT/dist/Sift.app"
STAGING_BUNDLE="$REPO_ROOT/dist/.Sift.app.staging"
PYTHON="$REPO_ROOT/.appenv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Sift development runtime: $PYTHON" >&2
    exit 1
fi

# Even APFS clones need metadata space for thousands of directory entries.
# Fail before touching either bundle when the volume is too full to stage a
# coherent app. Override exists for controlled CI fixtures only.
MIN_FREE_KIB="${SIFT_DEV_APP_MIN_FREE_KIB:-524288}"
FREE_KIB="$(df -Pk "$REPO_ROOT" | awk 'NR == 2 {print $4}')"
if [[ ! "$FREE_KIB" =~ ^[0-9]+$ ]] || (( FREE_KIB < MIN_FREE_KIB )); then
    echo "Sift development app build needs at least $((MIN_FREE_KIB / 1024)) MB free; only $((FREE_KIB / 1024)) MB is available." >&2
    exit 1
fi

if ! PYTHONPATH="$REPO_ROOT/src" "$PYTHON" -c "import sift.ui, webview"; then
    echo "The Sift development runtime is incomplete." >&2
    exit 1
fi

"$PYTHON" "$REPO_ROOT/packaging/generate_brand_assets.py" --check

APP_VERSION="$($PYTHON - <<'PY'
from pathlib import Path
import tomllib

with Path("pyproject.toml").open("rb") as stream:
    print(tomllib.load(stream)["project"]["version"])
PY
)"

rm -rf "$STAGING_BUNDLE"
trap 'rm -rf "$STAGING_BUNDLE"' EXIT
mkdir -p "$STAGING_BUNDLE/Contents/MacOS" "$STAGING_BUNDLE/Contents/Resources"

clone_tree() {
    local source="$1"
    local destination="$2"
    local attempt
    local clone_log="$STAGING_BUNDLE/clone-errors.log"
    mkdir -p "$destination"
    for attempt in 1 2 3; do
        # Extended attributes on dependency trees are not runtime material and
        # consume substantial metadata on nearly-full APFS volumes. Embedded
        # Mach-O signatures live in file contents. macOS cp rejects -c with
        # -X, so use its COPYFILE_DISABLE switch for the same effect.
        # -n makes retries incremental: files cloned successfully by an
        # earlier attempt are never recopied while APFS works through a
        # transiently unavailable file later in the tree.
        if COPYFILE_DISABLE=1 cp -cRn "$source/." "$destination/" \
            2>"$clone_log"; then
            rm -f "$clone_log"
            return 0
        fi
        echo "APFS clone attempt $attempt failed; retrying incomplete files." >&2
    done
    tail -20 "$clone_log" >&2 || true
    echo "Could not clone $source into the development app after 3 attempts." >&2
    return 1
}

# Keep Finder-launched code inside the application bundle. Reaching back into
# a source tree on Desktop makes macOS apply protected-folder access to the new
# app identity before Python can even finish initializing. `cp -c` uses APFS
# copy-on-write clones, so this remains space-efficient while giving the app an
# independent runtime and source snapshot.
clone_tree "$REPO_ROOT/.appenv" "$STAGING_BUNDLE/Contents/Resources/appenv"
clone_tree "$REPO_ROOT/src" "$STAGING_BUNDLE/Contents/Resources/src"

cat > "$STAGING_BUNDLE/Contents/MacOS/Sift" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

MACOS_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
RESOURCES="$(cd -- "$MACOS_DIR/../Resources" && pwd)"
PYTHON="$RESOURCES/appenv/bin/python"

show_launch_error() {
    /usr/bin/osascript -e \
        'display alert "Sift could not open" message "Its local development runtime is missing or incomplete. Rebuild the app from the Sift project folder." as critical' \
        >/dev/null 2>&1 || true
}

if [[ ! -x "$PYTHON" || ! -d "$RESOURCES/src/sift" ]]; then
    show_launch_error
    exit 1
fi

# Finder/Launch Services starts application executables with `/` as their
# working directory. Establish the app-owned resources directory before Python
# starts so environment discovery and every static asset resolve locally.
cd "$RESOURCES"
export PYTHONPATH="$RESOURCES/src"
# Launch Services injects the outer app's bundle identifier. Framework Python
# treats that inherited value as if its helper executable were the outer app
# and can block during path initialization. The real native identity is set by
# pywebview once Sift starts, so the child interpreter must not inherit it.
unset __CFBundleIdentifier
LAUNCH_DIAGNOSTIC="${TMPDIR:-/tmp}/sift-development-launch.log"
PYTHON_COMMAND=("$PYTHON")
# Launch Services may run a script CFBundleExecutable under Rosetta even on an
# Apple-silicon Mac. That architecture is inherited by Framework Python and
# makes its native arm64 wheels fail to load. Explicitly return to the hardware
# architecture before starting either the preflight or the real application.
if [[ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || true)" == "1" ]]; then
    PYTHON_COMMAND=(/usr/bin/arch -arm64 "$PYTHON")
fi
if ! "${PYTHON_COMMAND[@]}" -c "import sift.ui, webview" \
    >/dev/null 2>"$LAUNCH_DIAGNOSTIC"; then
    show_launch_error
    exit 1
fi
rm -f "$LAUNCH_DIAGNOSTIC"

exec "${PYTHON_COMMAND[@]}" -m sift
LAUNCHER
chmod +x "$STAGING_BUNDLE/Contents/MacOS/Sift"

cat > "$STAGING_BUNDLE/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>Sift</string>
    <key>CFBundleDisplayName</key>
    <string>Sift</string>
    <key>CFBundleIdentifier</key>
    <string>org.sapieninstitute.sift.development</string>
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
    <key>LSMinimumSystemVersion</key>
    <string>11.0</string>
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
    <key>SiftDevelopmentBuild</key>
    <true/>
</dict>
</plist>
PLIST

cp "$REPO_ROOT/packaging/Sift.icns" \
    "$STAGING_BUNDLE/Contents/Resources/Sift.icns"

touch "$STAGING_BUNDLE/Contents/Resources/DEVELOPMENT_BUILD"
/usr/bin/plutil -lint "$STAGING_BUNDLE/Contents/Info.plist" >/dev/null
PYTHONPATH="$STAGING_BUNDLE/Contents/Resources/src" \
    "$STAGING_BUNDLE/Contents/Resources/appenv/bin/python" \
    -c "import sift.ui, webview"

# The staged bundle is complete and launchable. Only now replace the previous
# development app, so a copy error never destroys the last working build.
rm -rf "$APP_BUNDLE"
mv "$STAGING_BUNDLE" "$APP_BUNDLE"
trap - EXIT

echo "Built $APP_BUNDLE"
