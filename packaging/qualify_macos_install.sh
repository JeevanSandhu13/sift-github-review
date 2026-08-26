#!/usr/bin/env bash
set -euo pipefail

[[ "$(uname -s)" == "Darwin" ]] || {
    echo "macOS installation qualification must run on macOS." >&2
    exit 1
}

SOURCE_APP="${1:-dist/Sift.app}"
[[ -d "$SOURCE_APP" ]] || { echo "Missing application: $SOURCE_APP" >&2; exit 1; }
SOURCE_APP="$(cd "$(dirname "$SOURCE_APP")" && pwd)/$(basename "$SOURCE_APP")"
# Stage beside the build by default. Large frozen runtimes can cross a
# sandboxed runner's separate system-temporary volume or policy boundary;
# same-volume staging also makes replacement behavior match /Applications.
TEST_PARENT="${SIFT_INSTALL_TEST_PARENT:-$(dirname "$SOURCE_APP")/.install-qualification}"
mkdir -p "$TEST_PARENT"
TEST_ROOT="$(mktemp -d "$TEST_PARENT/Sift.XXXXXX")"
INSTALL_APP="$TEST_ROOT/Applications/Sift.app"
STATE_ROOT="$TEST_ROOT/researcher-state"
cleanup() { rm -rf -- "$TEST_ROOT"; }
trap cleanup EXIT INT TERM

mkdir -p "$(dirname "$INSTALL_APP")" "$STATE_ROOT"
printf 'retain' > "$STATE_ROOT/session.sentinel"
/bin/cp -R "$SOURCE_APP" "$INSTALL_APP"

PLIST="$INSTALL_APP/Contents/Info.plist"
EXECUTABLE="$INSTALL_APP/Contents/Resources/sift/sift"
/usr/bin/plutil -lint "$PLIST" >/dev/null
[[ -x "$EXECUTABLE" ]] || { echo "Installed executable is missing." >&2; exit 1; }
"$EXECUTABLE" --platform-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$EXECUTABLE" --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$EXECUTABLE" --format-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$EXECUTABLE" --analysis-check >/dev/null
"$EXECUTABLE" --help >/dev/null

# An upgrade is a replacement of the app bundle, not user state. Use the same
# copy mechanism a DMG installation uses and prove external research state is
# untouched.
rm -rf -- "$INSTALL_APP"
/bin/cp -R "$SOURCE_APP" "$INSTALL_APP"
[[ "$(cat "$STATE_ROOT/session.sentinel")" == "retain" ]]
"$INSTALL_APP/Contents/Resources/sift/sift" --platform-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$INSTALL_APP/Contents/Resources/sift/sift" --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$INSTALL_APP/Contents/Resources/sift/sift" --format-check >/dev/null

rm -rf -- "$INSTALL_APP"
[[ ! -e "$INSTALL_APP" ]]
[[ "$(cat "$STATE_ROOT/session.sentinel")" == "retain" ]]
echo "macOS clean install, upgrade, execution, and removal qualification passed."
