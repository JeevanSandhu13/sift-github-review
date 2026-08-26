#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

[[ "$(uname -s)" == "Linux" ]] || {
    echo "Linux installation qualification must run on Linux." >&2
    exit 1
}

ARCHIVE="${1:?pass the Sift Linux archive}"
[[ -f "$ARCHIVE" ]] || { echo "Missing archive: $ARCHIVE" >&2; exit 1; }
TEST_ROOT="$(mktemp -d)"
cleanup() { rm -rf -- "$TEST_ROOT"; }
trap cleanup EXIT INT TERM

BUNDLE_ROOT="$TEST_ROOT/bundle"
USER_HOME="$TEST_ROOT/home with spaces % dollar\$ tick\`"
DATA_HOME="$USER_HOME/.local/share"
BIN_HOME="$USER_HOME/.local/bin"
mkdir -p "$BUNDLE_ROOT" "$USER_HOME/.sift-sessions/retained"
printf 'retain' > "$USER_HOME/.sift-sessions/retained/session.sentinel"
tar -xzf "$ARCHIVE" -C "$BUNDLE_ROOT"

run_install() {
    HOME="$USER_HOME" XDG_DATA_HOME="$DATA_HOME" XDG_BIN_HOME="$BIN_HOME" \
        "$BUNDLE_ROOT/Sift/install.sh" >/dev/null
}
run_uninstall() {
    HOME="$USER_HOME" XDG_DATA_HOME="$DATA_HOME" XDG_BIN_HOME="$BIN_HOME" \
        "$DATA_HOME/sift/uninstall.sh" >/dev/null
}

run_install
"$BIN_HOME/sift" --platform-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --format-check >/dev/null
xvfb-run -a "$BIN_HOME/sift" --renderer-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --analysis-check >/dev/null
"$SCRIPT_ROOT/linux/qualify_credential_store.sh" "$BIN_HOME/sift" >/dev/null
"$BIN_HOME/sift" --help >/dev/null
[[ -f "$DATA_HOME/applications/org.sapieninstitute.sift.desktop" ]]
[[ -x "$DATA_HOME/sift/uninstall.sh" ]]
[[ -f "$DATA_HOME/sift/INSTALL.txt" ]]
[[ -f "$DATA_HOME/sift/LICENSE.txt" ]]
desktop-file-validate "$DATA_HOME/applications/org.sapieninstitute.sift.desktop"
appstreamcli validate --no-net \
    "$DATA_HOME/metainfo/org.sapieninstitute.sift.metainfo.xml"

run_install
[[ "$(cat "$USER_HOME/.sift-sessions/retained/session.sentinel")" == "retain" ]]
"$BIN_HOME/sift" --platform-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --format-check >/dev/null

run_uninstall
[[ ! -e "$BIN_HOME/sift" ]]
[[ ! -e "$DATA_HOME/sift" ]]
[[ "$(cat "$USER_HOME/.sift-sessions/retained/session.sentinel")" == "retain" ]]
echo "Linux clean install, upgrade, execution, and uninstall qualification passed."
