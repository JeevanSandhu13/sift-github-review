#!/usr/bin/env bash
set -Eeuo pipefail

QUALIFICATION_PHASE="startup"
report_failure() {
    local status="$?"
    echo "Linux artifact qualification failed during: $QUALIFICATION_PHASE" >&2
    exit "$status"
}
trap report_failure ERR

phase() {
    QUALIFICATION_PHASE="$1"
    echo "Linux artifact qualification: $QUALIFICATION_PHASE"
}

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

[[ "$(uname -s)" == "Linux" ]] || {
    echo "Linux installation qualification must run on Linux." >&2
    exit 1
}

ARCHIVE="${1:?pass the Sift Linux archive}"
[[ -f "$ARCHIVE" ]] || { echo "Missing archive: $ARCHIVE" >&2; exit 1; }
if [[ -n "${SIFT_LINUX_QUALIFICATION_ROOT:-}" ]]; then
    TEST_ROOT="$SIFT_LINUX_QUALIFICATION_ROOT"
    [[ "$TEST_ROOT" == /* ]] || {
        echo "SIFT_LINUX_QUALIFICATION_ROOT must be absolute." >&2
        exit 1
    }
    [[ ! -e "$TEST_ROOT" ]] || {
        echo "Qualification root already exists: $TEST_ROOT" >&2
        exit 1
    }
    mkdir -m 0700 -p -- "$TEST_ROOT"
else
    TEST_ROOT="$(mktemp -d)"
fi
cleanup() { rm -rf -- "$TEST_ROOT"; }
trap cleanup EXIT INT TERM

BUNDLE_ROOT="$TEST_ROOT/bundle"
USER_HOME="$TEST_ROOT/home with spaces % dollar\$ tick\`"
DATA_HOME="$USER_HOME/.local/share"
BIN_HOME="$USER_HOME/.local/bin"
mkdir -p "$BUNDLE_ROOT" "$USER_HOME/.sift-sessions/retained"
printf 'retain' > "$USER_HOME/.sift-sessions/retained/session.sentinel"
phase "extract archive"
tar -xzf "$ARCHIVE" -C "$BUNDLE_ROOT"

run_install() {
    HOME="$USER_HOME" XDG_DATA_HOME="$DATA_HOME" XDG_BIN_HOME="$BIN_HOME" \
        "$BUNDLE_ROOT/Sift/install.sh" >/dev/null
}
run_uninstall() {
    HOME="$USER_HOME" XDG_DATA_HOME="$DATA_HOME" XDG_BIN_HOME="$BIN_HOME" \
        "$DATA_HOME/sift/uninstall.sh" >/dev/null
}

phase "clean install"
run_install
phase "installed platform check"
if ! PLATFORM_REPORT="$("$BIN_HOME/sift" --platform-check)"; then
    # The report is deliberately content-free and identifies the exact
    # renderer, credential-store, or confinement prerequisite that failed.
    printf '%s\n' "$PLATFORM_REPORT" >&2
    false
fi
phase "installed integration check"
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --integration-check >/dev/null
phase "installed format-worker check"
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --format-check >/dev/null
phase "installed renderer check"
xvfb-run -a "$BIN_HOME/sift" --renderer-check >/dev/null
phase "installed analysis-runtime check"
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --analysis-check >/dev/null
phase "installed credential-store check"
"$SCRIPT_ROOT/linux/qualify_credential_store.sh" "$BIN_HOME/sift" >/dev/null
phase "installed command and desktop metadata checks"
"$BIN_HOME/sift" --help >/dev/null
[[ -f "$DATA_HOME/applications/org.sapieninstitute.sift.desktop" ]]
[[ -x "$DATA_HOME/sift/uninstall.sh" ]]
[[ -f "$DATA_HOME/sift/INSTALL.txt" ]]
[[ -f "$DATA_HOME/sift/LICENSE.txt" ]]
desktop-file-validate "$DATA_HOME/applications/org.sapieninstitute.sift.desktop"
appstreamcli validate --no-net \
    "$DATA_HOME/metainfo/org.sapieninstitute.sift.metainfo.xml"

phase "in-place upgrade"
run_install
phase "upgrade state-preservation check"
[[ "$(cat "$USER_HOME/.sift-sessions/retained/session.sentinel")" == "retain" ]]
phase "upgraded runtime checks"
if ! PLATFORM_REPORT="$("$BIN_HOME/sift" --platform-check)"; then
    printf '%s\n' "$PLATFORM_REPORT" >&2
    false
fi
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 "$BIN_HOME/sift" --format-check >/dev/null

phase "uninstall"
run_uninstall
phase "uninstall and user-state preservation checks"
[[ ! -e "$BIN_HOME/sift" ]]
[[ ! -e "$DATA_HOME/sift" ]]
[[ "$(cat "$USER_HOME/.sift-sessions/retained/session.sentinel")" == "retain" ]]
echo "Linux clean install, upgrade, execution, and uninstall qualification passed."
