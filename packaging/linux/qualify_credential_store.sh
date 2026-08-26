#!/usr/bin/env bash
# Prove the frozen Linux app can use a real Secret Service vault without
# touching the maintainer's login keyring or retaining the random canary.
set -euo pipefail

EXECUTABLE="${1:?pass the frozen Sift executable}"
[[ -x "$EXECUTABLE" ]] || {
    echo "Missing executable: $EXECUTABLE" >&2
    exit 1
}
command -v dbus-run-session >/dev/null || {
    echo "dbus-run-session is required for credential-store qualification." >&2
    exit 1
}
command -v gnome-keyring-daemon >/dev/null || {
    echo "gnome-keyring-daemon is required for credential-store qualification." >&2
    exit 1
}

EXECUTABLE="$(cd -- "$(dirname -- "$EXECUTABLE")" && pwd)/$(basename -- "$EXECUTABLE")"
TEST_HOME="$(mktemp -d)"
cleanup() { rm -rf -- "$TEST_HOME"; }
trap cleanup EXIT INT TERM

HOME="$TEST_HOME" dbus-run-session -- bash -euo pipefail -c '
    eval "$(printf "\n" | gnome-keyring-daemon --unlock --components=secrets)"
    "$1" --credential-store-check >/dev/null
' bash "$EXECUTABLE"

echo "Linux Secret Service credential-store qualification passed."
