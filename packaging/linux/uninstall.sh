#!/usr/bin/env bash
# Remove the Sift application for the current Linux user; retain user data.
set -euo pipefail
umask 077

DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_HOME="${XDG_BIN_HOME:-$HOME/.local/bin}"
APP_HOME="$DATA_HOME/sift"
APPLICATIONS_HOME="$DATA_HOME/applications"
ICON_HOME="$DATA_HOME/icons/hicolor"
METAINFO_HOME="$DATA_HOME/metainfo"
DESKTOP_ID="org.sapieninstitute.sift.desktop"
EXPECTED_EXECUTABLE="$APP_HOME/app/sift"

fail() {
    echo "$1" >&2
    exit 1
}

for LOCATION in "$DATA_HOME" "$BIN_HOME"; do
    [[ "$LOCATION" == /* ]] || fail "Sift uninstall locations must be absolute paths."
    [[ "$LOCATION" != *$'\n'* && "$LOCATION" != *$'\r'* ]] \
        || fail "Sift uninstall locations cannot contain line breaks."
done

# Never recursively remove an arbitrary directory merely because it is named
# ``sift``. Every supported installer writes this exact ownership marker.
if [[ -e "$APP_HOME" ]]; then
    [[ -f "$APP_HOME/release-metadata.json" ]] \
        || fail "Refusing to remove an unrecognized directory at $APP_HOME"
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"sift-package-metadata"' \
        "$APP_HOME/release-metadata.json" \
        || fail "Refusing to remove an unrecognized directory at $APP_HOME"
fi

if [[ -L "$BIN_HOME/sift" && "$(readlink "$BIN_HOME/sift")" == "$EXPECTED_EXECUTABLE" ]]; then
    rm -- "$BIN_HOME/sift"
fi
rm -f -- "$APPLICATIONS_HOME/$DESKTOP_ID"
rm -f -- "$METAINFO_HOME/org.sapieninstitute.sift.metainfo.xml"
for ICON in "$ICON_HOME"/*x*/apps/org.sapieninstitute.sift.png; do
    [[ -e "$ICON" ]] && rm -- "$ICON"
done
rm -rf -- "$APP_HOME"

command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$APPLICATIONS_HOME" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 \
    && gtk-update-icon-cache -q -t "$ICON_HOME" >/dev/null 2>&1 || true

echo "Sift was removed. Sessions and credential-vault entries were retained."
