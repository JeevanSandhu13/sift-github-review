#!/usr/bin/env bash
# Install the unpacked Sift bundle for the current Linux user.
set -euo pipefail
umask 077

BUNDLE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_HOME="${XDG_DATA_HOME:-$HOME/.local/share}"
BIN_HOME="${XDG_BIN_HOME:-$HOME/.local/bin}"
APP_HOME="$DATA_HOME/sift"
APPLICATIONS_HOME="$DATA_HOME/applications"
ICON_HOME="$DATA_HOME/icons/hicolor"
METAINFO_HOME="$DATA_HOME/metainfo"
DESKTOP_ID="org.sapieninstitute.sift.desktop"
STAGING=""
ROLLBACK_ROOT=""
INSTALL_COMPLETE=0
APP_SWAPPED=0
HAD_PREVIOUS_APP=0
declare -a MANAGED_TARGETS=()
declare -a MANAGED_BACKUPS=()

cleanup() {
    local status=$?
    local index target backup
    trap - EXIT INT TERM
    set +e
    if [[ "$status" -ne 0 && "$INSTALL_COMPLETE" -ne 1 ]]; then
        # Restore shell-integration files in reverse order.  Never remove an
        # unexpected directory: every managed destination is checked before
        # it is registered with this transaction.
        for ((index=${#MANAGED_TARGETS[@]} - 1; index >= 0; index--)); do
            target="${MANAGED_TARGETS[$index]}"
            backup="${MANAGED_BACKUPS[$index]}"
            if [[ -e "$target" || -L "$target" ]]; then
                [[ ! -d "$target" || -L "$target" ]] && rm -f -- "$target"
            fi
            if [[ -e "$backup" || -L "$backup" ]]; then
                mkdir -p "$(dirname -- "$target")"
                cp -a -- "$backup" "$target"
            fi
        done

        # The prior application remains recoverable until every desktop,
        # icon, metadata, and launcher operation has succeeded.
        if [[ "$HAD_PREVIOUS_APP" -eq 1 && -e "$ROLLBACK_ROOT/app" ]]; then
            [[ "$APP_SWAPPED" -eq 0 ]] || rm -rf -- "$APP_HOME"
            mv -- "$ROLLBACK_ROOT/app" "$APP_HOME"
        elif [[ "$APP_SWAPPED" -eq 1 ]]; then
            rm -rf -- "$APP_HOME"
        fi
    fi
    [[ -z "$STAGING" ]] || rm -rf -- "$STAGING"
    [[ -z "$ROLLBACK_ROOT" ]] || rm -rf -- "$ROLLBACK_ROOT"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

fail() {
    echo "$1" >&2
    exit 1
}

# A desktop Exec field cannot safely represent embedded line breaks, and XDG
# paths are required to be absolute. Reject malformed environment overrides
# instead of producing an entry that launches a different command.
for LOCATION in "$DATA_HOME" "$BIN_HOME"; do
    [[ "$LOCATION" == /* ]] || fail "Sift install locations must be absolute paths."
    [[ "$LOCATION" != *$'\n'* && "$LOCATION" != *$'\r'* ]] \
        || fail "Sift install locations cannot contain line breaks."
done

test -x "$BUNDLE_ROOT/app/sift" || {
    echo "This Sift bundle is incomplete (app/sift is missing)." >&2
    exit 1
}
test -f "$BUNDLE_ROOT/share/applications/$DESKTOP_ID.in" || {
    echo "This Sift bundle is incomplete (desktop entry is missing)." >&2
    exit 1
}
for REQUIRED_FILE in install.sh uninstall.sh INSTALL.txt LICENSE.txt release-metadata.json; do
    test -f "$BUNDLE_ROOT/$REQUIRED_FILE" \
        || fail "This Sift bundle is incomplete ($REQUIRED_FILE is missing)."
done

# The install directory is owned only when it contains Sift's package marker.
# Refuse symlinks and unrelated files/directories rather than moving or
# replacing researcher-owned material that happens to use the same name.
if [[ -L "$APP_HOME" ]]; then
    fail "Refusing to replace a symbolic link at $APP_HOME"
fi
if [[ -e "$APP_HOME" ]]; then
    [[ -d "$APP_HOME" && -f "$APP_HOME/release-metadata.json" ]] \
        || fail "Refusing to replace an unrecognized path at $APP_HOME"
    grep -Eq '"format"[[:space:]]*:[[:space:]]*"sift-package-metadata"' \
        "$APP_HOME/release-metadata.json" \
        || fail "Refusing to replace an unrecognized directory at $APP_HOME"
fi

mkdir -p "$DATA_HOME" "$BIN_HOME" "$APPLICATIONS_HOME" "$METAINFO_HOME"
STAGING="$(mktemp -d "$DATA_HOME/.sift-installing.XXXXXXXX")"
ROLLBACK_ROOT="$(mktemp -d "$DATA_HOME/.sift-rollback.XXXXXXXX")"
cp -R "$BUNDLE_ROOT/app" "$STAGING/app"
cp "$BUNDLE_ROOT/release-metadata.json" "$STAGING/release-metadata.json"
cp "$BUNDLE_ROOT/install.sh" "$BUNDLE_ROOT/uninstall.sh" "$STAGING/"
cp "$BUNDLE_ROOT/INSTALL.txt" "$BUNDLE_ROOT/LICENSE.txt" "$STAGING/"
# Retaining the small integration tree makes the installed copy a complete,
# repeatable installer source.  This also supports the edge case where a
# researcher extracts the archive directly over the default install path.
cp -R "$BUNDLE_ROOT/share" "$STAGING/share"
chmod 0700 "$STAGING/install.sh" "$STAGING/uninstall.sh"
chmod 0600 "$STAGING/INSTALL.txt" "$STAGING/LICENSE.txt" \
    "$STAGING/release-metadata.json"

if [[ -e "$APP_HOME" ]]; then
    HAD_PREVIOUS_APP=1
    mv "$APP_HOME" "$ROLLBACK_ROOT/app"
fi
if ! mv "$STAGING" "$APP_HOME"; then
    echo "Sift could not be installed into $APP_HOME" >&2
    exit 1
fi
STAGING=""
APP_SWAPPED=1

backup_managed_target() {
    local target="$1"
    local name="$2"
    local backup="$ROLLBACK_ROOT/integration/$name"
    [[ ! -d "$target" || -L "$target" ]] \
        || fail "Cannot replace a directory at $target"
    mkdir -p "$(dirname -- "$backup")"
    if [[ -e "$target" || -L "$target" ]]; then
        cp -a -- "$target" "$backup"
    fi
    MANAGED_TARGETS+=("$target")
    MANAGED_BACKUPS+=("$backup")
}

EXECUTABLE="$APP_HOME/app/sift"
# Desktop Entry quoting requires these four characters to be backslash-
# escaped inside a quoted argument. Then escape sed's replacement syntax
# separately so an unusual but valid home directory cannot corrupt the file.
DESKTOP_EXECUTABLE="${EXECUTABLE//\\/\\\\}"
DESKTOP_EXECUTABLE="${DESKTOP_EXECUTABLE//\"/\\\"}"
DESKTOP_EXECUTABLE="${DESKTOP_EXECUTABLE//\`/\\\`}"
DESKTOP_EXECUTABLE="${DESKTOP_EXECUTABLE//\$/\\\$}"
# Percent introduces a Desktop Entry field code. A literal percent is %%.
DESKTOP_EXECUTABLE="${DESKTOP_EXECUTABLE//%/%%}"
SED_EXECUTABLE="${DESKTOP_EXECUTABLE//\\/\\\\}"
SED_EXECUTABLE="${SED_EXECUTABLE//&/\\&}"
SED_EXECUTABLE="${SED_EXECUTABLE//|/\\|}"
DESKTOP_TARGET="$APPLICATIONS_HOME/$DESKTOP_ID"
backup_managed_target "$DESKTOP_TARGET" "desktop"
DESKTOP_TEMP="$(mktemp "$APPLICATIONS_HOME/.sift-desktop.XXXXXXXX")"
sed "s|__SIFT_EXECUTABLE__|$SED_EXECUTABLE|g" \
    "$APP_HOME/share/applications/$DESKTOP_ID.in" \
    > "$DESKTOP_TEMP"
chmod 0644 "$DESKTOP_TEMP"
mv -f -- "$DESKTOP_TEMP" "$DESKTOP_TARGET"

install_file_atomic() {
    local source="$1"
    local destination="$2"
    local destination_dir temporary
    destination_dir="$(dirname -- "$destination")"
    mkdir -p "$destination_dir"
    [[ ! -d "$destination" ]] || fail "Cannot replace a directory at $destination"
    temporary="$(mktemp "$destination_dir/.sift-file.XXXXXXXX")"
    cp "$source" "$temporary"
    chmod 0644 "$temporary"
    mv -f -- "$temporary" "$destination"
}

for SIZE_DIR in "$APP_HOME"/share/icons/hicolor/*x*/apps; do
    [[ -d "$SIZE_DIR" ]] || continue
    RELATIVE="${SIZE_DIR#"$APP_HOME/share/icons/hicolor/"}"
    ICON_TARGET="$ICON_HOME/$RELATIVE/org.sapieninstitute.sift.png"
    backup_managed_target "$ICON_TARGET" "icons/$RELATIVE/org.sapieninstitute.sift.png"
    install_file_atomic \
        "$SIZE_DIR/org.sapieninstitute.sift.png" \
        "$ICON_TARGET"
done
METAINFO_TARGET="$METAINFO_HOME/org.sapieninstitute.sift.metainfo.xml"
backup_managed_target "$METAINFO_TARGET" "metainfo"
install_file_atomic \
    "$APP_HOME/share/metainfo/org.sapieninstitute.sift.metainfo.xml" \
    "$METAINFO_TARGET"

backup_managed_target "$BIN_HOME/sift" "launcher"
LAUNCHER_TEMP="$BIN_HOME/.sift-launcher.$RANDOM.$RANDOM"
ln -s -- "$EXECUTABLE" "$LAUNCHER_TEMP"
mv -f -- "$LAUNCHER_TEMP" "$BIN_HOME/sift"

command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$APPLICATIONS_HOME" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null 2>&1 \
    && gtk-update-icon-cache -q -t "$ICON_HOME" >/dev/null 2>&1 || true

INSTALL_COMPLETE=1
rm -rf -- "$ROLLBACK_ROOT"
ROLLBACK_ROOT=""
echo "Sift is installed. Open it from your applications menu or run: $BIN_HOME/sift"
if [[ ":$PATH:" != *":$BIN_HOME:"* ]]; then
    echo "Note: add $BIN_HOME to PATH to launch Sift by typing 'sift'."
fi
