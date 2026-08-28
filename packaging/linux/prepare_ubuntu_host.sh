#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Sift
# keeps that system protection enabled and installs two narrow allowances:
# Ubuntu's official bubblewrap profile for analysis confinement and a
# path-bound profile for Qt WebEngine's Chromium renderer sandbox. The latter
# is preferable to QTWEBENGINE_DISABLE_SANDBOX/--no-sandbox, which Sift never
# enables.

[[ "$(id -u)" == "0" ]] || {
    echo "Run this host-preparation helper as root (for example, with sudo)." >&2
    exit 1
}

TEST_USER="${SIFT_BWRAP_TEST_USER:-${SUDO_USER:-}}"
[[ -n "$TEST_USER" && "$TEST_USER" != "root" ]] || {
    echo "Set SIFT_BWRAP_TEST_USER to the non-root account that will run Sift." >&2
    exit 1
}
id "$TEST_USER" >/dev/null 2>&1 || {
    echo "The Sift test account '$TEST_USER' does not exist." >&2
    exit 1
}

probe_bwrap() {
    runuser -u "$TEST_USER" -- bwrap \
        --unshare-user --unshare-pid --unshare-net \
        --ro-bind / / --dev /dev --proc /proc /bin/true \
        >/dev/null 2>&1
}

[[ -r /etc/os-release ]] || { echo "Cannot identify the operating system." >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || {
    if command -v bwrap >/dev/null 2>&1 && probe_bwrap; then
        echo "Ubuntu bubblewrap confinement is already available."
        exit 0
    fi
    echo "Automatic policy preparation is supported only on Ubuntu 24.04." >&2
    echo "Sift did not weaken or alter this host's security policy." >&2
    exit 1
}

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    apparmor apparmor-profiles apparmor-utils bubblewrap

OFFICIAL_PROFILE="/usr/share/apparmor/extra-profiles/bwrap-userns-restrict"
INSTALLED_PROFILE="/etc/apparmor.d/bwrap-userns-restrict"
if ! probe_bwrap; then
    [[ -f "$OFFICIAL_PROFILE" && ! -L "$OFFICIAL_PROFILE" ]] || {
        echo "Ubuntu's official bubblewrap AppArmor profile is unavailable." >&2
        exit 1
    }
    if [[ -e "$INSTALLED_PROFILE" ]] && ! cmp -s "$OFFICIAL_PROFILE" "$INSTALLED_PROFILE"; then
        echo "A customized bubblewrap AppArmor profile already exists; refusing to overwrite it." >&2
        exit 1
    fi
    install -o root -g root -m 0644 "$OFFICIAL_PROFILE" "$INSTALLED_PROFILE"
    apparmor_parser -r "$INSTALLED_PROFILE"
fi

probe_bwrap || {
    echo "Ubuntu's official bubblewrap policy was installed, but the non-root confinement probe still failed." >&2
    exit 1
}

# Qt WebEngine's supported Linux sandbox uses unprivileged user namespaces;
# its setuid sandbox is deliberately unavailable. Ubuntu recommends a
# path-bound AppArmor profile with only `userns,` for applications that need
# this facility. Bind the allowance to Sift's bundled helper executable rather
# than weakening the host-wide restriction or disabling Chromium's sandbox.
TARGET_HOME="$(getent passwd "$TEST_USER" | cut -d: -f6)"
[[ "$TARGET_HOME" == /* && "$TARGET_HOME" != *$'\n'* && "$TARGET_HOME" != *$'\r'* ]] || {
    echo "Cannot determine a safe home directory for '$TEST_USER'." >&2
    exit 1
}
DEFAULT_QTWEBENGINE_PROCESS="$TARGET_HOME/.local/share/sift/app/_internal/PyQt6/Qt6/libexec/QtWebEngineProcess"
QTWEBENGINE_PROCESSES=("${SIFT_QTWEBENGINE_PROCESS:-$DEFAULT_QTWEBENGINE_PROCESS}")
if [[ -n "${SIFT_LINUX_QUALIFICATION_ROOT:-}" ]]; then
    [[ "$SIFT_LINUX_QUALIFICATION_ROOT" == /* ]] || {
        echo "SIFT_LINUX_QUALIFICATION_ROOT must be absolute." >&2
        exit 1
    }
    QTWEBENGINE_PROCESSES+=(
        "$SIFT_LINUX_QUALIFICATION_ROOT/home with spaces % dollar\$ tick\`/.local/share/sift/app/_internal/PyQt6/Qt6/libexec/QtWebEngineProcess"
    )
fi

SIFT_PROFILE="/etc/apparmor.d/org.sapieninstitute.sift.qtwebengine"
SIFT_PROFILE_TEMP="$(mktemp)"
trap 'rm -f -- "$SIFT_PROFILE_TEMP"' EXIT
{
    printf '%s\n' '# Managed by Sift prepare_ubuntu_host.sh.'
    printf '%s\n' 'abi <abi/4.0>,' 'include <tunables/global>' ''
    PROFILE_INDEX=0
    for PROCESS_PATH in "${QTWEBENGINE_PROCESSES[@]}"; do
        [[ "$PROCESS_PATH" == /* && "$PROCESS_PATH" != *$'\n'* && "$PROCESS_PATH" != *$'\r'* ]] || {
            echo "Refusing an unsafe Qt WebEngine process path." >&2
            exit 1
        }
        ATTACHMENT="${PROCESS_PATH//\\/\\\\}"
        ATTACHMENT="${ATTACHMENT//\"/\\\"}"
        printf 'profile sift-qtwebengine-%d "%s" flags=(unconfined) {\n' \
            "$PROFILE_INDEX" "$ATTACHMENT"
        printf '%s\n' '  userns,' '}' ''
        PROFILE_INDEX=$((PROFILE_INDEX + 1))
    done
} > "$SIFT_PROFILE_TEMP"

if [[ -e "$SIFT_PROFILE" ]] && ! grep -Fqx '# Managed by Sift prepare_ubuntu_host.sh.' "$SIFT_PROFILE"; then
    echo "A non-Sift AppArmor profile already exists at $SIFT_PROFILE; refusing to overwrite it." >&2
    exit 1
fi
install -o root -g root -m 0644 "$SIFT_PROFILE_TEMP" "$SIFT_PROFILE"
apparmor_parser -r "$SIFT_PROFILE"

echo "Ubuntu's bubblewrap and Qt WebEngine sandbox policies are active for Sift."
