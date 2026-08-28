#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor. Sift
# keeps that system protection enabled and installs Ubuntu's own narrow
# bubblewrap profile when the default policy blocks the confinement probe.

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

if command -v bwrap >/dev/null 2>&1 && probe_bwrap; then
    echo "Ubuntu bubblewrap confinement is already available."
    exit 0
fi

[[ -r /etc/os-release ]] || { echo "Cannot identify the operating system." >&2; exit 1; }
# shellcheck disable=SC1091
. /etc/os-release
[[ "${ID:-}" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]] || {
    echo "Automatic policy preparation is supported only on Ubuntu 24.04." >&2
    echo "Sift did not weaken or alter this host's security policy." >&2
    exit 1
}

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install --yes \
    apparmor apparmor-profiles apparmor-utils bubblewrap

OFFICIAL_PROFILE="/usr/share/apparmor/extra-profiles/bwrap-userns-restrict"
INSTALLED_PROFILE="/etc/apparmor.d/bwrap-userns-restrict"
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

probe_bwrap || {
    echo "Ubuntu's official bubblewrap policy was installed, but the non-root confinement probe still failed." >&2
    exit 1
}
echo "Ubuntu's official bubblewrap AppArmor policy is active for Sift's confinement backend."
