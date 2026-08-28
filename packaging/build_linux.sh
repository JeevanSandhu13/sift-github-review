#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

[[ "$(uname -s)" == "Linux" ]] || { echo "This release must be built on Linux." >&2; exit 1; }
command -v uv >/dev/null || { echo "uv is required." >&2; exit 1; }
command -v cc >/dev/null || { echo "A C compiler is required for native dependency builds (install build-essential on Ubuntu)." >&2; exit 1; }
command -v bwrap >/dev/null || { echo "bubblewrap is required." >&2; exit 1; }
command -v getconf >/dev/null || { echo "getconf is required." >&2; exit 1; }
command -v xvfb-run >/dev/null || { echo "xvfb-run is required for the renderer qualification." >&2; exit 1; }
command -v desktop-file-validate >/dev/null || { echo "desktop-file-validate is required." >&2; exit 1; }
command -v appstreamcli >/dev/null || { echo "appstreamcli is required." >&2; exit 1; }
command -v dbus-run-session >/dev/null || { echo "dbus-run-session is required." >&2; exit 1; }
command -v gnome-keyring-daemon >/dev/null || { echo "gnome-keyring-daemon is required." >&2; exit 1; }

require_shared_library() {
    local soname="$1"
    local package="$2"
    # Do not use grep -q here: with pipefail it can close the pipe early,
    # make ldconfig exit on SIGPIPE, and falsely report a library as absent.
    ldconfig -p 2>/dev/null | grep -F "$soname" >/dev/null || {
        echo "$soname is required for a complete Linux release build (install $package on Ubuntu)." >&2
        exit 1
    }
}

# These libraries are dependencies of advertised connectors, the Qt Linux
# platform/runtime modules, and Numba's accelerated thread pool.  Check them
# before the expensive freeze so a nominally installed Python wheel cannot
# produce an artifact with unresolved ELF dependencies.
require_shared_library "libodbc.so.2" "unixodbc"
require_shared_library "libxcb-shape.so.0" "libxcb-shape0"
require_shared_library "libxcb-icccm.so.4" "libxcb-icccm4"
require_shared_library "libxcb-keysyms.so.1" "libxcb-keysyms1"
require_shared_library "libxcb-dri3.so.0" "libxcb-dri3-0"
require_shared_library "libxcb-image.so.0" "libxcb-image0"
require_shared_library "libxcb-randr.so.0" "libxcb-randr0"
require_shared_library "libxcb-render-util.so.0" "libxcb-render-util0"
require_shared_library "libxcb-sync.so.1" "libxcb-sync1"
require_shared_library "libxcb-util.so.1" "libxcb-util1"
require_shared_library "libxcb-xfixes.so.0" "libxcb-xfixes0"
require_shared_library "libxcb-xkb.so.1" "libxcb-xkb1"
require_shared_library "libpulse.so.0" "libpulse0"
require_shared_library "libtbb.so.12" "libtbb12"
require_shared_library "libsnappy.so.1" "libsnappy1v5"
require_shared_library "libwebpdemux.so.2" "libwebpdemux2"
require_shared_library "libwebpmux.so.3" "libwebpmux3"
require_shared_library "libasound.so.2" "libasound2 (Ubuntu 22.04) or libasound2t64 (Ubuntu 24.04)"
require_shared_library "libminizip.so.1" "libminizip1 (Ubuntu 22.04) or libminizip1t64 (Ubuntu 24.04)"
require_shared_library "libgbm.so.1" "libgbm1"
require_shared_library "libEGL.so.1" "libegl1"
require_shared_library "libnspr4.so" "libnspr4"
require_shared_library "libnss3.so" "libnss3"
require_shared_library "libXdamage.so.1" "libxdamage1"
require_shared_library "libxkbfile.so.1" "libxkbfile1"
require_shared_library "libwayland-server.so.0" "libwayland-server0"

BUILD_ARCH="$(uname -m)"
UV_ARCH_SYNC_ARGS=()
if [[ "$BUILD_ARCH" == "aarch64" || "$BUILD_ARCH" == "arm64" ]]; then
    # cryptography's current manylinux ARM64 wheel can select instructions that
    # are reported, but not executable, by some ARM hypervisors.  The supported
    # release stays on the current security-fixed version and compiles it for
    # the qualification host's actual CPU instead of downgrading the package.
    command -v cargo >/dev/null || { echo "Rust/cargo is required for the secure ARM64 cryptography build." >&2; exit 1; }
    command -v rustc >/dev/null || { echo "Rust 1.83 or newer is required for the secure ARM64 cryptography build." >&2; exit 1; }
    command -v pkg-config >/dev/null || { echo "pkg-config is required for the secure ARM64 cryptography build." >&2; exit 1; }
    pkg-config --exists openssl || { echo "OpenSSL development headers are required for the secure ARM64 cryptography build." >&2; exit 1; }
    RUST_VERSION="$(rustc --version | awk '{print $2}')"
    if [[ "$(printf '%s\n' "1.83.0" "$RUST_VERSION" | sort -V | head -1)" != "1.83.0" ]]; then
        echo "Rust 1.83 or newer is required (found $RUST_VERSION)." >&2
        exit 1
    fi
    UV_ARCH_SYNC_ARGS+=(--no-binary-package cryptography --reinstall-package cryptography)
fi

# PyInstaller deliberately does not bundle glibc; an artifact built on a
# newer glibc can fail on an older workstation. Current Qt ARM64 wheels require
# glibc 2.39, while current x86_64 wheels still support the 2.35 baseline.
GLIBC_VERSION="$(getconf GNU_LIBC_VERSION | awk '{print $2}')"
if [[ "$BUILD_ARCH" == "aarch64" || "$BUILD_ARCH" == "arm64" ]]; then
    [[ "$GLIBC_VERSION" == "2.39" ]] || {
        echo "Linux ARM64 releases must be built on the glibc 2.39 baseline (found $GLIBC_VERSION)." >&2
        echo "Use the Ubuntu 24.04 ARM64 qualification image." >&2
        exit 1
    }
elif [[ "$(printf '%s\n' "2.35" "$GLIBC_VERSION" | sort -V | tail -1)" != "2.35" ]]; then
    echo "Linux x86_64 releases must be built with glibc 2.35 or older (found $GLIBC_VERSION)." >&2
    echo "Use the Ubuntu 22.04 x86_64 qualification image." >&2
    exit 1
fi

RELEASE_MODE="${SIFT_RELEASE_MODE:-development}"
RELEASE_CHANNEL="${SIFT_RELEASE_CHANNEL:-stable}"
[[ "$RELEASE_MODE" == "development" || "$RELEASE_MODE" == "production" ]] \
    || { echo "SIFT_RELEASE_MODE must be development or production." >&2; exit 1; }
[[ "$RELEASE_CHANNEL" == "stable" || "$RELEASE_CHANNEL" == "beta" ]] \
    || { echo "SIFT_RELEASE_CHANNEL must be stable or beta." >&2; exit 1; }
if [[ "$RELEASE_MODE" == "production" ]]; then
    [[ "${SIFT_SKIP_VENDOR:-0}" != "1" ]] \
        || { echo "Production cannot skip the bundled analysis runtime." >&2; exit 1; }
    [[ "${SIFT_SKIP_FREEZE:-0}" != "1" ]] \
        || { echo "Production cannot reuse a previously frozen application." >&2; exit 1; }
    [[ -n "${SIFT_RELEASE_PRIVATE_KEY_B64:-}" ]] \
        || { echo "Production requires SIFT_RELEASE_PRIVATE_KEY_B64." >&2; exit 1; }
    [[ -n "${SIFT_RELEASE_KEY_ID:-}" ]] \
        || { echo "Production requires SIFT_RELEASE_KEY_ID." >&2; exit 1; }
    PYTHONPATH=src uv run python -c '
from sift.update_config import load_update_policy
p = load_update_policy()
raise SystemExit(0 if p.get("configured") is True else "production update policy is not configured")
'
fi

# Establish the complete locked environment before invoking any Python build
# helper.  On ARM64 this guarantees the current cryptography release is built
# for the host CPU and prevents an incompatible generic wheel from ever being
# imported during the release pipeline.
uv sync --locked --all-extras "${UV_ARCH_SYNC_ARGS[@]}"
uv run python -c '
import cryptography
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
assert cryptography.__version__ == "50.0.0", cryptography.__version__
assert len(AESGCM.generate_key(bit_length=256)) == 32
'
if [[ "${SIFT_SKIP_VENDOR:-0}" != "1" ]]; then
    uv run --no-sync python packaging/vendor_python.py
fi
uv run python packaging/generate_brand_assets.py --check
uv run python packaging/verify_database_drivers.py

PYTHONPATH=src uv run python -m sift --platform-check
PYTHONPATH=src uv run python -m sift --integration-check >/dev/null
PYTHONPATH=src uv run python -m sift --format-check >/dev/null

PYTHONPATH=src uv run python -c '
from sift.env_detect import bwrap_baseline_result
ok, detail = bwrap_baseline_result()
raise SystemExit(0 if ok else "bubblewrap policy probe failed: " + detail)
'

if [[ "${SIFT_SKIP_TESTS:-0}" != "1" ]]; then
    # Materialize source-bound scientific evidence on the native reference
    # host before the broader suite consumes it. This keeps a clean checkout
    # honest while allowing non-reference macOS/Windows lanes to skip only the
    # three evidence-dependent assertions.
    uv run python scripts/method_qualification_evidence.py
    uv run pytest -q
    uv run python scripts/scientific_qualification.py
    uv run python scripts/performance_qualification.py

    # Bind local security and optional database preflight evidence to the
    # exact wheel/sdist produced from this tree. Public vulnerability lookup
    # and an independent penetration test remain explicit external gates.
    rm -f -- dist/sift-*.whl dist/sift-*.tar.gz
    uv build --wheel --sdist
    SECURITY_STATUS=0
    uv run python scripts/security_qualification.py --run-static \
        || SECURITY_STATUS=$?
    [[ "$SECURITY_STATUS" == "1" ]] || {
        echo "Local security qualification failed unexpectedly (status $SECURITY_STATUS)." >&2
        exit 1
    }
    uv run python -c '
import json
from pathlib import Path
r = json.loads(Path("dist/security/qualification.json").read_text())
allowed = {
    "dependency_scan_not_clear",
    "independent_third_party_penetration_test_not_supplied",
}
assert r["qualification_binding"]["release"]["status"] == "ready", r["blockers"]
assert r["secret_scan"]["status"] == "pass", r["secret_scan"]
assert r["static_analysis"]["status"] == "pass", r["static_analysis"]
assert r["bandit_static_analysis"]["status"] == "pass", r["bandit_static_analysis"]
assert r["dependency_scan"]["status"] == "not_run", r["dependency_scan"]
assert r["independent_penetration_test"]["status"] == "external_required", r["independent_penetration_test"]
assert set(r["blockers"]) == allowed, r["blockers"]
'
    uv run python scripts/database_qualification.py --preflight \
        > dist/database-preflight.json
    uv run python -c '
import json
from pathlib import Path
r = json.loads(Path("dist/database-preflight.json").read_text())
assert r["product_release_blocking"] is False
assert r["qualification_context_ready"] is True, r["qualification_context"]
'
fi

if [[ "${SIFT_SKIP_FREEZE:-0}" != "1" ]]; then
    uv run pyinstaller packaging/sift.spec --clean --noconfirm
else
    test -x dist/sift/sift || {
        echo "SIFT_SKIP_FREEZE requires an existing frozen dist/sift/sift bundle." >&2
        exit 1
    }
fi
test -x dist/sift/sift
dist/sift/sift --platform-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 dist/sift/sift --integration-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 dist/sift/sift --format-check >/dev/null
xvfb-run -a dist/sift/sift --renderer-check >/dev/null
PYTHONDONTWRITEBYTECODE=1 dist/sift/sift --analysis-check >/dev/null
packaging/linux/qualify_credential_store.sh dist/sift/sift >/dev/null
uv run python packaging/verify_frozen_bundle.py dist/sift
dist/sift/sift --help >/dev/null
uv run python packaging/verify_linux_elf_dependencies.py dist/sift

# The verified frozen tree is now self-contained. Reclaim build-only copies
# before staging and compressing the release archive; retaining all three can
# exhaust otherwise adequate hosted runners at the peak of packaging.
rm -rf -- build packaging/vendor
uv cache clean

case "$BUILD_ARCH" in
    x86_64) MANIFEST_ARCH="x86_64" ;;
    aarch64|arm64) MANIFEST_ARCH="aarch64" ;;
    *) echo "Unsupported Linux release architecture: $BUILD_ARCH" >&2; exit 1 ;;
esac
VERSION="$(uv run python -c 'try:
 import tomllib
except ModuleNotFoundError:
 import tomli as tomllib
print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
STAGE_PARENT="dist/.Sift-Linux-${MANIFEST_ARCH}.staging"
STAGE_ROOT="$STAGE_PARENT/Sift"
rm -rf -- "$STAGE_PARENT"
mkdir -p \
    "$STAGE_ROOT/share/applications" \
    "$STAGE_ROOT/share/metainfo" \
    "$STAGE_ROOT/share/icons"
# The frozen tree has already passed all source-independent runtime and ELF
# checks above. Move it into the release stage instead of holding a redundant
# multi-gigabyte copy throughout packaging.
mv dist/sift "$STAGE_ROOT/app"
cp packaging/linux/org.sapieninstitute.sift.desktop.in \
    "$STAGE_ROOT/share/applications/"
cp packaging/linux/org.sapieninstitute.sift.metainfo.xml \
    "$STAGE_ROOT/share/metainfo/"
cp -R packaging/linux/icons/hicolor "$STAGE_ROOT/share/icons/"
cp packaging/linux/install.sh packaging/linux/uninstall.sh \
    packaging/linux/prepare_ubuntu_host.sh packaging/linux/INSTALL.txt "$STAGE_ROOT/"
cp LICENSE "$STAGE_ROOT/LICENSE.txt"
chmod +x "$STAGE_ROOT/install.sh" "$STAGE_ROOT/uninstall.sh"
chmod +x "$STAGE_ROOT/prepare_ubuntu_host.sh"
PYTHONPATH=src uv run python packaging/write_package_metadata.py \
    "$STAGE_ROOT/release-metadata.json" \
    --version "$VERSION" --platform linux --architecture "$MANIFEST_ARCH"

# Exercise the per-user installer in an isolated home.  This proves the
# desktop entry has a real absolute executable, icons land at every declared
# size, the command launcher resolves, and uninstall removes only app files.
INSTALL_TEST_HOME="$(mktemp -d)"
cleanup_linux_stage() {
    rm -rf -- "$INSTALL_TEST_HOME" "$STAGE_PARENT"
}
trap cleanup_linux_stage EXIT INT TERM
HOME="$INSTALL_TEST_HOME" \
XDG_DATA_HOME="$INSTALL_TEST_HOME/.local/share" \
XDG_BIN_HOME="$INSTALL_TEST_HOME/.local/bin" \
    "$STAGE_ROOT/install.sh" >/dev/null
test -x "$INSTALL_TEST_HOME/.local/bin/sift"
test -x "$INSTALL_TEST_HOME/.local/share/sift/uninstall.sh"
test -f "$INSTALL_TEST_HOME/.local/share/sift/INSTALL.txt"
test -f "$INSTALL_TEST_HOME/.local/share/sift/LICENSE.txt"
! grep -q '__SIFT_EXECUTABLE__' \
    "$INSTALL_TEST_HOME/.local/share/applications/org.sapieninstitute.sift.desktop"
desktop-file-validate \
    "$INSTALL_TEST_HOME/.local/share/applications/org.sapieninstitute.sift.desktop"
appstreamcli validate --no-net \
    "$INSTALL_TEST_HOME/.local/share/metainfo/org.sapieninstitute.sift.metainfo.xml"
HOME="$INSTALL_TEST_HOME" \
XDG_DATA_HOME="$INSTALL_TEST_HOME/.local/share" \
XDG_BIN_HOME="$INSTALL_TEST_HOME/.local/bin" \
    "$INSTALL_TEST_HOME/.local/share/sift/uninstall.sh" >/dev/null
test ! -e "$INSTALL_TEST_HOME/.local/bin/sift"

ARCHIVE="dist/Sift-Linux-${MANIFEST_ARCH}.tar.gz"
rm -f "$ARCHIVE" "$ARCHIVE.sha256" "$ARCHIVE.sbom.cdx.json" "$ARCHIVE.sig.json"
tar -C "$STAGE_PARENT" -czf "$ARCHIVE" Sift

# The archive is now the authoritative input to the second lifecycle test.
# Release the staging copy before extracting the authoritative archive again.
rm -rf -- "$STAGE_PARENT" dist/sift
"$REPO_ROOT/packaging/qualify_linux_install.sh" "$ARCHIVE"

ARCHIVE_NAME="$(basename "$ARCHIVE")"
ARCHIVE_HASH="$(sha256sum "$ARCHIVE" | awk '{print $1}')"
printf '%s  %s\n' "$ARCHIVE_HASH" "$ARCHIVE_NAME" > "$ARCHIVE.sha256"
PYTHONPATH=src uv run python -m sift.release_manifest sbom \
    "$ARCHIVE" "$ARCHIVE.sbom.cdx.json" --version "$VERSION"
PYTHONPATH=src uv run python -m sift.release_manifest verify-sbom \
    "$ARCHIVE" "$ARCHIVE.sbom.cdx.json"
test -s "$ARCHIVE.sha256"
test -s "$ARCHIVE.sbom.cdx.json"

if [[ "$RELEASE_MODE" == "production" ]]; then
    SIGNED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    PYTHONPATH=src uv run python -m sift.release_manifest sign-file \
        "$ARCHIVE" "$ARCHIVE.sig.json" \
        --version "$VERSION" --channel "$RELEASE_CHANNEL" \
        --signed-at "$SIGNED_AT" --key-id "$SIFT_RELEASE_KEY_ID"
    test -s "$ARCHIVE.sig.json"
    echo "Built, checksummed, SBOM-recorded, and signed: $ARCHIVE"
else
    echo "Built, checksummed, and SBOM-recorded (development, unsigned): $ARCHIVE"
fi

rm -rf -- "$INSTALL_TEST_HOME" "$STAGE_PARENT"
trap - EXIT INT TERM
