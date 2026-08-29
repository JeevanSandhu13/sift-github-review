#!/usr/bin/env bash
#
# Release pipeline for Sift.
#
# Bundles the per-release flow into one command:
#   1. Pre-flight: verify signing identity + notary profile
#   2. Build dist/Sift.app   (signed via build_app.sh)
#   3. Build dist/Sift.dmg   (signed + notarized + stapled via build_dmg.sh)
#   4. Verify the artifacts pass codesign + spctl + stapler
#   5. Install to /Applications/Sift.app (with quarantine flag stripped)
#
# Usage (run from anywhere — script resolves the repo via its own path):
#   bash packaging/release.sh                # full pipeline + install
#   bash packaging/release.sh --no-install   # build only
#   bash packaging/release.sh --app-only     # skip the .dmg + notarization
#   bash packaging/release.sh --check-only   # run pre-flight, exit
#   bash packaging/release.sh --yes          # auto-confirm prompts (CI use)
#
# Required env (drop these in ~/.zshrc to make them stick):
#   SIFT_SIGN_IDENTITY     e.g. "Developer ID Application: Your Name (TEAMID)"
#   SIFT_NOTARIZE_PROFILE  notarytool keychain-profile name from
#                          `xcrun notarytool store-credentials`
#
# Optional env:
#   SIFT_RELEASE_YES=1     equivalent to --yes; auto-accepts the
#                          off-main-branch and behind-origin prompts so
#                          the script can run unattended (CI, cron).
#
# Why a wrapper at all:
#   - build_app.sh + build_dmg.sh skip signing silently when the env
#     vars are unset, producing a Gatekeeper-rejected .dmg with no error
#     message. The pre-flight catches this before the slow build starts.
#   - The release is built directly from this source tree and does not
#     consult or modify repository metadata.
#   - Apple's notary service occasionally hangs (a previous release sat
#     in --wait for 26 hours). The DMG script now caps polling at 30
#     minutes; this wrapper still verifies that stapling
#     succeeded so a partial run doesn't go unnoticed.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

INSTALL=true
CHECK_ONLY=false
APP_ONLY=false
# Default --yes from the env so this script can run unattended (CI,
# cron). Each interactive prompt below honours this flag instead of
# blocking forever on a non-TTY ``read``.
case "${SIFT_RELEASE_YES:-}" in
    1|true|yes|YES) ASSUME_YES=true ;;
    *)              ASSUME_YES=false ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-install)  INSTALL=false; shift ;;
        --check-only)  CHECK_ONLY=true; shift ;;
        --app-only)    APP_ONLY=true; shift ;;
        --yes|-y)      ASSUME_YES=true; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

# Confirm prompts honour --yes / SIFT_RELEASE_YES. When neither is set
# AND stdin isn't a TTY, refuse instead of blocking forever — CI logs
# would otherwise stall silently on a never-arriving newline.
# Both regexes match either the single-letter form (y / Y / n / N) or
# the full word (yes / YES / Yes, no / NO / No). Empty input falls
# through to whichever default the helper enforces. Explicit "no"
# typed at a default-yes prompt should mean no, not "garbage → default
# → yes" which is what an over-strict ^[Nn]$ would do.
_RE_YES='^[Yy]([Ee][Ss])?$'
_RE_NO='^[Nn]([Oo])?$'

confirm() {
    local prompt="$1"
    if [[ "$ASSUME_YES" == "true" ]]; then
        echo "    $prompt [y/N] y  (auto)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        echo "  ✗ $prompt — non-interactive shell. Re-run with --yes" >&2
        echo "    or set SIFT_RELEASE_YES=1 to bypass." >&2
        return 1
    fi
    local ans=""
    echo -n "    $prompt [y/N] "
    read -r ans
    [[ "$ans" =~ $_RE_YES ]]
}

# Default-yes counterpart of ``confirm``. Use for prompts where the
# obvious / expected answer is yes (e.g. "pull now?") so the user can
# accept by just hitting Enter. Non-TTY runs silently accept — CI's
# explicit choice is to not interact, and a default-yes prompt's whole
# point is that yes is the safe path.
confirm_yes() {
    local prompt="$1"
    if [[ "$ASSUME_YES" == "true" ]]; then
        echo "    $prompt [Y/n] y  (auto)"
        return 0
    fi
    if [[ ! -t 0 ]]; then
        return 0
    fi
    local ans=""
    echo -n "    $prompt [Y/n] "
    read -r ans
    [[ ! "$ans" =~ $_RE_NO ]]
}

# ── Pre-flight ────────────────────────────────────────────────────────

echo "==> Pre-flight"

RELEASE_CHANNEL="${SIFT_RELEASE_CHANNEL:-stable}"
[[ "$RELEASE_CHANNEL" == "stable" || "$RELEASE_CHANNEL" == "beta" ]] \
    || { echo "  ✗ SIFT_RELEASE_CHANNEL must be stable or beta." >&2; exit 1; }
if [[ -z "${SIFT_RELEASE_PRIVATE_KEY_B64:-}" || -z "${SIFT_RELEASE_KEY_ID:-}" ]]; then
    cat >&2 <<EOF
  ✗ Cross-platform release signing is not configured.
    Set SIFT_RELEASE_PRIVATE_KEY_B64 and SIFT_RELEASE_KEY_ID so the DMG and
    canonical all-platform manifest can be verified offline.
EOF
    exit 1
fi

# The app must know where to obtain the canonical signed manifest and which
# public release keys to trust before a production binary is created.  This is
# checked here as well as in build_app.sh so --check-only is a real release
# preflight, not a partial signing check.
if ! PYTHONPATH=src uv run python -c \
    'from sift.update_config import load_update_policy; p = load_update_policy(); raise SystemExit(0 if p.get("configured") is True else 1)'; then
    cat >&2 <<EOF
  ✗ Production update policy is not configured.
    Run packaging/configure_update_policy.py with the reviewed HTTPS manifest
    URL and public release trust store before release qualification.
EOF
    exit 1
fi
echo "  ✓ Signed update policy configured"

if [[ -z "${SIFT_SIGN_IDENTITY:-}" ]]; then
    cat >&2 <<EOF
  ✗ SIFT_SIGN_IDENTITY is not set.
    Add this to ~/.zshrc (or set in this shell) and re-run:
      export SIFT_SIGN_IDENTITY="Developer ID Application: Your Name (TEAMID)"
EOF
    exit 1
fi

# The cert must actually exist in the login keychain. Match by the
# (TEAMID) suffix so trailing whitespace in the env value doesn't trip
# us up.
TEAM_FRAGMENT="$(echo "$SIFT_SIGN_IDENTITY" \
                 | grep -oE '\([0-9A-Z]{10}\)' | head -1 || true)"
if [[ -z "$TEAM_FRAGMENT" ]] \
        || ! /usr/bin/security find-identity -v -p codesigning 2>/dev/null \
                | grep -q "$TEAM_FRAGMENT"; then
    cat >&2 <<EOF
  ✗ Signing cert for $SIFT_SIGN_IDENTITY not found in login keychain.
    Run: security find-identity -v -p codesigning
    to see what's actually installed.
EOF
    exit 1
fi
echo "  ✓ Signing identity: $SIFT_SIGN_IDENTITY"

if [[ "$APP_ONLY" == "false" ]]; then
    if [[ -z "${SIFT_NOTARIZE_PROFILE:-}" ]]; then
        cat >&2 <<EOF
  ✗ SIFT_NOTARIZE_PROFILE is not set (or pass --app-only to skip).
    Create one once with:
      xcrun notarytool store-credentials <profile-name> \\
          --apple-id you@example.com --team-id TEAMID \\
          --password APP-SPECIFIC-PASSWORD
EOF
        exit 1
    fi
    if ! xcrun notarytool history \
            --keychain-profile "$SIFT_NOTARIZE_PROFILE" >/dev/null 2>&1; then
        echo "  ✗ notarytool profile '$SIFT_NOTARIZE_PROFILE' is not stored." >&2
        exit 1
    fi
    echo "  ✓ Notary profile: $SIFT_NOTARIZE_PROFILE"
fi

# This source tree is intentionally releaseable without repository
# metadata. The artifact is validated from its actual files and frozen
# executable; release must never fetch, pull, or require commit state.
echo "  ✓ Source-tree release mode (repository metadata not consulted)"

if [[ "$CHECK_ONLY" == "true" ]]; then
    echo
    echo "Pre-flight passed. (--check-only: skipping build.)"
    exit 0
fi

# ── Build ─────────────────────────────────────────────────────────────

echo
echo "==> Building Sift.app"
SIFT_RELEASE_MODE=production bash "$REPO_ROOT/packaging/build_app.sh"

if [[ "$APP_ONLY" == "false" ]]; then
    echo
    echo "==> Building Sift.dmg"
    SIFT_RELEASE_MODE=production bash "$REPO_ROOT/packaging/build_dmg.sh"
fi

# ── Verify ────────────────────────────────────────────────────────────

echo
echo "==> Verifying artifacts"

APP="$REPO_ROOT/dist/Sift.app"
DMG="$REPO_ROOT/dist/Sift.dmg"

# codesign --verify walks every nested Mach-O when --deep is set, so a
# partial sign (where the bundle is signed but a nested binary isn't)
# fails this check. spctl --assess is what Gatekeeper itself runs at
# launch, so a pass here is a strong signal that end users won't see
# the "cannot be opened" dialog.
/usr/bin/codesign --verify --deep --strict "$APP"
echo "  ✓ codesign verify"

# Architecture check — the nested running binary MUST be arm64. The
# spec pins target_arch="arm64", but a future edit (or a stray
# Rosetta-installed PyInstaller) could silently revert that and ship
# an x86_64 bundle that runs fine on the build machine but triggers
# the Rosetta-deprecation banner on every user's Mac. Fail closed
# here so the bad artifact never reaches the install / DMG step.
NESTED_BIN="$APP/Contents/Resources/sift/sift"
APP_ARCHS="$(/usr/bin/lipo -archs "$NESTED_BIN" 2>/dev/null || true)"
if [[ "$APP_ARCHS" != "arm64" ]]; then
    echo "  ✗ Wrong architecture: $NESTED_BIN is '$APP_ARCHS' (want 'arm64')." >&2
    echo "    Likely cause: PyInstaller ran under an Intel Python." >&2
    echo "    Check: file \$(which python3); arch" >&2
    exit 1
fi
echo "  ✓ Architecture: arm64"

# Version-skew check: ``pyproject.toml``'s ``project.version`` must
# match the bundle's ``CFBundleShortVersionString``. ``build_app.sh``
# now derives the plist value from pyproject (so the two CAN'T diverge
# unless the derive step silently failed), but a release-time
# regression check pins the invariant — and catches a stale
# pre-derive bundle that wasn't rebuilt before this release pass.
PLIST_VERSION="$(
    /usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' \
        "$APP/Contents/Info.plist" 2>/dev/null || true
)"
# ``tomllib`` is stdlib only in Python 3.11+; macOS ships with
# 3.9 (Big Sur) or 3.10 (Ventura) at /usr/bin/python3, both of
# which raise ``ModuleNotFoundError`` on import. Prefer ``uv run
# python`` (the project's pinned interpreter, 3.11+ by default)
# and fall back to ``awk`` on pyproject.toml directly when uv is
# absent — the version line is unambiguous (``version = "X.Y.Z"``
# at the top of the [project] table) and that path avoids the
# Python-version trap entirely.
if command -v uv >/dev/null 2>&1; then
    PYPROJECT_VERSION="$(uv run python -c '
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
with open("pyproject.toml", "rb") as f:
    print(tomllib.load(f)["project"]["version"])
')"
else
    PYPROJECT_VERSION="$(
        /usr/bin/awk -F'"' '/^version[[:space:]]*=/ {print $2; exit}' pyproject.toml
    )"
fi
if [[ -z "$PLIST_VERSION" || -z "$PYPROJECT_VERSION" ]]; then
    echo "  ✗ Could not read both versions (plist='$PLIST_VERSION', pyproject='$PYPROJECT_VERSION')." >&2
    exit 1
fi
if [[ "$PLIST_VERSION" != "$PYPROJECT_VERSION" ]]; then
    echo "  ✗ Version skew: Info.plist='$PLIST_VERSION' but pyproject.toml='$PYPROJECT_VERSION'." >&2
    echo "    The bundle was likely built before build_app.sh's derive-from-pyproject" >&2
    echo "    step landed (or that step failed silently). Rebuild and re-run." >&2
    exit 1
fi
echo "  ✓ Version: $PLIST_VERSION (matches pyproject.toml)"

ASSESS="$( /usr/sbin/spctl --assess --verbose=2 --type execute "$APP" 2>&1 || true )"
if echo "$ASSESS" | grep -qE "accepted"; then
    echo "  ✓ spctl assess: $(echo "$ASSESS" | tr '\n' ' ' | sed 's/  */ /g')"
else
    # Gatekeeper would block this on a fresh Mac. Shipping anyway
    # defeats the wrapper's release-gating contract — the whole reason
    # to run release.sh instead of build_app.sh + build_dmg.sh directly
    # is so a rejected artifact never reaches the install / "Done" path.
    echo "  ✗ spctl assess REJECTED the .app — Gatekeeper would block this:" >&2
    echo "    $ASSESS" >&2
    echo
    echo "  Common causes:" >&2
    echo "    • Signature didn't apply (SIFT_SIGN_IDENTITY unset on the" >&2
    echo "      build_app.sh run)" >&2
    echo "    • Notarization hasn't completed for the bundled .app" >&2
    echo "    • The signing identity has expired or been revoked" >&2
    exit 1
fi

if [[ "$APP_ONLY" == "false" ]] && [[ -f "$DMG" ]]; then
    if xcrun stapler validate "$DMG" >/dev/null 2>&1; then
        echo "  ✓ DMG notarization stapled"
    else
        # build_dmg.sh staples after notarytool returns Accepted, so a
        # missing staple here means either the notarize step never ran
        # (env unset) or it timed out / errored. Either way the .dmg
        # isn't ready for distribution — fail rather than print Done.
        echo "  ✗ DMG is NOT stapled — first launch on a fresh Mac will hit Gatekeeper." >&2
        echo
        echo "  Recovery if Apple is just slow:" >&2
        echo "    xcrun notarytool history --keychain-profile \"\$SIFT_NOTARIZE_PROFILE\"" >&2
        echo "    # …wait for status: Accepted, then:" >&2
        echo "    xcrun stapler staple \"$DMG\"" >&2
        echo "    xcrun stapler validate \"$DMG\"" >&2
        exit 1
    fi

    # Emit a SHA-256 sidecar file alongside the .dmg. Researchers and
    # downstream packagers (homebrew-cask, internal IT distribution,
    # mirrors) need a way to verify the binary they got matches what
    # we shipped. Apple's notarization signs the bundle but doesn't
    # publish a per-release fingerprint anyone can check from the
    # outside, and Gatekeeper only catches "Apple no longer trusts
    # this developer" — not "the .dmg was modified after we built
    # it." A SHA-256 in the same dist directory is the conventional
    # fix; ``shasum -a 256`` ships with macOS so there's no toolchain
    # cost. The ``.sha256`` file format mirrors what GitHub Releases
    # accepts so users can ``shasum -a 256 -c Sift.dmg.sha256`` after
    # downloading both. Recompute on every release so the file always
    # corresponds to the .dmg next to it.
    SHA256_FILE="$DMG.sha256"
    # ``shasum`` writes ``<hash>  <path>``; rewrite to use the
    # basename so verification works regardless of where the user
    # downloaded the artifacts to.
    DMG_BASENAME="$(basename "$DMG")"
    DMG_HASH="$(/usr/bin/shasum -a 256 "$DMG" | awk '{print $1}')"
    printf '%s  %s\n' "$DMG_HASH" "$DMG_BASENAME" > "$SHA256_FILE"
    echo "  ✓ DMG SHA-256 written to $(basename "$SHA256_FILE")"
    echo "    $DMG_HASH"

    PYTHONPATH=src uv run python -m sift.release_manifest sbom \
        "$DMG" "$DMG.sbom.cdx.json" --version "$PYPROJECT_VERSION"
    PYTHONPATH=src uv run python -m sift.release_manifest verify-sbom \
        "$DMG" "$DMG.sbom.cdx.json"
    SIGNED_AT="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    PYTHONPATH=src uv run python -m sift.release_manifest sign-file \
        "$DMG" "$DMG.sig.json" --version "$PYPROJECT_VERSION" \
        --channel "$RELEASE_CHANNEL" --signed-at "$SIGNED_AT" \
        --key-id "$SIFT_RELEASE_KEY_ID"
    test -s "$DMG.sbom.cdx.json"
    test -s "$DMG.sig.json"
    echo "  ✓ DMG CycloneDX SBOM and detached Ed25519 signature written"
fi

# ── Install ───────────────────────────────────────────────────────────

if [[ "$INSTALL" == "true" ]]; then
    echo
    echo "==> Installing to /Applications/Sift.app"
    # If Sift is currently running, replacing the .app underneath it
    # would leave the running process in a weird state and the next
    # launch could load mismatched resources. Quit it cleanly first.
    #
    # Match by the actual executable name. Even though the .app's
    # CFBundleExecutable is ``Sift``, the launcher script ``exec``s
    # the bundled PyInstaller binary at ``Contents/Resources/sift/sift``
    # — so the running process shows up as ``sift`` (lowercase) in
    # ps / pgrep. The previous ``pgrep -xq Sift`` never matched, and
    # the install path went straight to ``rm -rf /Applications/Sift.app``
    # against a live process. Both names are checked here belt-and-
    # suspenders so a future packaging change that drops the exec hand-
    # off doesn't silently re-open the bug.
    if pgrep -xq sift || pgrep -xq Sift; then
        echo "  Quitting running Sift.app first..."
        osascript -e 'tell application "Sift" to quit' 2>/dev/null \
            || pkill -x sift 2>/dev/null \
            || pkill -x Sift 2>/dev/null \
            || true
        # Loop briefly until the process actually exits — sleep 1 was
        # a guess that fails under load (heavy GC inside an Anthropic
        # SDK shutdown can take 2-3s on a busy laptop). Bail with a
        # clear error if it never quits, rather than overwriting the
        # bundle out from under it.
        for _ in 1 2 3 4 5; do
            if ! pgrep -xq sift && ! pgrep -xq Sift; then break; fi
            sleep 1
        done
        if pgrep -xq sift || pgrep -xq Sift; then
            echo "  ✗ Sift is still running. Quit it manually, then re-run." >&2
            exit 1
        fi
    fi
    rm -rf /Applications/Sift.app
    # Use ``ditto`` instead of ``cp -R``: ditto preserves resource
    # forks, ACLs, and especially extended attributes that the code
    # signature relies on. ``cp -R`` on macOS strips some xattrs in
    # certain configurations, breaking the signature in subtle ways
    # that pass spctl but fail at first launch on a stricter machine.
    /usr/bin/ditto "$APP" /Applications/Sift.app
    # Strip the quarantine attr in case the .app was tagged after a
    # download / move. macOS only assigns it when the file crosses an
    # internet trust boundary, so this is usually a no-op for local
    # builds — but covers the case where dist/ came from somewhere else.
    xattr -dr com.apple.quarantine /Applications/Sift.app 2>/dev/null || true
    echo "  ✓ Installed at /Applications/Sift.app"
    echo "    (run 'killall Dock Finder' if the cached Dock icon stays stale)"
fi

# ── Summary ───────────────────────────────────────────────────────────

echo
echo "Done."
echo "  App: $APP"
if [[ "$APP_ONLY" == "false" ]] && [[ -f "$DMG" ]]; then
    echo "  DMG: $DMG  ($(du -h "$DMG" | cut -f1))"
    if [[ -f "$DMG.sha256" ]]; then
        echo "  SHA: $DMG.sha256"
    fi
    [[ -f "$DMG.sbom.cdx.json" ]] && echo "  SBOM: $DMG.sbom.cdx.json"
    [[ -f "$DMG.sig.json" ]] && echo "  Signature: $DMG.sig.json"
fi
