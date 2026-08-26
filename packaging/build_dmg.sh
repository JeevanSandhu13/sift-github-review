#!/usr/bin/env bash
#
# Build Sift.dmg — a drag-to-/Applications installer for Sift.app.
#
# Requires `dist/Sift.app` to already exist (run build_app.sh first
# if needed). Produces `dist/Sift.dmg`.
#
# Structure the .dmg presents when mounted:
#   Sift.dmg/
#     Sift.app         # drag this...
#     Applications -> /Applications  # ...into here
#
# The mounted window is branded from the same canonical artwork as the app:
# a Sift volume icon, a quiet install background, and fixed app/Applications
# positions.  No third-party DMG builder is required.
#
# Gatekeeper / signing:
#   - If $SIFT_SIGN_IDENTITY is set, the .dmg itself is signed too
#     (the .app inside should already be signed by build_app.sh).
#   - If $SIFT_NOTARIZE_PROFILE is also set (a keychain profile name
#     stored via `xcrun notarytool store-credentials`), the .dmg is
#     submitted to Apple's notary service and the resulting ticket is
#     stapled into the .dmg so first-launch works offline.
#   - If neither is set, the .dmg contains an unsigned .app and
#     researchers need the right-click → Open workaround documented in
#     docs/install.md.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DIST_DIR="$REPO_ROOT/dist"
APP_BUNDLE="$DIST_DIR/Sift.app"
DMG_OUT="$DIST_DIR/Sift.dmg"
STAGING="$DIST_DIR/dmg-staging"
RW_DMG="$DIST_DIR/.Sift-rw.$$.dmg"
MOUNT_POINT=""

cleanup_dmg_build() {
    if [[ -n "$MOUNT_POINT" && -d "$MOUNT_POINT" ]]; then
        /usr/bin/hdiutil detach "$MOUNT_POINT" -quiet >/dev/null 2>&1 || true
    fi
    rm -rf -- "$STAGING"
    rm -f -- "$RW_DMG"
}
trap cleanup_dmg_build EXIT INT TERM

if [[ ! -d "$APP_BUNDLE" ]]; then
    echo "Sift.app not found at $APP_BUNDLE — run build_app.sh first." >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    BRAND_PYTHON=("$(command -v uv)" run python)
elif [[ -x "$REPO_ROOT/.appenv/bin/python" ]]; then
    BRAND_PYTHON=("$REPO_ROOT/.appenv/bin/python")
else
    echo "uv is required to verify native brand assets." >&2
    exit 1
fi
"${BRAND_PYTHON[@]}" packaging/generate_brand_assets.py --check

# When the caller asks for a signed and/or notarized .dmg, the .app
# inside MUST already be signed. This script consumes a pre-built
# dist/Sift.app, so an easy mistake is to build the app unsigned, then
# rerun only this script with the signing variables set — Apple's
# notary service may even accept the submission, leaving you with a
# signed .dmg that wraps an unsigned app. Catch that here, before
# staging, so the failure is immediate and local. ``codesign --verify
# --deep --strict`` walks every nested Mach-O too, so a partial sign
# (where the bundle is signed but a nested binary isn't) is also
# rejected at this gate.
if [[ -n "${SIFT_SIGN_IDENTITY:-}" ]] || [[ -n "${SIFT_NOTARIZE_PROFILE:-}" ]]; then
    echo "==> Verifying $APP_BUNDLE is signed"
    if ! /usr/bin/codesign --verify --deep --strict "$APP_BUNDLE" 2>/dev/null; then
        echo "ERROR: $APP_BUNDLE is not signed (or its signature is invalid)." >&2
        echo "       Run build_app.sh with SIFT_SIGN_IDENTITY set, e.g.:" >&2
        echo "         SIFT_SIGN_IDENTITY=\"Developer ID Application: ... (TEAMID)\" \\" >&2
        echo "             bash packaging/build_app.sh" >&2
        echo "       then rerun this script. Notarizing an unsigned .app produces" >&2
        echo "       a release artifact that fails Gatekeeper at first launch." >&2
        exit 1
    fi
fi

echo "==> Preparing staging directory"
rm -rf -- "$STAGING"
rm -f -- "$DMG_OUT" "$RW_DMG"
mkdir -p "$STAGING"

# Copy the .app (don't move — we want the original to stay for
# re-runs / manual testing). ``ditto`` instead of ``cp -R``: cp on
# macOS strips some extended attributes the codesign signature
# relies on, and the bundle we're staging is ALREADY signed
# (build_app.sh ran codesign before this script). A corrupted
# signature passes spctl at the build host but fails at launch on
# stricter machines — silent ship-it bug. ``ditto`` preserves the
# attribute set codesign cares about. release.sh and build_app.sh
# already use ``ditto`` for the same reason; this closes the last
# gap.
/usr/bin/ditto "$APP_BUNDLE" "$STAGING/$(basename "$APP_BUNDLE")"

# The drag-to-Applications symlink. macOS Finder treats this specially
# when the .dmg is opened: shows as a real Applications folder the
# user can drop the app onto.
ln -s /Applications "$STAGING/Applications"

mkdir -p "$STAGING/.background"
cp "$REPO_ROOT/packaging/macos/installer-background.png" \
    "$STAGING/.background/installer-background.png"
cp "$REPO_ROOT/packaging/Sift.icns" "$STAGING/.VolumeIcon.icns"

echo "==> Building branded installer image"
# Create a temporary writable HFS+ image so Finder can store its icon layout
# and background in .DS_Store.  The final conversion is compressed/read-only.
/usr/bin/hdiutil create \
    -volname "Sift" \
    -srcfolder "$STAGING" \
    -fs HFS+ \
    -format UDRW \
    -ov \
    "$RW_DMG" >/dev/null

ATTACH_OUTPUT="$(/usr/bin/hdiutil attach "$RW_DMG" -readwrite -noverify -noautoopen)"
MOUNT_POINT="$(printf '%s\n' "$ATTACH_OUTPUT" | awk -F'\t' '/Apple_HFS/{print $NF; exit}')"
if [[ -z "$MOUNT_POINT" || ! -d "$MOUNT_POINT" ]]; then
    echo "Could not mount the writable Sift installer image." >&2
    exit 1
fi
VOLUME_NAME="$(basename "$MOUNT_POINT")"

/usr/bin/SetFile -a V "$MOUNT_POINT/.background" "$MOUNT_POINT/.VolumeIcon.icns"
/usr/bin/SetFile -a C "$MOUNT_POINT"

if ! /usr/bin/osascript - "$VOLUME_NAME" <<'APPLESCRIPT'
on run argv
    set volumeName to item 1 of argv
    tell application "Finder"
        tell disk volumeName
            open
            -- Finder 26 can return a transient -10006 while it creates the
            -- disk window. Resolve the window first, then keep unsupported
            -- chrome properties from aborting the critical icon/background
            -- layout that follows.
            set installerWindow to missing value
            repeat 20 times
                try
                    set installerWindow to container window
                    exit repeat
                on error
                    delay 0.25
                end try
            end repeat
            if installerWindow is missing value then error "Finder did not create the Sift disk window"
            set current view of installerWindow to icon view
            try
                set toolbar visible of installerWindow to false
            end try
            try
                set statusbar visible of installerWindow to false
            end try
            try
                set pathbar visible of installerWindow to false
            end try
            set bounds of installerWindow to {100, 100, 760, 500}
            set viewOptions to icon view options of installerWindow
            set arrangement of viewOptions to not arranged
            set icon size of viewOptions to 96
            set text size of viewOptions to 13
            set background picture of viewOptions to file ".background:installer-background.png"
            set position of item "Sift.app" of container window to {150, 205}
            set position of item "Applications" of container window to {510, 205}
            -- Closing the Finder window persists .DS_Store. Avoid Finder's
            -- recursive update command: on a multi-gigabyte scientific app it
            -- can exceed AppleEvent's timeout even though the layout is ready.
            delay 1
            close installerWindow
        end tell
    end tell
end run
APPLESCRIPT
then
    if [[ "${SIFT_RELEASE_MODE:-development}" == "production" ]]; then
        echo "Finder could not write the branded DMG layout; refusing a production artifact." >&2
        exit 1
    fi
    echo "WARNING: Finder could not save the custom DMG layout; the app and volume icons remain branded." >&2
fi

/bin/sync
/usr/bin/hdiutil detach "$MOUNT_POINT" -quiet
MOUNT_POINT=""
/usr/bin/hdiutil convert "$RW_DMG" -format UDZO -o "$DMG_OUT" >/dev/null
rm -f -- "$RW_DMG"
rm -rf -- "$STAGING"

if [[ ! -s "$DMG_OUT" ]]; then
    echo "Compressed Sift installer image was not created." >&2
    exit 1
fi

if [[ -n "${SIFT_SIGN_IDENTITY:-}" ]]; then
    echo "==> Signing $DMG_OUT"
    /usr/bin/codesign --force --timestamp \
        --sign "$SIFT_SIGN_IDENTITY" \
        "$DMG_OUT"
else
    echo "==> Skipping DMG signing (SIFT_SIGN_IDENTITY unset)"
fi

if [[ -n "${SIFT_NOTARIZE_PROFILE:-}" ]]; then
    if [[ -z "${SIFT_SIGN_IDENTITY:-}" ]]; then
        echo "SIFT_NOTARIZE_PROFILE is set but SIFT_SIGN_IDENTITY is not." >&2
        echo "Notarization requires the .dmg (and its .app) to be signed first." >&2
        exit 1
    fi

    echo "==> Submitting to Apple notary service"
    # We poll explicitly instead of using ``--wait``. ``--wait`` is a
    # tight in-process loop with no timeout; when Apple's notary queue
    # stalls (it does, occasionally — outage, accumulated backpressure,
    # something on their side), the loop hangs the build for hours
    # without ever giving up. A previous release of this DMG sat in
    # --wait for 26 hours before someone noticed. Manual polling lets
    # us bail with a recovery hint that lets the user resume from the
    # staple step once Apple eventually finishes.
    SUBMIT_OUT="$(xcrun notarytool submit "$DMG_OUT" \
                    --keychain-profile "$SIFT_NOTARIZE_PROFILE")"
    echo "$SUBMIT_OUT"
    SUBMISSION_ID="$(echo "$SUBMIT_OUT" \
                     | awk -F': ' '/^[[:space:]]*id:/{print $2; exit}')"
    if [[ -z "$SUBMISSION_ID" ]]; then
        echo "Could not parse submission id from notarytool output." >&2
        exit 1
    fi

    # Poll up to 30 minutes (default; both intervals env-overridable
    # so CI can shorten on fast notarization paths and lengthen on
    # known-slow days). Apple's median is ~5min; the default absorbs
    # routine backlogs while still surfacing genuinely-stuck
    # submissions in a workday rather than a workweek.
    POLL_INTERVAL="${SIFT_NOTARIZE_POLL_INTERVAL:-30}"
    POLL_TIMEOUT="${SIFT_NOTARIZE_POLL_TIMEOUT:-1800}"
    # Validate the env-overridable knobs. ``POLL_INTERVAL=0`` would
    # tight-loop forever: ``sleep 0`` returns immediately and
    # ``SECONDS_WAITED + 0`` stays at 0, so the timeout check never
    # trips and we hammer Apple's API until the user kills the
    # build. A non-positive interval or non-integer value is almost
    # always a typo in the caller's environment; refusing the build
    # is friendlier than the alternative.
    if ! [[ "$POLL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
        echo "SIFT_NOTARIZE_POLL_INTERVAL must be a positive integer; got '${POLL_INTERVAL}'" >&2
        exit 1
    fi
    if ! [[ "$POLL_TIMEOUT" =~ ^[1-9][0-9]*$ ]]; then
        echo "SIFT_NOTARIZE_POLL_TIMEOUT must be a positive integer; got '${POLL_TIMEOUT}'" >&2
        exit 1
    fi
    SECONDS_WAITED=0
    STATUS=""
    # Poll BEFORE the first sleep — Apple sometimes flips a tiny
    # submission to Accepted within seconds, and a leading 30s wait
    # was a guaranteed floor on every release. Layout is "check,
    # then sleep, then check again" so a fast accept exits in <1s,
    # a typical accept exits at the first natural-cadence interval,
    # and slow ones still cap at POLL_TIMEOUT.
    while true; do
        # ``--output-format json`` gives a stable shape regardless
        # of Apple's free-form column layout. The previous awk
        # ``status:`` parser was one Apple cosmetic change away from
        # silently returning empty status forever (which would loop
        # the script until POLL_TIMEOUT). The python one-liner is
        # vendored here so the script keeps its single-bash-file
        # contract — bringing in jq would add a dependency that
        # release machines may not have.
        STATUS_JSON="$(xcrun notarytool info "$SUBMISSION_ID" \
                        --keychain-profile "$SIFT_NOTARIZE_PROFILE" \
                        --output-format json 2>/dev/null || true)"
        STATUS="$(printf '%s' "$STATUS_JSON" \
                  | /usr/bin/python3 -c 'import json,sys
try:
  print(json.loads(sys.stdin.read()).get("status", ""))
except Exception:
  pass' 2>/dev/null)"
        echo "    [${SECONDS_WAITED}s] status: ${STATUS:-unknown}"
        case "$STATUS" in
            Accepted)
                break
                ;;
            Invalid|Rejected)
                # ``Invalid`` is Apple's documented terminal-failure
                # status. ``Rejected`` is included defensively in case
                # Apple introduces a synonym; treating it as failure
                # is correct either way.
                echo >&2
                echo "Notarization rejected (status=$STATUS). Submission log:" >&2
                xcrun notarytool log "$SUBMISSION_ID" \
                    --keychain-profile "$SIFT_NOTARIZE_PROFILE" >&2 || true
                exit 1
                ;;
            "In Progress"|"")
                # Still working / transient query failure. Loop.
                ;;
            *)
                # Unknown terminal status — treat as failure rather
                # than spinning until POLL_TIMEOUT. Apple may add new
                # statuses; we'd rather fail loudly than silently
                # consume a 30-min timeout.
                echo >&2
                echo "Notarization returned unexpected status: $STATUS" >&2
                echo "Submission id: $SUBMISSION_ID" >&2
                echo "Inspect with:" >&2
                echo "  xcrun notarytool info $SUBMISSION_ID --keychain-profile $SIFT_NOTARIZE_PROFILE" >&2
                exit 1
                ;;
        esac
        if [[ "$SECONDS_WAITED" -ge "$POLL_TIMEOUT" ]]; then
            break
        fi
        sleep "$POLL_INTERVAL"
        SECONDS_WAITED=$((SECONDS_WAITED + POLL_INTERVAL))
    done

    if [[ "$STATUS" != "Accepted" ]]; then
        # Apple is still chewing on it. The artifact and the
        # submission id are both salvageable — the user just runs the
        # staple step manually once `notarytool info` flips to Accepted.
        cat >&2 <<EOF

Notarization still pending after ${POLL_TIMEOUT}s. This is recoverable;
we just stopped watching. Apple's queue may be backlogged
(check https://developer.apple.com/system-status/).

Submission id: $SUBMISSION_ID

To finish the release once Apple flips it to Accepted, run:

    xcrun notarytool info $SUBMISSION_ID --keychain-profile $SIFT_NOTARIZE_PROFILE
    # ...wait until status: Accepted, then:
    xcrun stapler staple "$DMG_OUT"
    xcrun stapler validate "$DMG_OUT"

EOF
        exit 1
    fi

    echo "==> Stapling notarization ticket"
    xcrun stapler staple "$DMG_OUT"
    xcrun stapler validate "$DMG_OUT"
else
    echo "==> Skipping notarization (SIFT_NOTARIZE_PROFILE unset)"
fi

trap - EXIT INT TERM

echo
echo "Built: $DMG_OUT"
echo "Size:  $(du -sh "$DMG_OUT" | cut -f1)"
echo
echo "Qualification and installation steps: docs/install.md"
