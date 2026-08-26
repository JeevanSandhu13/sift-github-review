#!/usr/bin/env bash
#
# Compatibility wrapper: regenerate the complete native Sift icon set.
#
# build_app.sh expects Sift.icns to already exist; we keep it
# committed so the build is fully self-contained on a fresh checkout
# with no extra prerequisites. Run this script only when the source
# logo changes.
#
# The canonical command is packaging/make_icons.sh.  Keeping this name avoids
# breaking existing maintainer workflows while ensuring no platform drifts.

exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/make_icons.sh"
