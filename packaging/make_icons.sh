#!/usr/bin/env bash
# Regenerate every native icon/installer image from icon-source.png.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if command -v uv >/dev/null 2>&1; then
    exec uv run python packaging/generate_brand_assets.py
fi
if [[ -x "$REPO_ROOT/.appenv/bin/python" ]]; then
    exec "$REPO_ROOT/.appenv/bin/python" packaging/generate_brand_assets.py
fi
echo "uv is required to regenerate Sift brand assets." >&2
exit 1
