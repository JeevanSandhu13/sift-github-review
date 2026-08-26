#!/usr/bin/env bash
# packaging/vendor_python.sh — build Sift's bundled Python analysis runtime.
#
# WHAT THIS PRODUCES
# -------------------
# A portable, redistributable CPython (via astral-sh/python-build-
# standalone, fetched through `uv python install`) with Sift's
# complete maintained analysis stack (data readers, numerical engines,
# estimators, geospatial tooling, and Bayesian tooling) installed into its
# OWN site-packages, laid out at:
#
#   packaging/vendor/python/bin/python3
#
# This is a SEPARATE interpreter from the one PyInstaller freezes
# Sift's own UI/agent code into. It exists purely so
# `sift.env_detect.find_bundled_python()` has something to discover
# and hand to the executor's sandbox when a researcher has no (or an
# incomplete) Python of their own on PATH — the same role R and
# Stata detection already play for those languages, except Python's
# ecosystem can be fully vendored with no commercial licensing complication,
# closing a real first-run friction gap the other two languages
# cannot close the same way.
#
# WHEN TO RUN THIS
# -----------------
# Before any RELEASE build: run this once, then
# `uv run pyinstaller packaging/sift.spec --clean --noconfirm`.
# `packaging/sift.spec`'s VENDOR_PYTHON_DATAS picks up
# packaging/vendor/python/ automatically IF it exists at build time.
# Production build scripts make this runtime mandatory and run the
# frozen ``--analysis-check`` gate; only explicitly lightweight
# development builds may omit it. Dev runs
# (`uv run sift`) never need this at all.
#
# REQUIREMENTS (not satisfiable in every environment)
# ----------------------------------------------------
# - Must run ON macOS arm64 (matches packaging/sift.spec's own
#   target_arch — a vendored runtime built for the wrong platform
#   would fail every sandbox-health probe at app startup, and
#   find_bundled_python() would correctly refuse it, but that's a
#   wasted release-build cycle worth avoiding at the source).
# - Needs outbound network access to github.com (the
#   python-build-standalone release asset) and PyPI (the analysis
#   packages). Release builds must run this script on the target
#   platform and pass the frozen analysis-runtime qualification.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="${1:-$REPO_ROOT/packaging/vendor/python}"

# Pinned to the same minor version Sift's own dev environment targets
# (see pyproject.toml's requires-python). Update this alongside that
# constraint, not independently — a bundled runtime on a different
# minor version than the one Sift's own test suite runs against is
# an untested combination.
PYTHON_ID="cpython-3.12.11-macos-aarch64-none"

ANALYSIS_PACKAGES=(
  'pandas==2.3.3'
  'numpy==2.3.5'
  'scipy==1.17.1'
  'statsmodels==0.14.6'
  'matplotlib==3.11.1'
  'duckdb==1.5.5'
  'pyarrow==25.0.1'
  'openpyxl==3.1.5'
  'xlrd==2.0.2'
  'odfpy==1.4.1'
  'pyreadstat==1.3.6'
  'pyreadr==0.5.6'
  'scikit-learn==1.6.1'
  'factor-analyzer==0.5.1'
  'pyfixest==0.60.0'
  'rdrobust==2.0.0'
  'differences==0.3.0'
  'geopandas==1.1.4'
  'arviz==1.3.0'
  'pymc==6.3.1'
)

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "vendor_python.sh must run on macOS arm64 (packaging/sift.spec's target_arch)." >&2
  echo "Detected: $(uname -s) $(uname -m). Refusing to produce a runtime for the wrong platform." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (fetches the portable CPython build and installs packages)." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

echo "==> Fetching portable CPython ($PYTHON_ID) via uv python install..."
uv python install --install-dir "$WORK_DIR" "$PYTHON_ID"

# uv's on-disk layout under --install-dir is intentionally NOT
# hardcoded here -- it could not be verified against a real fetch
# from this script's authoring environment (no outbound network
# access to github.com in that sandbox). Locate the produced
# interpreter by search instead of by assumed path, so a future uv
# version nesting things differently doesn't silently break this
# script.
FOUND_BIN="$(find "$WORK_DIR" -type f -name 'python3*' -perm -u+x | head -n1)"
if [[ -z "$FOUND_BIN" ]]; then
  echo "Could not locate an installed python3 binary under $WORK_DIR after uv python install." >&2
  echo "Inspect $WORK_DIR manually -- uv's on-disk layout may have changed." >&2
  exit 1
fi
# python-build-standalone's "install_only" variant (what `uv python
# install` fetches) is built to be relocated as a whole directory
# tree, not just the binary -- the interpreter needs its adjacent
# lib/ (stdlib + lib-dynload) to start at all. Move the binary's
# grandparent (the actual distribution root, typically
# <...>/install/ or <...>/bin/../) rather than just python3 itself.
DIST_ROOT="$(cd "$(dirname "$FOUND_BIN")/.." && pwd)"

rm -rf "$VENDOR_ROOT"
mkdir -p "$(dirname "$VENDOR_ROOT")"
echo "==> Relocating vendored distribution to $VENDOR_ROOT ..."
cp -R "$DIST_ROOT" "$VENDOR_ROOT"

VENDORED_PY="$VENDOR_ROOT/bin/python3"
if [[ ! -x "$VENDORED_PY" ]]; then
  echo "Relocated distribution is missing bin/python3 at $VENDORED_PY -- layout assumption was wrong." >&2
  exit 1
fi

echo "==> Verifying the relocated interpreter still starts..."
"$VENDORED_PY" --version

echo "==> Installing Sift's analysis stack (${ANALYSIS_PACKAGES[*]})..."
# The copied uv-managed interpreter contains PEP 668's EXTERNALLY-MANAGED
# marker. It is Sift's private application runtime—not the host Python—so the
# override is intentionally restricted to this exact interpreter.
uv pip install --python "$VENDORED_PY" --break-system-packages "${ANALYSIS_PACKAGES[@]}"

echo "==> Verifying every package imports from the vendored interpreter..."
"$VENDORED_PY" -I -c "
import importlib
import sys
pkgs = ['pandas', 'numpy', 'scipy', 'statsmodels', 'matplotlib', 'duckdb', 'pyarrow', 'openpyxl', 'xlrd', 'odf', 'pyreadstat', 'pyreadr', 'sklearn', 'factor_analyzer', 'pyfixest', 'rdrobust', 'differences', 'geopandas', 'arviz', 'pymc']
failed = []
for package in pkgs:
    try:
        importlib.import_module(package)
    except Exception as exc:
        failed.append((package, type(exc).__name__, str(exc)))
if failed:
    print('FAILED IMPORTS:', failed, file=sys.stderr)
    sys.exit(1)
print('All analysis packages importable from the vendored interpreter.')
"

echo "==> Done. Vendored Python ready at: $VENDOR_ROOT"
echo "    env_detect.find_bundled_python() expects exactly this layout"
echo "    (<repo>/packaging/vendor/python/bin/python3 when VENDOR_ROOT"
echo "    is left at its default)."
echo "    Next: uv run pyinstaller packaging/sift.spec --clean --noconfirm"
