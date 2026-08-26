#!/bin/bash
#
# Sift.app launcher — execs the bundled web-UI binary in place.
#
# macOS launches this script when the researcher double-clicks
# Sift.app (it's Contents/MacOS/Sift per the Info.plist set in
# build_app.sh). The script runs in the GUI session and our binary
# is the pywebview shell, which opens its own native WKWebView
# window. The launcher does not spawn a Terminal; the application owns its
# native window and captures diagnostics through the Python entry point.
#
# The Python entry point captures stdout/stderr into bounded, redacted,
# enterprise-policy-aware diagnostics. The launcher deliberately does not
# redirect either stream because doing so would create a raw logging bypass.

set -euo pipefail

# Bridge the GUI process's PATH to the user's tool installs.
#
# macOS launches a double-clicked .app under launchd with a bare PATH
# (typically just /usr/bin:/bin:/usr/sbin:/sbin) — none of /usr/local/bin,
# /opt/homebrew/bin, ~/.local/bin, or any Node version manager dirs are
# visible. The Claude Agent SDK that backs Sift spawns the ``claude``
# CLI as a subprocess to run the agent loop; when ``claude`` is the
# npm-installed script (``#!/usr/bin/env node``), launching it from a
# double-clicked .app fails immediately with
#   env: node: No such file or directory
#   Fatal error in message reader: Command failed with exit code 127
# and the chat UI surfaces "session setup failed: Command failed with
# exit code 127". The same install works fine from a terminal because
# the shell's PATH was assembled by /etc/profile + ~/.zshrc.
#
# Fix: prepend the conventional macOS dev-tool locations to PATH so the
# subprocess can find both ``claude`` (native or npm) AND ``node``. This
# does NOT source the user's shell init — that risks running arbitrary
# code on every launch — it just adds well-known directories that
# either exist or don't. Order matters: user-scope paths come before
# system-scope so a per-user override wins over a system install.
#
# The list covers Homebrew (Apple Silicon + Intel), Volta, Bun, the
# default npm-global prefix, ``~/.local/bin``, the two common
# shell-shim managers (asdf, mise), and Python-specific interpreter
# managers (pyenv, uv, conda, python.org's framework installer).
# Anything else (exotic prefixes, custom ``$PREFIX`` builds) needs
# intervention beyond this launcher — neither ``~/.zshrc`` nor
# ``~/.zshenv`` is sourced from a Finder/launchd launch (the shebang
# above is bash, and launchd does not run shell init files for .app
# launches), so editing those files will NOT change the PATH this
# script sees. Researchers in that case should either (a) symlink the
# missing tool into one of the listed directories, (b) set the PATH
# via ``launchctl setenv PATH ...`` from a LaunchAgent, or (c) launch
# Sift from a terminal session where the shell init has already
# assembled PATH.
#
# Why the Python-specific paths matter: ``find_python()`` in
# env_detect.py picks the first ``python3`` on PATH. On a fresh macOS
# install with no developer tooling, that defaults to Apple's
# ``/usr/bin/python3`` — an xcselect stub that fails the sandbox-
# health probe (libxcrun lives outside Sift's read allowlist) and is
# then rejected as unusable. Adding pyenv shims / uv-managed Pythons /
# conda envs / python.org framework versions ahead of /usr/bin lets
# the researcher's actual Python win the search instead of the stub.
_SIFT_EXTRA_PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.volta/bin:$HOME/.bun/bin:$HOME/.asdf/shims:$HOME/.local/share/mise/shims:$HOME/.pyenv/shims:$HOME/miniconda3/bin:$HOME/anaconda3/bin:$HOME/opt/miniconda3/bin:$HOME/opt/anaconda3/bin:/opt/miniconda3/bin:/opt/anaconda3/bin:/opt/homebrew/bin:/usr/local/bin"

# Globbed paths: per-version subdirectories where uv and the python.org
# framework installer place their interpreters. uv installs Pythons
# under ``~/.local/share/uv/python/cpython-<version>-<platform>/bin``
# and does NOT symlink ``python3`` into the user's bin by default;
# the python.org installer puts the binary at
# ``/Library/Frameworks/Python.framework/Versions/<version>/bin/python3``
# and only adds that to PATH if the researcher ran the optional
# ``Update Shell Profile.command`` (which most skip).
#
# Bash leaves unmatched globs as literal patterns by default, so the
# ``-d`` test below correctly skips non-existent entries even when
# the user has installed neither manager. ``set -u`` doesn't affect
# pathname expansion.
for _sift_glob_dir in \
        "$HOME"/.local/share/uv/python/cpython-*/bin \
        /Library/Frameworks/Python.framework/Versions/*/bin; do
    [[ -d "$_sift_glob_dir" ]] || continue
    _SIFT_EXTRA_PATH="$_SIFT_EXTRA_PATH:$_sift_glob_dir"
done
unset _sift_glob_dir
export PATH="$_SIFT_EXTRA_PATH:${PATH:-/usr/bin:/bin:/usr/sbin:/sbin}"

# Resolve the .app's own path from wherever macOS launched us.
# $0 is Contents/MacOS/Sift (inside the app bundle); two dirs up
# gets us to the .app root.
LAUNCHER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd -- "$LAUNCHER_DIR/../.." && pwd)"
SIFT_BIN="$APP_ROOT/Contents/Resources/sift/sift"

if [[ ! -x "$SIFT_BIN" ]]; then
    # Self-display dialog via osascript — doesn't need Automation
    # entitlements. Fallback for a corrupted / partial install.
    /usr/bin/osascript -e "display dialog \"Sift.app is missing its bundled binary at $SIFT_BIN. The .app may have been damaged or incompletely installed.\" buttons {\"OK\"} default button \"OK\" with icon stop"
    exit 1
fi

# exec replaces this shell with the binary so the .app's process
# tree shows ``sift`` directly (cleaner Activity Monitor entry,
# Quit/Force-Quit work as expected). Sift's Python entry point owns
# diagnostic capture so credentials and local paths are redacted before
# writing and retention/size ceilings apply consistently on every OS.
# Shell-level redirection here would create an unredacted bypass.
exec "$SIFT_BIN"
