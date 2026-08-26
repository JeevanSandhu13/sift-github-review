"""Static release contracts for Sift's shared desktop GUI.

These tests intentionally avoid a browser or live pywebview renderer so they run
on every release host. Native smoke tests still run in each platform's packaging
pipeline; this module locks the cross-platform parts that must not silently drift.
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "src" / "sift" / "web"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_desktop_shell_assets_are_loaded_locally_in_order() -> None:
    html = _read(WEB / "index.html")

    assert 'href="style.css"' in html
    assert 'href="desktop-shell.css"' in html
    assert html.index('href="style.css"') < html.index(
        'href="desktop-shell.css"'
    )
    assert 'src="desktop-platform.js"' in html
    assert html.index('src="desktop-platform.js"') < html.index(
        'src="app.js"'
    )
    assert "https://fonts." not in html
    assert "cdn.jsdelivr" not in html


def test_primary_workspace_landmarks_and_controls_are_named() -> None:
    html = _read(WEB / "index.html")

    assert '<html lang="en">' in html
    assert 'id="skip-link" class="skip-link" href="#messages"' in html
    assert 'id="auth-content" class="auth-card" tabindex="-1"' in html
    assert 'id="landing-content" class="landing-card" tabindex="-1"' in html
    assert 'aria-label="Research sessions"' in html
    assert 'aria-label="Research transcript"' in html
    assert 'aria-label="Message Sift"' in html
    assert 'id="checkpoints-label-input"' in html
    assert 'aria-label="Checkpoint label"' in html
    assert 'id="status-line" class="status-line" aria-live="polite"' in html
    assert html.count('data-role="status" role="status" aria-live="polite"') == 4
    assert html.count('role="status" aria-live="polite"') >= 7
    for provider in (
        "Anthropic", "OpenAI", "Google Gemini", "Custom endpoint",
    ):
        assert f'aria-label="{provider} API key"' in html
    assert 'id="sidebar-toggle"' in html
    assert 'aria-expanded="true"' in html
    assert 'aria-controls="sidebar-list"' in html
    assert html.count('aria-haspopup="dialog"') >= 3
    assert 'role="dialog" aria-label="Session files"' in html
    assert 'role="dialog" aria-label="Data permissions"' in html
    assert 'role="dialog" aria-label="Model and reasoning effort"' in html
    assert 'role="dialog" aria-modal="true"' in html


def test_about_and_updates_panel_is_accessible_and_user_initiated() -> None:
    html = _read(WEB / "index.html")
    script = _read(WEB / "app.js")

    assert 'id="updates-overlay"' in html
    assert 'aria-labelledby="updates-title"' in html
    assert 'id="updates-status" class="updates-status" role="status" aria-live="polite"' in html
    assert html.count("data-open-updates") >= 3
    assert "update_configuration" in script
    assert "check_for_updates(download)" in script
    assert "addEventListener('click', openUpdates)" in script
    assert "setInterval" not in script[script.index("async function openUpdates"):script.index("function closeUpdates")]


def test_privacy_and_credential_copy_is_accurate_and_cross_platform() -> None:
    html = _read(WEB / "index.html")

    assert "operating system's protected credential store" in html
    assert "Raw data is processed on this machine" in html
    assert "active permission tier" in html
    assert "Local workspace" in html
    assert html.count("Local analysis. Controlled disclosure.") == 2
    assert "macOS Keychain, never" not in html
    assert "No Data Leaves" not in html
    assert "Claude CLI" not in html


def test_loading_status_is_provider_neutral_and_offline() -> None:
    html = _read(WEB / "index.html")
    script = _read(WEB / "app.js")
    css = _read(WEB / "style.css")

    assert "Sift is ' + label" in script
    assert "Claude is ' + label" not in script
    assert "lottie-player" not in html
    assert "cat-loading" not in css


def test_optional_language_runtimes_do_not_create_startup_errors() -> None:
    app_js = _read(WEB / "app.js")

    # .dta ingestion is provided by the bundled reader. A missing optional
    # Stata/R installation must not be presented as a product problem unless
    # no analysis language is usable at all.
    assert "report.blocked && r.status === 'unavailable'" in app_js
    assert "r.status === 'blocked'" in app_js


def test_platform_adapter_covers_all_desktop_hosts_and_shortcuts() -> None:
    platform_js = _read(WEB / "desktop-platform.js")
    app_js = _read(WEB / "app.js")
    html = _read(WEB / "index.html")

    for platform in ("macos", "windows", "linux"):
        assert f"'{platform}'" in platform_js
    assert "data-shortcut-mod" in html
    assert "resolved === 'macos' ? '⌘' : 'Ctrl'" in platform_js
    assert "Finder" in app_js
    assert "File Explorer" in app_js
    assert "Credential Manager" in app_js
    assert "nativeFileManager" in app_js
    assert "protectedCredentialStore" in app_js


def test_shell_has_required_accessibility_and_reflow_modes() -> None:
    css = _read(WEB / "desktop-shell.css")

    assert "outline: 2px solid var(--shell-focus)" in css
    assert "min-width: 28px" in css
    assert "min-height: 28px" in css
    assert "@media (prefers-reduced-motion: reduce)" in css
    assert "@media (prefers-contrast: more)" in css
    assert "@media (forced-colors: active)" in css
    assert "@media (max-width: 920px)" in css
    assert "@media (max-width: 720px)" in css
    assert "@media (max-width: 480px)" in css
    assert 'data-platform="windows"' in css
    assert 'data-platform="linux"' in css
    assert "#sources-chip { display: none; }" in css
    assert ".topbar-chip:nth-of-type(-n + 2) { display: none; }" not in css
    assert "justify-self: stretch" in css


def test_shell_is_restrained_not_a_decorative_terminal_skin() -> None:
    css = _read(WEB / "desktop-shell.css").lower()
    html = _read(WEB / "index.html").lower()

    for unwanted in ("scanline", "matrix", "neon", "glitch"):
        assert unwanted not in css
        assert unwanted not in html
    # Monospace is applied to technical surfaces, not the whole document.
    body_rule = re.search(r"\nbody\s*\{(?P<body>.*?)\n\}", css, re.DOTALL)
    assert body_rule is not None
    assert "font-family: var(--shell-font)" in body_rule.group("body")


def test_native_window_contract_supports_resizing_zoom_and_private_storage() -> None:
    ui = _read(ROOT / "src" / "sift" / "ui.py")
    create_window = ui.split("window = webview.create_window(", 1)[1].split(
        "bridge.attach(window)", 1
    )[0]

    assert "width=1180" in create_window
    assert "height=780" in create_window
    assert "min_size=(880, 600)" in create_window
    assert "resizable=True" in create_window
    assert "text_select=True" in create_window
    assert "zoomable=True" in create_window
    assert 'background_color="#f4f5f2"' in create_window
    assert "gui=preferred_webview_gui()" in ui
    assert "debug=False" in ui
    assert "private_mode=True" in ui


def test_new_web_assets_are_automatically_bundled_on_every_platform() -> None:
    spec = _read(ROOT / "packaging" / "sift.spec")
    windows = _read(ROOT / "packaging" / "build_windows.ps1")
    linux = _read(ROOT / "packaging" / "build_linux.sh")
    mac = _read(ROOT / "packaging" / "build_app.sh")

    assert 'WEB_DIR.rglob("*")' in spec
    assert "WEB_DATAS" in spec
    assert "def runtime_submodules(package)" in spec
    for excluded in ("test", "tests", "testing", "benchmark", "benchmarks"):
        assert f'"{excluded}"' in spec
    for framework in ("Cocoa", "Foundation", "AppKit", "Quartz", "WebKit"):
        assert f'collect_submodules("{framework}")' in spec
    for build_script in (windows, linux, mac):
        assert "packaging/sift.spec" in build_script


def test_sidebar_and_theme_controls_keep_state_accessible() -> None:
    app_js = _read(WEB / "app.js")

    assert "renderSidebarToggle" in app_js
    assert "aria-expanded" in app_js
    assert "Expand sidebar" in app_js
    assert "Collapse sidebar" in app_js
    assert "aria-pressed" in app_js
    assert "Switch to light theme" in app_js
    assert "Switch to dark theme" in app_js
    assert "function activeModal()" in app_js
    assert "function trapModalTab(event, modal)" in app_js
    assert "if (trapModalTab(e, modal)) return" in app_js
    assert "filesChip.setAttribute('aria-expanded', 'false')" in app_js


def test_composer_prompts_are_professional_research_actions() -> None:
    app_js = _read(WEB / "app.js")
    block = app_js.split("const PLACEHOLDERS = [", 1)[1].split("];", 1)[0]

    assert "Profile the variables and data quality" in block
    assert "Build a reproducible analysis" in block
    for stale in (
        "Reviewer 2 is asleep",
        "The data is restless tonight",
        "Pun buffer",
        "Fixed effects, flexible morals",
    ):
        assert stale not in block


def test_dataset_profile_renderer_accepts_partial_bridge_payloads() -> None:
    app_js = _read(WEB / "app.js")
    block = app_js.split("function loadDatasetProfile(name) {", 1)[1].split(
        "function renderDatasetHealth(p) {", 1
    )[0]

    assert "Number.isFinite(rows)" in block
    assert "Number.isFinite(columns)" in block
    assert "Number.isFinite(missingPct)" in block
    assert "Array.isArray(p.variables)" in block
    assert "p.rows.toLocaleString()" not in block
    assert "Profile details are not available for this file yet." in block
    assert "Close and reopen Data, then try again." in block
