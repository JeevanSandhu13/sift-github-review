"""Desktop-platform qualification shared by source and frozen builds.

The GUI intentionally uses one reviewed renderer per operating system:
WKWebView on macOS, WebView2 on Windows, and Qt WebEngine on Linux.  Falling
back to an older or merely-present renderer is unsafe for Sift's modern web
shell and makes release behavior differ from development behavior.
"""

from __future__ import annotations

import importlib
import importlib.util
import hmac
import json
import os
import platform
import secrets
import shutil
import struct
import sys
import sysconfig
import threading
from pathlib import Path
from typing import Any, Literal


WINDOWS_WEBVIEW2_CLIENT = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
WINDOWS_WEBVIEW2_MINIMUM_VERSION = (86, 0, 616, 0)
WINDOWS_11_MINIMUM_BUILD = 22_000
SUPPORTED_ARCHITECTURES = {
    "darwin": {"arm64", "aarch64"},
    "win32": {"amd64", "x86_64"},
    "linux": {"x86_64", "amd64", "aarch64", "arm64"},
}


def normalized_platform(value: str | None = None) -> str:
    current = (value or sys.platform).lower()
    if current.startswith("win"):
        return "win32"
    if current.startswith("linux"):
        return "linux"
    if current == "darwin":
        return "darwin"
    return current


def runtime_architecture(
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    python_platform: str | None = None,
) -> str:
    """Return the application process target, not merely the host CPU.

    This distinction matters on Windows 11 ARM, where the supported Sift x64
    build runs through Windows' x64 emulation.  ``platform.machine()`` can
    report the ARM host there; ``sysconfig.get_platform()`` remains bound to
    the Python/PE target that PyInstaller will freeze.
    """
    current = normalized_platform(platform_name)
    if current == "win32":
        target = (python_platform or sysconfig.get_platform()).lower()
        if target in {"win-amd64", "win-x86_64"}:
            return "amd64"
        if target in {"win-arm64", "win-aarch64"}:
            return "arm64"
        if target in {"win32", "win-x86"}:
            return "x86"
        return target or "unknown"
    return (machine or platform.machine()).lower()


def windows_x64_emulation(
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    python_platform: str | None = None,
) -> bool:
    """Return whether an x64 Sift process is running on Windows ARM.

    Windows exposes the native host architecture through
    ``platform.machine()`` even to an emulated x64 process, while sysconfig
    continues to describe the x64 Python target.  This precise distinction is
    needed by native extensions whose CPUID probes cannot execute when Python
    reports the ARM host architecture.
    """
    return (
        normalized_platform(platform_name) == "win32"
        and (machine or platform.machine()).lower() in {"arm64", "aarch64"}
        and runtime_architecture(
            platform_name=platform_name,
            machine=machine,
            python_platform=python_platform,
        ) == "amd64"
    )


def preferred_webview_gui(
    value: str | None = None,
) -> Literal["cocoa", "edgechromium", "qt"]:
    """Return the only renderer Sift qualifies on the selected platform."""
    current = normalized_platform(value)
    if current == "darwin":
        return "cocoa"
    if current == "win32":
        return "edgechromium"
    if current == "linux":
        return "qt"
    raise RuntimeError(f"unsupported desktop platform: {current}")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _module_importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001 - native loader failures become a failed check
        return False


def _version_key(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def windows_webview2_runtime_version(registry: Any | None = None) -> str | None:
    """Return the installed Evergreen WebView2 version, if present.

    Microsoft documents both per-user and per-machine installations.  On a
    64-bit host the updater may expose its 32-bit registry view explicitly,
    so all reviewed locations are checked and the newest valid version wins.
    """
    if registry is None:
        if normalized_platform() != "win32":
            return None
        try:
            import winreg as registry  # type: ignore[no-redef]
        except ImportError:
            return None

    registry_api = registry
    if registry_api is None:  # Defensive narrowing for static and runtime safety.
        return None
    roots = [
        getattr(registry_api, "HKEY_CURRENT_USER", None),
        getattr(registry_api, "HKEY_LOCAL_MACHINE", None),
    ]
    paths = [
        rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{WINDOWS_WEBVIEW2_CLIENT}",
        rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{WINDOWS_WEBVIEW2_CLIENT}",
    ]
    versions: list[str] = []
    for root in roots:
        if root is None:
            continue
        for path in paths:
            try:
                with registry_api.OpenKey(root, path) as key:
                    value, _ = registry_api.QueryValueEx(key, "pv")
            except (OSError, AttributeError):
                continue
            candidate = str(value).strip()
            if candidate and _version_key(candidate) and any(_version_key(candidate)):
                versions.append(candidate)
    return max(versions, key=_version_key) if versions else None


def windows_webview2_runtime_supported(version: str | None = None) -> bool:
    """Return whether the installed Evergreen Runtime can load WebView2.

    Microsoft documents 86.0.616.0 as the minimum Runtime capable of loading
    WebView2. Merely finding a non-empty updater registry value is therefore
    insufficient on a machine whose enterprise update policy has retained an
    obsolete Runtime.
    """
    candidate = version if version is not None else windows_webview2_runtime_version()
    key = _version_key(candidate) if candidate else ()
    padded = key[:4] + (0,) * max(0, 4 - len(key))
    return bool(key) and padded >= WINDOWS_WEBVIEW2_MINIMUM_VERSION


def windows_build_number(version_info: Any | None = None) -> int | None:
    """Return the Windows kernel build without exposing other host details."""
    if version_info is None:
        if normalized_platform() != "win32":
            return None
        try:
            get_windows_version = getattr(sys, "getwindowsversion", None)
            if not callable(get_windows_version):
                return None
            version_info = get_windows_version()
        except OSError:
            return None
    try:
        return int(version_info.build)
    except (AttributeError, TypeError, ValueError):
        return None


def windows_11_or_newer(version_info: Any | None = None) -> bool:
    """Return whether the host meets Sift's Windows 11 support floor."""
    build = windows_build_number(version_info)
    return build is not None and build >= WINDOWS_11_MINIMUM_BUILD


def _webview_library_files_present() -> bool:
    spec = importlib.util.find_spec("webview")
    if spec is None or spec.origin is None:
        return False
    library = Path(spec.origin).resolve().parent / "lib"
    return all(
        (library / filename).is_file()
        for filename in (
            "Microsoft.Web.WebView2.Core.dll",
            "Microsoft.Web.WebView2.WinForms.dll",
        )
    )


def desktop_runtime_report(*, require_sandbox: bool = False) -> dict[str, Any]:
    """Return a content-free report suitable for build and support checks."""
    current = normalized_platform()
    machine = runtime_architecture(platform_name=current)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    supported = current in SUPPORTED_ARCHITECTURES
    add("operating_system", supported, current)
    architecture_ok = machine in SUPPORTED_ARCHITECTURES.get(current, set())
    add("architecture", architecture_ok, machine or "unknown")
    add("python_64_bit", struct.calcsize("P") * 8 == 64, f"{struct.calcsize('P') * 8}-bit")

    web_root = Path(__file__).resolve().parent / "web"
    required_assets = ("index.html", "app.js", "sources.js", "style.css", "desktop-shell.css")
    missing_assets = [name for name in required_assets if not (web_root / name).is_file()]
    add("web_assets", not missing_assets, "complete" if not missing_assets else ", ".join(missing_assets))
    webview_ready = _module_importable("webview")
    add("pywebview", webview_ready, "importable" if webview_ready else "missing or unloadable")

    if current == "darwin":
        cocoa = all(_module_importable(name) for name in ("Cocoa", "Quartz", "WebKit"))
        add("renderer_cocoa", cocoa, "WKWebView/PyObjC" if cocoa else "PyObjC framework missing")
        add("credential_store", _module_importable("keyring.backends.macOS"), "macOS Keychain")
        sandbox_present = Path("/usr/bin/sandbox-exec").is_file()
        add("sandbox_binary", sandbox_present, "sandbox-exec" if sandbox_present else "missing")
    elif current == "win32":
        windows_modules = _module_importable("clr") and _module_importable("webview.platforms.winforms")
        add("renderer_bindings", windows_modules, "pythonnet/WinForms" if windows_modules else "pythonnet missing")
        add("webview2_loader", _webview_library_files_present(), "bundled loader assemblies")
        runtime_version = windows_webview2_runtime_version()
        runtime_supported = windows_webview2_runtime_supported(runtime_version)
        add(
            "webview2_runtime",
            runtime_supported,
            str(runtime_version)
            if runtime_supported
            else "Evergreen Runtime missing or too old",
        )
        add("credential_store", _module_importable("keyring.backends.Windows"), "Windows Credential Locker")
        sandbox_present = True  # the substantive native probe runs below
    elif current == "linux":
        qt = all(
            _module_importable(name)
            for name in ("PyQt6", "PyQt6.QtWidgets", "PyQt6.QtWebEngineWidgets")
        )
        add("renderer_qt", qt, "Qt 6 WebEngine" if qt else "Qt 6 WebEngine missing")
        secret_service = _module_importable("secretstorage") and _module_importable("keyring.backends.SecretService")
        add("credential_store", secret_service, "Freedesktop Secret Service" if secret_service else "Secret Service binding missing")
        sandbox_present = shutil.which("bwrap") is not None
        add("sandbox_binary", sandbox_present, "bubblewrap" if sandbox_present else "bubblewrap missing")
    else:
        sandbox_present = False

    if require_sandbox and supported and sandbox_present:
        try:
            from sift.env_detect import detect_environment

            sandbox_working = detect_environment().has_sandbox_backend()
        except Exception as exc:  # noqa: BLE001 - normalized content-free report
            sandbox_working = False
            detail = type(exc).__name__
        else:
            detail = "native confinement probe passed" if sandbox_working else "native confinement probe failed"
        add("sandbox_probe", sandbox_working, detail)

    report = {
        "schema_version": 1,
        "platform": current,
        "architecture": machine,
        "renderer": preferred_webview_gui(current) if supported else None,
        "checks": checks,
    }
    report["ok"] = all(item["ok"] for item in checks)
    return report


def credential_store_roundtrip(keyring_module: Any | None = None) -> tuple[bool, str]:
    """Prove the selected secure OS vault can write, read, and delete.

    The canary is random, contains no user material, is never returned, and is
    removed in a ``finally`` block.  This probe is intentionally explicit
    rather than part of ordinary startup because some operating systems may
    ask the user to unlock their credential vault.
    """
    try:
        if keyring_module is None:
            import keyring as keyring_module  # type: ignore[no-redef]
        from sift.integration_core import keyring_module_is_secure

        ring: Any = keyring_module
        if not keyring_module_is_secure(ring):
            return False, "secure OS credential backend unavailable"
        account = f"qualification-{os.getpid()}-{secrets.token_hex(8)}"
        value = secrets.token_urlsafe(32)
        service = "org.sapieninstitute.sift.qualification"
        wrote = False
        readback_matches = False
        cleanup_failed = False
        try:
            ring.set_password(service, account, value)
            wrote = True
            recovered = ring.get_password(service, account)
            readback_matches = isinstance(recovered, str) and hmac.compare_digest(
                recovered, value,
            )
        finally:
            if wrote:
                try:
                    ring.delete_password(service, account)
                except Exception:  # noqa: BLE001 - report cleanup failure below
                    cleanup_failed = True
        if cleanup_failed:
            return False, "OS credential-store canary cleanup failed"
        if not readback_matches:
            return False, "OS credential-store readback mismatch"
        if ring.get_password(service, account) is not None:
            return False, "OS credential-store canary remained after deletion"
    except Exception as exc:  # noqa: BLE001 - normalized, content-free detail
        return False, type(exc).__name__
    return True, "secure OS credential-store round-trip passed"


def qt_webengine_runtime_probe(*, timeout_ms: int = 15_000) -> tuple[bool, str]:
    """Create and load a real Qt WebEngine view on Linux.

    Imports alone cannot reveal missing XCB libraries, an unusable display, or
    a Chromium helper that immediately crashes. Native Linux release builds
    run this under Xvfb against the frozen executable.
    """
    if normalized_platform() != "linux":
        return False, "Qt WebEngine probe is Linux-only"
    try:
        from PyQt6.QtCore import QEventLoop, QTimer
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtWebEngineWidgets import QWebEngineView

        app = QApplication.instance() or QApplication(["sift-renderer-check"])
        loop = QEventLoop()
        view = QWebEngineView()
        outcome = {"loaded": False, "finished": False}

        def finish(loaded: bool) -> None:
            outcome["loaded"] = bool(loaded)
            outcome["finished"] = True
            loop.quit()

        view.loadFinished.connect(finish)
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)
        timer.start(timeout_ms)
        view.setHtml("<!doctype html><meta charset=utf-8><title>Sift check</title>")
        loop.exec()
        timer.stop()
        view.close()
        view.deleteLater()
        app.processEvents()
    except Exception as exc:  # noqa: BLE001 - native loader failures are reported
        return False, type(exc).__name__
    if not outcome["finished"]:
        return False, "renderer timed out"
    detail = "local page rendered" if outcome["loaded"] else "local page load failed"
    return bool(outcome["loaded"]), detail


def windows_webview2_runtime_probe(
    *, timeout_seconds: float = 20.0,
) -> tuple[bool, str]:
    """Create a hidden Edge WebView2 window and load local HTML on Windows."""
    if normalized_platform() != "win32":
        return False, "WebView2 renderer probe is Windows-only"
    if not windows_webview2_runtime_supported():
        return False, "Evergreen Runtime missing or too old"
    try:
        import webview

        loaded = threading.Event()
        window = webview.create_window(
            "Sift renderer check",
            html="<!doctype html><meta charset=utf-8><title>Sift check</title>",
            width=320,
            height=200,
            hidden=True,
        )
        if window is None:
            return False, "renderer window creation was cancelled"

        timer: threading.Timer

        def close_loaded_window() -> None:
            loaded.set()
            timer.cancel()
            window.destroy()

        def close_timed_out_window() -> None:
            try:
                window.destroy()
            except Exception:
                pass

        window.events.loaded += close_loaded_window
        timer = threading.Timer(timeout_seconds, close_timed_out_window)
        timer.daemon = True
        timer.start()
        try:
            webview.start(gui="edgechromium", debug=False, private_mode=True)
        finally:
            timer.cancel()
    except Exception as exc:  # noqa: BLE001 - native loader failures are reported
        return False, type(exc).__name__
    return (True, "local page rendered") if loaded.is_set() else (False, "renderer timed out")


def format_runtime_report(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, separators=(",", ":"))


__all__ = [
    "desktop_runtime_report",
    "format_runtime_report",
    "normalized_platform",
    "preferred_webview_gui",
    "runtime_architecture",
    "credential_store_roundtrip",
    "qt_webengine_runtime_probe",
    "windows_webview2_runtime_probe",
    "windows_webview2_runtime_version",
    "windows_webview2_runtime_supported",
    "windows_build_number",
    "windows_11_or_newer",
]
