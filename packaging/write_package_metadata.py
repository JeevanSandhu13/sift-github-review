"""Write deterministic metadata embedded in portable release archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", choices=("windows", "linux"), required=True)
    parser.add_argument("--architecture", required=True)
    args = parser.parse_args()
    executable = "Sift.exe" if args.platform == "windows" else "app/sift"
    runtime_requirements = (
        [
            "64-bit Windows 11 (build 22000 or newer)",
            "Microsoft Edge WebView2 Evergreen Runtime 86.0.616.0 or newer",
        ]
        if args.platform == "windows"
        else [
            (
                "64-bit Linux ARM64 with glibc 2.39 or newer"
                if args.architecture == "aarch64"
                else "64-bit Linux x86_64 with glibc 2.35 or newer"
            ),
            "X11 or Wayland desktop session",
            "bubblewrap confinement",
            "Freedesktop Secret Service-compatible credential vault",
        ]
    )
    document = {
        "format": "sift-package-metadata",
        "schema_version": 2,
        "name": "Sift",
        "publisher": "Sapien Institute",
        "version": args.version,
        "platform": args.platform,
        "architecture": args.architecture,
        "package_kind": (
            "installer_and_portable_archive"
            if args.platform == "windows"
            else "portable_archive_with_per_user_installer"
        ),
        "executable": executable,
        "install_scope": "per-user",
        "requires_administrator": False,
        "host_preparation": (
            {
                "may_require_administrator": True,
                "command": "sudo ./prepare_ubuntu_host.sh",
                "scope": "Ubuntu 24.04 bubblewrap AppArmor policy only",
            }
            if args.platform == "linux" else None
        ),
        "silent_installer_available": True,
        "installer": (
            {
                "filename": "Sift-Windows-x64-Setup.exe",
                "scope": "per-user",
                "silent_arguments": ["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"],
            }
            if args.platform == "windows"
            else {
                "filename": "install.sh",
                "scope": "per-user",
                "silent_arguments": [],
            }
        ),
        "runtime_requirements": runtime_requirements,
        "uninstall": (
            "Run the Sift uninstaller from Windows Settings. User sessions "
            "and credential-vault entries are retained."
            if args.platform == "windows"
            else (
                "Run $XDG_DATA_HOME/sift/uninstall.sh, or "
                "~/.local/share/sift/uninstall.sh when XDG_DATA_HOME is unset. "
                "User sessions and credential-vault entries are retained."
            )
        ),
    }
    args.output.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
