"""Serve the production desktop shell with the browser-only bridge fixture.

This is a visual-QA utility, never part of a packaged Sift application.  It
keeps the preview faithful by serving the real HTML/CSS/JS and injecting only
the pywebview API fixture immediately before the production scripts run.
"""

from __future__ import annotations

import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).parents[1]
WEB = ROOT / "src" / "sift" / "web"
BOOTSTRAP = ROOT / "tests" / "gui_preview_bootstrap.js"


class PreviewHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB), **kwargs)

    def do_GET(self) -> None:  # noqa: N802 — stdlib handler protocol
        if self.path.split("?", 1)[0] in {"", "/", "/index.html"}:
            html = (WEB / "index.html").read_text(encoding="utf-8")
            html = html.replace(
                '<script src="app.js"></script>',
                '<script src="/__preview_bridge.js"></script>\n'
                '  <script src="app.js"></script>',
                1,
            )
            # Mirror the production cache-busting contract for every local
            # script and stylesheet. Selectively busting only sources.js/css
            # made visual QA silently reuse stale app.js and shell CSS after
            # an edit, so a repaired layout could look broken (or vice versa).
            build_id = str(max(
                child.stat().st_mtime_ns
                for child in WEB.iterdir()
                if child.suffix.lower() in {".js", ".css", ".html"}
            ))
            html = re.sub(
                r'(src|href)="([^"]+\.(?:js|css))"',
                lambda match: (
                    f'{match.group(1)}="{match.group(2)}?v={build_id}"'
                ),
                html,
            )
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path.split("?", 1)[0] == "/__preview_bridge.js":
            payload = BOOTSTRAP.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/javascript; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8765), PreviewHandler).serve_forever()
