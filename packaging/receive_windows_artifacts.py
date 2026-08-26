#!/usr/bin/env python3
"""Receive the fixed Windows release artifact set from an isolated VM.

The server accepts only the six expected artifact/sidecar filenames, writes
each upload atomically, and binds to the VM-only host interface by default.
It never replaces files in ``dist`` directly; callers verify the complete
incoming set before promoting it.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path, PurePosixPath


EXPECTED = {
    "Sift-Windows-x64-Setup.exe",
    "Sift-Windows-x64-Setup.exe.sha256",
    "Sift-Windows-x64-Setup.exe.sbom.cdx.json",
    "Sift-Windows-x64.zip",
    "Sift-Windows-x64.zip.sha256",
    "Sift-Windows-x64.zip.sbom.cdx.json",
}
MAX_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024


def handler_for(output_dir: Path) -> type[BaseHTTPRequestHandler]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    class ArtifactHandler(BaseHTTPRequestHandler):
        server_version = "SiftArtifactReceiver/1"

        def do_PUT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = PurePosixPath(self.path.split("?", 1)[0])
            filename = path.name
            if path.parent != PurePosixPath("/") or filename not in EXPECTED:
                self.send_error(404, "unexpected release artifact")
                return
            raw_length = self.headers.get("Content-Length")
            try:
                length = int(raw_length) if raw_length is not None else -1
            except ValueError:
                length = -1
            if length < 0 or length > MAX_ARTIFACT_BYTES:
                self.send_error(413, "invalid artifact size")
                return

            destination = output_dir / filename
            temporary = output_dir / f".{filename}.uploading"
            remaining = length
            try:
                with temporary.open("wb") as handle:
                    while remaining:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ConnectionError("upload ended before Content-Length")
                        handle.write(chunk)
                        remaining -= len(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, destination)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise

            self.send_response(201)
            self.send_header("Content-Length", "0")
            self.end_headers()
            print(f"received {filename} ({length} bytes)", flush=True)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.client_address[0]} {format % args}", flush=True)

    return ArtifactHandler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--host", default="192.168.64.1")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_for(args.output_dir)
    )
    print(f"receiving on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
