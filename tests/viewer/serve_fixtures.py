"""Tiny stand-in for the app's ViewerServer, used by the Playwright browser tests.

Serves the viewer web app plus a scene library with the same routes as
``vrphoto.webxr.server.ViewerServer``: ``/`` (gallery), ``/view.html``, ``/api/scenes``
(:func:`vrphoto.webxr.scene.list_scenes`), ``/scenes/<id>/<file>`` (with HTTP Range) and plain
static files.  Plain HTTP on 127.0.0.1, which browsers treat as a secure context.

Usage::

    python tests/viewer/serve_fixtures.py --port 8765            # builds a fixture library in a temp dir
    python tests/viewer/serve_fixtures.py --port 8765 --library DIR
    python tests/viewer/serve_fixtures.py --port 8765 --app-server   # use the real ViewerServer (HTTP only)

Works without ``pip install -e .`` as long as numpy and Pillow are importable (the repository's
``src`` folder is added to ``sys.path`` when ``vrphoto`` is not installed).
"""

from __future__ import annotations

import argparse
import json
import posixpath
import re
import sys
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VIEWER_DIR = REPO / "src" / "vrphoto" / "webxr" / "viewer"

try:  # make the repository importable without an editable install
    import vrphoto  # noqa: F401
except ImportError:  # pragma: no cover - depends on the environment
    sys.path.insert(0, str(REPO / "src"))

#: Explicit types - never trust the OS registry (Windows can map .js to text/plain).
MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".wasm": "application/wasm",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".ply": "application/octet-stream",
    ".splat": "application/octet-stream",
    ".spz": "application/octet-stream",
    ".md": "text/plain; charset=utf-8",
}
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)$")


def make_handler(library: Path, viewer: Path) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "vrphoto-test/1"

        def log_message(self, fmt: str, *args) -> None:  # quiet
            pass

        def _send_bytes(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _not_found(self) -> None:
            self._send_bytes(404, b"Not found", "text/plain; charset=utf-8")

        def _send_file(self, path: Path) -> None:
            try:
                size = path.stat().st_size
            except OSError:
                self._not_found()
                return
            ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
            start, end = 0, size - 1
            status = HTTPStatus.OK
            rng = self.headers.get("Range")
            if rng:
                m = _RANGE_RE.match(rng.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1):
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                    else:
                        start = max(0, size - int(m.group(2)))
                    end = min(end, size - 1)
                    if start > end or start >= size:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{size}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    status = HTTPStatus.PARTIAL_CONTENT
            length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self.command == "HEAD":
                return
            with path.open("rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        @staticmethod
        def _inside(base: Path, rel: str) -> Path | None:
            rel = posixpath.normpath("/" + rel).lstrip("/")
            if not rel or rel.startswith(".."):
                return None
            p = (base / rel).resolve()
            try:
                p.relative_to(base.resolve())
            except ValueError:
                return None
            return p if p.is_file() else None

        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(urlsplit(self.path).path)
            if path == "/api/scenes":
                from vrphoto.webxr.scene import list_scenes

                body = json.dumps(list_scenes(library)).encode("utf-8")
                self._send_bytes(200, body, "application/json; charset=utf-8")
                return
            if path.startswith("/scenes/"):
                f = self._inside(library, path[len("/scenes/") :])
            else:
                f = self._inside(viewer, "index.html" if path in ("", "/") else path.lstrip("/"))
            if f is None:
                self._not_found()
            else:
                self._send_file(f)

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--library", type=Path, default=None, help="existing library (default: build fixtures)")
    ap.add_argument("--viewer", type=Path, default=VIEWER_DIR)
    ap.add_argument("--app-server", action="store_true", help="serve with vrphoto.webxr.server.ViewerServer")
    args = ap.parse_args(argv)

    tmp = None
    library = args.library
    if library is None:
        sys.path.insert(0, str(HERE))
        from make_fixtures import build_library

        tmp = tempfile.TemporaryDirectory(prefix="vrphoto-fixtures-")
        library = Path(tmp.name)
        build_library(library)
    if args.app_server:
        from vrphoto.webxr.server import ViewerServer

        app = ViewerServer(
            library, https_port=None, http_port=args.port, viewer_dir=args.viewer, bind=args.host
        )
        app.start()
        sys.stdout.write(f"READY (ViewerServer) {app.urls()} library={library}\n")
        sys.stdout.flush()
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            app.stop()
            if tmp is not None:
                tmp.cleanup()
        return 0

    server = ThreadingHTTPServer((args.host, args.port), make_handler(library, args.viewer))
    server.daemon_threads = True
    sys.stdout.write(f"READY http://{args.host}:{server.server_address[1]}/ library={library}\n")
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
