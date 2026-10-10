"""Local web server that delivers the WebXR viewer and exported scenes to a Quest.

Two stdlib :class:`~http.server.ThreadingHTTPServer` instances run in daemon threads:

* **HTTPS** on ``bind:https_port`` (default ``0.0.0.0:8443``) with the self-signed
  certificate from :func:`vrphoto.webxr.cert.ensure_cert` - Quest Browser over Wi-Fi.
  A plain-HTTP request that reaches this port is answered with a redirect to ``https://``.
* **HTTP** on ``127.0.0.1:http_port`` (default 8080) - for ``adb reverse`` over USB
  (``http://localhost`` is a secure context, so WebXR works) and the desktop browser.

Routes::

    GET /                      viewer index.html (gallery)
    GET /<file>                static viewer files (view.html, js/, css/, vendor/ ...)
    GET /scenes/<id>/<file>    exported scene assets from the library folder
    GET /api/scenes            JSON scene list (webxr.scene.list_scenes)
    GET /api/info              JSON {app, version, urls, ...}
    GET /files/<path>[?download=1]  optional: files from ``files_dir`` (outputs)
    GET /api/files             optional: JSON list of media files in ``files_dir``

Static files support ``HEAD``, single byte ``Range`` requests (206/416; required for
video seeking on the Quest), ``ETag``/``Last-Modified`` revalidation (304) and
``Cache-Control: no-cache``.  Paths are strictly confined to their root folder: ``..``,
encoded dots, backslashes, drive letters, NTFS streams, device names and dot-files are
rejected and there are no directory listings.

``Access-Control-Allow-Origin: *`` is sent for ``/api/*`` only.  The viewer loads scene
assets same-origin, so they deliberately carry no CORS header: otherwise any web page
open in a browser that can reach this server (the desktop browser via localhost, or
Quest Browser after the certificate was accepted) could read the user's photos.
"""

from __future__ import annotations

import errno
import json
import mimetypes
import os
import re
import socket
import socketserver
import ssl
import sys
import threading
from datetime import UTC, datetime
from email.utils import formatdate, parsedate_to_datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlsplit

from .. import APP_NAME, __version__
from ..core.config import get_settings, resource_path
from ..core.errors import VRPhotoError
from ..core.log import get_logger

log = get_logger(__name__)

__all__ = ["ViewerServer", "ServerError", "content_type"]

_CHUNK = 1 << 20
_HANDSHAKE_TIMEOUT = 15.0
_REQUEST_TIMEOUT = 30.0
_MAX_FILE_LIST = 2000

_DISCONNECT_ERRORS: tuple[type[BaseException], ...] = (
    BrokenPipeError,
    ConnectionResetError,
    ConnectionAbortedError,
    TimeoutError,
    ssl.SSLError,
)

_MIME: dict[str, str] = {
    ".html": "text/html",
    ".htm": "text/html",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".css": "text/css",
    ".json": "application/json",
    ".map": "application/json",
    ".webmanifest": "application/manifest+json",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".xml": "application/xml",
    ".svg": "image/svg+xml",
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".bin": "application/octet-stream",
    ".wasm": "application/wasm",
    ".splat": "application/octet-stream",
    ".ksplat": "application/octet-stream",
    ".ply": "application/octet-stream",
    ".spz": "application/octet-stream",
    ".sog": "application/octet-stream",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".avif": "image/avif",
    ".heic": "image/heic",
    ".ktx2": "image/ktx2",
    ".hdr": "image/vnd.radiance",
    ".exr": "image/x-exr",
    ".ico": "image/x-icon",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".wav": "audio/wav",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".otf": "font/otf",
}
_TEXTUAL = ("text/", "application/json", "application/manifest+json", "application/xml", "image/svg+xml",
            "model/gltf+json")
_DOWNLOADABLE = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".avif", ".mp4", ".m4v", ".mov", ".mkv",
                 ".webm", ".glb", ".gltf", ".ply", ".splat", ".spz", ".obj", ".zip", ".jps", ".mpo"}
_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"} | {
    f"{p}{i}" for p in ("COM", "LPT") for i in range(1, 10)
}
_HOST_OK = re.compile(r"^[A-Za-z0-9.\-]+(?::\d{1,5})?$|^\[[0-9A-Fa-f:.]+\](?::\d{1,5})?$")


class ServerError(VRPhotoError):
    """The viewer server could not start (port busy, missing viewer files, ...)."""


def content_type(path: Path | str) -> str:
    """MIME type for a served file (own table first: the Windows registry often maps
    ``.js`` to ``text/plain``, which breaks ES modules).  Text types carry UTF-8."""
    ext = Path(path).suffix.lower()
    ctype = _MIME.get(ext) or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
    if ctype.startswith(_TEXTUAL):
        ctype += "; charset=utf-8"
    return ctype


# --------------------------------------------------------------------------- path safety
class _Forbidden(Exception):
    pass


def _safe_segments(raw_path: str) -> list[str]:
    """Split a URL path into safe, decoded segments; raises :class:`_Forbidden`."""
    if "\\" in raw_path or "%5c" in raw_path.lower():
        raise _Forbidden("backslash")
    path = unquote(raw_path, errors="replace")
    if "\\" in path or "\x00" in path:
        raise _Forbidden("bad character")
    segs: list[str] = []
    for seg in path.split("/"):
        if not seg:
            continue
        if seg in (".", ".."):
            raise _Forbidden("dot segment")
        if ":" in seg or any(ord(c) < 32 for c in seg):
            raise _Forbidden("drive letter / stream / control character")
        if seg[-1] in ". ":
            raise _Forbidden("trailing dot or space")  # Windows would silently strip these
        if seg.split(".")[0].upper().rstrip() in _RESERVED_NAMES:
            raise _Forbidden("device name")
        segs.append(seg)
    return segs


def _split_target(target: str) -> tuple[str, str]:
    """``(path, query)`` of a request target.

    Origin-form targets (``/a/b?x``) are split by hand: ``urlsplit("//x/y")`` would take
    ``x`` for a host name.  Absolute-form targets (``http://host/a``) use ``urlsplit``.
    """
    if target.startswith("/"):
        path, _, query = target.partition("#")[0].partition("?")
        return path, query
    split = urlsplit(target)
    return split.path or "/", split.query


def _resolve_in(root: Path, segs: list[str]) -> Path | None:
    """Existing regular file ``root/segs`` that stays inside ``root`` (after symlinks)."""
    if not segs or any(s.startswith(".") for s in segs):
        return None
    cand = root.joinpath(*segs)
    try:
        real = cand.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        return None
    if not real.is_relative_to(root):
        raise _Forbidden("outside root")
    try:
        return real if real.is_file() else None
    except OSError:
        return None


# --------------------------------------------------------------------------- range / caching
_IGNORE = object()


def _parse_range(header: str, size: int) -> Any:
    """Return ``(start, end)`` (inclusive), ``None`` for 416, or ``_IGNORE`` (serve 200)."""
    unit, sep, spec = header.strip().partition("=")
    if not sep or unit.strip().lower() != "bytes":
        return _IGNORE
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        return None
    if len(parts) > 1:
        return _IGNORE  # multi-range is optional (RFC 9110); browsers never send it
    m = re.fullmatch(r"(\d*)\s*-\s*(\d*)", parts[0])
    if not m or (not m.group(1) and not m.group(2)):
        return None
    first, last = m.groups()
    if not first:  # suffix range: last N bytes
        n = int(last)
        if n <= 0 or size <= 0:
            return None
        return max(0, size - n), size - 1
    start = int(first)
    if start >= size:
        return None
    if last:
        end = int(last)
        if end < start:
            return None
        return start, min(end, size - 1)
    return start, size - 1


def _etag_matches(header: str, etag: str) -> bool:
    if header.strip() == "*":
        return True
    tags = [t.strip() for t in header.split(",")]
    bare = etag.removeprefix("W/")
    return any(t.removeprefix("W/") == bare for t in tags)


def _http_date_ts(value: str) -> float | None:
    try:
        dt = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


# --------------------------------------------------------------------------- scene listing
def _fallback_list_scenes(library: Path) -> list[dict]:
    """Minimal ``list_scenes``: scan ``<library>/*/scene.json`` (used if scene.py is absent)."""
    out: list[dict] = []
    try:
        entries = sorted(library.iterdir())
    except OSError:
        return out
    for d in entries:
        if d.name.startswith(".") or not d.is_dir():
            continue
        sj = d / "scene.json"
        try:
            manifest = json.loads(sj.read_text(encoding="utf-8"))
            mtime = sj.stat().st_mtime
        except (OSError, ValueError):
            continue
        if not isinstance(manifest, dict):
            continue
        sid = d.name
        thumb = manifest.get("thumbnail")
        thumb_url = None
        for name in dict.fromkeys([thumb if isinstance(thumb, str) else "", "thumb.jpg"]):
            try:
                ok = bool(name) and _resolve_in(d.resolve(), _safe_segments(name)) is not None
            except _Forbidden:
                ok = False
            if ok:
                thumb_url = f"/scenes/{quote(sid)}/{quote(name)}"
                break
        created = manifest.get("created")
        if not isinstance(created, str) or not created:
            created = datetime.fromtimestamp(mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        out.append(
            {
                "id": sid,
                "title": str(manifest.get("title") or sid),
                "mode": str(manifest.get("mode") or ""),
                "thumbnail": thumb_url,
                "url": f"/view.html?scene={quote(sid)}",
                "created": created,
            }
        )
    out.sort(key=lambda s: s["created"], reverse=True)
    return out


def _list_scenes(library: Path) -> list[dict]:
    try:
        from .scene import list_scenes  # owned by the viewer/scene engineer; lazy import
    except ImportError:
        return _fallback_list_scenes(library)
    try:
        return list(list_scenes(library))
    except NotImplementedError:
        return _fallback_list_scenes(library)
    except Exception as exc:
        log.warning("scene.list_scenes failed (%s); using a plain folder scan", exc)
        return _fallback_list_scenes(library)


# --------------------------------------------------------------------------- HTTP plumbing
class _HTTPServer(ThreadingHTTPServer):
    """Threaded server with per-connection TLS (handshake off the accept thread),
    no port sharing on Windows and tracked connections for a prompt stop()."""

    daemon_threads = True
    block_on_close = False
    request_queue_size = 64
    # Same policy as socket.create_server(): on Windows SO_REUSEADDR would let us bind a
    # port another program already uses (so "port busy" went undetected), and it is not
    # needed there to rebind while old connections sit in TIME_WAIT.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, addr: tuple[str, int], handler: type, *, app: ViewerServer,
                 ssl_context: ssl.SSLContext | None, local_only: bool):
        self.app = app
        self.ssl_context = ssl_context
        self.local_only = local_only
        self.scheme = "https" if ssl_context is not None else "http"
        self._active: set[socket.socket] = set()
        self._active_lock = threading.Lock()
        if ":" in addr[0]:
            self.address_family = socket.AF_INET6
        super().__init__(addr, handler)

    def server_bind(self) -> None:
        # Skip HTTPServer.server_bind: its getfqdn() can stall for seconds on Windows.
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    def _track(self, sock: socket.socket, add: bool) -> None:
        with self._active_lock:
            (self._active.add if add else self._active.discard)(sock)

    def finish_request(self, request: Any, client_address: Any) -> None:
        if self.ssl_context is None:
            self._track(request, True)
            try:
                self.RequestHandlerClass(request, client_address, self)
            finally:
                self._track(request, False)
            return
        request.settimeout(_HANDSHAKE_TIMEOUT)
        try:
            first = request.recv(1, socket.MSG_PEEK)
        except OSError:
            return
        if not first:
            return
        if first != b"\x16":  # not a TLS handshake record: someone typed http://ip:8443
            _RedirectHandler(request, client_address, self)
            return
        try:
            tls = self.ssl_context.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError) as exc:
            # Normal the first time: Quest Browser rejects the self-signed cert until accepted.
            log.debug("TLS handshake with %s failed: %s", client_address[0], exc)
            return
        self._track(tls, True)
        try:
            self.RequestHandlerClass(tls, client_address, self)
        finally:
            self._track(tls, False)
            try:
                tls.close()
            except OSError:
                pass

    def close_connections(self) -> None:
        """Abort keep-alive / streaming connections so their threads end promptly."""
        with self._active_lock:
            socks = list(self._active)
        for s in socks:
            try:
                socket.socket.shutdown(s, socket.SHUT_RDWR)  # raw shutdown, also for SSLSocket
            except OSError:
                pass

    def handle_error(self, request: Any, client_address: Any) -> None:
        exc = sys.exc_info()[1]
        if isinstance(exc, _DISCONNECT_ERRORS) or isinstance(exc, OSError):
            log.debug("Connection from %s ended: %s", client_address[0] if client_address else "?", exc)
        else:
            log.warning("Error while serving %s", client_address[0] if client_address else "?", exc_info=True)


class _QuietHandlerMixin:
    """Route http.server logging to our logger and swallow client disconnects."""

    def version_string(self) -> str:
        return f"VRPhotoStudio/{__version__}"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        log.debug("%s %s", self.client_address[0], format % args)  # type: ignore[attr-defined]

    def log_error(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("%s %s", self.client_address[0], format % args)  # type: ignore[attr-defined]

    def handle(self) -> None:
        try:
            super().handle()  # type: ignore[misc]
        except _DISCONNECT_ERRORS as exc:
            self.close_connection = True
            log.debug("Client %s disconnected: %s", self.client_address[0], exc)  # type: ignore[attr-defined]

    def finish(self) -> None:
        try:
            super().finish()  # type: ignore[misc]
        except _DISCONNECT_ERRORS + (OSError,):
            pass


class _RedirectHandler(_QuietHandlerMixin, BaseHTTPRequestHandler):
    """Answers plain HTTP on the HTTPS port with a redirect to the https:// URL."""

    protocol_version = "HTTP/1.0"
    timeout = 10

    def _redirect(self) -> None:
        host = self.headers.get("Host", "")
        if not _HOST_OK.match(host):
            host = f"{self.server.app.primary_host()}:{self.server.server_port}"  # type: ignore[attr-defined]
        path = self.path if self.path.startswith("/") else "/"
        path = re.sub(r"[\r\n]", "", path)
        body = b"This server only speaks HTTPS.\n"
        self.send_response(HTTPStatus.MOVED_PERMANENTLY)
        self.send_header("Location", f"https://{host}{path}")
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = do_HEAD = do_POST = do_PUT = do_DELETE = do_OPTIONS = _redirect


class _Handler(_QuietHandlerMixin, BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = _REQUEST_TIMEOUT
    server: _HTTPServer

    # ------------------------------------------------------------------ methods
    def do_GET(self) -> None:
        self._dispatch(head=False)

    def do_HEAD(self) -> None:
        self._dispatch(head=True)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type, If-None-Match, If-Range")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _not_allowed(self) -> None:
        # The request body (if any) is not read, so the connection cannot be reused.
        self.close_connection = True
        self._send_error(HTTPStatus.METHOD_NOT_ALLOWED, "Only GET and HEAD are supported", allow=True)

    do_POST = do_PUT = do_DELETE = do_PATCH = _not_allowed

    # ------------------------------------------------------------------ helpers
    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range, Accept-Ranges, ETag")

    def _is_api(self) -> bool:
        return _split_target(self.path)[0].lstrip("/").startswith("api")

    def _send_error(self, code: int, message: str = "", *, allow: bool = False, head: bool = False) -> None:
        status = HTTPStatus(code)
        body = json.dumps({"error": message or status.phrase, "status": int(code)}).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        if allow:
            self.send_header("Allow", "GET, HEAD, OPTIONS")
        if self.close_connection:
            self.send_header("Connection", "close")
        if self._is_api():
            self._cors_headers()
        self.end_headers()
        if not head and self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, obj: Any, *, head: bool) -> None:
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self._cors_headers()
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def _host_allowed(self) -> bool:
        """DNS-rebinding guard for the localhost server: only IP literals / localhost."""
        if not self.server.local_only:
            return True
        host = self.headers.get("Host", "")
        if host.startswith("["):
            return True  # IPv6 literal
        name = host.rsplit(":", 1)[0].lower() if host.count(":") == 1 else host.lower()
        if name in ("", "localhost") or name.endswith(".localhost"):
            return True
        return bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", name))

    # ------------------------------------------------------------------ routing
    def _dispatch(self, head: bool) -> None:
        app = self.server.app
        try:
            if not self._host_allowed():
                self._send_error(HTTPStatus.FORBIDDEN, "Unknown host name", head=head)
                return
            raw_path, raw_query = _split_target(self.path)
            segs = _safe_segments(raw_path or "/")
            query = parse_qs(raw_query)
            if not segs:
                segs = ["index.html"]
            if segs[0] == "api":
                self._api(segs[1:], head=head)
                return
            download = False
            if segs[0] == "scenes":
                path = _resolve_in(app.library_root, segs[1:])
            elif segs[0] == "files" and app.files_root is not None:
                path = _resolve_in(app.files_root, segs[1:])
                download = query.get("download", ["0"])[0].lower() in ("1", "true", "yes")
            else:
                path = _resolve_in(app.viewer_root, segs)
            if path is None:
                self._send_error(HTTPStatus.NOT_FOUND, "Not found", head=head)
                return
            # same-origin content: no CORS header (see module docstring)
            self._serve_file(path, head=head, cors=False, download=download)
        except _Forbidden as exc:
            log.debug("Rejected path %r from %s: %s", self.path, self.client_address[0], exc)
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden", head=head)
        except _DISCONNECT_ERRORS:
            self.close_connection = True
        except Exception:
            log.exception("Viewer server error for %s", self.path)
            self.close_connection = True
            try:
                self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Internal server error", head=head)
            except OSError:
                pass

    def _api(self, segs: list[str], *, head: bool) -> None:
        app = self.server.app
        if segs == ["scenes"]:
            self._send_json(_list_scenes(app.library_root), head=head)
        elif segs == ["info"]:
            host = self.headers.get("Host", "").lower()
            secure = self.server.scheme == "https" or host.startswith(("localhost", "127.", "[::1]"))
            info = {
                "app": APP_NAME,
                "version": __version__,
                "urls": app.urls(),
                "https_port": app.https_port,
                "http_port": app.http_port,
                "scheme": self.server.scheme,
                "secure_context": secure,
                "files": app.files_root is not None,
                "time": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            self._send_json(info, head=head)
        elif segs == ["files"] and app.files_root is not None:
            self._send_json(app.list_files(), head=head)
        else:
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint", head=head)

    # ------------------------------------------------------------------ files
    def _serve_file(self, path: Path, *, head: bool, cors: bool, download: bool) -> None:
        try:
            st = path.stat()
        except OSError:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found", head=head)
            return
        size = st.st_size
        etag = f'"{size:x}-{st.st_mtime_ns:x}"'
        last_modified = formatdate(st.st_mtime, usegmt=True)

        inm = self.headers.get("If-None-Match")
        ims = self.headers.get("If-Modified-Since")
        not_modified = False
        if inm is not None:
            not_modified = _etag_matches(inm, etag)
        elif ims:
            ts = _http_date_ts(ims)
            not_modified = ts is not None and int(st.st_mtime) <= ts
        if not_modified:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Cache-Control", "no-cache")
            if cors:
                self._cors_headers()
            self.end_headers()
            return

        start, end, status = 0, size - 1, HTTPStatus.OK
        rng = self.headers.get("Range")
        if rng and self._if_range_ok(etag, st.st_mtime):
            parsed = _parse_range(rng, size)
            if parsed is None:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.send_header("Accept-Ranges", "bytes")
                if cors:
                    self._cors_headers()
                self.end_headers()
                return
            if parsed is not _IGNORE:
                start, end = parsed
                status = HTTPStatus.PARTIAL_CONTENT
        length = max(0, end - start + 1)

        try:
            fh = open(path, "rb")  # noqa: SIM115 - closed below
        except OSError:
            self._send_error(HTTPStatus.NOT_FOUND, "Not found", head=head)
            return
        with fh:
            self.send_response(status)
            self.send_header("Content-Type", content_type(path))
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", last_modified)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            if status == HTTPStatus.PARTIAL_CONTENT:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if cors:
                self._cors_headers()
            if download:
                self.send_header("Content-Disposition", _content_disposition(path.name))
            self.end_headers()
            if head or length == 0:
                return
            self._copy(fh, start, length)

    def _if_range_ok(self, etag: str, mtime: float) -> bool:
        """Honour ``If-Range``: a Range applies only if the validator still matches."""
        cond = self.headers.get("If-Range")
        if not cond:
            return True
        cond = cond.strip()
        if cond.startswith(('"', "W/")):
            return not cond.startswith("W/") and cond == etag
        ts = _http_date_ts(cond)
        return ts is not None and int(mtime) == int(ts)

    def _copy(self, fh: Any, start: int, length: int) -> None:
        conn = self.connection
        if not isinstance(conn, ssl.SSLSocket) and hasattr(conn, "sendfile"):
            try:
                sent = conn.sendfile(fh, start, length)
            except _DISCONNECT_ERRORS:
                self.close_connection = True
                return
            if sent < length:
                self.close_connection = True  # file shrank while sending
            return
        fh.seek(start)
        remaining = length
        while remaining > 0:
            chunk = fh.read(min(_CHUNK, remaining))
            if not chunk:
                self.close_connection = True
                return
            try:
                self.wfile.write(chunk)
            except _DISCONNECT_ERRORS:
                self.close_connection = True
                return
            remaining -= len(chunk)


def _content_disposition(name: str) -> str:
    ascii_name = re.sub(r'[^A-Za-z0-9._ -]', "_", name) or "download"
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(name)}"


def _make_ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(str(cert), str(key))
    try:
        ctx.set_alpn_protocols(["http/1.1"])
    except (NotImplementedError, ssl.SSLError):
        pass
    return ctx


_PORT_BUSY_ERRNOS = {errno.EADDRINUSE, errno.EACCES, 10048, 10013}  # incl. WSAEADDRINUSE / WSAEACCES


# --------------------------------------------------------------------------- public class
class ViewerServer:
    """Serves the WebXR viewer and the scene library over HTTPS (LAN) and HTTP (localhost).

    Typical use::

        srv = ViewerServer(settings.library_dir, https_port=settings.https_port,
                           http_port=settings.http_port)
        srv.start()               # raises ServerError (a VRPhotoError) with .hint
        print(srv.urls())         # {"https": "https://192.168.1.5:8443/", "local": "http://localhost:8080/"}
        srv.stop()

    Port ``0`` picks a free port (``urls()`` then reports the real one); ``None``
    disables that listener.  ``files_dir`` (optional, not in the frozen contract) also
    exposes a folder of finished outputs under ``/files/`` for download in Quest Browser.
    ``start``/``stop`` are thread-safe and idempotent; the instance can be restarted.
    """

    def __init__(
        self,
        library: Path,
        *,
        https_port: int | None = 8443,
        http_port: int | None = 8080,
        viewer_dir: Path | None = None,
        cert_dir: Path | None = None,
        bind: str = "0.0.0.0",
        files_dir: Path | None = None,
    ) -> None:
        if https_port is None and http_port is None:
            raise ServerError("At least one of the HTTPS and HTTP servers must be enabled")
        if https_port and http_port and int(https_port) == int(http_port):
            raise ServerError(
                f"The HTTPS and HTTP viewer servers cannot both use port {https_port}.",
                hint="Choose two different ports in Settings (defaults: 8443 for HTTPS, 8080 for HTTP).",
            )
        self.library = Path(library)
        self.viewer_dir = Path(viewer_dir) if viewer_dir is not None else resource_path("webxr", "viewer")
        self.cert_dir = Path(cert_dir) if cert_dir is not None else None
        self.files_dir = Path(files_dir) if files_dir is not None else None
        self.bind = bind or "0.0.0.0"
        self._https_port_cfg = https_port
        self._http_port_cfg = http_port
        self._lock = threading.RLock()
        self._servers: list[_HTTPServer] = []
        self._threads: list[threading.Thread] = []
        self._ports: dict[str, int] = {}
        self._ips: list[str] | None = None
        self._running = False
        # resolved roots (set by start(); usable by handlers)
        self.library_root = self.library
        self.viewer_root = self.viewer_dir
        self.files_root: Path | None = None

    # ------------------------------------------------------------------ properties
    @property
    def running(self) -> bool:
        """True between a successful :meth:`start` and :meth:`stop`."""
        return self._running

    @property
    def https_port(self) -> int | None:
        """Actual HTTPS port while running (else the configured one); None if disabled."""
        return self._ports.get("https", self._https_port_cfg)

    @property
    def http_port(self) -> int | None:
        """Actual HTTP (localhost) port while running (else the configured one)."""
        return self._ports.get("http", self._http_port_cfg)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        """Bind both listeners and serve in daemon threads.

        Creates/renews the HTTPS certificate for the current LAN addresses.  Raises
        :class:`ServerError` (with a ``hint``) if a port is busy or the viewer files are
        missing.  Calling start() on a running server does nothing.
        """
        with self._lock:
            if self._running:
                return
            index = self.viewer_dir / "index.html"
            if not index.is_file():
                raise ServerError(
                    f"The VR viewer files are missing ({index}).",
                    hint="Reinstall VR Photo Studio; the 'webxr/viewer' folder must be next to the program.",
                )
            try:
                self.library.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ServerError(f"Cannot create the library folder {self.library}: {exc}") from exc
            self.library_root = self.library.resolve()
            self.viewer_root = self.viewer_dir.resolve()
            self.files_root = self.files_dir.resolve() if self.files_dir and self.files_dir.is_dir() else None

            ssl_ctx = None
            self._ips = self._detect_ips()
            if self._https_port_cfg is not None:
                from .cert import ensure_cert

                cert_dir = self.cert_dir or get_settings().certs_dir
                cert, key = ensure_cert(cert_dir, self._ips)
                try:
                    ssl_ctx = _make_ssl_context(cert, key)
                except (OSError, ssl.SSLError) as exc:
                    raise ServerError(
                        f"The HTTPS certificate could not be loaded: {exc}",
                        hint=f"Delete the folder {cert_dir} and start the server again.",
                    ) from exc

            servers: list[tuple[str, _HTTPServer]] = []
            try:
                if self._https_port_cfg is not None:
                    servers.append(("https", self._bind(self.bind, self._https_port_cfg, ssl_ctx, False)))
                if self._http_port_cfg is not None:
                    servers.append(("http", self._bind("127.0.0.1", self._http_port_cfg, None, True)))
            except BaseException:
                for _, srv in servers:
                    srv.server_close()
                raise

            self._servers = [srv for _, srv in servers]
            self._ports = {name: srv.server_port for name, srv in servers}
            self._threads = []
            for name, srv in servers:
                t = threading.Thread(
                    target=srv.serve_forever, kwargs={"poll_interval": 0.1},
                    name=f"vrphoto-viewer-{name}", daemon=True,
                )
                t.start()
                self._threads.append(t)
            self._running = True
            log.info("Viewer server started: %s", ", ".join(self.urls().values()))

    def stop(self) -> None:
        """Stop serving, close the sockets and abort open connections.  Idempotent."""
        with self._lock:
            servers, threads = self._servers, self._threads
            self._servers, self._threads = [], []
            was_running = self._running
            self._running = False
            # shutdown() waits for the serve loop's next poll; stop all listeners in parallel
            stoppers = [threading.Thread(target=srv.shutdown, daemon=True) for srv in servers]
            for t in stoppers:
                t.start()
            for t in stoppers:
                t.join(timeout=5)
            for srv in servers:
                srv.server_close()
                srv.close_connections()
            for t in threads:
                t.join(timeout=5)
            self._ports = {}
            if was_running:
                log.info("Viewer server stopped")

    def __enter__(self) -> ViewerServer:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------ URLs
    def _detect_ips(self) -> list[str]:
        if self.bind not in ("0.0.0.0", "::", ""):
            return [self.bind]
        from .net import lan_ips

        try:
            return lan_ips()
        except Exception:  # never let address detection break the server
            log.debug("LAN address detection failed", exc_info=True)
            return []

    def primary_host(self) -> str:
        """Host part of the main HTTPS URL (first LAN IP, else ``localhost``)."""
        ips = self._ips if self._ips is not None else self._detect_ips()
        return ips[0] if ips else "localhost"

    def lan_urls(self) -> list[str]:
        """HTTPS URLs for every LAN address (best first); empty if HTTPS is disabled."""
        port = self.https_port
        if port is None:
            return []
        ips = self._ips if self._ips is not None else self._detect_ips()
        hosts = [f"[{ip}]" if ":" in ip else ip for ip in ips] or ["localhost"]
        return [f"https://{h}:{port}/" for h in hosts]

    def urls(self) -> dict[str, str]:
        """``{"https": main LAN URL, "local": "http://localhost:<port>/"}``.

        Additional LAN addresses are listed as ``"https_2"``, ``"https_3"``...  Keys for
        disabled listeners are omitted.  While running, the actual ports are reported.
        """
        out: dict[str, str] = {}
        lan = self.lan_urls()
        if lan:
            out["https"] = lan[0]
            for i, url in enumerate(lan[1:], start=2):
                out[f"https_{i}"] = url
        if self.http_port is not None:
            out["local"] = f"http://localhost:{self.http_port}/"
        return out

    # ------------------------------------------------------------------ files listing
    def list_files(self) -> list[dict]:
        """Media files under ``files_dir`` (newest first, at most 2000) for ``/api/files``."""
        root = self.files_root
        if root is None:
            return []
        items: list[dict] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
            for name in filenames:
                p = Path(dirpath) / name
                if name.startswith(".") or p.suffix.lower() not in _DOWNLOADABLE:
                    continue
                try:
                    st = p.stat()
                    rel = p.relative_to(root).as_posix()
                except (OSError, ValueError):
                    continue
                url = "files/" + quote(rel)
                items.append({
                    "name": rel,
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime, UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "url": url,
                    "download": url + "?download=1",
                })
                if len(items) >= _MAX_FILE_LIST:
                    break
            if len(items) >= _MAX_FILE_LIST:
                break
        items.sort(key=lambda d: d["modified"], reverse=True)
        return items

    # ------------------------------------------------------------------ internals
    def _bind(self, host: str, port: int, ctx: ssl.SSLContext | None, local_only: bool) -> _HTTPServer:
        try:
            return _HTTPServer((host, int(port)), _Handler, app=self, ssl_context=ctx, local_only=local_only)
        except OSError as exc:
            kind = "HTTPS" if ctx is not None else "HTTP"
            if exc.errno in _PORT_BUSY_ERRNOS or getattr(exc, "winerror", None) in (10048, 10013):
                raise ServerError(
                    f"Port {port} is already in use or reserved, so the {kind} viewer server cannot start.",
                    hint=(
                        "Close the other program that uses this port (or another running copy of "
                        "VR Photo Studio), or choose a different port in Settings. On Windows, Hyper-V/WSL "
                        "may reserve ports: check with 'netsh interface ipv4 show excludedportrange "
                        "protocol=tcp'."
                    ),
                ) from exc
            raise ServerError(
                f"The {kind} viewer server cannot listen on {host}:{port}: {exc.strerror or exc}",
                hint="Check the server address and port in Settings.",
            ) from exc

