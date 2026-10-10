"""End-to-end tests for vrphoto.webxr.server.ViewerServer (ephemeral ports)."""

from __future__ import annotations

import http.client
import json
import os
import socket
import ssl
import sys
import threading
import time
import types
from email.utils import formatdate
from pathlib import Path

import pytest

from vrphoto.core.errors import VRPhotoError
from vrphoto.webxr import net
from vrphoto.webxr import server as srvmod
from vrphoto.webxr.server import ServerError, ViewerServer, content_type

INDEX = b"<!doctype html><title>Gallery</title>"
VIDEO = bytes(range(256)) * 1200  # 307200 bytes with a recognisable pattern
SECRET = b"TOP-SECRET-CONTENT"


# --------------------------------------------------------------------------- fixtures / helpers
@pytest.fixture
def site(tmp_path) -> dict[str, Path]:
    viewer = tmp_path / "viewer"
    (viewer / "js").mkdir(parents=True)
    (viewer / "css").mkdir()
    (viewer / "vendor").mkdir()
    (viewer / "index.html").write_bytes(INDEX)
    (viewer / "view.html").write_bytes(b"<!doctype html><title>View</title>")
    (viewer / "js" / "app.js").write_text("export const x = 1;\n", encoding="utf-8")
    (viewer / "js" / "mod.mjs").write_text("export default 2;\n", encoding="utf-8")
    (viewer / "css" / "viewer.css").write_text("body{}", encoding="utf-8")
    (viewer / "vendor" / "lib.wasm").write_bytes(b"\0asm\1\0\0\0")
    (viewer / "icon.svg").write_text("<svg/>", encoding="utf-8")
    (viewer / ".hidden").write_text("hidden", encoding="utf-8")

    library = tmp_path / "library"
    scene = library / "beach-01"
    scene.mkdir(parents=True)
    manifest = {
        "schema": "vrphoto.scene/1",
        "id": "beach-01",
        "title": "Beach",
        "created": "2026-10-09T12:00:00Z",
        "thumbnail": "thumb.jpg",
        "mode": "stereo",
        "stereo": {"media": "video_LR.mp4", "kind": "video", "layout": "sbs"},
    }
    (scene / "scene.json").write_text(json.dumps(manifest), encoding="utf-8")
    (scene / "video_LR.mp4").write_bytes(VIDEO)
    (scene / "thumb.jpg").write_bytes(b"\xff\xd8\xff\xe0jpeg")
    (scene / "bg.glb").write_bytes(b"glTF\x02\x00\x00\x00")
    (scene / "x.splat").write_bytes(b"\0" * 64)
    (scene / "x.ply").write_bytes(b"ply\n")
    (scene / "x.spz").write_bytes(b"\0" * 8)
    (scene / "photo.png").write_bytes(b"\x89PNG")
    (scene / "photo.webp").write_bytes(b"RIFF")
    (tmp_path / "secret.txt").write_bytes(SECRET)
    return {"root": tmp_path, "viewer": viewer, "library": library, "scene": scene,
            "secret": tmp_path / "secret.txt"}


@pytest.fixture
def server(site, tmp_path):
    srv = ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "certs", bind="127.0.0.1")
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


def _ctx(tmp_path: Path) -> ssl.SSLContext:
    return ssl.create_default_context(cafile=str(tmp_path / "certs" / "cert.pem"))


def req(port: int, method: str, path: str, headers: dict | None = None, *,
        ctx: ssl.SSLContext | None = None) -> tuple[int, dict[str, str], bytes]:
    if ctx is None:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    else:
        conn = http.client.HTTPSConnection("127.0.0.1", port, timeout=10, context=ctx)
    try:
        conn.request(method, path, headers=headers or {})
        r = conn.getresponse()
        body = r.read()
        return r.status, {k.lower(): v for k, v in r.getheaders()}, body
    finally:
        conn.close()


def get(srv: ViewerServer, path: str, headers: dict | None = None):
    return req(srv.http_port, "GET", path, headers)


# --------------------------------------------------------------------------- basics
def test_index_and_static_files(server):
    st, h, body = get(server, "/")
    assert st == 200 and body == INDEX
    assert h["content-type"] == "text/html; charset=utf-8"
    assert h["content-length"] == str(len(INDEX))
    assert h["accept-ranges"] == "bytes"
    assert get(server, "/index.html")[2] == INDEX
    st, _, body = get(server, "/view.html?scene=beach-01")
    assert st == 200 and b"View" in body


@pytest.mark.parametrize(
    "path,ctype",
    [
        ("/js/app.js", "text/javascript; charset=utf-8"),
        ("/js/mod.mjs", "text/javascript; charset=utf-8"),
        ("/css/viewer.css", "text/css; charset=utf-8"),
        ("/vendor/lib.wasm", "application/wasm"),
        ("/icon.svg", "image/svg+xml; charset=utf-8"),
        ("/scenes/beach-01/scene.json", "application/json; charset=utf-8"),
        ("/scenes/beach-01/bg.glb", "model/gltf-binary"),
        ("/scenes/beach-01/video_LR.mp4", "video/mp4"),
        ("/scenes/beach-01/thumb.jpg", "image/jpeg"),
        ("/scenes/beach-01/photo.png", "image/png"),
        ("/scenes/beach-01/photo.webp", "image/webp"),
        ("/scenes/beach-01/x.splat", "application/octet-stream"),
        ("/scenes/beach-01/x.ply", "application/octet-stream"),
        ("/scenes/beach-01/x.spz", "application/octet-stream"),
    ],
)
def test_mime_types(server, path, ctype):
    st, h, _ = get(server, path)
    assert st == 200
    assert h["content-type"] == ctype
    assert h["x-content-type-options"] == "nosniff"


def test_content_type_helper():
    assert content_type("a/B.JS") == "text/javascript; charset=utf-8"
    assert content_type("x.glb") == "model/gltf-binary"
    assert content_type("x.unknownext") == "application/octet-stream"
    assert content_type(Path("m.json")) == "application/json; charset=utf-8"


def test_json_files_have_validators_and_no_cache(server):
    st, h, body = get(server, "/scenes/beach-01/scene.json")
    assert st == 200 and json.loads(body)["id"] == "beach-01"
    assert h["cache-control"] == "no-cache"
    assert h["etag"].startswith('"') and h["last-modified"].endswith("GMT")


def test_cors_only_on_api(server):
    """Scene assets are same-origin only: a foreign web page must not be able to read photos."""
    for path in ("/scenes/beach-01/scene.json", "/scenes/beach-01/video_LR.mp4", "/scenes/beach-01/nope.jpg", "/"):
        _, h, _ = get(server, path, {"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in h, path
    _, h, _ = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": "bytes=0-1"})
    assert "access-control-allow-origin" not in h
    for path in ("/api/scenes", "/api/info", "/api/unknown"):
        _, h, _ = get(server, path, {"Origin": "https://example.org"})
        assert h["access-control-allow-origin"] == "*", path


# --------------------------------------------------------------------------- ranges
@pytest.mark.parametrize(
    "rng,start,end",
    [
        ("bytes=100-199", 100, 199),
        ("bytes=0-0", 0, 0),
        ("bytes=-50", len(VIDEO) - 50, len(VIDEO) - 1),
        ("bytes=307190-", len(VIDEO) - 10, len(VIDEO) - 1),
        ("bytes=300000-999999999", 300000, len(VIDEO) - 1),
        ("bytes=0-", 0, len(VIDEO) - 1),
        ("bytes=-999999999", 0, len(VIDEO) - 1),
    ],
)
def test_range_206_exact_bytes(server, rng, start, end):
    st, h, body = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": rng})
    assert st == 206
    assert h["content-range"] == f"bytes {start}-{end}/{len(VIDEO)}"
    assert int(h["content-length"]) == end - start + 1
    assert body == VIDEO[start : end + 1]


def test_range_over_https_exact_bytes(server, tmp_path):
    st, h, body = req(server.https_port, "GET", "/scenes/beach-01/video_LR.mp4",
                      {"Range": "bytes=1000-200999"}, ctx=_ctx(tmp_path))
    assert st == 206 and body == VIDEO[1000:201000]


@pytest.mark.parametrize("rng", [f"bytes={len(VIDEO)}-", "bytes=500-100", "bytes=abc", "bytes=-0", "bytes="])
def test_range_416(server, rng):
    st, h, body = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": rng})
    assert st == 416
    assert h["content-range"] == f"bytes */{len(VIDEO)}"
    assert body == b""


@pytest.mark.parametrize("rng", ["items=0-10", "bytes=0-10,20-30"])
def test_range_ignored_serves_full(server, rng):
    st, h, body = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": rng})
    assert st == 200 and body == VIDEO


def test_if_range(server):
    _, h, _ = get(server, "/scenes/beach-01/video_LR.mp4")
    etag = h["etag"]
    st, _, body = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": "bytes=0-9", "If-Range": etag})
    assert st == 206 and body == VIDEO[:10]
    st, _, body = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": "bytes=0-9", "If-Range": '"stale"'})
    assert st == 200 and body == VIDEO
    st, _, _ = get(server, "/scenes/beach-01/video_LR.mp4", {"Range": "bytes=0-9", "If-Range": h["last-modified"]})
    assert st == 206


def test_parse_range_unit():
    p = srvmod._parse_range
    assert p("bytes=0-4", 10) == (0, 4)
    assert p("bytes=5-", 10) == (5, 9)
    assert p("bytes=-3", 10) == (7, 9)
    assert p("bytes=10-", 10) is None
    assert p("bytes=0-", 0) is None
    assert p("bytes=1-2,4-5", 10) is srvmod._IGNORE
    assert p("lines=1-2", 10) is srvmod._IGNORE
    assert p("bytes = 2 - 3", 10) == (2, 3)


# --------------------------------------------------------------------------- HEAD / caching / keep-alive
def test_head_has_headers_but_no_body(server):
    conn = http.client.HTTPConnection("127.0.0.1", server.http_port, timeout=10)
    try:
        conn.request("HEAD", "/scenes/beach-01/video_LR.mp4")
        r = conn.getresponse()
        assert r.status == 200
        assert r.getheader("Content-Length") == str(len(VIDEO))
        assert r.getheader("Content-Type") == "video/mp4"
        assert r.read() == b""
        # same keep-alive connection still works (no stray body bytes were sent)
        conn.request("GET", "/index.html")
        r2 = conn.getresponse()
        assert r2.status == 200 and r2.read() == INDEX
        conn.request("HEAD", "/api/scenes")
        r3 = conn.getresponse()
        assert r3.status == 200 and r3.read() == b""
    finally:
        conn.close()


def test_conditional_requests_304(server, site):
    _, h, _ = get(server, "/js/app.js")
    st, h2, body = get(server, "/js/app.js", {"If-None-Match": h["etag"]})
    assert st == 304 and body == b"" and h2["etag"] == h["etag"]
    st, _, _ = get(server, "/js/app.js", {"If-None-Match": '"other", ' + h["etag"]})
    assert st == 304
    st, _, _ = get(server, "/js/app.js", {"If-Modified-Since": h["last-modified"]})
    assert st == 304
    old = formatdate(time.time() - 10 * 86400, usegmt=True)
    os.utime(site["viewer"] / "js" / "app.js", (time.time(), time.time()))
    st, _, body = get(server, "/js/app.js", {"If-Modified-Since": old})
    assert st == 200 and body
    (site["viewer"] / "js" / "app.js").write_text("export const x = 22;\n", encoding="utf-8")
    st, h3, _ = get(server, "/js/app.js", {"If-None-Match": h["etag"]})
    assert st == 200 and h3["etag"] != h["etag"]


# --------------------------------------------------------------------------- path safety
TRAVERSALS = [
    "/../secret.txt",
    "/%2e%2e/secret.txt",
    "/%2E%2E/%2e%2e/secret.txt",
    "/scenes/%2e%2e/%2e%2e/secret.txt",
    "/scenes/../../secret.txt",
    "/..%2fsecret.txt",
    "/scenes/beach-01/..%2f..%2f..%2fsecret.txt",
    "/scenes/beach-01/..\\..\\..\\secret.txt",
    "/%5c..%5c..%5csecret.txt",
    "/..%5csecret.txt",
    "//etc/passwd",
    "/C:/Windows/win.ini",
    "/c:%5cwindows%5cwin.ini",
    "/scenes/beach-01/scene.json%00.jpg",
    "/index.html::$DATA",
    "/index.html.",
    "/index.html%20",
    "/con",
    "/scenes/NUL.txt",
    "/.hidden",
    "/js/",
    "/scenes/",
    "/scenes/beach-01/",
    "/scenes/beach-01",
    "/api/../secret.txt",
]


@pytest.mark.parametrize("path", TRAVERSALS)
def test_path_traversal_and_listing_rejected(server, site, path):
    st, _, body = get(server, path)
    assert st in (403, 404), (path, st)
    assert SECRET not in body


def test_absolute_filesystem_path_rejected(server, site):
    p = site["secret"].resolve().as_posix()
    st, _, body = get(server, "/" + p.lstrip("/"))
    assert st in (403, 404) and SECRET not in body
    st, _, body = get(server, "/scenes/" + p.lstrip("/"))
    assert st in (403, 404) and SECRET not in body


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_symlink_escape_rejected(server, site):
    (site["viewer"] / "leak.txt").symlink_to(site["secret"])
    (site["scene"] / "leakdir").symlink_to(site["root"], target_is_directory=True)
    for path in ("/leak.txt", "/scenes/beach-01/leakdir/secret.txt"):
        st, _, body = get(server, path)
        assert st in (403, 404) and SECRET not in body


def test_safe_segments_unit():
    assert srvmod._safe_segments("/a//b/c.js") == ["a", "b", "c.js"]
    assert srvmod._safe_segments("/sp%20ace/%C3%BC.png") == ["sp ace", "ü.png"]
    for bad in ("/a/../b", "/a/%2e%2e/b", "/a\\b", "/a%5cb", "/a/b:c", "/x/COM1.txt", "/a.", "/a/%00"):
        with pytest.raises(srvmod._Forbidden):
            srvmod._safe_segments(bad)


def test_split_target_unit():
    st = srvmod._split_target
    assert st("/a/b.js?x=1#frag") == ("/a/b.js", "x=1")
    assert st("//scenes/x/y.json?q") == ("//scenes/x/y.json", "q")  # not a host name
    assert st("/view.html?scene=a?b") == ("/view.html", "scene=a?b")
    assert st("http://192.168.1.5:8443/api/scenes?x=2") == ("/api/scenes", "x=2")
    assert st("http://host") == ("/", "")


def test_double_slash_and_absolute_form_targets(server):
    st, _, body = get(server, "//scenes/beach-01/scene.json")
    assert st == 200 and json.loads(body)["id"] == "beach-01"
    st, _, body = get(server, "//api/scenes")
    assert st == 200 and isinstance(json.loads(body), list)
    s = socket.create_connection(("127.0.0.1", server.http_port), timeout=5)
    try:
        s.sendall(b"GET http://localhost/index.html HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    finally:
        s.close()
    assert data.startswith(b"HTTP/1.1 200") and data.endswith(INDEX)


def test_unicode_file_names(server, site):
    (site["scene"] / "Strand ü.jpg").write_bytes(b"uu")
    st, _, body = get(server, "/scenes/beach-01/Strand%20%C3%BC.jpg")
    assert st == 200 and body == b"uu"


# --------------------------------------------------------------------------- API
def test_api_scenes_fallback_lists_scene(server, monkeypatch):
    monkeypatch.setitem(sys.modules, "vrphoto.webxr.scene", None)  # import -> ImportError
    st, h, body = get(server, "/api/scenes")
    assert st == 200
    assert h["content-type"] == "application/json; charset=utf-8"
    assert h["access-control-allow-origin"] == "*"
    assert h["cache-control"] == "no-store"
    scenes = json.loads(body)
    assert [s["id"] for s in scenes] == ["beach-01"]
    s = scenes[0]
    assert s["title"] == "Beach" and s["mode"] == "stereo"
    assert s["thumbnail"] == "/scenes/beach-01/thumb.jpg"
    assert s["url"] == "/view.html?scene=beach-01"
    assert s["created"] == "2026-10-09T12:00:00Z"
    # the thumbnail URL is servable
    assert get(server, s["thumbnail"])[0] == 200


def test_fallback_skips_broken_and_sorts(site):
    lib = site["library"]
    (lib / "broken").mkdir()
    (lib / "broken" / "scene.json").write_text("{not json", encoding="utf-8")
    (lib / "nojson").mkdir()
    (lib / ".tmp-export").mkdir()
    (lib / ".tmp-export" / "scene.json").write_text("{}", encoding="utf-8")
    newer = lib / "newer"
    newer.mkdir()
    (newer / "scene.json").write_text(json.dumps({"mode": "sphere", "created": "2026-10-10T00:00:00Z"}),
                                      encoding="utf-8")
    (newer / "thumb.jpg").write_bytes(b"x")
    scenes = srvmod._fallback_list_scenes(lib)
    assert [s["id"] for s in scenes] == ["newer", "beach-01"]
    assert scenes[0]["title"] == "newer" and scenes[0]["thumbnail"] == "/scenes/newer/thumb.jpg"


def test_api_scenes_uses_scene_module(server, monkeypatch, site):
    seen = {}

    def fake_list(library=None):
        seen["library"] = library
        return [{"id": "abc", "title": "From module", "mode": "photo3d", "thumbnail": None,
                 "url": "/view.html?scene=abc", "created": "2026-01-01T00:00:00Z"}]

    monkeypatch.setitem(sys.modules, "vrphoto.webxr.scene", types.SimpleNamespace(list_scenes=fake_list))
    scenes = json.loads(get(server, "/api/scenes")[2])
    assert scenes[0]["title"] == "From module"
    assert Path(seen["library"]) == site["library"].resolve()


@pytest.mark.parametrize("exc", [NotImplementedError, RuntimeError])
def test_api_scenes_module_failure_falls_back(server, monkeypatch, exc):
    def broken(library=None):
        raise exc("nope")

    monkeypatch.setitem(sys.modules, "vrphoto.webxr.scene", types.SimpleNamespace(list_scenes=broken))
    st, _, body = get(server, "/api/scenes")
    assert st == 200 and [s["id"] for s in json.loads(body)] == ["beach-01"]


def test_api_info(server):
    st, h, body = get(server, "/api/info")
    assert st == 200 and h["access-control-allow-origin"] == "*"
    info = json.loads(body)
    from vrphoto import __version__

    assert info["version"] == __version__
    assert info["https_port"] == server.https_port and info["http_port"] == server.http_port
    assert info["urls"] == server.urls()
    assert info["secure_context"] is True  # http://localhost counts as secure
    assert get(server, "/api/nope")[0] == 404


def test_options_and_methods(server):
    st, h, body = req(server.http_port, "OPTIONS", "/api/scenes")
    assert st == 204 and h["access-control-allow-origin"] == "*"
    assert "GET" in h["access-control-allow-methods"]
    st, h, _ = req(server.http_port, "POST", "/api/scenes", {"Content-Length": "0"})
    assert st == 405 and "GET" in h["allow"]
    assert h["connection"] == "close"  # unread request bodies must not poison keep-alive
    st, _, _ = req(server.http_port, "DELETE", "/scenes/beach-01/scene.json")
    assert st == 405


def test_local_server_rejects_foreign_host_header(server):
    st, _, _ = get(server, "/api/scenes", {"Host": "evil.example:80"})
    assert st == 403
    assert get(server, "/api/scenes", {"Host": f"localhost:{server.http_port}"})[0] == 200
    assert get(server, "/api/scenes", {"Host": f"127.0.0.1:{server.http_port}"})[0] == 200


# --------------------------------------------------------------------------- HTTPS
def test_https_with_trusted_cert(server, tmp_path):
    ctx = _ctx(tmp_path)
    st, h, body = req(server.https_port, "GET", "/", ctx=ctx)
    assert st == 200 and body == INDEX
    st, _, body = req(server.https_port, "GET", "/api/scenes", ctx=ctx)
    assert st == 200 and isinstance(json.loads(body), list)
    # foreign Host headers are fine on the LAN server
    st, _, _ = req(server.https_port, "GET", "/", {"Host": "192.168.1.5:8443"}, ctx=ctx)
    assert st == 200


def test_https_untrusted_client_fails_but_server_survives(server, tmp_path):
    with pytest.raises(ssl.SSLError):
        req(server.https_port, "GET", "/", ctx=ssl.create_default_context())
    st, _, _ = req(server.https_port, "GET", "/", ctx=_ctx(tmp_path))
    assert st == 200


def test_plain_http_to_https_port_redirects(server):
    st, h, _ = req(server.https_port, "GET", "/view.html?scene=beach-01",
                   {"Host": f"127.0.0.1:{server.https_port}"})
    assert st == 301
    assert h["location"] == f"https://127.0.0.1:{server.https_port}/view.html?scene=beach-01"


def test_silent_tcp_client_does_not_block_others(server, tmp_path):
    idle = socket.create_connection(("127.0.0.1", server.https_port), timeout=5)
    try:
        st, _, _ = req(server.https_port, "GET", "/", ctx=_ctx(tmp_path))
        assert st == 200
    finally:
        idle.close()


# --------------------------------------------------------------------------- lifecycle
def test_urls_report_actual_ports(site, tmp_path, monkeypatch):
    monkeypatch.setattr(net, "lan_ips", lambda: ["192.0.2.10", "198.51.100.7"])
    srv = ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "certs")
    srv.start()
    try:
        assert srv.running
        hp, lp = srv.https_port, srv.http_port
        assert hp and lp and hp != 0 and lp != 0
        urls = srv.urls()
        assert urls["https"] == f"https://192.0.2.10:{hp}/"
        assert urls["https_2"] == f"https://198.51.100.7:{hp}/"
        assert urls["local"] == f"http://localhost:{lp}/"
        assert srv.lan_urls() == [urls["https"], urls["https_2"]]
        # the certificate covers the advertised LAN addresses
        from vrphoto.webxr.cert import cert_info

        info = cert_info(tmp_path / "certs" / "cert.pem")
        assert {"192.0.2.10", "198.51.100.7", "127.0.0.1"} <= set(info["ips"])
        # bound on all interfaces -> reachable via loopback too
        st, _, _ = req(hp, "GET", "/", ctx=_ctx(tmp_path))
        assert st == 200
    finally:
        srv.stop()
    assert not srv.running


def test_urls_without_lan_and_disabled_listeners(site, tmp_path, monkeypatch):
    monkeypatch.setattr(net, "lan_ips", lambda: [])
    srv = ViewerServer(site["library"], https_port=8443, http_port=None, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "c")
    assert srv.urls() == {"https": "https://localhost:8443/"}
    srv2 = ViewerServer(site["library"], https_port=None, http_port=8080, viewer_dir=site["viewer"])
    assert srv2.urls() == {"local": "http://localhost:8080/"}
    with pytest.raises(ServerError):
        ViewerServer(site["library"], https_port=None, http_port=None)


def test_same_port_for_both_listeners_rejected(site):
    with pytest.raises(ServerError) as ei:
        ViewerServer(site["library"], https_port=8443, http_port=8443, viewer_dir=site["viewer"])
    assert "8443" in str(ei.value) and ei.value.hint
    ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=site["viewer"])  # 0 = any: fine


def test_certificate_error_surfaces_as_vrphoto_error(site, tmp_path, monkeypatch):
    from vrphoto.webxr import cert as certmod

    def broken(cert_dir, ips=None):
        raise certmod.CertificateError("Cannot write the HTTPS certificate", hint="Check the folder.")

    monkeypatch.setattr(certmod, "ensure_cert", broken)
    srv = ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "certs", bind="127.0.0.1")
    with pytest.raises(VRPhotoError) as ei:
        srv.start()
    assert ei.value.hint and not srv.running


def test_http_only_server_needs_no_certificate(site, tmp_path):
    srv = ViewerServer(site["library"], https_port=None, http_port=0, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "nocerts")
    with srv:
        assert get(srv, "/")[0] == 200
    assert not (tmp_path / "nocerts").exists()


def test_port_busy_raises_friendly_error(site, tmp_path):
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    busy = blocker.getsockname()[1]
    try:
        srv = ViewerServer(site["library"], https_port=0, http_port=busy, viewer_dir=site["viewer"],
                           cert_dir=tmp_path / "certs", bind="127.0.0.1")
        with pytest.raises(VRPhotoError) as ei:
            srv.start()
        assert isinstance(ei.value, ServerError)
        assert str(busy) in str(ei.value) and ei.value.hint
        assert not srv.running
    finally:
        blocker.close()


def test_second_server_on_same_port_fails_first_keeps_working(server, site, tmp_path):
    other = ViewerServer(site["library"], https_port=server.https_port, http_port=0,
                         viewer_dir=site["viewer"], cert_dir=tmp_path / "certs", bind="127.0.0.1")
    with pytest.raises(ServerError):
        other.start()
    assert not other.running
    assert get(server, "/")[0] == 200
    assert req(server.https_port, "GET", "/", ctx=_ctx(tmp_path))[0] == 200


def test_stop_restart_and_idempotence(site, tmp_path):
    srv = ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "certs", bind="127.0.0.1")
    srv.stop()  # stop before start is harmless
    srv.start()
    srv.start()  # no-op
    port = srv.http_port
    assert get(srv, "/")[0] == 200
    srv.stop()
    srv.stop()
    assert not srv.running
    with pytest.raises(OSError):
        req(port, "GET", "/")
    srv.start()
    try:
        assert srv.running and get(srv, "/")[0] == 200
    finally:
        srv.stop()


def test_stop_aborts_open_keepalive_connections(site, tmp_path):
    srv = ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=site["viewer"],
                       cert_dir=tmp_path / "certs", bind="127.0.0.1")
    srv.start()
    conn = http.client.HTTPConnection("127.0.0.1", srv.http_port, timeout=10)
    conn.request("GET", "/index.html")
    assert conn.getresponse().read() == INDEX
    t0 = time.monotonic()
    srv.stop()
    assert time.monotonic() - t0 < 5
    with pytest.raises((OSError, http.client.HTTPException)):
        conn.request("GET", "/index.html")
        conn.getresponse().read()
    conn.close()


def test_missing_viewer_files(site, tmp_path):
    srv = ViewerServer(site["library"], https_port=0, http_port=0, viewer_dir=tmp_path / "nope",
                       cert_dir=tmp_path / "certs")
    with pytest.raises(ServerError) as ei:
        srv.start()
    assert ei.value.hint


def test_default_viewer_dir_is_bundled_resource(site):
    srv = ViewerServer(site["library"])
    from vrphoto.core.config import resource_path

    assert srv.viewer_dir == resource_path("webxr", "viewer")


def test_library_folder_is_created(site, tmp_path):
    lib = tmp_path / "new" / "library"
    with ViewerServer(lib, https_port=None, http_port=0, viewer_dir=site["viewer"]) as srv:
        assert lib.is_dir()
        assert json.loads(get(srv, "/api/scenes")[2]) == []


# --------------------------------------------------------------------------- robustness
def test_client_disconnect_mid_transfer_is_quiet(server, site, capfd):
    big = site["scene"] / "big.mp4"
    big.write_bytes(os.urandom(1 << 20) * 8)
    for port, wrap in ((server.http_port, False), (server.https_port, True)):
        raw = socket.create_connection(("127.0.0.1", port), timeout=10)
        if wrap:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            raw = ctx.wrap_socket(raw, server_hostname="127.0.0.1")
        raw.sendall(b"GET /scenes/beach-01/big.mp4 HTTP/1.1\r\nHost: localhost\r\n\r\n")
        assert raw.recv(1024).startswith(b"HTTP/1.1 200")
        raw.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        raw.close()
    time.sleep(0.3)
    assert get(server, "/")[0] == 200
    err = capfd.readouterr().err
    assert "Traceback" not in err and "Exception" not in err


def test_garbage_request_gets_400_and_server_survives(server):
    s = socket.create_connection(("127.0.0.1", server.http_port), timeout=5)
    s.sendall(b"\x00\x01garbage\r\n\r\n")
    data = s.recv(1024)
    s.close()
    assert data.startswith(b"HTTP/") and b" 400 " in data.split(b"\r\n")[0] + b" "
    assert get(server, "/")[0] == 200


def test_concurrent_requests(server):
    results: list[int] = []
    lock = threading.Lock()

    def work(i: int) -> None:
        path = "/scenes/beach-01/video_LR.mp4" if i % 2 else "/api/scenes"
        st = get(server, path, {"Range": "bytes=0-99"} if i % 2 else None)[0]
        with lock:
            results.append(st)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(20)
    assert sorted(set(results)) == [200, 206] and len(results) == 24


# --------------------------------------------------------------------------- optional /files
def test_files_endpoint(site, tmp_path):
    outputs = tmp_path / "outputs"
    (outputs / "sub").mkdir(parents=True)
    (outputs / "beach_LR.mp4").write_bytes(b"m" * 1000)
    (outputs / "sub" / "pano_360.jpg").write_bytes(b"j" * 10)
    (outputs / "notes.txt").write_text("skip", encoding="utf-8")
    (outputs / ".secret.mp4").write_bytes(b"s")
    srv = ViewerServer(site["library"], https_port=None, http_port=0, viewer_dir=site["viewer"],
                       files_dir=outputs)
    with srv:
        st, h, body = get(srv, "/files/beach_LR.mp4?download=1")
        assert st == 200 and body == b"m" * 1000
        assert h["content-disposition"].startswith("attachment;")
        assert "beach_LR.mp4" in h["content-disposition"]
        st, h, _ = get(srv, "/files/sub/pano_360.jpg")
        assert st == 200 and "content-disposition" not in h
        st, _, body = get(srv, "/files/beach_LR.mp4", {"Range": "bytes=10-19"})
        assert st == 206 and len(body) == 10
        listing = json.loads(get(srv, "/api/files")[2])
        names = {item["name"] for item in listing}
        assert names == {"beach_LR.mp4", "sub/pano_360.jpg"}
        assert all(item["download"].endswith("?download=1") for item in listing)
        assert get(srv, "/files/../library/beach-01/scene.json")[0] == 403
        assert json.loads(get(srv, "/api/info")[2])["files"] is True


def test_files_endpoint_disabled_by_default(server):
    assert get(server, "/files/anything.mp4")[0] == 404
    assert get(server, "/api/files")[0] == 404


# --------------------------------------------------------------------------- packaging
def test_serves_real_bundled_viewer(tmp_path):
    with ViewerServer(tmp_path / "lib", https_port=None, http_port=0) as srv:
        st, h, body = get(srv, "/")
        assert st == 200 and b"<html" in body.lower()
        st, h, _ = get(srv, "/js/gallery.js")
        assert st == 200 and h["content-type"].startswith("text/javascript")
        assert get(srv, "/scene.schema.json")[1]["content-type"].startswith("application/json")


def test_package_lazy_exports():
    import vrphoto.webxr as wx

    assert wx.ViewerServer is ViewerServer
    assert callable(wx.ensure_cert) and callable(wx.find_adb) and callable(wx.qr_png)
    assert "ViewerServer" in dir(wx)
    with pytest.raises(AttributeError):
        wx.does_not_exist  # noqa: B018


def test_imports_stay_light():
    import subprocess

    code = (
        "import sys, vrphoto.webxr.server, vrphoto.webxr.quest, vrphoto.webxr.net, vrphoto.webxr.cert\n"
        "bad = [m for m in ('PySide6', 'onnxruntime', 'torch', 'qrcode', 'cryptography') if m in sys.modules]\n"
        "print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == ""
