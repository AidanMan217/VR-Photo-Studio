"""Resumable downloads and safe archive extraction."""

from __future__ import annotations

import hashlib
import http.server
import io
import os
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from vrphoto.addons import fetch
from vrphoto.core.errors import AddonError, Cancelled
from vrphoto.core.types import CancelToken

PAYLOAD = os.urandom(300_000)


class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, bytes] = {}
    hits: list[tuple[str, str | None]] = []
    fail_after: dict[str, int] = {}

    def log_message(self, *args):  # silence
        pass

    def do_GET(self):  # noqa: N802
        data = self.files.get(self.path)
        self.hits.append((self.path, self.headers.get("Range")))
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        start = 0
        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            start = int(rng[6:].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cut = self.fail_after.pop(self.path, None)
        if cut is not None:
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.shutdown(2)  # simulate a dropped connection
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    _Handler.files = {"/f.bin": PAYLOAD}
    _Handler.hits = []
    _Handler.fail_after = {}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture(autouse=True)
def _no_proxy(monkeypatch):
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setattr(fetch, "_sleep", lambda s, c: None)


def test_download_and_checksum(server, tmp_path):
    seen = []
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    p = fetch.download(f"{server}/f.bin", tmp_path / "f.bin", sha256=sha, on_bytes=lambda d, t: seen.append((d, t)))
    assert p.read_bytes() == PAYLOAD
    assert seen[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert not (tmp_path / "f.bin.part").exists()
    # cached file with the right hash: no new request
    n = len(_Handler.hits)
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin", sha256=sha)
    assert len(_Handler.hits) == n


def test_download_resumes_after_drop(server, tmp_path):
    _Handler.fail_after["/f.bin"] = 100_000
    p = fetch.download(f"{server}/f.bin", tmp_path / "f.bin", sha256=hashlib.sha256(PAYLOAD).hexdigest())
    assert p.read_bytes() == PAYLOAD
    ranges = [r for path, r in _Handler.hits if path == "/f.bin"]
    assert ranges[0] is None and ranges[-1] is not None and ranges[-1].startswith("bytes=")


def test_download_resumes_existing_part(server, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(PAYLOAD[:1234])
    fetch.download(f"{server}/f.bin", tmp_path / "f.bin")
    assert (tmp_path / "f.bin").read_bytes() == PAYLOAD
    assert _Handler.hits[-1][1] == "bytes=1234-"


def test_download_checksum_mismatch(server, tmp_path):
    with pytest.raises(AddonError, match="checksum"):
        fetch.download(f"{server}/f.bin", tmp_path / "f.bin", sha256="0" * 64)
    assert not (tmp_path / "f.bin").exists() and not (tmp_path / "f.bin.part").exists()


def test_download_404(server, tmp_path):
    with pytest.raises(AddonError, match="not found"):
        fetch.download(f"{server}/missing.zip", tmp_path / "m.zip")


def test_download_unreachable(tmp_path):
    with pytest.raises(AddonError, match="failed"):
        fetch.download("http://127.0.0.1:9/x.bin", tmp_path / "x.bin", timeout=2)


def test_download_cancel(server, tmp_path):
    token = CancelToken()
    token.cancel()
    with pytest.raises(Cancelled):
        fetch.download(f"{server}/f.bin", tmp_path / "f.bin", cancel=token)


def _zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_extract_zip_strips_top_folder(tmp_path):
    a = tmp_path / "a.zip"
    a.write_bytes(_zip({"repo-abc/": b"", "repo-abc/x.py": b"x", "repo-abc/sub/y.txt": b"y"}))
    dest = fetch.extract(a, tmp_path / "out")
    assert (dest / "x.py").read_bytes() == b"x"
    assert (dest / "sub" / "y.txt").read_bytes() == b"y"
    assert not (tmp_path / "out.extracting").exists()


def test_extract_zip_no_strip_for_flat(tmp_path):
    a = tmp_path / "a.zip"
    a.write_bytes(_zip({"brush_app.exe": b"exe", "LICENSE": b"l"}))
    dest = fetch.extract(a, tmp_path / "out")
    assert (dest / "brush_app.exe").is_file() and (dest / "LICENSE").is_file()


def test_extract_replaces_existing(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "old.txt").write_text("old")
    a = tmp_path / "a.zip"
    a.write_bytes(_zip({"new.txt": b"n"}))
    fetch.extract(a, tmp_path / "out")
    assert not (tmp_path / "out" / "old.txt").exists() and (tmp_path / "out" / "new.txt").exists()


@pytest.mark.parametrize("bad", ["../evil.txt", "/abs/evil.txt", "a/../../evil.txt", "C:/evil.txt",
                                 # Windows: pathlib joins "sub" / "D:evil" as the drive-relative "D:evil"
                                 "sub/D:evil.txt", "a:b.txt", "file.txt:stream", "sub/.../evil.txt",
                                 "sub/. ./evil.txt", "CON", "sub/nul.txt", "lpt1.log"])
def test_extract_rejects_unsafe_zip(tmp_path, bad):
    a = tmp_path / "a.zip"
    a.write_bytes(_zip({"ok.txt": b"1", bad: b"2"}))
    with pytest.raises(AddonError, match="unsafe"):
        fetch.extract(a, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
    assert not (tmp_path / "out").exists()


def test_extract_tar_xz(tmp_path):
    a = tmp_path / "brush.tar.xz"
    with tarfile.open(a, "w:xz") as tf:
        for name, data in {"brush-app-x86_64/brush_app": b"#!bin", "brush-app-x86_64/README.md": b"r"}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
        link = tarfile.TarInfo("brush-app-x86_64/evil")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        tf.addfile(link)
    dest = fetch.extract(a, tmp_path / "brush")
    assert (dest / "brush_app").read_bytes() == b"#!bin"
    assert not (dest / "evil").exists()
    if os.name != "nt":
        assert os.access(dest / "brush_app", os.X_OK)


def test_extract_damaged(tmp_path):
    a = tmp_path / "a.zip"
    a.write_bytes(b"PK\x03\x04garbage")
    with pytest.raises(AddonError):
        fetch.extract(a, tmp_path / "out")


def test_dir_size_and_rmtree(tmp_path):
    d = tmp_path / "d"
    (d / "s").mkdir(parents=True)
    (d / "s" / "a").write_bytes(b"12345")
    ro = d / "ro.txt"
    ro.write_bytes(b"123")
    ro.chmod(0o444)
    assert fetch.dir_size(d) == 8
    fetch.rmtree(d)
    assert not d.exists()
    fetch.rmtree(d)  # missing is fine
    assert fetch.dir_size(Path(tmp_path / "missing")) == 0


# ------------------------------------------------------------------ review regressions
def test_windows_drive_relative_member_would_escape():
    """Documents why ':' is refused: on Windows the member lands outside the folder."""
    from pathlib import PureWindowsPath

    assert str(PureWindowsPath("C:/out").joinpath("sub", "D:evil")) == "D:evil"
    assert fetch._safe_rel("sub/D:evil") is None
    assert fetch._safe_rel("sub/ok.txt").as_posix() == "sub/ok.txt"
    assert fetch._safe_rel("./a/./b.txt").as_posix() == "a/b.txt"
    assert fetch._safe_rel("console/readme.txt").as_posix() == "console/readme.txt"
    assert fetch._safe_rel("sub/nul.txt") is None  # device names are not files on Windows


def _tar(path: Path, members: list[tuple[str, str, bytes | str]]) -> Path:
    """members: (name, kind, data-or-linkname) with kind 'f' file, 'l' symlink, 'h' hard link."""
    with tarfile.open(path, "w:gz") as tf:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            if kind == "f":
                data = payload if isinstance(payload, bytes) else payload.encode()
                info.size = len(data)
                info.mode = 0o644
                tf.addfile(info, io.BytesIO(data))
            else:
                info.type = tarfile.SYMTYPE if kind == "l" else tarfile.LNKTYPE
                info.linkname = str(payload)
                tf.addfile(info)
    return path


@pytest.mark.skipif(os.name == "nt", reason="symlinks need a privilege on Windows")
def test_extract_tar_links_inside_are_kept_escaping_ones_dropped(tmp_path):
    a = _tar(tmp_path / "t.tar.gz", [
        ("top/lib/libfoo.so.1", "f", b"elf"),
        ("top/lib/libfoo.so", "l", "libfoo.so.1"),           # ok: same folder
        ("top/bin/libfoo.so", "l", "../lib/libfoo.so.1"),    # ok: '..' but stays inside
        ("top/bin/up", "l", ".."),                           # ok: top/ itself after stripping
        ("top/bin/escape", "l", "../.."),                    # archive root = outside after strip
        ("top/bin/abs", "l", "/etc/passwd"),
        ("top/bin/drive", "l", "C:/Windows/win.ini"),
        ("top/bin/hard", "h", "top/lib/libfoo.so.1"),        # hard link -> copied
        ("top/bin/hard_out", "h", "../outside"),
    ])
    dest = fetch.extract(a, tmp_path / "out")
    assert (dest / "lib" / "libfoo.so").read_bytes() == b"elf"
    assert (dest / "bin" / "libfoo.so").read_bytes() == b"elf"
    assert (dest / "bin" / "hard").read_bytes() == b"elf" and not (dest / "bin" / "hard").is_symlink()
    assert (dest / "bin" / "up").is_symlink()
    for name in ("escape", "abs", "drive", "hard_out"):
        assert not os.path.lexists(dest / "bin" / name), name


@pytest.mark.skipif(os.name == "nt", reason="symlinks need a privilege on Windows")
def test_extract_tar_link_then_file_below_it_stays_inside(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    # the classic two-step attack: a link pointing out, then a file "inside" the link
    a = _tar(tmp_path / "t.tar.gz", [("x/b", "l", "../../outside"), ("x/b/evil.txt", "f", b"pwn")])
    dest = fetch.extract(a, tmp_path / "out", strip=False)
    assert (dest / "x" / "b" / "evil.txt").read_bytes() == b"pwn"
    assert not (dest / "x" / "b").is_symlink()
    assert not list(outside.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="symlinks need a privilege on Windows")
def test_write_target_check_refuses_links_out_of_the_folder(tmp_path):
    root = tmp_path / "root"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    os.symlink(tmp_path / "outside", root / "sub" / "link")
    with pytest.raises(AddonError, match="unsafe"):
        fetch._check_write_target(root, root / "sub" / "link" / "evil.txt", Path("a.tar"))
    os.symlink(tmp_path / "outside" / "f", root / "f")
    with pytest.raises(AddonError, match="unsafe"):
        fetch._check_write_target(root, root / "f", Path("a.tar"))
    fetch._check_write_target(root, root / "sub" / "ok.txt", Path("a.tar"))  # fine


class _ChunkedHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = 0

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        type(self).calls += 1
        self.send_response(200)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        half = len(PAYLOAD) // 2
        self.wfile.write(f"{half:x}\r\n".encode() + PAYLOAD[:half] + b"\r\n")
        if type(self).calls == 1:  # first try: announce a chunk, then hang up mid-way
            self.wfile.write(f"{len(PAYLOAD) - half:x}\r\n".encode() + PAYLOAD[half:half + 10])
            self.wfile.flush()
            self.connection.shutdown(2)
            return
        self.wfile.write(f"{len(PAYLOAD) - half:x}\r\n".encode() + PAYLOAD[half:] + b"\r\n0\r\n\r\n")


def test_download_retries_incomplete_chunked_response(tmp_path):
    """http.client raises IncompleteRead (not an OSError) for a cut chunked body."""
    _ChunkedHandler.calls = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ChunkedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_address[1]}/c.bin"
        p = fetch.download(url, tmp_path / "c.bin", sha256=hashlib.sha256(PAYLOAD).hexdigest())
        assert p.read_bytes() == PAYLOAD
        assert _ChunkedHandler.calls >= 2
    finally:
        srv.shutdown()
        srv.server_close()


def test_stale_part_with_wrong_content_restarts_from_scratch(server, tmp_path):
    (tmp_path / "f.bin.part").write_bytes(b"garbage from an older file" * 100)
    p = fetch.download(f"{server}/f.bin", tmp_path / "f.bin", sha256=hashlib.sha256(PAYLOAD).hexdigest())
    assert p.read_bytes() == PAYLOAD
    ranges = [r for path, r in _Handler.hits if path == "/f.bin"]
    assert ranges[0] is not None and ranges[-1] is None  # resumed first, then a clean download


def test_replace_with_retry_waits_out_sharing_violations(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.sys, "platform", "win32")
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    real = os.replace
    fails = {"n": 3}

    def flaky(a, b):
        if fails["n"]:
            fails["n"] -= 1
            raise PermissionError(13, "The process cannot access the file")
        real(a, b)

    monkeypatch.setattr(fetch.os, "replace", flaky)
    (tmp_path / "a").write_text("x")
    fetch.replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert (tmp_path / "b").read_text() == "x" and fails["n"] == 0


def test_replace_with_retry_posix_raises_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.sys, "platform", "linux")
    calls = []

    def denied(a, b):
        calls.append(1)
        raise PermissionError(13, "denied")

    monkeypatch.setattr(fetch.os, "replace", denied)
    with pytest.raises(PermissionError):
        fetch.replace_with_retry(tmp_path / "a", tmp_path / "b")
    assert len(calls) == 1


def test_long_path_prefix(monkeypatch):
    import ntpath

    assert fetch.long_path("/x/y") == "/x/y" or os.name == "nt"
    monkeypatch.setattr(fetch.sys, "platform", "win32")
    monkeypatch.setattr(fetch.os.path, "abspath", ntpath.abspath)
    assert fetch.long_path(r"C:\Users\me\venv") == "\\\\?\\C:\\Users\\me\\venv"
    assert fetch.long_path(r"\\server\share\x") == "\\\\?\\UNC\\server\\share\\x"
    assert fetch.long_path("\\\\?\\C:\\x") == "\\\\?\\C:\\x"


def test_extract_never_merges_into_an_unremovable_leftover(tmp_path, monkeypatch):
    stale = tmp_path / "out.extracting"
    stale.mkdir()
    (stale / "stale.txt").write_text("old")
    real = fetch.rmtree

    def locked(path):
        if Path(path) == stale:
            raise PermissionError(32, "being used by another process")
        real(path)

    monkeypatch.setattr(fetch, "rmtree", locked)
    a = tmp_path / "a.zip"
    a.write_bytes(_zip({"new.txt": b"n"}))
    dest = fetch.extract(a, tmp_path / "out")
    assert sorted(p.name for p in dest.iterdir()) == ["new.txt"]
