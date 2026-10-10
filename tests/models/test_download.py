"""Model downloader against a local HTTP server (Range, resume, checksums, auth, cancel)."""

from __future__ import annotations

import dataclasses
import errno
import hashlib
import json
import os
import socket
import threading
import time
import urllib.error
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from vrphoto.core.errors import Cancelled, DownloadError, ModelUnavailable
from vrphoto.core.types import CancelToken, ProgressReporter
from vrphoto.models import download, registry
from vrphoto.models.download import (
    delete_model,
    download_url,
    ensure_model,
    is_downloaded,
    model_dir,
    recorded_hashes,
)
from vrphoto.models.registry import ModelSpec

PAYLOAD = bytes(range(256)) * 4096 + b"tail"  # ~1 MiB, not chunk aligned
SHA = hashlib.sha256(PAYLOAD).hexdigest()
MD5 = hashlib.md5(PAYLOAD).hexdigest()


@dataclasses.dataclass
class Behaviour:
    data: bytes = PAYLOAD
    statuses: list[int] = dataclasses.field(default_factory=list)  # served (and popped) first
    cut_after: list[int] = dataclasses.field(default_factory=list)  # per request: bytes then drop
    etag: str | None = '"v1"'
    ignore_range: bool = False
    require_token: str | None = None
    redirect_to: str | None = None
    retry_after: str | None = None
    stall_after: int | None = None  # send this many bytes, then hang for stall_s seconds
    stall_s: float = 0.0
    content_type: str | None = None


class FileServer:
    def __init__(self) -> None:
        self.routes: dict[str, Behaviour] = {}
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_GET(self):  # noqa: N802
                with server.lock:
                    server.requests.append((self.path, {k: v for k, v in self.headers.items()}))
                    b = server.routes.get(self.path)
                    status = b.statuses.pop(0) if b and b.statuses else None
                    cut = b.cut_after.pop(0) if b and b.cut_after else None
                if b is None:
                    self.send_error(404)
                    return
                if b.redirect_to:
                    self.send_response(302)
                    self.send_header("Location", b.redirect_to)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if status is not None:
                    self.send_response(status)
                    if b.retry_after:
                        self.send_header("Retry-After", b.retry_after)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if b.require_token and self.headers.get("Authorization") != f"Bearer {b.require_token}":
                    self.send_error(401)
                    return
                data = b.data
                start = 0
                rng = self.headers.get("Range")
                if_range = self.headers.get("If-Range")
                if rng and not b.ignore_range and (if_range is None or if_range == b.etag):
                    start = int(rng.split("=")[1].split("-")[0])
                    if start >= len(data):
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(data)}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                else:
                    self.send_response(200)
                body = data[start:]
                self.send_header("Content-Length", str(len(body)))
                if b.content_type:
                    self.send_header("Content-Type", b.content_type)
                if b.etag:
                    self.send_header("ETag", b.etag)
                self.end_headers()
                if b.stall_after is not None:
                    try:
                        self.wfile.write(body[: b.stall_after])
                        self.wfile.flush()
                        server.stop.wait(b.stall_s)
                        self.wfile.write(body[b.stall_after :])
                    except OSError:
                        pass  # client gave up (cancel) - expected
                    return
                if cut is not None:
                    self.wfile.write(body[:cut])
                    self.wfile.flush()
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True
        )
        self.thread.start()

    def url(self, path: str, host: str = "127.0.0.1") -> str:
        return f"http://{host}:{self.port}{path}"

    def close(self) -> None:
        self.stop.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server(monkeypatch) -> Iterator[FileServer]:
    for var in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    s = FileServer()
    yield s
    s.close()


@pytest.fixture
def fake_registry(monkeypatch, server):
    """Register test models served by the local server (HF-style and direct URL)."""
    monkeypatch.setenv("HF_ENDPOINT", server.url(""))
    specs = {
        "t-hf": ModelSpec(
            id="t-hf",
            task="depth",
            title="Test HF",
            repo="org/repo",
            revision="main",
            files=("onnx/model.onnx",),
            sha256={"onnx/model.onnx": SHA},
            size_mb=1.0,
            license="MIT",
        ),
        "t-url": ModelSpec(
            id="t-url",
            task="segment",
            title="Test URL",
            repo="",
            revision="v0",
            files=("m.onnx",),
            url=server.url("/dl/m.onnx"),
            sha256={"m.onnx": "md5:" + MD5},
            size_mb=1.0,
            license="MIT",
        ),
        "t-nohash": ModelSpec(
            id="t-nohash",
            task="depth",
            title="No hash",
            repo="org/nohash",
            revision="main",
            files=("model.onnx",),
            size_mb=1.0,
            license="MIT",
        ),
        "t-multi": ModelSpec(
            id="t-multi",
            task="depth",
            title="Multi",
            repo="org/multi",
            revision="main",
            files=("model.onnx", "model.onnx_data"),
            size_mb=2.0,
            license="MIT",
            sha256={"model.onnx": SHA, "model.onnx_data": hashlib.sha256(b"x" * 5000).hexdigest()},
            sizes={"model.onnx": len(PAYLOAD), "model.onnx_data": 5000},
            fp16_file="model_fp16.onnx",
        ),
    }
    monkeypatch.setattr(registry, "MODELS", {**registry.MODELS, **specs})
    server.routes["/org/repo/resolve/main/onnx/model.onnx"] = Behaviour()
    server.routes["/dl/m.onnx"] = Behaviour()
    server.routes["/org/nohash/resolve/main/model.onnx"] = Behaviour()
    server.routes["/org/multi/resolve/main/model.onnx"] = Behaviour()
    server.routes["/org/multi/resolve/main/model.onnx_data"] = Behaviour(data=b"x" * 5000)
    server.routes["/org/multi/resolve/main/model_fp16.onnx"] = Behaviour(data=b"h" * 100)
    return specs


def _range_requests(server: FileServer, path: str) -> list[str | None]:
    return [h.get("Range") for p, h in server.requests if p == path]


def test_ensure_model_downloads_and_verifies(fake_registry, server):
    events: list[tuple[float, str]] = []
    p = ensure_model("t-hf", progress=ProgressReporter(lambda f, m: events.append((f, m))))
    assert p == model_dir("t-hf") / "onnx" / "model.onnx"
    assert p.read_bytes() == PAYLOAD
    assert not p.with_name("model.onnx.part").exists()
    assert not p.with_name("model.onnx.part.json").exists()
    assert is_downloaded("t-hf")
    assert events and events[-1][0] == pytest.approx(1.0)
    assert any("Test HF" in m and "MB" in m for _, m in events)
    assert all(0.0 <= f <= 1.0 for f, _ in events)
    # second call: no network
    n = len(server.requests)
    assert ensure_model("t-hf") == p
    assert len(server.requests) == n


def test_user_agent_and_no_token_by_default(fake_registry, server):
    ensure_model("t-hf")
    hdrs = server.requests[-1][1]
    assert hdrs["User-Agent"].startswith("VRPhotoStudio/")
    assert "Authorization" not in hdrs


def test_direct_url_with_md5(fake_registry, server):
    p = ensure_model("t-url")
    assert p.read_bytes() == PAYLOAD
    assert server.requests[-1][0] == "/dl/m.onnx"


def test_token_sent_to_hf_only_and_dropped_on_cross_host_redirect(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(redirect_to=server.url("/cdn/blob", host="localhost"))
    server.routes["/cdn/blob"] = Behaviour()
    ensure_model("t-hf", token="hf_secret")
    first = [h for p, h in server.requests if p == path][0]
    cdn = [h for p, h in server.requests if p == "/cdn/blob"][0]
    assert first.get("Authorization") == "Bearer hf_secret"
    assert "Authorization" not in cdn
    # never sent to non-HF hosts
    ensure_model("t-url", token="hf_secret")
    assert "Authorization" not in [h for p, h in server.requests if p == "/dl/m.onnx"][-1]


def test_token_from_settings(fake_registry, server):
    from vrphoto.core.config import get_settings

    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(require_token="hf_abc")
    get_settings().hf_token = "hf_abc"
    assert ensure_model("t-hf").read_bytes() == PAYLOAD


def test_resume_after_interrupted_transfer(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(cut_after=[300_000, 200_000])
    p = ensure_model("t-hf")
    assert p.read_bytes() == PAYLOAD
    ranges = _range_requests(server, path)
    assert ranges[0] is None
    assert ranges[1] == "bytes=300000-"
    assert ranges[2] == "bytes=500000-"
    if_range = [h.get("If-Range") for q, h in server.requests if q == path]
    assert if_range[1] == '"v1"'


def test_resume_across_calls_after_cancel(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    tok = CancelToken()
    seen = []

    def on_progress(frac, msg):
        seen.append(frac)
        if frac > 0.2:
            tok.cancel()

    with pytest.raises(Cancelled):
        ensure_model("t-hf", progress=ProgressReporter(on_progress), cancel=tok)
    part = model_dir("t-hf") / "onnx" / "model.onnx.part"
    assert part.exists() and 0 < part.stat().st_size < len(PAYLOAD)
    assert not (model_dir("t-hf") / "onnx" / "model.onnx").exists()
    assert not is_downloaded("t-hf")
    have = part.stat().st_size
    p = ensure_model("t-hf")
    assert p.read_bytes() == PAYLOAD  # verified sha256 although the first part was hashed from disk
    assert _range_requests(server, path)[-1] == f"bytes={have}-"


def test_server_ignoring_range_restarts(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(cut_after=[100_000], ignore_range=True)
    p = ensure_model("t-hf")
    assert p.read_bytes() == PAYLOAD


def test_changed_etag_restarts_from_scratch(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(cut_after=[100_000])
    tok = CancelToken()
    # first attempt is cut; cancel during the retry backoff
    orig = download._sleep

    def cancel_sleep(seconds, cancel):
        tok.cancel()
        orig(seconds, cancel)

    download._sleep = cancel_sleep
    try:
        with pytest.raises(Cancelled):
            ensure_model("t-hf", cancel=tok)
    finally:
        download._sleep = orig
    server.routes[path] = Behaviour(etag='"v2"')  # file changed upstream
    p = ensure_model("t-hf")
    assert p.read_bytes() == PAYLOAD
    last = [h for q, h in server.requests if q == path][-1]
    assert last.get("If-Range") == '"v1"'  # server answered 200 -> full restart


def test_stale_part_for_other_url_is_discarded(fake_registry, server):
    d = model_dir("t-hf") / "onnx"
    d.mkdir(parents=True)
    (d / "model.onnx.part").write_bytes(b"garbage" * 1000)
    (d / "model.onnx.part.json").write_text(json.dumps({"url": "http://elsewhere/x"}), encoding="utf-8")
    p = ensure_model("t-hf")
    assert p.read_bytes() == PAYLOAD
    assert _range_requests(server, "/org/repo/resolve/main/onnx/model.onnx")[-1] is None


def test_range_not_satisfiable_with_complete_part(fake_registry, server):
    spec = dataclasses.replace(fake_registry["t-hf"], sizes={"onnx/model.onnx": len(PAYLOAD)})
    registry.MODELS["t-hf"] = spec
    d = model_dir("t-hf") / "onnx"
    d.mkdir(parents=True)
    url = server.url("/org/repo/resolve/main/onnx/model.onnx")
    (d / "model.onnx.part").write_bytes(PAYLOAD)
    (d / "model.onnx.part.json").write_text(json.dumps({"url": url, "etag": '"v1"'}), encoding="utf-8")
    assert ensure_model("t-hf").read_bytes() == PAYLOAD


def test_checksum_mismatch(fake_registry, server):
    server.routes["/org/repo/resolve/main/onnx/model.onnx"] = Behaviour(data=b"evil" * 1000)
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    assert "checksum" in str(ei.value).lower()
    assert ei.value.hint
    d = model_dir("t-hf") / "onnx"
    assert not (d / "model.onnx").exists()
    assert not (d / "model.onnx.part").exists()


def test_size_mismatch(fake_registry, server):
    registry.MODELS["t-nohash"] = dataclasses.replace(fake_registry["t-nohash"], sizes={"model.onnx": 10})
    with pytest.raises(DownloadError, match="wrong size"):
        ensure_model("t-nohash")


def test_unknown_hash_is_recorded(fake_registry, server, caplog):
    with caplog.at_level("WARNING"):
        p = ensure_model("t-nohash")
    assert p.read_bytes() == PAYLOAD
    rec = recorded_hashes()
    assert rec["t-nohash/model.onnx"]["sha256"] == SHA
    assert rec["t-nohash/model.onnx"]["size"] == len(PAYLOAD)
    assert any("No pinned checksum" in r.message for r in caplog.records)
    delete_model("t-nohash")
    assert "t-nohash/model.onnx" not in recorded_hashes()


def test_multi_file_model_and_variants(fake_registry, server):
    p = ensure_model("t-multi")
    assert p.name == "model.onnx"
    assert (p.parent / "model.onnx_data").read_bytes() == b"x" * 5000
    assert is_downloaded("t-multi")
    assert not is_downloaded("t-multi", variant="fp16")
    p16 = ensure_model("t-multi", variant="fp16")
    assert p16.name == "model_fp16.onnx" and p16.read_bytes() == b"h" * 100
    with pytest.raises(ModelUnavailable):
        ensure_model("t-hf", variant="fp16")
    # truncated file (size known) -> not downloaded -> fetched again
    (p.parent / "model.onnx_data").write_bytes(b"x" * 10)
    assert not is_downloaded("t-multi")
    ensure_model("t-multi")
    assert (p.parent / "model.onnx_data").stat().st_size == 5000


def test_progress_aggregates_across_files(fake_registry, server):
    fracs: list[float] = []
    ensure_model("t-multi", progress=ProgressReporter(lambda f, m: fracs.append(f)))
    assert fracs == sorted(fracs)
    assert fracs[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("code", [401, 403])
def test_auth_errors_are_friendly_and_not_retried(fake_registry, server, code):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(statuses=[code] * 10)
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    assert f"HTTP {code}" in str(ei.value)
    assert "token" in (ei.value.hint or "").lower()
    assert len([1 for p, _ in server.requests if p == path]) == 1


def test_404_not_retried(fake_registry, server):
    server.routes.pop("/dl/m.onnx")
    with pytest.raises(DownloadError, match="not found"):
        ensure_model("t-url")
    assert len(server.requests) == 1


def test_transient_5xx_retried(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(statuses=[503, 500], retry_after="0")
    assert ensure_model("t-hf").read_bytes() == PAYLOAD
    assert len([1 for p, _ in server.requests if p == path]) == 3


def test_persistent_5xx_gives_up(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(statuses=[502] * 20)
    with pytest.raises(DownloadError, match="502"):
        ensure_model("t-hf")
    assert len([1 for p, _ in server.requests if p == path]) == download.MAX_ATTEMPTS


def test_offline_error(fake_registry, monkeypatch):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = s.getsockname()[1]
    monkeypatch.setenv("HF_ENDPOINT", f"http://127.0.0.1:{dead}")
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    assert "offline" in str(ei.value).lower() or "reach" in str(ei.value).lower()
    assert "internet" in (ei.value.hint or "").lower()


def test_disk_space_precheck(fake_registry, monkeypatch):
    import shutil as _sh

    monkeypatch.setattr(download.shutil, "disk_usage", lambda p: _sh._ntuple_diskusage(10, 10, 1000))
    with pytest.raises(DownloadError, match="disk space") as ei:
        ensure_model("t-hf")
    assert ei.value.hint


def test_disk_full_while_writing(fake_registry, server, monkeypatch):
    real_open = open

    class FullFile:
        def __init__(self, f):
            self.f = f

        def write(self, b):
            raise OSError(errno.ENOSPC, "No space left on device")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.f.close()

    def fake_open(path, mode="r", *a, **k):
        f = real_open(path, mode, *a, **k)
        return (
            FullFile(f) if str(path).endswith(".part") and "b" in mode and ("w" in mode or "a" in mode) else f
        )

    monkeypatch.setattr("builtins.open", fake_open)
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    assert "disk space" in str(ei.value).lower()


def test_cancel_before_start(fake_registry, server):
    tok = CancelToken()
    tok.cancel()
    with pytest.raises(Cancelled):
        ensure_model("t-hf", cancel=tok)


def test_env_override(fake_registry, tmp_path, monkeypatch, server):
    f = tmp_path / "local.onnx"
    f.write_bytes(b"onnx")
    monkeypatch.setenv("VRPHOTO_MODEL_T_HF", str(f))
    assert ensure_model("t-hf") == f
    assert is_downloaded("t-hf")
    assert download.local_path("t-hf") == f
    assert server.requests == []
    delete_model("t-hf")  # never deletes the user's file
    assert f.exists()
    monkeypatch.setenv("VRPHOTO_MODEL_T_HF", f'"{f}"')  # quoted as Windows users often do
    assert ensure_model("t-hf") == f
    monkeypatch.setenv("VRPHOTO_MODEL_T_HF", str(tmp_path / "missing.onnx"))
    with pytest.raises(ModelUnavailable, match="does not exist"):
        ensure_model("t-hf")
    assert not is_downloaded("t-hf")


def test_delete_model_and_size(fake_registry, server):
    ensure_model("t-hf")
    assert download.downloaded_size_mb("t-hf") == pytest.approx(len(PAYLOAD) / 1e6, rel=1e-3)
    delete_model("t-hf")
    assert not model_dir("t-hf").exists()
    assert not is_downloaded("t-hf")
    delete_model("t-hf")  # idempotent
    assert download.downloaded_size_mb("t-hf") == 0.0


def test_unknown_model_id():
    with pytest.raises(ModelUnavailable):
        ensure_model("does-not-exist")
    assert is_downloaded("does-not-exist") is False


def test_concurrent_ensure_downloads_once(fake_registry, server):
    results: list[Path] = []
    errors: list[BaseException] = []

    def worker():
        try:
            results.append(ensure_model("t-hf"))
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert not errors
    assert len(set(results)) == 1
    assert len([1 for p, _ in server.requests if p.endswith("onnx/model.onnx")]) == 1


def test_download_url_public_helper(server, tmp_path):
    server.routes["/x.bin"] = Behaviour()
    got: list[tuple[int, int | None]] = []
    out = download_url(
        server.url("/x.bin"),
        tmp_path / "sub" / "x.bin",
        checksum=SHA,
        on_bytes=lambda d, t: got.append((d, t)),
    )
    assert out.read_bytes() == PAYLOAD
    assert got[-1] == (len(PAYLOAD), len(PAYLOAD))
    with pytest.raises(DownloadError, match="Unsupported checksum"):
        download_url(server.url("/x.bin"), tmp_path / "y.bin", checksum="crc32:1234")


def test_invalid_repo_paths_rejected(monkeypatch):
    bad = ModelSpec(
        id="t-bad",
        task="depth",
        title="Bad",
        repo="o/r",
        revision="main",
        files=("../evil.onnx",),
        size_mb=1,
        license="MIT",
    )
    monkeypatch.setattr(registry, "MODELS", {**registry.MODELS, "t-bad": bad})
    with pytest.raises(ModelUnavailable, match="Invalid file path"):
        ensure_model("t-bad")


def test_interprocess_lock_helper(tmp_path):
    lock = tmp_path / ".lock"
    with download._interprocess_lock(lock, None):
        assert lock.exists()
    with download._interprocess_lock(lock, None):  # re-acquirable after release
        pass


def test_hint_mentions_hf_licence_on_403(fake_registry, server):
    server.routes["/org/repo/resolve/main/onnx/model.onnx"] = Behaviour(statuses=[403])
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    assert "licen" in (ei.value.hint or "").lower()
    assert os.fspath(model_dir("t-hf"))  # dir still resolvable


def test_range_not_satisfiable_uses_sidecar_total(fake_registry, server):
    d = model_dir("t-nohash")
    d.mkdir(parents=True)
    url = server.url("/org/nohash/resolve/main/model.onnx")
    (d / "model.onnx.part").write_bytes(PAYLOAD)
    meta = {"url": url, "etag": '"v1"', "size": len(PAYLOAD)}
    (d / "model.onnx.part.json").write_text(json.dumps(meta), encoding="utf-8")
    assert ensure_model("t-nohash").read_bytes() == PAYLOAD
    assert len([1 for p, _ in server.requests if p.endswith("model.onnx")]) == 1


def test_invalid_token_retries_anonymously(fake_registry, server, caplog):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(statuses=[401])  # first (token) request rejected
    with caplog.at_level("WARNING"):
        p = ensure_model("t-hf", token="hf_expired")
    assert p.read_bytes() == PAYLOAD
    hdrs = [h for q, h in server.requests if q == path]
    assert hdrs[0].get("Authorization") == "Bearer hf_expired"
    assert "Authorization" not in hdrs[1]
    assert any("token rejected" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- review additions
@pytest.mark.parametrize(
    "rel",
    [
        "C:evil.onnx",
        "C:/Windows/evil.onnx",
        "a\\..\\..\\evil.onnx",
        "model.onnx:ads",
        "dir/../x.onnx",
        "x.onnx.",
        "x.onnx ",
        "/abs.onnx",
    ],
)
def test_local_rejects_paths_that_could_escape_on_windows(monkeypatch, rel):
    bad = ModelSpec(
        id="t-bad",
        task="depth",
        title="Bad",
        repo="o/r",
        revision="main",
        files=(rel,),
        size_mb=1,
        license="MIT",
    )
    monkeypatch.setattr(registry, "MODELS", {**registry.MODELS, "t-bad": bad})
    with pytest.raises(ModelUnavailable, match="Invalid file path"):
        download.local_path("t-bad")
    assert is_downloaded("t-bad") is False


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_local_rejects_symlinked_folder_escape(monkeypatch, tmp_path):
    spec = ModelSpec(
        id="t-link",
        task="depth",
        title="Link",
        repo="o/r",
        revision="main",
        files=("sub/model.onnx",),
        size_mb=1,
        license="MIT",
    )
    monkeypatch.setattr(registry, "MODELS", {**registry.MODELS, "t-link": spec})
    d = model_dir("t-link")
    d.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (d / "sub").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ModelUnavailable, match="Invalid file path"):
        download.local_path("t-link")


def test_rename_retries_transient_windows_sharing_violation(fake_registry, server, monkeypatch):
    real = os.replace
    fails = {"n": 0}

    def flaky(src, dst):
        if str(src).endswith(".part") and fails["n"] < 2:
            fails["n"] += 1
            raise PermissionError(13, "The process cannot access the file (WinError 32)")
        return real(src, dst)

    monkeypatch.setattr(download.os, "replace", flaky)
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    assert ensure_model("t-hf").read_bytes() == PAYLOAD
    assert fails["n"] == 2


def test_rename_permanently_locked_gives_friendly_error(fake_registry, server, monkeypatch):
    real = os.replace

    def locked(src, dst):
        if str(src).endswith(".part"):
            raise PermissionError(13, "Access is denied")
        return real(src, dst)

    monkeypatch.setattr(download.os, "replace", locked)
    monkeypatch.setattr(download, "REPLACE_RETRIES", 3)
    monkeypatch.setattr(download.time, "sleep", lambda s: None)
    with pytest.raises(DownloadError, match="permission denied") as ei:
        ensure_model("t-hf")
    assert "antivirus" in (ei.value.hint or "").lower()


def test_cancel_takes_effect_while_connection_stalls(fake_registry, server):
    path = "/org/repo/resolve/main/onnx/model.onnx"
    server.routes[path] = Behaviour(stall_after=200_000, stall_s=20.0)
    tok = CancelToken()
    threading.Timer(0.4, tok.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        ensure_model("t-hf", cancel=tok)
    assert time.monotonic() - t0 < 5.0  # not the 30 s socket timeout
    part = model_dir("t-hf") / "onnx" / "model.onnx.part"
    assert part.exists() and part.stat().st_size == 200_000  # kept for resuming
    server.routes[path] = Behaviour()
    assert ensure_model("t-hf").read_bytes() == PAYLOAD


def test_waiting_for_another_threads_download_honours_cancel(fake_registry, server):
    tok = CancelToken()
    with download._model_lock("t-hf"):  # another thread is "downloading" this model
        threading.Timer(0.3, tok.cancel).start()
        t0 = time.monotonic()
        with pytest.raises(Cancelled):
            ensure_model("t-hf", cancel=tok)
        assert time.monotonic() - t0 < 3.0


def test_offline_fails_fast(fake_registry, monkeypatch):
    attempts = []

    class DeadOpener:
        def open(self, req, timeout=None):
            attempts.append(req.full_url)
            raise urllib.error.URLError(socket.gaierror(11001, "getaddrinfo failed"))

    monkeypatch.setattr(download, "_build_opener", lambda: DeadOpener())
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    assert "offline" in str(ei.value).lower()
    assert len(attempts) == download.OFFLINE_ATTEMPTS < download.MAX_ATTEMPTS


def test_is_offline_classification():
    assert download._is_offline(socket.gaierror(-2, "Name or service not known"))
    assert download._is_offline(ConnectionRefusedError())
    assert download._is_offline(OSError(errno.ENETUNREACH, "Network is unreachable"))
    assert not download._is_offline(TimeoutError("timed out"))
    assert not download._is_offline(ConnectionResetError())


def test_401_hint_also_mentions_moved_models(fake_registry, server):
    server.routes["/org/repo/resolve/main/onnx/model.onnx"] = Behaviour(statuses=[401])
    with pytest.raises(DownloadError) as ei:
        ensure_model("t-hf")
    hint = (ei.value.hint or "").lower()
    assert "token" in hint and "moved" in hint


@pytest.mark.skipif(
    os.name == "nt", reason="flock semantics; msvcrt locks are per-handle too but untested here"
)
def test_waiting_for_another_process_is_reported(fake_registry, server):
    import fcntl

    d = model_dir("t-hf")
    d.mkdir(parents=True)
    other = open(d / ".download.lock", "a+b")  # noqa: SIM115 - "another process" holds the lock
    fcntl.flock(other.fileno(), fcntl.LOCK_EX)
    events: list[str] = []
    threading.Timer(0.5, lambda: (fcntl.flock(other.fileno(), fcntl.LOCK_UN), other.close())).start()
    p = ensure_model("t-hf", progress=ProgressReporter(lambda f, m: events.append(m)))
    assert p.read_bytes() == PAYLOAD
    assert any("Waiting for another download" in m for m in events)


def test_captive_portal_html_is_rejected(fake_registry, server):
    path = "/org/nohash/resolve/main/model.onnx"  # no pinned hash: would otherwise be accepted
    server.routes[path] = Behaviour(
        data=b"<html>Please log in to the hotel Wi-Fi</html>", content_type="text/html; charset=utf-8"
    )
    with pytest.raises(DownloadError, match="web page") as ei:
        ensure_model("t-nohash")
    assert "browser" in (ei.value.hint or "")
    assert not is_downloaded("t-nohash")
    assert recorded_hashes() == {}
    server.routes[path] = Behaviour(content_type="application/octet-stream")
    assert ensure_model("t-nohash").read_bytes() == PAYLOAD
