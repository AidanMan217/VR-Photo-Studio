"""Tests for the viewer tooling: scripts/vendor_web.py, the fixture library builder and the
stand-in HTTP server used by the browser tests."""

from __future__ import annotations

import http.client
import importlib.util
import io
import json
import os
import sys
import tarfile
import threading
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VIEWER = REPO / "src" / "vrphoto" / "webxr" / "viewer"
VENDOR = VIEWER / "vendor"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def vw():
    return _load("vendor_web_under_test", REPO / "scripts" / "vendor_web.py")


@pytest.fixture(scope="module")
def fixtures_mod():
    return _load("make_fixtures_under_test", HERE / "make_fixtures.py")


@pytest.fixture(scope="module")
def server_mod():
    return _load("serve_fixtures_under_test", HERE / "serve_fixtures.py")


# --------------------------------------------------------------------------- vendor_web.py
def test_relative_and_bare_imports(vw):
    src = (
        "import {\n  A,\n  B\n} from 'three';\n"
        "import { C } from '../loaders/GLTFLoader.js';\n"
        "export { D } from './XRHandMeshModel.js';\n"
        "import './side-effect.js';\n"
        "const m = import('./lazy.js');\n"
    )
    assert vw.relative_imports(src) == [
        "../loaders/GLTFLoader.js",
        "./XRHandMeshModel.js",
        "./lazy.js",
        "./side-effect.js",
    ]
    assert vw.bare_imports(src) == ["three"]


def test_close_addons_follows_relative_imports(vw, tmp_path):
    jsm = tmp_path / "jsm"
    (jsm / "webxr").mkdir(parents=True)
    (jsm / "loaders").mkdir()
    (jsm / "utils").mkdir()
    (jsm / "webxr" / "F.js").write_text("import { L } from '../loaders/L.js';", encoding="utf-8")
    (jsm / "loaders" / "L.js").write_text(
        "import { u } from '../utils/U.js';\nimport * as T from 'three';", encoding="utf-8"
    )
    (jsm / "utils" / "U.js").write_text("export const u = 1;", encoding="utf-8")
    assert vw.close_addons(jsm, ["webxr/F.js"]) == ["loaders/L.js", "utils/U.js", "webxr/F.js"]
    with pytest.raises(vw.VendorError, match="not found"):
        vw.close_addons(jsm, ["webxr/Missing.js"])
    (jsm / "utils" / "U.js").write_text("import x from '../../outside.js';", encoding="utf-8")
    with pytest.raises(vw.VendorError, match="outside"):
        vw.close_addons(jsm, ["webxr/F.js"])


def test_filtered_profiles_list(vw, tmp_path):
    full = {pid: {"path": f"{pid}/profile.json"} for pid in [*vw.INPUT_PROFILES, "htc-vive", "pico-4"]}
    src = tmp_path / "profilesList.json"
    src.write_text(json.dumps(full), encoding="utf-8")
    out = json.loads(vw.filtered_profiles_list(src))
    assert set(out) == set(vw.INPUT_PROFILES) | set(vw.PROFILE_ALIASES)
    assert out["generic-trigger"]["path"] == "generic-trigger-squeeze-thumbstick/profile.json"
    full.pop("generic-hand")
    src.write_text(json.dumps(full), encoding="utf-8")
    with pytest.raises(vw.VendorError, match="generic-hand"):
        vw.filtered_profiles_list(src)


def test_apply_patch_requires_exactly_one_match(vw):
    p = vw.Patch("x.js", "const A = 1;", "const A = 2;", "test")
    assert vw.apply_patch(b"const A = 1;\n", p) == b"const A = 2;\n"
    with pytest.raises(vw.VendorError, match="found 0"):
        vw.apply_patch(b"const B = 1;", p)
    with pytest.raises(vw.VendorError, match="found 2"):
        vw.apply_patch(b"const A = 1;const A = 1;", p)


def test_patches_match_vendored_files(vw):
    lock = vw.load_lock(VENDOR)
    patched = {p["path"]: p for p in lock["patches"]}
    assert set(patched) == {p.path for p in vw.PATCHES}
    for patch in vw.PATCHES:
        text = (VENDOR / patch.path).read_text(encoding="utf-8")
        assert patch.new in text and patch.old not in text


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return path


def test_safe_extract(vw, tmp_path):
    good = _tar(tmp_path / "good.tgz", {"package/package.json": b"{}", "package/dist/a.js": b"x"})
    root = vw.safe_extract(good, tmp_path / "x1")
    assert (root / "dist" / "a.js").read_bytes() == b"x"
    evil = _tar(tmp_path / "evil.tgz", {"package/package.json": b"{}", "../escape.txt": b"x"})
    with pytest.raises(vw.VendorError, match="escapes"):
        vw.safe_extract(evil, tmp_path / "x2")
    assert not (tmp_path / "escape.txt").exists()
    nopkg = _tar(tmp_path / "nopkg.tgz", {"other/file.txt": b"x"})
    with pytest.raises(vw.VendorError, match="package/"):
        vw.safe_extract(nopkg, tmp_path / "x3")


def test_verify_detects_drift(vw, tmp_path):
    d = tmp_path / "vendor"
    (d / "a").mkdir(parents=True)
    (d / "a" / "f.js").write_bytes(b"one")
    lock = {
        "packages": {
            "three": {"version": vw.THREE_VERSION, "files": {"a/f.js": vw.sha256_bytes(b"one")}},
            "spark": {"version": vw.SPARK_VERSION, "files": {}},
            "webxr-input-profiles": {"version": vw.PROFILES_VERSION, "files": {"a/g.js": "0" * 64}},
        },
        "generated": [],
    }
    (d / "VENDOR.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    problems = vw.verify(d, vw.load_lock(d))
    assert problems == [
        *(f"lock does not record the generated file {g}" for g in vw.GENERATED_FILES),
        "missing: a/g.js",
    ]
    for g in vw.GENERATED_FILES:
        (d / g).parent.mkdir(parents=True, exist_ok=True)
        (d / g).write_bytes(b"gen")
        lock["generated"].append({"path": g, "sha256": vw.sha256_bytes(b"gen")})
    (d / "VENDOR.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    assert vw.verify(d, vw.load_lock(d)) == ["missing: a/g.js"]
    (d / "a" / "f.js").write_bytes(b"two")
    (d / "extra.js").write_bytes(b"x")
    problems = vw.verify(d, vw.load_lock(d))
    assert "modified: a/f.js" in problems and "unexpected file: extra.js" in problems
    lock["packages"]["three"]["version"] = "0.0.1"
    assert any("script wants" in p for p in vw.verify(d, lock))
    assert vw.verify(tmp_path / "nothing", None) == ["VENDOR.lock.json is missing or unreadable"]


def test_verify_accepts_crlf_checkout(vw, tmp_path):
    """A Windows checkout with core.autocrlf=true must still pass --check (same content)."""
    import shutil

    copy = tmp_path / "vendor"
    shutil.copytree(VENDOR, copy)
    js = copy / "three" / "examples" / "jsm" / "controls" / "OrbitControls.js"
    raw = js.read_bytes()
    assert b"\r\n" not in raw
    js.write_bytes(raw.replace(b"\n", b"\r\n"))
    assert vw.verify(copy, vw.load_lock(copy)) == []
    js.write_bytes(raw.replace(b"\n", b"\r\n") + b"// extra\r\n")  # a real change is still caught
    assert vw.verify(copy, vw.load_lock(copy)) == ["modified: three/examples/jsm/controls/OrbitControls.js"]


def test_vendor_gitattributes_disables_eol_conversion(vw):
    text = (VENDOR / ".gitattributes").read_text(encoding="utf-8")
    assert text == vw.GITATTRIBUTES
    assert "* -text" in text.splitlines()
    lock = vw.load_lock(VENDOR)
    assert {g["path"] for g in lock["generated"]} == set(vw.GENERATED_FILES)


def test_vendor_is_noop_when_up_to_date(vw, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("npm must not be called when the vendor folder matches the lock")

    monkeypatch.setattr(vw, "find_npm", no_network)
    monkeypatch.setattr(vw, "npm_pack", no_network)
    assert vw.vendor(VENDOR) is False
    assert vw.main(["--check"]) == 0


def test_vendor_check_fails_on_modified_copy(vw, tmp_path):
    import shutil

    copy = tmp_path / "vendor"
    shutil.copytree(VENDOR, copy)
    (copy / "three" / "LICENSE").write_text("changed", encoding="utf-8")
    assert vw.main(["--check", "--vendor-dir", str(copy)]) == 1


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def library(tmp_path_factory, fixtures_mod):
    lib = tmp_path_factory.mktemp("fixture_lib")
    ids = fixtures_mod.build_library(lib)
    return lib, ids


def test_fixture_library_has_every_mode(library):
    from vrphoto.webxr.scene import list_scenes, read_manifest

    lib, ids = library
    items = list_scenes(lib)
    assert [i["id"] for i in items] == ids  # created stamps make the order deterministic
    modes = {i["mode"] for i in items}
    assert modes == {"photo3d", "stereo", "sphere", "object", "splat"}
    for item in items:
        m = read_manifest(lib / item["id"])
        assert item["thumbnail"] == f"/scenes/{item['id']}/thumb.jpg"
        assert m["schema"] == "vrphoto.scene/1"


def test_fixture_glbs_are_valid(library):
    pygltflib = pytest.importorskip("pygltflib")
    lib, _ = library
    for glb in sorted(lib.glob("*/*.glb")):
        g = pygltflib.GLTF2().load(str(glb))
        assert g.meshes and g.images and g.materials
        assert all("KHR_materials_unlit" in (mat.extensions or {}) for mat in g.materials)
        blob = g.binary_blob()
        for view in g.bufferViews:
            assert view.byteOffset % 4 == 0 and view.byteOffset + view.byteLength <= len(blob)


def test_fixture_splats_are_32_byte_records(library, fixtures_mod):
    import numpy as np

    lib, _ = library
    files = sorted(lib.glob("*/*.splat"))
    assert {f.parent.name for f in files} >= {"fx-object-splat", "fx-splat"}
    for f in files:
        data = f.read_bytes()
        assert len(data) % 32 == 0 and len(data) >= 32 * 1000
        rec = np.frombuffer(data, dtype=[("p", "<f4", 3), ("s", "<f4", 3), ("c", "u1", 4), ("q", "u1", 4)])
        assert np.isfinite(rec["p"]).all() and (rec["s"] > 0).all()
        if not f.parent.name.startswith("fx-real-"):  # inline writer: identity quaternion, w first
            assert (rec["q"][:, 0] == 255).all()


def test_fixture_real_scenes_use_app_geometry(library, fixtures_mod):
    """fx-real-* scenes come from the app's own geometry writers (integration with the viewer)."""
    pytest.importorskip("vrphoto.geometry.gltf")
    pytest.importorskip("vrphoto.geometry.splat")
    from vrphoto.geometry.gltf import read_glb_meshes
    from vrphoto.geometry.splat import read_splats

    lib, ids = library
    assert set(fixtures_mod.REAL_SCENE_IDS) <= set(ids)
    meshes = read_glb_meshes(lib / "fx-real-photo3d" / "scene.glb")
    assert len(meshes) >= 2
    # glTF frame: the photo is in front of the eye (0, 1.5, 0), i.e. at negative z.
    assert all(float(m.vertices[:, 2].max()) < 0 for m in meshes)
    obj = read_glb_meshes(lib / "fx-real-object" / "object.glb")
    lo = min(float(m.vertices[:, 1].min()) for m in obj)
    hi = max(float(m.vertices[:, 1].max()) for m in obj)
    assert lo == pytest.approx(0.0, abs=1e-3) and hi == pytest.approx(1.7, abs=1e-2)  # feet on the floor
    assert len(read_splats(lib / "fx-real-splat-ply" / "world.ply")) > 1000


def test_fixture_library_without_real_scenes(tmp_path, fixtures_mod):
    ids = fixtures_mod.build_library(tmp_path / "lib", real=False)
    assert ids == [i for i in fixtures_mod.SCENE_IDS if i in ids]
    assert not any(i.startswith("fx-real-") for i in ids)


# --------------------------------------------------------------------------- test server
@pytest.fixture(scope="module")
def server(library, server_mod):
    from http.server import ThreadingHTTPServer

    lib, _ = library
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_mod.make_handler(lib, VIEWER))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _get(port: int, path: str, headers: dict | None = None, method: str = "GET"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(method, path, headers=headers or {})
    res = conn.getresponse()
    body = res.read()
    conn.close()
    return res, body


def test_server_routes(server):
    res, body = _get(server, "/")
    assert (
        res.status == 200
        and b"VR Photo Studio" in body
        and res.getheader("Content-Type").startswith("text/html")
    )
    res, body = _get(server, "/api/scenes")
    scenes = json.loads(body)
    assert res.status == 200 and len(scenes) >= 7
    res, body = _get(server, "/scenes/fx-stereo/scene.json")
    assert res.status == 200 and json.loads(body)["mode"] == "stereo"
    res, _ = _get(server, "/vendor/three/build/three.module.js")
    assert res.status == 200 and res.getheader("Content-Type").startswith("text/javascript")
    res, _ = _get(server, "/view.html?scene=fx-stereo")
    assert res.status == 200
    res, body = _get(server, "/js/app.js", method="HEAD")
    assert res.status == 200 and body == b"" and int(res.getheader("Content-Length")) > 1000


def test_server_range_and_safety(server, library):
    lib, _ = library
    full = (lib / "fx-splat" / "room.splat").read_bytes()
    res, body = _get(server, "/scenes/fx-splat/room.splat", {"Range": "bytes=32-95"})
    assert res.status == 206 and body == full[32:96]
    assert res.getheader("Content-Range") == f"bytes 32-95/{len(full)}"
    res, body = _get(server, "/scenes/fx-splat/room.splat", {"Range": "bytes=-32"})
    assert res.status == 206 and body == full[-32:]
    res, _ = _get(server, "/scenes/fx-splat/room.splat", {"Range": f"bytes={len(full) + 10}-"})
    assert res.status == 416
    for bad in (
        "/scenes/../../etc/passwd",
        "/scenes/%2e%2e/%2e%2e/pyproject.toml",
        "/../pyproject.toml",
        "/nope.js",
    ):
        res, _ = _get(server, bad)
        assert res.status == 404, bad


def test_viewer_works_with_app_server(library):
    """Integration: the real ViewerServer serves this viewer and the scene list."""
    pytest.importorskip("vrphoto.webxr.server")
    import socket
    import subprocess

    lib, ids = library
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            str(HERE / "serve_fixtures.py"),
            "--port",
            str(port),
            "--library",
            str(lib),
            "--app-server",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},  # temp paths may be non-ASCII on Windows
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("READY"), line + (proc.stdout.read() if proc.poll() is not None else "")
        res, body = _get(port, "/api/scenes")
        assert res.status == 200 and [s["id"] for s in json.loads(body)] == ids
        res, body = _get(port, "/view.html?scene=fx-stereo")
        assert res.status == 200 and b'type="importmap"' in body
        res, _ = _get(port, "/vendor/spark/dist/spark.module.js", method="HEAD")
        assert res.status == 200 and res.getheader("Content-Type").startswith("text/javascript")
        res, body = _get(port, "/scenes/fx-splat/room.splat", {"Range": "bytes=0-31"})
        assert res.status == 206 and len(body) == 32
    finally:
        proc.terminate()
        proc.wait(timeout=10)
