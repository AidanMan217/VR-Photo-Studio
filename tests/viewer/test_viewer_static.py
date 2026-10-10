"""Static checks of the WebXR viewer web app (no browser needed; Node.js for the JS parts).

* every module specifier in every page / module resolves to a local (vendored) file,
* ``node --check`` (ES-module goal) passes for every .js file,
* no CDN URL is referenced by anything the pages load at runtime,
* vendored files match ``vendor/VENDOR.lock.json`` and licences are shipped,
* DOM ids used by the scripts exist in the pages, all modes are registered,
* the viewer's JS manifest validator agrees with the Python one on what is loadable.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
VIEWER = REPO / "src" / "vrphoto" / "webxr" / "viewer"
VENDOR = VIEWER / "vendor"
NODE = shutil.which("node")

needs_node = pytest.mark.skipif(NODE is None, reason="Node.js is not installed")

CDN_HOSTS = (
    "cdn.jsdelivr.net",
    "unpkg.com",
    "cdnjs.cloudflare.com",
    "esm.sh",
    "cdn.skypack.dev",
    "ga.jspm.io",
    "sparkjs.dev",
    "storage.googleapis.com",
    "fonts.googleapis.com",
    "raw.githubusercontent.com",
    "huggingface.co",
)


def _run_node(*args: str, stdin: str | None = None, timeout: float = 120) -> subprocess.CompletedProcess:
    # Node speaks UTF-8 whatever the Windows code page is.
    return subprocess.run(
        [NODE, *args],
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=HERE,
        check=False,
    )


def _check_imports_json(viewer: Path = VIEWER) -> tuple[int, dict, str]:
    proc = _run_node(str(HERE / "check_imports.mjs"), str(viewer), "--json")
    try:
        data = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except ValueError:
        data = {}
    return proc.returncode, data, proc.stderr


# --------------------------------------------------------------------------- module graph
@needs_node
def test_all_imports_resolve_to_local_files():
    rc, data, err = _check_imports_json()
    assert rc == 0, f"check_imports.mjs found problems:\n{err}"
    reachable = set(data["reachable"])
    # The pages really pull in what we expect (catches a silently empty graph).
    for must in (
        "js/app.js",
        "js/gallery.js",
        "js/modes/photo3d.js",
        "js/modes/stereo.js",
        "js/modes/sphere.js",
        "js/modes/object.js",
        "js/modes/splat.js",
        "vendor/three/build/three.module.js",
        "vendor/three/build/three.core.js",
        "vendor/spark/dist/spark.module.js",
        "vendor/three/examples/jsm/postprocessing/Pass.js",
        "vendor/three/examples/jsm/webxr/XRControllerModelFactory.js",
        "vendor/three/examples/jsm/webxr/XRHandModelFactory.js",
        "vendor/three/examples/jsm/loaders/GLTFLoader.js",
        "vendor/three/examples/jsm/controls/OrbitControls.js",
    ):
        assert must in reachable, f"{must} is not reachable from the pages"


@needs_node
def test_import_checker_detects_problems(tmp_path):
    """The checker must fail on a missing module, a CDN import and a CDN fetch URL."""
    v = tmp_path / "viewer"
    (v / "js").mkdir(parents=True)
    (v / "index.html").write_text(
        '<script type="importmap">{"imports":{"three":"./vendor/three.module.js"}}</script>'
        '<script type="module" src="js/main.js"></script><link rel="stylesheet" href="css/missing.css">',
        encoding="utf-8",
    )
    (v / "js" / "main.js").write_text(
        "import * as THREE from 'three';\n"
        "import { x } from './missing.js';\n"
        "import { y } from 'https://cdn.jsdelivr.net/npm/foo/foo.js';\n"
        "// import { z } from './only-in-a-comment.js';\n"
        "const s = \"import('./not-real-in-string.js')\";\n"
        "export const go = () => import('./modes/' + s);\n"
        "fetch('https://unpkg.com/thing/data.json');\n",
        encoding="utf-8",
    )
    rc, data, err = _check_imports_json(v)
    assert rc == 1
    text = err
    assert "three.module.js does not exist" in text
    assert "missing.js" in text
    assert "remote URL" in text and "cdn.jsdelivr.net" in text
    assert "unpkg.com" in text
    assert "css/missing.css" in text
    assert "computed specifier" in text
    assert "only-in-a-comment" not in text  # real parser: comments are ignored


@needs_node
def test_node_check_every_js_file():
    """`node --check` in ES-module mode (plain `node --check x.js` silently skips ESM syntax)."""
    files = sorted(VIEWER.rglob("*.js"))
    assert len(files) >= 20

    def check(path: Path) -> tuple[Path, int, str]:
        proc = subprocess.run(
            [NODE, "--input-type=module", "--check"],
            input=path.read_bytes(),
            capture_output=True,
            timeout=120,
            check=False,
        )
        return path, proc.returncode, proc.stderr.decode("utf-8", "replace")

    with ThreadPoolExecutor(max_workers=4) as pool:
        failures = [(p, e) for p, rc, e in pool.map(check, files) if rc != 0]
    assert not failures, "\n".join(f"{p.relative_to(VIEWER)}:\n{e}" for p, e in failures)


@needs_node
def test_node_check_catches_syntax_errors(tmp_path):
    bad = tmp_path / "bad.js"
    bad.write_text("export const a = ;\n", encoding="utf-8")
    proc = subprocess.run(
        [NODE, "--input-type=module", "--check"], input=bad.read_bytes(), capture_output=True, check=False
    )
    assert proc.returncode != 0


# --------------------------------------------------------------------------- no CDN at runtime
def _url_hosts(text: str) -> set[str]:
    return {m.group(1).lower() for m in re.finditer(r"https?://([^/\s\"'`)<>\\]+)", text)}


def test_own_viewer_code_has_no_remote_urls():
    own = [p for p in VIEWER.rglob("*") if p.is_file() and "vendor" not in p.relative_to(VIEWER).parts]
    own = [p for p in own if p.suffix in (".js", ".html", ".css")]
    assert own
    allowed = {"www.w3.org", "localhost", "localhost:8080", "127.0.0.1"}
    for p in own:
        hosts = _url_hosts(p.read_text(encoding="utf-8")) - allowed
        assert not hosts, f"{p.relative_to(VIEWER)} references remote hosts {sorted(hosts)}"


@needs_node
def test_runtime_vendored_files_reference_no_cdn():
    rc, data, err = _check_imports_json()
    assert rc == 0, err
    for rel in data["reachable"]:
        hosts = _url_hosts((VIEWER / rel).read_text(encoding="utf-8"))
        bad = sorted(h for h in hosts if any(h == c or h.endswith("." + c) for c in CDN_HOSTS))
        assert not bad, f"{rel} references CDN hosts {bad}"


def test_controller_and_hand_factories_point_to_local_profiles():
    jsm = VENDOR / "three" / "examples" / "jsm" / "webxr"
    ctrl = (jsm / "XRControllerModelFactory.js").read_text(encoding="utf-8")
    hand = (jsm / "XRHandMeshModel.js").read_text(encoding="utf-8")
    assert "cdn.jsdelivr.net" not in ctrl and "cdn.jsdelivr.net" not in hand
    assert "webxr-input-profiles/profiles" in ctrl and "generic-hand" in hand
    xr = (VIEWER / "js" / "xr.js").read_text(encoding="utf-8")
    assert ".setPath(PROFILES_PATH)" in xr and ".setPath(HAND_PROFILE_PATH)" in xr
    assert "'../vendor/webxr-input-profiles/profiles'" in xr


# --------------------------------------------------------------------------- vendored files
def _vendor_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("vendor_web", REPO / "scripts" / "vendor_web.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vendor_web"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_vendored_files_match_lock():
    vw = _vendor_module()
    lock = vw.load_lock(VENDOR)
    assert lock is not None, "run: python scripts/vendor_web.py"
    assert vw.verify(VENDOR, lock) == []
    pk = lock["packages"]
    assert pk["three"]["version"] == "0.186.1"
    assert pk["spark"]["version"] == "2.3.1"
    assert pk["webxr-input-profiles"]["version"] == "1.0.20"
    for entry in pk.values():
        assert entry["integrity"].startswith("sha512-")
        for lic in entry["license_files"]:
            assert (VENDOR / lic).is_file()
            assert "Permission is hereby granted" in (VENDOR / lic).read_text(encoding="utf-8")


def test_vendored_three_build_pair_and_addons_present():
    build = VENDOR / "three" / "build"
    module = (build / "three.module.js").read_text(encoding="utf-8")
    assert "./three.core.js" in module
    assert (build / "three.core.js").is_file()
    jsm = VENDOR / "three" / "examples" / "jsm"
    for rel in (
        "webxr/VRButton.js",
        "webxr/ARButton.js",
        "webxr/XRButton.js",
        "webxr/XRControllerModelFactory.js",
        "webxr/XRHandModelFactory.js",
        "webxr/XRHandMeshModel.js",
        "webxr/XRHandPrimitiveModel.js",
        "libs/motion-controllers.module.js",
        "loaders/GLTFLoader.js",
        "utils/BufferGeometryUtils.js",
        "utils/SkeletonUtils.js",
        "controls/OrbitControls.js",
        "postprocessing/Pass.js",
    ):
        assert (jsm / rel).is_file(), rel


def test_input_profiles_list_is_consistent():
    base = VENDOR / "webxr-input-profiles" / "profiles"
    plist = json.loads((base / "profilesList.json").read_text(encoding="utf-8"))
    for pid in (
        "meta-quest-touch-plus",
        "oculus-touch-v3",
        "oculus-touch-v2",
        "generic-hand",
        "generic-trigger-squeeze-thumbstick",
    ):
        assert pid in plist
    # three's XRControllerModelFactory default profile must resolve too.
    assert "generic-trigger" in plist
    for pid, entry in plist.items():
        prof_path = base / entry["path"]
        assert prof_path.is_file(), f"{pid}: {entry['path']} missing"
        prof = json.loads(prof_path.read_text(encoding="utf-8"))
        for hand, layout in prof.get("layouts", {}).items():
            if "assetPath" in layout:
                assert (prof_path.parent / layout["assetPath"]).is_file(), (
                    f"{pid}/{hand}: {layout['assetPath']}"
                )
    for side in ("left", "right"):
        assert (base / "generic-hand" / f"{side}.glb").is_file()


def test_spark_bundle_is_self_contained():
    src = (VENDOR / "spark" / "dist" / "spark.module.js").read_text(encoding="utf-8")
    imports = set(re.findall(r'^import .*? from "([^"]+)";', src[:5000], re.MULTILINE))
    assert imports <= {"three", "three/addons/postprocessing/Pass.js"}, imports
    # WASM and the worker are inlined (no separate files to serve).
    assert "WebAssembly.compile" in src and "createObjectURL" in src


# --------------------------------------------------------------------------- pages and modes
def _ids_in_html(path: Path) -> set[str]:
    return set(re.findall(r'\bid="([^"]+)"', path.read_text(encoding="utf-8")))


def test_dom_ids_used_by_scripts_exist():
    view_ids = _ids_in_html(VIEWER / "view.html")
    gallery_ids = _ids_in_html(VIEWER / "index.html")
    pat = re.compile(r"""\$\(\s*'([\w-]+)'\s*\)|getElementById\(\s*'([\w-]+)'\s*\)""")

    def used(js: Path) -> set[str]:
        return {a or b for a, b in pat.findall(js.read_text(encoding="utf-8"))}

    for js in ("ui.js", "app.js"):
        missing = used(VIEWER / "js" / js) - view_ids
        assert not missing, f"js/{js} uses ids missing from view.html: {sorted(missing)}"
    missing = used(VIEWER / "js" / "gallery.js") - gallery_ids
    assert not missing, f"js/gallery.js uses ids missing from index.html: {sorted(missing)}"


def test_view_html_import_map_and_secure_context_message():
    html = (VIEWER / "view.html").read_text(encoding="utf-8")
    m = re.search(r'<script type="importmap">(.*?)</script>', html, re.S)
    assert m
    imports = json.loads(m.group(1))["imports"]
    assert set(imports) == {"three", "three/addons/", "@sparkjsdev/spark"}
    assert all(v.startswith("./vendor/") for v in imports.values())
    ui = (VIEWER / "js" / "ui.js").read_text(encoding="utf-8")
    assert "isSecureContext" in (VIEWER / "js" / "xr.js").read_text(encoding="utf-8")
    assert "https://" in ui and "USB mode" in ui and "certificate" in ui
    assert "Enter AR (passthrough)" in html and "Enter VR" in html


def test_all_modes_registered_and_export_load():
    schema = json.loads((VIEWER / "scene.schema.json").read_text(encoding="utf-8"))
    modes = schema["properties"]["mode"]["enum"]
    app = (VIEWER / "js" / "app.js").read_text(encoding="utf-8")
    for mode in modes:
        assert f"{mode}: () => import('./modes/{mode}.js')" in app
        src = (VIEWER / "js" / "modes" / f"{mode}.js").read_text(encoding="utf-8")
        assert re.search(r"export async function load\(ctx, scene\)", src), mode


def test_xr_session_features():
    xr = (VIEWER / "js" / "xr.js").read_text(encoding="utf-8")
    assert "requiredFeatures: ['local-floor']" in xr
    for feat in ("hand-tracking", "hit-test", "anchors", "plane-detection", "layers", "bounded-floor"):
        assert f"'{feat}'" in xr
    assert "getOffsetReferenceSpace" in xr and "XRRigidTransform" in xr
    assert "setFoveation(1)" in xr


# --------------------------------------------------------------------------- JS vs Python validation
GOOD = [
    {
        "schema": "vrphoto.scene/1",
        "id": "a1",
        "title": "A",
        "mode": "photo3d",
        "photo3d": {"meshes": ["bg.glb", "fg.glb"]},
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "s",
        "title": "S",
        "mode": "stereo",
        "stereo": {"media": "x_LR.mp4", "kind": "video", "layout": "sbs", "aspect": 1.78},
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "p",
        "title": "P",
        "mode": "sphere",
        "sphere": {"media": "x.jpg", "kind": "image", "coverage": 360, "layout": "tb", "depth": None},
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "o",
        "title": "O",
        "mode": "object",
        "object": {"asset": "p.splat", "format": "splat", "height_m": 1.7, "pivot": "center"},
        "env": {"prefer": "ar", "background": "#102030"},
        "notices": ["NC"],
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "w",
        "title": "W",
        "mode": "splat",
        "splat": {"asset": "w.spz", "format": "spz", "up": "z"},
    },
]
BAD_FOR_VIEWER = [
    {"schema": "vrphoto.scene/1", "id": "b", "title": "B", "mode": "hologram"},
    {"schema": "vrphoto.scene/1", "id": "b", "title": "B", "mode": "stereo"},
    {"schema": "vrphoto.scene/1", "id": "b", "title": "B", "mode": "photo3d", "photo3d": {"meshes": []}},
    {
        "schema": "vrphoto.scene/1",
        "id": "b",
        "title": "B",
        "mode": "photo3d",
        "photo3d": {"meshes": ["../x.glb"]},
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "b",
        "title": "B",
        "mode": "sphere",
        "sphere": {"media": "x.jpg", "kind": "image", "coverage": 270, "layout": "mono"},
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "b",
        "title": "B",
        "mode": "object",
        "object": {"asset": "C:\\x.glb", "format": "glb"},
    },
    {
        "schema": "vrphoto.scene/9",
        "id": "b",
        "title": "B",
        "mode": "splat",
        "splat": {"asset": "w.ply", "format": "ply"},
    },
    {
        "schema": "vrphoto.scene/1",
        "id": "b",
        "title": "B",
        "mode": "stereo",
        "stereo": {"media": "x.jpg", "kind": "image", "layout": "sbs", "distance_m": -1},
    },
]


@needs_node
def test_js_validator_agrees_with_python():
    from vrphoto.core.errors import InputError
    from vrphoto.webxr.scene import MODES, SCHEMA_ID, validate_manifest

    proc = _run_node(
        str(HERE / "js_manifest_check.mjs"), str(VIEWER), stdin=json.dumps(GOOD + BAD_FOR_VIEWER)
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["modes"] == list(MODES)
    assert out["schema"] == SCHEMA_ID
    assert out["relative"] == "2 hours ago"
    results = out["results"]
    for case, res in zip(GOOD, results[: len(GOOD)]):
        validate_manifest(case)  # Python accepts ...
        assert res["ok"], (case, res["errors"])  # ... so the viewer must load it
    for case, res in zip(BAD_FOR_VIEWER, results[len(GOOD) :]):
        with pytest.raises(InputError):
            validate_manifest(case)
        assert not res["ok"], case
        assert res["errors"] and all(isinstance(e, str) and e for e in res["errors"])
    # Defaults are filled in for the modes.
    assert results[0]["manifest"]["photo3d"]["eye_height_m"] == 1.5
    assert results[2]["manifest"]["sphere"]["radius_m"] == 20
    assert results[3]["manifest"]["object"]["placement"] == "hit-test"
