"""Installation self-test: ``vrphoto-cli --selftest [--json]``.

:func:`run_selftest` checks, in a few seconds and without network access (unless
``models=True``), that every part of the app works on this machine and in this build
(including the PyInstaller-frozen exe):

* ``imports`` - every vrphoto module and the third-party libraries it bundles;
* ``onnxruntime`` - the available execution providers, and a tiny embedded ONNX model
  run on each of them (CPU, CUDA, DirectML...);
* ``ffmpeg`` - encode 10 frames with the preferred encoder and decode them again;
* ``spatial_metadata`` - inject + read VR180 metadata in an MP4;
* ``https_server`` - certificate generation, viewer server start on ephemeral localhost
  ports, fetching ``index.html`` over HTTPS and HTTP, stop;
* ``viewer_assets`` - the bundled WebXR viewer files (three.js, Spark);
* ``addons`` - add-on manifests load and their worker scripts are bundled;
* ``uv`` - the uv package manager used to install add-ons;
* ``geometry`` - heuristic depth -> layers (with inpainting) -> stereo pair -> GLB;
* ``data_folder`` - the app data folder is writable;
* ``models`` (only with ``models=True``) - download and run Depth Anything V2 Small.

The result is a JSON-able dict ``{"ok": bool, "checks": {name: {"ok", "seconds",
"detail"/"error", ...}}, ...}``.  Informational sub-results (e.g. a GPU provider that is
not present) never fail the test.
"""

from __future__ import annotations

import base64
import json
import platform
import ssl
import sys
import tempfile
import time
import traceback
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .core.log import get_logger

log = get_logger(__name__)

#: A 669-byte ONNX model (opset 13): y = relu(2 * (conv3x3(x, w=1/27, pad=1) + b)),
#: x: float32[1,3,8,8] -> y: float32[1,4,8,8], b = [0, 0.5, 1, -1].  Embedded so the check
#: also works in the frozen app, where the ``onnx`` package is not bundled.
TINY_ONNX_B64 = (
    "CAgSEHZycGhvdG8tc2VsZnRlc3Q6gAUKJQoBeAoBdwoBYhIBYyIEQ29udioRCgRwYWRzQAFAAUABQAGgAQcKEAoBYwoDdHdv"
    "EgFtIgNNdWwKDAoBbRIBeSIEUmVsdRIQdnJwaG90b19zZWxmdGVzdCrAAwgECAMIAwgDEAFCAXdKsAMmtBc9JrQXPSa0Fz0m"
    "tBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0m"
    "tBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0m"
    "tBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0m"
    "tBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0m"
    "tBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0m"
    "tBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0mtBc9JrQXPSa0Fz0qGQgEEAFCAWJKEAAA"
    "AAAAAAA/AACAPwAAgL8qDRABQgN0d29KBAAAAEBaGwoBeBIWChQIARIQCgIIAQoCCAMKAggICgIICGIbCgF5EhYKFAgBEhAK"
    "AggBCgIIBAoCCAgKAggIQgQKABAN"
)
_TINY_BIAS = np.array([0.0, 0.5, 1.0, -1.0], np.float32)

#: Third-party modules the app needs at run time.
THIRD_PARTY = (
    "numpy",
    "cv2",
    "PIL.Image",
    "pillow_heif",
    "onnxruntime",
    "imageio_ffmpeg",
    "cryptography.x509",
    "qrcode",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
)
#: Modules that must import even if package walking finds nothing (e.g. a broken frozen bundle).
KEY_MODULES = (
    "vrphoto.cli",
    "vrphoto.pipeline.run",
    "vrphoto.pipeline.queue",
    "vrphoto.depth.da2",
    "vrphoto.depth.pano",
    "vrphoto.inpaint.lama",
    "vrphoto.segment.birefnet",
    "vrphoto.geometry.warp",
    "vrphoto.geometry.gltf",
    "vrphoto.video.encode",
    "vrphoto.video.spatial",
    "vrphoto.webxr.server",
    "vrphoto.webxr.scene",
    "vrphoto.addons.manager",
    "vrphoto.models.ort",
    "vrphoto.gui.app",
)
#: vrphoto sub-packages that are not imported by the walk (run elsewhere / vendored).
_SKIP_PREFIXES = ("vrphoto.addons.workers", "vrphoto.third_party", "vrphoto.__main__")

#: Files of the WebXR viewer that must be bundled (relative to ``webxr/viewer``).
VIEWER_FILES = (
    "index.html",
    "view.html",
    "scene.schema.json",
    "vendor/three/build/three.module.js",
    "vendor/three/build/three.core.js",
    "vendor/spark/dist/spark.module.js",
)


def tiny_onnx_model() -> bytes:
    """Bytes of the embedded test model (see :data:`TINY_ONNX_B64`)."""
    return base64.b64decode(TINY_ONNX_B64)


def tiny_onnx_reference(x: np.ndarray) -> np.ndarray:
    """NumPy reference output of the embedded test model for input ``x`` [1,3,8,8]."""
    pad = np.pad(x[0].astype(np.float64), ((0, 0), (1, 1), (1, 1)))
    s = np.zeros((8, 8))
    for dy in range(3):
        for dx in range(3):
            s += pad[:, dy : dy + 8, dx : dx + 8].sum(0)
    conv = s / 27.0
    y = np.stack([np.maximum(2.0 * (conv + b), 0.0) for b in _TINY_BIAS])[None]
    return y.astype(np.float32)


class CheckFailed(Exception):
    """A self-test check failed (message is shown to the user)."""


# --------------------------------------------------------------------------- checks
def check_imports() -> dict[str, Any]:
    import importlib
    import pkgutil

    import vrphoto

    failed: dict[str, str] = {}
    third: dict[str, str] = {}
    for name in THIRD_PARTY:
        try:
            mod = importlib.import_module(name)
            third[name] = str(
                getattr(mod, "__version__", "")
                or getattr(sys.modules.get(name.split(".")[0]), "__version__", "")
            )
        except Exception as e:  # noqa: BLE001 - report every import problem
            failed[name] = f"{type(e).__name__}: {e}"
    count = 0
    names = sorted(
        m.name
        for m in pkgutil.walk_packages(
            vrphoto.__path__, "vrphoto.", onerror=lambda n: failed.setdefault(n, "walk failed")
        )
        if not m.name.startswith(_SKIP_PREFIXES)
    )
    for name in dict.fromkeys([*KEY_MODULES, *names]):
        try:
            importlib.import_module(name)
            count += 1
        except Exception as e:  # noqa: BLE001
            failed[name] = f"{type(e).__name__}: {e}"
    if failed:
        raise CheckFailed("Cannot import: " + "; ".join(f"{k} ({v})" for k, v in sorted(failed.items())))
    return {"modules": count, "third_party": third}


def check_onnxruntime() -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        raise CheckFailed(f"ONNX Runtime cannot be loaded: {e}") from e
    providers = list(ort.get_available_providers())
    model = tiny_onnx_model()
    rng = np.random.default_rng(0)
    x = rng.random((1, 3, 8, 8), dtype=np.float32)
    ref = tiny_onnx_reference(x)
    runs: dict[str, str] = {}
    skip = {"AzureExecutionProvider", "TensorrtExecutionProvider", "OpenVINOExecutionProvider"}
    for p in providers:
        if p in skip:
            runs[p] = "skipped"
            continue
        try:
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if p == "DmlExecutionProvider":  # DirectML needs these (as models.ort sets them)
                so.enable_mem_pattern = False
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            chain = [p] if p == "CPUExecutionProvider" else [p, "CPUExecutionProvider"]
            sess = ort.InferenceSession(model, sess_options=so, providers=chain)
            used = sess.get_providers()[0]
            y = sess.run(None, {"x": x})[0]
            err = float(np.abs(y - ref).max())
            if used != p:
                runs[p] = f"not usable here (fell back to {used})"
            elif err > 1e-3:
                runs[p] = f"WRONG RESULT (max error {err:.3g})"
            else:
                runs[p] = "ok"
        except Exception as e:  # noqa: BLE001 - GPU providers may fail on machines without the GPU
            runs[p] = f"failed: {str(e).splitlines()[0][:200]}"
    if runs.get("CPUExecutionProvider") != "ok":
        raise CheckFailed(
            f"The CPU provider could not run the test model: {runs.get('CPUExecutionProvider')}"
        )
    bad = [p for p, r in runs.items() if r.startswith("WRONG")]
    if bad:
        raise CheckFailed(f"Wrong results from {', '.join(bad)}")
    return {"version": ort.__version__, "providers": providers, "runs": runs}


def _synthetic_frames(n: int, w: int = 256, h: int = 256) -> list[np.ndarray]:
    frames = []
    for i in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[..., 0] = (i * 20) % 256
        f[..., 1] = (np.arange(w) * 255 // (w - 1))[None, :]
        f[..., 2] = (np.arange(h) * 255 // (h - 1))[:, None]
        frames.append(f)
    return frames


def check_ffmpeg(tmp: Path) -> dict[str, Any]:
    from .core.config import get_settings
    from .video.decode import iter_frames
    from .video.encode import VideoWriter, choose_encoder, ffmpeg_exe, probe_encoders, resolve_preset

    exe = ffmpeg_exe()
    encoders = probe_encoders()
    settings = get_settings()
    preset = resolve_preset("auto", settings.quest_model)
    enc = choose_encoder(preset, settings.video_encoder or "auto")
    frames = _synthetic_frames(10)
    out = tmp / "selftest.mp4"
    with VideoWriter(out, 256, 256, 10.0, preset, encoder=settings.video_encoder or "auto") as vw:
        for f in frames:
            vw.write(f)
        used = vw.encoder
    got = list(iter_frames(out))
    if len(got) != 10:
        raise CheckFailed(f"Decoded {len(got)} of 10 frames (encoder {used})")
    if got[0].shape != (256, 256, 3):
        raise CheckFailed(f"Decoded frame size {got[0].shape} != (256, 256, 3)")
    err = float(np.abs(got[5].astype(np.float32) - frames[5].astype(np.float32)).mean())
    if err > 12.0:
        raise CheckFailed(f"Decoded frames differ too much from the input (mean error {err:.1f})")
    return {
        "ffmpeg": exe,
        "preset": preset,
        "encoder": used,
        "preferred": enc,
        "encoders": [k for k, v in encoders.items() if v],
        "mean_error": round(err, 2),
    }


def check_spatial(tmp: Path) -> dict[str, Any]:
    from .video.encode import VideoWriter
    from .video.spatial import inject, read

    path = tmp / "spatial_180_LR.mp4"
    with VideoWriter(path, 256, 128, 10.0, "h264_compat") as vw:
        for f in _synthetic_frames(3, 256, 128):
            vw.write(f)
    inject(path, coverage=180, stereo="lr")
    meta = read(path)
    if not meta.get("spherical") or meta.get("stereo") != "lr" or meta.get("coverage") != 180:
        raise CheckFailed(f"Metadata not read back correctly: {meta}")
    return {"metadata": {k: meta.get(k) for k in ("spherical", "stereo", "coverage", "projection", "v2")}}


def _fetch(url: str, *, insecure: bool = False, timeout: float = 15.0) -> tuple[int, bytes]:
    ctx = None
    if insecure:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    handler = urllib.request.HTTPSHandler(context=ctx) if insecure else urllib.request.HTTPHandler()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), handler)
    with opener.open(url, timeout=timeout) as resp:
        return int(resp.status), resp.read()


def check_https_server(tmp: Path) -> dict[str, Any]:
    from .webxr.server import ViewerServer

    lib = tmp / "library"
    srv = ViewerServer(lib, https_port=0, http_port=0, cert_dir=tmp / "certs", bind="127.0.0.1")
    srv.start()
    try:
        https_port, http_port = srv.https_port, srv.http_port
        status, body = _fetch(f"https://127.0.0.1:{https_port}/index.html", insecure=True)
        if status != 200 or b"<html" not in body.lower():
            raise CheckFailed(f"HTTPS index.html returned status {status} ({len(body)} bytes)")
        status2, body2 = _fetch(f"http://127.0.0.1:{http_port}/api/scenes")
        if status2 != 200:
            raise CheckFailed(f"HTTP /api/scenes returned status {status2}")
        scenes = json.loads(body2.decode("utf-8"))
        cert = tmp / "certs" / "cert.pem"
        return {
            "https_port": https_port,
            "http_port": http_port,
            "index_bytes": len(body),
            "scenes": len(scenes) if isinstance(scenes, list) else None,
            "cert": cert.is_file(),
        }
    finally:
        srv.stop()


def check_viewer_assets() -> dict[str, Any]:
    from .core.config import resource_path

    root = resource_path("webxr", "viewer")
    missing = [f for f in VIEWER_FILES if not (root / f).is_file() or (root / f).stat().st_size == 0]
    if missing:
        raise CheckFailed(f"Missing viewer files in {root}: {', '.join(missing)}")
    sizes = {f: (root / f).stat().st_size for f in VIEWER_FILES}
    return {"folder": str(root), "bytes": sum(sizes.values())}


def check_addons() -> dict[str, Any]:
    from .addons import manager

    manager.reload_manifests()
    items = manager.list_addons(include_experimental=True)
    if not items:
        raise CheckFailed("No add-on manifests were found.")
    missing = []
    for m in items:
        if m.kind == "venv":
            w = m.worker_path()
            if w is None or not Path(w).is_file():
                missing.append(f"{m.id} ({m.worker})")
    if missing:
        raise CheckFailed(f"Add-on worker scripts missing: {', '.join(missing)}")
    return {"addons": [m.id for m in items]}


def check_uv() -> dict[str, Any]:
    import subprocess

    from .addons.env import popen_kwargs
    from .addons.manager import find_uv

    uv = find_uv()
    try:
        proc = subprocess.run([uv, "--version"], capture_output=True, text=True, timeout=60, **popen_kwargs())
    except (OSError, subprocess.SubprocessError) as e:
        raise CheckFailed(f"uv ({uv}) cannot be started: {e}") from e
    if proc.returncode != 0:
        raise CheckFailed(f"'uv --version' failed: {(proc.stderr or proc.stdout).strip()[:300]}")
    return {"uv": uv, "version": proc.stdout.strip()}


def _synthetic_photo(h: int = 192, w: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx * 255 // w, yy * 255 // h, (xx + yy) * 255 // (w + h)], -1).astype(np.uint8)
    img[h // 2 : h - 10, w // 3 : 2 * w // 3] = (220, 70, 50)
    return img


def check_geometry(tmp: Path) -> dict[str, Any]:
    from .core.types import DepthScale, StereoParams
    from .depth import get_estimator
    from .geometry.camera import estimate_intrinsics
    from .geometry.gltf import write_glb
    from .geometry.layers import build_layers
    from .geometry.mesh import layered_meshes
    from .geometry.stereo import compose, stereo_pair
    from .inpaint import get_inpainter

    img = _synthetic_photo()
    res = get_estimator("heuristic").estimate(img)
    if res.disparity.shape != img.shape[:2]:
        raise CheckFailed(f"Depth map size {res.disparity.shape} != image size {img.shape[:2]}")
    K = estimate_intrinsics(img.shape)
    inv = DepthScale(1.2, 15.0).to_inv_z(res.disparity)
    li = build_layers(img, inv, K, max_baseline_m=0.4, inpainter=get_inpainter("opencv"))
    left, right = stereo_pair(li, StereoParams())
    sbs = compose(left, right, "sbs")
    if sbs.shape != (img.shape[0], 2 * img.shape[1], 3):
        raise CheckFailed(f"Side-by-side size {sbs.shape} is wrong")
    meshes = layered_meshes(li, target_faces=5000, anchor_distance_m=2.0)
    glb = write_glb(meshes, tmp / "selftest.glb")
    size = Path(glb).stat().st_size
    if size < 1000:
        raise CheckFailed(f"GLB file is too small ({size} bytes)")
    return {"layers": len(li.layers), "meshes": len(meshes), "glb_bytes": size}


def check_data_folder() -> dict[str, Any]:
    from .core.config import data_root

    root = data_root()
    probe = root / f".selftest-{time.time_ns()}.tmp"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as e:
        raise CheckFailed(f"The data folder {root} is not writable: {e}") from e
    return {"folder": str(root)}


def check_models(tmp: Path) -> dict[str, Any]:
    from .depth import get_estimator
    from .models.download import ensure_model

    path = ensure_model("da2-small")
    est = get_estimator("da2-small")
    img = _synthetic_photo()
    res = est.estimate(img)
    d = res.disparity
    if d.shape != img.shape[:2] or not np.isfinite(d).all() or d.min() < -1e-3 or d.max() > 1.0 + 1e-3:
        raise CheckFailed("Depth Anything V2 Small returned an invalid depth map.")
    provider = getattr(getattr(est, "_model", None), "provider", None)
    return {"model": str(path), "provider": provider, "model_id": res.model_id}


def _devices() -> dict[str, Any]:
    from .models.gpu import system_info

    return system_info()


# --------------------------------------------------------------------------- runner
def _plan(tmp: Path, models: bool) -> list[tuple[str, Callable[[], dict[str, Any]], bool]]:
    """``(name, check, required)`` in run order; checks that write files use ``tmp``."""
    plan: list[tuple[str, Callable[[], dict[str, Any]], bool]] = [
        ("imports", check_imports, True),
        ("data_folder", check_data_folder, True),
        ("onnxruntime", check_onnxruntime, True),
        ("ffmpeg", lambda: check_ffmpeg(tmp), True),
        ("spatial_metadata", lambda: check_spatial(tmp), True),
        ("https_server", lambda: check_https_server(tmp), True),
        ("viewer_assets", check_viewer_assets, True),
        ("addons", check_addons, True),
        ("uv", check_uv, True),
        ("geometry", lambda: check_geometry(tmp), True),
        ("devices", _devices, False),
    ]
    if models:
        plan.append(("models", lambda: check_models(tmp), True))
    return plan


def check_names(models: bool = False) -> list[str]:
    """Names of the checks :func:`run_selftest` runs, in order (e.g. for a progress bar)."""
    return [name for name, _fn, _req in _plan(Path(tempfile.gettempdir()), models)]


def _run_check(
    results: dict[str, Any], name: str, fn: Callable[[], dict[str, Any]], *, required: bool = True
) -> None:
    t0 = time.perf_counter()
    entry: dict[str, Any] = {"ok": False, "required": required}
    try:
        detail = fn()
        entry["ok"] = True
        entry["detail"] = detail
    except Exception as e:  # noqa: BLE001 - a check must never crash the self-test
        msg = str(e) or type(e).__name__
        hint = getattr(e, "hint", None)
        entry["error"] = msg + (f" ({hint})" if hint else "")
        if not isinstance(e, CheckFailed):
            entry["traceback"] = traceback.format_exc(limit=8)
        log.warning("Self-test check %s failed: %s", name, entry["error"])
    entry["seconds"] = round(time.perf_counter() - t0, 3)
    results[name] = entry


def run_selftest(
    out: Path | None = None, *, models: bool = False, progress: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Run all checks; returns ``{"ok": bool, "checks": {...}, ...}`` (JSON-able).

    ``out``: also write the JSON report there.  ``models``: also download and run Depth
    Anything V2 Small (needs internet on first use).  ``progress(name)`` is called before
    each check.
    """
    from .core.config import is_frozen

    t0 = time.perf_counter()
    checks: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="vrphoto-selftest-", ignore_cleanup_errors=True) as td:
        for name, fn, required in _plan(Path(td), models):
            if progress is not None:
                progress(name)
            _run_check(checks, name, fn, required=required)
    ok = all(c["ok"] for c in checks.values() if c.get("required", True))
    report: dict[str, Any] = {
        "ok": ok,
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "frozen": is_frozen(),
        "executable": sys.executable,
        "elapsed_s": round(time.perf_counter() - t0, 3),
        "failed": [n for n, c in checks.items() if not c["ok"] and c.get("required", True)],
        "checks": checks,
    }
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("Self-test %s in %.1f s", "passed" if ok else "FAILED", report["elapsed_s"])
    return report


__all__ = [
    "run_selftest",
    "tiny_onnx_model",
    "tiny_onnx_reference",
    "TINY_ONNX_B64",
    "VIEWER_FILES",
    "CheckFailed",
    "check_names",
]
