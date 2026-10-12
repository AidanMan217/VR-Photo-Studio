"""Job runner: turns a job spec (:mod:`vrphoto.pipeline.jobs`) into output files.

:func:`run_job` dispatches on ``spec.kind`` and processes every input path in turn (the
``capture`` mode uses all inputs together).  Each mode is built from the module contracts
(docs/CONTRACTS.md): load + downscale the photo, estimate depth (cached), choose the camera
intrinsics, build a layered depth image with "look behind" inpainting, then render /
mesh / encode, save files with Quest player name tokens and (optionally) export a scene
to the WebXR viewer library.

Behaviour shared by all modes:

* progress is reported in sub-ranges per input and per step, cancellation is checked
  between steps (and per frame for videos);
* outputs are never overwritten (``name (2)_LR.jpg``);
* fallbacks (heuristic depth because the AI model is unavailable, simple fill instead of
  AI look-behind, blur instead of AI outpainting) are reported in ``JobResult.warnings``;
* results produced with non-commercial models / add-ons carry a notice in the viewer
  scene manifest (and a warning);
* with several inputs, a failing input is reported as a warning and the others still
  run; the job only fails when every input failed.

Heavy modules (geometry, video, webxr, addons) are imported lazily so importing this
module (e.g. by the GUI job queue) stays fast.
"""

from __future__ import annotations

import errno
import inspect
import io
import json
import math
import os
import secrets
import shutil
import struct
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import Settings
from ..core.errors import (
    MODE_VIDEO,
    UI_ADDONS,
    UI_LOG,
    UI_SETTINGS_MODELS,
    AddonError,
    AddonNotInstalled,
    Cancelled,
    InputError,
    VRPhotoError,
)
from ..core.fsutil import atomic_write_bytes, replace_with_retry, temp_sibling
from ..core.imaging import is_image, is_video, limit_long, load_image
from ..core.log import get_logger
from ..core.types import (
    RGB,
    CancelToken,
    DepthResult,
    DepthScale,
    FloatMap,
    Intrinsics,
    LayeredImage,
    ProgressReporter,
    StereoParams,
)
from .context import RunContext
from .jobs import (
    CaptureJob,
    CommonOptions,
    DepthJob,
    JobBase,
    JobResult,
    ObjectJob,
    PanoJob,
    ParallaxJob,
    SceneJob,
    StereoJob,
    VideoJob,
    VR180Job,
)
from .naming import output_name, release_reserved, safe_base, unique_path

log = get_logger(__name__)

# --------------------------------------------------------------------------- presets
#: VR180 photo eye size (pixels per eye, 180x180 degrees) by headset.
VR180_EYE_PX: dict[str, int] = {"quest3": 3840, "quest2": 2880, "pcvr": 3840}
#: VR180 video eye size by headset (videos are rendered per frame on the CPU/GPU).
VIDEO_VR180_EYE_PX: dict[str, int] = {"quest3": 2048, "quest2": 1920, "pcvr": 2048}
#: 360 equirect width by headset (stereo top-bottom output is width x width).
PANO_WIDTH: dict[str, int] = {"quest3": 6144, "quest2": 4096, "pcvr": 6144}
JPEG_QUALITY = 92
#: Frame rate of "still" videos (a photo shown for N seconds, for players without photo support).
STILL_VIDEO_FPS = 10.0
#: Height of the original camera (the viewer's eye) in photo3d scenes.
EYE_HEIGHT_M = 1.5
#: Viewer sphere radius for VR180 / 360 scenes.
SPHERE_RADIUS_M = 20.0
#: Virtual screen for flat 3D photos / videos in the viewer: at most this wide (the viewer's
#: default) and this tall (centred at 1.6 m, a taller screen would reach into the floor).
SCREEN_MAX_WIDTH_M = 3.0
SCREEN_MAX_HEIGHT_M = 2.2
#: Longest side of a flat "living photo" video when no width is given: plays smoothly on
#: phones and PCs (H.264 level limits) and keeps files small.
FLAT_VIDEO_MAX_LONG = 1920
#: Highest frame rate of converted videos (faster clips are converted at this rate).
MAX_VIDEO_FPS = 60.0
#: Most Gaussians in an object / add-on splat (what a Quest 3 browser shows smoothly).
MAX_OBJECT_SPLATS = 750_000
WIGGLE_FRAMES = 12
WIGGLE_MAX_PX = 640
DEPTH_SCENE_PX = 1024

#: Object methods that run in an add-on: method -> (add-on id, task).
ADDON_METHODS: dict[str, tuple[str, str]] = {
    "trellis": ("trellis", "image_to_3d"),
    "trellis2": ("trellis2", "image_to_3d"),
    "hunyuan": ("hunyuan", "image_to_3d"),
    "sharp": ("sharp", "image_to_splat"),
    # SHARP 360: SHARP's photo front + a predicted back (pipeline/sharp360.py); the back
    # sources are optional and fall back with a warning, so only SHARP is required
    "sharp360": ("sharp", "image_to_splat"),
}

STEREO_LAYOUTS = ("sbs", "half_sbs", "tb", "half_tb", "anaglyph", "anaglyph_gray", "cross")
LOOK_BEHIND = ("none", "band")
SCENE_LOOK_BEHIND = ("none", "band", "cleanplate")
FILLS = ("blur", "black", "outpaint")
IMAGE_FORMATS = ("jpg", "jpeg", "png")
PARALLAX_OUTPUTS = ("flat", "sbs", "vr180")
VIDEO_OUTPUTS = ("sbs", "half_sbs", "tb", "half_tb", "vr180")
OBJECT_METHODS = ("billboard", *ADDON_METHODS)
PLACEMENTS = ("hit-test", "front-floor")
BACKS = ("diffuse", "mirror")
#: SHARP 360 options (see pipeline/sharp360.py).
BACK_SOURCES = ("auto", "shell")
BACK_QUALITIES = ("fast", "balanced", "best")
SIZE_FROM = ("auto", "height", "photo")
QUEST_MODELS = ("quest3", "quest2", "pcvr")

_MODEL_ALIASES = {
    "moge2": "moge2-vitl",
    "moge": "moge2-vitl",
    "da3": "da3-mono-large",
    "isnet": "isnet-general",
}
_SIMPLE_INPAINTERS = ("opencv", "telea", "ns", "pushpull")


# --------------------------------------------------------------------------- helpers
def _call_opt(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` dropping keyword arguments it does not accept.

    Several modules accept ``progress=``/``cancel=`` as additive extensions of their
    contract; passing them only where supported keeps this module working with any
    implementation that matches the contract.
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return fn(*args, **kwargs)
    return fn(*args, **{k: v for k, v in kwargs.items() if k in params})


def _choice(value: Any, allowed: tuple[str, ...], what: str) -> str:
    v = str(value).strip().lower() if value is not None else ""
    if v not in allowed:
        raise InputError(f"Unknown {what} {value!r}.", hint=f"Choose one of: {', '.join(allowed)}.")
    return v


def _positive(value: Any, what: str, *, allow_zero: bool = False) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise InputError(f"{what} must be a number (got {value!r}).") from None
    if not math.isfinite(v) or v < 0 or (v == 0 and not allow_zero):
        raise InputError(
            f"{what} must be {'zero or ' if allow_zero else ''}a positive number (got {value!r})."
        )
    return v


def _as_reporter(progress: Any) -> ProgressReporter:
    if progress is None:
        return ProgressReporter.null()
    if isinstance(progress, ProgressReporter):
        return progress
    if callable(progress):
        return ProgressReporter(progress)
    raise TypeError("progress must be a ProgressReporter or a callable(frac, msg)")


def _prefixed(rep: ProgressReporter, prefix: str) -> ProgressReporter:
    if not prefix:
        return rep
    return ProgressReporter(lambda f, m: rep(f, f"{prefix}{m}"))


def _resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    if img.shape[1] == w and img.shape[0] == h:
        return img
    interp = cv2.INTER_AREA if (w < img.shape[1] and h < img.shape[0]) else cv2.INTER_CUBIC
    return cv2.resize(img, (int(w), int(h)), interpolation=interp)


def _is_equirect(rgb: np.ndarray) -> bool:
    h, w = rgb.shape[:2]
    return h > 0 and w >= 256 and abs(w / h - 2.0) <= 0.02


def _asset_name(path: Path, fallback: str) -> str:
    """Use the output file name inside the viewer scene when it is plain ASCII."""
    from ..webxr.scene import is_safe_asset_name

    name = path.name
    if name.isascii() and " " not in name and is_safe_asset_name(name):
        return name
    return fallback


def _concat_splats(clouds: list[Any]) -> Any:
    from ..core.types import SplatCloud

    return SplatCloud(
        means=np.concatenate([c.means for c in clouds]).astype(np.float32),
        scales=np.concatenate([c.scales for c in clouds]).astype(np.float32),
        quats=np.concatenate([c.quats for c in clouds]).astype(np.float32),
        colors=np.concatenate([c.colors for c in clouds]).astype(np.float32),
        opacity=np.concatenate([c.opacity for c in clouds]).astype(np.float32),
    )


def _splats_cv_to_gltf(sc: Any) -> Any:
    """Convert a splat cloud from the OpenCV camera frame (y down, z forward) to glTF."""
    from ..core.types import SplatCloud

    q = np.asarray(sc.quats, dtype=np.float32).copy()
    q[:, 2] *= -1.0  # conjugation by a 180 degree rotation about x: (w, x, -y, -z)
    q[:, 3] *= -1.0
    return SplatCloud(
        means=(np.asarray(sc.means, dtype=np.float32) * np.array([1.0, -1.0, -1.0], np.float32)),
        scales=np.asarray(sc.scales, dtype=np.float32),
        quats=q,
        colors=np.asarray(sc.colors, dtype=np.float32),
        opacity=np.asarray(sc.opacity, dtype=np.float32),
    )


# --------------------------------------------------------------------------- GLB fitting
def _mat_trs(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T  # column-major
    t = np.array(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    x, y, z, w = (float(v) for v in node.get("rotation", [0.0, 0.0, 0.0, 1.0]))
    s = np.array(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    M = np.eye(4)
    M[:3, :3] = R * s[None, :]
    M[:3, 3] = t
    return M


def _root_nodes(doc: dict[str, Any]) -> list[int]:
    """Root node indices of the default scene (nodes nobody references, if there is no scene)."""
    nodes = doc.get("nodes", [])
    scenes = doc.get("scenes", [])
    if scenes:
        return [int(i) for i in scenes[int(doc.get("scene", 0))].get("nodes", [])]
    children = {int(c) for n in nodes for c in n.get("children", [])}
    return [i for i in range(len(nodes)) if i not in children]


def glb_bounds(doc: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """World-space bounding box of a glTF document's default scene (from accessor min/max)."""
    nodes = doc.get("nodes", [])
    roots = _root_nodes(doc)
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    stack = [(int(r), np.eye(4)) for r in roots]
    seen = 0
    while stack:
        idx, parent = stack.pop()
        seen += 1
        if seen > 100_000:
            return None
        node = nodes[idx]
        M = parent @ _mat_trs(node)
        if "mesh" in node:
            for prim in doc["meshes"][int(node["mesh"])].get("primitives", []):
                acc_i = prim.get("attributes", {}).get("POSITION")
                if acc_i is None:
                    continue
                acc = doc["accessors"][int(acc_i)]
                if "min" not in acc or "max" not in acc:
                    return None
                a, b = np.array(acc["min"], float), np.array(acc["max"], float)
                corners = np.array(
                    [[x, y, z, 1.0] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])]
                )
                wc = (M @ corners.T).T[:, :3]
                lo = np.minimum(lo, wc.min(0))
                hi = np.maximum(hi, wc.max(0))
        for c in node.get("children", []):
            stack.append((int(c), M))
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return None
    return lo, hi


def fit_glb_height(src: Path, dst: Path, height_m: float) -> bool:
    """Rescale a GLB losslessly so it is ``height_m`` tall, feet at y=0, centred on x/z.

    Adds one root node with scale + translation (materials, textures and animations are
    untouched).  Returns False (and writes nothing) if the file cannot be analysed; a
    failure to write ``dst`` (disk full, folder not writable) raises ``OSError``.
    """
    try:
        data = Path(src).read_bytes()
        magic, version, length = struct.unpack_from("<III", data, 0)
        if magic != 0x46546C67 or version != 2:
            return False
        jlen, jtype = struct.unpack_from("<II", data, 12)
        if jtype != 0x4E4F534A:
            return False
        doc = json.loads(data[20 : 20 + jlen].decode("utf-8"))
        rest = data[20 + jlen : length]
        bounds = glb_bounds(doc)
        if bounds is None:
            return False
        lo, hi = bounds
        height = float(hi[1] - lo[1])
        if not (height > 1e-6 and math.isfinite(height)):
            return False
        s = float(height_m) / height
        cx, cz = 0.5 * float(lo[0] + hi[0]), 0.5 * float(lo[2] + hi[2])
        roots = _root_nodes(doc)
        nodes = doc.setdefault("nodes", [])
        scenes = doc.get("scenes") or [{"nodes": roots}]
        si = int(doc.get("scene", 0)) if doc.get("scenes") else 0
        nodes.append(
            {
                "name": "vrphoto_fit_height",
                "scale": [s, s, s],
                "translation": [-cx * s, -float(lo[1]) * s, -cz * s],
                "children": roots,
            }
        )
        scenes[si]["nodes"] = [len(nodes) - 1]
        doc["scenes"] = scenes
        doc["scene"] = si
        js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        body = struct.pack("<II", len(js), 0x4E4F534A) + js + rest
        out = struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body
    except (OSError, ValueError, KeyError, IndexError, TypeError, struct.error) as e:
        log.info("Could not rescale %s: %s", src, e)
        return False
    # temp file + rename that waits out a briefly locked file (antivirus / indexer on Windows)
    atomic_write_bytes(dst, out, fsync=False)
    return True


# --------------------------------------------------------------------------- work folders
#: Work folders older than this are left over from a crashed / killed app and are deleted.
STALE_WORK_S = 2 * 24 * 3600.0
_swept: set[str] = set()


def _sweep_stale_work(root: Path, max_age_s: float = STALE_WORK_S) -> int:
    """Delete ``job-*`` work folders older than ``max_age_s`` (once per folder and process).

    Work folders are removed when a job ends, but a crash or a killed process leaves them
    behind (a capture can leave gigabytes).  Returns the number of folders removed.
    """
    key = os.path.normcase(str(root))
    if key in _swept:
        return 0
    _swept.add(key)
    now = time.time()
    removed = 0
    try:
        entries = list(root.glob("job-*"))
    except OSError:
        return 0
    for d in entries:
        try:
            if d.is_dir() and now - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    if removed:
        log.info("Removed %d stale work folder(s) from %s", removed, root)
    return removed


# --------------------------------------------------------------------------- output folder
#: Hint for an output folder the app may not write to.
WRITE_HINT = (
    "Choose another output folder in the options or in Settings. If Windows Security "
    "'Controlled folder access' is on, allow VR Photo Studio there."
)


#: Name prefix of the empty file :func:`_check_writable` creates (and removes at once).
WRITE_TEST_PREFIX = ".vrphoto-write-test-"


def _check_writable(folder: Path) -> None:
    """Fail before any slow work when files cannot be created in ``folder``.

    ``mkdir(exist_ok=True)`` succeeds for an existing read-only folder (or one protected by
    Windows 'Controlled folder access'), which would otherwise only show up when the
    result is saved, after all the processing.

    One plain create of a uniquely named file, deliberately not :mod:`tempfile`: on Windows
    its name loop takes every "access denied" for a name clash while the folder looks
    writable and tries again - up to ``os.TMP_MAX`` (2**31) times on Python 3.11 and older
    3.12 releases, i.e. a hang on exactly such a folder - and Controlled folder access
    reports every attempt as a blocked change.
    """
    o_temp = getattr(os, "O_TEMPORARY", 0)  # Windows: deleted when closed, even if antivirus peeks
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOINHERIT", 0) | o_temp
    for _ in range(3):  # (a name clash is practically impossible)
        probe = folder / f"{WRITE_TEST_PREFIX}{os.getpid()}-{secrets.token_hex(6)}.tmp"
        try:
            fd = os.open(probe, flags, 0o600)
        except FileExistsError:
            continue
        except OSError as e:
            raise InputError(f"Cannot save files into {folder} ({e.strerror or e}).", hint=WRITE_HINT) from e
        os.close(fd)
        if not o_temp:
            try:
                probe.unlink()
            except OSError as e:  # pragma: no cover - created but not removable: harmless
                log.debug("Could not remove the write test file %s: %s", probe, e)
        return


# --------------------------------------------------------------------------- run state
@dataclass
class _Item:
    """One input being processed."""

    index: int
    path: Path
    base: str
    prefix: str = ""
    notices: list[str] = field(default_factory=list)
    work: Path | None = None
    label: str = ""  # human-readable title (unsanitised name), default: base

    @property
    def title(self) -> str:
        return (self.label or self.base).strip()[:200] or self.base

    def notice(self, text: str) -> None:
        if text not in self.notices:
            self.notices.append(text)


@dataclass
class _Photo:
    rgb: RGB
    scale: float
    exif_hfov: float | None
    alpha: FloatMap | None = None  # the file's own transparency (only when asked for and present)


@dataclass
class _Depth:
    res: DepthResult  # as produced by the estimator (possibly cached)
    disparity: FloatMap  # refined, [0, 1], 1 = near
    inv_z: FloatMap
    K: Intrinsics

    def result(self) -> DepthResult:
        """The refined depth as a DepthResult (for consumers that take one)."""
        r = self.res
        return DepthResult(self.disparity, r.model_id, metric_m=r.metric_m, intrinsics=r.intrinsics)


class _Run:
    def __init__(self, spec: JobBase, ctx: RunContext, cancel: CancelToken) -> None:
        self.spec = spec
        self.common: CommonOptions = spec.common if spec.common is not None else CommonOptions()
        self.ctx = ctx
        self.settings: Settings = ctx.settings
        self.cancel = cancel
        self.result = JobResult()
        self.multi = len(spec.inputs) > 1
        self.reserved: list[Path] = []  # output names reserved by this job (released at the end)
        out = self.common.output_dir
        try:
            self.out_dir = Path(out).expanduser() if out else Path(self.settings.outputs_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise InputError(
                f"Cannot create the output folder {out or self.settings.output_dir}: {e}",
                hint="Choose another output folder in the options or in Settings.",
            ) from e
        _check_writable(self.out_dir)
        self.quest = _choice(self.common.quest_model or "quest3", QUEST_MODELS, "headset")

    # ---------------------------------------------------------------- helpers
    def check(self) -> None:
        self.cancel.check()

    def warn(self, msg: str) -> None:
        if msg not in self.result.warnings:
            self.result.warnings.append(msg)
            log.warning("%s", msg)

    def out_path(self, item: _Item, kind: str, ext: str) -> Path:
        p = unique_path(self.out_dir / output_name(item.base, kind, ext))
        self.reserved.append(p)
        return p

    def add_output(self, item: _Item, label: str, path: Path | str) -> None:
        key = f"{item.base}: {label}" if self.multi else label
        k, n = key, 2
        while k in self.result.outputs:
            k = f"{key} ({n})"
            n += 1
        self.result.outputs[k] = str(path)

    def work_dir(self, item: _Item) -> Path:
        if item.work is None:
            root = Path(self.settings.cache_dir) / "work"
            root.mkdir(parents=True, exist_ok=True)
            _sweep_stale_work(root)
            item.work = Path(tempfile.mkdtemp(prefix="job-", dir=root))
        return item.work

    def model_notice(self, item: _Item, model_id: str | None) -> None:
        """Add a non-commercial notice when ``model_id`` is a non-commercial registry model."""
        if not model_id:
            return
        mid = str(model_id).split(":")[-1].strip().lower()
        mid = _MODEL_ALIASES.get(mid, mid)
        try:
            from ..models.registry import get_spec

            spec = get_spec(mid)
        except Exception:
            return
        if getattr(spec, "noncommercial", False):
            text = f"Contains output of a non-commercial model ({spec.title}; {spec.license})."
            item.notice(text)
            self.warn(f"Note: {text} Do not use this result commercially.")

    def addon_notice(self, item: _Item, manifest: Any) -> None:
        if getattr(manifest, "noncommercial", False):
            text = f"Contains output of {manifest.title} (non-commercial use only: {manifest.license})."
            item.notice(text)
            self.warn(f"Note: {text}")
        regions = getattr(manifest, "region_restricted", None)
        if regions:
            item.notice(f"Made with {manifest.title}; its license is not valid in: {', '.join(regions)}.")


# --------------------------------------------------------------------------- shared steps
def _max_process_px(run: _Run) -> int:
    max_px = int(run.common.max_process_px or 0) or int(run.settings.max_process_px)
    if max_px < 256:
        raise InputError(f"Max processing size must be at least 256 px (got {max_px}).")
    return max_px


def _load_photo(
    run: _Run, item: _Item, rep: ProgressReporter, *, with_alpha: bool = False, max_px: int | None = None
) -> _Photo:
    """Load + downscale (``max_process_px``, or ``max_px`` when given) a photo.  ``with_alpha``
    also returns the file's transparency as ``_Photo.alpha`` when it has a real one (e.g. an
    iPhone "lift subject" PNG)."""
    from ..core.imaging import exif_hfov

    rep(0.0, f"Loading {item.path.name}")
    if is_video(item.path):
        raise InputError(
            f"{item.path.name} is a video; this mode needs photos.",
            hint=f"Use '{MODE_VIDEO}' for videos.",
        )
    limit = _max_process_px(run) if max_px is None else int(max_px)
    img = load_image(item.path, with_alpha=with_alpha, max_px=limit)  # reduced-scale JPEG decode
    img, s = limit_long(img, limit)
    alpha: FloatMap | None = None
    if img.ndim == 3 and img.shape[2] == 4:
        a = img[..., 3].astype(np.float32) / 255.0
        img = img[..., :3]
        # only a real cut-out counts: some transparent AND some opaque pixels
        if float((a < 0.5).mean()) > 0.01 and float((a >= 0.5).mean()) > 0.002:
            alpha = a
    rgb = img
    if s < 1.0:
        log.info(
            "%s downscaled by %.3f to %dx%d for processing", item.path.name, s, rgb.shape[1], rgb.shape[0]
        )
    try:
        hf = exif_hfov(item.path)
    except InputError:
        hf = None
    run.check()
    rep(1.0, "Photo loaded")
    return _Photo(np.ascontiguousarray(rgb), s, hf, alpha)


def _depth_scale(run: _Run) -> DepthScale:
    near = _positive(run.common.near_m, "Nearest distance")
    far = _positive(run.common.far_m, "Farthest distance")
    if far <= near:
        raise InputError(
            f"The farthest distance ({far} m) must be larger than the nearest distance ({near} m)."
        )
    return DepthScale(near, far)


def _note_depth_fallback(run: _Run, est: Any, res: DepthResult) -> None:
    requested = run.ctx.resolve_name("depth", run.common.depth_model).lower()
    used = str(res.model_id or "").lower()
    if requested != "heuristic" and (used == "heuristic" or used.endswith(":heuristic")):
        reason = getattr(est, "fallback_reason", None)
        run.warn(
            "The AI depth model could not be used"
            + (f" ({reason})" if reason else "")
            + ". A simple built-in depth estimate was used instead, so the 3D effect may be weaker or "
            "less accurate. Models download on first use - check your internet connection, or pick a "
            f"depth model in {UI_SETTINGS_MODELS}."
        )


def _refine_disparity(run: _Run, disp: FloatMap, rgb: RGB) -> FloatMap:
    strength = float(run.common.depth_refine or 0.0)
    if strength <= 1e-3:
        return disp
    from ..depth.post import refine

    return refine(disp, rgb, strength)


def _inv_from(run: _Run, res: DepthResult, disparity: FloatMap, rgb: RGB) -> FloatMap:
    """Inverse depth (1/m): metric when the model knows distances, else DepthScale(near, far)."""
    if res.metric_m is not None and np.asarray(res.metric_m).shape == disparity.shape:
        z = np.asarray(res.metric_m, dtype=np.float32)
        good = np.isfinite(z) & (z > 0)
        if good.mean() > 0.5:
            z = np.where(good, z, np.float32(1e4))
            inv = 1.0 / np.clip(z, 0.05, 1e4)
            strength = float(run.common.depth_refine or 0.0)
            if strength > 1e-3:
                from ..depth.post import refine

                lo, hi = float(inv.min()), float(inv.max())
                if hi - lo > 1e-9:
                    d = refine(((inv - lo) / (hi - lo)).astype(np.float32), rgb, strength)
                    inv = lo + d * (hi - lo)
            return np.maximum(inv, 1e-4).astype(np.float32)
    return _depth_scale(run).to_inv_z(disparity)


def _estimate_depth(run: _Run, item: _Item, photo: _Photo, rep: ProgressReporter) -> _Depth:
    from ..geometry.camera import estimate_intrinsics

    rgb = photo.rgb
    _depth_scale(run)  # validate early
    est = run.ctx.estimator(run.common.depth_model, run.common.device)
    rep(0.0, "Estimating depth")
    res = run.ctx.depth(rgb, estimator=est, source=item.path, progress=rep.sub(0.0, 0.9), cancel=run.cancel)
    run.check()
    _note_depth_fallback(run, est, res)
    run.model_notice(item, res.model_id)
    disp = np.asarray(res.disparity, dtype=np.float32)
    if disp.shape != rgb.shape[:2]:
        disp = np.clip(_resize(disp, rgb.shape[1], rgb.shape[0]), 0.0, 1.0)
    rep(0.92, "Refining depth")
    disp = _refine_disparity(run, disp, rgb)
    inv = _inv_from(run, res, disp, rgb)
    K = estimate_intrinsics(rgb.shape, hfov_deg=run.common.hfov_deg, exif_hfov=photo.exif_hfov, depth=res)
    log.info("Depth: %s, camera %.1f deg horizontal FOV", res.model_id, K.hfov_deg)
    rep(1.0, "Depth ready")
    return _Depth(res, disp, inv, K)


def _note_inpainter(run: _Run, item: _Item, inp: Any) -> None:
    requested = run.ctx.resolve_name("inpaint", run.common.inpaint_model).lower()
    resolved = str(getattr(inp, "resolved", None) or getattr(inp, "id", "") or "").lower()
    if requested == "auto" and resolved in _SIMPLE_INPAINTERS:
        run.warn(
            "The AI look-behind model could not be used; hidden areas were filled with a simple method, "
            "so edges may look smeared. Models download on first use - check your internet connection."
        )
    run.model_notice(item, resolved)


def _build_layers(
    run: _Run,
    item: _Item,
    photo: _Photo,
    depth: _Depth,
    rep: ProgressReporter,
    *,
    max_baseline: float,
    look_behind: str,
    alpha: FloatMap | None = None,
) -> LayeredImage:
    from ..geometry.layers import build_layers

    mode = look_behind
    if mode == "band" and not max_baseline > 0:
        mode = "none"
    inp = run.ctx.inpainter(run.common.inpaint_model, run.common.device) if mode != "none" else None
    rep(0.0, "Preparing look-behind layers" if mode != "none" else "Preparing layers")
    li = build_layers(
        photo.rgb,
        depth.inv_z,
        depth.K,
        max_baseline_m=float(max(max_baseline, 0.0)),
        inpainter=inp,
        alpha=alpha,
        mode=mode,  # type: ignore[arg-type]
        progress=rep,
        cancel=run.cancel,
    )
    if inp is not None and len(li.layers) > 1:
        _note_inpainter(run, item, inp)
    run.check()
    return li


def _save_jpeg_or_png(path: Path, img: RGB, *, xmp: bytes | None = None) -> Path:
    """Save an RGB image (JPEG quality 92 with optional XMP, or PNG).

    :func:`~vrphoto.core.imaging.save_image` encodes in memory (retrying a failed optimised
    JPEG pass itself) and writes atomically, so a full disk raises instead of leaving a
    truncated file behind.
    """
    from ..core.imaging import save_image

    if path.suffix.lower() not in (".jpg", ".jpeg"):
        return save_image(path, img)
    return save_image(path, img, quality=JPEG_QUALITY, xmp=xmp)


def _export_scene(
    run: _Run,
    item: _Item,
    *,
    mode: str,
    block: dict[str, Any],
    assets: dict[str, Path | bytes],
    thumbnail: RGB | None,
    env: dict[str, Any] | None = None,
    rep: ProgressReporter | None = None,
) -> str | None:
    """Add a scene to the viewer library (when ``common.export_scene``); returns its id."""
    if not run.common.export_scene:
        return None
    from ..webxr.scene import export_scene

    if rep is not None:
        rep(0.0, "Adding to the VR viewer library")
    manifest: dict[str, Any] = {"title": item.title, "mode": mode, mode: block, "source": item.path.name}
    if env:
        manifest["env"] = env
    if item.notices:
        manifest["notices"] = list(item.notices)
    try:
        scene_dir = export_scene(
            manifest, assets, library=Path(run.settings.library_dir), thumbnail=thumbnail
        )
    except Cancelled:
        raise
    except (VRPhotoError, OSError) as e:
        run.warn(f"The result was saved, but it could not be added to the VR viewer library: {e}")
        return None
    sid = Path(scene_dir).name
    if run.result.scene_id is None:
        run.result.scene_id = sid
    run.add_output(item, "VR viewer scene", scene_dir)
    if rep is not None:
        rep(1.0, "Added to the VR viewer library")
    return sid


def _video_preset(run: _Run, preset: str | None, *, output: str | None = None) -> str:
    """Encoder preset for ``preset`` ("auto" -> by ``output`` kind and headset).

    ``output`` is the kind of video: ``"flat"`` is for phones / PCs (H.264, which every
    player decodes), anything else is watched in the headset (the Quest HEVC presets).
    """
    from ..video import encode

    p = str(preset or "auto").strip().lower()
    if p == "auto":
        fn = getattr(encode, "resolve_preset", None)
        if fn is not None:
            return str(_call_opt(fn, "auto", run.quest, output=output))
        if output == "flat":
            return "h264_compat"
        return {"quest3": "quest3_hevc", "quest2": "quest2_hevc"}.get(run.quest, "h264_compat")
    if p not in encode.PRESETS:
        raise InputError(
            f"Unknown video format {preset!r}.", hint=f"Choose one of: auto, {', '.join(encode.PRESETS)}."
        )
    return p


def _exact_fps(fps: float) -> float:
    """``fps`` with NTSC rates made exact: ffmpeg prints 24000/1001 as ``23.98``, and decoding
    at the rounded rate while the encoder writes 24000/1001 makes the picture drift behind
    the sound (about 0.2 ms per second)."""
    try:
        from ..video.encode import ntsc_rate
    except ImportError:  # pragma: no cover - older video module
        return fps
    exact = ntsc_rate(fps)
    return float(exact) if exact is not None else fps


def _video_writer(run: _Run, path: Path, w: int, h: int, fps: float, preset: str, **kw: Any) -> Any:
    from ..video.encode import VideoWriter

    encoder = str(getattr(run.settings, "video_encoder", "auto") or "auto")
    return _call_opt(VideoWriter, path, w, h, fps, preset, encoder=encoder, **kw)


def _close_video(run: _Run, vw: Any) -> Path:
    """Finish a VideoWriter and pass its non-fatal problems (e.g. sound dropped) to the result."""
    path = Path(vw.close())
    for w in getattr(vw, "warnings", None) or []:
        run.warn(str(w))
    return path


def _write_still_video(
    run: _Run,
    frame: RGB,
    path: Path,
    seconds: float,
    *,
    coverage: int | None,
    stereo: str,
    rep: ProgressReporter,
) -> Path:
    """Encode ``frame`` as an N-second video (+ spatial metadata when ``coverage`` is set)."""
    preset = _video_preset(run, "auto")
    n = max(1, int(round(seconds * STILL_VIDEO_FPS)))
    h, w = frame.shape[:2]
    vw = _video_writer(run, path, w, h, STILL_VIDEO_FPS, preset)
    try:
        for i in range(n):
            run.check()
            vw.write(frame)
            if i % 5 == 0:
                rep(i / n, "Writing video")
        out = _close_video(run, vw)
    except BaseException:
        vw.abort()
        raise
    if coverage:
        from ..video.spatial import inject

        inject(out, coverage=coverage, stereo=stereo)
    rep(1.0, "Video written")
    return out


def _addon_manifest(addon_id: str) -> Any:
    from ..addons import manager

    try:
        return manager.get_manifest(addon_id)
    except AddonError as e:
        raise AddonNotInstalled(
            f"The '{addon_id}' add-on is not available in this version of VR Photo Studio.",
            hint=f"Update VR Photo Studio, then install it in the {UI_ADDONS}.",
        ) from e


def _addon_installed(addon_id: str) -> bool:
    from ..addons import manager

    try:
        return bool(manager.status(addon_id).get("installed"))
    except AddonError:
        return False


def _require_addon(addon_id: str) -> Any:
    m = _addon_manifest(addon_id)
    if not _addon_installed(addon_id):
        raise AddonNotInstalled(
            f"{m.title} is not installed.",
            hint=f"Install it in the {UI_ADDONS}; it is downloaded once and then works offline.",
        )
    return m


def _free_app_gpu() -> None:
    """Close the app's own ONNX Runtime sessions (depth, cut-out, fill models) so an add-on
    process gets the graphics memory; they are re-opened on next use.  Never fails the job."""
    try:
        from ..models import ort

        ort.unload_model(wait=True)
    except Exception:
        log.warning("Could not release the app's AI models before running an add-on", exc_info=True)


def _run_addon(
    run: _Run, item: _Item, addon_id: str, task: str, inputs: dict, params: dict, rep: ProgressReporter
) -> dict[str, Any]:
    from ..addons import manager

    _free_app_gpu()  # 8 GB cards: a loaded depth / cut-out model can push the add-on out of memory
    work = run.work_dir(item) / addon_id
    work.mkdir(parents=True, exist_ok=True)
    try:
        outs = manager.run(addon_id, task, inputs, params, work, progress=rep, cancel=run.cancel)
    except AddonNotInstalled as e:
        raise AddonNotInstalled(str(e), hint=e.hint or f"Install it in the {UI_ADDONS}.") from e
    run.check()
    return dict(outs or {})


def _outpaint(
    run: _Run, item: _Item, img: RGB, fill_mask: np.ndarray, rep: ProgressReporter, *, wrap: str
) -> RGB | None:
    """AI-outpaint ``fill_mask`` (True = fill) with the sdxl360 add-on; None = fall back to blur."""
    if not _addon_installed("sdxl360"):
        run.warn(
            "AI outpainting needs the 'AI outpainting for VR180 / 360' add-on, which is not installed, so "
            f"a soft blur was used instead. Install it in the {UI_ADDONS} to use AI outpainting."
        )
        return None
    from ..core.imaging import save_image

    work = run.work_dir(item)
    img_p = save_image(work / "outpaint_input.png", img)
    mask_p = save_image(
        work / "outpaint_mask.png", np.repeat((fill_mask.astype(np.uint8) * 255)[..., None], 3, 2)
    )
    try:
        outs = _run_addon(
            run,
            item,
            "sdxl360",
            "outpaint_360",
            {"image": str(img_p), "mask": str(mask_p)},
            {"seed": 0, "name": "outpainted", "wrap": wrap},
            rep,
        )
    except Cancelled:
        raise
    except AddonError as e:
        run.warn(f"AI outpainting failed ({e}); a soft blur was used instead.")
        return None
    out = outs.get("image")
    if not out or not Path(out).is_file():
        run.warn("AI outpainting returned no image; a soft blur was used instead.")
        return None
    try:
        res = load_image(out)
    except InputError as e:
        run.warn(f"AI outpainting returned an unreadable image ({e}); a soft blur was used instead.")
        return None
    try:
        manifest = _addon_manifest("sdxl360")
    except AddonError:
        manifest = None
    if manifest is not None:
        run.addon_notice(item, manifest)
        _addon_warnings(run, manifest.title, outs)
    return _resize(res, img.shape[1], img.shape[0])


# --------------------------------------------------------------------------- modes
def _mode_depth(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    from ..core.imaging import colorize_depth, save_depth16

    spec: DepthJob = run.spec  # type: ignore[assignment]
    if not (spec.save_16bit or spec.save_color):
        raise InputError("Nothing to save: switch on the 16-bit depth PNG or the colour preview.")
    photo = _load_photo(run, item, rep.sub(0.0, 0.05))
    d = _estimate_depth(run, item, photo, rep.sub(0.05, 0.9))
    rep(0.92, "Saving depth maps")
    if spec.save_16bit:
        p = run.out_path(item, "depth", "png")
        save_depth16(p, d.disparity)
        run.add_output(item, "Depth map (16-bit PNG)", p)
    if spec.save_color:
        p = run.out_path(item, "depth_color", "jpg")
        _save_jpeg_or_png(p, colorize_depth(d.disparity))
        run.add_output(item, "Depth map (colour preview)", p)
    rep(1.0, "Done")


def _screen_width_m(aspect: float) -> float:
    """Width of the viewer's virtual screen for a picture of ``aspect`` (width / height).

    :data:`SCREEN_MAX_WIDTH_M` wide, but never taller than :data:`SCREEN_MAX_HEIGHT_M`
    (portrait photos / vertical videos), so the screen does not reach into the floor.  The
    same width goes into ``StereoParams.screen_width_m`` (the limit that keeps the
    background from needing eye divergence) and into the scene manifest, so the viewer
    shows the pair at the size it was made for.
    """
    a = float(aspect) if aspect and math.isfinite(aspect) and aspect > 0 else 16 / 9
    return round(min(SCREEN_MAX_WIDTH_M, SCREEN_MAX_HEIGHT_M * a), 4)


def _flat_stereo(aspect: float, **kw: Any) -> StereoParams:
    """Flat-mode stereo parameters for a picture of ``aspect`` shown on the viewer screen."""
    return StereoParams(mode="flat", screen_width_m=_screen_width_m(aspect), **kw)


def _aspect(img_or_K: Any) -> float:
    if isinstance(img_or_K, Intrinsics):
        return img_or_K.width / max(img_or_K.height, 1)
    h, w = img_or_K.shape[:2]
    return w / max(h, 1)


def _stereo_scene_block(media: str, kind: str, layout: str, left: RGB) -> dict[str, Any]:
    """Viewer block for a flat stereo (or mono) picture; ``left`` is one eye's view."""
    lay = "tb" if layout in ("tb", "half_tb") else "mono" if layout == "mono" else "sbs"
    aspect = _aspect(left)
    return {
        "media": media,
        "kind": kind,
        "layout": lay,
        "aspect": round(aspect, 6),
        "screen_width_m": _screen_width_m(aspect),
    }


def _save_gif(path: Path, frames: list[RGB], frame_ms: int = 80) -> Path:
    from PIL import Image

    ims = []
    for f in frames:
        small, _ = limit_long(f, WIGGLE_MAX_PX)
        ims.append(
            Image.fromarray(np.ascontiguousarray(small)).convert(
                "P", palette=Image.Palette.ADAPTIVE, colors=256
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()  # encoded in memory: a full disk raises instead of truncating the file
    ims[0].save(buf, "GIF", save_all=True, append_images=ims[1:], duration=frame_ms, loop=0, disposal=1)
    atomic_write_bytes(path, buf.getvalue(), fsync=False)
    return path


def _mode_stereo(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    from ..geometry.stereo import compose, stereo_pair, wiggle_frames

    spec: StereoJob = run.spec  # type: ignore[assignment]
    layout = _choice(spec.layout, STEREO_LAYOUTS, "stereo layout")
    look = _choice(spec.look_behind, LOOK_BEHIND, "look-behind setting")
    fmt = _choice(spec.image_format, IMAGE_FORMATS, "image format")
    ipd = _positive(spec.ipd_m, "Eye distance (IPD)")
    strength = _positive(spec.strength, "3D strength", allow_zero=True)

    photo = _load_photo(run, item, rep.sub(0.0, 0.05))
    d = _estimate_depth(run, item, photo, rep.sub(0.05, 0.4))
    li = _build_layers(run, item, photo, d, rep.sub(0.4, 0.6), max_baseline=ipd * strength, look_behind=look)
    params = _flat_stereo(
        _aspect(li.K),
        ipd_m=ipd,
        strength=strength,
        convergence_m=spec.convergence_m,
        max_parallax_pct=float(spec.max_parallax_pct),
        swap_eyes=bool(spec.swap_eyes),
    )
    rep(0.6, "Rendering the two eye views")
    left, right = _call_opt(stereo_pair, li, params, progress=rep.sub(0.6, 0.85), cancel=run.cancel)
    run.check()
    frame = compose(left, right, layout)
    ext = "png" if fmt == "png" else "jpg"
    rep(0.86, "Saving the 3D photo")
    p = run.out_path(item, layout, ext)
    _save_jpeg_or_png(p, frame)
    labels = {
        "sbs": "3D photo (side-by-side)",
        "half_sbs": "3D photo (half side-by-side)",
        "tb": "3D photo (top-bottom)",
        "half_tb": "3D photo (half top-bottom)",
        "anaglyph": "3D photo (red/cyan anaglyph)",
        "anaglyph_gray": "3D photo (grey red/cyan anaglyph)",
        "cross": "3D photo (cross-eyed)",
    }
    run.add_output(item, labels[layout], p)
    if spec.wiggle_gif:
        rep(0.88, "Rendering the wiggle animation")
        frames = _call_opt(
            wiggle_frames, li, params, WIGGLE_FRAMES, cancel=run.cancel, progress=rep.sub(0.88, 0.95)
        )
        g = run.out_path(item, "wiggle", "gif")
        _save_gif(g, frames)
        run.add_output(item, "Wiggle animation (GIF)", g)
    if layout in ("sbs", "half_sbs", "tb", "half_tb"):
        media = _asset_name(p, f"media.{ext}")
        _export_scene(
            run,
            item,
            mode="stereo",
            block=_stereo_scene_block(media, "image", layout, left),
            assets={media: p},
            thumbnail=left,
            rep=rep.sub(0.95, 1.0),
        )
    rep(1.0, "Done")


def _auto_focus(li: LayeredImage) -> float | None:
    try:
        from ..geometry.stereo import auto_convergence, visible_inv_z

        return float(auto_convergence(visible_inv_z(li)))
    except Exception:  # pragma: no cover - optional helper
        try:
            from ..geometry.stereo import auto_convergence

            return float(auto_convergence(li.front.inv_z))
        except Exception:
            return None


def _mode_parallax(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    from ..video.paths import make_path
    from ..video.render import render_path_frames, render_vr180_frames

    spec: ParallaxJob = run.spec  # type: ignore[assignment]
    output = _choice(spec.output, PARALLAX_OUTPUTS, "video type")
    look = _choice(spec.look_behind, LOOK_BEHIND, "look-behind setting")
    fps = _positive(spec.fps, "Frame rate")
    duration = _positive(spec.duration_s, "Duration")
    amp = _positive(spec.amplitude_m, "Movement size", allow_zero=True)
    ipd = _positive(spec.ipd_m, "Eye distance (IPD)")
    preset = _video_preset(run, spec.encoder_preset, output=output)
    n_frames = max(1, int(round(duration * fps)))
    path_args: dict[str, Any] = {
        "n_frames": n_frames,
        "amplitude_m": amp,
        "fps": fps,
        "loop": bool(spec.loop),
        "keyframes": spec.keyframes,
    }
    _call_opt(make_path, spec.path, **path_args)  # bad path / keyframes fail before the slow steps

    photo = _load_photo(run, item, rep.sub(0.0, 0.03))
    d = _estimate_depth(run, item, photo, rep.sub(0.03, 0.2))
    max_b = 2.0 * amp + (ipd if output != "flat" else 0.0)
    li = _build_layers(run, item, photo, d, rep.sub(0.2, 0.3), max_baseline=max_b, look_behind=look)
    focus = _auto_focus(li)
    path = _call_opt(make_path, spec.path, **path_args, focus_m=focus)
    K = li.K
    if spec.width:
        w = int(spec.width)
        if output == "sbs":
            w = max(2, w // 2)
        if w < 16:
            raise InputError(f"Video width must be at least 16 px (got {spec.width}).")
        if output != "vr180":
            K = K.scaled(w / K.width)
    elif output != "vr180":
        if output == "flat" and max(K.width, K.height) > FLAT_VIDEO_MAX_LONG:
            # a video for phones / PCs: a size every H.264 decoder plays, and small files
            K = K.scaled(FLAT_VIDEO_MAX_LONG / max(K.width, K.height))
        # never render more pixels than the video format can hold (e.g. 4096 px for H.264)
        n_eyes = 2 if output == "sbs" else 1
        enc_w = _encoded_width(n_eyes * K.width, K.height, preset) // n_eyes
        if enc_w < K.width:
            K = K.scaled(enc_w / K.width)
    render_rep = rep.sub(0.3, 0.97)
    if output == "flat":
        frames: Iterator[RGB] = _call_opt(
            render_path_frames, li, path, out_K=K, layout="mono", cancel=run.cancel
        )
        kind, label = "parallax", "Living photo video"
    elif output == "sbs":
        stereo = _flat_stereo(_aspect(K), ipd_m=ipd)
        frames = _call_opt(
            render_path_frames, li, path, out_K=K, stereo=stereo, layout="sbs", cancel=run.cancel
        )
        kind, label = "parallax_sbs", "Living photo video (3D side-by-side)"
    else:
        eye = int(spec.width) // 2 if spec.width else _video_eye_px(run, li.K)
        if eye < 16:
            raise InputError(f"Video width must be at least 32 px (got {spec.width}).")
        frames = _call_opt(
            render_vr180_frames,
            _scale_layers(li, _vr180_render_K(li.K, eye)),
            path,
            eye_px=eye,
            stereo=StereoParams(ipd_m=ipd, mode="vr"),
            cancel=run.cancel,
        )
        kind, label = "parallax_vr180", "Living photo video (VR180)"
    out = run.out_path(item, kind, "mp4")
    vw = None
    thumb: RGB | None = None
    try:
        for i, frame in enumerate(frames):
            run.check()
            if vw is None:
                vw = _video_writer(run, out, frame.shape[1], frame.shape[0], fps, preset)
                thumb = frame
            vw.write(frame)
            render_rep(i / n_frames, f"Rendering frame {i + 1}/{n_frames}")
        if vw is None:
            raise InputError("The camera path produced no frames.")
        rep(0.97, "Finishing the video")
        out = _close_video(run, vw)
    except BaseException:
        if vw is not None:
            vw.abort()
        raise
    if output == "vr180":
        from ..video.spatial import inject

        inject(out, coverage=180, stereo="lr")
    run.add_output(item, label, out)
    media = _asset_name(out, "media.mp4")
    if output == "vr180":
        block = {
            "media": media,
            "kind": "video",
            "coverage": 180,
            "layout": "sbs",
            "radius_m": SPHERE_RADIUS_M,
        }
        _export_scene(
            run,
            item,
            mode="sphere",
            block=block,
            assets={media: out},
            thumbnail=thumb[:, : thumb.shape[1] // 2] if thumb is not None else None,
        )
    else:
        if output == "sbs":
            eye_img = thumb[:, : thumb.shape[1] // 2] if thumb is not None else photo.rgb
            block = _stereo_scene_block(media, "video", "sbs", eye_img)
        else:
            eye_img = thumb if thumb is not None else photo.rgb
            block = _stereo_scene_block(media, "video", "mono", eye_img)
        _export_scene(run, item, mode="stereo", block=block, assets={media: out}, thumbnail=eye_img)
    rep(1.0, "Done")


def _vr180_render_K(K: Intrinsics, eye: int, size_mult: float = 1.0) -> Intrinsics:
    """Camera for rendering VR180 eye views at the output's angular resolution (never above the source)."""
    big = max(K.hfov_deg, K.vfov_deg) * size_mult
    need_w = eye * min(big, 150.0) / 180.0 * (K.width / max(K.width, K.height))
    s = min(1.0, need_w / K.width) if need_w > 0 else 1.0
    if s >= 0.999:
        return K
    w, h = max(16, int(round(K.width * s))), max(16, int(round(K.height * s)))
    return K.resized(w, h)


def _scale_layers(li: LayeredImage, K2: Intrinsics) -> LayeredImage:
    """Resample every layer to the size of ``K2`` (used to render big photos at the size needed)."""
    import cv2

    from ..core.types import Layer

    if (K2.width, K2.height) == (li.K.width, li.K.height):
        return li
    size = (K2.width, K2.height)
    layers = [
        Layer(
            cv2.resize(lay.rgb, size, interpolation=cv2.INTER_AREA),
            cv2.resize(np.asarray(lay.inv_z, np.float32), size, interpolation=cv2.INTER_NEAREST),
            np.clip(cv2.resize(np.asarray(lay.alpha, np.float32), size, interpolation=cv2.INTER_AREA), 0, 1),
        )
        for lay in li.layers
    ]
    return LayeredImage(layers, K2)


def _encoded_width(width: int, height: int, preset: str) -> int:
    """Width the encoder will produce for a frame (preset size caps), or ``width``."""
    try:
        from ..video.encode import output_size

        return int(output_size(width, height, preset)[0])
    except Exception:  # pragma: no cover - optional helper
        return width


def _video_eye_px(run: _Run, K: Intrinsics) -> int:
    """VR180 video eye size: enough for the source's angular resolution, capped by the headset preset."""
    big = max(K.hfov_deg, K.vfov_deg, 1.0)
    need = max(K.width, K.height) * 180.0 / big
    return int(max(256, min(VIDEO_VR180_EYE_PX[run.quest], round(need / 2.0) * 2)))


def _vr180_mask(left: RGB, K: Intrinsics, eye: int, size_mult: float) -> np.ndarray:
    """True where a VR180 eye image has no photo content (to be filled)."""
    import cv2

    from ..geometry.projection import vr180_frame

    white = np.full_like(left, 255)
    cov = vr180_frame(white, white, K, eye, size_mult=size_mult, fill="black", layout="sbs")[:, :eye, 0]
    mask = cov < 128
    return cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def _mode_vr180(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    from ..core.imaging import gpano_xmp
    from ..geometry.projection import vr180_frame
    from ..geometry.stereo import stereo_pair

    spec: VR180Job = run.spec  # type: ignore[assignment]
    fill = _choice(spec.fill, FILLS, "fill style")
    look = _choice(spec.look_behind, LOOK_BEHIND, "look-behind setting")
    fmt = _choice(spec.image_format, IMAGE_FORMATS, "image format")
    ipd = _positive(spec.ipd_m, "Eye distance (IPD)")
    size_mult = _positive(spec.size_mult, "Photo size")
    as_video = _positive(spec.as_video_s, "Video length", allow_zero=True)
    eye = int(spec.eye_px) if spec.eye_px else VR180_EYE_PX[run.quest]
    if eye < 64:
        raise InputError(f"VR180 eye size must be at least 64 px (got {eye}).")

    photo = _load_photo(run, item, rep.sub(0.0, 0.05))
    d = _estimate_depth(run, item, photo, rep.sub(0.05, 0.35))
    li = _build_layers(run, item, photo, d, rep.sub(0.35, 0.5), max_baseline=ipd, look_behind=look)
    # render the eyes at the angular resolution of the output (never above the source)
    dst_K = _vr180_render_K(li.K, eye, size_mult)
    rep(0.5, "Rendering the two eye views")
    params = StereoParams(ipd_m=ipd, mode="vr")
    left, right = _call_opt(stereo_pair, li, params, dst_K, progress=rep.sub(0.5, 0.75), cancel=run.cancel)
    run.check()
    fill_arg: Any = fill
    if fill == "outpaint":
        rep(0.75, "AI outpainting the surroundings")
        mono = vr180_frame(left, left, dst_K, eye, size_mult=size_mult, fill="black", layout="sbs")[:, :eye]
        painted = _outpaint(
            run, item, mono, _vr180_mask(left, dst_K, eye, size_mult), rep.sub(0.75, 0.85), wrap="no"
        )
        fill_arg = painted if painted is not None else "blur"
    rep(0.86, "Projecting to VR180")
    frame = vr180_frame(left, right, dst_K, eye, size_mult=size_mult, fill=fill_arg, layout="sbs")
    run.check()
    ext = "png" if fmt == "png" else "jpg"
    p = run.out_path(item, "vr180", ext)
    xmp = gpano_xmp(2 * eye, eye, eye, eye, eye // 2, 0)
    _save_jpeg_or_png(p, frame, xmp=xmp)
    run.add_output(item, "VR180 photo", p)
    if as_video > 0:
        v = run.out_path(item, "vr180", "mp4")
        _write_still_video(run, frame, v, as_video, coverage=180, stereo="lr", rep=rep.sub(0.9, 0.97))
        run.add_output(item, "VR180 video (still)", v)
    media = _asset_name(p, f"media.{ext}")
    block = {"media": media, "kind": "image", "coverage": 180, "layout": "sbs", "radius_m": SPHERE_RADIUS_M}
    _export_scene(
        run, item, mode="sphere", block=block, assets={media: p}, thumbnail=left, rep=rep.sub(0.97, 1.0)
    )
    rep(1.0, "Done")


def _mode_pano(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    import cv2

    from ..core.imaging import gpano_xmp

    spec: PanoJob = run.spec  # type: ignore[assignment]
    fill = _choice(spec.fill, FILLS, "fill style")
    ipd = _positive(spec.ipd_m, "Eye distance (IPD)", allow_zero=True)
    as_video = _positive(spec.as_video_s, "Video length", allow_zero=True)
    max_px = _max_process_px(run)
    try:
        want_w = int(spec.width or 0)
    except (TypeError, ValueError):
        raise InputError(f"360 width must be a whole number of pixels (got {spec.width!r}).") from None
    # A 360 photo is resampled straight to the output width (its depth is estimated on small
    # cube faces), so load it at up to that width instead of 'Max processing size'.
    photo = _load_photo(run, item, rep.sub(0.0, 0.04), max_px=max(max_px, want_w, PANO_WIDTH[run.quest]))
    rgb = photo.rgb
    is_360 = _is_equirect(rgb) if spec.input_is_360 is None else bool(spec.input_is_360)
    if not is_360 and max(rgb.shape[:2]) > max_px:
        rgb = limit_long(rgb, max_px)[0]  # a normal photo is processed at the usual size
    assets: dict[str, Path | bytes] = {}
    depth_block: dict[str, Any] = {}
    if is_360:
        from ..depth.pano import estimate_equirect

        if abs(rgb.shape[1] / rgb.shape[0] - 2.0) > 0.1:
            raise InputError(
                f"{item.path.name} is not a 360 photo (it should be twice as wide as high).",
                hint="Untick '360 input' to place a normal photo inside a 360 sphere.",
            )
        width = int(spec.width) if spec.width else min(rgb.shape[1], PANO_WIDTH[run.quest])
        width = max(256, width // 2 * 2)
        eq = _resize(rgb, width, width // 2)
        scale = _depth_scale(run)
        est = run.ctx.estimator(run.common.depth_model, run.common.device)
        drep = rep.sub(0.04, 0.45)
        drep(0.0, "Estimating 360 depth")
        res = run.ctx.depth(
            eq, estimator=est, source=item.path, extra="pano:518",
            compute=lambda e: estimate_equirect(eq, e, face=518, progress=drep, cancel=run.cancel),
        )  # fmt: skip
        run.check()
        _note_depth_fallback(run, est, res)
        run.model_notice(item, res.model_id)
        disp = np.asarray(res.disparity, dtype=np.float32)
        if disp.shape != eq.shape[:2]:
            disp = np.clip(_resize(disp, eq.shape[1], eq.shape[0]), 0, 1)
        disp = _refine_disparity(run, disp, eq)
        dist = 1.0 / scale.to_inv_z(disp)
        if spec.stereo and ipd > 0:
            from ..geometry.ods import ods_pair
            from ..geometry.stereo import compose

            inp = run.ctx.inpainter(run.common.inpaint_model, run.common.device)
            rep(0.45, "Rendering stereo 360 (two eye views)")
            left, right = _call_opt(
                ods_pair, eq, dist, ipd, inpainter=inp, progress=rep.sub(0.45, 0.88), cancel=run.cancel
            )
            run.check()
            _note_inpainter(run, item, inp)
            frame = compose(left, right, "tb")
            kind, label, layout, stereo_meta = "360_tb", "360 stereo photo (top-bottom)", "tb", "tb"
            thumb = left
        else:
            frame = eq
            kind, label, layout, stereo_meta = "360", "360 photo", "mono", "mono"
            thumb = eq
            dsmall = _resize(disp, DEPTH_SCENE_PX, DEPTH_SCENE_PX // 2)
            assets["depth.png"] = _png_bytes((np.clip(dsmall, 0, 1) * 255 + 0.5).astype(np.uint8))
            depth_block = {
                "depth": "depth.png",
                "depth_near_m": scale.near_m,
                "depth_far_m": scale.far_m,  # the viewer clamps to the sphere radius
            }
        eq_w, eq_h = eq.shape[1], eq.shape[0]
    else:
        from ..geometry.camera import estimate_intrinsics
        from ..geometry.projection import pano_from_photo

        K = estimate_intrinsics(rgb.shape, hfov_deg=run.common.hfov_deg, exif_hfov=photo.exif_hfov)
        width = int(spec.width) if spec.width else PANO_WIDTH[run.quest]
        width = max(256, width // 2 * 2)
        rep(0.1, "Placing the photo in a 360 sphere")
        if fill == "outpaint":
            black, valid = pano_from_photo(rgb, K, width, fill="black")
            mask = cv2.dilate((~valid).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            painted = _outpaint(run, item, black, mask, rep.sub(0.15, 0.85), wrap="yes")
            frame = painted if painted is not None else pano_from_photo(rgb, K, width, fill="blur")[0]
        else:
            frame, _valid = pano_from_photo(rgb, K, width, fill=fill)
        run.check()
        if spec.stereo:
            log.info("Stereo 360 needs a 360 input photo; writing a mono 360 for %s", item.path.name)
        kind, label, layout, stereo_meta = "360", "360 photo", "mono", "mono"
        thumb = rgb
        eq_w, eq_h = frame.shape[1], frame.shape[1] // 2
    rep(0.9, "Saving")
    p = run.out_path(item, kind, "jpg")
    _save_jpeg_or_png(p, frame, xmp=gpano_xmp(eq_w, eq_h, eq_w, eq_h, 0, 0))
    run.add_output(item, label, p)
    if as_video > 0:
        v = run.out_path(item, kind, "mp4")
        _write_still_video(run, frame, v, as_video, coverage=360, stereo=stereo_meta, rep=rep.sub(0.9, 0.97))
        run.add_output(item, label.replace("photo", "video (still)"), v)
    media = _asset_name(p, "media.jpg")
    assets[media] = p
    block = {
        "media": media,
        "kind": "image",
        "coverage": 360,
        "layout": layout,
        "radius_m": SPHERE_RADIUS_M,
        **depth_block,
    }
    _export_scene(
        run, item, mode="sphere", block=block, assets=assets, thumbnail=thumb, rep=rep.sub(0.97, 1.0)
    )
    rep(1.0, "Done")


def _png_bytes(img: np.ndarray) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, "PNG")
    return buf.getvalue()


def _segment(run: _Run, item: _Item, rgb: RGB, rep: ProgressReporter) -> FloatMap:
    from ..segment.matte import clean_matte

    seg = run.ctx.segmenter(run.common.segment_model, run.common.device)
    rep(0.0, "Finding the subject")
    alpha = _call_opt(seg.segment, rgb, progress=rep.sub(0.0, 0.9), cancel=run.cancel)
    run.check()
    requested = run.ctx.resolve_name("segment", run.common.segment_model).lower()
    used = str(getattr(seg, "resolved", None) or getattr(seg, "id", "") or "").lower()
    if requested == "auto" and used == "grabcut":
        run.warn(
            "The AI cut-out model could not be used; a simple method (GrabCut) found the subject, so the "
            "outline may be rough. Models download on first use - check your internet connection."
        )
    run.model_notice(item, used)
    alpha = clean_matte(np.asarray(alpha, dtype=np.float32))
    rep(1.0, "Subject found")
    return alpha


#: Smallest anchor scale used to size the look-behind layers of a 3D scene (a photo whose
#: centre is sky or far horizon); below it the inpainted band is at its widest anyway.
MIN_ANCHOR_SCALE = 0.1


def _scene_anchor_scale(depth: _Depth, anchor_m: float, alpha: FloatMap | None) -> float:
    """The scale :func:`~vrphoto.geometry.mesh.layered_meshes` will apply for ``anchor_m``
    (median depth of the central window of the front layer -> ``anchor_m``)."""
    try:
        from ..geometry.mesh import anchor_scale
    except ImportError:  # pragma: no cover - older geometry module
        return 1.0
    try:
        s = float(anchor_scale(depth.inv_z, depth.K, anchor_m, alpha=alpha))
    except VRPhotoError as e:  # no valid depth in the window: layered_meshes reports it
        log.info("Cannot compute the scene scale up front: %s", e)
        return 1.0
    return s if math.isfinite(s) and s > MIN_ANCHOR_SCALE else MIN_ANCHOR_SCALE


def _mode_scene(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    from ..geometry.gltf import write_glb, write_obj
    from ..geometry.mesh import layered_meshes

    spec: SceneJob = run.spec  # type: ignore[assignment]
    look = _choice(spec.look_behind, SCENE_LOOK_BEHIND, "look-behind setting")
    anchor = _positive(spec.anchor_distance_m, "Distance to the photo")
    max_b = _positive(spec.max_baseline_m, "Movement range", allow_zero=True)
    if int(spec.target_faces) < 1000:
        raise InputError(f"Mesh detail must be at least 1000 triangles (got {spec.target_faces}).")

    photo = _load_photo(run, item, rep.sub(0.0, 0.04))
    d = _estimate_depth(run, item, photo, rep.sub(0.04, 0.35))
    alpha = None
    if look == "cleanplate":
        alpha = _segment(run, item, photo.rgb, rep.sub(0.35, 0.45))
        if float((alpha > 0.5).mean()) < 0.002:
            run.warn("No subject was found for the clean background; used normal look-behind instead.")
            look, alpha = "band", None
    # The meshes are scaled by s about the eye so the photo's centre is 'anchor' metres away:
    # a real head movement m is then a movement of m / s in the photo's own depth units, so
    # that is what the look-behind layers must be prepared for.
    s = _scene_anchor_scale(d, anchor, alpha if look == "cleanplate" else None)
    log.info(
        "Scene scale %.3f: look-behind prepared for %.2f m of head movement (%.2f m in photo units)",
        s, max_b, max_b / s,
    )  # fmt: skip
    li = _build_layers(
        run, item, photo, d, rep.sub(0.45, 0.65), max_baseline=max_b / s, look_behind=look, alpha=alpha
    )
    rep(0.65, "Building 3D meshes")
    meshes = _call_opt(
        layered_meshes,
        li,
        target_faces=int(spec.target_faces),
        anchor_distance_m=anchor,
        eye_height_m=EYE_HEIGHT_M,
        progress=rep.sub(0.65, 0.85),
        cancel=run.cancel,
    )
    run.check()
    rep(0.86, "Saving the 3D model")
    keep_glb = bool(spec.export_glb) or not (run.common.export_scene or spec.export_obj)
    glb = run.out_path(item, "scene", "glb") if keep_glb else run.work_dir(item) / "scene.glb"
    _call_opt(write_glb, meshes, glb, cancel=run.cancel)
    if keep_glb:
        run.add_output(item, "3D scene (GLB)", glb)
    if spec.export_obj:
        o = run.out_path(item, "scene", "obj")
        write_obj(meshes, o)
        run.add_output(item, "3D scene (OBJ)", o)
    block = {"meshes": ["scene.glb"], "anchor_distance_m": anchor, "eye_height_m": EYE_HEIGHT_M}
    _export_scene(
        run,
        item,
        mode="photo3d",
        block=block,
        assets={"scene.glb": glb},
        thumbnail=photo.rgb,
        rep=rep.sub(0.95, 1.0),
    )
    rep(1.0, "Done")


def _cutout_rgba(rgb: RGB, alpha: FloatMap, margin: float = 0.08) -> np.ndarray:
    """RGBA crop around the subject with a margin (transparent background)."""
    ys, xs = np.nonzero(alpha > 0.5)
    h, w = alpha.shape
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    m = int(round(margin * max(y1 - y0, x1 - x0)))
    y0, y1, x0, x1 = max(0, y0 - m), min(h, y1 + m), max(0, x0 - m), min(w, x1 + m)
    a = (np.clip(alpha[y0:y1, x0:x1], 0, 1) * 255 + 0.5).astype(np.uint8)
    return np.dstack([rgb[y0:y1, x0:x1], a])


#: Viewer environment of object scenes: shown in passthrough AR on the Quest.
OBJECT_ENV: dict[str, str] = {"prefer": "ar", "background": "passthrough"}


def _save_cutout(run: _Run, item: _Item, photo: _Photo, alpha: FloatMap) -> Path:
    """Save the subject cut-out (``<base>_cutout.png``) as an output; returns its path."""
    from ..core.imaging import save_image

    cut = run.out_path(item, "cutout", "png")
    save_image(cut, _cutout_rgba(photo.rgb, alpha))
    run.add_output(item, "Cut-out (PNG with transparency)", cut)
    return cut


def _sharp_inputs(run: _Run, item: _Item, photo: _Photo) -> tuple[Path, dict[str, Any]]:
    """SHARP's input image and parameters: the whole photo (SHARP models the scene, the
    subject is picked out afterwards) and its horizontal field of view."""
    from ..core.imaging import save_image
    from ..geometry.camera import estimate_intrinsics

    K = estimate_intrinsics(photo.rgb.shape, hfov_deg=run.common.hfov_deg, exif_hfov=photo.exif_hfov)
    params: dict[str, Any] = {"hfov_deg": round(K.hfov_deg, 3), "name": "model"}
    image_in = run.work_dir(item) / "sharp_input.png"
    save_image(image_in, photo.rgb)
    return image_in, params


def _mode_object(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    """Walk-around person / object.

    The subject comes from the photo's own transparency when it has one (an already cut-out
    PNG / HEIC), else from the segmenter.  ``billboard`` inflates it locally (GLB + optional
    splat); the add-on methods get the RGBA cut-out (SHARP: the whole photo, its Gaussians are
    then reduced to the subject) and their GLB is rescaled to ``height_m`` losslessly.
    ``sharp360`` continues in :func:`vrphoto.pipeline.sharp360.mode_sharp360` once the
    subject is found.
    """
    spec: ObjectJob = run.spec  # type: ignore[assignment]
    method = _choice(spec.method, OBJECT_METHODS, "3D method")
    placement = _choice(spec.placement, PLACEMENTS, "placement")
    height = _positive(spec.height_m, "Height")
    back = _choice(spec.back, BACKS, "back texture")
    thickness = _positive(spec.thickness, "Thickness", allow_zero=True)
    s360_opts = None
    if method == "sharp360":
        from . import sharp360

        s360_opts = sharp360.resolve_options(spec, installed=_addon_installed)  # bad options: before work
    manifest = None
    if method in ADDON_METHODS:
        manifest = _require_addon(ADDON_METHODS[method][0])  # fail fast, before any work
    photo = _load_photo(run, item, rep.sub(0.0, 0.04), with_alpha=True)
    found = 0.10 if method == "sharp360" else 0.25  # SHARP 360 needs the time for the back
    if photo.alpha is not None:
        from ..segment.matte import clean_matte

        # an already cut-out photo (PNG/HEIC with transparency): use its own outline
        log.info("%s has transparency; using it as the cut-out", item.path.name)
        alpha = clean_matte(photo.alpha)
        rep(found, "Using the photo's transparent background as the cut-out")
    else:
        alpha = _segment(run, item, photo.rgb, rep.sub(0.04, found))
    if float((alpha > 0.5).mean()) < 0.002:
        raise InputError(
            f"No person or object could be found in {item.path.name}.",
            hint="Use a photo where the subject stands out clearly from the background, or try another cut-out model.",
        )
    if s360_opts is not None:
        sharp360.mode_sharp360(
            run, item, photo, alpha, spec, placement=placement, height=height, rep=rep, opts=s360_opts
        )
        return
    assets: dict[str, Path | bytes] = {}
    env = dict(OBJECT_ENV)
    if method == "billboard":
        from ..geometry.billboard import billboard
        from ..geometry.gltf import write_glb

        d = _estimate_depth(run, item, photo, rep.sub(0.25, 0.55))
        rep(0.55, "Building the 3D cut-out")
        meshes = _call_opt(
            billboard,
            photo.rgb,
            alpha,
            d.result(),
            d.K,
            height_m=height,
            thickness=thickness,
            back=back,
            progress=rep.sub(0.55, 0.8),
            cancel=run.cancel,
        )
        run.check()
        glb = run.out_path(item, "object", "glb")
        _call_opt(write_glb, meshes, glb, cancel=run.cancel)
        run.add_output(item, "3D object (GLB)", glb)
        assets["object.glb"] = glb
        block: dict[str, Any] = {
            "asset": "object.glb",
            "format": "glb",
            "height_m": height,
            "pivot": "base",
            "placement": placement,
        }
        if spec.export_splat:
            from ..geometry.splat import splats_from_mesh, write_splats

            rep(0.85, "Making the splat version")
            sc = _concat_splats([splats_from_mesh(m) for m in meshes])
            if len(sc) > MAX_OBJECT_SPLATS:
                from ..geometry.splat import decimate_splats

                sc = decimate_splats(sc, max_n=MAX_OBJECT_SPLATS)
            sp = run.out_path(item, "object", "splat")
            write_splats(sc, sp)
            run.add_output(item, "3D object (Gaussian splat)", sp)
            assets["object.splat"] = sp
            block["alt_assets"] = ["object.splat"]
        _export_scene(
            run,
            item,
            mode="object",
            block=block,
            assets=assets,
            thumbnail=photo.rgb,
            env=env,
            rep=rep.sub(0.95, 1.0),
        )
        rep(1.0, "Done")
        return

    addon_id, task = ADDON_METHODS[method]
    run.addon_notice(item, manifest)
    cut = _save_cutout(run, item, photo, alpha)
    params: dict[str, Any] = {"seed": int(spec.seed), "name": "model"}
    if method == "hunyuan":
        params["texture"] = bool(getattr(spec, "hunyuan_texture", False))  # paint a texture (slower)
    image_in = cut
    if method == "sharp":
        if int(spec.seed):
            run.warn(f"{manifest.title} always gives the same result; the 'Variation' number is not used.")
        image_in, params = _sharp_inputs(run, item, photo)
    rep(0.3, f"Running {manifest.title}")
    outs = _run_addon(run, item, addon_id, task, {"image": str(image_in)}, params, rep.sub(0.3, 0.9))
    title = manifest.title
    _addon_warnings(run, title, outs)
    glb_src = outs.get("glb")
    ply_src = outs.get("ply") or outs.get("splat")
    if glb_src and Path(glb_src).is_file():
        dst = run.out_path(item, f"_{method}", "glb")
        if not fit_glb_height(Path(glb_src), dst, height):
            _copy_file(glb_src, dst)
        run.add_output(item, f"3D object (GLB, {title})", dst)
        assets["object.glb"] = dst
        block = {
            "asset": "object.glb",
            "format": "glb",
            "height_m": height,
            "pivot": "base",
            "placement": placement,
        }
        if ply_src and Path(ply_src).is_file():
            extra = run.out_path(item, f"_{method}", Path(ply_src).suffix.lstrip(".") or "ply")
            _copy_file(ply_src, extra)
            run.add_output(item, f"3D Gaussians ({Path(ply_src).suffix.lstrip('.').upper()}, {title})", extra)
    elif ply_src and Path(ply_src).is_file():
        from ..geometry.splat import (
            crop_splats,
            decimate_splats,
            normalize_orientation,
            read_splats,
            write_splats,
        )

        rep(0.9, "Converting the splat")
        sc = read_splats(ply_src)
        if str(outs.get("frame", "")).lower() == "opencv":
            if method == "sharp":  # SHARP models the whole photo: keep only the subject
                sc = _subject_splats(run, sc, alpha, outs, params)
            sc = _splats_cv_to_gltf(sc)
        sc = crop_splats(sc, radius_pct=98.0)
        sc = normalize_orientation(sc)
        sc = decimate_splats(sc, max_n=MAX_OBJECT_SPLATS)
        dst = run.out_path(item, f"_{method}", "splat")
        write_splats(sc, dst)
        run.add_output(item, f"3D object (Gaussian splat, {title})", dst)
        assets["object.splat"] = dst
        block = {
            "asset": "object.splat",
            "format": "splat",
            "height_m": height,
            "pivot": "base",
            "placement": placement,
        }
    else:
        raise AddonError(
            f"{title} finished but did not return a 3D model.",
            hint=f"Details are in the log ({UI_LOG}). Try again, or reinstall the add-on in the {UI_ADDONS}.",
        )
    _export_scene(
        run,
        item,
        mode="object",
        block=block,
        assets=assets,
        thumbnail=photo.rgb,
        env=env,
        rep=rep.sub(0.95, 1.0),
    )
    rep(1.0, "Done")


def _addon_warnings(run: _Run, title: str, outs: dict[str, Any]) -> None:
    """Pass an add-on worker's ``warning`` / ``warnings`` outputs on to the job result."""
    msgs: list[Any] = []
    for key in ("warning", "warnings"):
        v = outs.get(key)
        if isinstance(v, str):
            msgs.append(v)
        elif isinstance(v, list | tuple):
            msgs.extend(v)
    for m in msgs:
        text = str(m).strip()
        if text:
            run.warn(f"{title}: {text}")


def _subject_splats(run: _Run, sc: Any, alpha: FloatMap, outs: dict[str, Any], params: dict[str, Any]) -> Any:
    """Keep the Gaussians of a whole-photo reconstruction (OpenCV camera frame) whose centres
    project onto the subject's cut-out (slightly dilated) and are not far behind the subject,
    i.e. drop the background.

    Falls back to the full reconstruction (with a warning) when hardly anything is left,
    e.g. if the add-on used another camera model than expected.
    """
    import cv2

    h, w = alpha.shape
    try:
        out_w = float(outs.get("width") or w)
        out_h = float(outs.get("height") or h)
        f = float(outs.get("f_px") or 0.0)
    except (TypeError, ValueError):
        out_w, out_h, f = float(w), float(h), 0.0
    if not (f > 0 and math.isfinite(f)):
        hf = float(params.get("hfov_deg") or 65.0)
        f = (out_w / 2.0) / math.tan(math.radians(hf) / 2.0)
    P = np.asarray(sc.means, dtype=np.float64)
    z = P[:, 2]
    front = np.isfinite(P).all(axis=1) & (z > 1e-3)
    zs = np.where(front, z, 1.0)
    # pixel coordinates in the add-on's image, then in the matte's resolution
    u = (f * P[:, 0] / zs + out_w / 2.0) * (w / out_w)
    v = (f * P[:, 1] / zs + out_h / 2.0) * (h / out_h)
    ui = np.clip(np.round(u - 0.5), 0, w - 1).astype(np.int64)  # SHARP's centre is at W/2
    vi = np.clip(np.round(v - 0.5), 0, h - 1).astype(np.int64)
    inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    r = max(2, int(round(0.01 * max(h, w))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    matte = cv2.dilate((alpha > 0.5).astype(np.uint8), k) > 0
    keep = inside & matte[vi, ui]
    if int(keep.sum()) >= 50:
        # drop what lies far behind the subject inside its outline (hidden background the
        # add-on reconstructed): the visible front of the subject dominates the near depths,
        # and a person / object is rarely deeper than half its distance from the camera
        p5, p25 = (float(v) for v in np.percentile(z[keep], [5.0, 25.0]))
        keep &= z <= 1.5 * p25 + 2.0 * (p25 - p5)
    n_keep = int(keep.sum())
    if n_keep < max(500, int(0.01 * len(P))):
        run.warn(
            "The subject could not be separated from the background in the add-on's 3D result, so the "
            "whole scene was kept."
        )
        return sc
    log.info("Kept %d of %d Gaussians on the subject", n_keep, len(P))
    from ..core.types import SplatCloud

    return SplatCloud(
        means=np.asarray(sc.means)[keep],
        scales=np.asarray(sc.scales)[keep],
        quats=np.asarray(sc.quats)[keep],
        colors=np.asarray(sc.colors)[keep],
        opacity=np.asarray(sc.opacity)[keep],
        sh_rest=None if getattr(sc, "sh_rest", None) is None else np.asarray(sc.sh_rest)[keep],
    )


def _first_frame(path: Path) -> RGB | None:
    try:
        if is_image(path):
            return limit_long(load_image(path), 1024)[0]
        if is_video(path):
            from ..video.decode import iter_frames

            it = iter_frames(path, max_long=1024)
            try:
                return next(iter(it))
            finally:
                close = getattr(it, "close", None)
                if close is not None:
                    close()
        if path.is_dir():
            for p in sorted(path.iterdir()):
                if is_image(p):
                    return limit_long(load_image(p), 1024)[0]
    except (VRPhotoError, OSError, StopIteration) as e:
        log.debug("No thumbnail from %s: %s", path, e)
    return None


def _keep_capture_log(run: _Run, item: _Item, work: Path) -> Path | None:
    """Copy a failed capture's log from the work folder to the output folder."""
    try:
        logs = sorted(work.glob("*capture.log"), key=lambda q: q.stat().st_mtime)
    except OSError:
        return None
    if not logs:
        return None
    dst = run.out_path(item, "_capture_log", "txt")
    try:
        shutil.copyfile(logs[-1], dst)
    except OSError as e:
        log.info("Could not keep the capture log %s: %s", logs[-1], e)
        return None
    return dst


def _capture_scale(splat: Path) -> dict[str, Any] | None:
    """The metric scale record ``<stem>.capture.json`` written next to a captured splat.

    Returns ``{"scale", "height_m", "method"}`` (``height_m``: the subject's height when it
    was given, else the splat's height, in metres) or None when there is no usable record.
    """
    p = splat.with_name(f"{splat.stem}.capture.json")
    try:
        info = json.loads(p.read_text(encoding="utf-8"))
        scale = float(info["scale"])
        method = str(info.get("method") or "none")
        # the subject height the user typed (when given) else the splat bounding-box height;
        # with scale 1.0 the viewer shows the splat 1:1 and height_m only describes it
        height = float(info.get("subject_height_m") or info.get("height_m") or 0.0)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as e:
        log.info("Ignoring the capture scale record %s: %s", p, e)
        return None
    if not (math.isfinite(scale) and scale > 0):
        return None
    if not (math.isfinite(height) and height > 0):
        height = 0.0
    log.info("Capture scale x%.4g (%s): %.2f m tall", scale, method, height)
    return {"scale": scale, "height_m": round(height, 4), "method": method}


def _copy_file(src: Path | str, dst: Path) -> None:
    """Copy a file to its output name through a temp file: ``dst`` is never half-written
    (disk full), and the final rename waits out a brief Windows file lock."""
    tmp = temp_sibling(dst)
    try:
        shutil.copyfile(src, tmp)
        replace_with_retry(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _move_file(src: Path, dst: Path) -> None:
    """Move a finished file from the work folder to its output name.

    A rename when both are on one drive (retried while antivirus or the Windows search
    indexer briefly holds the new file), else :func:`_copy_file`.
    """
    try:
        replace_with_retry(src, dst)
        return
    except OSError as e:  # another drive (or the file stays locked): copy it
        log.debug("Cannot rename %s to %s (%s); copying it", src, dst, e)
    _copy_file(src, dst)
    try:
        os.unlink(src)
    except OSError:  # the work folder is deleted at the end of the job anyway
        pass


def _mode_capture(run: _Run, items: list[_Item], rep: ProgressReporter) -> None:
    from ..addons.capture import capture_to_splat

    spec: CaptureJob = run.spec  # type: ignore[assignment]
    item = items[0]
    run.multi = False  # one result for all inputs
    if run.common.name:
        item.base = safe_base(run.common.name.strip())
    else:
        first = item.path
        item.base = safe_base(
            first.name if first.is_dir() else (first.parent.name if len(items) > 1 else first.stem)
        )
    subject_h = _positive(getattr(spec, "subject_height_m", 0.0) or 0.0, "Subject height", allow_zero=True)
    paths = [it.path for it in items]
    work = run.work_dir(item) / "capture"
    work.mkdir(parents=True, exist_ok=True)
    _free_app_gpu()  # COLMAP's GPU SIFT and Brush do not go through _run_addon
    try:
        splat = _call_opt(
            capture_to_splat, paths, work,
            video_fps=float(spec.video_fps), max_images=int(spec.max_images), steps=int(spec.steps),
            max_splats=int(spec.max_splats), crop_pct=float(spec.crop_pct), name="capture",
            subject_height_m=subject_h, progress=rep.sub(0.0, 0.95), cancel=run.cancel,
        )  # fmt: skip
    except AddonNotInstalled as e:
        raise AddonNotInstalled(str(e), hint=e.hint or f"Install it in the {UI_ADDONS}.") from e
    except Cancelled:
        raise
    except VRPhotoError as e:
        # the reconstruction log is written into the job's work folder, which is deleted
        # at the end of the job: keep it next to the outputs so the user can look at it
        kept = _keep_capture_log(run, item, work)
        if kept is None:
            raise
        hint = f"{e.hint + ' ' if e.hint else ''}The reconstruction log was saved as {kept}."
        try:
            err: VRPhotoError = type(e)(str(e), hint=hint)
        except TypeError:
            err = VRPhotoError(str(e), hint=hint)
        raise err from e
    run.check()
    splat = Path(splat)
    size = _capture_scale(splat)
    if size is None or size["method"] == "none":
        msg = "The real-world size of the captured scene could not be worked out"
        if spec.as_object:
            msg += "; in the viewer, resize it with both grip buttons if it looks too big or too small."
        else:  # a walk-in splat scene cannot be resized in the viewer
            msg += ", so it may look too big or too small in the viewer."
        run.warn(msg + ("" if subject_h else " Or set 'Subject height' and run it again."))
    dst = run.out_path(item, "capture", "splat")
    _move_file(splat, dst)
    run.add_output(item, "Captured 3D scene (Gaussian splat)", dst)
    ply = splat.with_suffix(".ply")
    if ply.is_file():
        dply = run.out_path(item, "capture", "ply")
        _move_file(ply, dply)
        run.add_output(item, "Captured 3D scene (PLY)", dply)
    media = _asset_name(dst, "capture.splat")
    thumb = _first_frame(item.path)
    if spec.as_object:
        block = {"asset": media, "format": "splat", "pivot": "base", "placement": "hit-test"}
        if size is not None and size["method"] != "none":
            # already real size: show it 1:1 ('height_m' then names the subject's height)
            block["scale"] = 1.0
            if size["height_m"] > 0:
                block["height_m"] = size["height_m"]
        _export_scene(
            run,
            item,
            mode="object",
            block=block,
            assets={media: dst},
            thumbnail=thumb,
            env={"prefer": "ar", "background": "passthrough"},
            rep=rep.sub(0.95, 1.0),
        )
    else:
        block = {"asset": media, "format": "splat", "up": "y", "scale": 1.0}
        _export_scene(
            run, item, mode="splat", block=block, assets={media: dst}, thumbnail=thumb, rep=rep.sub(0.95, 1.0)
        )
    rep(1.0, "Done")


def _mode_video(run: _Run, item: _Item, rep: ProgressReporter) -> None:
    """Normal video -> 3D (flat stereo layouts) or VR180 video, frame by frame.

    Frames are decoded at a constant rate (at most :data:`MAX_VIDEO_FPS`) so the copied sound
    stays in sync; depth is smoothed over time (``temporal_smooth``) and so is the flat-mode
    screen distance, so the 3D effect does not "breathe".  At a cut (a new shot whose depth
    has nothing to do with the previous frame) both start over from the new frame.
    """
    from ..depth.post import scene_cut, temporal_align
    from ..geometry.camera import estimate_intrinsics
    from ..geometry.layers import build_layers
    from ..geometry.projection import vr180_frame
    from ..geometry.stereo import auto_convergence, compose, stereo_pair
    from ..video.decode import iter_frames, probe

    spec: VideoJob = run.spec  # type: ignore[assignment]
    output = _choice(spec.output, VIDEO_OUTPUTS, "video type")
    ipd = _positive(spec.ipd_m, "Eye distance (IPD)")
    strength = _positive(spec.strength, "3D strength", allow_zero=True)
    ema = min(max(float(spec.temporal_smooth), 0.0), 0.95)
    preset = _video_preset(run, spec.encoder_preset, output=output)
    if not is_video(item.path):
        raise InputError(
            f"{item.path.name} is not a video.", hint="Use '3D photo' or another photo mode for images."
        )
    scale = _depth_scale(run)
    rep(0.0, f"Reading {item.path.name}")
    info = probe(item.path)
    start = max(0.0, float(spec.start_s or 0.0))
    end = float(spec.end_s) if spec.end_s else None
    if end is not None and end <= start:
        raise InputError(f"The end time ({end} s) must be after the start time ({start} s).")
    if info.duration > 0 and start >= info.duration:
        raise InputError(
            f"The start time ({start:g} s) is after the end of {item.path.name} ({info.duration:.1f} s long).",
            hint="Choose a start time within the video.",
        )
    span = (min(end, info.duration) if end else info.duration) - start if info.duration > 0 else 0.0
    # Decode at a constant frame rate: phone videos often have a variable one, and the output
    # is written at a fixed rate, so passing frames through would drift from the sound.
    # High-speed clips (120/240 fps) are converted at MAX_VIDEO_FPS: 3D needs a depth map per
    # frame, and headsets cannot play 4K stereo at those rates anyway.
    src_fps = float(info.fps or 0.0)
    out_fps = min(src_fps, MAX_VIDEO_FPS) if src_fps > 0 and math.isfinite(src_fps) else 30.0
    if src_fps > MAX_VIDEO_FPS:
        log.info("%s has %.0f fps; converting at %.0f fps", item.path.name, src_fps, out_fps)
    out_fps = _exact_fps(out_fps)  # 23.98 -> 24000/1001: decoder and encoder use the same rate
    cfr = out_fps
    n_est = (
        max(1, int(round(span * out_fps)))
        if span > 0
        else max(1, int(round(info.n_frames * out_fps / max(src_fps, out_fps))))
    )
    frames = _call_opt(
        iter_frames, item.path, max_long=int(spec.max_long), start=start, end=end, fps=cfr, cancel=run.cancel
    )
    est = run.ctx.estimator(run.common.depth_model, run.common.device)
    keep_audio = bool(spec.keep_audio) and info.has_audio
    eye = 0  # set from the first frame
    kind = {"sbs": "sbs", "half_sbs": "half_sbs", "tb": "tb", "half_tb": "half_tb", "vr180": "vr180"}[output]
    out = run.out_path(item, kind, "mp4")
    frep = rep.sub(0.02, 0.97)
    vw = None
    thumb: RGB | None = None
    K: Intrinsics | None = None
    prev: FloatMap | None = None
    inv_c: float | None = None
    first_res: DepthResult | None = None
    i = 0
    try:
        for i, frame in enumerate(frames):  # noqa: B007 - i used after the loop
            run.check()
            if K is None:
                K = estimate_intrinsics(frame.shape, hfov_deg=run.common.hfov_deg)
                eye = _video_eye_px(run, K)
            res = _call_opt(est.estimate, frame, cancel=run.cancel)
            if first_res is None:
                first_res = res
                _note_depth_fallback(run, est, res)
                run.model_notice(item, res.model_id)
            d = np.asarray(res.disparity, dtype=np.float32)
            if d.shape != frame.shape[:2]:
                d = np.clip(_resize(d, frame.shape[1], frame.shape[0]), 0, 1)
            if ema > 0:
                if prev is not None and scene_cut(prev, d):
                    log.debug("Scene cut at frame %d: depth and screen plane start over", i)
                    prev, inv_c = None, None
                d = temporal_align(prev, d, ema=ema)
            prev = d
            d = _refine_disparity(run, d, frame)
            inv = scale.to_inv_z(d)
            li = build_layers(
                frame, inv, K, max_baseline_m=ipd * max(strength, 1.0), inpainter=None, mode="none"
            )
            if output == "vr180":
                eye_K = _vr180_render_K(K, eye)
                left, right = _call_opt(
                    stereo_pair, li, StereoParams(ipd_m=ipd, mode="vr"), eye_K, cancel=run.cancel
                )
                out_frame = vr180_frame(left, right, eye_K, eye, fill="blur", layout="sbs")
            else:
                zc = float(auto_convergence(inv))
                c = 1.0 / zc if zc > 0 and math.isfinite(zc) else 0.0
                inv_c = c if inv_c is None else (ema * inv_c + (1.0 - ema) * c)  # steady screen plane
                params = _flat_stereo(
                    _aspect(K),
                    ipd_m=ipd,
                    strength=strength,
                    convergence_m=(1.0 / inv_c) if inv_c and inv_c > 0 else None,
                    max_parallax_pct=float(spec.max_parallax_pct),
                )
                left, right = _call_opt(stereo_pair, li, params, cancel=run.cancel)
                out_frame = compose(left, right, output)
            if vw is None:
                thumb = left
                vw = _video_writer(
                    run,
                    out,
                    out_frame.shape[1],
                    out_frame.shape[0],
                    out_fps,
                    preset,
                    audio_from=item.path if keep_audio else None,
                    audio_start_s=start,
                )
            vw.write(out_frame)
            frep(min(1.0, (i + 1) / n_est), f"Converting frame {i + 1}/{max(n_est, i + 1)}")
        if vw is None:
            raise InputError(f"No video frames were found in {item.path.name} for the selected time range.")
        rep(0.97, "Finishing the video")
        out = _close_video(run, vw)
    except BaseException:
        if vw is not None:
            vw.abort()
        raise
    finally:
        close = getattr(frames, "close", None)
        if close is not None:
            close()
    log.info("Converted %d frames of %s", i + 1, item.path.name)
    labels = {
        "sbs": "3D video (side-by-side)",
        "half_sbs": "3D video (half side-by-side)",
        "tb": "3D video (top-bottom)",
        "half_tb": "3D video (half top-bottom)",
        "vr180": "VR180 video",
    }
    if output == "vr180":
        from ..video.spatial import inject

        inject(out, coverage=180, stereo="lr")
    run.add_output(item, labels[output], out)
    media = _asset_name(out, "media.mp4")
    if output == "vr180":
        block = {
            "media": media,
            "kind": "video",
            "coverage": 180,
            "layout": "sbs",
            "radius_m": SPHERE_RADIUS_M,
        }
        _export_scene(run, item, mode="sphere", block=block, assets={media: out}, thumbnail=thumb)
    else:
        block = _stereo_scene_block(
            media, "video", output, thumb if thumb is not None else np.zeros((9, 16, 3))
        )
        _export_scene(run, item, mode="stereo", block=block, assets={media: out}, thumbnail=thumb)
    rep(1.0, "Done")


_HANDLERS: dict[str, Callable[[_Run, _Item, ProgressReporter], None]] = {
    "depth": _mode_depth,
    "stereo": _mode_stereo,
    "parallax": _mode_parallax,
    "vr180": _mode_vr180,
    "pano": _mode_pano,
    "scene": _mode_scene,
    "object": _mode_object,
    "video": _mode_video,
}
#: Every job kind :func:`run_job` understands.
SUPPORTED_KINDS: tuple[str, ...] = (*_HANDLERS, "capture")


# --------------------------------------------------------------------------- entry point
def _validate_inputs(spec: JobBase) -> list[Path]:
    raw = list(spec.inputs or [])
    if not raw:
        raise InputError("No input files were given.", hint="Add one or more photos (or a video) first.")
    paths: list[Path] = []
    for r in raw:
        p = Path(str(r)).expanduser()
        if spec.kind == "capture" and p.is_dir():
            paths.append(p)
            continue
        if p.is_dir():
            raise InputError(
                f"{p.name} is a folder, not a file.",
                hint="Choose the photo or video files inside it (folders are only used by 'Capture to 3D').",
            )
        if not p.is_file():
            raise InputError(f"File not found: {p}", hint="It may have been moved, renamed or deleted.")
        paths.append(p)
    return paths


def _validate_common(c: CommonOptions) -> None:
    """Check common options before any slow work starts."""
    if c.hfov_deg is not None:
        try:
            h = float(c.hfov_deg)
        except (TypeError, ValueError):
            h = float("nan")
        if not (math.isfinite(h) and 1.0 <= h <= 179.0):
            raise InputError(
                f"Field of view {c.hfov_deg} deg is not valid; use a value between 1 and 179 degrees, "
                "or leave it on 'Auto'."
            )
    try:
        px = int(c.max_process_px or 0)
    except (TypeError, ValueError):
        px = -1
    if px and px < 256:
        raise InputError(f"Max processing size must be at least 256 px (got {c.max_process_px}).")
    for name in ("near_m", "far_m"):
        _positive(getattr(c, name), "Nearest distance" if name == "near_m" else "Farthest distance")
    if float(c.far_m) <= float(c.near_m):
        raise InputError(
            f"The farthest distance ({c.far_m} m) must be larger than the nearest distance ({c.near_m} m)."
        )
    if c.quest_model and str(c.quest_model).strip().lower() not in QUEST_MODELS:
        raise InputError(
            f"Unknown headset {c.quest_model!r}.", hint=f"Choose one of: {', '.join(QUEST_MODELS)}."
        )


def _friendly(e: BaseException) -> VRPhotoError | None:
    """Translate low-level failures into friendly errors (None = leave as is)."""
    if isinstance(e, MemoryError):
        return VRPhotoError(
            "Not enough memory to process this file.",
            hint="Lower 'Max processing size' in the advanced options, or close other programs.",
        )
    text = str(e)
    if type(e).__name__ == "error" and type(e).__module__.startswith("cv2"):
        if "memory" in text.lower() or "allocate" in text.lower():
            return VRPhotoError(
                "Not enough memory to process this file.",
                hint="Lower 'Max processing size' in the advanced options.",
            )
    if isinstance(e, OSError):
        where = f" ({e.filename})" if getattr(e, "filename", None) else ""
        if e.errno == errno.ENOSPC:  # also Windows ERROR_DISK_FULL / ERROR_HANDLE_DISK_FULL
            return VRPhotoError(
                f"The disk is full: the result could not be saved{where}.",
                hint="Free some disk space or choose an output folder on another drive, then try again.",
            )
        if isinstance(e, PermissionError):
            return VRPhotoError(f"A file could not be saved: permission denied{where}.", hint=WRITE_HINT)
    return None


def run_job(
    spec: JobBase,
    ctx: RunContext | None = None,
    *,
    progress: ProgressReporter | Callable[[float, str], None] | None = None,
    cancel: CancelToken | None = None,
) -> JobResult:
    """Run one job and return its :class:`JobResult` (outputs, scene id, warnings, time).

    Raises :class:`InputError` for bad options/inputs, :class:`AddonNotInstalled` when the
    chosen method needs an add-on that is not installed, :class:`Cancelled` when
    ``cancel`` fires, or another :class:`VRPhotoError` subclass with a friendly message.
    """
    t0 = time.perf_counter()
    if not isinstance(spec, JobBase):
        raise InputError(f"Not a job specification: {type(spec).__name__}")
    kind = getattr(spec, "kind", None)
    if kind not in SUPPORTED_KINDS:
        raise InputError(f"Unknown job kind {kind!r}.", hint=f"Valid kinds: {', '.join(SUPPORTED_KINDS)}.")
    rep = _as_reporter(progress)
    cancel = cancel if cancel is not None else CancelToken()
    ctx = ctx if ctx is not None else RunContext()
    paths = _validate_inputs(spec)
    _validate_common(spec.common if spec.common is not None else CommonOptions())
    run = _Run(spec, ctx, cancel)
    cancel.check()
    log.info("Job %s started: %d input(s) -> %s", kind, len(paths), run.out_dir)
    name = (run.common.name or "").strip()
    items = []
    for i, p in enumerate(paths):
        if name:
            label = name if len(paths) == 1 else f"{name} {i + 1}"
            base = safe_base(name if len(paths) == 1 else f"{name}_{i + 1}")
        else:
            label, base = p.stem, safe_base(p.stem)
        prefix = f"[{i + 1}/{len(paths)}] " if len(paths) > 1 else ""
        items.append(_Item(index=i, path=p, base=base, prefix=prefix, label=label))
    try:
        if kind == "capture":
            _guarded(run, lambda: _mode_capture(run, items, rep))
        else:
            handler = _HANDLERS[kind]
            failures: list[tuple[_Item, BaseException]] = []
            n = len(items)
            for item in items:
                cancel.check()
                sub = _prefixed(rep.sub(item.index / n, (item.index + 1) / n), item.prefix)
                try:
                    _guarded(run, lambda item=item, sub=sub: handler(run, item, sub))
                except Cancelled:
                    raise
                except Exception as e:
                    if n == 1:
                        raise
                    failures.append((item, e))
                    log.error("%s failed", item.path.name, exc_info=not isinstance(e, VRPhotoError))
                    hint = getattr(e, "hint", None)
                    run.warn(f"{item.path.name} could not be processed: {e}" + (f" ({hint})" if hint else ""))
                finally:
                    _cleanup(item)
            if failures and len(failures) == n:
                first = failures[0][1]
                if isinstance(first, VRPhotoError):
                    text = f"None of the {n} files could be processed. First error: {first}"
                    try:
                        err: VRPhotoError = type(first)(text, hint=first.hint)
                    except TypeError:  # a subclass with another constructor
                        err = VRPhotoError(text, hint=first.hint)
                    raise err from first
                raise first
    finally:
        for item in items:
            _cleanup(item)
        release_reserved(run.reserved)  # written files now protect their names themselves
    run.result.elapsed_s = round(time.perf_counter() - t0, 3)
    rep(1.0, "Done")
    log.info(
        "Job %s finished in %.1f s: %d output(s), %d warning(s)",
        kind,
        run.result.elapsed_s,
        len(run.result.outputs),
        len(run.result.warnings),
    )
    return run.result


def _guarded(run: _Run, fn: Callable[[], None]) -> None:
    try:
        fn()
    except VRPhotoError:
        raise
    except Exception as e:
        if run.cancel.cancelled:
            raise Cancelled("Operation cancelled") from e
        friendly = _friendly(e)
        if friendly is not None:
            raise friendly from e
        raise


def _cleanup(item: _Item) -> None:
    if item.work is not None:
        shutil.rmtree(item.work, ignore_errors=True)
        item.work = None


__all__ = [
    "run_job",
    "SUPPORTED_KINDS",
    "ADDON_METHODS",
    "VR180_EYE_PX",
    "VIDEO_VR180_EYE_PX",
    "PANO_WIDTH",
    "fit_glb_height",
    "glb_bounds",
]
