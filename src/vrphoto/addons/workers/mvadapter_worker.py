"""MV-Adapter worker (experimental): one framed cut-out -> 6 orthographic views with cameras.

Runs inside the ``mvadapter`` add-on venv (Python 3.11, torch 2.8.0 + CUDA 12.8, diffusers)
with https://github.com/huanngzh/MV-Adapter extracted to ``<addon_dir>/src/mvadapter`` (not
pip-installed), Stable Diffusion 2.1 base at ``MVADAPTER_BASE`` and
``mvadapter_i2mv_sd21.safetensors`` in ``MVADAPTER_WEIGHTS``.

Task ``image_to_views``:
    inputs:  ``image`` - a square RGBA picture (512 x 512) already framed by the app: the
             subject's long side is about 90 % of the picture.  It is composited on 0.5 grey
             exactly like MV-Adapter's ``preprocess_image`` but **never re-cropped or re-scaled**,
             so an input pixel keeps its place in view 0 (design 3.4 step 6).
    params:  ``steps`` (30, 5-100), ``guidance`` (3.0, 1-10), ``seeds`` (list) or ``seed`` (0) -
             one set of views per seed, the model is loaded once; ``prompt`` (default
             :data:`DEFAULT_PROMPT`), ``negative_prompt``, ``offload`` ("auto" | "model" |
             "sequential" | "none"), ``name`` ("views").
    outputs: ``sets`` = ``[{"seed", "views": [6 PNG paths], "cameras": cameras.json path}]``,
             ``frame`` = "opencv", ``warnings`` (plain sentences), ``input`` (the grey-composited
             picture given to the model), ``offload`` / ``precision`` actually used.

Cameras (``scripts/inference_i2mv_sd.py`` at the pinned commit): orthographic, box +-0.55,
distance 1.8, elevation 0, azimuths ``[0, 45, 90, 180, 270, 315]`` handed to
``get_orthogonal_camera`` as ``azimuth - 90``.  :func:`mv_c2w_gl` is a line-by-line port of
its ``get_c2w``: MV-Adapter's world is z-up, the camera of azimuth ``a`` (as passed) sits at
``d (cos a, sin a, 0)`` and its c2w columns are ``[right, up, -lookat]`` (OpenGL / Blender
axes).  View 0 (``a = -90``) therefore sits at ``-y`` looking along ``+y`` with ``right = +x``
and ``up = +z``: the input picture's right is MV-Adapter's ``+x`` and its "into the picture"
is ``+y``.

``cameras.json`` re-expresses every camera in the **input view's OpenCV frame** W (x = right
in the input picture, y = down, z = away from the input camera; origin = the orbit centre,
which projects to the centre of every view): ``W = A M`` with ``A`` = :data:`MV_TO_CV_WORLD`
(x -> x, y -> -z, z -> y; a proper rotation), camera axes flipped ``diag(1, -1, -1)`` from
OpenGL to OpenCV.  View 0 becomes ``R = I`` at ``(0, 0, -1.8)``.

``phi_deg`` (the app's azimuth, CONTRACTS 13: ``forward(phi) = cos(phi) z - sin(phi) x``,
0 = the input view, 90 = "you walked to your right", 180 = behind) is read off that c2w: the
camera centre is ``-distance * forward = distance * (sin phi, 0, -cos phi)``, so
``phi = atan2(C_x, -C_z)`` (:func:`phi_elevation`).  For MV-Adapter's azimuth 90 the camera is
at ``a = 0`` -> ``M = (1.8, 0, 0)`` -> ``W = (1.8, 0, 0)``: on the ``+x`` side of the input
picture, looking along ``-x`` with its right = ``+z`` - the subject's left side, phi = 90.
Azimuth 180 -> ``W = (0, 0, 1.8)`` looking along ``-z`` (the back, mirrored), phi = 180.  The
phi values equal MV-Adapter's azimuths; the app still derives them from ``c2w``.

Projection (``pixel_center`` 0.5): a point P lands at ``u = (P - C) . R[:, 0] / m + 256``,
``v = (P - C) . R[:, 1] / m + 256`` with ``m = 2 * 0.55 / 512`` world units per pixel; pixel
``(i, j)`` spans ``u in [j, j + 1)``.  The world unit is arbitrary: the app scales it so view
0 matches its levelled input.

Runtime: fp16 UNet / text encoder / condition encoder (PyTorch runs the fp16 GEMMs with fp32
math on Pascal), fp32 VAE (its inputs and outputs are cast by :func:`cast_vae_io`), PyTorch
SDPA attention (no xFormers / flash attention on GTX 10-series), sliced VAE decoding, the
DDPM scheduler the README recommends for SD 2.1 wrapped in MV-Adapter's ShiftSNR schedule.
``offload="auto"`` uses model CPU offload below 7 GB free VRAM.  Out of memory -> free the
cache, reload with sequential CPU offload, then a plain error; NaN or black pictures -> one
retry with an fp32 UNet and text encoder (model offload below 12 GB free, sequential offload
below 10 GB: an 8 GB card goes straight to sequential).  Seeds use a CPU generator, so a seed
gives the same views with any offload mode.  Each set is written as soon as it is made, so the
sets finished before an error or a cancel stay on disk for the app's view cache.

The pure numpy helpers at the top are unit-tested in the app's test-suite
(``tests/addons/test_mvadapter_worker.py``).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from _common import (
    Progress,
    WorkerError,
    add_sys_path,
    addon_dir,
    cuda_info,
    free_cuda,
    log,
    param,
    require_input,
    run,
    src_dir,
)

VERSION = "1.0"
TASK = "image_to_views"

#: MV-Adapter's own view azimuths (degrees; 0 = the input view), in output order.
AZIMUTHS = (0, 45, 90, 180, 270, 315)
#: ``scripts/inference_i2mv_sd.py`` hands ``azimuth - 90`` to ``get_orthogonal_camera``.
AZIMUTH_OFFSET = -90.0
DISTANCE = 1.8
ORTHO_HALF = 0.55  # left = bottom = -0.55, right = top = 0.55
ORTHO_SCALE = 1.1  # passed to get_plucker_embeds_from_cameras_ortho (unused by it)
ELEVATION = 0.0
SIZE = 512
INPUT_FILL = 0.9  # preprocess_image: the subject's long side is 90 % of the picture
GREY = 0.5
PIXEL_CENTER = 0.5
SHIFT_SCALE = 8.0  # ShiftSNRScheduler.from_scheduler(shift_mode="interpolated", shift_scale=8.0)

DEFAULT_PROMPT = "a person, full body, high quality, realistic photo"
DEFAULT_NEGATIVE = "watermark, ugly, deformed, noisy, blurry, low contrast"  # the script's default
DEFAULT_STEPS = 30
STEPS_RANGE = (5, 100)
DEFAULT_GUIDANCE = 3.0
GUIDANCE_RANGE = (1.0, 10.0)
MAX_SEEDS = 8
MAX_SEED = 2**63 - 1  # torch.Generator.manual_seed range
MAX_PROMPT_CHARS = 500
OFFLOAD_MODES = ("auto", "model", "sequential", "none")
#: Placements from the least to the most memory-saving.
PLACEMENT_ORDER = ("none", "model", "sequential")
#: ``offload="auto"``: below this much free VRAM (GB) use model CPU offload (fp16 weights).
OFFLOAD_FREE_GB = 7.0
#: fp32 weights (the NaN / black retry): everything on the GPU from 12 GB free, model offload from
#: 10 GB, else sequential.  With model offload the fp32 UNet + adapter (about 4.2 GB), the VAE
#: (0.33 GB) and the reference pass's cached attention inputs (6 views x CFG, kept twice by the
#: pipeline: about 2.2 GB) are resident together - more than a GTX 1080's ~7 GB free.
OFFLOAD_FREE_GB_FP32 = 12.0
MODEL_OFFLOAD_FREE_GB_FP32 = 10.0
#: The subject's measured fill outside this range gets a warning (the app frames at 0.9).
FILL_WARN = (0.75, 0.97)

ADAPTER_FILE = "mvadapter_i2mv_sd21.safetensors"
#: Files of the SD 2.1 base folder the worker loads (fp16 UNet / text encoder, fp32 VAE).
BASE_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)
#: MV-Adapter source files the worker imports (relative to ``src/mvadapter``).  They only have
#: to exist: ``mvadapter/__init__.py`` is an empty file at the pinned commit.
SOURCE_FILES = (
    "mvadapter/__init__.py",
    "mvadapter/pipelines/pipeline_mvadapter_i2mv_sd.py",
    "mvadapter/schedulers/scheduling_shift_snr.py",
    "mvadapter/schedulers/scheduler_utils.py",  # imported by scheduling_shift_snr.py
    "mvadapter/loaders/__init__.py",  # the pipeline imports CustomAdapterMixin from the package
    "mvadapter/loaders/custom_adapter.py",
    "mvadapter/models/attention_processor.py",
    "mvadapter/utils/geometry.py",
)
REINSTALL_HINT = "Reinstall the MV-Adapter add-on from the Add-ons dialog."
OOM_HINT = ("Close other programs that use the graphics card (games, browsers, video players) and try "
            "again, or let 'SHARP 360' use TRELLIS or the built-in back.")

#: MV-Adapter world (z up, input camera at -y) -> the input view's OpenCV frame W.
MV_TO_CV_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
#: OpenGL camera axes (x right, y up, z backward) -> OpenCV (x right, y down, z forward).
GL_TO_CV_CAMERA = np.diag([1.0, -1.0, -1.0])

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


# --------------------------------------------------------------------------- cameras (pure)
def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def mv_c2w_gl(azimuth_deg: float | Sequence[float], elevation_deg: float | Sequence[float] = ELEVATION,
              distance: float | Sequence[float] = DISTANCE) -> np.ndarray:
    """Port of MV-Adapter's ``get_c2w``: camera-to-world matrices ``(N, 4, 4)`` (``(4, 4)`` for
    a scalar azimuth) in MV-Adapter's z-up world with OpenGL camera axes ``[right, up, -lookat]``.

    ``azimuth_deg`` is the value *passed* to ``get_orthogonal_camera`` (MV-Adapter's view
    azimuth plus :data:`AZIMUTH_OFFSET`)."""
    scalar = np.ndim(azimuth_deg) == 0
    az = np.radians(np.atleast_1d(np.asarray(azimuth_deg, dtype=np.float64)))
    el = np.radians(np.broadcast_to(np.asarray(elevation_deg, dtype=np.float64), az.shape))
    dist = np.broadcast_to(np.asarray(distance, dtype=np.float64), az.shape)
    if np.any(np.abs(el) >= math.radians(89.9)):
        raise ValueError("elevation must stay below +-89.9 degrees (the camera 'up' is world z)")
    pos = np.stack([dist * np.cos(el) * np.cos(az), dist * np.cos(el) * np.sin(az), dist * np.sin(el)], axis=-1)
    up0 = np.broadcast_to(np.array([0.0, 0.0, 1.0]), pos.shape)
    lookat = _normalize(-pos)
    right = _normalize(np.cross(lookat, up0))
    up = _normalize(np.cross(right, lookat))
    c2w = np.zeros(az.shape + (4, 4))
    c2w[:, :3, :3] = np.stack([right, up, -lookat], axis=-1)
    c2w[:, :3, 3] = pos
    c2w[:, 3, 3] = 1.0
    return c2w[0] if scalar else c2w


def c2w_opencv(azimuth_deg: float, elevation_deg: float = ELEVATION, distance: float = DISTANCE) -> np.ndarray:
    """Camera-to-world ``(4, 4)`` of MV-Adapter's view ``azimuth_deg`` (0 = the input view) in the
    input view's OpenCV frame W (x right, y down, z away from the input camera; camera axes
    x right, y down, z forward).  View 0 is ``R = I`` at ``(0, 0, -distance)``."""
    gl = mv_c2w_gl(float(azimuth_deg) + AZIMUTH_OFFSET, elevation_deg, distance)
    out = np.eye(4)
    out[:3, :3] = MV_TO_CV_WORLD @ gl[:3, :3] @ GL_TO_CV_CAMERA
    out[:3, 3] = MV_TO_CV_WORLD @ gl[:3, 3]
    return out


def phi_elevation(c2w: np.ndarray) -> tuple[float, float]:
    """``(phi_deg, elevation_deg)`` of an OpenCV c2w in frame W: the camera centre is
    ``distance * (sin phi cos el, -sin el, -cos phi cos el)`` (app contract: ``forward(phi) =
    cos(phi) z - sin(phi) x``; elevation > 0 = camera above the subject)."""
    c = np.asarray(c2w, dtype=np.float64)[:3, 3]
    r = float(np.linalg.norm(c))
    if r <= 0.0:
        raise ValueError("the camera sits on the orbit centre")
    phi = round(math.degrees(math.atan2(c[0], -c[2])), 6) % 360.0
    elev = round(math.degrees(math.asin(max(-1.0, min(1.0, -c[1] / r)))), 6)
    return phi + 0.0, elev + 0.0


def _clean(m: np.ndarray) -> list:
    """Matrix as nested lists without float noise or negative zeros (JSON)."""
    return (np.round(np.asarray(m, dtype=np.float64), 12) + 0.0).tolist()


def views_meta(azimuths: Iterable[float] = AZIMUTHS, *, elevation_deg: float = ELEVATION,
               distance: float = DISTANCE) -> list[dict[str, Any]]:
    """One ``cameras.json`` view entry per azimuth: ``file`` (``view_<k>.png``), ``azimuth_deg``
    (MV-Adapter's), ``phi_deg`` / ``elevation_deg`` (the app's, derived from ``c2w``) and the
    OpenCV ``c2w``."""
    out = []
    for k, az in enumerate(azimuths):
        c2w = c2w_opencv(float(az), elevation_deg, distance)
        phi, elev = phi_elevation(c2w)
        out.append({"file": f"view_{k}.png", "azimuth_deg": float(az), "phi_deg": phi, "elevation_deg": elev,
                    "c2w": _clean(c2w)})
    return out


def cameras_doc(seed: int, *, azimuths: Iterable[float] = AZIMUTHS,
                input_fill_measured: float | None = None) -> dict[str, Any]:
    """The ``cameras.json`` document of one set of views."""
    doc: dict[str, Any] = {
        "projection": "orthographic",
        "ortho_half_extent": ORTHO_HALF,
        "distance": DISTANCE,
        "width": SIZE,
        "height": SIZE,
        "input_fill": INPUT_FILL,
        "pixel_center": PIXEL_CENTER,
        "convention": "c2w_opencv",
        "frame": "opencv",
        "world": "x = right in the input picture, y = down, z = away from the input camera; "
                 "origin = the orbit centre (centre of every view); units: the view box is +-0.55",
        "azimuth_convention": "MV-Adapter azimuth_deg (get_orthogonal_camera receives azimuth_deg - 90)",
        "phi_convention": "0 = the input view, 90 = camera on the input picture's +x side (you walked to your "
                          "right; the subject's left side), 180 = behind",
        "seed": int(seed),
        "worker_version": VERSION,
        "views": views_meta(azimuths),
    }
    if input_fill_measured is not None:
        doc["input_fill_measured"] = round(float(input_fill_measured), 4)
    return doc


# --------------------------------------------------------------------------- input (pure)
def subject_fill(alpha: np.ndarray, threshold: int = 127) -> float:
    """Long side of the bounding box of ``alpha > threshold`` (uint8 alpha; half-covered pixels
    count, faint resampling fringes do not) as a fraction of the picture size (0 when empty)."""
    ys, xs = np.nonzero(np.asarray(alpha) > threshold)
    if ys.size == 0:
        return 0.0
    h, w = np.shape(alpha)[:2]
    return max((ys.max() - ys.min() + 1) / h, (xs.max() - xs.min() + 1) / w)


def composite_on_grey(img: np.ndarray, grey: float = GREY) -> tuple[np.ndarray, dict[str, Any]]:
    """MV-Adapter's ``preprocess_image`` without its crop / re-scale: ``rgb * a + (1 - a) * grey``
    in float, then ``(x * 255).clip(0, 255).astype(uint8)`` as upstream.  ``img`` is a square
    :data:`SIZE` picture (H x W x 4 RGBA, H x W x 3 RGB or H x W grey, uint8).  Returns the RGB
    picture and ``{"fill", "transparent", "warnings"}``."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise WorkerError(f"The input picture must have 8-bit channels (got {a.dtype}).")
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise WorkerError(f"The input picture has an unsupported layout {a.shape}.")
    h, w = a.shape[:2]
    if (h, w) != (SIZE, SIZE):
        raise WorkerError(f"The input picture must be {SIZE} x {SIZE} pixels, framed by VR Photo Studio "
                          f"(got {w} x {h}).")
    warnings: list[str] = []
    if a.shape[2] == 4:
        alpha = a[..., 3]
        if not np.any(alpha > 0):
            raise WorkerError("The input picture is empty (fully transparent).",
                              hint="The cut-out of the subject failed; try another photo.")
        transparent = bool(np.any(alpha < 255))
    else:
        alpha = np.full((h, w), 255, np.uint8)
        transparent = False
    f = a.astype(np.float32) / 255.0
    if a.shape[2] == 4:
        out = f[..., :3] * f[..., 3:4] + (1.0 - f[..., 3:4]) * grey
    else:
        out = f
    rgb = (out * 255).clip(0, 255).astype(np.uint8)
    fill = subject_fill(alpha) if transparent else 1.0
    if not transparent:
        warnings.append("The input picture has no transparent background, so it was used as it is "
                        "(MV-Adapter expects a cut-out on grey).")
    elif not FILL_WARN[0] <= fill <= FILL_WARN[1]:
        warnings.append(f"The subject fills {fill * 100:.0f} % of the input picture; MV-Adapter expects about "
                        f"{INPUT_FILL * 100:.0f} %, so the views may be less consistent.")
    return rgb, {"fill": float(fill), "transparent": transparent, "warnings": warnings}


def load_input(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read the framed input picture and composite it (:func:`composite_on_grey`).  A square
    picture of another size is resized to :data:`SIZE` as a whole (the framing is unchanged; a
    warning says so); a non-square one is refused."""
    from PIL import Image

    try:
        im = Image.open(path)
        im.load()
    except Exception as e:
        raise WorkerError(f"Cannot read the input picture {Path(path).name}: {e}") from e
    if im.width != im.height:
        raise WorkerError(f"The input picture must be square, framed by VR Photo Studio (got {im.width} x "
                          f"{im.height}).")
    resized = None
    im = im.convert("RGBA") if ("A" in im.getbands() or im.mode == "P") else im.convert("RGB")
    if im.width != SIZE:
        resized = im.width
        if im.mode == "RGBA":  # premultiplied resampling: no dark fringes from transparent pixels
            im = im.convert("RGBa").resize((SIZE, SIZE), Image.Resampling.LANCZOS).convert("RGBA")
        else:
            im = im.resize((SIZE, SIZE), Image.Resampling.LANCZOS)
    rgb, info = composite_on_grey(np.asarray(im))
    if resized is not None:
        info["warnings"].insert(0, f"The input picture was {resized} px; it was resized to {SIZE} px "
                                   "(the framing is unchanged).")
    return rgb, info


# --------------------------------------------------------------------------- parameters (pure)
@dataclass
class Settings:
    """Validated request parameters."""

    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    seeds: tuple[int, ...] = (0,)
    prompt: str = DEFAULT_PROMPT
    negative: str = DEFAULT_NEGATIVE
    offload: str = "auto"
    name: str = "views"
    warnings: list[str] = field(default_factory=list)


def _whole(v: Any) -> int:
    """A whole number from an int, a whole float or a digit string (ValueError otherwise)."""
    if isinstance(v, bool):
        raise ValueError(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and math.isfinite(v) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*[+-]?\d+\s*", v):
        return int(v)
    raise ValueError(v)


def _seed(v: Any) -> int:
    try:
        s = _whole(v)
    except ValueError:
        s = -1
    if not 0 <= s <= MAX_SEED:
        raise WorkerError(f"Invalid seed {v!r}: use a whole number from 0 to {MAX_SEED}.")
    return s


def parse_params(params: dict[str, Any]) -> Settings:
    """Validate the request parameters (before torch is imported).  Numbers outside their range
    are clamped like :func:`_common.param`; malformed values raise :class:`WorkerError`."""
    p = params or {}
    steps = param(p, "steps", DEFAULT_STEPS, _whole, lo=STEPS_RANGE[0], hi=STEPS_RANGE[1])
    guidance = param(p, "guidance", DEFAULT_GUIDANCE, float)
    if not math.isfinite(guidance):
        raise WorkerError(f"Invalid value for 'guidance': {p.get('guidance')!r}")
    guidance = min(GUIDANCE_RANGE[1], max(GUIDANCE_RANGE[0], guidance))
    raw = p.get("seeds")
    if raw is None or (isinstance(raw, (list, tuple)) and not raw):
        raw = [0 if p.get("seed") is None else p.get("seed")]
    elif not isinstance(raw, (list, tuple)):
        raw = [raw]
    seeds = [_seed(v) for v in raw]
    warnings: list[str] = []
    unique = list(dict.fromkeys(seeds))
    if len(unique) < len(seeds):
        warnings.append("Repeated seeds were generated only once.")
    if len(unique) > MAX_SEEDS:
        raise WorkerError(f"Too many seeds ({len(unique)}): at most {MAX_SEEDS} sets of views per run.")
    prompt = " ".join(str(p.get("prompt") or "").split())[:MAX_PROMPT_CHARS] or DEFAULT_PROMPT
    neg_raw = p.get("negative_prompt")
    negative = DEFAULT_NEGATIVE if neg_raw is None else " ".join(str(neg_raw).split())[:MAX_PROMPT_CHARS]
    offload = str(p.get("offload") or "auto").strip().lower()
    if offload not in OFFLOAD_MODES:
        raise WorkerError(f"Invalid value for 'offload': {p.get('offload')!r} (use {', '.join(OFFLOAD_MODES)}).")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(p.get("name") or "views")).strip("_")[:40] or "views"
    return Settings(steps=int(steps), guidance=float(guidance), seeds=tuple(unique), prompt=prompt,
                    negative=negative, offload=offload, name=name, warnings=warnings)


def choose_offload(mode: str, free_gb: float | None, *, fp32: bool = False) -> str:
    """Placement for ``offload`` ``mode``: ``auto`` -> ``"model"`` (model CPU offload) when less than
    7 GB of VRAM is free or unknown, else ``"none"`` (all on the GPU).  fp32 weights: ``"none"``
    from 12 GB free, ``"model"`` from 10 GB, else ``"sequential"`` (an 8 GB card)."""
    if mode != "auto":
        return mode
    if fp32:
        if free_gb is None or free_gb < MODEL_OFFLOAD_FREE_GB_FP32:
            return "sequential"
        return "model" if free_gb < OFFLOAD_FREE_GB_FP32 else "none"
    return "model" if free_gb is None or free_gb < OFFLOAD_FREE_GB else "none"


def stricter_placement(a: str, b: str) -> str:
    """The more memory-saving of two placements (:data:`PLACEMENT_ORDER`)."""
    return max(a, b, key=PLACEMENT_ORDER.index)


def is_oom(exc: BaseException) -> bool:
    """True for a CUDA / allocator out-of-memory error (``torch.cuda.OutOfMemoryError`` & co.)."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    text = str(exc)
    return ("out of memory" in text.lower() or "CUBLAS_STATUS_ALLOC_FAILED" in text
            or "cudaErrorMemoryAllocation" in text)


def bad_views(images: Sequence[np.ndarray]) -> list[tuple[int, str]]:
    """``(index, "nan" | "black")`` for each float view (H x W x 3 in [0, 1]) that holds non-finite
    values (fp16 overflow) or is black.  A healthy view always shows the 0.5 grey background."""
    out = []
    for k, im in enumerate(images):
        a = np.asarray(im, dtype=np.float32)
        if a.size == 0 or not np.all(np.isfinite(a)):
            out.append((k, "nan"))
        elif float(a.max()) < 0.02:
            out.append((k, "black"))
    return out


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Float picture in [0, 1] -> uint8 (as diffusers' ``numpy_to_pil``: ``round(x * 255)``)."""
    return (np.clip(np.nan_to_num(np.asarray(img, dtype=np.float32)), 0.0, 1.0) * 255).round().astype(np.uint8)


def step_message(step: int, total: int, attempt: int, attempts: int) -> str:
    """``"Generating the views (try 1/2, step 12/30)"`` (no "try" part for a single set)."""
    tries = f"try {attempt}/{attempts}, " if attempts > 1 else ""
    return f"Generating the views ({tries}step {step}/{total})"


def step_callback(progress: Callable[[float, str], None], steps: int, attempt: int = 1,
                  attempts: int = 1) -> Callable[..., dict]:
    """``callback_on_step_end`` reporting ``(i + 1) / total`` to ``progress``; ``total`` is the
    pipeline's real timestep count (``_num_timesteps``) when it is known."""

    def on_step(pipe: Any, i: int, _t: Any, kwargs: dict) -> dict:
        try:
            total = int(getattr(pipe, "_num_timesteps", 0) or steps)
        except (TypeError, ValueError):
            total = steps
        total = max(1, total)
        n = min(int(i) + 1, total)
        progress(n / total, step_message(n, total, attempt, attempts))
        return kwargs

    return on_step


def adapter_fit(adapter_keys: Iterable[str], unet_keys: Iterable[str], cond_keys: Iterable[str]) -> dict[str, Any]:
    """How well the adapter's state dict fits the patched UNet and the condition encoder.
    ``load_custom_adapter`` loads with ``strict=False``, so a renamed module (another diffusers
    version) would silently leave random multi-view weights: ``ok`` needs >= 99 % of the keys
    to fit, multi-view attention weights among them and a condition encoder."""
    keys = list(adapter_keys)
    unet, cond = set(unet_keys), set(cond_keys)
    fit_unet = [k for k in keys if k in unet]
    fit_cond = [k for k in keys if k in cond]
    matched = len(set(fit_unet) | set(fit_cond))
    mv = sum(1 for k in fit_unet if "_mv" in k)
    frac = matched / len(keys) if keys else 0.0
    return {"keys": len(keys), "matched": matched, "fraction": frac, "mv": mv, "cond": len(fit_cond),
            "ok": bool(keys) and frac >= 0.99 and mv > 0 and len(fit_cond) > 0}


# --------------------------------------------------------------------------- dtype casting around the VAE
class _CastDist:
    """A latent distribution whose samples come back in ``dtype`` (the pipeline's)."""

    def __init__(self, dist: Any, dtype: Any):
        self._dist = dist
        self._dtype = dtype

    def sample(self, generator: Any = None) -> Any:
        return self._dist.sample(generator).to(self._dtype)

    def mode(self) -> Any:
        return self._dist.mode().to(self._dtype)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dist, name)


class _CastEncoderOutput:
    def __init__(self, out: Any, dtype: Any):
        self.latent_dist = _CastDist(out.latent_dist, dtype)


def cast_vae_io(vae: Any) -> Any:
    """Let an fp32 VAE work inside an fp16 pipeline: ``encode`` casts its input to the VAE's dtype
    and returns latents in the caller's dtype, ``decode`` casts the latents to the VAE's dtype
    (the decoded picture stays fp32).  The original bound methods are called, so diffusers'
    CPU-offload hooks on ``encode`` / ``decode`` keep working."""
    encode, decode = vae.encode, vae.decode

    def encode_cast(x: Any, *args: Any, **kwargs: Any) -> Any:
        dtype = x.dtype
        out = encode(x.to(vae.dtype), *args, **kwargs)
        if isinstance(out, tuple):  # return_dict=False
            return (_CastDist(out[0], dtype), *out[1:])
        return _CastEncoderOutput(out, dtype)

    def decode_cast(z: Any, *args: Any, **kwargs: Any) -> Any:
        return decode(z.to(vae.dtype), *args, **kwargs)

    vae.encode = encode_cast
    vae.decode = decode_cast
    return vae


# --------------------------------------------------------------------------- generation ladder (torch-free)
def generate_all(seeds: Sequence[int], *, make: Callable[[str, bool], Any],
                 generate: Callable[[Any, int, int], list[np.ndarray]], mode: str,
                 free: Callable[[], None] = free_cuda, free_gb: Callable[[], float | None] = lambda: None,
                 warnings: list[str] | None = None,
                 on_set: Callable[[int, int, list[np.ndarray]], Any] | None = None,
                 ) -> tuple[list[list[np.ndarray]], dict[str, Any]]:
    """Run every seed with one loaded pipeline and the fallback ladder.

    ``make(mode, fp32)`` loads and places a pipeline (``mode`` = "none" | "model" | "sequential");
    ``generate(pipe, seed, index)`` returns 6 float views.  Out of memory (while loading or
    generating) -> ``free()`` and a reload with sequential CPU offload; out of memory there ->
    :class:`WorkerError`.  NaN / black views -> one reload with fp32 weights (placed as
    :func:`choose_offload` says for fp32 at the free VRAM then, never less saving than before)
    and the seed again; bad views again -> :class:`WorkerError`.  ``on_set(index, seed, views)``
    runs as soon as a seed's views are good (the worker saves the set there, so a later failure
    or a cancel leaves the finished sets on disk)."""
    warns = warnings if warnings is not None else []
    state = {"offload": mode, "precision": "fp16", "reloads": 0}

    # The out-of-memory handling runs AFTER the ``except`` block: inside it the exception's
    # traceback still references the frames holding the pipeline and its CUDA tensors, so
    # ``free()`` could not release them.
    def escalate(message: str) -> None:
        free()
        if state["offload"] == "sequential":
            raise WorkerError("The graphics card ran out of memory while making the views, even with the "
                              "slowest memory-saving mode.", hint=OOM_HINT)
        log(f"out of memory with offload '{state['offload']}': {message}", "warning")
        warns.append("MV-Adapter ran out of GPU memory and continued with sequential CPU offload (slower).")
        state["offload"] = "sequential"
        state["reloads"] += 1

    def oom_text(e: BaseException) -> str:
        text = str(e).strip()
        return text.splitlines()[0][:200] if text else type(e).__name__

    def load() -> Any:
        while True:
            oom = None
            try:
                return make(str(state["offload"]), state["precision"] == "fp32")
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = oom_text(e)
            escalate(oom)

    results: list[list[np.ndarray]] = []
    pipe: Any = load()
    for i, seed in enumerate(seeds):
        while True:
            oom = None
            try:
                images = generate(pipe, int(seed), i)
            except Exception as e:
                if not is_oom(e):
                    raise
                oom = oom_text(e)
            if oom is not None:
                pipe = None
                escalate(oom)
                pipe = load()
                continue
            bad = bad_views(images)
            if not bad:
                break
            what = ", ".join(f"view {k} {kind}" for k, kind in bad)
            if state["precision"] == "fp32":
                raise WorkerError(f"MV-Adapter produced broken pictures even in full precision ({what}).",
                                  hint="Try another seed ('Variation') or let 'SHARP 360' use TRELLIS or the "
                                       "built-in back.")
            log(f"seed {seed}: broken fp16 output ({what}); retrying with fp32 weights", "warning")
            warns.append("MV-Adapter's half-precision result was broken (black or NaN pictures); it was made "
                         "again in full precision (slower).")
            pipe = None
            free()
            state["precision"] = "fp32"
            state["offload"] = stricter_placement(str(state["offload"]), choose_offload("auto", free_gb(), fp32=True))
            state["reloads"] += 1
            pipe = load()
        results.append(images)
        if on_set is not None:
            on_set(i, int(seed), images)
    pipe = None
    free()
    return results, state


# --------------------------------------------------------------------------- files
@dataclass(frozen=True)
class ModelPaths:
    src: Path
    base: Path
    adapter: Path


def model_paths() -> ModelPaths:
    """Where the installer put the code and the weights (``MVADAPTER_BASE`` / ``MVADAPTER_WEIGHTS``)."""
    root = addon_dir()
    base = Path(os.environ.get("MVADAPTER_BASE") or root / "MODELS" / "sd21-base")
    weights = Path(os.environ.get("MVADAPTER_WEIGHTS") or root / "MODELS" / "mv-adapter")
    return ModelPaths(src=src_dir("mvadapter"), base=base, adapter=weights / ADAPTER_FILE)


def missing_files(paths: ModelPaths) -> list[str]:
    """Required files that are missing (display names).  A model file must also be non-empty; a
    source file need not be (``mvadapter/__init__.py`` is empty at the pinned commit)."""

    def ok(p: Path, *, empty_ok: bool = False) -> bool:
        try:
            return p.is_file() and (empty_ok or p.stat().st_size > 0)
        except OSError:
            return False

    out = [f"src/mvadapter/{f}" for f in SOURCE_FILES if not ok(paths.src / f, empty_ok=True)]
    out += [f"sd21-base/{f}" for f in BASE_FILES if not ok(paths.base / f)]
    if not ok(paths.adapter):
        out.append(ADAPTER_FILE)
    return out


def require_models(paths: ModelPaths) -> None:
    missing = missing_files(paths)
    if missing:
        more = f" and {len(missing) - 4} more" if len(missing) > 4 else ""
        raise WorkerError(f"MV-Adapter files are missing: {', '.join(missing[:4])}{more}.", hint=REINSTALL_HINT)


def save_set(out_dir: Path, name: str, seed: int, views: Sequence[np.ndarray], *,
             input_fill: float | None = None) -> dict[str, Any]:
    """Write one set (``<out_dir>/<name>_<seed>/view_<k>.png``, then ``cameras.json`` atomically)
    and return its ``{"seed", "views", "cameras"}`` entry."""
    from PIL import Image

    if len(views) != len(AZIMUTHS):
        raise WorkerError(f"MV-Adapter returned {len(views)} pictures instead of {len(AZIMUTHS)}.")
    folder = Path(out_dir) / f"{name}_{int(seed)}"
    folder.mkdir(parents=True, exist_ok=True)
    doc = cameras_doc(seed, input_fill_measured=input_fill)
    files = []
    for entry, img in zip(doc["views"], views, strict=True):
        arr = to_uint8(img) if np.asarray(img).dtype != np.uint8 else np.asarray(img)
        if arr.shape[:2] != (SIZE, SIZE):
            raise WorkerError(f"MV-Adapter returned a {arr.shape[1]} x {arr.shape[0]} picture instead of "
                              f"{SIZE} x {SIZE}.")
        path = folder / entry["file"]
        Image.fromarray(np.ascontiguousarray(arr[..., :3])).save(path)
        files.append(str(path))
    cams = folder / "cameras.json"
    tmp = folder / "cameras.json.tmp"  # written last and atomically: its presence marks a whole set
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, cams)
    return {"seed": int(seed), "views": files, "cameras": str(cams)}


# --------------------------------------------------------------------------- torch / diffusers
def _load_by_path(module_name: str, path: Path) -> Any:
    """Import one source file without running its package's ``__init__``."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _import_mvadapter(src: Path) -> tuple[Any, Any, Callable[..., Any]]:
    """``(MVAdapterI2MVSDPipeline, ShiftSNRScheduler, get_plucker_embeds_from_cameras_ortho)``.

    The pipeline and scheduler modules import only torch / diffusers / transformers /
    safetensors / einops.  ``mvadapter.utils`` is never imported: its ``__init__`` pulls in the
    texturing stack (nvdiffrast, triton, trimesh, matplotlib, jaxtyping, ...), so the camera
    maths is ported (:func:`mv_c2w_gl`) and ``utils/geometry.py`` (numpy + torch only) is
    loaded straight from its file."""
    add_sys_path(src)
    from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    geometry = _load_by_path("_mvadapter_geometry", Path(src) / "mvadapter" / "utils" / "geometry.py")
    return MVAdapterI2MVSDPipeline, ShiftSNRScheduler, geometry.get_plucker_embeds_from_cameras_ortho


def _free_vram_gb() -> float | None:
    import torch

    try:
        free, _total = torch.cuda.mem_get_info()
        return free / 1024**3
    except Exception:
        return None


def _load_pipeline(paths: ModelPaths, *, fp32: bool) -> Any:
    """Build the i2mv pipeline like ``prepare_pipeline`` of the script (DDPM, as the README
    recommends for SD 2.1), on the CPU: UNet / text encoder / condition encoder in fp16 (fp32
    with ``fp32``), VAE in fp32 with cast I/O, sliced decoding."""
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler
    from safetensors.torch import load_file

    pipe_cls, shift_cls, _ = _import_mvadapter(paths.src)
    dtype = torch.float32 if fp32 else torch.float16
    vae = AutoencoderKL.from_pretrained(str(paths.base), subfolder="vae", torch_dtype=torch.float32,
                                        use_safetensors=True, local_files_only=True)
    pipe = pipe_cls.from_pretrained(str(paths.base), vae=vae, safety_checker=None, feature_extractor=None,
                                    torch_dtype=dtype, variant="fp16", use_safetensors=True,
                                    local_files_only=True)
    pipe.scheduler = shift_cls.from_scheduler(pipe.scheduler, shift_mode="interpolated",
                                              shift_scale=SHIFT_SCALE, scheduler_class=DDPMScheduler)
    pipe.init_custom_adapter(num_views=len(AZIMUTHS))
    state = load_file(str(paths.adapter))
    fit = adapter_fit(state.keys(), pipe.unet.state_dict().keys(), pipe.cond_encoder.state_dict().keys())
    log(f"adapter weights: {fit['matched']}/{fit['keys']} fit ({fit['mv']} multi-view, {fit['cond']} encoder)")
    if not fit["ok"]:
        raise WorkerError(f"The MV-Adapter weights do not fit the installed MV-Adapter code ({fit['matched']} of "
                          f"{fit['keys']} weights fit).", hint=REINSTALL_HINT)
    pipe.load_custom_adapter(state, weight_name=ADAPTER_FILE)
    del state
    pipe.unet.to(dtype=dtype)
    pipe.text_encoder.to(dtype=dtype)
    pipe.cond_encoder.to(dtype=dtype)
    pipe.vae.to(dtype=torch.float32)
    cast_vae_io(pipe.vae)
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _place(pipe: Any, mode: str) -> Any:
    """Put the pipeline on the GPU (``none``) or behind model / sequential CPU offload.  The
    condition encoder is not a registered pipeline component, so it always goes to the GPU
    itself (about 150 MB in fp16)."""
    if mode == "none":
        pipe.to("cuda")
    elif mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        raise WorkerError(f"Unknown offload mode {mode!r}.")
    pipe.cond_encoder.to("cuda")
    return pipe


def _control_images(plucker_fn: Callable[..., Any]) -> Any:
    """The camera condition of the script: ``(plucker + 1) / 2`` of the six orthographic cameras."""
    import torch

    c2w = torch.from_numpy(mv_c2w_gl([a + AZIMUTH_OFFSET for a in AZIMUTHS])).float()
    plucker = plucker_fn(c2w, [ORTHO_SCALE] * len(AZIMUTHS), SIZE)
    return ((plucker + 1.0) / 2.0).clamp(0, 1)


def _generate(pipe: Any, reference: Any, control: Any, s: Settings, seed: int,
              progress: Callable[[float, str], None], attempt: int, attempts: int) -> list[np.ndarray]:
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    out = pipe(
        s.prompt,
        height=SIZE,
        width=SIZE,
        num_inference_steps=s.steps,
        guidance_scale=s.guidance,
        num_images_per_prompt=len(AZIMUTHS),
        control_image=control,
        control_conditioning_scale=1.0,
        reference_image=reference,
        reference_conditioning_scale=1.0,
        negative_prompt=s.negative,
        generator=generator,
        output_type="np",
        callback_on_step_end=step_callback(progress, s.steps, attempt, attempts),
    )
    return [np.asarray(im, dtype=np.float32) for im in out.images]


def handle(task: str, inputs: dict, params: dict, out_dir: Path, progress: Progress) -> dict:
    if task != TASK:
        raise WorkerError(f"Unknown task '{task}'.")
    # everything that can be checked without torch comes first
    image_path = require_input(inputs, "image")
    s = parse_params(params)
    rgb, info = load_input(image_path)
    paths = model_paths()
    require_models(paths)
    warnings = [*s.warnings, *info["warnings"]]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is local; never wait for the network

    cuda = cuda_info(require=True)
    from PIL import Image

    ref_path = Path(out_dir) / f"{s.name}_input.png"
    Image.fromarray(rgb).save(ref_path)
    reference = Image.fromarray(rgb)
    mode = choose_offload(s.offload, _free_vram_gb())
    log(f"{cuda.get('gpu', 'GPU')} ({cuda.get('vram_gb', '?')} GB): offload '{mode}', {s.steps} steps, "
        f"guidance {s.guidance:g}, seeds {list(s.seeds)}, input fill {info['fill']:.3f}")
    progress(0.02, "Loading MV-Adapter")
    try:
        _, _, plucker_fn = _import_mvadapter(paths.src)
    except ImportError as e:  # a source file gone after the install (antivirus, partial clean-up)
        raise WorkerError(f"The MV-Adapter code could not be loaded ({type(e).__name__}: {e}).",
                          hint=REINSTALL_HINT) from e
    control = _control_images(plucker_fn)
    n = len(s.seeds)
    gen_lo, gen_hi = 0.15, 0.97
    span = (gen_hi - gen_lo) / n
    track = {"loads": 0, "frac": 0.02}  # a reload reports at the current fraction (never backwards)

    def make(placement: str, fp32: bool) -> Any:
        if track["loads"]:
            progress(track["frac"], f"Reloading MV-Adapter ({'full' if fp32 else 'half'} precision, "
                                    f"offload {placement})")
        track["loads"] += 1
        t0 = time.monotonic()
        pipe = _place(_load_pipeline(paths, fp32=fp32), placement)
        log(f"pipeline loaded in {time.monotonic() - t0:.0f} s ({'fp32' if fp32 else 'fp16'}, offload {placement})")
        track["frac"] = max(track["frac"], gen_lo)
        progress(track["frac"], "MV-Adapter loaded")
        return pipe

    def generate(pipe: Any, seed: int, i: int) -> list[np.ndarray]:
        track["frac"] = gen_lo + i * span
        sub = progress.sub(gen_lo + i * span, gen_lo + (i + 1) * span)
        t0 = time.monotonic()
        views = _generate(pipe, reference, control, s, seed, sub, i + 1, n)
        log(f"seed {seed}: {len(views)} views in {time.monotonic() - t0:.0f} s")
        return views

    sets: list[dict[str, Any]] = []

    def on_set(i: int, seed: int, views: list[np.ndarray]) -> None:
        # each set is written as soon as it is made (cameras.json last), so a later failure, an
        # out-of-memory error or a cancel (the app kills the process) keeps the finished sets
        sets.append(save_set(out_dir, s.name, seed, views, input_fill=info["fill"]))
        log(f"seed {seed}: saved")

    _results, state = generate_all(s.seeds, make=make, generate=generate, mode=mode, free=free_cuda,
                                   free_gb=_free_vram_gb, warnings=warnings, on_set=on_set)
    progress(1.0, "Done")
    return {
        "sets": sets,
        "frame": "opencv",
        "warnings": warnings,
        "input": str(ref_path),
        "offload": state["offload"],
        "precision": state["precision"],
        "steps": s.steps,
        "guidance": s.guidance,
        "azimuths": list(AZIMUTHS),
        "worker_version": VERSION,
    }


def check(require_gpu: bool) -> dict:
    import importlib

    info = cuda_info(require=require_gpu)
    failures: dict[str, str] = {}
    for mod in ("diffusers", "transformers", "accelerate", "safetensors", "einops", "PIL"):
        try:
            importlib.import_module(mod)
        except Exception as e:
            failures[mod] = f"{type(e).__name__}: {e}"
    paths = model_paths()
    if not failures:
        try:
            _import_mvadapter(paths.src)
        except Exception as e:
            failures["MV-Adapter code"] = f"{type(e).__name__}: {e}"
    missing = missing_files(paths)
    if missing:
        failures["model files"] = "missing " + ", ".join(missing[:6]) + (" ..." if len(missing) > 6 else "")
    elif not failures:
        try:
            from safetensors import safe_open

            with safe_open(str(paths.adapter), framework="pt") as f:
                info["adapter_tensors"] = len(list(f.keys()))
        except Exception as e:
            failures["MV-Adapter weights"] = f"{type(e).__name__}: {e}"
    if failures:
        detail = "; ".join(f"{k}: {v}" for k, v in failures.items())
        raise WorkerError(f"Broken MV-Adapter environment ({detail})", hint=REINSTALL_HINT)
    import diffusers

    info["diffusers"] = diffusers.__version__
    return info


if __name__ == "__main__":
    sys.exit(run(handle, name="mvadapter", version=VERSION, check=check, tasks=[TASK]))
