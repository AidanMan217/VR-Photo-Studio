"""SHARP 360: SHARP's photo-real front plus a predicted back, life-size (object method ``sharp360``).

Design: docs/design/SHARP360.md (sections 1.3-1.5, 3.1, 3.12 and 4.2); conventions and the
front guarantee: docs/CONTRACTS.md section 13.  :func:`vrphoto.pipeline.run._mode_object`
validates the options (:func:`resolve_options`), checks that SHARP is installed and finds
the subject, then hands over to :func:`mode_sharp360`:

1. SHARP reconstructs the whole photo (the same add-on call as the plain ``sharp`` method).
2. :func:`~vrphoto.geometry.splat_fusion.sharp_front` cleans it down to the subject's
   photographed front (hidden wall layer, edge streaks and background removed) and
   :func:`~vrphoto.geometry.splat_fusion.canonical_frame` finds the subject's upright frame.
3. A **back source** makes the hidden side.  Sources are tried in order (:data:`SOURCES`,
   ``Opts.sources``): one that is missing, fails or returns an unusable back only adds a
   warning and the next one runs; the built-in shell is always last, and when even it fails
   the cleaned front is exported alone.  A cancel is never turned into a fallback
   (``Cancelled`` is a ``VRPhotoError``, so it is re-raised first).
4. Front and back are stitched (:mod:`~vrphoto.geometry.splat_merge`: dithered seam,
   colour feathering, floater removal).  Any unexpected failure there exports the cleaned
   front with a warning - once SHARP has succeeded the job only fails when cancelled.
5. Export (design 3.12): glTF frame, rotated upright about the found "up", life size from
   the entered height or from SHARP's own metric size, box crop, decimation to the viewer
   budget; the scene block carries ``scale: 1.0`` so the viewer shows it 1:1.

Back sources (:data:`SOURCES`): the built-in ``shell`` and, from 0.2.0, ``trellis`` (the
TRELLIS add-on's whole-body Gaussians, lined up behind SHARP's front, see
:func:`_src_trellis`; "auto" tries it first when it is installed, except with "fast"
quality) and ``images`` (the user's own back photo plus optional side photos, or back
views made with any AI tool, bent onto the photo's outline and lifted onto a mould, see
:func:`_src_images`; never chosen by "auto").  MV-Adapter and Qwen join in later
milestones, as does the Brush polish.  The 'Variation' number (``seed``) is the AI
sources' seed (attempt ``i`` uses ``seed + i``) and the seam dither's; the photographed
front never depends on it.

Outputs: ``<base>_cutout.png``, ``<base>_sharp360.splat``, ``<base>_sharp360_preview.jpg``
(a turntable sheet of the final splat, 8 views; a failure there is only a warning),
``<base>_sharp360_views.png`` (a contact sheet of the back / side photos as lined up, only
when the ``images`` source ran) and, with ``save_extras``, ``_sharp360.ply``,
``_sharp360_front.splat`` (the cleaned SHARP front alone, same frame),
``_sharp360_back.splat`` and ``_sharp360_debug.json``.  The debug data is always written
to the job's work folder and summarised in one log line.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import (
    UI_ADDONS,
    UI_LOG,
    AddonError,
    AddonNotInstalled,
    Cancelled,
    InputError,
    VRPhotoError,
)
from ..core.fsutil import atomic_write_bytes
from ..core.log import get_logger
from ..core.types import FloatMap, ProgressReporter, SplatCloud
from ..geometry.back_mould import BackPart, shell_back
from ..geometry.camera import cv_to_gltf
from ..geometry.register import apply_color_affine, color_affine_fit, register_cloud_to_front
from ..geometry.splat import (
    crop_splats,
    decimate_splats,
    normalize_orientation,
    read_splats,
    subset_splats,
    write_splats,
)
from ..geometry.splat_fusion import (
    MOULD_LONG,
    Canon,
    FrontModel,
    SharpCam,
    canonical_frame,
    depth_tol,
    sharp_front,
)
from ..geometry.splat_merge import add_underlayer, feather_colors, merge_parts, remove_floaters, to_life_size
from . import run as R

log = get_logger(__name__)

#: The source that always works and ends every chain (no AI, no download).
FINAL_SOURCE = "shell"
#: Seam angle limits (deg) of the "Keep the photo up to" option.
SEAM_RANGE = (35.0, 80.0)
#: Thickness (a fraction of the height) the life-size back accepts.  The option is shared
#: with Billboard, whose presets go up to 0.24 for a stronger 3D look; a back that deep
#: makes a person about twice as deep as a real body, so it is clamped (with a warning).
THICKNESS_RANGE = (0.06, 0.18)
#: Most AI attempts per source ('AI attempts' / ``back_tries``; 0 = the quality's default).
MAX_TRIES = 4
#: Most AI generation steps ('AI steps' / ``back_steps``; 0 = each source's default).
MAX_STEPS = 100
#: Largest 'Variation' number: attempt ``i`` uses ``seed + i``, which every generator and
#: numpy's seam dither take as is (a negative seed is an error for numpy).
MAX_SEED = 2**31 - 1


@dataclass(frozen=True)
class Quality:
    """What a ``back_quality`` setting means (design 1.3)."""

    mould_long: int  # long side of the isotropic mould grid (cells)
    stride: int  # one back Gaussian every ``stride`` mould cells
    auto: tuple[str, ...]  # sources "auto" tries, best first (only installed ones are used)
    tries: int = 1  # AI attempts when 'AI attempts' is Automatic (the best fit is kept)
    trellis_steps: tuple[int, int] = (12, 12)  # TRELLIS sparse-structure / structured-latent steps
    relief: str = "depth"  # what 'Back detail' Automatic means for a back made from a picture
    fill_holes: bool = True  # side photos also fill side pixels nothing else covers (design V10)
    preview_px: int = 384  # size of one turntable preview view


QUALITY: dict[str, Quality] = {
    # "fast" never waits minutes for an AI back: the built-in one (an explicit source still runs)
    "fast": Quality(MOULD_LONG["fast"], 2, (FINAL_SOURCE,), relief="flat", fill_holes=False, preview_px=256),
    "balanced": Quality(MOULD_LONG["balanced"], 1, ("trellis", FINAL_SOURCE)),
    "best": Quality(
        MOULD_LONG["best"], 1, ("trellis", FINAL_SOURCE), tries=2, trellis_steps=(25, 25), preview_px=512
    ),
}

#: How each source is named in messages (mid-sentence) and in the output's label.
SOURCE_NAMES: dict[str, str] = {
    "shell": "the built-in back",
    "trellis": "TRELLIS",
    "images": "your back photo",
}
SOURCE_LABELS: dict[str, str] = {
    "shell": "built-in back",
    "trellis": "back by TRELLIS",
    "images": "back from your photo",
}
#: The add-on a source needs; "auto" skips sources whose add-on is not installed.  The
#: built-in back needs none.
SOURCE_ADDONS: dict[str, str] = {"trellis": "trellis"}


@dataclass(frozen=True)
class Opts:
    """Validated SHARP 360 options of one job (see :func:`resolve_options`)."""

    back_source: str  # as chosen: "auto" or a source id
    quality: str  # fast | balanced | best
    sources: tuple[str, ...]  # tried in this order; always ends with FINAL_SOURCE
    mould_long: int
    stride: int
    seam: float  # degrees
    size_from: str  # auto | height | photo
    save_extras: bool
    back: str  # built-in back colours: diffuse | mirror
    thickness: float  # median body thickness as a fraction of the height (clamped)
    seed: int  # 'Variation': the AI sources' seed (attempt i uses seed + i) and the seam dither
    thickness_asked: float | None = None  # the job's value when it was clamped, else None
    tries: int = 1  # AI attempts per AI source (the job's 'AI attempts', else the quality's)
    steps: int = 0  # AI generation steps asked by the job; 0 = each source's default for the quality
    color_match: bool = True  # match the back's colours / exposure to the photo
    relief: str = "auto"  # 'Back detail' of a back made from a picture: auto | depth | flat
    back_images: str | None = None  # 'Back photo' (checked) when the images source is chosen


def _whole(value: Any, what: str, lo: int, hi: int, hint: str) -> int:
    """``value`` as an int in ``[lo, hi]`` (whole-number floats and strings from job files
    are accepted), else :class:`InputError`."""
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        v = float("nan")
    if not (math.isfinite(v) and v == math.floor(v) and lo <= v <= hi):
        raise InputError(f"{what} must be a whole number from {lo} to {hi} (got {value!r}).", hint=hint)
    return int(v)


def _seam(value: Any) -> float:
    lo, hi = SEAM_RANGE
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = float("nan")
    if not (math.isfinite(v) and lo <= v <= hi):
        raise InputError(
            f"'Keep the photo up to' must be between {lo:g} and {hi:g} degrees (got {value!r}).",
            hint="60 degrees suits most photos.",
        )
    return v


def _is_back(phi: float) -> bool:
    return abs((float(phi) - 180.0 + 180.0) % 360.0 - 180.0) < 1.0


def _back_photo(value: Any) -> str:
    """The 'Back photo' of the ``images`` source, checked before any work: given, an existing
    picture (or a folder of views) with a view from behind among the views it names, and a
    readable picture header.  Surrounding quotes are removed (Windows Explorer's 'Copy as
    path' adds them).  Returns the path (``~`` expanded); :class:`InputError` otherwise.  A
    picture that only fails while decoding fails the source later (the job then warns and
    uses the built-in back).  Qt-free: the GUI checks a job with it before queueing."""
    from ..core.imaging import open_pil
    from ..geometry.views_lift import views_from_folder

    text = str(value).strip().strip('"').strip("'").strip() if value is not None else ""
    if not text:
        raise InputError(
            "'Your own back photo' needs a picture of the back ('Back photo').",
            hint="Choose the photo of the subject from behind, or pick another 'Back made by'.",
        )
    path = Path(text).expanduser()
    entries = views_from_folder(path)  # InputError: not found / not a picture
    backs = [p for phi, p in entries if _is_back(phi)]
    if not backs:
        raise InputError(
            f"No picture of the back was found in {path}.",
            hint="Pick the photo taken from behind, or name it back.jpg (side photos next to it: "
            "right.jpg, left.jpg, back_right.jpg, ...).",
        )
    try:
        open_pil(backs[0], load=False)  # header only: cheap even for a 200 MP photo
    except InputError as e:  # "Cannot read image back.jpg: ..."
        raise InputError(
            f"The back photo cannot be used. {e}",
            hint="Save it again as JPEG or PNG and choose it again ('Back photo').",
        ) from e
    return str(path)


def resolve_options(spec: Any, *, installed: Callable[[str], bool]) -> Opts:
    """Check the SHARP 360 options of an object job and pick the back sources to try.

    Raises :class:`InputError` for an unknown back source, quality or size setting, a
    seam angle outside 35-80 degrees, or a bad thickness / back colour, so a bad job fails
    before any slow work.  A valid thickness outside :data:`THICKNESS_RANGE` is clamped
    into it (``Opts.thickness_asked`` keeps the job's value; the job warns).  ``"auto"``
    tries the quality's :attr:`Quality.auto` sources whose add-on is ``installed``; an
    explicit source is tried even when its add-on is missing (the job then warns and falls
    back).  The built-in back always ends the chain.
    ``seed`` ('Variation', 0 to :data:`MAX_SEED`), ``back_tries`` (0 = Automatic, up to
    :data:`MAX_TRIES`) and ``back_steps`` (0 = Automatic, up to :data:`MAX_STEPS`) must be
    whole numbers in range; ``tries`` resolves Automatic to the quality's
    :attr:`Quality.tries`, ``steps`` stays 0 (each AI source has its own default).
    ``back_relief`` must be one of ``run.BACK_RELIEFS``.  With ``back_source="images"`` the
    'Back photo' (``back_images``) must name a readable picture of the back
    (:func:`_back_photo`); "auto" never picks that source (it needs the user's picture)."""
    source = R._choice(spec.back_source, R.BACK_SOURCES, "back source")
    quality = R._choice(spec.back_quality, R.BACK_QUALITIES, "back quality")
    relief = R._choice(getattr(spec, "back_relief", "auto"), R.BACK_RELIEFS, "'Back detail' setting")
    back_images = _back_photo(getattr(spec, "back_images", None)) if source == "images" else None
    size_from = R._choice(spec.size_from, R.SIZE_FROM, "'Size from' setting")
    seam = _seam(spec.seam_deg)
    back = R._choice(spec.back, R.BACKS, "back texture")
    asked = R._positive(spec.thickness, "Thickness", allow_zero=True)
    thickness = min(max(asked, THICKNESS_RANGE[0]), THICKNESS_RANGE[1])
    seed = _whole(getattr(spec, "seed", 0), "'Variation'", 0, MAX_SEED, "0 is the usual value.")
    tries = _whole(
        getattr(spec, "back_tries", 0),
        "'AI attempts'",
        0,
        MAX_TRIES,
        "0 (Automatic) makes 1 attempt, or 2 with 'Best' back quality.",
    )
    steps = _whole(
        getattr(spec, "back_steps", 0),
        "'AI steps'",
        0,
        MAX_STEPS,
        "0 (Automatic) uses 12 steps, or 25 with 'Best' back quality.",
    )
    q = QUALITY[quality]
    if source == "auto":
        chain = [
            s for s in q.auto if s in SOURCES and (s not in SOURCE_ADDONS or installed(SOURCE_ADDONS[s]))
        ]
    else:
        chain = [source]
    if FINAL_SOURCE not in chain:
        chain.append(FINAL_SOURCE)
    return Opts(
        back_source=source,
        quality=quality,
        sources=tuple(chain),
        mould_long=q.mould_long,
        stride=q.stride,
        seam=seam,
        size_from=size_from,
        save_extras=bool(spec.save_extras),
        back=back,
        thickness=thickness,
        seed=seed,
        thickness_asked=None if thickness == asked else asked,
        tries=tries or q.tries,
        steps=steps,
        color_match=bool(getattr(spec, "color_match", True)),
        relief=relief,
        back_images=back_images,
    )


# --------------------------------------------------------------------------- back sources
@dataclass
class Ctx:
    """What a back source gets: the job, the photo and SHARP's cleaned front."""

    run: Any  # run._Run
    item: Any  # run._Item
    photo: Any  # run._Photo
    alpha: FloatMap
    sc_full: SplatCloud  # SHARP's whole reconstruction (frame C)
    front: FrontModel
    canon: Canon
    opts: Opts
    #: What a source leaves for the previews, e.g. ``"views"``: ``(OrthoInput, [GenView])``
    #: for the contact sheet of a view-based source (also when its back was not usable).
    previews: dict[str, Any] = field(default_factory=dict)


def _src_shell(ctx: Ctx, rep: ProgressReporter) -> BackPart:
    """The built-in back: the silhouette inflated behind the front (no AI, no download)."""
    o = ctx.opts
    return shell_back(
        ctx.front,
        ctx.canon,
        thickness=o.thickness,
        back=o.back,
        stride=o.stride,
        progress=rep,
        cancel=ctx.run.cancel,
    )


# --------------------------------------------------------------------------- TRELLIS (design 3.5.5)
TRELLIS_ADDON = "trellis"
_TRELLIS_TASK = "image_to_3d"
_TRELLIS_INPUT_PX = 1024  # long side of the picture TRELLIS gets (it works at 518 px itself)
_TRELLIS_OPAQUE = 0.3  # TRELLIS Gaussians that define its back depth and colour (registration's "opaque")
_COLOR_MIN_PAIRS = 64  # fewer photographed cells than this: TRELLIS keeps its own colours
_COLOR_GRAD_PCT = 30.0  # colour pairs only where the photo is smoothest (design V4) ...
_COLOR_RIM_FRAC = 0.01  # ... at least this fraction of the grid's long side (>= 3 cells) inside the outline
_COLOR_GAIN_MIN = 0.02  # linear RGB: darker source channels say nothing about a gain
_COLOR_GAIN_RANGE = (0.6, 1.6)  # as the affine's diagonal (register.color_affine_fit)


@dataclass
class _Try:
    """One TRELLIS attempt: its seed, the registered cloud (frame C) when the gates passed,
    the registration diagnostics and, when it failed before registering, why."""

    index: int  # 1-based
    seed: int
    sc: SplatCloud | None = None
    diag: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0

    @property
    def score(self) -> float:
        """The registration's own candidate score (design 3.5.5 step 3), higher is better:
        ``IoU - 0.5 min(1, residual / 10 cm)``; -inf when it could not be measured."""
        iou, resid = self.diag.get("iou"), self.diag.get("resid_m")
        if iou is None or resid is None or not (math.isfinite(iou) and math.isfinite(resid)):
            return -math.inf
        return float(iou) - 0.5 * min(1.0, float(resid) / 0.1)

    def summary(self) -> dict[str, Any]:
        keys = ("ok", "reason", "iou", "resid_m", "scale", "up", "yaw_deg", "tilt_deg", "n_in")
        out = {"try": self.index, "seed": self.seed, "seconds": round(self.seconds, 2)}
        out.update({k: self.diag[k] for k in keys if k in self.diag})
        if self.error:
            out.update(ok=False, error=self.error)
        return out


def _empty_cloud() -> SplatCloud:
    z3 = np.zeros((0, 3), np.float32)
    return SplatCloud(z3, z3.copy(), np.zeros((0, 4), np.float32), z3.copy(), np.zeros(0, np.float32))


def _cutout_input(ctx: Ctx) -> np.ndarray:
    """The photo's RGBA cut-out (``run._cutout_rgba``), at most :data:`_TRELLIS_INPUT_PX` on
    its long side."""
    import cv2

    cut = R._cutout_rgba(np.asarray(ctx.photo.rgb), np.asarray(ctx.alpha, dtype=np.float32))
    h, w = cut.shape[:2]
    k = _TRELLIS_INPUT_PX / max(h, w)
    if k < 1.0:
        size = (max(1, round(w * k)), max(1, round(h * k)))
        cut = cv2.resize(cut, size, interpolation=cv2.INTER_AREA)
    return cut


def _trellis_input(ctx: Ctx) -> tuple[np.ndarray, str]:
    """TRELLIS's input picture (RGBA uint8) and its kind (``"levelled"`` or ``"cutout"``).

    Preferred: the levelled orthographic picture (design 3.4, ``views_lift.
    levelled_ortho_input(...).rgba_ref``, 1024 px): the photo forward-splatted onto SHARP's
    front and seen square-on from the horizontal.  TRELLIS models what it sees as an upright
    object under a near-orthographic camera, so taking out the camera pitch and the
    perspective (~12 % magnification of near parts) gives it the subject's true proportions.
    When that picture cannot be made, the plain cut-out: TRELLIS crops and centres any RGBA
    picture itself, and the registration does not rely on the framing, only the shape can
    come out a little distorted."""
    try:
        from ..geometry.views_lift import levelled_ortho_input
    except ImportError:  # the levelled input is not part of this build
        return _cutout_input(ctx), "cutout"
    try:
        oi = R._call_opt(
            levelled_ortho_input, ctx.photo.rgb, ctx.alpha, ctx.front, ctx.canon, cancel=ctx.run.cancel
        )
        rgba = np.asarray(oi.rgba_ref)
        if rgba.ndim != 3 or rgba.shape[2] != 4 or not (rgba[..., 3] > 127).any():
            raise ValueError(f"unusable levelled picture {rgba.shape}")
        if rgba.dtype != np.uint8:
            rgba = (np.clip(rgba, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return rgba, "levelled"
    except Cancelled:
        raise
    except Exception:  # a bug or an odd front there must not cost the AI back
        log.exception("SHARP 360: the levelled TRELLIS input failed; using the plain cut-out")
        return _cutout_input(ctx), "cutout"


def _trellis_try(
    ctx: Ctx,
    title: str,
    image: Path,
    t: _Try,
    steps: tuple[int, int],
    n_tries: int,
    rep: ProgressReporter,
) -> None:
    """Run TRELLIS once (seed ``t.seed``) and register its Gaussians to the front; fills ``t``.
    Raises what the add-on raises, or :class:`AddonError` when it returns no Gaussians."""
    run, item = ctx.run, ctx.item
    label = "TRELLIS" if n_tries == 1 else f"TRELLIS (try {t.index}/{n_tries})"
    gen = rep.sub(0.0, 0.85)  # registration takes seconds, TRELLIS minutes

    def worker_progress(frac: float, msg: str = "") -> None:
        gen(frac, f"{label}: {msg or 'generating the shape'}")

    gen(0.0, f"{label}: generating the shape")
    params = {
        "seed": t.seed,
        "ss_steps": steps[0],
        "slat_steps": steps[1],
        "glb": False,  # Gaussians only: baking the textured mesh would add a minute per try
        "ply": True,
        "name": f"back{t.index}",
    }
    outs = R._run_addon(
        run,
        item,
        TRELLIS_ADDON,
        _TRELLIS_TASK,
        {"image": str(image)},
        params,
        ProgressReporter(worker_progress),
    )
    R._addon_warnings(run, title, outs)
    ply = outs.get("ply")
    if not (ply and Path(ply).is_file()):
        raise AddonError(
            f"{title} finished but did not return its 3D Gaussians.",
            hint=f"Details are in the log ({UI_LOG}). Try again, or reinstall the add-on in the {UI_ADDONS}.",
        )
    cloud = read_splats(ply)
    run.check()
    t.sc, t.diag = register_cloud_to_front(
        cloud, ctx.front, ctx.canon, progress=rep.sub(0.85, 1.0), cancel=run.cancel
    )


def _photo_gradient(rgb: np.ndarray) -> np.ndarray:
    """Gradient magnitude of the (slightly blurred) grey photo on the mould grid, flat."""
    import cv2

    g = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy).ravel()


def _reliable_cells(front: FrontModel, seam_deg: float) -> np.ndarray:
    """Flat bool over the mould grid: photographed cells whose colour is trustworthy for
    the colour fit - inside the matte, facing the camera within ``seam - 10`` deg (where
    the photo is kept anyway), away from the outline (edge pixels mix in the background and
    TRELLIS's edges are soft) and among the smoothest 30 % of the photo (a cell or two of
    registration error must not pair two different colours)."""
    grid = front.grid
    D = np.asarray(front.D_f, dtype=np.float64).ravel()
    cos = np.nan_to_num(np.asarray(front.cos_theta, dtype=np.float64).ravel(), nan=-1.0)
    dt = np.asarray(front.dt, dtype=np.float64).ravel()
    rim = max(3.0, _COLOR_RIM_FRAC * max(grid.Hm, grid.Wm))
    cos_min = math.cos(math.radians(max(0.0, seam_deg - 10.0)))
    good = np.asarray(front.matte, dtype=bool).ravel() & np.isfinite(D) & (cos >= cos_min) & (dt >= rim)
    if good.any():
        grad = _photo_gradient(front.rgb)
        good &= grad <= np.percentile(grad[good], _COLOR_GRAD_PCT)
    return good


def _surface_colours(front: FrontModel, sc: SplatCloud, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Opacity-weighted mean colour per mould cell of the opaque Gaussians of ``sc`` (frame C)
    that lie on the photographed surface (``|z - D_f| <= tau``) of a cell in ``cells``;
    ``(means (n_cells, 3), weights (n_cells,))``, weight 0 where there is none."""
    n_cells = int(front.grid.Hm) * int(front.grid.Wm)
    P = np.asarray(sc.means, dtype=np.float64)
    op = np.nan_to_num(np.asarray(sc.opacity, dtype=np.float64), nan=0.0)
    col = np.asarray(sc.colors, dtype=np.float64)
    u, v, z = front.cam.project(P)
    cell = front.grid.cell_of(u, v)
    idx = np.flatnonzero((cell >= 0) & (op >= _TRELLIS_OPAQUE) & np.isfinite(col).all(axis=1))
    idx = idx[cells[cell[idx]]]
    D = np.asarray(front.D_f, dtype=np.float64).ravel()[cell[idx]]
    idx = idx[np.abs(z[idx] - D) <= depth_tol(D)]
    c, w = cell[idx], op[idx]
    wsum = np.bincount(c, weights=w, minlength=n_cells)
    sums = np.stack([np.bincount(c, weights=w * col[idx, k], minlength=n_cells) for k in range(3)], axis=1)
    return sums / np.maximum(wsum, 1e-12)[:, None], wsum


def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _channel_gain(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-channel gain ``dst ~= g * src`` in LINEAR RGB between paired sRGB colours: the
    median ratio over the pairs whose source channel is not dark; a channel with fewer than
    8 such pairs takes the measured channels' geometric mean (exposure is common to all
    channels), 1 when none is measured.  Clamped like the affine's diagonal."""
    s, d = _srgb_to_linear(src), _srgb_to_linear(dst)
    g = np.full(3, np.nan)
    for c in range(3):
        m = s[:, c] >= _COLOR_GAIN_MIN
        if int(m.sum()) >= 8:
            g[c] = float(np.median(d[m, c] / s[m, c]))
    known = np.isfinite(g) & (g > 0)
    g = np.where(known, g, float(np.exp(np.log(g[known]).mean())) if known.any() else 1.0)
    return np.clip(g, *_COLOR_GAIN_RANGE)


def _colour_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.median(np.abs(a - b).sum(axis=1)))


def _trellis_colours(front: FrontModel, sc: SplatCloud, seam_deg: float) -> tuple[SplatCloud, dict[str, Any]]:
    """Design 3.5.5 "Colour": TRELLIS paints its shape from the photo, but softer and with
    its own exposure / white balance.  Pairs of mould cells where TRELLIS lies on the
    photographed surface (:func:`_reliable_cells`): TRELLIS's mean colour there -> SHARP's
    mean colour there (both opacity-weighted).  The map is fitted in two stages and applied
    to every TRELLIS Gaussian, so the made-up back carries the photo's look:

    1. a per-channel gain in linear RGB (:func:`_channel_gain`) - exposure and white
       balance scale linear light, and a gain carries over to colours the photo did not
       show (black hair on the back stays black);
    2. the V4 affine map (Huber, ridge toward the identity, clamped -
       :func:`~vrphoto.geometry.register.color_affine_fit`) on the gain-corrected pairs, for
       what a gain cannot explain.

    (An addition to the design, which fits the affine alone: on a front of one main colour
    - one outfit - its ridge keeps ``A`` at the identity in the directions the pairs do not
    span and the free offset ``b`` absorbs the exposure difference instead, which then
    tints every other colour of the back by up to the 0.12 clamp.)  Kept only when it
    lowers the pairs' median colour error.  Only the DC colour changes (TRELLIS v1 writes no
    higher SH bands)."""
    good = _reliable_cells(front, seam_deg)
    t_col, t_w = _surface_colours(front, sc, good)
    s_col, s_w = _surface_colours(front, front.sc, good)
    pair = (t_w > 0) & (s_w > 0)
    n = int(pair.sum())
    info: dict[str, Any] = {"pairs": n, "ok": False}
    if n < _COLOR_MIN_PAIRS:
        info["reason"] = f"too few photographed cells to compare ({n})"
        return sc, info
    src, dst = t_col[pair], s_col[pair]
    g = _channel_gain(src, dst)
    G = np.diag(g)
    zero = np.zeros(3)
    A2, b2, ok2 = color_affine_fit(apply_color_affine(src.astype(np.float32), G, zero), dst)
    A, b = (A2 @ G, b2) if ok2 else (G.copy(), zero)
    A[np.diag_indices(3)] = np.clip(np.diag(A), *_COLOR_GAIN_RANGE)  # the two stages together
    before = _colour_error(src, dst)
    after = _colour_error(apply_color_affine(src.astype(np.float32), A, b), dst)
    info.update(
        gain=np.round(g, 4).tolist(),
        A=np.round(A, 4).tolist(),
        b=np.round(b, 4).tolist(),
        affine=bool(ok2),
        median_error_before=round(before, 4),
        median_error_after=round(after, 4),
    )
    if not after < before:
        info["reason"] = "the colours already match"
        return sc, info
    info["ok"] = True
    colors = apply_color_affine(np.asarray(sc.colors, dtype=np.float32), A, b)
    return replace(sc, colors=colors.astype(np.float32)), info


def _cloud_back_z(front: FrontModel, sc: SplatCloud) -> np.ndarray:
    """``back_z`` (design 3.5.5): per mould cell of the matte, the largest ``z`` of the opaque
    Gaussians of ``sc`` behind the photographed surface (``z > D_f + tau``) whose centres
    project into it.  The closure under-layer
    (:func:`~vrphoto.geometry.splat_merge.add_underlayer`) is clipped to stay in front of it.

    A generated cloud is sparser than the mould grid (TRELLIS: about one Gaussian per few
    cells on each side), so the matte cells without one get the push-pull fill of their
    neighbours: left unclipped, the under-layer there would stand out behind the cloud in
    the built-in back's colours, and a real hole in the cloud's back is closed at the depth
    of the back around it.  NaN outside the matte (everywhere when no Gaussian is behind
    the front)."""
    from ..core.fill import push_pull_fill

    grid = front.grid
    Hm, Wm = int(grid.Hm), int(grid.Wm)
    matte = np.asarray(front.matte, dtype=bool)
    P = np.asarray(sc.means, dtype=np.float64)
    op = np.nan_to_num(np.asarray(sc.opacity, dtype=np.float64), nan=0.0)
    P = P[np.isfinite(P).all(axis=1) & (op >= _TRELLIS_OPAQUE)]
    u, v, _ = front.cam.project(P)
    cell = grid.cell_of(u, v)
    ok = cell >= 0
    ok[ok] = matte.ravel()[cell[ok]]
    D = np.asarray(front.D_f, dtype=np.float64).ravel()
    with np.errstate(invalid="ignore"):
        ok[ok] = P[ok, 2] > D[cell[ok]] + depth_tol(D[cell[ok]])
    z = np.full(Hm * Wm, -np.inf)
    np.maximum.at(z, cell[ok], P[ok, 2])
    z = z.reshape(Hm, Wm)
    known = np.isfinite(z)
    if not known.any():
        return np.full((Hm, Wm), np.nan, np.float32)
    filled = push_pull_fill(np.where(known, z, 0.0).astype(np.float32), known)
    return np.where(matte, filled, np.nan).astype(np.float32)


def _src_trellis(ctx: Ctx, rep: ProgressReporter) -> BackPart:
    """The TRELLIS back (design 3.5.5): the TRELLIS v1 add-on builds the whole subject in 3D
    from one picture; its Gaussians are lined up behind SHARP's front and only what the
    photo did not see survives the merge (a ``"cloud"`` part, closed by the under-layer).

    1. Input: the levelled orthographic picture of the subject, or the plain cut-out
       (:func:`_trellis_input`), saved as ``trellis_input.png`` in the work folder.
    2. ``Opts.tries`` attempts with seeds ``seed, seed + 1, ...`` and the quality's
       sparse-structure / latent steps (25 / 25 for best, else 12 / 12; 'AI steps' sets
       both); Gaussians only (``glb=False, ply=True``).  Each cloud is registered by
       :func:`~vrphoto.geometry.register.register_cloud_to_front` (axis convention, yaw,
       scale and position searched, then gated).  The attempt with the best registration
       score among those passing the gates is kept.  An add-on failure (out of memory, a
       crash) ends the attempts, since another would most likely fail the same way.
    3. Colours matched to the photo (:func:`_trellis_colours`) when ``color_match``.
    4. ``back_z`` per mould cell (:func:`_cloud_back_z`) for the under-layer.

    TRELLIS's non-commercial notice is added only when its back is returned.  Returns
    ``ok=False`` with a plain reason when no attempt lined up; raises the add-on's error when
    no attempt produced a cloud at all (and :class:`AddonNotInstalled` when TRELLIS is not
    installed), so the chain warns and falls back."""
    from ..core.imaging import save_image

    run, item, o = ctx.run, ctx.item, ctx.opts
    manifest = R._require_addon(TRELLIS_ADDON)  # not installed -> the chain warns and falls back
    q = QUALITY.get(o.quality, QUALITY["balanced"])
    steps = (o.steps, o.steps) if o.steps else q.trellis_steps
    n_tries = max(1, int(o.tries))
    rep(0.0, "TRELLIS: preparing the picture")
    rgba, kind = _trellis_input(ctx)
    image = save_image(run.work_dir(item) / "trellis_input.png", rgba)
    run.check()
    tries: list[_Try] = []
    first_error: BaseException | None = None
    span = 0.96 / n_tries
    for i in range(n_tries):
        t = _Try(index=i + 1, seed=o.seed + i)
        tries.append(t)
        t0 = time.perf_counter()
        try:
            _trellis_try(
                ctx, manifest.title, image, t, steps, n_tries, rep.sub(0.02 + i * span, 0.02 + (i + 1) * span)
            )
        except Cancelled:
            raise  # Cancelled is a VRPhotoError: never an attempt that "failed"
        except AddonNotInstalled:
            raise
        except (AddonError, VRPhotoError, MemoryError, OSError) as e:
            t.error = str(e) or type(e).__name__
            first_error = first_error or e
            log.warning("TRELLIS attempt %d/%d failed: %s", t.index, n_tries, t.error)
            if isinstance(e, AddonError):
                break
        finally:
            t.seconds = time.perf_counter() - t0
        run.check()
    passed = [t for t in tries if t.sc is not None]
    registered = [t for t in tries if t.diag]
    summary = [t.summary() for t in tries]
    log.info("TRELLIS attempts: %s", summary)
    if not passed:
        if not registered:  # no attempt produced a cloud: the chain names the add-on's own error
            raise first_error or AddonError(f"{manifest.title} made no 3D shape.")
        best = max(registered, key=lambda t: t.score)
        why = str(best.diag.get("reason") or "it could not be lined up")
        # count the shapes that were made: an attempt that crashed made none (and is named,
        # so the advice to try another Variation is not followed for an out-of-memory error)
        n_made = len(registered)
        lead = "its shape did not line up" if n_made == 1 else f"none of its {n_made} shapes lined up"
        crashed = "".join(f"; attempt {t.index} failed ({t.error})" for t in tries if t.error)
        return BackPart(
            sc=_empty_cloud(),
            kind="cloud",
            source="trellis",
            ok=False,
            reason=f"{lead} with the photo: {why}{crashed}",
            diag={"tries": summary, "input": kind, "steps": list(steps), "best": best.diag},
        )
    best = max(passed, key=lambda t: t.score)
    assert best.sc is not None
    sc = best.sc
    fin = np.isfinite(np.asarray(sc.means)).all(axis=1) & np.isfinite(np.asarray(sc.scales)).all(axis=1)
    if not fin.all():
        sc = subset_splats(sc, fin)
    rep(0.98, "Matching the TRELLIS colours to the photo")
    diag: dict[str, Any] = {
        **best.diag,
        "tries": summary,
        "chosen": best.index,
        "seed": best.seed,
        "steps": list(steps),
        "input": kind,
    }
    if o.color_match:
        sc, diag["color"] = _trellis_colours(ctx.front, sc, o.seam)
    else:
        diag["color"] = {"ok": False, "reason": "off ('Match colours to the photo')"}
    run.check()
    back_z = _cloud_back_z(ctx.front, sc)
    diag["back_cells"] = int(np.isfinite(back_z).sum())
    crashed = [t for t in tries if t.error]
    if crashed:
        t = crashed[0]
        diag["warning"] = (
            f"TRELLIS attempt {t.index} of {n_tries} failed ({t.error}); attempt {best.index} was used."
        )
    run.addon_notice(item, manifest)  # its output is in the result from here on
    log.info(
        "TRELLIS back: attempt %d/%d (seed %d), %d Gaussians, IoU %.3f, residual %.1f cm, colours %s",
        best.index,
        n_tries,
        best.seed,
        len(sc),
        float(best.diag.get("iou", math.nan)),
        100.0 * float(best.diag.get("resid_m", math.nan)),
        "matched" if diag["color"].get("ok") else "kept",
    )
    rep(1.0, "TRELLIS back ready")
    return BackPart(sc=sc, kind="cloud", source="trellis", back_z=back_z, diag=diag)


# --------------------------------------------------------------------------- your own back photo (design 3.5.4)
_LEVELLED_PX = 512  # the levelled frame the views are lined up in (design 3.4)
_LEVELLED_REF_PX = 1024


def _view_segmenter(run: Any) -> Any:
    """The app's segmenter (``common.segment_model``) for the back / side photos, or None
    when it cannot be created (the views' flat background is then keyed out)."""
    try:
        return run.ctx.segmenter(run.common.segment_model, run.common.device)
    except Cancelled:
        raise
    except Exception as e:  # a model problem must not cost the user's back photo
        log.warning("SHARP 360: no segmenter for the back photos (%s); keying their background", e)
        return None


def _relief_fn(ctx: Ctx) -> tuple[str, Callable[[np.ndarray], Any] | None, list[str]]:
    """``(relief mode, depth_fn, models used)`` for :func:`~vrphoto.geometry.views_lift.
    view_based_back`.  'Back detail' Automatic is the quality's :attr:`Quality.relief`
    (smooth with "fast", design 1.3); otherwise the app's depth model
    (``common.depth_model``) gives the disparity of the locked back view.  With the model-free
    heuristic estimator, "auto" quietly makes a smooth back and "depth" says why it is
    smooth (views_lift).  The ids of the models that ran are appended to the list, for the
    licence notice."""
    o, run = ctx.opts, ctx.run
    q = QUALITY.get(o.quality, QUALITY["balanced"])
    mode = o.relief
    if mode == "flat" or (mode == "auto" and q.relief == "flat"):
        return "flat", None, []
    used: list[str] = []

    def depth_fn(rgb: np.ndarray) -> Any:
        est = run.ctx.estimator(run.common.depth_model, run.common.device)
        res = run.ctx.depth(rgb, estimator=est, cancel=run.cancel)
        used.append(str(getattr(res, "model_id", "") or ""))
        return res

    return mode, depth_fn, used


def _src_images(ctx: Ctx, rep: ProgressReporter) -> BackPart:
    """'Your own back photo' (design 3.5.4 and the shared view pipeline 3.5.2): a picture of
    the subject from behind - a real photo or a back view made with any AI tool - plus the
    optional side photos next to it (``right``, ``left``, ``back_right``, ... see
    :func:`~vrphoto.geometry.views_lift.views_from_folder`).

    1. The views are read (long side <= 1024 px) and cut out by the app's segmenter (a flat
       background is keyed out when it finds nothing; a cut-out PNG keeps its own matte).
    2. The levelled orthographic picture of the photo (design 3.4) is the frame they are
       lined up in.
    3. :func:`~vrphoto.geometry.views_lift.view_based_back`: the back is fitted to the
       mirrored front outline and locked onto it by a thin-plate spline (a different pose is
       rejected), side photos give the orbit axis and the visual hull, the depth model's
       relief shapes the back (:func:`_relief_fn`), the Poisson mould builds it and every
       Gaussian takes its colour from the photos that see it best.  ``color_match``,
       ``stride``, ``seed`` and the quality's hole filling come from the options.

    Returns the part (``ok=False`` with a plain reason when the back photo does not fit, or
    cannot be read or shows no subject - the reason names the back photo's own problem - so
    the chain warns and uses the built-in back).  Side photos that cannot be read or show no
    subject are named in ``diag["warning"]`` / ``diag["dropped"]`` and left out of
    ``diag["files"]`` (the pictures in use).  The contact sheet of the lined-up views is
    left in ``ctx.previews["views"]`` whenever the levelled frame was made."""
    from ..geometry import views_lift as vl

    run, item, o = ctx.run, ctx.item, ctx.opts
    q = QUALITY.get(o.quality, QUALITY["balanced"])
    rep(0.0, "Reading your back photo")
    if not o.back_images:  # resolve_options requires it; a hand-made Opts may not
        raise InputError("No back photo was given ('Back photo').")
    entries = vl.views_from_folder(o.back_images)
    dropped: list[str] = []  # plain sentences: pictures that cannot be read or show no subject
    try:
        views = vl.load_views(entries, dropped=dropped)
    except InputError:
        if not dropped:
            raise
        views = []  # none could be read: said below, through the back photo's own entry
    run.check()
    rep(0.03, "Finding the subject in your back photo")
    seg = _view_segmenter(run) if views else None
    views = vl.segment_views(views, seg, cancel=run.cancel, dropped=dropped)
    if seg is not None and any(v.meta.get("matte") == "segmenter" for v in views):
        run.model_notice(item, str(getattr(seg, "resolved", None) or getattr(seg, "id", "") or ""))
    run.check()
    files = {f"{v.phi_deg:g}": str(v.meta.get("file", v.name)) for v in views}  # the pictures in use
    if not any(_is_back(v.phi_deg) for v in views):
        # the back photo's own problem, not the first side photo's
        back_file = next((p.name for phi, p in entries if _is_back(phi)), None)
        own = [d for d in dropped if back_file and f"({back_file})" in d]
        why = (own[0] if own else "there is no picture of the back").rstrip(".")
        return BackPart(
            sc=_empty_cloud(),
            kind="mould",
            source="images",
            ok=False,
            reason=why[:1].lower() + why[1:],
            diag={"views": [e[1].name for e in entries], "files": files, "dropped": dropped},
        )
    rep(0.12, "Levelling the photo")
    ortho = vl.levelled_ortho_input(
        ctx.photo.rgb,
        ctx.alpha,
        ctx.front,
        ctx.canon,
        size=_LEVELLED_PX,
        ref=_LEVELLED_REF_PX,
        progress=rep.sub(0.12, 0.25),
        cancel=run.cancel,
    )
    ctx.previews["views"] = (ortho, views)
    relief, depth_fn, used = _relief_fn(ctx)
    part = vl.view_based_back(
        ctx.front,
        ctx.canon,
        ortho,
        views,
        depth_fn=depth_fn,
        relief=relief,
        color_match=o.color_match,
        stride=o.stride,
        seed=o.seed,
        fill_holes=q.fill_holes,
        source="images",
        progress=rep.sub(0.25, 1.0),
        cancel=run.cancel,
    )
    if part.views:  # registered (or rejected) views: the contact sheet shows what happened
        ctx.previews["views"] = (ortho, list(part.views))
    diag = dict(part.diag or {})
    diag["files"] = files
    if dropped:
        diag["dropped"] = dropped
        if part.ok:  # an unreadable or empty side photo: say so next to the back's own notes
            notes = [*dropped, str(diag["warning"])] if diag.get("warning") else dropped
            diag["warning"] = " ".join(dict.fromkeys(notes))
    part.diag = diag
    if part.ok and (diag.get("relief") or {}).get("source") == "depth" and used:
        run.model_notice(item, used[-1])  # the depth model's output shapes the result
    return part


#: Back sources by id: ``fn(ctx, rep) -> BackPart`` in frame C.  A source raises (or returns
#: ``ok=False`` with a plain-language ``reason``) when it cannot make a back.
SOURCES: dict[str, Callable[[Ctx, ProgressReporter], BackPart]] = {
    "shell": _src_shell,
    "trellis": _src_trellis,
    "images": _src_images,
}


def _name(src: str) -> str:
    return SOURCE_NAMES.get(src, src)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def run_sources(ctx: Ctx, rep: ProgressReporter, attempts: list[dict[str, Any]]) -> BackPart | None:
    """Try ``ctx.opts.sources`` in order and return the first usable back (None: none worked).

    Every failure is one plain warning naming what is used instead; ``attempts`` gets one
    JSON-safe record per source tried (for debug.json)."""
    run = ctx.run
    srcs = ctx.opts.sources
    for i, src in enumerate(srcs):
        nxt = _name(srcs[i + 1]) if i + 1 < len(srcs) else "the photo front alone"
        who = _cap(_name(src))
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"source": src, "ok": False}
        attempts.append(rec)
        fn = SOURCES.get(src)
        if fn is None:  # named by a job file from a newer version
            rec["reason"] = "not available in this version"
            run.warn(f"{who} is not available in this version; {nxt} was used instead.")
            continue
        try:
            part = fn(ctx, rep)
            rec["seconds"] = round(time.perf_counter() - t0, 2)
            if part is not None and part.ok and len(part.sc):
                rec["ok"] = True
                return part
            reason = (part.reason if part is not None else "") or "it came out empty"
            rec["reason"] = reason
            if part is not None and part.diag:  # e.g. why a TRELLIS shape did not line up
                rec["diag"] = part.diag
            run.warn(f"{who} could not make a usable back ({reason}); {nxt} was used instead.")
        except Cancelled:
            raise  # Cancelled is a VRPhotoError: never a fallback
        except AddonNotInstalled as e:
            rec["reason"] = f"not installed: {e}"
            run.warn(f"{who} is not installed; {nxt} was used instead.")
        except (AddonError, VRPhotoError, MemoryError, OSError) as e:
            rec["reason"] = str(e) or type(e).__name__
            run.warn(f"{who} could not make the back ({rec['reason']}); {nxt} was used instead.")
        except Exception as e:  # a bug in the fusion code: degrade, keep the traceback
            log.exception("SHARP 360 back source %s failed", src)
            rec["reason"] = f"{type(e).__name__}: {e}"
            run.warn(f"{who} failed unexpectedly ({type(e).__name__}); {nxt} was used instead.")
    return None


# --------------------------------------------------------------------------- stitching and export
def _stitch(
    ctx: Ctx, part: BackPart | None, rep: ProgressReporter, dbg: dict[str, Any]
) -> tuple[SplatCloud, np.ndarray]:
    """Merge the back into the front, close cloud backs, feather colours, drop floaters."""
    run, front, canon, o = ctx.run, ctx.front, ctx.canon, ctx.opts
    rep(0.0, "Stitching front and back")
    merged, is_front, dbg["merge"] = merge_parts(
        front,
        part,
        seam_deg=o.seam,
        seed=o.seed,
        budget=R.MAX_OBJECT_SPLATS,
        cancel=run.cancel,
        progress=rep.sub(0.0, 0.45),
    )
    if part is not None and part.kind == "cloud":
        rep(0.45, "Closing holes in the back")
        merged, is_front = add_underlayer(merged, is_front, front, canon, part, cancel=run.cancel)
    rep(0.5, "Matching colours")
    merged = feather_colors(merged, is_front, cancel=run.cancel)
    run.check()
    rep(0.6, "Removing floaters")
    merged, is_front, dbg["floaters"] = remove_floaters(
        merged,
        is_front,
        front,
        canon,
        hull=None if part is None else part.hull,
        seam_deg=o.seam,
        spacing_m=None if part is None else part.spacing_m,
        cancel=run.cancel,
        progress=rep.sub(0.6, 1.0),
    )
    return merged, is_front


@dataclass
class _Export:
    sc: SplatCloud  # final cloud (glTF frame, feet at y = 0, life-size, cropped, decimated)
    front: SplatCloud  # its front rows before decimation (save_extras)
    back: SplatCloud  # its generated rows before decimation (save_extras)
    height: float  # final height (m)
    size_mode: str  # height | photo
    info: dict[str, Any] = field(default_factory=dict)


def _export_cloud(
    sc: SplatCloud,
    is_front: np.ndarray,
    canon: Canon,
    o: Opts,
    height: float,
    run: Any,
    rep: ProgressReporter,
) -> _Export:
    """Design 3.12: glTF frame -> upright about ``canon.up_C`` (floor at y = 0) -> life size ->
    box crop -> decimate to the viewer budget.  Every step before the decimation keeps the
    row order, so the front / back split of ``is_front`` still applies.

    The box crop removes generated rows only: the front rows are SHARP's cleaned subject
    (matte- and depth-filtered already), and the percentile box would cut thin parts held
    out from the body (a cane, a pointing hand, fingertips above a raised arm) out of the
    protected photo front."""
    P = np.asarray(sc.means)
    ok = np.isfinite(P).all(axis=1) & np.isfinite(np.asarray(sc.scales)).all(axis=1)
    if not ok.all():  # normalize_orientation would drop them and break the row order
        sc, is_front = subset_splats(sc, ok), np.asarray(is_front)[ok]
    rep(0.0, "Making it life-size")
    g = R._splats_cv_to_gltf(sc)
    g = normalize_orientation(g, up=cv_to_gltf(np.asarray(canon.up_C, dtype=np.float64)))
    cut = bool(canon.cut_top or canon.cut_bottom)
    g, h, box, used = to_life_size(g, mode=o.size_from, height_m=height, cut=cut)
    isf = np.asarray(is_front, dtype=bool)
    P = np.asarray(g.means, dtype=np.float64)
    in_box = np.all(P >= box[0], axis=1) & np.all(P <= box[1], axis=1)
    keep = isf | in_box
    main = subset_splats(g, keep)
    front_part = subset_splats(g, isf)
    back_part = subset_splats(g, keep & ~isf)
    run.check()
    n_crop = len(main)
    main = decimate_splats(main, max_n=R.MAX_OBJECT_SPLATS, progress=rep.sub(0.3, 1.0), cancel=run.cancel)
    info = {
        "size_from": o.size_from,
        "size_used": used,
        "height_m": h,
        "box": np.asarray(box).tolist(),
        "front_outside_box": int((isf & ~in_box).sum()),
        "after_crop": n_crop,
        "final": len(main),
    }
    return _Export(main, front_part, back_part, h, used, info)


def _cut_where(canon: Canon) -> str:
    if canon.cut_top and canon.cut_bottom:
        return "at the top and bottom"
    return "at the top" if canon.cut_top else "at the bottom"


# --------------------------------------------------------------------------- debug data
def _jsonable(x: Any) -> Any:
    """``x`` with numpy scalars / small arrays, paths and non-finite floats made JSON-safe."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist()) if x.size <= 64 else f"<array {x.shape}>"
    if isinstance(x, np.generic):
        return _jsonable(x.item())
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, bool | int | str) or x is None:
        return x
    if isinstance(x, Path):
        return str(x)
    return str(x)


def _canon_info(canon: Canon) -> dict[str, Any]:
    return {
        "up_source": canon.up_source,
        "up_C": canon.up_C,
        "pitch_deg": canon.pitch_deg,
        "height_s": canon.height_s,
        "width_s": canon.width_s,
        "T_prior": canon.T_prior,
        "z_axis": canon.z_axis,
        "cut_top": canon.cut_top,
        "cut_bottom": canon.cut_bottom,
        "diag": canon.diag,
    }


def _write_debug(path: Path, dbg: dict[str, Any]) -> Path:
    data = json.dumps(_jsonable(dbg), indent=2, ensure_ascii=False).encode("utf-8")
    atomic_write_bytes(path, data)
    return path


# --------------------------------------------------------------------------- the mode
def mode_sharp360(
    run: Any,
    item: Any,
    photo: Any,
    alpha: FloatMap,
    spec: Any,
    *,
    placement: str,
    height: float,
    rep: ProgressReporter,
    opts: Opts | None = None,
) -> None:
    """Run SHARP 360 for one input once the subject's matte ``alpha`` (photo resolution) is
    known; writes the outputs and the viewer scene (see the module docstring).

    ``opts`` (an optional addition) are options already checked by :func:`resolve_options`
    (``run._mode_object`` validates them before any work); None resolves them here.
    Raises :class:`AddonError` when SHARP fails or returns nothing (as the plain ``sharp``
    method) and :class:`Cancelled`; every later failure becomes a warning."""
    o = opts if opts is not None else resolve_options(spec, installed=R._addon_installed)
    t_start = time.perf_counter()
    times: dict[str, float] = {}
    dbg: dict[str, Any] = {"version": 1, "options": asdict(o)}

    def lap(name: str, t0: float) -> float:
        t = time.perf_counter()
        times[name] = round(t - t0, 3)
        return t

    # ---- 1. SHARP (fails the job like the plain sharp method)
    manifest = R._addon_manifest("sharp")
    run.addon_notice(item, manifest)
    if o.thickness_asked is not None:
        lo, hi = THICKNESS_RANGE
        run.warn(
            f"SHARP 360° used a Thickness of {o.thickness:.2f} instead of {o.thickness_asked:.2f}: its "
            f"life-size back allows {lo:.2f}-{hi:.2f} (0.08-0.15 suits most people; the presets set "
            "Thickness for Billboard)."
        )
    R._save_cutout(run, item, photo, alpha)
    image_in, params = R._sharp_inputs(run, item, photo)
    rep(0.10, "Running SHARP (front)")
    t = time.perf_counter()
    outs = R._run_addon(
        run, item, "sharp", "image_to_splat", {"image": str(image_in)}, params, rep.sub(0.10, 0.25)
    )
    R._addon_warnings(run, manifest.title, outs)
    ply = outs.get("ply") or outs.get("splat")
    if not (ply and Path(ply).is_file()):
        raise AddonError(
            f"{manifest.title} finished but did not return a 3D model.",
            hint=f"Details are in the log ({UI_LOG}). Try again, or reinstall the add-on in the {UI_ADDONS}.",
        )
    frame = str(outs.get("frame", "opencv") or "opencv").lower()
    if frame != "opencv":
        log.warning("SHARP reported frame %r; SHARP 360 assumes OpenCV camera coordinates", frame)
    sc_full = read_splats(ply)
    if not len(sc_full):
        raise AddonError(
            f"{manifest.title} returned an empty 3D result.",
            hint=f"Try again; if it keeps happening, reinstall the add-on in the {UI_ADDONS}.",
        )
    t = lap("sharp", t)
    run.check()

    # ---- 2. clean the front, find the upright frame
    rep(0.25, "Cleaning the SHARP front (removing background layers and edge streaks)")
    try:
        cam = SharpCam.from_outputs(outs, params, n_raw=len(sc_full), shape=alpha.shape)
        dbg["camera"] = {k: getattr(cam, k) for k in ("f", "width", "height", "n_raw", "grid", "layers")}
        front = sharp_front(
            sc_full,
            alpha,
            cam,
            photo.rgb,
            mould_long=o.mould_long,
            progress=rep.sub(0.25, 0.29),
            cancel=run.cancel,
        )
        dbg["front"] = {"kept": len(front.sc), "stats": front.stats}
        rep(0.29, "Finding the subject's upright direction")
        canon = canonical_frame(front, sc_full, alpha, thickness=o.thickness)
        dbg["canon"] = _canon_info(canon)
    except Cancelled:
        raise
    except Exception as e:
        log.exception("SHARP 360: cleaning SHARP's result failed")
        why = str(e) if isinstance(e, VRPhotoError) else type(e).__name__
        run.warn(
            f"SHARP 360 could not clean SHARP's result ({why}); plain SHARP's subject was saved instead, "
            "without a back."
        )
        dbg["front_failed"] = f"{type(e).__name__}: {e}"
        _plain_fallback(run, item, photo, alpha, sc_full, outs, params, o, placement, height, rep, dbg)
        return
    for w in list(front.stats.get("warnings") or []) + list(canon.diag.get("warnings") or []):
        run.warn(str(w))
    t = lap("front", t)
    run.check()

    # ---- 3. the back
    ctx = Ctx(run, item, photo, alpha, sc_full, front, canon, o)
    attempts: list[dict[str, Any]] = []
    part = run_sources(ctx, rep.sub(0.30, 0.72), attempts)
    dbg["sources"] = attempts
    if part is not None:
        dbg["back"] = {"source": part.source, "kind": part.kind, "n": len(part.sc), "diag": part.diag or {}}
        warning = (part.diag or {}).get("warning")
        if warning:
            run.warn(str(warning))
    t = lap("back", t)
    run.check()

    # ---- 4. stitch (never fails the job: falls back to the cleaned front)
    stitched = True
    try:
        merged, is_front = _stitch(ctx, part, rep.sub(0.72, 0.88), dbg)
    except Cancelled:
        raise
    except Exception as e:
        log.exception("SHARP 360: stitching failed")
        run.warn("Stitching the front and back failed; the cleaned SHARP front was saved instead.")
        dbg["stitch_failed"] = f"{type(e).__name__}: {e}"
        merged, is_front, stitched = front.sc, np.ones(len(front.sc), bool), False
    t = lap("stitch", t)

    # ---- 5. export (an export problem with the back retries with the front alone)
    try:
        ex = _export_cloud(merged, is_front, canon, o, height, run, rep.sub(0.88, 0.92))
    except Cancelled:
        raise
    except Exception as e:
        if not stitched or part is None:
            raise
        log.exception("SHARP 360: export with the back failed")
        run.warn("The back could not be added to the final splat; the cleaned SHARP front was saved instead.")
        dbg["export_failed"] = f"{type(e).__name__}: {e}"
        stitched = False
        ex = _export_cloud(front.sc, np.ones(len(front.sc), bool), canon, o, height, run, rep.sub(0.88, 0.92))
    if o.size_from == "auto" and ex.size_mode == "photo":
        run.warn(
            f"The subject is cut off {_cut_where(canon)} of the photo, so SHARP's own size estimate was used "
            f"({ex.height:.2f} m). Set 'Size from' to 'Real-world height' to force your height."
        )
    dbg["export"] = ex.info
    used_back = part is not None and stitched and len(ex.back) > 0
    what = "front only"
    if used_back and part is not None:
        what = SOURCE_LABELS.get(part.source, f"back by {_name(part.source)}")
    dst = run.out_path(item, "_sharp360", "splat")
    write_splats(ex.sc, dst)
    run.add_output(item, f"3D object (Gaussian splat, SHARP 360°, {what})", dst)
    t = lap("export", t)

    # ---- 6. previews (never fail the job)
    dbg["previews"] = _write_previews(run, item, ctx, ex.sc, o, rep.sub(0.92, 0.97))
    t = lap("previews", t)

    # ---- 7. extras, debug data, viewer scene
    times["total"] = round(time.perf_counter() - t_start, 3)
    dbg["seconds"] = times
    dbg["warnings"] = list(run.result.warnings)
    _write_extras(run, item, ex, o, used_back)
    _save_debug(run, item, dbg, o.save_extras)
    log.info(
        "SHARP 360 %s: %d SHARP Gaussians -> front %d, back %d (%s), final %d, height %.2f m (%s), %.1f s",
        item.path.name,
        len(sc_full),
        len(front.sc),
        0 if part is None else len(part.sc),
        what,
        len(ex.sc),
        ex.height,
        ex.size_mode,
        times["total"],
    )
    # the viewer shows the splat 1:1 (it is already life-size); height_m only names the height
    block = {"asset": "object.splat", "format": "splat", "height_m": round(float(ex.height), 4), "scale": 1.0}
    _scene(run, item, photo, dst, {**block, "pivot": "base", "placement": placement}, rep)


def _scene(
    run: Any, item: Any, photo: Any, splat: Path, block: dict[str, Any], rep: ProgressReporter
) -> None:
    R._export_scene(
        run,
        item,
        mode="object",
        block=block,
        assets={"object.splat": splat},
        thumbnail=photo.rgb,
        env=dict(R.OBJECT_ENV),
        rep=rep.sub(0.97, 1.0),
    )
    rep(1.0, "Done")


#: Views of the turntable preview (design 1.5: a 2 x 4 sheet at 0, 45, ..., 315 degrees).
PREVIEW_VIEWS = 8
_PREVIEW_COLS = 4
_PREVIEW_JPEG_QUALITY = 88
_VIEWS_TILE_PX = 256  # one picture of the back photos' contact sheet


def _why(e: BaseException) -> str:
    """A failure in a few plain words: the message of our own errors, else the type."""
    return (str(e) or type(e).__name__) if isinstance(e, VRPhotoError | OSError) else type(e).__name__


def _write_previews(
    run: Any, item: Any, ctx: Ctx, sc: SplatCloud, o: Opts, rep: ProgressReporter
) -> dict[str, Any]:
    """Design 1.5 / 3.13: the preview pictures, written next to the splat.

    * ``<base>_sharp360_preview.jpg`` - a turntable sheet of the final splat ``sc`` (glTF
      frame, upright, facing +z): :data:`PREVIEW_VIEWS` orthographic views around the
      vertical axis (0 = the photo's side, 90 = from the subject's left, 180 = behind) of
      :attr:`Quality.preview_px` pixels each, rendered on the CPU
      (:func:`~vrphoto.geometry.splat_render.turntable_sheet`).
    * ``<base>_sharp360_views.png`` - only when a view-based source ran (``images``): its
      contact sheet (:func:`~vrphoto.geometry.views_lift.contact_sheet`), the levelled
      photo and every back / side photo with the outline it was expected to have, also when
      the photos could not be used (it shows why).

    A failure is one warning (the splat is already written); a cancel propagates.  Returns
    JSON-safe notes for debug.json."""
    from ..core.imaging import save_image

    info: dict[str, Any] = {}
    q = QUALITY.get(o.quality, QUALITY["balanced"])
    rep(0.0, "Rendering a turntable preview")
    t0 = time.perf_counter()
    try:
        from ..geometry.splat_render import turntable_sheet

        sheet = turntable_sheet(
            sc,
            n=PREVIEW_VIEWS,
            size=q.preview_px,
            cols=_PREVIEW_COLS,
            progress=rep.sub(0.0, 0.9),
            cancel=run.cancel,
        )
        p = run.out_path(item, "_sharp360_preview", "jpg")
        save_image(p, sheet, quality=_PREVIEW_JPEG_QUALITY)
        run.add_output(item, "SHARP 360° turntable preview (JPEG)", p)
        info["turntable"] = {
            "file": p.name,
            "views": PREVIEW_VIEWS,
            "px": q.preview_px,
            "seconds": round(time.perf_counter() - t0, 2),
        }
    except Cancelled:
        raise
    except Exception as e:  # a preview never costs the result
        log.exception("SHARP 360: the turntable preview failed")
        info["turntable"] = {"error": f"{type(e).__name__}: {e}"}
        run.warn(f"The turntable preview could not be made ({_why(e)}); the 3D result is not affected.")
    views = ctx.previews.get("views")
    if views is not None:
        rep(0.92, "Saving the contact sheet of your back photos")
        try:
            from ..geometry.views_lift import contact_sheet

            ortho, vlist = views
            p = run.out_path(item, "_sharp360_views", "png")
            save_image(p, contact_sheet(ortho, vlist, tile=_VIEWS_TILE_PX))
            run.add_output(item, "SHARP 360° back photos as lined up (PNG)", p)
            info["views"] = {"file": p.name, "views": len(vlist)}
        except Cancelled:
            raise
        except Exception as e:
            log.exception("SHARP 360: the contact sheet failed")
            info["views"] = {"error": f"{type(e).__name__}: {e}"}
            run.warn(
                f"The contact sheet of the back photos could not be saved ({_why(e)}); the 3D result is not "
                "affected."
            )
    rep(1.0, "Previews ready")
    return info


def _save_debug(run: Any, item: Any, dbg: dict[str, Any], as_output: bool) -> None:
    """debug.json in the work folder (always, for the log) and, with ``save_extras``, as an
    output.  Never fails the job."""
    try:
        _write_debug(run.work_dir(item) / "sharp360_debug.json", dbg)
        if as_output:
            dj = run.out_path(item, "_sharp360_debug", "json")
            _write_debug(dj, dbg)
            run.add_output(item, "SHARP 360° debug data (JSON)", dj)
    except OSError as e:
        log.warning("Could not write the SHARP 360 debug data: %s", e)


def _write_extras(run: Any, item: Any, ex: _Export, o: Opts, used_back: bool) -> None:
    """With ``save_extras``: the PLY and the front / back parts (same frame and size as the
    main splat).  Extras never fail the job."""
    if not o.save_extras:
        return
    try:
        ply = run.out_path(item, "_sharp360", "ply")
        write_splats(ex.sc, ply)
        run.add_output(item, "3D object (PLY, SHARP 360°)", ply)
        parts = [("_sharp360_front", ex.front, "SHARP 360° photo front only (Gaussian splat)")]
        if used_back:
            parts.append(("_sharp360_back", ex.back, "SHARP 360° back only (Gaussian splat)"))
        for kind, sc, label in parts:
            if len(sc) > R.MAX_OBJECT_SPLATS:
                sc = decimate_splats(sc, max_n=R.MAX_OBJECT_SPLATS, cancel=run.cancel)
            p = run.out_path(item, kind, "splat")
            write_splats(sc, p)
            run.add_output(item, label, p)
    except Cancelled:
        raise
    except (VRPhotoError, OSError) as e:
        run.warn(f"The extra SHARP 360 files could not all be saved: {e}")


def _plain_fallback(
    run: Any,
    item: Any,
    photo: Any,
    alpha: FloatMap,
    sc_full: SplatCloud,
    outs: dict[str, Any],
    params: dict[str, Any],
    o: Opts,
    placement: str,
    height: float,
    rep: ProgressReporter,
    dbg: dict[str, Any],
) -> None:
    """Last resort when the front cannot be cleaned: the plain ``sharp`` method's result
    (matte filter, fitted to the height by the viewer), saved under the SHARP 360 name."""
    rep(0.9, "Converting the splat")
    sc = R._subject_splats(run, sc_full, alpha, outs, params)
    sc = R._splats_cv_to_gltf(sc)
    sc = crop_splats(sc, radius_pct=98.0)
    sc = normalize_orientation(sc)
    sc = decimate_splats(sc, max_n=R.MAX_OBJECT_SPLATS, cancel=run.cancel)
    dst = run.out_path(item, "_sharp360", "splat")
    write_splats(sc, dst)
    run.add_output(item, "3D object (Gaussian splat, SHARP 360°, plain SHARP front)", dst)
    dbg["warnings"] = list(run.result.warnings)
    _save_debug(run, item, dbg, o.save_extras)
    block = {"asset": "object.splat", "format": "splat", "height_m": height, "pivot": "base"}
    _scene(run, item, photo, dst, {**block, "placement": placement}, rep)


__all__ = [
    "FINAL_SOURCE",
    "MAX_SEED",
    "MAX_STEPS",
    "MAX_TRIES",
    "PREVIEW_VIEWS",
    "QUALITY",
    "SEAM_RANGE",
    "SOURCES",
    "SOURCE_ADDONS",
    "SOURCE_LABELS",
    "SOURCE_NAMES",
    "THICKNESS_RANGE",
    "Ctx",
    "Opts",
    "Quality",
    "mode_sharp360",
    "resolve_options",
    "run_sources",
]
