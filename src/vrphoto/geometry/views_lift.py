"""SHARP 360: lift pictures of the subject's back and sides onto SHARP's photographed front.

The view-based back sources (docs/design/SHARP360.md sections 3.4, 3.5.2 V1-V11, 3.5.3,
3.5.4 and 3.13; frames: docs/CONTRACTS.md section 13).  Two kinds of views share the code
behind :class:`GenView`: NOMINAL views - the user's own back / side photos or AI edits made
elsewhere (the ``images`` source; Qwen / ComfyUI in M7), of which only the azimuth is known
- and ORTHOGRAPHIC views with exact cameras (``kind="ortho"``: MV-Adapter's six views,
:func:`views_from_cameras`, or a saved views folder, :func:`save_views_folder`).  Pure
numpy / OpenCV, CPU only; the depth model is injected (``depth_fn``) so this module stays
model-free.

**Levelled frame.**  Everything is lined up in one upright orthographic frame of ``S x S``
pixels with ``m`` metres per pixel (:func:`levelled_ortho_input`): the photo's visible front,
forward-splatted with SHARP's front depth and seen square-on from the horizontal (no camera
pitch, no perspective).  A point with S coordinates ``P = (x, y, z)`` (frame S of
:class:`~vrphoto.geometry.splat_fusion.Canon`) lands in a view at azimuth ``phi`` at

    ``u = (P . w(phi) - o) / m + S/2``,  ``v = (y - y0) / m + S/2``,  depth ``P . f(phi)``

with ``w(phi) = (cos phi, 0, sin phi)`` (the view's "right"), ``f(phi) = (-sin phi, 0,
cos phi)`` (its viewing direction), ``y0`` the levelled centre row's height and ``o`` the
view's horizontal offset (``-x0`` for the levelled input mirrored to the back, solved per
side view below).  Continuous coordinates as in design 3.0: pixel ``(i, j)`` spans
``u in [j, j+1)``, ``v in [i, i+1)``; OpenCV samplers get ``u - 0.5``.

**Pipeline** (:func:`view_based_back`):

* V1 :func:`segment_views` - the app's segmenter, else a flat-background key from a 4 px
  border ring; largest component, ``clean_matte``; generated views covering < 2 % and user
  photos whose subject is under 64 px across are dropped ("too small", not "no subject").
* V2/V3 the back (phi 180): its matte is fitted to the mirrored front silhouette by a
  symmetric-chamfer similarity (scale, shift, roll <= 5 deg), refined by a perspective
  stage (a camera turn: aspect and keystone, robust chamfer - a phone held above the
  subject's middle tilts down; :func:`~vrphoto.geometry.register.chamfer_perspective_fit`;
  gate IoU >= 0.75), then locked exactly onto it with a 64-point thin-plate spline.  It is
  rejected as a different pose when, after the global fit, 2 % of the outline is still
  more than 6 % of the subject's size away from the other outline
  (:func:`~vrphoto.geometry.register.outline_gap`).  Back silhouette = mirrored front
  silhouette is exact in orthographic projection.  Spline, perspective and similarity are
  composed into one resampling.
* V2 side / quarter views: invariant rows - at elevation 0 the subject's top and bottom
  rows are the same in every view - give scale and vertical shift; the column offset is
  left to V5.  A view (or photo) cut off at the top or bottom keeps the other row and takes
  the scale that makes its front edge follow the photo's best.
* V4 colour: Reinhard LAB transfer between the view's and the photo's rim bands (3 % of the
  size), strength 0.5, with the colour part bounded to a white-balance-sized shift (a back
  of another colour keeps it).
* V5 offsets: in a view at ``phi``, the silhouette edge facing the photo camera is formed
  by the photographed front, so per row ``o = r_f - (u_f - S/2) m`` with ``r_f`` the 2nd
  percentile of ``sign(sin phi) P.w`` over the front points of that row; the median over
  rows is the view's offset (row MAD gate 3 cm).  A view whose mirror hypothesis
  (``360 - phi``, the usual left/right naming mix-up) fits much better on its worst rows is
  turned round with a warning - a mirrored pair (right / left) both or neither, judged
  together; at most one view per azimuth; the back edges implied by the left and right
  views must agree.  A photo has no rotation centre of its own, so the orbit axis reported for nominal
  views is the middle of the subject's depth (photo front edge to side back edge).
* V6 :func:`hull_intervals` - per mould-grid ray, the first contiguous run inside the
  visual hull of the side / quarter views (all-but-one voting, 3 px dilation; a left / right
  view without its mirror partner has two votes, as the pair would).
* V7 back depth: with left/right views the back's per-row extent from their back edges,
  shaped across the row by the relief (``depth_fn`` disparity of the locked back view, else
  the silhouette's inflation profile); without them the shell prior plus bounded high-pass
  relief detail.  Never carves more than 40 % into its envelope (hull, else prior).
* V8 :func:`~vrphoto.geometry.back_mould.mould_back` - Poisson back + rim wall.
* V9 colours from the views, weight ``max(0, n . -f)^8``, sampled from images bled past
  their mattes; visibility by point z-buffer, each Gaussian a disc of its own projected
  footprint (1-4 px) and a depth tolerance that grows on slanted surfaces.  V10 side-view
  pixels nothing covers are lifted onto the hull entry along that view's ray; V11 a QC
  cost for choosing between tries (:func:`qc_cost`, :func:`choose_view_set`).

**Orthographic sets** (a back view of ``kind="ortho"``; design V2-V5 "mvadapter").  The
generator keeps the levelled input's framing: ``m = m_in`` (at the input's size), the orbit
axis on the centre column at ``x = x0``, rows centred on ``y0`` - so a view at ``phi`` has
the offset ``o = x0 cos(phi) + z_a sin(phi)`` with one unknown, the axis depth ``z_a``.

* V2 lateral check: view 0 against the levelled front silhouette; a symmetric-chamfer
  similarity (scale, shift) is fitted and applied to EVERY view (they share the framing)
  when it moves the outline by more than half a pixel and raises the overlap by a point;
  below IoU 0.75 the set is rejected.  Invariant rows: each view's top / bottom rows give
  its scale against view 0's; > 5 % off drops the view, otherwise it is rescaled about its
  centre column with the bottom row aligned (a cut-off end is not used as an anchor).
* V3 the back is locked onto the mirrored front silhouette (IoU gate 0.75, the outline-gap
  pose gate, thin-plate spline) as for nominal views, without the similarity / perspective
  stages (its camera is exact).
* V4 colour: one Huber affine in linear RGB (:func:`~vrphoto.geometry.register.
  color_affine_fit`) from view 0 to the levelled input on flat, confidently inside pixels,
  applied to every view (one exposure); the bounded rim-band Reinhard without view 0.
* V5 shared axis: ``z_a`` per row from the front edge of the +-90 views (else the +-45
  views), median over both, which must agree (``max(4 cm, 2 % H)``) and keep their row MAD
  small; sign safety net - with the azimuth sign reversed both views pick the back edge,
  so they still agree but their row deviations grow: when the reversed hypothesis costs less
  than half and SHARP's front rendered at the quarter views' azimuths agrees (design V5's
  second test), every view - the dropped ones too - is turned round (``phi -> 360 - phi``)
  with a warning; without that agreement the named sides are kept (warning).  Quarter views
  must line up with the solved axis.

``ok=False`` with a plain ``reason`` (no back view, outline mismatch, different pose)
makes the orchestrator fall back to the next source.
"""

from __future__ import annotations

import inspect
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..core.errors import Cancelled, InputError
from ..core.fill import push_pull_fill
from ..core.fsutil import atomic_write_bytes
from ..core.imaging import IMAGE_EXTS, load_image, save_image
from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter, SplatCloud
from ..segment.matte import clean_matte, to_alpha
from .back_mould import (
    DETAIL_M,
    MIN_GAP_M,
    BackPart,
    HullIntervals,
    _boundary,
    _front_maps,
    inflation_prior,
    mould_back,
)
from .billboard import _masked_blur, inflation_profile
from .mesh import _as_reporter, _check
from .register import (
    _linear_to_srgb,
    _similarity_affine,
    _srgb_to_linear,
    apply_color_affine,
    chamfer_perspective_fit,
    chamfer_similarity_fit,
    color_affine_fit,
    contour_points,
    nelder_mead,
    outline_gap,
    point_zbuffer,
    silhouette_iou,
    tps_fit,
)
from .splat import concat_splats, quat_to_rotmat, rotmat_to_quat
from .splat_fusion import Canon, FrontModel, _nearest_fill, depth_tol

log = get_logger(__name__)

#: ``GenView.kind`` values: exact orthographic cameras (MV-Adapter, M4) or a nominal azimuth only.
VIEW_KINDS = ("ortho", "nominal")
#: ``relief`` values of :func:`view_based_back` (``sharp`` joins in M6).
RELIEFS = ("auto", "depth", "flat")
#: Views-folder file names (whole stem, any case, ``_`` / ``-`` / space between words) and
#: their azimuth: "right" means you walked to your right (you see the subject's left side).
VIEW_NAMES: dict[str, float] = {
    "front_right": 45.0,
    "right": 90.0,
    "back_right": 135.0,
    "back": 180.0,
    "behind": 180.0,
    "rear": 180.0,
    "back_left": 225.0,
    "left": 270.0,
    "front_left": 315.0,
}
#: Canonical view name per azimuth (other azimuths are called ``azNNN``).  ``front`` (0) is
#: only a name for saved views: a folder's ``front`` picture is used through ``views.json``.
PHI_NAMES: dict[int, str] = {0: "front", 45: "front_right", 90: "right", 135: "back_right", 180: "back"}
PHI_NAMES.update({225: "back_left", 270: "left", 315: "front_left"})
#: Camera file of a saved views folder: azimuths, and the exact orthographic cameras.
VIEWS_JSON = "views.json"
#: ``format`` written into a saved ``views.json``.
VIEWS_FORMAT = "vrphoto-views"
#: The photo is downscaled to this long side before forward splatting.
MAX_PHOTO_LONG = 2048
#: Views are downscaled to this long side by default.
VIEW_MAX_LONG = 1024

# ---- levelled input (design 3.4)
_ALPHA_MIN = 0.02  # photo pixels with more alpha than this belong to the subject
_FILL = 0.9  # the subject's long side fills 90 % of the levelled picture
_FRONT_TOL_M = 0.02  # samples within this of a pixel's front z are averaged
_MAX_SAMPLES = 4_000_000  # forward-splat samples (supersampling is reduced above this)
_MAX_SUPERSAMPLE = 8
_SUPERSAMPLE_MARGIN = 1.3  # samples land < 1/1.3 px apart at the median depth
_EXTENT_PCT = 0.05  # robust extent: 0.05th .. 99.95th percentile of the subject's pixels
_SAMPLE_CHUNK = 1 << 20

# ---- V1 segmentation
_SEG_MIN_COVER = 0.02  # generated ("ortho") views fill their frame ...
_SEG_MIN_PX = 64  # ... a user's photo needs a subject this many pixels across ...
_SEG_MIN_AREA_PX = 400  # ... and this much area
_SEG_MAX_COVER = 0.97  # a "subject" covering more than this is the whole picture
_KEY_RING_PX = 4
_KEY_THRESH = 0.06

# ---- V2 / V3 registration
_BACK_IOU_GATE = 0.75
_MAX_ROLL_DEG = 5.0
_MAX_KEYSTONE = 0.5  # perspective stage: bound of the projective terms (normalised coordinates)
_TPS_POINTS = 64
_TPS_LAM = 1e-3
_PAIR_WINDOW = 0.04  # partner search window (share of the outline)
_POSE_MAX_FRAC = 0.06  # pose gate: of the subject's long side in levelled pixels ...
_POSE_GAP_PCT = 98.0  # ... at this percentile of the outline gap after the global fit
_TPS_EXACT_PX = 256 * 256  # larger frames evaluate the composed map on a 4 px lattice
_TPS_LATTICE = 4
_SIDE_MIN_SIN = 0.3  # |sin phi| above this: a side or quarter view (hull, V5, V10)
_SIDE_EXACT_SIN = 0.95  # ... above this: a left / right view (back edges, QC)
_SCALE_RANGE = (0.2, 5.0)  # plausible registration scales of a nominal view
_BORDER_PX = 2  # a side matte this near its picture's top / bottom is cut off there
_CUT_MIN_SHOWN = 0.5  # a cut-off picture still shows at least half the subject's height
_SCALE_STEPS = 321  # scale search of a cut-off side view: log-spaced steps over its bracket

# ---- V4 colour
_RIM_BAND_FRAC = 0.03
_REINHARD_STRENGTH = 0.5
_AB_SHIFT_MAX = 6.0  # LAB units: the colour (not lightness) part of the rim-band transfer

# ---- V5 offsets
_EDGE_PCT = 2.0
_MIN_ROW_PTS = 3
_MIN_ROWS_FRAC = 0.2
_MAD_MAX_M = 0.03
_MAD_MAX_FRAC = 0.02  # of the height
_FLIP_RATIO = 0.5
_FLIP_MIN_MISFIT_M = 0.02
_FLIP_PCT = 90.0  # row-offset percentile that judges the mirror hypothesis
_AGREE_M = 0.04
_AGREE_FRAC = 0.03

# ---- orthographic sets (V2-V5 "mvadapter")
_ORTHO_IOU_GATE = 0.75  # view 0 (after the similarity) / the back vs the photo's outline
_ORTHO_FIT_GAIN = 0.02  # the lateral similarity is applied when it raises the IoU this much ...
_ORTHO_FIT_PX = 0.5  # ... and moves the outline by more than this somewhere
_ORTHO_SCALE_MAX = 0.05  # invariant rows: a view whose scale differs more is dropped
_ORTHO_ROWS_PX = 0.25  # row corrections below this are not resampled
_ORTHO_ELEV_MAX = 2.0  # deg: views must be level
_AXIS_AGREE_FRAC = 0.02  # of the height: +-90 views' axis depths agree within max(4 cm, this)
_SIGN_PCT = 90.0  # the azimuth-sign test compares this percentile of the row deviations ...
_SIGN_MIN_MISFIT_M = 0.01  # ... and the named hypothesis must miss by this much (per view) to flip
_SIGN_RENDER_PX = 192  # the second sign test: SHARP's front rendered at this size ...
_SIGN_RENDER_MARGIN = 0.05  # ... the reversed hypothesis must score this much better per view
_SIGN_SSIM_WEIGHT = 0.5  # score = IoU + 0.5 SSIM + 0.5 (1 - colour MAE / 0.25)
_SIGN_COLOUR_WEIGHT = 0.5
_SIGN_COLOUR_SCALE = 0.25
_SIGN_RENDER_DIST_M = 10.0  # the orthographic render camera stands this far in front of the axis
_OUTLINE_MAX_PX = 1.5  # lateral fit: a uniform outline offset up to this is not taken for a scale
_OUTLINE_PTS = 1500  # outline points per mask of that fit
_COLOR_FIT_ALPHA = 0.9  # V4 pixel pairs: both mattes above this ...
_COLOR_FIT_ERODE = 4  # ... eroded by this many px ...
_COLOR_FIT_GRAD_PCT = 30.0  # ... and the input's gradient at most its 30th percentile
_COLOR_FIT_MIN = 64  # fewer pairs: the Reinhard rim-band transfer instead
_QC_DROPPED = 0.5  # QC cost per view dropped from an orthographic set
_QC_FLIPPED = 0.5  # ... for a set turned round (the generator's cameras were wrong) ...
_QC_SIGN_UNSURE = 0.25  # ... and for one whose outlines fit better mirrored, its pictures not

# ---- V6 hull
_HULL_SAMPLES = 96
_HULL_RANGE_M = 0.25
_HULL_RANGE_FRAC = 0.5  # of the height
_HULL_DILATE_PX = 3
_HULL_BISECT = 3
_HULL_CHUNK = 4096  # rays per chunk

# ---- V7 target
_RELIEF_BLUR_FRAC = 0.04
_RELIEF_HUBER_M = 0.01
_RELIEF_SMOOTH_FRAC = 0.05
_TARGET_GAP_M = 0.015
_HULL_SLACK_M = 0.02
_CARVE_MAX = 0.4
_TARGET_ITERS = 2

# ---- V9 / V10
_COLOR_POWER = 8
_VIS_TOL_M = 0.015
_VIS_SLOPE_MAX_M = 0.05  # cap of the slope term of the visibility tolerance
_SPLAT_R = (1.0, 4.0)
_MAX_OCCLUDERS = 400_000
_HOLE_ERODE_PX = 2
_HOLE_SCALE = 0.7
_HOLE_MAX_FRAC = 0.2
_HOLE_OPACITY = 0.95
_HOLE_SAMPLES = 96
_S3_MIN = 0.0005

_GREY = 128


# =========================================================================== data
@dataclass
class GenView:
    """One picture of the subject from azimuth ``phi_deg`` (design 3.0: 0 = the photo's
    side, 90 = "you walked to your right", 180 = behind).

    ``kind`` is ``"ortho"`` (an exact orthographic camera at elevation 0 whose picture
    centre is on the orbit axis: MV-Adapter, a saved views folder) or ``"nominal"`` (a user
    photo / AI edit: only the azimuth is known).  For ``"ortho"`` views ``m`` is the metres
    per pixel and ``meta["frame_center_S"]`` the ``(x0, y0)`` (frame S) at the picture
    centre; ``m=None`` / no centre means the levelled input's framing (the generator keeps
    it: ``m = m_in * S / width``, centre ``OrthoInput.center_S``).  ``rgb`` is uint8
    ``H x W x 3``, ``alpha`` float32 ``H x W`` (1 = subject), ``depth`` an optional float32
    map, ``meta`` JSON-friendly notes (file, matte method, registration).  Views returned
    by :func:`view_based_back` are registered: ``S x S`` in the levelled frame, ``m`` set,
    ``meta["registered"] = True`` and ``meta["offset_m"]`` the horizontal offset ``o``."""

    name: str
    phi_deg: float
    rgb: np.ndarray
    alpha: np.ndarray
    kind: str = "nominal"
    m: float | None = None
    depth: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in VIEW_KINDS:
            raise InputError(f"Unknown view kind {self.kind!r}; use one of {', '.join(VIEW_KINDS)}.")
        rgb = np.asarray(self.rgb)
        a = np.asarray(self.alpha)
        if rgb.ndim != 3 or rgb.shape[2] < 3 or a.shape != rgb.shape[:2]:
            raise InputError(
                f"View '{self.name}': the picture ({rgb.shape}) must be H x W x 3 and its matte "
                f"({a.shape}) H x W."
            )
        try:
            phi = float(self.phi_deg)
        except (TypeError, ValueError):
            phi = math.nan
        if not math.isfinite(phi):
            raise InputError(f"View '{self.name}': invalid azimuth {self.phi_deg!r}.")
        self.phi_deg = phi % 360.0


@dataclass
class OrthoInput:
    """The levelled orthographic input (design 3.4): the photo's visible front seen
    square-on from the horizontal.  Pixel ``(i, j)`` of the ``S x S`` picture covers the S
    points with ``x in [x0 + (j - S/2) m, x0 + (j + 1 - S/2) m)`` and likewise ``y`` with
    ``(x0, y0) = center_S`` and ``m = m_in``; ``rgba_ref`` is the same at ``ref`` pixels
    (``m_in * size / ref`` metres per pixel).  Colours are bled past the matte (straight,
    not premultiplied); compositing on 0.5 grey is the worker's job."""

    rgba: np.ndarray  # uint8 S x S x 4 (generator input)
    rgba_ref: np.ndarray  # uint8 R x R x 4 (colour reference / TRELLIS input)
    alpha0: np.ndarray  # bool S x S: orthographic front silhouette
    m_in: float  # metres per pixel at size S
    size: int
    ref: int
    center_S: np.ndarray  # (2,) (x0, y0) in frame S at the picture centre
    meta: dict[str, Any] = field(default_factory=dict)  # addition: supersampling, extent

    @property
    def m_ref(self) -> float:
        """Metres per pixel of ``rgba_ref``."""
        return float(self.m_in) * self.size / self.ref


@dataclass
class _Frame:
    """The levelled frame shared by every registered view."""

    S: int
    m: float
    cx: float
    cy: float
    R: np.ndarray
    c: np.ndarray
    alpha0: np.ndarray
    L_px: float  # the subject's long side in levelled pixels

    def to_S(self, P: np.ndarray) -> np.ndarray:
        return (np.asarray(P, dtype=np.float64) - self.c) @ self.R

    def to_C(self, P: np.ndarray) -> np.ndarray:
        return np.asarray(P, dtype=np.float64) @ self.R.T + self.c


# =========================================================================== small helpers
def _axes(phi_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """``(w, f)`` in S: the view's right ``(cos, 0, sin)`` and viewing direction ``(-sin, 0, cos)``."""
    p = math.radians(float(phi_deg))
    c, s = math.cos(p), math.sin(p)
    c = 0.0 if abs(c) < 1e-12 else c
    s = 0.0 if abs(s) < 1e-12 else s
    return np.array([c, 0.0, s]), np.array([-s, 0.0, c])


def _project(P_S: np.ndarray, phi: float, offset: float, fr: _Frame) -> tuple[np.ndarray, ...]:
    """Continuous ``(u, v, depth)`` of S points in the registered view ``(phi, offset)``."""
    w, f = _axes(phi)
    u = (P_S @ w - offset) / fr.m + fr.S / 2.0
    v = (P_S[..., 1] - fr.cy) / fr.m + fr.S / 2.0
    return u, v, P_S @ f


def _az_diff(a: float, b: float) -> float:
    return abs((float(a) - float(b) + 180.0) % 360.0 - 180.0)


def _is_back(phi: float) -> bool:
    return _az_diff(phi, 180.0) < 1.0


def _sin(phi: float) -> float:
    return math.sin(math.radians(float(phi)))


def _phi_name(phi: float) -> str:
    k = int(round(float(phi))) % 360
    if abs(k - float(phi) % 360.0) < 1e-6 and k in PHI_NAMES:
        return PHI_NAMES[k]
    return f"az{k:03d}"


def _rgb_u8(img: Any, what: str) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] < 3:
        raise InputError(f"The {what} must be an H x W x 3 picture (got shape {a.shape}).")
    a = a[..., :3]
    if a.dtype == np.uint8:
        return np.ascontiguousarray(a)
    if a.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(a, initial=0.0)) <= 1.0 else 1.0
        return np.clip(np.rint(np.nan_to_num(a.astype(np.float64)) * scale), 0, 255).astype(np.uint8)
    return np.clip(a, 0, 255).astype(np.uint8)


def _remap_points(img: np.ndarray, x: Any, y: Any, *, border: Any = 0.0) -> np.ndarray:
    """Sample ``img`` (``H x W`` or ``H x W x C``) at OpenCV coordinates (pixel centres at
    integers) of any shape; returns ``x.shape`` (``+ (C,)``) float32.  Non-finite or outside
    coordinates get ``border``."""
    xs = np.asarray(x, dtype=np.float64)
    shape = xs.shape
    xs = xs.ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    a = np.asarray(img)
    a = a.astype(np.float32) if a.dtype != np.float32 else a
    chans = () if a.ndim == 2 else (a.shape[2],)
    n = len(xs)
    if n == 0:
        return np.zeros(shape + chans, np.float32)
    cols = 1024
    rows = (n + cols - 1) // cols
    mx = np.full(rows * cols, -1e4, np.float32)
    my = np.full(rows * cols, -1e4, np.float32)
    ok = np.isfinite(xs) & np.isfinite(ys)
    mx[:n][ok] = np.clip(xs[ok], -1e4, 1e5)
    my[:n][ok] = np.clip(ys[ok], -1e4, 1e5)
    bv = float(border) if np.ndim(border) == 0 else tuple(float(b) for b in np.ravel(border))
    out = cv2.remap(
        np.ascontiguousarray(a),
        mx.reshape(rows, cols),
        my.reshape(rows, cols),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bv,
    )
    return out.reshape(rows * cols, *chans)[:n].reshape(shape + chans)


def _dist_in(mask: np.ndarray) -> np.ndarray:
    """Distance (px) of inside pixels to the nearest outside pixel (borders count as outside)."""
    pad = np.pad(mask.astype(np.uint8), 1)
    return cv2.distanceTransform(pad, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]


def _disc(r: int) -> np.ndarray:
    r = max(1, int(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _row_edges(alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sub-pixel left / right silhouette edge per row (continuous ``u``; NaN for empty rows).

    For a vertical edge at ``e`` the pixel coverage is ``clip(j + 1 - e, 0, 1)`` on its left
    side, so ``e = j + 1 - a[j] - a[j-1]`` with ``j`` the first pixel with ``a >= 0.5`` (exact
    for anti-aliased mattes, the pixel boundary for hard ones); mirrored on the right."""
    a = np.clip(np.nan_to_num(np.asarray(alpha, dtype=np.float64)), 0.0, 1.0)
    h, w = a.shape
    inside = a >= 0.5
    has = inside.any(axis=1)
    r = np.arange(h)
    first = np.argmax(inside, axis=1)
    last = w - 1 - np.argmax(inside[:, ::-1], axis=1)
    a_prev = np.where(first > 0, a[r, np.maximum(first - 1, 0)], 0.0)
    a_next = np.where(last < w - 1, a[r, np.minimum(last + 1, w - 1)], 0.0)
    left = first + 1.0 - a[r, first] - a_prev
    right = last + a[r, last] + a_next
    nan = np.full(h, np.nan)
    return np.where(has, left, nan), np.where(has, right, nan)


def _top_bottom(alpha: np.ndarray) -> tuple[float, float] | None:
    """Sub-pixel top / bottom edge (continuous ``v``) of a silhouette, from the per-row
    maximum coverage like :func:`_row_edges`; None when empty."""
    a = np.clip(np.nan_to_num(np.asarray(alpha, dtype=np.float64)), 0.0, 1.0)
    cov = a.max(axis=1)
    rows = np.flatnonzero(cov >= 0.5)
    if not len(rows):
        return None
    t, b = int(rows[0]), int(rows[-1])
    top = t + 1.0 - cov[t] - (cov[t - 1] if t > 0 else 0.0)
    bottom = b + cov[b] + (cov[b + 1] if b + 1 < len(cov) else 0.0)
    return float(top), float(bottom)


def _empty_cloud() -> SplatCloud:
    f32 = np.float32
    return SplatCloud(
        means=np.zeros((0, 3), f32),
        scales=np.zeros((0, 3), f32),
        quats=np.zeros((0, 4), f32),
        colors=np.zeros((0, 3), f32),
        opacity=np.zeros(0, f32),
    )


def _size(x: Any, what: str) -> int:
    try:
        v = int(x)
    except (TypeError, ValueError):
        v = 0
    if v < 16 or v != x:
        raise InputError(f"The levelled picture {what} must be a whole number >= 16 (got {x!r}).")
    return v


def _bleed(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Colours of the matte (eroded by 1 px: its edge mixes in the background) continued
    outward by push-pull, so samples just outside never pick up the background."""
    inside = cv2.erode((np.asarray(alpha) >= 0.5).astype(np.uint8), _disc(1)) > 0
    if not inside.any():
        inside = np.asarray(alpha) >= 0.5
    if not inside.any():
        return np.asarray(rgb).copy()
    return push_pull_fill(np.ascontiguousarray(rgb), inside)


# =========================================================================== levelled input
def levelled_ortho_input(
    rgb: np.ndarray,
    alpha: FloatMap,
    front: FrontModel,
    canon: Canon,
    *,
    size: int = 512,
    ref: int = 1024,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> OrthoInput:
    """The photo forward-splatted with its front depth into an upright orthographic view
    (design 3.4): no camera pitch, no perspective magnification, so a pixel maps to S
    exactly.

    1. Photo pixels (downscaled to a long side <= 2048) whose alpha exceeds 0.02, plus a
       2 px ring for the soft edge, get the front depth ``D_f`` resampled bilinearly from
       the mould grid and become points in S.  Outside the mould matte the depth is that of
       the nearest matte cell (the design's ``D_ext`` is the eroded, i.e. nearest, depth: at
       a convex outline it is too near and pulled the levelled edge ~1 px inward).
    2. ``L_S`` = the subject's long extent in S (0.05th-99.95th percentile of its pixels),
       ``m_in = L_S / (0.9 size)``, centred on the middle of that box (``center_S``).
    3. Each target pixel keeps the samples within 2 cm of its nearest one (z-buffer by
       ``np.minimum.at``); colours are their alpha-weighted mean, alpha the mean matte value
       (samples outside the subject always count, so edges stay anti-aliased).  The photo
       is supersampled (up to 8 x 8) where it is magnified, so samples land < 1 px apart.
    4. Morphological close of the covered pixels, push-pull fill inside; colours bled
       outward.  Returned at ``size`` and ``ref`` pixels; ``alpha0 = alpha_size >= 0.5``.

    ``rgb`` (``H x W x 3``) and ``alpha`` (``H x W``) are the photo and matte at any
    resolution with SHARP's field of view.  ``progress`` / ``cancel`` are additions."""
    rep = _as_reporter(progress)
    S, Rr = _size(size, "size"), _size(ref, "reference size")
    if Rr < S:
        raise InputError(f"The reference size ({Rr}) must be at least the picture size ({S}).")
    img = _rgb_u8(rgb, "photo")
    a = np.asarray(alpha, dtype=np.float32)
    if a.ndim == 3:
        a = a[..., 0]
    if a.shape != img.shape[:2]:
        raise InputError(f"The matte ({a.shape}) must match the photo ({img.shape[:2]}).")
    a = np.clip(np.nan_to_num(a), 0.0, 1.0)
    h, w = a.shape
    if max(h, w) > MAX_PHOTO_LONG:
        k0 = MAX_PHOTO_LONG / max(h, w)
        dsize = (max(1, round(w * k0)), max(1, round(h * k0)))
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, dsize, interpolation=cv2.INTER_AREA)
        h, w = a.shape
    subj = a > _ALPHA_MIN
    if not subj.any():
        raise InputError("No subject was found in the cut-out.", hint="Check the subject cut-out (matte).")
    _check(cancel)
    rep(0.0, "Levelling the photo")

    cam, grid = front.cam, front.grid
    matte_m, D_f = _front_maps(front)
    # outside the matte (the soft edge) each pixel takes the depth of the nearest matte cell:
    # a convex outline curves away from the camera, and the smoother continuations (D_ext,
    # the push-pull ring) average in the nearer interior, which pulls the edge toward the
    # optical axis by about a pixel
    Dm = np.ascontiguousarray(_nearest_fill(np.where(matte_m, D_f, 0.0).astype(np.float32), matte_m))
    sx, sy = cam.width / w, cam.height / h
    cell = float(grid.cell_px)

    def points(uu: np.ndarray, vv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """S points and C depths of photo coordinates (continuous, matte pixels)."""
        us, vs = uu * sx, vv * sy
        z = _remap_points(Dm, us / cell - 0.5, vs / cell - 0.5, border=np.nan).astype(np.float64)
        z = np.where(np.isfinite(z), z, float(np.nanmedian(Dm)))
        return canon.to_S(cam.rays(us, vs) * z[:, None]), z

    # ---- 2. scale and centre from the hard matte's pixel centres
    hard = a >= 0.5
    ii, jj = np.nonzero(hard if hard.any() else subj)
    step = max(1, int(math.ceil(len(ii) / 400_000)))
    P0, z0 = points(jj[::step] + 0.5, ii[::step] + 0.5)
    lo = np.percentile(P0[:, :2], _EXTENT_PCT, axis=0)
    hi = np.percentile(P0[:, :2], 100.0 - _EXTENT_PCT, axis=0)
    L = float(max(hi - lo))
    if not (math.isfinite(L) and L > 1e-6):
        raise InputError("The subject's size could not be measured from its front.")
    cx, cy = (float(t) for t in 0.5 * (lo + hi))
    m_in = L / (_FILL * S)
    m_ref = L / (_FILL * Rr)
    _check(cancel)

    # ---- supersampling so neighbouring samples land < 1 px apart at the reference size
    f_ph = float(cam.f) / (0.5 * (sx + sy))
    mag = float(np.median(z0)) / f_ph / m_ref  # reference pixels per photo pixel
    region = cv2.dilate(subj.astype(np.uint8), _disc(2)) > 0
    rr = np.flatnonzero(region.any(axis=1))
    cc = np.flatnonzero(region.any(axis=0))
    y0, y1, x0, x1 = int(rr[0]), int(rr[-1]) + 1, int(cc[0]), int(cc[-1]) + 1
    n_reg = int(region[y0:y1, x0:x1].sum())
    k = int(min(_MAX_SUPERSAMPLE, max(1, math.ceil(mag * _SUPERSAMPLE_MARGIN))))
    while k > 1 and n_reg * k * k > _MAX_SAMPLES:
        k -= 1
    crop_rgb = img[y0:y1, x0:x1]
    crop_a = a[y0:y1, x0:x1]
    crop_r = region[y0:y1, x0:x1].astype(np.uint8)
    if k > 1:
        dsize = ((x1 - x0) * k, (y1 - y0) * k)
        crop_rgb = cv2.resize(crop_rgb, dsize, interpolation=cv2.INTER_LINEAR)
        crop_a = cv2.resize(crop_a, dsize, interpolation=cv2.INTER_LINEAR)
        crop_r = cv2.resize(crop_r, dsize, interpolation=cv2.INTER_NEAREST)
    si, sj = np.nonzero(crop_r)
    n = len(si)
    # one compact record per sample: target pixel at both sizes, z_S, alpha, colour
    res = (S, Rr)
    m_of = (m_in, m_ref)
    idx = [np.empty(n, np.int32) for _ in res]  # N*N <= 2048^2 fits
    zs = np.empty(n, np.float32)
    al = crop_a[si, sj].astype(np.float32)
    col = crop_rgb[si, sj]
    for c0 in range(0, n, _SAMPLE_CHUNK):
        _check(cancel)
        sl = slice(c0, c0 + _SAMPLE_CHUNK)
        P, _ = points(x0 + (sj[sl] + 0.5) / k, y0 + (si[sl] + 0.5) / k)
        zs[sl] = P[:, 2]
        for t, (N, mm) in enumerate(zip(res, m_of, strict=True)):
            ju = np.floor((P[:, 0] - cx) / mm + N / 2.0)
            iv = np.floor((P[:, 1] - cy) / mm + N / 2.0)
            ok = (ju >= 0) & (ju < N) & (iv >= 0) & (iv < N)
            idx[t][sl] = np.where(ok, iv * N + ju, -1).astype(np.int32)
        rep(0.1 + 0.6 * min(1.0, (c0 + _SAMPLE_CHUNK) / max(n, 1)), "Levelling the photo")

    outs = []
    for t, N in enumerate(res):
        _check(cancel)
        outs.append(_splat_level(idx[t], zs, al, col, N))
    rep(1.0, "Levelled picture ready")
    alpha0 = outs[0][..., 3] >= 128
    log.info(
        "Levelled input: subject %.3f m long -> %.2f mm/px at %d px (x%d supersampling, %d samples)",
        L,
        1000.0 * m_in,
        S,
        k,
        n,
    )
    return OrthoInput(
        rgba=outs[0],
        rgba_ref=outs[1],
        alpha0=alpha0,
        m_in=float(m_in),
        size=S,
        ref=Rr,
        center_S=np.array([cx, cy], np.float64),
        meta={"long_m": L, "supersample": k, "samples": n, "magnification": round(mag, 3)},
    )


def _splat_level(idx: np.ndarray, zs: np.ndarray, al: np.ndarray, col: np.ndarray, N: int) -> np.ndarray:
    """Forward-splat samples (target pixel, z_S, alpha, colour) into an ``N x N`` RGBA picture."""
    inb = idx >= 0
    subj = inb & (al > _ALPHA_MIN)
    zmin = np.full(N * N, np.inf, np.float32)
    np.minimum.at(zmin, idx[subj], zs[subj])
    near = subj & (zs <= zmin[np.where(inb, idx, 0)] + _FRONT_TOL_M)
    use_a = near | (inb & ~subj)  # outside samples dilute the edge; hidden ones do not count
    cnt = np.bincount(idx[use_a], minlength=N * N).astype(np.float64)
    asum = np.bincount(idx[use_a], weights=al[use_a], minlength=N * N)
    wsum = np.bincount(idx[near], weights=al[near], minlength=N * N)
    rgb = np.zeros((N * N, 3), np.float64)
    for ch in range(3):
        rgb[:, ch] = np.bincount(idx[near], weights=al[near] * col[near, ch], minlength=N * N)
    covered = (cnt > 0).reshape(N, N)
    alpha = np.where(cnt > 0, asum / np.maximum(cnt, 1.0), 0.0).reshape(N, N).astype(np.float32)
    known_c = (wsum > 0).reshape(N, N)
    rgb = np.where((wsum > 0)[:, None], rgb / np.maximum(wsum, 1e-12)[:, None], 0.0).reshape(N, N, 3)
    # pixels no sample reached inside the subject (disocclusions, magnified parts)
    closed = cv2.morphologyEx(covered.astype(np.uint8), cv2.MORPH_CLOSE, _disc(2)) > 0
    holes = closed & ~covered
    if holes.any() and covered.any():
        alpha = np.where(holes, push_pull_fill(alpha, covered), alpha)
    alpha = np.where(closed, alpha, 0.0)
    if known_c.any():
        rgb = push_pull_fill(rgb.astype(np.float32), known_c)
    rgba = np.empty((N, N, 4), np.uint8)
    rgba[..., :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return rgba


# =========================================================================== views folder
_COMPOUND_RE = re.compile(r"^(front|back)[ _-]?(left|right)$")
_AZ_RE = re.compile(r"^az[ _-]?(\d{1,3})$")


def _phi_of_stem(stem: str) -> float | None:
    """Azimuth of a views-folder file name (whole stem, case-insensitive), None if none."""
    s = stem.strip().lower()
    m = _COMPOUND_RE.match(s)
    if m:
        return VIEW_NAMES[f"{m.group(1)}_{m.group(2)}"]
    if s in VIEW_NAMES:
        return VIEW_NAMES[s]
    m = _AZ_RE.match(s)
    if m:
        return float(int(m.group(1)) % 360)
    return None


def _views_json_records(folder: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """``(records, top)`` of a saved folder's ``views.json``: ``records`` maps each listed
    file name (lower case) to its entry with the azimuth as ``"phi"`` (from ``phi_deg``, else
    ``azimuth_deg``; entries without a usable name or azimuth are skipped), ``top`` is the
    whole document (the shared ``frame`` of :func:`save_views_folder`).  Both are empty when
    the file is missing or unreadable."""
    path = folder / VIEWS_JSON
    if not path.is_file():
        return {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        out: dict[str, dict[str, Any]] = {}
        for v in data.get("views", []):
            if not isinstance(v, dict):
                continue
            name = str(v.get("file", ""))
            phi = v.get("phi_deg", v.get("azimuth_deg"))
            try:
                phi_f = float(phi) if phi is not None else math.nan
            except (TypeError, ValueError):
                phi_f = math.nan
            if name and math.isfinite(phi_f):
                out[Path(name).name.lower()] = {**v, "phi": phi_f % 360.0}
        return out, data
    except (OSError, ValueError, TypeError, AttributeError) as e:
        log.warning("Ignoring %s: %s", path, e)
        return {}, {}


def _views_json(folder: Path) -> dict[str, float]:
    """``file name (lower case) -> azimuth`` from a saved folder's ``views.json`` (empty when
    missing or unreadable)."""
    return {k: float(r["phi"]) for k, r in _views_json_records(folder)[0].items()}


def views_from_folder(back_path: str | Path) -> list[tuple[float, Path]]:
    """The views to use for a back photo (design 3.5.4): ``[(phi_deg, path), ...]`` sorted by
    azimuth.

    ``back_path`` is the back picture (phi 180, unless its own name follows a rule below) or
    a folder.  Pictures in the same folder are used when their whole file name (any case,
    any picture extension) is ``right`` (90: you walked to your right), ``left`` (270),
    ``back_right`` (135), ``back_left`` (225), ``front_right`` (45), ``front_left`` (315),
    ``back`` / ``behind`` / ``rear`` (180) or ``azNNN`` (NNN degrees); compound names may use
    ``_``, ``-``, a space or nothing between the words.  Other names (also ``my_back``,
    ``left2``) are ignored.  A ``views.json`` with ``{"views": [{"file", "phi_deg" or
    "azimuth_deg"}]}`` (a saved views folder, :func:`save_views_folder`) sets the azimuth of
    the files it lists, whatever their names; its exact orthographic cameras are restored
    by :func:`load_views`.  One picture per azimuth: the given file first, then names in
    sorted order.  Raises :class:`InputError` when the picture does not exist or is not a
    picture."""
    p = Path(str(back_path)).expanduser()
    if p.is_dir():
        folder, given = p, None
    else:
        if not p.is_file():
            raise InputError(
                f"The back photo was not found: {p}",
                hint="Pick the picture of the back again ('Back photo').",
            )
        if p.suffix.lower() not in IMAGE_EXTS:
            raise InputError(
                f"'{p.name}' is not a picture.",
                hint="Pick a JPEG, PNG or WebP picture of the subject's back.",
            )
        folder, given = p.parent, p
    exact = _views_json(folder)
    chosen: dict[float, Path] = {}
    if given is not None:
        phi = exact.get(given.name.lower(), _phi_of_stem(given.stem))
        chosen[180.0 if phi is None else phi] = given
    try:
        siblings = sorted(
            (q for q in folder.iterdir() if q.is_file() and q.suffix.lower() in IMAGE_EXTS),
            key=lambda q: q.name.lower(),
        )
    except OSError as e:
        log.warning("Could not list %s: %s", folder, e)
        siblings = []
    for q in siblings:
        if given is not None and q == given:
            continue
        phi = exact.get(q.name.lower(), _phi_of_stem(q.stem))
        if phi is None:
            continue
        if phi in chosen:
            log.info("View %s ignored: %s already shows %.0f deg", q.name, chosen[phi].name, phi)
            continue
        chosen[phi] = q
    return sorted(chosen.items(), key=lambda t: t[0])


def load_views(
    entries: Sequence[tuple[float, str | Path]],
    *,
    max_long: int = VIEW_MAX_LONG,
    dropped: list[str] | None = None,
) -> list[GenView]:
    """Read the pictures of :func:`views_from_folder` as :class:`GenView` objects,
    downscaled to a long side <= ``max_long`` (``INTER_AREA``).

    A picture listed in its folder's ``views.json`` as ``"kind": "ortho"`` (a saved views
    folder, :func:`save_views_folder`) whose size is still the recorded one becomes an
    orthographic view with its exact camera restored (``m`` - adjusted when downscaled -,
    ``meta["frame_center_S"]``, the colour match already applied, ``meta["camera"]``); a
    picture replaced by one of another size (a photo) is nominal.  Other pictures are
    nominal.  A picture with a real alpha channel (a cut-out PNG) keeps it as its matte
    (``meta["alpha_from_file"]``); otherwise ``alpha`` is all ones until
    :func:`segment_views`.  Unreadable pictures are skipped with a logged warning and a
    plain sentence appended to ``dropped`` (an addition, like :func:`segment_views`'s);
    :class:`InputError` when none can be read."""
    try:
        cap = int(max_long)
    except (TypeError, ValueError):
        cap = 0
    if cap < 16:
        raise InputError(f"max_long must be at least 16 pixels (got {max_long!r}).")
    out: list[GenView] = []
    errors: list[str] = []
    saved: dict[Path, tuple[dict[str, dict[str, Any]], dict[str, Any]]] = {}
    for phi, path in entries:
        p = Path(path)
        try:
            img = load_image(p, with_alpha=True, max_px=cap)
        except (InputError, OSError, ValueError) as e:
            log.warning("Could not read the view %s: %s", p, e)
            errors.append(f"{p.name}: {e}")
            if dropped is not None:
                why = str(e).strip().rstrip(".") or type(e).__name__
                why = why.removeprefix(f"Cannot read image {p.name}: ")  # the name is said already
                dropped.append(f"The {_phi_name(phi)} picture ({p.name}) could not be read ({why}).")
            continue
        h0, w0 = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3])
        a8 = img[..., 3] if img.shape[2] == 4 else None
        scale = min(1.0, cap / max(h0, w0))
        if scale < 1.0:
            dsize = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
            rgb = cv2.resize(rgb, dsize, interpolation=cv2.INTER_AREA)
            if a8 is not None:
                a8 = cv2.resize(a8, dsize, interpolation=cv2.INTER_AREA)
        from_file = a8 is not None and bool((a8 < 128).any()) and bool((a8 >= 128).any())
        alpha = a8.astype(np.float32) / 255.0 if from_file else np.ones(rgb.shape[:2], np.float32)
        if p.parent not in saved:
            saved[p.parent] = _views_json_records(p.parent)
        records, top = saved[p.parent]
        kind, m, extra = _saved_camera(records.get(p.name.lower()), top, (w0, h0), scale)
        out.append(
            GenView(
                name=_phi_name(phi),
                phi_deg=float(phi),
                rgb=rgb,
                alpha=alpha,
                kind=kind,
                m=m,
                meta={
                    "file": p.name,
                    "path": str(p),
                    "source_size": [int(w0), int(h0)],  # as decoded (JPEG draft scaling)
                    "scale": float(scale),
                    "alpha_from_file": bool(from_file),
                    **extra,
                },
            )
        )
    if not out:
        detail = f" ({errors[0]})" if errors else ""
        raise InputError(f"None of the view pictures could be read{detail}.", hint="Check the back photo.")
    return out


def _pos_float(x: Any) -> float | None:
    """``x`` as a finite positive float, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def _pair(x: Any) -> list[float] | None:
    """``x`` as a finite ``[a, b]``, else None."""
    try:
        a = [float(t) for t in np.asarray(x, dtype=np.float64).ravel()]
    except (TypeError, ValueError):
        return None
    return a[:2] if len(a) >= 2 and all(math.isfinite(t) for t in a[:2]) else None


def _saved_camera(
    rec: dict[str, Any] | None, top: dict[str, Any], size: tuple[int, int], scale: float
) -> tuple[str, float | None, dict[str, Any]]:
    """``(kind, m, meta)`` of a picture from its ``views.json`` entry: an orthographic camera
    when the entry says ``"kind": "ortho"`` and the picture still has the recorded size."""
    if not rec or str(rec.get("kind", "")).strip().lower() != "ortho":
        return "nominal", None, {}
    w, h = rec.get("width"), rec.get("height")
    try:
        same = w is None or h is None or (int(w), int(h)) == (int(size[0]), int(size[1]))
    except (TypeError, ValueError):
        same = False
    if not same:
        log.info(
            "View %s: %dx%d px instead of the saved %sx%s - used as a photo (its camera no longer applies)",
            rec.get("file"),
            size[0],
            size[1],
            w,
            h,
        )
        return "nominal", None, {"camera_dropped": "the picture's size changed"}
    frame = top.get("frame") if isinstance(top.get("frame"), dict) else {}
    m = _pos_float(rec.get("m"))
    extra: dict[str, Any] = {"framing": "saved", "camera": {k: v for k, v in rec.items() if k != "phi"}}
    centre = _pair(rec.get("center_S", frame.get("center_S")))
    if centre is not None:
        extra["frame_center_S"] = centre
    if rec.get("color_match"):
        extra["color_match"] = str(rec["color_match"])
    for key in ("ortho_half_extent", "input_fill"):
        if key in top:
            extra[key] = top[key]
    return "ortho", (m / scale if m is not None else None), extra


_C2W_CONVENTIONS = {"c2w_opencv": (1.0, 1.0), "c2w_opengl": (-1.0, -1.0)}  # signs of the y / z columns


def _cam_axes(c2w: Any, convention: str, what: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(right, down, forward)`` unit vectors (world) of a camera-to-world matrix."""
    try:
        M = np.asarray(c2w, dtype=np.float64)
    except (TypeError, ValueError):
        M = np.zeros(0)
    if M.ndim != 2 or M.shape[0] < 3 or M.shape[1] < 3 or not np.isfinite(M[:3, :3]).all():
        raise InputError(f"The camera of {what} is not a 3 x 3 / 4 x 4 matrix.")
    sd, sf = _C2W_CONVENTIONS[convention]
    out = []
    for col in (M[:3, 0], sd * M[:3, 1], sf * M[:3, 2]):
        n = float(np.linalg.norm(col))
        if not n > 1e-9:
            raise InputError(f"The camera of {what} has a zero axis.")
        out.append(col / n)
    return out[0], out[1], out[2]


def _phi_from_c2w(
    axes: tuple[np.ndarray, np.ndarray, np.ndarray], ref: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[float, float]:
    """``(phi, elevation)`` in degrees of a camera relative to the reference (input) camera,
    world axes unknown: ``phi`` is the angle about the reference's up from its direction
    (subject -> camera) toward its right (design 3.0: 90 = "you walked to your right")."""
    right0, down0, fwd0 = ref
    up = -down0
    d = -axes[2]  # subject -> camera
    elev = math.degrees(math.asin(float(np.clip(d @ up, -1.0, 1.0))))
    dh = d - (d @ up) * up
    phi = math.degrees(math.atan2(float(dh @ right0), float(dh @ -fwd0)))
    return phi % 360.0, elev


def views_from_cameras(
    cameras_json_path_or_dict: str | Path | Mapping[str, Any], *, base_dir: str | Path | None = None
) -> list[GenView]:
    """The orthographic views of a generator's camera file (design 4.3 ``cameras.json``:
    ``{"projection": "orthographic", "convention": "c2w_opencv", "ortho_half_extent",
    "input_fill", "width", "height", "views": [{"file", "phi_deg", "azimuth_deg",
    "elevation_deg", "c2w"}, ...]}``) as ``kind="ortho"`` :class:`GenView` objects sorted by
    azimuth.

    The azimuth is ``phi_deg`` (the app's convention, written by the worker), else derived
    from ``c2w`` relative to the input's camera (the view whose ``azimuth_deg`` is 0, else
    the first; ``c2w_opencv`` or ``c2w_opengl`` axes; the world axes are not assumed), else
    the generator's own ``azimuth_deg`` (its sign is then only checked by the V5 safety
    net).  Views must be level (within 2 deg).  ``m`` is left None: the generator keeps the
    levelled input's framing, so ``m = m_in * S / width`` in :func:`view_based_back`;
    ``meta`` keeps ``ortho_half_extent``, ``input_fill`` and the camera entry.  Picture paths
    are relative to ``base_dir`` (default: the JSON file's folder; required for a relative
    path in a dict).  A picture with a real alpha channel keeps it as its matte.  Two views
    at one azimuth: the first is kept.  Raises :class:`InputError` (plain sentence) for an
    unreadable or malformed file, a non-orthographic projection, a tilted view or a picture
    that cannot be read."""
    src = cameras_json_path_or_dict
    base: Path | None = Path(base_dir).expanduser() if base_dir is not None else None
    if isinstance(src, Mapping):
        data: Any = dict(src)
    elif isinstance(src, (str, Path)):
        path = Path(src).expanduser()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise InputError(
                f"The camera file of the generated views could not be read ({path.name}: {e})."
            ) from e
        if base is None:
            base = path.parent
    else:
        raise InputError(f"Expected a camera file or its contents (got {type(src).__name__}).")
    if not isinstance(data, dict):
        raise InputError("The camera file of the generated views is not a JSON object.")
    proj = str(data.get("projection", "orthographic")).strip().lower()
    if proj != "orthographic":
        raise InputError(f"The generated views use '{proj}' cameras; only orthographic views can be used.")
    conv = str(data.get("convention", "c2w_opencv")).strip().lower()
    if conv not in _C2W_CONVENTIONS:
        raise InputError(f"Unknown camera convention {conv!r} in the camera file.")
    entries = data.get("views")
    if not isinstance(entries, list) or not entries:
        raise InputError("The camera file lists no views.")
    if not all(isinstance(e, dict) for e in entries):
        raise InputError("The camera file's views must be objects with a 'file' and an azimuth.")
    ref = None
    with_c2w = [e for e in entries if "c2w" in e]
    if any("phi_deg" not in e for e in with_c2w):
        zero = [e for e in with_c2w if _az_diff(_num_or(e.get("azimuth_deg"), math.nan), 0.0) < 0.5]
        first = (zero or with_c2w)[0]
        ref = _cam_axes(first["c2w"], conv, str(first.get("file", "the first view")))
    out: list[GenView] = []
    for i, e in enumerate(entries):
        file = str(e.get("file") or "").strip()
        if not file:
            raise InputError(f"View {i + 1} of the camera file names no picture.")
        elev = _num_or(e.get("elevation_deg"), 0.0)
        if "phi_deg" in e:
            phi, how = _num_or(e.get("phi_deg"), math.nan), "phi_deg"
        elif "c2w" in e and ref is not None:
            phi, elev_c = _phi_from_c2w(_cam_axes(e["c2w"], conv, file), ref)
            elev = elev_c if abs(elev_c) > abs(elev) else elev  # relative to the input's camera
            how = "c2w"
        elif "azimuth_deg" in e:
            phi, how = _num_or(e.get("azimuth_deg"), math.nan), "azimuth_deg"
        else:
            raise InputError(f"The generated view {file} has no azimuth in the camera file.")
        if not math.isfinite(phi) or not math.isfinite(elev):
            raise InputError(f"The generated view {file} has an invalid camera in the camera file.")
        if abs(elev) > _ORTHO_ELEV_MAX:
            where = "above" if elev > 0 else "below"
            raise InputError(
                f"The generated view {file} is not level (its camera is {abs(elev):.0f}° {where} the "
                f"subject's middle); only level views can be used."
            )
        p = Path(file).expanduser()
        if not p.is_absolute():
            if base is None:
                raise InputError(f"The generated view {file} has a relative path, but no folder was given.")
            p = base / p
        try:
            img = load_image(p, with_alpha=True, max_px=VIEW_MAX_LONG)
        except (InputError, OSError, ValueError) as err:
            raise InputError(f"The generated view {p.name} could not be read ({err}).") from err
        h0, w0 = img.shape[:2]
        rgb = np.ascontiguousarray(img[..., :3])
        a8 = img[..., 3] if img.shape[2] == 4 else None
        scale = min(1.0, VIEW_MAX_LONG / max(h0, w0))
        if scale < 1.0:
            dsize = (max(1, round(w0 * scale)), max(1, round(h0 * scale)))
            rgb = cv2.resize(rgb, dsize, interpolation=cv2.INTER_AREA)
            if a8 is not None:
                a8 = cv2.resize(a8, dsize, interpolation=cv2.INTER_AREA)
        from_file = a8 is not None and bool((a8 < 128).any()) and bool((a8 >= 128).any())
        alpha = a8.astype(np.float32) / 255.0 if from_file else np.ones(rgb.shape[:2], np.float32)
        phi = phi % 360.0
        if any(_az_diff(v.phi_deg, phi) < 0.5 for v in out):
            log.info("Generated view %s ignored: another view already shows %.0f deg", p.name, phi)
            continue
        meta: dict[str, Any] = {
            "file": p.name,
            "path": str(p),
            "source_size": [int(w0), int(h0)],
            "alpha_from_file": bool(from_file),
            "framing": "input",
            "camera": {
                "index": i,
                "phi_source": how,
                "azimuth_deg": e.get("azimuth_deg"),
                "elevation_deg": float(elev),
            },
        }
        for key in ("ortho_half_extent", "input_fill"):
            if key in data:
                meta[key] = data[key]
        if how == "azimuth_deg":
            log.info("Generated view %s: azimuth from the generator's own angle (sign not confirmed)", p.name)
        out.append(GenView(_phi_name(phi), phi, rgb, alpha, kind="ortho", m=None, meta=meta))
    return sorted(out, key=lambda v: v.phi_deg)


def _num_or(x: Any, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v


def _ortho_framing_of(v: GenView, ortho: OrthoInput | None) -> tuple[float | None, list[float] | None]:
    """``(m, (x0, y0))`` of an orthographic view's camera (None where unknown)."""
    w = int(np.shape(v.alpha)[1])
    m = _pos_float(v.m)
    if m is None and ortho is not None:
        m = float(ortho.m_in) * int(ortho.size) / max(w, 1)
    centre = _pair(v.meta.get("frame_center_S"))
    if centre is None and ortho is not None:
        centre = _pair(ortho.center_S)
    return m, centre


def save_views_folder(views: Sequence[GenView], folder: str | Path, *, ortho: OrthoInput | None) -> Path:
    """Write views as an editable "images"-style folder (design 0.3 "where generated views
    are saved", 3.5.4): one PNG per azimuth named by the folder rule (``back.png``,
    ``right.png``, ``left.png``, ``front_right.png``, ``front_left.png``, ... ``front.png``
    for 0 deg, ``azNNN.png`` otherwise) composited over 0.5 grey like the generator's output
    (any matte is re-made by :func:`segment_views`), plus ``views.json`` with every file's
    azimuth and, for orthographic views, the exact camera: ``kind``, picture size, ``m``
    (metres per pixel) and ``center_S`` (frame S at the picture centre; for views in the
    levelled input's framing these come from ``ortho``, which also gives the shared
    ``frame``), and the colour match already applied.  Pick ``back.png`` as the 'Back photo'
    (source ``images``) to re-run with them: :func:`views_from_folder` + :func:`load_views`
    restore the cameras, so an edited view goes through the same orthographic checks.

    ``views`` are raw (:func:`views_from_cameras`) or registered (``BackPart.views``); of two
    views at one azimuth the one the back was built with is saved and the other is skipped
    (logged).  Returns ``folder``.  :class:`InputError` when
    there is nothing to save; ``OSError`` from writing propagates."""
    out = Path(folder).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    recs: list[dict[str, Any]] = []
    top_extra: dict[str, Any] = {}
    for v in views:
        if not isinstance(v, GenView):
            raise InputError(f"Expected GenView objects (got {type(v).__name__}).")
    # one picture per azimuth: the first view the back was built with, else the first one
    chosen: dict[str, int] = {}
    for i, v in enumerate(views):
        file = f"{_phi_name(v.phi_deg)}.png"
        j = chosen.get(file)
        if j is None or (views[j].meta.get("used") is False and v.meta.get("used") is not False):
            chosen[file] = i
    for i, v in enumerate(views):
        name = _phi_name(v.phi_deg)
        file = f"{name}.png"
        if chosen[file] != i:
            log.warning(
                "Saving views: %s (%s) was not saved - %s holds another view at %.0f deg",
                v.name, v.meta.get("file", "?"), file, v.phi_deg,
            )  # fmt: skip
            continue
        rgb = _rgb_u8(v.rgb, f"view '{v.name}'")
        a = np.clip(np.nan_to_num(np.asarray(v.alpha, dtype=np.float32)), 0.0, 1.0)[..., None]
        comp = np.clip(np.rint(rgb * a + _GREY * (1.0 - a)), 0, 255).astype(np.uint8)
        save_image(out / file, comp)
        h, w = comp.shape[:2]
        rec: dict[str, Any] = {
            "file": file,
            "phi_deg": float(v.phi_deg),
            "kind": v.kind,
            "width": w,
            "height": h,
        }
        if v.kind == "ortho":
            m, centre = _ortho_framing_of(v, ortho)
            if m is not None:
                rec["m"] = m
            if centre is not None:
                rec["center_S"] = centre
            for key in ("ortho_half_extent", "input_fill"):
                if key in v.meta and key not in top_extra:
                    top_extra[key] = v.meta[key]
        if v.meta.get("color_match"):
            rec["color_match"] = str(v.meta["color_match"])
        if v.meta.get("used") is False:
            rec["used"] = False
            if v.meta.get("reason"):
                rec["reason"] = str(v.meta["reason"])
        if v.meta.get("file"):
            rec["source_file"] = str(v.meta["file"])
        recs.append(rec)
    if not recs:
        raise InputError("There are no views to save.")
    doc: dict[str, Any] = {
        "format": VIEWS_FORMAT,
        "version": 1,
        "projection": "orthographic" if any(r["kind"] == "ortho" for r in recs) else "nominal",
        **top_extra,
    }
    if ortho is not None:
        doc["frame"] = {"size": int(ortho.size), "m": float(ortho.m_in), "center_S": _pair(ortho.center_S)}
    doc["views"] = recs
    atomic_write_bytes(out / VIEWS_JSON, (json.dumps(_jsonable(doc), indent=2) + "\n").encode("utf-8"))
    log.info("Saved %d view(s) to %s", len(recs), out)
    return out


# =========================================================================== V1 segmentation
def _accepts_cancel(fn: Callable[..., Any]) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "cancel" in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _key_flat_background(rgb: np.ndarray) -> np.ndarray:
    """Matte by keying out a flat background: its colour is the median of a 4 px border
    ring; pixels differing from it by more than 0.06 (any channel, 0-1 units) are subject."""
    h, w = rgb.shape[:2]
    r = min(_KEY_RING_PX, max(1, min(h, w) // 4))
    ring = np.ones((h, w), bool)
    ring[r : h - r, r : w - r] = False
    bg = np.median(rgb[ring].reshape(-1, 3).astype(np.float64), axis=0)
    diff = np.abs(rgb.astype(np.float64) - bg).max(axis=2) / 255.0
    return (diff > _KEY_THRESH).astype(np.float32)


def _too_small(mask: np.ndarray, kind: str) -> str:
    """Why a view's matte is too small to use ("" when it is not): empty, under 2 % of a
    generated (``"ortho"``) view, or for a user's photo under 64 pixels across or 400 of
    area (the picture is not required to be filled: a person 3 m away covers 2 %)."""
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n == 0:
        return "no subject was found"
    cover = n / m.size
    if kind == "ortho":
        ok = cover >= _SEG_MIN_COVER
    else:
        rows = np.flatnonzero(m.any(axis=1))
        cols = np.flatnonzero(m.any(axis=0))
        across = max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
        ok = across >= _SEG_MIN_PX and n >= _SEG_MIN_AREA_PX
    if ok:
        return ""
    return (
        f"the subject is too small in the picture (it fills {100 * cover:.1f} % of it) - take it from "
        f"closer or crop the picture around the subject"
    )


def segment_views(
    views: Sequence[GenView],
    segmenter: Any,
    *,
    cancel: CancelToken | None = None,
    dropped: list[str] | None = None,
) -> list[GenView]:
    """V1: a matte for every view.

    A matte that came with the picture (``meta["alpha_from_file"]``) is kept; otherwise
    ``segmenter.segment(rgb)`` (the app's segmenter; None skips it).  When it fails or
    finds too little (:func:`_too_small`), the flat background is keyed out instead
    (``|rgb - bg| > 0.06`` with ``bg`` from a 4 px border ring - generated views sit on a
    plain background).  Then the largest component, holes filled and a 1 px feather
    (``clean_matte``); views whose matte is too small or covers more than 97 % (no
    background) are dropped, with a plain sentence appended to ``dropped`` (an addition)
    that tells "no subject" from "too small".  "Too small" is the design's 2 % of the
    picture for generated (``"ortho"``) views, which fill their frame; a user's photo
    (``"nominal"``) needs a subject at least 64 pixels across (and 400 pixels of area): a
    slim person seen side-on in a whole-body portrait covers about 2 %.
    ``meta["matte"]`` says which method won.  :class:`Cancelled` propagates."""
    out: list[GenView] = []
    for v in views:
        _check(cancel)
        rgb = _rgb_u8(v.rgb, f"view '{v.name}'")
        h, w = rgb.shape[:2]
        method = ""
        raw: np.ndarray | None = None
        if v.meta.get("alpha_from_file"):
            raw, method = to_alpha(np.asarray(v.alpha, dtype=np.float32)), "file"
        elif segmenter is not None:
            try:
                fn = segmenter.segment
                res = fn(rgb, cancel=cancel) if _accepts_cancel(fn) else fn(rgb)
                a = to_alpha(np.asarray(res, dtype=np.float32))
                if a.shape != (h, w):
                    a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                if not _too_small(a >= 0.5, v.kind):
                    raw, method = a, "segmenter"
                else:
                    log.info("View %s: the segmenter found almost nothing; keying the background", v.name)
            except Cancelled:
                raise
            except Exception as e:  # a model failure must not lose the view
                log.warning("View %s: segmentation failed (%s); keying the background", v.name, e)
        if raw is None:
            raw, method = _key_flat_background(rgb), "background key"
        matte = clean_matte(raw, threshold=0.5, feather_px=1.0, largest_only=True, max_hole_frac=0.02)
        cover = float((matte >= 0.5).mean())
        small = _too_small(matte >= 0.5, v.kind)
        if small or cover > _SEG_MAX_COVER:
            why = small or "the background could not be separated"
            msg = f"The {v.name} picture ({v.meta.get('file', v.name)}) was not used: {why}."
            log.warning(msg)
            if dropped is not None:
                dropped.append(msg)
            continue
        out.append(
            replace(
                v,
                rgb=rgb,
                alpha=matte.astype(np.float32),
                meta={**v.meta, "matte": method, "coverage": cover},
            )
        )
    return out


# =========================================================================== V2 / V3 registration
def _frame(ortho: OrthoInput, canon: Canon) -> _Frame:
    a0 = np.asarray(ortho.alpha0, dtype=bool)
    S = int(ortho.size)
    if a0.shape != (S, S) or not a0.any():
        raise InputError("The levelled front silhouette is missing or empty.")
    m = float(ortho.m_in)
    if not (math.isfinite(m) and m > 0):
        raise InputError(f"Invalid levelled picture scale {ortho.m_in!r}.")
    cx, cy = (float(t) for t in np.asarray(ortho.center_S, dtype=np.float64).ravel()[:2])
    rows = np.flatnonzero(a0.any(axis=1))
    cols = np.flatnonzero(a0.any(axis=0))
    L_px = float(max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1))
    return _Frame(
        S=S,
        m=m,
        cx=cx,
        cy=cy,
        R=np.asarray(canon.R, dtype=np.float64),
        c=np.asarray(canon.c, dtype=np.float64),
        alpha0=a0,
        L_px=L_px,
    )


def _composed_map(tps: Any, G: np.ndarray, S: int) -> tuple[np.ndarray, np.ndarray]:
    """``(map_x, map_y)`` sending output pixel ``q`` to ``G^-1(tps(q))`` (``G`` the 3 x 3
    similarity-then-perspective map of the source into the target): the TPS lock and the
    global fit in one resampling (a 4 px lattice, bilinearly refined, above 256 x 256)."""
    Gi = np.linalg.inv(np.asarray(G, dtype=np.float64))

    def src(q: np.ndarray) -> np.ndarray:
        p = tps(q)
        w = p @ Gi[2, :2] + Gi[2, 2]
        w = np.where(np.abs(w) > 1e-9, w, 1e-9)  # never reached inside the frame (G is bounded)
        return (p @ Gi[:2, :2].T + Gi[:2, 2]) / w[:, None]

    if S * S <= _TPS_EXACT_PX:
        gy, gx = np.mgrid[0:S, 0:S]
        p = src(np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64))
        return p[:, 0].reshape(S, S).astype(np.float32), p[:, 1].reshape(S, S).astype(np.float32)
    k = _TPS_LATTICE
    nc = (S - 1) // k + 2
    gy, gx = np.mgrid[0:nc, 0:nc]
    p = src(np.stack([gx.ravel() * k, gy.ravel() * k], axis=1).astype(np.float64))
    cxm = p[:, 0].reshape(nc, nc).astype(np.float32)
    cym = p[:, 1].reshape(nc, nc).astype(np.float32)
    fy, fx = np.mgrid[0:S, 0:S].astype(np.float32) / np.float32(k)
    return cv2.remap(cxm, fx, fy, cv2.INTER_LINEAR), cv2.remap(cym, fx, fy, cv2.INTER_LINEAR)


def _prefilter(img: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """``(img, q)``: ``img`` area-downscaled by about ``scale`` when the registration shrinks
    it (OpenCV's warps sample bilinearly without an area filter, which aliases fine texture
    such as hair), ``q = (qx, qy)`` the exact factors (``1`` otherwise).  A source pixel
    ``x`` is then at ``(x + 0.5) qx - 0.5``."""
    if scale >= 0.75:
        return img, np.ones(2)
    h, w = img.shape[:2]
    dsize = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, dsize, interpolation=cv2.INTER_AREA), np.array([dsize[0] / w, dsize[1] / h])


def _registered(v: GenView, rgb: np.ndarray, alpha: np.ndarray, fr: _Frame, **meta: Any) -> GenView:
    return replace(
        v,
        rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
        alpha=np.ascontiguousarray(alpha, dtype=np.float32),
        m=float(fr.m),
        depth=None,
        meta={**v.meta, "registered": True, "frame_size": fr.S, **meta},
    )


def _contour_pairs(src: np.ndarray, dst: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """``(src_pts, dst_pts)``: ``n`` control points by arc length on the target outline (from
    the top, clockwise - :func:`~vrphoto.geometry.register.contour_points`) and their
    partners on the source outline.

    Plain arc-length pairing drifts by the difference of the two outlines' lengths (a 1 %
    longer outline is ~10 px off by the end), which bends the lock at the arm tips and the
    feet.  Each partner is therefore the source outline point nearest to the target point
    within a window of +-4 % of the outline around its arc-length position (a window, not a
    global search, so a point never jumps to the other side of a thin arm or leg).

    Each outline starts at its own topmost point, which is a different feature on the two
    when two of them are nearly level (a raised hand 4 mm below the head top): every pair
    would then be wrong.  Both outlines are in the same frame here, so the source outline is
    first turned to the cyclic start that best matches the target points as a whole."""
    dst_pts = contour_points(dst, n)
    dense_n = max(4 * n, int(round(cv2.arcLength(np.asarray(dst_pts, np.float32), True))))
    dense = contour_points(src, dense_n)
    centre = np.rint(np.arange(n) * dense_n / n).astype(np.int64)
    off = np.arange(dense_n)
    cost = np.empty(dense_n)
    for a in range(0, dense_n, 256):  # (offsets x n x 2) in blocks: a few MB at most
        o = off[a : a + 256]
        cost[a : a + 256] = ((dense[(centre[None, :] + o[:, None]) % dense_n] - dst_pts) ** 2).sum(
            axis=(1, 2)
        )
    dense = np.roll(dense, -int(np.argmin(cost)), axis=0)
    win = max(2, int(round(_PAIR_WINDOW * dense_n)))
    cand = (centre[:, None] + np.arange(-win, win + 1)[None, :]) % dense_n
    d2 = ((dense[cand] - dst_pts[:, None, :]) ** 2).sum(axis=2)
    pick = cand[np.arange(n), np.argmin(d2, axis=1)]
    return dense[pick], dst_pts


def _tps_lock(
    rgb_v: np.ndarray,
    a_v: np.ndarray,
    a_g: np.ndarray,
    target: np.ndarray,
    G: np.ndarray,
    shrink: float,
    fr: _Frame,
) -> tuple[np.ndarray, np.ndarray, float]:
    """V3 lock: a 64-point thin-plate spline from the globally fitted matte ``a_g`` onto
    ``target``, composed with the global map ``G`` (3 x 3, source picture -> frame) into one
    resampling of the source picture (area-prefiltered by about ``shrink`` when it shrinks).
    Returns ``(rgb, alpha, largest spline move in px)``."""
    S = fr.S
    src_pts, dst_pts = _contour_pairs(a_g >= 0.5, target, _TPS_POINTS)
    tps = tps_fit(src_pts, dst_pts, lam=_TPS_LAM)
    disp = tps.max_displacement((S, S), mask=target)
    mx, my = _composed_map(tps, G, S)
    rgb_q, q = _prefilter(rgb_v, shrink)
    rgb_l = cv2.remap(
        rgb_q,
        ((mx + 0.5) * q[0] - 0.5).astype(np.float32),
        ((my + 0.5) * q[1] - 0.5).astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(_GREY,) * 3,
    )
    a_l = cv2.remap(a_v, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rgb_l, a_l, float(disp)


def _lock_back(v: GenView, fr: _Frame, cancel: CancelToken | None) -> tuple[GenView, dict[str, Any]]:
    """V2 + V3 for the back view: similarity (roll <= 5 deg) to the mirrored front
    silhouette, perspective stage (a camera turn), IoU gate, pose gate (the outline gap
    after that global fit), TPS lock.  Returns the view in the levelled back frame
    (``offset = -x0``) and the diagnostics (``ok``, ``reason``)."""
    S = fr.S
    target = fr.alpha0[:, ::-1]
    a_v = np.clip(np.nan_to_num(np.asarray(v.alpha, dtype=np.float32)), 0.0, 1.0)
    info: dict[str, Any] = {"name": v.name, "phi": float(v.phi_deg), "file": v.meta.get("file"), "ok": False}
    if not (a_v >= 0.5).any():
        info["reason"] = "the back picture shows no subject"
        return replace(v, meta={**v.meta, "used": False, "reason": info["reason"]}), info
    params, cost = chamfer_similarity_fit(
        a_v >= 0.5, target, allow_roll=True, max_roll_deg=_MAX_ROLL_DEG, cancel=cancel
    )
    M = _similarity_affine(params)
    a_sim = cv2.warpAffine(
        a_v, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    iou_sim = silhouette_iou(a_sim >= 0.5, target)
    _check(cancel)
    # perspective stage: a photo taken from above (or below) the subject's middle is a smooth
    # keystone of the square-on view that no similarity absorbs; left in, the pose check
    # below would measure it as a different pose
    Hp, cost_h, _ = chamfer_perspective_fit(a_sim >= 0.5, target, max_keystone=_MAX_KEYSTONE, cancel=cancel)
    G = Hp @ np.vstack([M, [0.0, 0.0, 1.0]])
    a_g = cv2.warpPerspective(
        a_v, G, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    iou = silhouette_iou(a_g >= 0.5, target)
    info.update(
        similarity={k: round(float(x), 5) for k, x in params.items()},
        perspective=[[round(float(x), 7) for x in row] for row in Hp],
        chamfer_px=round(float(cost), 3),
        chamfer_perspective_px=round(float(cost_h), 3),
        iou_similarity=round(iou_sim, 4),
        iou=round(iou, 4),
    )
    rgb_v = _rgb_u8(v.rgb, f"view '{v.name}'")

    def unused(reason: str) -> tuple[GenView, dict[str, Any]]:
        info["reason"] = reason
        rgb_s = cv2.warpPerspective(
            rgb_v, G, (S, S), borderMode=cv2.BORDER_CONSTANT, borderValue=(_GREY,) * 3
        )
        return _registered(v, rgb_s, a_g, fr, offset_m=-fr.cx, used=False, reason=reason), info

    if iou < _BACK_IOU_GATE:
        return unused(
            f"the back picture does not match the photo's outline (overlap {100 * iou:.0f} %, "
            f"at least {100 * _BACK_IOU_GATE:.0f} % is needed)"
        )
    # the pose check: how far apart the outlines still are after the global (similarity +
    # perspective) fit - what the lock would have to move, whatever pairs it picks - at a
    # high percentile (a fingertip or a strand of hair may stick out, a limb in another
    # place is a long stretch of the outline)
    gap = outline_gap(a_g >= 0.5, target, _POSE_GAP_PCT)
    frac = gap / fr.L_px
    info.update(pose_px=round(gap, 3), pose_frac=round(frac, 4))
    if frac > _POSE_MAX_FRAC:
        return unused(
            f"the back picture shows a different pose (after lining it up, parts of its outline are "
            f"still {100 * frac:.0f} % of its size away from the photo's; at most "
            f"{100 * _POSE_MAX_FRAC:.0f} % is accepted)"
        )
    _check(cancel)
    # the source is shrunk by about the similarity's scale (the perspective stage is close
    # to the identity), so the area prefilter keys on it
    rgb_l, a_l, disp = _tps_lock(rgb_v, a_v, a_g, target, G, float(params["s"]), fr)
    info.update(tps_max_px=round(float(disp), 3), tps_max_frac=round(float(disp / fr.L_px), 4))
    iou_l = silhouette_iou(a_l >= 0.5, target)
    info.update(ok=True, iou_locked=round(iou_l, 4))
    log.info(
        "Back view %s: similarity s=%.3f roll=%.1f deg, IoU %.3f -> %.3f (perspective) -> %.3f after the "
        "lock (outlines %.1f %% apart, lock move up to %.1f %%)",
        v.name,
        params["s"],
        params["roll_deg"],
        iou_sim,
        iou,
        iou_l,
        100 * frac,
        100 * disp / fr.L_px,
    )
    out = _registered(v, rgb_l, a_l, fr, offset_m=-fr.cx, used=True, reason="", iou=iou, iou_locked=iou_l)
    return out, info


def _preregistered(v: GenView, fr: _Frame) -> bool:
    """A nominal view already in this levelled frame (``S x S``, same ``m``, a finite offset):
    the registered views of an earlier run.  (Orthographic sets always go through their own
    checks, which leave an already registered set as it is.)"""
    try:
        o = float(v.meta.get("offset_m", math.nan))
    except (TypeError, ValueError):
        return False
    return (
        bool(v.meta.get("registered"))
        and np.shape(v.alpha) == (fr.S, fr.S)
        and v.m is not None
        and abs(float(v.m) - fr.m) <= 1e-6 * fr.m
        and math.isfinite(o)
    )


def _touches(alpha: np.ndarray) -> tuple[bool, bool]:
    """Whether a matte reaches its picture's top / bottom border (within 2 px): the subject
    is probably cut off there, so that row is not the subject's top / bottom."""
    rows = np.flatnonzero((np.asarray(alpha) >= 0.5).any(axis=1))
    if not len(rows):
        return False, False
    return bool(rows[0] < _BORDER_PX), bool(rows[-1] >= alpha.shape[0] - _BORDER_PX)


def _front_edge_rows(phi: float, P_S: np.ndarray, fr: _Frame) -> np.ndarray:
    """``r_f(j)`` of :func:`_offset_rows`: per levelled row, the front points' extreme
    ``P.w`` on the side of a view at ``phi`` (NaN where unknown)."""
    sgn = 1.0 if _sin(phi) > 0 else -1.0
    w, _ = _axes(phi)
    rows = np.floor((P_S[:, 1] - fr.cy) / fr.m + fr.S / 2.0).astype(np.int64)
    return sgn * _row_percentile(rows, sgn * (P_S @ w), fr.S, _EDGE_PCT)


def _search_side_scale(
    a_v: np.ndarray,
    phi: float,
    anchor_v: float,
    anchor_0: float,
    bracket: tuple[float, float],
    P_S: np.ndarray,
    fr: _Frame,
) -> tuple[float, float]:
    """``(scale, spread)``: the scale of a side view that is cut off (or whose photo is):
    with one invariant row (``anchor_v`` in the view, ``anchor_0`` in the levelled frame)
    the scale is the one that makes the V5 front-edge offsets of the rows agree best (their
    median absolute deviation, m), searched within ``bracket`` for the named azimuth and its
    mirror (the left / right check comes later).  ``(nan, inf)`` when no scale fits."""
    left, right = _row_edges(a_v)
    h = a_v.shape[0]
    lo, hi = (float(t) for t in bracket)
    vq = np.arange(fr.S) + 0.5  # levelled row centres

    def spread(scales: np.ndarray) -> np.ndarray:
        best = np.full(len(scales), np.inf)
        for ph in (phi, (360.0 - phi) % 360.0):
            r_f = _front_edge_rows(ph, P_S, fr)
            need = max(5, int(math.ceil(_MIN_ROWS_FRAC * int(np.isfinite(r_f).sum()))))
            u_src = left if _sin(ph) > 0 else right
            i = np.floor((vq[None, :] - anchor_0) / scales[:, None] + anchor_v).astype(np.int64)
            inside = (i >= 0) & (i < h)
            u = np.where(inside, u_src[np.clip(i, 0, h - 1)], np.nan)
            # o(j) = r_f(j) - s (u_f(j) - c) m; the constant c does not change the spread
            o = r_f[None, :] - scales[:, None] * u * fr.m
            n = np.isfinite(o).sum(axis=1)
            enough = n >= need
            o = np.where(enough[:, None], o, 0.0)  # too few rows: never chosen (and no all-NaN rows)
            med = np.nanmedian(o, axis=1)
            mad = np.nanmedian(np.abs(o - med[:, None]), axis=1)
            best = np.minimum(best, np.where(enough, mad, np.inf))
        return best

    # one dense global grid (0.2 % steps over a factor of 2): the spread is a median over
    # rows, piecewise flat with shallow local minima, so a coarse grid refined locally can
    # settle a percent or two off
    grid = np.exp(np.linspace(math.log(lo), math.log(hi), _SCALE_STEPS))
    sp = spread(grid)
    k = int(np.argmin(sp))
    if not math.isfinite(float(sp[k])):
        return math.nan, math.inf
    return float(grid[k]), float(sp[k])


def _register_side(
    v: GenView, fr: _Frame, P_S: np.ndarray, cut0: tuple[bool, bool] = (False, False)
) -> tuple[GenView | None, dict[str, Any]]:
    """V2 for a nominal side / quarter view: the invariant top and bottom rows give the
    scale and vertical shift; the silhouette's box centre goes to the middle column (the
    real column offset is solved by V5).

    Both rows must be the subject's own: a view whose matte touches its picture's top or
    bottom border, or a photo cut off there (``cut0`` = the photo's ``Canon.cut_top`` /
    ``cut_bottom``), keeps only the other row as the anchor and takes the scale that makes
    its front edge follow the photo's best (:func:`_search_side_scale`) - the row extent
    alone would scale a picture with the feet cut off 20 % too large, and a fifth more
    thickness would follow.  A view with no reliable row at all is not used."""
    info: dict[str, Any] = {
        "name": v.name,
        "phi": float(v.phi_deg),
        "file": v.meta.get("file"),
        "used": False,
    }
    a_v = np.clip(np.nan_to_num(np.asarray(v.alpha, dtype=np.float32)), 0.0, 1.0)
    tb_v = _top_bottom(a_v)
    tb_0 = _top_bottom(fr.alpha0.astype(np.float32))
    if tb_v is None or tb_0 is None or tb_v[1] - tb_v[0] < 4:
        info["reason"] = "no subject in the picture"
        return None, info
    cut_v = _touches(a_v)
    top_ok = not (cut_v[0] or cut0[0])
    bottom_ok = not (cut_v[1] or cut0[1])
    s_rows = (tb_0[1] - tb_0[0]) / (tb_v[1] - tb_v[0])
    info.update(cut_top=bool(cut_v[0]), cut_bottom=bool(cut_v[1]))
    if top_ok and bottom_ok:
        s, anchor = s_rows, "rows"
        a_src, a_dst = tb_v[0], tb_0[0]
    elif top_ok or bottom_ok:
        a_src, a_dst = (tb_v[0], tb_0[0]) if top_ok else (tb_v[1], tb_0[1])
        anchor = "top" if top_ok else "bottom"
        # the cut picture shows less of the subject than the uncut one: the row-extent scale
        # is an upper bound when only the view is cut, a lower bound when only the photo is
        view_cut, photo_cut = cut_v[0] or cut_v[1], cut0[0] or cut0[1]
        if view_cut and not photo_cut:
            bracket = (_CUT_MIN_SHOWN * s_rows, s_rows)
        elif photo_cut and not view_cut:
            bracket = (s_rows, s_rows / _CUT_MIN_SHOWN)
        else:
            bracket = (_CUT_MIN_SHOWN * s_rows, s_rows / _CUT_MIN_SHOWN)
        s, mad = _search_side_scale(a_v, float(v.phi_deg), a_src, a_dst, bracket, P_S, fr)
        info.update(scale_rows=round(float(s_rows), 5), scale_spread_m=round(float(mad), 5))
        if not math.isfinite(s):
            info["reason"] = "it does not show enough of the subject to be lined up"
            return None, info
    else:
        where = (
            "the photo and the picture"
            if (cut0[0] or cut0[1]) and (cut_v[0] or cut_v[1])
            else ("the photo" if (cut0[0] or cut0[1]) else "the picture")
        )
        info["reason"] = (
            f"it cannot be lined up: {where} cut the subject off at the top and the bottom, so no "
            f"height in it is known"
        )
        return None, info
    if not (_SCALE_RANGE[0] <= s <= _SCALE_RANGE[1]):
        info["reason"] = f"its size is implausible (scale {s:.2f})"
        return None, info
    left, right = _row_edges(a_v)
    c_k = 0.5 * (float(np.nanmin(left)) + float(np.nanmax(right)))
    S = fr.S
    # continuous u' = s (u - c_k) + S/2, v' = s (v - a_src) + a_dst; OpenCV coords are u - 0.5
    M = np.array(
        [[s, 0.0, s * (0.5 - c_k) + S / 2.0 - 0.5], [0.0, s, s * (0.5 - a_src) + a_dst - 0.5]], np.float64
    )
    rgb_q, q = _prefilter(_rgb_u8(v.rgb, f"view '{v.name}'"), s)
    Mq = M.copy()  # the same map from the prefiltered picture: x = (x_q + 0.5) / q - 0.5
    Mq[:, :2] /= q[None, :]
    Mq[:, 2] += M[:, :2] @ (0.5 * (1.0 / q - 1.0))
    rgb = cv2.warpAffine(
        rgb_q, Mq, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(_GREY,) * 3
    )
    a = cv2.warpAffine(a_v, M, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    info.update(scale=round(float(s), 5), anchor=anchor, used=True)
    if anchor != "rows":
        log.info(
            "Side view %s is cut off: scale %.3f from its front edge (rows alone: %.3f)", v.name, s, s_rows
        )
    return _registered(v, rgb, a, fr, scale=float(s), anchor=anchor, used=True, reason=""), info


# =========================================================================== V4 colour
def _rim_band(alpha: np.ndarray, width_px: float) -> np.ndarray:
    m = np.asarray(alpha) >= 0.5
    d = _dist_in(m)
    return m & (d >= 1.5) & (d <= max(2.5, width_px))


def _rim_transfer(
    img: np.ndarray, band: np.ndarray, ref: np.ndarray, ref_band: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Reinhard LAB transfer at strength 0.5 (as :func:`~vrphoto.geometry.register.reinhard_lab`)
    with the colour part bounded: lightness gets the mean / std transfer (exposure,
    contrast), the ``a`` / ``b`` channels only a mean shift clamped to +-6 units - white
    balance moves them by a few units, while a large difference between the rim bands is a
    real one (a jacket's back, a backpack) that must keep its colour.  Returns the picture
    and the applied shifts."""
    lab = cv2.cvtColor(img.astype(np.float32) / np.float32(255.0), cv2.COLOR_RGB2Lab)
    lab_r = cv2.cvtColor(ref.astype(np.float32) / np.float32(255.0), cv2.COLOR_RGB2Lab)
    if int(band.sum()) < 16 or int(ref_band.sum()) < 16:
        return img.copy(), {"skipped": True}
    mu, sd = lab[band].mean(axis=0, dtype=np.float64), lab[band].std(axis=0, dtype=np.float64)
    mu_r, sd_r = lab_r[ref_band].mean(axis=0, dtype=np.float64), lab_r[ref_band].std(axis=0, dtype=np.float64)
    k = _REINHARD_STRENGTH
    ratio = np.clip(sd_r / np.maximum(sd, 1e-6), 0.5, 2.0)
    gain = 1.0 + k * (ratio - 1.0)
    gain[1:] = 1.0  # the chroma spread is the garments', not the camera's
    shift = k * (mu_r - mu)
    shift[1:] = np.clip(shift[1:], -_AB_SHIFT_MAX, _AB_SHIFT_MAX)
    out = (lab - mu) * gain.astype(np.float32) + (mu + shift).astype(np.float32)
    rgb = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_Lab2RGB)
    info = {"L_shift": float(shift[0]), "a_shift": float(shift[1]), "b_shift": float(shift[2])}
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), info


def _match_colours(views: list[GenView], ortho: OrthoInput, fr: _Frame) -> list[GenView]:
    """V4 (nominal): Reinhard LAB transfer, strength 0.5, from each view's rim band toward
    the photo's rim band (within 3 % of the subject's size of the outline); see
    :func:`_rim_transfer` for the bounded colour shift."""
    width = _RIM_BAND_FRAC * fr.L_px
    ref_rgb = np.asarray(ortho.rgba)[..., :3]
    ref_band = _rim_band(fr.alpha0, width)
    out = []
    for v in views:
        rgb, info = _rim_transfer(v.rgb, _rim_band(v.alpha, width), ref_rgb, ref_band)
        out.append(replace(v, rgb=rgb, meta={**v.meta, "color_match": "reinhard", "color_shift": info}))
    return out


# =========================================================================== V5 offsets
def _front_points(front: FrontModel, D_f: np.ndarray, matte: np.ndarray) -> np.ndarray:
    """C points of the photographed front at the mould-cell centres of the matte."""
    u, v = front.grid.centers()
    rays = front.cam.rays(u[matte], v[matte])
    return rays * D_f[matte][:, None]


def _row_percentile(rows: np.ndarray, vals: np.ndarray, n_rows: int, pct: float) -> np.ndarray:
    """Per row (``0 .. n_rows-1``) the ``pct`` percentile (nearest rank) of ``vals``; NaN
    for rows with fewer than 3 values."""
    out = np.full(n_rows, np.nan)
    ok = (rows >= 0) & (rows < n_rows) & np.isfinite(vals)
    r, x = rows[ok], vals[ok]
    if not len(r):
        return out
    order = np.lexsort((x, r))
    r, x = r[order], x[order]
    starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
    counts = np.diff(np.r_[starts, len(r)])
    pick = starts + np.floor(pct / 100.0 * (counts - 1)).astype(np.int64)
    good = counts >= _MIN_ROW_PTS
    out[r[starts[good]]] = x[pick[good]]
    return out


def _offset_rows(
    alpha: np.ndarray, phi: float, P_S: np.ndarray, fr: _Frame, bias: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Per row the offset estimate ``o(j) = r_f(j) - (u_f(j) - S/2) m`` of a view at ``phi``
    (``r = P.w``; the silhouette edge facing the photo camera against the front points'
    extreme ``r``) and the opposite edge ``u_b(j)`` (NaN where unknown).  ``bias`` (px,
    an addition): the matte's outline lies this far outside the subject's (a keyed matte
    of an orthographic set, measured on view 0), so both edges are moved inward by it."""
    sgn = 1.0 if _sin(phi) > 0 else -1.0
    w, _ = _axes(phi)
    rows = np.floor((P_S[:, 1] - fr.cy) / fr.m + fr.S / 2.0).astype(np.int64)
    r_f = sgn * _row_percentile(rows, sgn * (P_S @ w), fr.S, _EDGE_PCT)
    left, right = _row_edges(alpha)
    if bias:
        left, right = left + bias, right - bias
    u_f, u_b = (left, right) if sgn > 0 else (right, left)
    return r_f - (u_f - fr.S / 2.0) * fr.m, u_b


def _hypotheses(v: GenView, P_S: np.ndarray, fr: _Frame) -> list[dict[str, Any]]:
    """The V5 offset of a view as named (``phi``) and mirrored (``360 - phi``): offset
    (median over rows), row MAD and misfit (90th percentile of the row deviations)."""
    sil_rows = int((np.asarray(v.alpha) >= 0.5).any(axis=1).sum())
    need = max(5, int(math.ceil(_MIN_ROWS_FRAC * sil_rows)))
    hyp = []
    for phi in (float(v.phi_deg), (360.0 - float(v.phi_deg)) % 360.0):
        o_rows, _ = _offset_rows(v.alpha, phi, P_S, fr)
        ok = np.isfinite(o_rows)
        if int(ok.sum()) < need:
            hyp.append(
                {
                    "phi": phi,
                    "offset_m": math.nan,
                    "mad_m": math.inf,
                    "misfit_m": math.inf,
                    "rows": int(ok.sum()),
                }
            )
            continue
        o = float(np.median(o_rows[ok]))
        dev = np.abs(o_rows[ok] - o)
        hyp.append(
            {
                "phi": phi,
                "offset_m": o,
                "mad_m": float(np.median(dev)),
                "misfit_m": float(np.percentile(dev, _FLIP_PCT)),
                "rows": int(ok.sum()),
            }
        )
    return hyp


def _flip_decisions(views: list[GenView], hyps: list[list[dict[str, Any]]]) -> tuple[list[bool], list[tuple]]:
    """Which views to turn round (used as ``360 - phi``), and the complementary pairs that
    were judged together.

    The mirror hypothesis (the usual left / right naming mix-up) is judged on the worst rows:
    front and back of a body have much the same outline except where something sticks out
    (a bag, an arm in front, the face), so the median (MAD) barely changes while the 90th
    percentile of the row offsets does.  A view alone is turned round when its mirror
    misfit is under half its own.  Two views named as each other's mirror (right / left,
    back_right / back_left, front_right / front_left) are judged TOGETHER - both or neither,
    on the summed misfits: in a perspective photo an arm raised toward one camera is
    magnified and can make one picture fit its wrong name, and turning only the other one
    would put both on the same side.  When the pair's sums do not decide but one picture
    alone would have been turned, neither is (the caller warns)."""
    n = len(views)
    flip = [False] * n

    def single(h: list[dict[str, Any]]) -> bool:
        a, b = h
        return (
            math.isfinite(b["misfit_m"])
            and a["misfit_m"] > _FLIP_MIN_MISFIT_M
            and b["misfit_m"] < _FLIP_RATIO * a["misfit_m"]
            and b["phi"] != a["phi"]
        )

    pairs: list[tuple] = []
    paired: set[int] = set()
    for i in range(n):
        if i in paired or views[i].meta.get("offset_given"):
            continue
        for j in range(i + 1, n):
            if j in paired or views[j].meta.get("offset_given"):
                continue
            if _az_diff(views[j].phi_deg, 360.0 - views[i].phi_deg) < 1.0 and _sin(views[i].phi_deg) != 0:
                named = hyps[i][0]["misfit_m"] + hyps[j][0]["misfit_m"]
                swapped = hyps[i][1]["misfit_m"] + hyps[j][1]["misfit_m"]
                both = (
                    math.isfinite(swapped)
                    and named > 2 * _FLIP_MIN_MISFIT_M
                    and swapped < _FLIP_RATIO * named
                )
                alone = single(hyps[i]) or single(hyps[j])
                flip[i] = flip[j] = bool(both)
                pairs.append((i, j, named, swapped, "both" if both else ("unsure" if alone else "as named")))
                paired.update((i, j))
                break
    for i in range(n):
        if i not in paired and not views[i].meta.get("offset_given"):
            flip[i] = single(hyps[i])
    return flip, pairs


def _solve_offsets(
    views: list[GenView], P_S: np.ndarray, fr: _Frame, canon: Canon, warnings: list[str]
) -> tuple[list[GenView], dict[str, Any]]:
    """V5 for nominal side / quarter views: per-view offset from the front edge (median over
    rows, MAD gate), the left/right naming check (mirror hypothesis ``360 - phi``, mirrored
    pairs judged together: :func:`_flip_decisions`), at most one view per azimuth (the
    better-fitting one) and the back-edge agreement of the left and right views.  Returns
    the views (offsets set, rejected ones with ``meta["used"] = False``) and the
    diagnostics."""
    H = float(canon.height_s)
    mad_max = max(_MAD_MAX_M, _MAD_MAX_FRAC * H)
    info: dict[str, Any] = {"views": []}
    out: list[GenView] = []
    hyps = [[] if v.meta.get("offset_given") else _hypotheses(v, P_S, fr) for v in views]
    flips, pairs = _flip_decisions(views, hyps)
    info["pairs"] = [
        {"views": [views[i].name, views[j].name], "named_m": a, "swapped_m": b, "decision": d}
        for i, j, a, b, d in pairs
    ]
    for i, j, _, _, d in pairs:
        if d == "unsure":
            msg = (
                f"The left / right naming of the {views[i].name} and {views[j].name} pictures "
                f"({views[i].meta.get('file', views[i].name)}, {views[j].meta.get('file', views[j].name)}) "
                f"could not be confirmed from their outlines, so they were used as named. ('Right' means you "
                f"walked to your right around the subject.)"
            )
            warnings.append(msg)
            log.warning(msg)
    for v, hyp, flip in zip(views, hyps, flips, strict=True):
        rec: dict[str, Any] = {"name": v.name, "phi": float(v.phi_deg)}
        if v.meta.get("offset_given"):
            o = float(v.meta["offset_m"])
            rec.update(phi_used=float(v.phi_deg), offset_m=o, mad_m=0.0, rows=0, used=True, given=True)
            out.append(replace(v, meta={**v.meta, "mad_m": float(v.meta.get("mad_m", 0.0))}))
            info["views"].append(rec)
            continue
        rec["hypotheses"] = hyp
        h = hyp[1] if flip else hyp[0]
        phi, o, mad, n_rows = h["phi"], h["offset_m"], h["mad_m"], h["rows"]
        if flip:
            msg = (
                f"The {v.name} picture ({v.meta.get('file', v.name)}) seems to show the other side of the "
                f"subject; it was used as the {_phi_name(phi)} view. ('Right' means you walked to "
                f"your right around the subject.)"
            )
            warnings.append(msg)
            log.warning(msg)
            v = replace(v, phi_deg=phi, name=_phi_name(phi), meta={**v.meta, "flipped": True})
        rec.update(phi_used=phi, offset_m=o, mad_m=mad, rows=n_rows, misfit_m=h["misfit_m"])
        if not math.isfinite(o):
            reason = "it could not be lined up with the photo"
        elif mad > mad_max:
            reason = f"it does not line up with the photo (its front edge wanders by {100 * mad:.0f} cm)"
        else:
            reason = ""
        rec["used"] = not reason
        if reason:
            rec["reason"] = reason
            msg = f"The {v.name} picture ({v.meta.get('file', v.name)}) was not used: {reason}."
            warnings.append(msg)
            log.warning(msg)
        out.append(
            replace(
                v,
                meta={
                    **v.meta,
                    "offset_m": o,
                    "mad_m": mad,
                    "misfit_m": h["misfit_m"],
                    "used": not reason,
                    "reason": reason,
                },
            )
        )
        info["views"].append(rec)

    # never two views at one azimuth (a turned-round picture can land on another's side):
    # the one whose front edge fits better is kept
    for a in range(len(out)):
        for b in range(a + 1, len(out)):
            va, vb = out[a], out[b]
            if not (va.meta.get("used") and vb.meta.get("used")) or _az_diff(va.phi_deg, vb.phi_deg) >= 1.0:
                continue
            fa, fb = float(va.meta.get("misfit_m", 0.0)), float(vb.meta.get("misfit_m", 0.0))
            keep, drop = (a, b) if fa <= fb else (b, a)
            vk, vd = out[keep], out[drop]
            reason = f"the {vk.meta.get('file', vk.name)} picture already shows that side"
            msg = f"The {vd.name} picture ({vd.meta.get('file', vd.name)}) was not used: {reason}."
            warnings.append(msg)
            log.warning(msg)
            out[drop] = replace(vd, meta={**vd.meta, "used": False, "reason": reason})
            for rec in info["views"]:
                if rec["name"] == vd.name and rec.get("used") and rec.get("phi_used") == vd.phi_deg:
                    rec.update(used=False, reason=reason)
                    break

    # back edges of the left / right views must agree (design V5, nominal QC)
    exact = [v for v in out if v.meta.get("used") and abs(_sin(v.phi_deg)) > _SIDE_EXACT_SIN]
    pos = [v for v in exact if _sin(v.phi_deg) > 0]
    neg = [v for v in exact if _sin(v.phi_deg) < 0]
    info["agree_m"] = None
    named_sides = {(_sin(v.phi_deg) > 0) != bool(v.meta.get("flipped")) for v in exact}
    if len(exact) >= 2 and not (pos and neg) and len(named_sides) == 2:
        # named for both sides, but turned round onto one: the outlines contradict the names
        # and nothing says which picture is which, so neither is trusted
        msg = "The left and right pictures seem to show the same side of the subject; they were not used."
        warnings.append(msg)
        log.warning(msg)
        info["same_side"] = True
        drop = {id(v) for v in exact}
        out = [
            replace(v, meta={**v.meta, "used": False, "reason": "left and right show the same side"})
            if id(v) in drop
            else v
            for v in out
        ]
    elif pos and neg:
        zb = []
        for v in (pos[0], neg[0]):
            _, u_b = _offset_rows(v.alpha, v.phi_deg, P_S, fr)
            zb.append(math.copysign(1.0, _sin(v.phi_deg)) * (v.meta["offset_m"] + (u_b - fr.S / 2.0) * fr.m))
        both = np.isfinite(zb[0]) & np.isfinite(zb[1])
        diff = float(np.median(zb[0][both] - zb[1][both])) if both.any() else math.inf
        tol = max(_AGREE_M, _AGREE_FRAC * H)
        info.update(agree_m=diff if math.isfinite(diff) else None, agree_tol_m=tol)
        if not abs(diff) <= tol:
            by = f" (their back edges differ by {100 * abs(diff):.0f} cm)" if math.isfinite(diff) else ""
            msg = f"The left and right pictures do not agree on the subject's depth{by}; they were not used."
            warnings.append(msg)
            log.warning(msg)
            drop = {id(v) for v in exact}
            out = [
                replace(v, meta={**v.meta, "used": False, "reason": "left and right disagree"})
                if id(v) in drop
                else v
                for v in out
            ]
    # the orbit axis of nominal views: a photo has no rotation centre of its own (its offset
    # only lines the picture up), so the axis is the middle of the subject's depth - the
    # photo's front edge to the side views' back edge - median over the rows
    exact = [v for v in out if v.meta.get("used") and abs(_sin(v.phi_deg)) > _SIDE_EXACT_SIN]
    if exact:
        rows = np.floor((P_S[:, 1] - fr.cy) / fr.m + fr.S / 2.0).astype(np.int64)
        zf = _row_percentile(rows, P_S[:, 2], fr.S, _EDGE_PCT)
        mid = 0.5 * (zf + _side_back_rows(exact, P_S, fr))
        mid = mid[np.isfinite(mid)]
        if len(mid):
            a_z = float(np.median(mid))
            info.update(z_axis_S=a_z, z_axis=float(canon.z_axis) + a_z, z_axis_prior=float(canon.z_axis))
    return out, info


# =========================================================================== orthographic sets (V2-V5)
@dataclass
class _Reg:
    """What V2-V5 hand to the shared lifting stages (V6-V11)."""

    back: GenView | None = None
    binfo: dict[str, Any] = field(default_factory=dict)
    sides: list[GenView] = field(default_factory=list)  # registered side / quarter views (used or not)
    orbit: dict[str, Any] = field(default_factory=dict)
    views: list[GenView] = field(default_factory=list)  # every view, for BackPart.views
    qc: dict[str, float] = field(default_factory=dict)  # QC terms of the registration
    reason: str = ""  # non-empty: the set cannot be used


def _ortho_affine(v: GenView, fr: _Frame) -> np.ndarray:
    """3 x 3 map (OpenCV pixel coordinates) of an orthographic view's picture into the
    levelled frame: its pixel scale to ``m_in`` about the picture centre (on the orbit axis),
    and its frame centre onto the levelled one - a saved set made with another levelled
    picture moves by ``(x0' - x0) cos(phi)`` horizontally (the axis depth term cancels) and
    ``y0' - y0`` vertically.  The identity for a generator's view at the input's size."""
    h, w = np.shape(v.alpha)
    m_v = _pos_float(v.m) or fr.m * fr.S / max(w, 1)
    centre = _pair(v.meta.get("frame_center_S"))
    cx_v, cy_v = centre if centre is not None else (fr.cx, fr.cy)
    k = m_v / fr.m
    dx = (cx_v - fr.cx) * math.cos(math.radians(float(v.phi_deg))) / fr.m
    dy = (cy_v - fr.cy) / fr.m
    S = fr.S
    return np.array(
        [
            [k, 0.0, (0.5 - w / 2.0) * k + S / 2.0 - 0.5 + dx],
            [0.0, k, (0.5 - h / 2.0) * k + S / 2.0 - 0.5 + dy],
            [0.0, 0.0, 1.0],
        ]
    )


def _warp_alpha(alpha: np.ndarray, M: np.ndarray, S: int) -> np.ndarray:
    a = np.clip(np.nan_to_num(np.asarray(alpha, dtype=np.float32)), 0.0, 1.0)
    return cv2.warpAffine(
        a, M[:2], (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


def _warp_rgb(rgb: np.ndarray, M: np.ndarray, S: int) -> np.ndarray:
    """``rgb`` resampled by the affine ``M`` into ``S x S`` (area-prefiltered when shrunk)."""
    s = math.sqrt(abs(float(np.linalg.det(M[:2, :2]))))
    rgb_q, q = _prefilter(rgb, s)
    Mq = np.asarray(M[:2], dtype=np.float64).copy()
    Mq[:, :2] /= q[None, :]
    Mq[:, 2] += M[:2, :2] @ (0.5 * (1.0 / q - 1.0))
    return cv2.warpAffine(
        rgb_q, Mq, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(_GREY,) * 3
    )


def _corner_move(M: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> float:
    """Largest displacement (px) of the box corners under the affine ``M``."""
    P = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]])
    return float(np.abs(P @ M[:2].T - P[:, :2]).max())


def _signed_dist(mask: np.ndarray) -> np.ndarray:
    """Signed distance (px) of pixel centres to a binary mask's outline (the boundary
    between its inside and outside pixels): negative inside, positive outside."""
    m = np.ascontiguousarray(mask, dtype=np.uint8)
    d_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_in = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(m > 0, 0.5 - d_in, d_out - 0.5).astype(np.float64)


def _bilinear_at(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear lookup at OpenCV coordinates; outside the picture the distance to it is
    added (a distance map never looks better off the frame)."""
    H, W = img.shape
    xc = np.clip(x, 0.0, W - 1.0)
    yc = np.clip(y, 0.0, H - 1.0)
    x0 = np.minimum(np.floor(xc).astype(np.int64), W - 2)
    y0 = np.minimum(np.floor(yc).astype(np.int64), H - 2)
    fx, fy = xc - x0, yc - y0
    top = img[y0, x0] * (1.0 - fx) + img[y0, x0 + 1] * fx
    bot = img[y0 + 1, x0] * (1.0 - fx) + img[y0 + 1, x0 + 1] * fx
    return top * (1.0 - fy) + bot * fy + np.hypot(x - xc, y - yc)


def _outline_pts(mask: np.ndarray) -> np.ndarray:
    """``(n, 2)`` outline pixels (inside pixels with an outside neighbour), at most 1500."""
    m8 = mask.astype(np.uint8)
    er = cv2.erode(m8, np.ones((3, 3), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.nonzero((m8 > 0) & (er == 0))
    k = max(1, int(math.ceil(len(xs) / _OUTLINE_PTS)))
    return np.stack([xs[::k], ys[::k]], axis=1).astype(np.float64)


def _outline_fit(
    src: np.ndarray, dst: np.ndarray, init: dict[str, float], cancel: CancelToken | None
) -> tuple[np.ndarray, float, Callable[[np.ndarray], float]]:
    """Similarity (scale, shift) laying ``src``'s outline onto ``dst``'s up to a uniform
    outline offset: the cost is the mean absolute deviation of the signed distances of
    each outline's points to the other outline from their median (both directions), the
    median bounded to +-1.5 px.  A keyed matte lies ~half a pixel outside the subject's
    outline all round; the plain chamfer takes that for a 1-1.5 % larger subject (limbs
    are a few pixels wide) and shrinks every view.  Starts from the chamfer fit ``init``.
    Returns ``(T 3 x 3, cost, offset(T))``: ``offset(T)`` (px) is how far ``src``'s outline
    lies outside ``dst``'s under ``T``."""
    sd_s, sd_d = _signed_dist(src), _signed_dist(dst)
    P, Q = _outline_pts(src), _outline_pts(dst)
    ys, xs = np.nonzero(src)
    c = np.array([xs.mean(), ys.mean()])
    s0 = float(init.get("s", 1.0))
    c_in = np.array([float(init.get("cx", c[0])), float(init.get("cy", c[1]))])
    t0 = np.array([float(init.get("tx", 0.0)), float(init.get("ty", 0.0))]) + (1.0 - s0) * (c_in - c)

    def T_of(x: np.ndarray) -> np.ndarray:
        s = math.exp(float(x[0]))
        return np.array(
            [[s, 0.0, (1.0 - s) * c[0] + x[1]], [0.0, s, (1.0 - s) * c[1] + x[2]], [0.0, 0.0, 1.0]]
        )

    def dists(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = math.sqrt(abs(float(np.linalg.det(T[:2, :2]))))
        f = P @ T[:2, :2].T + T[:2, 2]
        Ti = np.linalg.inv(T)
        b = Q @ Ti[:2, :2].T + Ti[:2, 2]
        return _bilinear_at(sd_d, f[:, 0], f[:, 1]), s * _bilinear_at(sd_s, b[:, 0], b[:, 1])

    lo, hi = -0.5 - _OUTLINE_MAX_PX, -0.5 + _OUTLINE_MAX_PX  # outline pixels sit 0.5 px inside

    def cost(x: np.ndarray) -> float:
        df, db = dists(T_of(x))
        mf = min(max(float(np.median(df)), lo), hi)
        mb = min(max(float(np.median(db)), lo), hi)
        return float(np.abs(df - mf).mean() + np.abs(db - mb).mean())

    def offset(T: np.ndarray) -> float:
        df, db = dists(T)
        return 0.5 * ((float(np.median(df)) + 0.5) - (float(np.median(db)) + 0.5))

    x0 = np.array([math.log(max(s0, 1e-6)), t0[0], t0[1]])
    x, fx = nelder_mead(cost, x0, np.array([0.01, 1.0, 1.0]), max_iter=300, tol=1e-4, cancel=cancel)
    return T_of(x), float(fx), offset


def _lateral_fit(
    a0w: np.ndarray, fr: _Frame, cancel: CancelToken | None
) -> tuple[np.ndarray, dict[str, Any]]:
    """V2 lateral check: view 0 (in the levelled frame) against the front silhouette.  A
    similarity (scale, shift; no roll) is always fitted (~0.1 s: the symmetric chamfer,
    refined to ignore a uniform outline offset, :func:`_outline_fit`) and kept when it moves
    the outline by more than half a pixel and raises the overlap by at least 2 points - the
    design fits only below IoU 0.85, which leaves a 3 % scale or a 3 px shift of a 512 px
    set (IoU 0.89) uncorrected.  Returns ``(T, info)``: ``T`` the 3 x 3 correction for every
    view (the identity when not kept); ``info["iou"]`` the overlap after it and
    ``info["outline_offset_px"]`` how far view 0's matte lies outside the photo's outline
    (the matte bias every view's edges are corrected by in V5 / V7)."""
    a = a0w >= 0.5
    iou_raw = silhouette_iou(a, fr.alpha0)
    info: dict[str, Any] = {
        "iou_raw": round(iou_raw, 4),
        "applied": False,
        "iou": round(iou_raw, 4),
        "outline_offset_px": 0.0,
    }
    if not a.any():
        return np.eye(3), info
    params, cost = chamfer_similarity_fit(a, fr.alpha0, allow_roll=False, cancel=cancel)
    T, cost_o, offset = _outline_fit(a, fr.alpha0, params, cancel)
    iou_fit = silhouette_iou(_warp_alpha(a0w, T, fr.S) >= 0.5, fr.alpha0)
    rows = np.flatnonzero(a.any(axis=1))
    cols = np.flatnonzero(a.any(axis=0))
    move = _corner_move(T, float(cols[0]), float(rows[0]), float(cols[-1]), float(rows[-1]))
    apply = iou_fit >= iou_raw + _ORTHO_FIT_GAIN and move > _ORTHO_FIT_PX
    T_used = T if apply else np.eye(3)
    bias = min(max(offset(T_used), -_OUTLINE_MAX_PX), _OUTLINE_MAX_PX)
    info.update(
        similarity={
            "s": round(float(T[0, 0]), 5),
            "tx": round(float(T[0, 2]), 4),
            "ty": round(float(T[1, 2]), 4),
        },
        chamfer_px=round(float(cost), 3),
        outline_cost_px=round(cost_o, 4),
        iou_fit=round(iou_fit, 4),
        move_px=round(move, 3),
        applied=bool(apply),
        iou=round(iou_fit if apply else iou_raw, 4),
        outline_offset_px=round(bias, 4),
    )
    return T_used, info


def _ortho_rows(
    a_w: np.ndarray, ref: tuple[float, float], ref_ok: tuple[bool, bool], fr: _Frame
) -> tuple[np.ndarray, dict[str, Any], str]:
    """V2 invariant rows of one orthographic view (in the levelled frame): at elevation 0
    the subject's top and bottom rows are the same in every view, so ``s = (b0 - t0) / (b -
    t)`` against the reference rows (view 0's).  More than 5 % off: dropped (the reason is
    returned); otherwise the 3 x 3 correction scales the view about its centre column (the
    orbit axis) and puts its bottom row on the reference's.  A row cut off by the picture
    border (or the photo's cut-off end, ``ref_ok`` False) is no anchor: one good row only
    aligns (scale 1), none leaves the view as it is - but the measured ratio is still a
    bound (a cut view shows less of the subject, a cut reference less of it), so a view
    that is more than 5 % too large even so (e.g. one filling its picture top to bottom)
    is dropped."""
    S = fr.S
    tb = _top_bottom(a_w)
    if tb is None or tb[1] - tb[0] < 4:
        return np.eye(3), {"scale": 1.0, "anchor": "none"}, "it shows no subject"
    cut = _touches(a_w)
    top_ok = ref_ok[0] and not cut[0]
    bottom_ok = ref_ok[1] and not cut[1]
    (t0, b0), (t, b) = ref, tb
    s_meas = (b0 - t0) / (b - t)
    view_cut, ref_cut = cut[0] or cut[1], not (ref_ok[0] and ref_ok[1])
    s, ya, yb, anchor = 1.0, 0.0, 0.0, "none"
    off = ""
    if top_ok and bottom_ok:
        s, anchor = s_meas, "rows"
        if abs(s - 1.0) > _ORTHO_SCALE_MAX:
            off = f"{100 * abs(s - 1.0):.0f} % off"
        ya, yb = b - 0.5, b0 - 0.5
    else:
        if view_cut and not ref_cut and s_meas < 1.0 - _ORTHO_SCALE_MAX:  # s_true <= s_meas
            off = f"at least {100 * (1.0 - s_meas):.0f} % too large"
        elif ref_cut and not view_cut and s_meas > 1.0 + _ORTHO_SCALE_MAX:  # s_true >= s_meas
            off = f"at least {100 * (s_meas - 1.0):.0f} % too small"
        if bottom_ok:
            ya, yb, anchor = b - 0.5, b0 - 0.5, "bottom"
        elif top_ok:
            ya, yb, anchor = t - 0.5, t0 - 0.5, "top"
    if off:
        info = {"scale": round(float(s_meas), 5), "anchor": anchor}
        return (
            np.eye(3),
            info,
            f"its size does not match the other views ({off}; at most {100 * _ORTHO_SCALE_MAX:.0f} % is accepted)",
        )
    xc = S / 2.0 - 0.5
    C = np.array([[s, 0.0, xc * (1.0 - s)], [0.0, s, yb - s * ya], [0.0, 0.0, 1.0]])
    move = _corner_move(C, 0.0, 0.0, S - 1.0, S - 1.0)
    info = {"scale": round(float(s), 5), "anchor": anchor, "shift_px": round(float(yb - ya), 3)}
    if move <= _ORTHO_ROWS_PX:
        return np.eye(3), {**info, "applied": False}, ""
    return C, {**info, "applied": True}, ""


def _lock_back_ortho(
    v: GenView, M: np.ndarray, fr: _Frame, cancel: CancelToken | None
) -> tuple[GenView, dict[str, Any]]:
    """V3 for an orthographic back view placed by ``M`` (its exact camera plus the set's V2
    corrections): IoU gate 0.75 against the mirrored front silhouette, the outline-gap pose
    gate and the thin-plate spline lock - no similarity / perspective stage, the camera is
    exact.  Returns the view in the levelled back frame (offset ``-x0``) and the
    diagnostics (``ok``, ``reason``)."""
    S = fr.S
    target = fr.alpha0[:, ::-1]
    a_v = np.clip(np.nan_to_num(np.asarray(v.alpha, dtype=np.float32)), 0.0, 1.0)
    a_g = _warp_alpha(a_v, M, S)
    iou = silhouette_iou(a_g >= 0.5, target)
    info: dict[str, Any] = {
        "name": v.name,
        "phi": float(v.phi_deg),
        "file": v.meta.get("file"),
        "camera": "exact",
        "ok": False,
        "iou": round(iou, 4),
    }
    rgb_v = _rgb_u8(v.rgb, f"view '{v.name}'")

    def unused(reason: str) -> tuple[GenView, dict[str, Any]]:
        info["reason"] = reason
        rgb_s = _warp_rgb(rgb_v, M, S)
        return _registered(v, rgb_s, a_g, fr, offset_m=-fr.cx, used=False, reason=reason), info

    if not (a_g >= 0.5).any():
        return unused("the back view shows no subject")
    if iou < _ORTHO_IOU_GATE:
        return unused(
            f"the back view does not match the photo's outline (overlap {100 * iou:.0f} %, at least "
            f"{100 * _ORTHO_IOU_GATE:.0f} % is needed)"
        )
    gap = outline_gap(a_g >= 0.5, target, _POSE_GAP_PCT)
    frac = gap / fr.L_px
    info.update(pose_px=round(gap, 3), pose_frac=round(frac, 4))
    if frac > _POSE_MAX_FRAC:
        return unused(
            f"the back view shows a different pose (parts of its outline are {100 * frac:.0f} % of the "
            f"subject's size away from the photo's; at most {100 * _POSE_MAX_FRAC:.0f} % is accepted)"
        )
    _check(cancel)
    shrink = math.sqrt(abs(float(np.linalg.det(M[:2, :2]))))
    rgb_l, a_l, disp = _tps_lock(rgb_v, a_v, a_g, target, M, shrink, fr)
    iou_l = silhouette_iou(a_l >= 0.5, target)
    info.update(
        ok=True,
        iou_locked=round(iou_l, 4),
        tps_max_px=round(disp, 3),
        tps_max_frac=round(disp / fr.L_px, 4),
    )
    log.info(
        "Back view %s (exact camera): IoU %.3f -> %.3f after the lock (outlines %.1f %% apart)",
        v.name,
        iou,
        iou_l,
        100 * frac,
    )
    out = _registered(v, rgb_l, a_l, fr, offset_m=-fr.cx, used=True, reason="", iou=iou, iou_locked=iou_l)
    return out, info


def _colour_map_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """``(A, b, info)``: the colour map ``lin_dst ~ A lin_src + b`` (linear RGB) of paired
    sRGB colours (uint8 ``N x 3``).  A per-channel gain first (median ratio, clamped to
    [0.6, 1.6]), then :func:`~vrphoto.geometry.register.color_affine_fit` on the
    gain-corrected pairs, as for TRELLIS (docs/CONTRACTS.md section 13): view 0 is mostly a
    few colours (the front of the clothes), with which a full affine cannot tell a gain from
    an offset - an offset that fixes a red front adds red to a blue back.  Each stage is
    kept only when it lowers the median colour error (the affine against the gain alone:
    a few mismatched pairs, e.g. a detail the generator drew differently, tilt it); the
    identity when neither does."""
    Xs = _srgb_to_linear(np.asarray(src, dtype=np.float64) / 255.0)
    Yd = _srgb_to_linear(np.asarray(dst, dtype=np.float64) / 255.0)
    g = np.ones(3)
    for c in range(3):
        ok = Xs[:, c] > 0.01
        if int(ok.sum()) >= 8:
            g[c] = float(np.clip(np.median(Yd[ok, c] / Xs[ok, c]), 0.6, 1.6))
    A_r, b_r, ok_r = color_affine_fit(_linear_to_srgb(Xs * g), np.asarray(dst, dtype=np.float64) / 255.0)

    def err(A: np.ndarray, b: np.ndarray) -> float:
        return float(np.median(np.linalg.norm(Xs @ A.T + b - Yd, axis=1)))

    eye, zero = np.eye(3), np.zeros(3)
    err0, err_g = err(eye, zero), err(np.diag(g), zero)
    err_a = err(A_r @ np.diag(g), b_r) if ok_r else math.inf
    info: dict[str, Any] = {
        "gain": [round(float(x), 4) for x in g],
        "error_before": round(err0, 5),
        "error_gain": round(err_g, 5),
        "error_affine": round(err_a, 5) if math.isfinite(err_a) else None,
    }
    if err_a < min(err_g, err0) - 1e-4:
        return A_r @ np.diag(g), b_r, {**info, "kept": "affine", "error_after": round(err_a, 5)}
    if err_g < err0 - 1e-4:
        return np.diag(g), zero, {**info, "kept": "gain", "error_after": round(err_g, 5)}
    return eye, zero, {**info, "kept": "none", "error_after": round(err0, 5)}


def _ortho_colours(
    views: list[GenView], v0: GenView | None, ortho: OrthoInput, fr: _Frame
) -> tuple[list[GenView], dict[str, Any]]:
    """V4 for an orthographic set: pixel pairs between view 0 (registered) and the levelled
    input where both mattes exceed 0.9 (eroded 4 px) and the input's gradient is at most its
    30th percentile; one colour map in linear RGB (:func:`_colour_map_fit`: a per-channel
    gain, then the Huber affine) applied to every view (one exposure; ``meta["color_match"]``
    ``"gain"`` / ``"affine"``, ``"identity"`` when nothing was changed).  When it does not
    lower the colour error the views already match and are left as they are (the design's
    Reinhard fallback would pull a back of another colour toward the front's lightness);
    without view 0 or with fewer than 64 pairs, the bounded rim-band Reinhard of nominal
    views (:func:`_match_colours`).  Views matched before (a saved set) are skipped."""
    todo = [i for i, v in enumerate(views) if "color_match" not in v.meta]
    info: dict[str, Any] = {"method": "none", "pairs": 0}
    if not todo:
        info["method"] = "matched before"
        return views, info
    out = list(views)
    if v0 is not None:
        ref = np.asarray(ortho.rgba)[..., :3]
        a_in = np.asarray(ortho.rgba)[..., 3].astype(np.float32) / 255.0
        both = (np.asarray(v0.alpha) >= _COLOR_FIT_ALPHA) & (a_in >= _COLOR_FIT_ALPHA)
        both = cv2.erode(both.astype(np.uint8), _disc(_COLOR_FIT_ERODE)) > 0
        sel = both
        if both.any():
            g = cv2.cvtColor(np.ascontiguousarray(ref), cv2.COLOR_RGB2GRAY).astype(np.float32)
            grad = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
            sel = both & (grad <= np.percentile(grad[both], _COLOR_FIT_GRAD_PCT))
        info["pairs"] = int(sel.sum())
        if info["pairs"] >= _COLOR_FIT_MIN:
            A, b, finfo = _colour_map_fit(v0.rgb[sel], ref[sel])
            kept = finfo["kept"] != "none"
            info.update(
                finfo, A=[[round(float(x), 5) for x in r] for r in A], b=[round(float(x), 5) for x in b]
            )
            method = str(finfo["kept"]) if kept else "identity"  # "gain" / "affine"
            for i in todo:
                v = out[i]
                rgb = apply_color_affine(v.rgb, A, b) if kept else v.rgb
                out[i] = replace(v, rgb=rgb, meta={**v.meta, "color_match": method})
            info["method"] = method
            return out, info
    done = _match_colours([out[i] for i in todo], ortho, fr)
    for i, v in zip(todo, done, strict=True):
        out[i] = v
    info["method"] = "reinhard"
    return out, info


def _masked_ssim(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean SSIM (Gaussian window, sigma 1.5 px) of two grey pictures in [0, 1] over ``mask``
    (0 when the mask is empty)."""
    if not mask.any():
        return 0.0
    c1, c2 = 0.01**2, 0.03**2

    def blur(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (0, 0), 1.5)

    mu_a, mu_b = blur(a), blur(b)
    s_aa = blur(a * a) - mu_a * mu_a
    s_bb = blur(b * b) - mu_b * mu_b
    s_ab = blur(a * b) - mu_a * mu_b
    ssim = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return float(np.mean(ssim[mask]))


def _sign_render_check(
    sides: list[GenView],
    z_axes: Mapping[bool, float],
    front_sc: SplatCloud | None,
    fr: _Frame,
    cancel: CancelToken | None,
) -> dict[str, Any]:
    """The second azimuth-sign test of design V5: SHARP's photographed front rendered
    (:func:`~vrphoto.geometry.splat_render.render_splats`, orthographic, in the views'
    levelled framing about each hypothesis' own orbit axis ``z_axes[flip]``) at each quarter
    view's azimuth (the +-90 views when there is no quarter view) - as named and mirrored -
    and compared with the (colour-matched) view at 192 px: ``IoU + 0.5 SSIM + 0.5 (1 - MAE /
    0.25)`` with the grey SSIM and the mean absolute RGB difference over both mattes (the
    render un-premultiplied).  The colour term is an addition to the design's ``IoU + 0.5
    SSIM``: on the synthetic person that alone separated the hypotheses by only 0.03 for side
    views drawn 18 deg off their azimuth (the mean colour difference: 0.10 vs 0.22).  The render
    shows only the photographed half, so neither hypothesis scores 2; the right one scores
    clearly more.  ``contradicts`` is True when the mirrored hypothesis scores at least 0.05
    more per view (the cameras look reversed); ``ran`` is False when nothing could be
    compared."""
    out: dict[str, Any] = {"ran": False, "contradicts": False}
    quarter = [v for v in sides if _SIDE_MIN_SIN < abs(_sin(v.phi_deg)) <= _SIDE_EXACT_SIN]
    check = quarter or [v for v in sides if abs(_sin(v.phi_deg)) > _SIDE_EXACT_SIN]
    if (
        front_sc is None
        or not len(front_sc)
        or not check
        or not all(math.isfinite(z) for z in z_axes.values())
    ):
        out["reason"] = "nothing to compare"
        return out
    from .splat_render import SplatCamera, prepare_splats, render_splats

    S = fr.S
    N = int(min(S, _SIGN_RENDER_PX))
    m = fr.m * S / N
    prep = prepare_splats(front_sc, cancel=cancel)
    R = fr.R
    down = R[:, 1]
    scores: dict[bool, list[float]] = {False: [], True: []}
    per_view = []
    for v in check:
        a_v = cv2.resize(np.asarray(v.alpha, dtype=np.float32), (N, N), interpolation=cv2.INTER_AREA)
        c_v = (
            cv2.resize(_rgb_u8(v.rgb, v.name), (N, N), interpolation=cv2.INTER_AREA).astype(np.float32)
            / 255.0
        )
        g_v = cv2.cvtColor(c_v, cv2.COLOR_RGB2GRAY)
        rec: dict[str, Any] = {"name": v.name, "phi": float(v.phi_deg)}
        for flip in (False, True):
            _check(cancel)
            phi = (360.0 - float(v.phi_deg)) % 360.0 if flip else float(v.phi_deg)
            p = math.radians(phi)
            right = R @ np.array([math.cos(p), 0.0, math.sin(p)])
            fwd = R @ np.array([-math.sin(p), 0.0, math.cos(p)])
            o = fr.cx * math.cos(p) + z_axes[flip] * math.sin(p)
            centre = fr.c + o * right + fr.cy * down - _SIGN_RENDER_DIST_M * fwd
            cam = SplatCamera(
                R=np.stack([right, down, fwd]), center=centre, width=N, height=N, kind="ortho", m=m
            )
            r = render_splats(prep, cam, cancel=cancel)
            al = np.asarray(r.alpha, dtype=np.float32)
            ra, va = al >= 0.5, a_v >= 0.5
            both = ra & va
            c_r = np.clip(
                np.asarray(r.rgb, dtype=np.float32) / 255.0 / np.maximum(al, 1e-3)[..., None], 0.0, 1.0
            )
            g_r = cv2.cvtColor(c_r, cv2.COLOR_RGB2GRAY)
            mae = float(np.abs(c_r - c_v)[both].mean()) if both.any() else 1.0
            score = (
                silhouette_iou(ra, va)
                + _SIGN_SSIM_WEIGHT * _masked_ssim(g_r, g_v, both)
                + _SIGN_COLOUR_WEIGHT * max(0.0, 1.0 - mae / _SIGN_COLOUR_SCALE)
            )
            scores[flip].append(score)
            rec["reversed" if flip else "named"] = round(score, 4)
        per_view.append(rec)
    named, mirrored = float(np.mean(scores[False])), float(np.mean(scores[True]))
    out.update(
        ran=True,
        views=per_view,
        named=round(named, 4),
        reversed=round(mirrored, 4),
        contradicts=bool(mirrored > named + _SIGN_RENDER_MARGIN),
    )
    return out


def _turned_round(v: GenView) -> GenView:
    """``v`` at the mirrored azimuth ``360 - phi`` (the azimuth sign of its camera reversed),
    renamed, with ``meta["flipped"]`` and the azimuth it was named for (``phi_named``)."""
    phi = (360.0 - float(v.phi_deg)) % 360.0
    return replace(
        v, phi_deg=phi, name=_phi_name(phi), meta={**v.meta, "flipped": True, "phi_named": float(v.phi_deg)}
    )


def _ortho_axis(
    sides: list[GenView],
    Pf_S: np.ndarray,
    fr: _Frame,
    canon: Canon,
    warnings: list[str],
    front_sc: SplatCloud | None = None,
    cancel: CancelToken | None = None,
) -> tuple[list[GenView], dict[str, Any]]:
    """V5 for an orthographic set: the shared orbit axis depth ``z_a`` (frame S) and the
    azimuth sign.

    In a view at ``phi`` the silhouette edge facing the photo camera is the photographed
    front, so per row ``z_a(j) = (o(j) - x0 cos(phi)) / sin(phi)`` with ``o(j)`` the
    front-edge offset of :func:`_offset_rows`.  The +-90 views (else the quarter views) give
    ``z_a`` = the median over the rows of both; their medians must agree within ``max(4 cm,
    2 % H)`` and each view's row MAD stay within ``max(3 cm, 2 % H)``, otherwise every side
    and quarter view is dropped (their offsets depend on ``z_a``) with a warning.  Sign
    safety net: with the sign reversed (``360 - phi``) each view picks its back edge, so the
    two still agree; the rows then measure the subject's mid-depth, which changes only where
    something sticks out at the front or the back (an arm in front, a bag, the feet), so the
    test is the 90th percentile of the row deviations (the MAD of the design barely changes:
    measured 2.5 vs 2.9 mm on the synthetic person, against 6 vs 40 mm at the 90th
    percentile).  When the reversed hypothesis costs less than half (summed over the views),
    the second test of the design decides (:func:`_sign_render_check`: SHARP's front rendered
    at the quarter views' azimuths, as named and mirrored, compared with the views): every view
    is turned round with a warning only when it agrees.  Otherwise the named sides are kept
    with a warning (side views rendered ~18 deg off their azimuth also fit better mirrored by
    rows; turning them round mirrored the subject).  Then every view's offset is ``x0 cos(phi) +
    z_a sin(phi)``; a quarter view whose own front edge disagrees with that (beyond the same
    tolerances) is dropped.  Row edges are corrected by each view's
    ``meta["edge_bias_px"]`` (the matte's outline offset measured on view 0)."""
    H = float(canon.height_s)
    mad_max = max(_MAD_MAX_M, _MAD_MAX_FRAC * H)
    tol = max(_AGREE_M, _AXIS_AGREE_FRAC * H)
    info: dict[str, Any] = {"views": [], "agree_m": None, "agree_tol_m": tol, "mad_max_m": mad_max}
    if not sides:
        return sides, info
    exact = [i for i, v in enumerate(sides) if abs(_sin(v.phi_deg)) > _SIDE_EXACT_SIN]
    axis_idx = exact or list(range(len(sides)))

    def bias(v: GenView) -> float:
        return float(v.meta.get("edge_bias_px", 0.0))

    def rows_z(v: GenView, phi: float) -> dict[str, Any] | None:
        o_rows, _ = _offset_rows(v.alpha, phi, Pf_S, fr, bias(v))
        z = (o_rows - fr.cx * math.cos(math.radians(phi))) / _sin(phi)
        ok = np.isfinite(z)
        sil_rows = int((np.asarray(v.alpha) >= 0.5).any(axis=1).sum())
        if int(ok.sum()) < max(5, int(math.ceil(_MIN_ROWS_FRAC * sil_rows))):
            return None
        med = float(np.median(z[ok]))
        dev = np.abs(z[ok] - med)
        return {
            "z": z[ok],
            "med": med,
            "mad": float(np.median(dev)),
            "misfit": float(np.percentile(dev, _SIGN_PCT)),
            "rows": int(ok.sum()),
        }

    hyp: dict[bool, tuple[list[dict[str, Any] | None], float, float]] = {}
    for flip in (False, True):
        res = [
            rows_z(sides[i], (360.0 - sides[i].phi_deg) % 360.0 if flip else sides[i].phi_deg)
            for i in axis_idx
        ]
        full = all(r is not None for r in res)
        misfit = sum(r["misfit"] for r in res if r is not None) if full else math.inf
        mad = sum(r["mad"] for r in res if r is not None) if full else math.inf
        hyp[flip] = (res, misfit, mad)
    named, reversed_ = hyp[False][1], hyp[True][1]
    proposed = (
        math.isfinite(reversed_)
        and named > _SIGN_MIN_MISFIT_M * len(axis_idx)
        and reversed_ < _FLIP_RATIO * named
    )
    info["sign"] = {
        "named_misfit_m": named,
        "reversed_misfit_m": reversed_,
        "named_mad_m": hyp[False][2],
        "reversed_mad_m": hyp[True][2],
        "rows_say_reversed": bool(proposed),
        "flipped": False,
        "unconfirmed": False,
    }
    flip = False
    if proposed:  # the second test (design V5): flip only when both contradict the cameras
        z_axes = {
            h: float(np.median(np.concatenate([r["z"] for r in hyp[h][0] if r is not None])))
            for h in (False, True)
        }
        check = _sign_render_check(list(sides), z_axes, front_sc, fr, cancel)
        info["sign"]["render"] = check
        flip = bool(check.get("contradicts"))
        info["sign"]["flipped"] = flip
        if not flip:
            info["sign"]["unconfirmed"] = True
            msg = (
                "The generated side views fit the photo better as mirror images, but SHARP's front "
                "does not confirm that their left and right were swapped; they were used as made (the "
                "AI may have drawn the subject turned)."
            )
            warnings.append(msg)
            log.warning(msg)
    if flip:
        sides = [_turned_round(v) for v in sides]
        msg = (
            "The generated side views were turned round: the azimuth sign of their cameras looks "
            "reversed (the left and right views were swapped)."
        )
        warnings.append(msg)
        log.warning(msg)
    res = hyp[flip][0]
    info["axis_views"] = [sides[i].name for i in axis_idx]

    def drop_all(reason: str, msg: str) -> tuple[list[GenView], dict[str, Any]]:
        warnings.append(msg)
        log.warning(msg)
        info["dropped"] = reason
        return [replace(v, meta={**v.meta, "used": False, "reason": reason}) for v in sides], info

    if any(r is None for r in res):
        return drop_all(
            "not lined up",
            "The generated side views could not be lined up with the photo; they were not used.",
        )
    good: list[dict[str, Any]] = [r for r in res if r is not None]
    z_a = float(np.median(np.concatenate([r["z"] for r in good])))
    for i, r in zip(axis_idx, good, strict=True):
        info["views"].append(
            {
                "name": sides[i].name,
                "phi": float(sides[i].phi_deg),
                "z_m": r["med"],
                "mad_m": r["mad"],
                "rows": r["rows"],
            }
        )
    pos = [r["med"] for i, r in zip(axis_idx, good, strict=True) if _sin(sides[i].phi_deg) > 0]
    neg = [r["med"] for i, r in zip(axis_idx, good, strict=True) if _sin(sides[i].phi_deg) < 0]
    if pos and neg:
        agree = float(np.median(pos) - np.median(neg))
        info["agree_m"] = agree
        if abs(agree) > tol:
            return drop_all(
                "left and right disagree",
                f"The generated left and right views do not agree on the subject's depth (by "
                f"{100 * abs(agree):.0f} cm); the side views were not used.",
            )
    worst = max(r["mad"] for r in good)
    if worst > mad_max:
        return drop_all(
            "does not line up",
            f"The generated side views do not line up with the photo (their front edge wanders by "
            f"{100 * worst:.0f} cm); they were not used.",
        )
    info.update(z_axis_S=z_a, z_axis=float(canon.z_axis) + z_a, z_axis_prior=float(canon.z_axis))
    mads = {i: r["mad"] for i, r in zip(axis_idx, good, strict=True)}
    out: list[GenView] = []
    for i, v in enumerate(sides):
        phi = float(v.phi_deg)
        o = fr.cx * math.cos(math.radians(phi)) + z_a * _sin(phi)
        mad, reason, dev = mads.get(i, 0.0), "", 0.0
        if i not in mads:  # a quarter view: its own front edge must follow the solved axis
            o_rows, _ = _offset_rows(v.alpha, phi, Pf_S, fr, bias(v))
            ok = np.isfinite(o_rows)
            if int(ok.sum()) >= 5:
                dev = float(np.median(o_rows[ok]) - o)
                mad = float(np.median(np.abs(o_rows[ok] - o - dev)))
                if abs(dev) > tol or mad > mad_max:
                    reason = f"it does not line up with the side views (its front edge is {100 * abs(dev):.0f} cm off)"
            else:
                reason = "it could not be lined up with the photo"
            info["views"].append(
                {"name": v.name, "phi": phi, "dev_m": dev, "mad_m": mad, "used": not reason, "reason": reason}
            )
            if reason:
                msg = f"The {v.name} view ({v.meta.get('file', v.name)}) was not used: {reason}."
                warnings.append(msg)
                log.warning(msg)
        out.append(
            replace(v, meta={**v.meta, "offset_m": o, "mad_m": mad, "used": not reason, "reason": reason})
        )
    log.info(
        "Orthographic views: orbit axis %.3f m (prior %.3f m), +-90 agree %s, row MAD %.1f mm%s",
        float(canon.z_axis) + z_a,
        float(canon.z_axis),
        "n/a" if info["agree_m"] is None else f"{1000 * info['agree_m']:.1f} mm",
        1000 * worst,
        ", azimuth sign reversed" if flip else "",
    )
    return out, info


def _register_ortho_set(
    vlist: list[GenView],
    canon: Canon,
    ortho: OrthoInput,
    fr: _Frame,
    Pf_S: np.ndarray,
    color_match: bool,
    warnings: list[str],
    diag: dict[str, Any],
    rep: Any,
    cancel: CancelToken | None,
    front_sc: SplatCloud | None = None,
) -> _Reg:
    """V2-V5 for a set of orthographic views (see the module docstring); ``front_sc`` (SHARP's
    cleaned front) is rendered by the second azimuth-sign test."""
    reg = _Reg()
    S = fr.S
    rep(0.0, "Lining up the views")
    view0: GenView | None = None
    back: GenView | None = None
    sides_in: list[GenView] = []
    unused: list[GenView] = []
    for v in vlist:
        why = ""
        if v.kind != "ortho":
            why = "it was not made with the cameras of the other views"
        elif _is_back(v.phi_deg):
            if back is None:
                back = v
                continue
            why = "second back"
        elif _az_diff(v.phi_deg, 0.0) < 1.0:
            if view0 is None:
                view0 = v
                continue
            why = "second front view"
        elif abs(_sin(v.phi_deg)) > _SIDE_MIN_SIN:
            sides_in.append(v)
            continue
        else:
            why = "a front view"
        diag["views"].append({"name": v.name, "phi": v.phi_deg, "used": False, "reason": why})
        if v.kind != "ortho":
            msg = f"The {v.name} picture ({v.meta.get('file', v.name)}) was not used: {why}."
            warnings.append(msg)
            log.warning(msg)
        unused.append(replace(v, meta={**v.meta, "used": False, "reason": why}))
    assert back is not None  # the caller routes sets with an orthographic back here
    items: list[tuple[GenView, np.ndarray]] = [(v, _ortho_affine(v, fr)) for v in (back, *sides_in)]
    M0 = _ortho_affine(view0, fr) if view0 is not None else None

    def shown(reason: str) -> list[GenView]:
        """Every view in the levelled frame as far as it got, for the contact sheet."""
        out = []
        for v, M in ([(view0, M0)] if view0 is not None else []) + items:
            a = _warp_alpha(v.alpha, M, S)
            out.append(
                _registered(v, _warp_rgb(_rgb_u8(v.rgb, v.name), M, S), a, fr, used=False, reason=reason)
            )
        return out + unused

    # ---- V2: lateral check against the photo's outline (view 0)
    lat: dict[str, Any] = {"skipped": "no front view"}
    if view0 is not None and M0 is not None:
        T, lat = _lateral_fit(_warp_alpha(view0.alpha, M0, S), fr, cancel)
        if lat["iou"] < _ORTHO_IOU_GATE:
            diag["lateral"] = lat
            reg.reason = (
                f"the generated views do not match the photo's outline (the front view overlaps it by "
                f"{100 * lat['iou']:.0f} %, at least {100 * _ORTHO_IOU_GATE:.0f} % is needed)"
            )
            reg.views = shown(reg.reason)
            return reg
        if lat["applied"]:
            M0 = T @ M0
            items = [(v, T @ M) for v, M in items]
    diag["lateral"] = lat
    if view0 is not None:
        diag["views"].append(
            {
                "name": view0.name,
                "phi": view0.phi_deg,
                "file": view0.meta.get("file"),
                "used": True,
                "role": "outline check and colour reference",
            }
        )
    _check(cancel)

    # ---- V2: invariant rows (against view 0's rows, else the photo's)
    cut0 = (bool(getattr(canon, "cut_top", False)), bool(getattr(canon, "cut_bottom", False)))
    a_ref = (
        _warp_alpha(view0.alpha, M0, S)
        if view0 is not None and M0 is not None
        else fr.alpha0.astype(np.float32)
    )
    ref = _top_bottom(a_ref)
    cut_ref = _touches(a_ref)
    ref_ok = (not (cut0[0] or cut_ref[0]), not (cut0[1] or cut_ref[1]))
    kept: list[tuple[GenView, np.ndarray, dict[str, Any]]] = []
    recs: list[dict[str, Any]] = []  # diag records of the kept views (the back first)
    rejected: list[GenView] = []
    rejected_recs: list[dict[str, Any]] = []

    def warn_rejected() -> None:
        """The dropped views' warnings, named as finally placed (V5 may turn the set round)."""
        for r in rejected:
            msg = (
                f"The {r.name} view ({r.meta.get('file', r.name)}) was not used: {r.meta.get('reason', '')}."
            )
            warnings.append(msg)
            log.warning(msg)

    for v, M in items:
        a_w = _warp_alpha(v.alpha, M, S)
        if ref is None:
            C, rinfo, why = np.eye(3), {"scale": 1.0, "anchor": "none"}, ""
        else:
            C, rinfo, why = _ortho_rows(a_w, ref, ref_ok, fr)
        rec = {"name": v.name, "phi": v.phi_deg, "file": v.meta.get("file"), "used": not why, **rinfo}
        diag["views"].append(rec)
        if why:
            rec["reason"] = why
        if why:
            if v is back:
                reg.reason = f"the back view could not be used ({why})"
                warn_rejected()
                reg.views = shown(reg.reason)
                return reg
            rgb = _warp_rgb(_rgb_u8(v.rgb, v.name), M, S)
            rejected.append(_registered(v, rgb, a_w, fr, used=False, reason=why, scale=rinfo.get("scale")))
            rejected_recs.append(rec)
            continue
        kept.append((v, C @ M, rinfo))
        recs.append(rec)
    _check(cancel)

    # ---- V3: the back lock
    rep(0.06, "Fitting the back view to the outline")
    _, Mb, binfo_rows = kept[0]
    back_r, binfo = _lock_back_ortho(back, Mb, fr, cancel)
    diag["back"] = binfo
    if not binfo["ok"]:
        warn_rejected()
        reg.reason = binfo["reason"]
        reg.views = shown(reg.reason)  # [view 0, back, sides..., unused...]
        reg.views[1 if view0 is not None else 0] = back_r
        return reg
    centre = [float(fr.cx), float(fr.cy)]
    back_r = replace(
        back_r, meta={**back_r.meta, "frame_center_S": centre, "scale": binfo_rows.get("scale", 1.0)}
    )
    sides: list[GenView] = []
    edge_bias = float(lat.get("outline_offset_px", 0.0))  # the mattes' outline offset (view 0's)
    for v, M, rinfo in kept[1:]:
        _check(cancel)
        rgb = _warp_rgb(_rgb_u8(v.rgb, v.name), M, S)
        sides.append(
            _registered(
                v,
                rgb,
                _warp_alpha(v.alpha, M, S),
                fr,
                used=True,
                reason="",
                scale=rinfo.get("scale", 1.0),
                anchor=rinfo.get("anchor"),
                frame_center_S=centre,
                edge_bias_px=edge_bias,
            )
        )
    v0r: GenView | None = None
    if view0 is not None and M0 is not None:
        rgb0 = _warp_rgb(_rgb_u8(view0.rgb, view0.name), M0, S)
        v0r = _registered(
            view0,
            rgb0,
            _warp_alpha(view0.alpha, M0, S),
            fr,
            offset_m=fr.cx,
            used=True,
            reason="",
            frame_center_S=centre,
        )
    _check(cancel)

    # ---- V4: one colour map for the whole set
    if color_match:
        group = [*([v0r] if v0r is not None else []), back_r, *sides]
        group, cinfo = _ortho_colours(group, v0r, ortho, fr)
        if v0r is not None:
            v0r, back_r, *sides = group
        else:
            back_r, *sides = group
        diag["colour_fit"] = cinfo
    diag["color_match"] = bool(color_match)
    _check(cancel)

    # ---- V5: the shared orbit axis and the azimuth sign
    rep(0.12, "Finding the orbit axis")
    sides, orbit = _ortho_axis(sides, Pf_S, fr, canon, warnings, front_sc=front_sc, cancel=cancel)
    diag["orbit"] = orbit
    for rec, v in zip(recs[1:], sides, strict=True):  # V5 may turn views round or drop them
        rec.update(phi_used=float(v.phi_deg), used=bool(v.meta.get("used")))
        if not v.meta.get("used"):
            rec["reason"] = v.meta.get("reason", "")
    if (orbit.get("sign") or {}).get("flipped"):  # the dropped side views are turned round too
        rejected = [_turned_round(v) for v in rejected]
        for rec, v in zip(rejected_recs, rejected, strict=True):
            rec.update(phi_used=float(v.phi_deg))
    warn_rejected()
    n_dropped = len(rejected) + sum(1 for v in sides if not v.meta.get("used"))
    spread = [abs(float(r.get("scale", 1.0)) - 1.0) for _, _, r in kept]
    sign = orbit.get("sign") or {}
    reg.qc = {
        "height_spread": (max(spread) / _ORTHO_SCALE_MAX) if spread else 0.0,
        "lateral": (1.0 - float(lat.get("iou", 0.0))) if v0r is not None else 0.0,
        "dropped": _QC_DROPPED * n_dropped,
        # an unflipped set wins between tries: a turned-round or doubtful one had wrong cameras
        "sign": _QC_FLIPPED if sign.get("flipped") else (_QC_SIGN_UNSURE if sign.get("unconfirmed") else 0.0),
    }
    reg.back, reg.binfo, reg.sides, reg.orbit = back_r, binfo, sides, orbit
    reg.views = [*([v0r] if v0r is not None else []), back_r, *sides, *rejected, *unused]
    rep(0.2, "Lining up the views")
    return reg


# =========================================================================== V6 hull
def _constraining(views: Sequence[GenView], fr: _Frame) -> list[tuple[float, float, np.ndarray, int]]:
    """``(phi, offset, dilated matte (flat bool), votes)`` of the registered side / quarter
    views.  A left / right view (``|sin phi| > 0.95``) whose mirror partner (``phi + 180``)
    is not among them has 2 votes: in orthographic projection the two carry the same
    silhouette, so in a whole set that silhouette carves with two votes.  Alone with one
    vote it no longer carved behind the body, where the quarter views' silhouettes are
    wider in depth (measured: the hull 8.7 instead of 3.9 cm beyond the true back at the
    median with the 270 view dropped)."""
    out: list[tuple[float, float, np.ndarray, int]] = []
    for v in views:
        if not (v.meta.get("registered") and v.meta.get("used", True)):
            continue
        if abs(_sin(v.phi_deg)) <= _SIDE_MIN_SIN or not math.isfinite(
            float(v.meta.get("offset_m", math.nan))
        ):
            continue
        a = np.asarray(v.alpha)
        if a.shape != (fr.S, fr.S):
            continue
        dil = cv2.dilate((a >= 0.5).astype(np.uint8), _disc(_HULL_DILATE_PX)) > 0
        out.append((float(v.phi_deg), float(v.meta["offset_m"]), dil.ravel(), 1))
    exact = [k for k, c in enumerate(out) if abs(_sin(c[0])) > _SIDE_EXACT_SIN]
    for k in exact:
        phi = out[k][0]
        if not any(j != k and _az_diff(out[j][0], phi + 180.0) < 22.5 for j in exact):
            out[k] = (*out[k][:3], 2)
    return out


def _vote(
    P_S: np.ndarray, cviews: list[tuple[float, float, np.ndarray, int]], fr: _Frame, rule: str
) -> np.ndarray:
    """Inside the visual hull: inside at least K-1 of the K votes of the views that see the
    point (all of them when K <= 2, or with ``rule="all"``; a lone left / right view has 2
    votes, :func:`_constraining`); a point no view sees is not constrained."""
    shape = P_S.shape[:-1]
    n_in = np.zeros(shape, np.int16)
    n_known = np.zeros(shape, np.int16)
    S = fr.S
    for phi, o, mask, votes in cviews:
        u, v, _ = _project(P_S, phi, o, fr)
        with np.errstate(invalid="ignore"):
            ok = (u >= 0) & (u < S) & (v >= 0) & (v < S)
        iu = np.floor(np.where(ok, u, 0.0)).astype(np.int64)
        iv = np.floor(np.where(ok, v, 0.0)).astype(np.int64)
        flat = np.clip(iv * S + iu, 0, S * S - 1)
        n_in += (ok & mask[flat]) * np.int16(votes)
        n_known += ok * np.int16(votes)
    if rule == "all":
        return n_in == n_known
    return np.where(n_known <= 2, n_in == n_known, n_in >= n_known - 1)


def _rays_S(front: FrontModel, fr: _Frame, cells: np.ndarray) -> np.ndarray:
    """``R^T ray`` (S axes) of the z = 1 rays through mould cells (flat indices)."""
    u, v = front.grid.centers()
    rays = front.cam.rays(u.ravel()[cells], v.ravel()[cells])
    return rays @ fr.R


def hull_intervals(
    front: FrontModel,
    canon: Canon,
    ortho: OrthoInput,
    views: Sequence[GenView],
    *,
    rule: str = "vote",
    samples: int = _HULL_SAMPLES,
    cancel: CancelToken | None = None,
) -> HullIntervals:
    """V6: per mould-grid ray of the matte, the first contiguous ``[z_in, z_out]`` (depths in
    C) inside the visual hull of the registered side / quarter ``views`` (from
    :func:`view_based_back`; the back view adds nothing - its silhouette is the photo's).

    ``samples`` points from ``D_f`` to ``D_f + max(25 cm, 0.5 height)`` along each ray; a
    point is inside view ``k`` when it projects into the view's matte dilated by 3 px
    (outside the picture = unknown); ``rule="vote"``: inside at least K-1 of the K views
    that see it (all when K <= 2), ``"all"``: inside every one (the strict carve, for
    comparison).  ``z_out`` is refined by 3 bisection steps.  ``valid`` marks rays whose
    run ends inside the sampled range.  Rays are processed in chunks with a cancel check."""
    if rule not in ("vote", "all"):
        raise InputError(f"Unknown hull rule {rule!r}; use 'vote' or 'all'.")
    fr = _frame(ortho, canon)
    matte, D_f = _front_maps(front)
    shape = matte.shape
    z_in = np.full(shape, np.nan, np.float32)
    z_out = np.full(shape, np.nan, np.float32)
    valid = np.zeros(shape, bool)
    cviews = _constraining(views, fr)
    if not cviews:
        return HullIntervals(z_in=z_in, z_out=z_out, valid=valid)
    n_s = max(4, int(samples))
    span = max(_HULL_RANGE_M, _HULL_RANGE_FRAC * float(canon.height_s))
    t = np.linspace(0.0, 1.0, n_s)
    cells = np.flatnonzero(matte.ravel())
    c_S = fr.c @ fr.R
    zi_f, zo_f, va_f = z_in.ravel(), z_out.ravel(), valid.ravel()
    for a in range(0, len(cells), _HULL_CHUNK):
        _check(cancel)
        cl = cells[a : a + _HULL_CHUNK]
        rS = _rays_S(front, fr, cl)
        d0 = D_f.ravel()[cl]
        z = d0[:, None] + span * t[None, :]  # (n, s)
        P = z[..., None] * rS[:, None, :] - c_S
        inside = _vote(P, cviews, fr, rule)
        has = inside.any(axis=1)
        start = np.argmax(inside, axis=1)
        after = np.arange(n_s)[None, :] >= start[:, None]
        gap = after & ~inside
        closed = has & gap.any(axis=1)
        end = np.argmax(gap, axis=1)  # first outside sample after the run
        rows = np.flatnonzero(closed)
        if not len(rows):
            continue
        lo = z[rows, end[rows] - 1]
        hi = z[rows, end[rows]]
        for _ in range(_HULL_BISECT):
            mid = 0.5 * (lo + hi)
            ins = _vote(mid[:, None] * rS[rows] - c_S, cviews, fr, rule)
            lo = np.where(ins, mid, lo)
            hi = np.where(ins, hi, mid)
        idx = cl[rows]
        zi_f[idx] = z[rows, start[rows]]
        zo_f[idx] = 0.5 * (lo + hi)
        va_f[idx] = True
    return HullIntervals(z_in=z_in, z_out=z_out, valid=valid)


def _hull_volume(front: FrontModel, hull: HullIntervals) -> float:
    """Volume (m^3) of the hull intervals: each ray's cell is a pyramid frustum."""
    k = (float(front.grid.cell_px) / float(front.cam.f)) ** 2
    zi = np.asarray(hull.z_in, dtype=np.float64)[hull.valid]
    zo = np.asarray(hull.z_out, dtype=np.float64)[hull.valid]
    return float(k * ((zo**3 - zi**3) / 3.0).sum())


# =========================================================================== V7 back depth
def _back_q(P_S: np.ndarray, fr: _Frame) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV coordinates of S points in the locked back view (offset ``-x0``)."""
    u, v, _ = _project(P_S, 180.0, -fr.cx, fr)
    return u - 0.5, v - 0.5


def _relief_map(
    back: GenView, fr: _Frame, depth_fn: Callable[[np.ndarray], Any] | None, mode: str, warnings: list[str]
) -> tuple[np.ndarray | None, str]:
    """``(d, source)``: the depth model's disparity of the locked back view (``S x S`` in
    [0, 1], 1 = nearest the back camera) or None (``"flat"``)."""
    if mode == "flat":
        return None, "flat"
    if depth_fn is None:
        if mode == "depth":
            warnings.append("No depth model was available for the back picture, so the back is smooth.")
        return None, "flat"
    a = np.clip(np.asarray(back.alpha, dtype=np.float32), 0.0, 1.0)[..., None]
    comp = np.clip(np.rint(back.rgb.astype(np.float32) * a + _GREY * (1.0 - a)), 0, 255).astype(np.uint8)
    try:
        res = depth_fn(comp)
    except Cancelled:
        raise
    except Exception as e:  # the relief is a nicety: a smooth back instead
        log.warning("Depth of the back picture failed (%s); smooth back", e)
        warnings.append("The depth of the back picture could not be estimated, so the back is smooth.")
        return None, "flat"
    if res is None or getattr(res, "model_id", "") == "heuristic":
        if mode == "depth":
            warnings.append("No depth model was available for the back picture, so the back is smooth.")
        return None, "flat"
    d = np.asarray(getattr(res, "disparity", res), dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.ndim != 2 or not np.isfinite(d).any():
        warnings.append("The depth of the back picture was unusable, so the back is smooth.")
        return None, "flat"
    if d.shape != (fr.S, fr.S):
        d = cv2.resize(d, (fr.S, fr.S), interpolation=cv2.INTER_LINEAR)
    d = np.nan_to_num(d, nan=float(np.nanmedian(d)))
    # normalised over the subject only: the model's values on the grey background are arbitrary
    inside = np.asarray(back.alpha) >= 0.5
    vals = d[inside] if inside.any() else d.ravel()
    lo, hi = (float(t) for t in np.percentile(vals, [0.5, 99.5]))
    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0) if hi > lo else np.zeros_like(d)
    return d.astype(np.float32), "depth"


def _huber_line(x: np.ndarray, y: np.ndarray, delta: float, iters: int = 10) -> tuple[float, float]:
    """Huber IRLS fit ``y ~ b + a x``; ``a`` is clipped at 0 (then ``b`` alone is refitted)."""
    w = np.ones(len(x))
    a, b = 0.0, float(np.median(y))
    fit_a = float(np.ptp(x)) > 1e-3
    for _ in range(iters):
        sw = np.sqrt(w)
        if fit_a:
            X = np.stack([x, np.ones_like(x)], axis=1)
            sol = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
            a, b = float(sol[0]), float(sol[1])
        else:
            b = float((w * y).sum() / w.sum())
        r = np.abs(y - (b + a * x))
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
        if fit_a and a <= 0:
            fit_a, a = False, 0.0
    return a, b


def _smooth_rows(vals: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware Gaussian smoothing along a 1-D row profile, extended to every row."""
    ok = np.isfinite(vals)
    out = np.full(len(vals), np.nan)
    if not ok.any():
        return out
    r = max(1, int(math.ceil(3 * sigma)))
    kx = np.arange(-r, r + 1)
    ker = np.exp(-0.5 * (kx / max(sigma, 1e-6)) ** 2)
    num = np.convolve(np.where(ok, vals, 0.0), ker, mode="same")
    den = np.convolve(ok.astype(np.float64), ker, mode="same")
    good = den > 0.05 * ker.sum()
    out[good] = num[good] / den[good]
    idx = np.arange(len(vals))
    return np.interp(idx, idx[good], out[good]) if good.any() else out


def _side_back_rows(sides: list[GenView], P_S: np.ndarray, fr: _Frame) -> np.ndarray:
    """``z_back(j)`` (z_S): the farthest back edge of the left / right views per row."""
    zb = np.full(fr.S, np.nan)
    for v in sides:
        if abs(_sin(v.phi_deg)) <= _SIDE_EXACT_SIN:
            continue
        _, u_b = _offset_rows(v.alpha, v.phi_deg, P_S, fr, float(v.meta.get("edge_bias_px", 0.0)))
        z = math.copysign(1.0, _sin(v.phi_deg)) * (v.meta["offset_m"] + (u_b - fr.S / 2.0) * fr.m)
        zb = np.fmax(zb, z)
    return zb


# =========================================================================== V9 / V10
@dataclass
class _Pts:
    """Splat centres in S with their local axes scaled by the axis sigmas (columns, S frame),
    for per-view screen footprints."""

    P: np.ndarray  # (N, 3)
    B: np.ndarray  # (N, 3, 3)

    @staticmethod
    def of(sc: SplatCloud, fr: _Frame, rows: np.ndarray | None = None) -> _Pts:
        means = np.asarray(sc.means, dtype=np.float64)
        quats = np.asarray(sc.quats, dtype=np.float64)
        scales = np.abs(np.asarray(sc.scales, dtype=np.float64))
        if rows is not None:
            means, quats, scales = means[rows], quats[rows], scales[rows]
        B = np.einsum("ji,njk->nik", fr.R, quat_to_rotmat(quats)) * scales[:, None, :]
        return _Pts(P=fr.to_S(means), B=B)

    def __add__(self, other: _Pts) -> _Pts:
        return _Pts(P=np.concatenate([self.P, other.P]), B=np.concatenate([self.B, other.B]))

    def __len__(self) -> int:
        return len(self.P)


def _footprint_px(B: np.ndarray, f: np.ndarray, m: float) -> np.ndarray:
    """Splat radius (px, clipped to [1, 4] as in design V9) of each Gaussian in a view
    looking along ``f``: twice its largest axis sigma projected onto the picture plane (a
    surface seen at a slant covers less of the picture than its 3D size)."""
    along = np.einsum("nik,i->nk", B, f)
    proj2 = np.maximum((B * B).sum(axis=1) - along * along, 0.0)
    return np.clip(2.0 * np.sqrt(proj2.max(axis=1)) / m, *_SPLAT_R)


def _zbuffer(pts: _Pts, v: GenView, fr: _Frame) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Point z-buffer of ``pts`` in the registered view ``v`` (each point a disc of its own
    footprint radius, grouped by whole pixels so every group takes the fast single-radius
    path of :func:`~vrphoto.geometry.register.point_zbuffer`); also returns ``(u, v,
    depth, radius)`` of the points."""
    u, vv, d = _project(pts.P, v.phi_deg, float(v.meta["offset_m"]), fr)
    _, f = _axes(v.phi_deg)
    r = _footprint_px(pts.B, f, fr.m)
    rq = np.rint(r)
    zbuf = np.full((fr.S, fr.S), np.inf)
    for rr in np.unique(rq):
        g = rq == rr
        zb, _ = point_zbuffer(u[g], vv[g], d[g], (fr.S, fr.S), radius=float(rr))
        np.minimum(zbuf, zb, out=zbuf)
    return zbuf, (u, vv, d, r)


def _colour_from_views(
    gen: _Pts,
    n_S: np.ndarray,
    occ: _Pts,
    views: list[GenView],
    bled: list[np.ndarray],
    fr: _Frame,
    cancel: CancelToken | None,
) -> tuple[np.ndarray, np.ndarray]:
    """V9: weighted view colours ``(colours in [0, 1] or NaN, weight sums)`` of the generated
    Gaussians ``gen`` with outward normals ``n_S``, weight ``max(0, n . -f)^8``.  A Gaussian
    is visible in a view when its depth is within 1.5 cm of the point z-buffer (occluders
    ``occ`` + the generated ones) plus the depth change across its own disc on a slanted
    surface (``radius * m * tan(theta)``, at most 5 cm) - a fixed tolerance hides a sloped
    surface behind its own neighbours."""
    n = len(gen)
    acc = np.zeros((n, 3))
    wsum = np.zeros(n)
    allp = occ + gen if len(occ) else gen
    S = fr.S
    for v, img in zip(views, bled, strict=True):
        _check(cancel)
        zbuf, (u, vv, d, r) = _zbuffer(allp, v, fr)
        u, vv, d, r = u[-n:], vv[-n:], d[-n:], r[-n:]
        with np.errstate(invalid="ignore"):
            ok = (u >= 0) & (u < S) & (vv >= 0) & (vv < S)
        iu = np.clip(np.floor(np.where(ok, u, 0)).astype(np.int64), 0, S - 1)
        iv = np.clip(np.floor(np.where(ok, vv, 0)).astype(np.int64), 0, S - 1)
        _, f = _axes(v.phi_deg)
        c = np.maximum(0.0, -(n_S @ f))
        slope = np.sqrt(np.maximum(1.0 - c * c, 0.0)) / np.maximum(c, 0.25)
        tol = _VIS_TOL_M + np.minimum(_VIS_SLOPE_MAX_M, r * fr.m * slope)
        vis = ok & (d <= zbuf[iv, iu] + tol)
        w = np.where(vis, c**_COLOR_POWER, 0.0)
        sel = np.flatnonzero(w > 1e-6)
        if not len(sel):
            continue
        col = _remap_points(img, u[sel] - 0.5, vv[sel] - 0.5, border=_GREY).astype(np.float64) / 255.0
        acc[sel] += w[sel, None] * col
        wsum[sel] += w[sel]
    out = np.where(wsum[:, None] > 1e-6, acc / np.maximum(wsum, 1e-12)[:, None], np.nan)
    return out, wsum


def _photo_inside(P_C: np.ndarray, front: FrontModel, matte: np.ndarray, D_f: np.ndarray) -> np.ndarray:
    """Inside the photo's subject: projects into the mould matte and lies behind its front."""
    u, v, z = front.cam.project(P_C)
    cell = front.grid.cell_of(u, v)
    ok = cell >= 0
    ci = np.where(ok, cell, 0)
    zf = D_f.ravel()[ci]
    with np.errstate(invalid="ignore"):
        return ok & matte.ravel()[ci] & (z >= zf - depth_tol(np.nan_to_num(zf, nan=1.0)))


def _fill_holes(
    sides: list[GenView],
    bled: list[np.ndarray],
    pts: _Pts,
    n_back: int,
    front: FrontModel,
    canon: Canon,
    matte: np.ndarray,
    D_f: np.ndarray,
    cviews: list[tuple[float, float, np.ndarray, int]],
    fr: _Frame,
    warnings: list[str],
    info: dict[str, Any],
    cancel: CancelToken | None,
) -> tuple[SplatCloud, np.ndarray]:
    """V10: silhouette pixels of a side view (matte eroded 2 px) that no Gaussian covers are
    lifted onto the hull entry along that view's ray (inside the photo's subject and the
    voted hull), as one view-facing Gaussian of 0.7 px with that pixel's colour.  A view
    with more holes than 20 % of the back's Gaussians does not fit and is skipped."""
    span = 0.75 * max(float(canon.height_s), float(canon.width_s)) + 0.3
    t = np.linspace(-span, span, _HOLE_SAMPLES)
    clouds: list[SplatCloud] = []
    normals: list[np.ndarray] = []
    for v, img in zip(sides, bled, strict=True):
        _check(cancel)
        zbuf, _ = _zbuffer(pts, v, fr)
        sil = cv2.erode((np.asarray(v.alpha) >= 0.5).astype(np.uint8), _disc(_HOLE_ERODE_PX)) > 0
        holes = sil & ~np.isfinite(zbuf)
        n_h = int(holes.sum())
        rec = {"name": v.name, "holes": n_h, "lifted": 0}
        info.setdefault("views", []).append(rec)
        if n_h == 0:
            continue
        if n_h > _HOLE_MAX_FRAC * max(n_back, 1):
            rec["skipped"] = True
            msg = (
                f"The {v.name} picture ({v.meta.get('file', v.name)}) does not fit the rest of the shape; "
                f"its uncovered parts were not filled in."
            )
            warnings.append(msg)
            log.warning(msg)
            continue
        iv, iu = np.nonzero(holes)
        w, f = _axes(v.phi_deg)
        o = float(v.meta["offset_m"])
        base = (o + (iu + 0.5 - fr.S / 2.0) * fr.m)[:, None] * w
        base[:, 1] = fr.cy + (iv + 0.5 - fr.S / 2.0) * fr.m
        hit = np.full(len(iv), np.nan)

        def inside(q: np.ndarray) -> np.ndarray:
            return _photo_inside(fr.to_C(q), front, matte, D_f) & _vote(q, cviews, fr, "vote")

        for a in range(0, len(iv), _HULL_CHUNK):  # bounded memory: chunks of rays x samples
            _check(cancel)
            b0 = base[a : a + _HULL_CHUNK]
            ins = inside((b0[:, None, :] + t[None, :, None] * f).reshape(-1, 3)).reshape(len(b0), -1)
            first = np.argmax(ins, axis=1)
            rows = np.flatnonzero(ins.any(axis=1) & (first > 0))  # the entry is crossed, not started in
            lo, hi = t[first[rows] - 1], t[first[rows]]
            for _ in range(_HULL_BISECT):
                mid = 0.5 * (lo + hi)
                ok = inside(b0[rows] + mid[:, None] * f)
                hi = np.where(ok, mid, hi)
                lo = np.where(ok, lo, mid)
            hit[a + rows] = hi
        rows = np.flatnonzero(np.isfinite(hit))
        if not len(rows):
            continue
        P_hit = base[rows] + hit[rows, None] * f
        n = len(rows)
        nrm_S = np.broadcast_to(-f, (n, 3))
        a2 = np.cross(-f, w)
        frame_S = np.stack([np.broadcast_to(w, (n, 3)), np.broadcast_to(a2, (n, 3)), nrm_S], axis=2)
        frame_C = np.einsum("ij,njk->nik", fr.R, frame_S)
        s12 = _HOLE_SCALE * fr.m
        col = img[iv[rows], iu[rows]].astype(np.float32) / 255.0
        sc = SplatCloud(
            means=fr.to_C(P_hit).astype(np.float32),
            scales=np.tile(np.array([s12, s12, max(_S3_MIN, 0.1 * s12)], np.float32), (n, 1)),
            quats=rotmat_to_quat(frame_C).astype(np.float32),
            colors=col,
            opacity=np.full(n, _HOLE_OPACITY, np.float32),
        )
        clouds.append(sc)
        normals.append((nrm_S @ fr.R.T).astype(np.float32))
        B = frame_S * np.array([s12, s12, max(_S3_MIN, 0.1 * s12)])[None, None, :]
        pts = pts + _Pts(P=P_hit, B=B)  # later views see these as covered
        rec["lifted"] = n
    if not clouds:
        return _empty_cloud(), np.zeros((0, 3), np.float32)
    return concat_splats(clouds), np.concatenate(normals)


# =========================================================================== the source
def _failed(source: str, reason: str, views: list[GenView], diag: dict[str, Any], t0: float) -> BackPart:
    diag["ok"] = False
    diag["reason"] = reason
    diag["seconds"] = round(time.perf_counter() - t0, 3)
    log.warning("View-based back (%s) not usable: %s", source, reason)
    return BackPart(
        sc=_empty_cloud(), kind="mould", source=source, ok=False, reason=reason, views=views, diag=diag
    )


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(float(x)) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    return x


def _register_nominal_set(
    vlist: list[GenView],
    front: FrontModel,
    canon: Canon,
    ortho: OrthoInput,
    fr: _Frame,
    color_match: bool,
    warnings: list[str],
    diag: dict[str, Any],
    rep: Any,
    cancel: CancelToken | None,
) -> tuple[_Reg, np.ndarray]:
    """V2-V5 for nominal views: the back lock, side registration, the rim-band colours and
    the per-view offsets.  Returns the registration and the front points in S."""
    reg = _Reg()
    matte, D_f = _front_maps(front)
    backs = [v for v in vlist if _is_back(v.phi_deg)]
    sides_in = [v for v in vlist if not _is_back(v.phi_deg) and abs(_sin(v.phi_deg)) > _SIDE_MIN_SIN]
    for v in vlist:
        if not _is_back(v.phi_deg) and abs(_sin(v.phi_deg)) <= _SIDE_MIN_SIN:
            diag["views"].append({"name": v.name, "phi": v.phi_deg, "used": False, "reason": "a front view"})
    if not backs:
        reg.reason, reg.views = "there is no picture of the back (a view from behind, 180°)", vlist
        return reg, np.zeros((0, 3))

    # ---- V2 / V3: the back
    rep(0.0, "Fitting your back photo to the outline")
    back, binfo = _lock_back(backs[0], fr, cancel)
    diag["back"] = binfo
    for extra in backs[1:]:
        diag["views"].append(
            {"name": extra.name, "phi": extra.phi_deg, "used": False, "reason": "second back"}
        )
    if not binfo["ok"]:
        reg.reason, reg.views = binfo["reason"], [back, *sides_in]
        return reg, np.zeros((0, 3))
    _check(cancel)

    # ---- V2: side / quarter views
    rep(0.12, "Lining up the side photos")
    Pf_C = _front_points(front, D_f, matte)
    Pf_S = fr.to_S(Pf_C)
    cut0 = (bool(getattr(canon, "cut_top", False)), bool(getattr(canon, "cut_bottom", False)))
    sides: list[GenView] = []
    rejected: list[GenView] = []
    for v in sides_in:
        if _preregistered(v, fr):  # a view set registered before
            sides.append(replace(v, meta={**v.meta, "used": True, "reason": "", "offset_given": True}))
            diag["views"].append({"name": v.name, "phi": v.phi_deg, "used": True, "registered": "given"})
            continue
        rv, info = _register_side(v, fr, Pf_S, cut0)
        diag["views"].append(info)
        if rv is None:
            msg = f"The {v.name} picture ({v.meta.get('file', v.name)}) was not used: {info['reason']}."
            warnings.append(msg)
            rejected.append(replace(v, meta={**v.meta, "used": False, "reason": info["reason"]}))
        else:
            sides.append(rv)

    # ---- V4: colours toward the photo's rim
    if color_match:  # (not twice: a view set registered before is matched already)
        todo = [i for i, v in enumerate([back, *sides]) if "color_match" not in v.meta]
        done = _match_colours([[back, *sides][i] for i in todo], ortho, fr)
        both = [back, *sides]
        for i, v in zip(todo, done, strict=True):
            both[i] = v
        back, *sides = both
    diag["color_match"] = bool(color_match)
    _check(cancel)

    # ---- V5: offsets of the side views
    orbit: dict[str, Any] = {}
    if sides:
        sides, orbit = _solve_offsets(sides, Pf_S, fr, canon, warnings)
    diag["orbit"] = orbit
    rep(0.2, "Lining up the side photos")
    reg.back, reg.binfo, reg.sides, reg.orbit = back, binfo, sides, orbit
    reg.views = [back, *sides, *rejected]
    return reg, Pf_S


def qc_cost(diag: Mapping[str, Any] | BackPart | None) -> float:
    """V11: the QC cost of a view-based back for choosing between tries (lower is better):
    ``diag["qc"]["cost"]`` of :func:`view_based_back` (the sum of its ``terms``: ``1 -
    IoU_back``, the views' height spread, the orbit-axis disagreement of the +-90 views over
    its tolerance, the worst row MAD / 3 cm, and for orthographic sets ``1 - IoU`` of view 0
    and 0.5 per dropped view; the yaw and face terms are 0 until M6).  ``inf`` for an
    unusable back (``ok`` False) or a diag without a QC cost.  Accepts the ``BackPart`` or
    its ``diag``."""
    if isinstance(diag, BackPart):
        if not diag.ok:
            return math.inf
        diag = diag.diag
    if not isinstance(diag, Mapping) or diag.get("ok") is False:
        return math.inf
    qc = diag.get("qc")
    if not isinstance(qc, Mapping):
        return math.inf
    cost = qc.get("cost")
    if cost is None:
        terms = qc.get("terms")
        if not isinstance(terms, Mapping):
            return math.inf
        cost = sum(_num_or(t, math.inf) for t in terms.values() if t is not None)
    c = _num_or(cost, math.inf)
    return c if math.isfinite(c) else math.inf


def choose_view_set(results: Sequence[BackPart | Mapping[str, Any] | None]) -> int | None:
    """Design 3.5.3: the index of the usable try with the lowest :func:`qc_cost` (the first
    on a tie), None when no try is usable.  ``results`` are the ``BackPart`` objects (or
    their diags) of :func:`view_based_back`, one per generated view set."""
    best: int | None = None
    best_c = math.inf
    for i, r in enumerate(results):
        c = qc_cost(r) if r is not None else math.inf
        if c < best_c:
            best, best_c = i, c
    return best


def view_based_back(
    front: FrontModel,
    canon: Canon,
    ortho: OrthoInput,
    views: Sequence[GenView],
    *,
    depth_fn: Callable[[np.ndarray], Any] | None = None,
    relief: str = "auto",
    color_match: bool = True,
    stride: int = 1,
    seed: int = 0,
    fill_holes: bool = True,
    source: str = "images",
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> BackPart:
    """The back from views of the subject (design 3.5.2 V2-V11; see the module docstring).

    ``views`` are segmented :class:`GenView` objects (:func:`segment_views`); one at
    phi 180 (the back) is required, side / quarter views (|sin phi| > 0.3) are optional
    (front views are ignored).  When the back view is ``kind="ortho"`` the set has exact
    orthographic cameras (MV-Adapter, :func:`views_from_cameras`, or a saved views folder):
    the orthographic checks of the module docstring replace the nominal registration
    (nominal pictures among them are not used, view 0 is the lateral check and the colour
    reference).  Otherwise every view is nominal; side views already registered in this
    levelled frame (``meta["registered"]``, ``S x S``, ``m == ortho.m_in``,
    ``meta["offset_m"]``: the views of an earlier run) skip V2 and V5.
    ``depth_fn(rgb) -> disparity`` (``S x S`` in [0, 1], 1 = nearest the back camera; a
    ``DepthResult`` or None is accepted, a ``"heuristic"`` result counts as none) gives the
    relief when ``relief != "flat"``.  ``color_match`` applies V4 (the Reinhard rim-band
    transfer; one affine colour map for an orthographic set).  ``stride`` is the mould's
    Gaussian stride, ``seed`` fixes the subsampling of occluders, ``fill_holes`` (addition)
    switches V10 off for the fast quality, ``source`` (addition) names the result.

    Returns ``BackPart(kind="mould", source=source, views=registered views, hull=...,
    diag=...)``; ``ok=False`` with a plain ``reason`` when there is no back view, its outline
    does not match the photo's (IoU < 0.75 after the similarity and perspective stages; for
    an orthographic set also view 0's after the lateral similarity) or it shows a different
    pose (after them, 2 % of the outlines are still more than 6 % of the subject's size
    apart), or an orthographic back's size differs from view 0's by more than 5 %.  ``diag``
    is JSON-ready: ``back`` (similarity, perspective matrix, IoU after the similarity /
    perspective (``iou``) / lock, ``pose_frac`` - the outline gap of the pose check - and
    the lock's largest move), ``orbit`` (per-view offsets and row MAD, mirrored pairs and
    their naming decision, left/right agreement, ``z_axis`` - the orbit axis depth in C
    when left / right views were used; for an orthographic set also ``sign`` - the MAD of
    both azimuth-sign hypotheses and whether the views were turned round), ``lateral`` and
    ``colour_fit`` (orthographic sets), ``views`` (also each side view's scale and anchor
    row), ``hull``, ``relief``, ``colours``, ``holes``, ``qc`` (``cost`` and its terms,
    :func:`qc_cost`); ``diag["warning"]`` joins the plain sentences for the user (also
    listed in ``diag["warnings"]``).  Raises :class:`InputError` for invalid inputs and
    :class:`Cancelled`."""
    t_start = time.perf_counter()
    rep = _as_reporter(progress)
    mode = str(relief).strip().lower() if relief is not None else "auto"
    if mode not in RELIEFS:
        raise InputError(f"Unknown back relief {relief!r}.", hint=f"Choose one of: {', '.join(RELIEFS)}.")
    fr = _frame(ortho, canon)
    matte, D_f = _front_maps(front)
    vlist = list(views or [])
    for v in vlist:
        if not isinstance(v, GenView):
            raise InputError(f"Expected GenView objects (got {type(v).__name__}).")
    warnings: list[str] = []
    diag: dict[str, Any] = {"source": source, "seed": int(seed), "views": [], "warnings": warnings}
    _check(cancel)

    backs = [v for v in vlist if _is_back(v.phi_deg)]
    if backs and backs[0].kind == "ortho":
        diag["cameras"] = "orthographic"
        Pf_S = fr.to_S(_front_points(front, D_f, matte))
        reg = _register_ortho_set(
            vlist, canon, ortho, fr, Pf_S, color_match, warnings, diag, rep, cancel, front_sc=front.sc
        )
    else:
        reg, Pf_S = _register_nominal_set(
            vlist, front, canon, ortho, fr, color_match, warnings, diag, rep, cancel
        )
    if reg.reason:
        return _failed(source, reg.reason, reg.views, _jsonable(diag), t_start)
    back, binfo, sides, orbit = reg.back, reg.binfo, reg.sides, reg.orbit
    assert back is not None
    used_sides = [v for v in sides if v.meta.get("used")]
    exact = [v for v in used_sides if abs(_sin(v.phi_deg)) > _SIDE_EXACT_SIN]
    _check(cancel)

    # ---- V6: hull
    cviews = _constraining(used_sides, fr)
    hull: HullIntervals | None = None
    if cviews:
        hull = hull_intervals(front, canon, ortho, used_sides, cancel=cancel)
        n_valid = int(hull.valid.sum())
        diag["hull"] = {
            "views": len(cviews),
            "valid_rays": n_valid,
            "matte_rays": int(matte.sum()),
            "volume_m3": _hull_volume(front, hull),
        }
        if n_valid == 0:
            hull = None
    rep(0.4, "Building the back surface")
    _check(cancel)

    # ---- V7: the back depth target
    tb = fr.alpha0[:, ::-1]
    d, d_src = _relief_map(back, fr, depth_fn, mode, warnings)
    T_prior = float(canon.T_prior)
    H = float(canon.height_s)
    thick = T_prior / H if H > 0 else 0.11
    Zp, _ = inflation_prior(front, canon, thickness=thick)
    Zp = np.maximum(np.asarray(Zp, dtype=np.float64), D_f + MIN_GAP_M)
    cells = np.flatnonzero(matte.ravel())
    rS = _rays_S(front, fr, cells)
    c_S = fr.c @ fr.R
    zhat = fr.R[:, 2]
    rz = rS[:, 2]  # = zhat . ray
    zc = float(zhat @ fr.c)
    df = D_f.ravel()[cells]
    zp = Zp.ravel()[cells]
    hv = np.zeros(len(cells), bool) if hull is None else np.asarray(hull.valid).ravel()[cells]
    z_out = np.full(len(cells), np.nan) if hull is None else np.asarray(hull.z_out, np.float64).ravel()[cells]
    rel: dict[str, Any] = {"asked": mode, "source": d_src}
    zb_rows = _side_back_rows(exact, Pf_S, fr) if exact else np.full(fr.S, np.nan)
    sil_rows = np.flatnonzero(tb.any(axis=1))
    if d is None:  # no relief model: the back silhouette's round profile shapes each row
        prof, _ = inflation_profile(tb)
        dd = prof / max(float(prof.max()), 1e-9)
    else:
        dd = d
    dd = push_pull_fill(np.where(tb, dd, 0.0).astype(np.float32), tb)
    dmax = _row_percentile(np.repeat(np.arange(fr.S), fr.S)[tb.ravel()], dd.ravel()[tb.ravel()], fr.S, 99.0)
    ok_r = np.isfinite(dmax) & np.isfinite(zb_rows)
    with_sides = bool(exact) and int(ok_r.sum()) >= max(3, _MIN_ROWS_FRAC * len(sil_rows))
    if with_sides:
        rel["shape"] = "profile" if d is None else "depth"
        a_fit, b_fit = _huber_line(dmax[ok_r], zb_rows[ok_r], _RELIEF_HUBER_M)
        resid = np.where(ok_r, zb_rows - (b_fit + a_fit * dmax), np.nan)
        b_rows = b_fit + _smooth_rows(resid, max(1.0, _RELIEF_SMOOTH_FRAC * len(sil_rows)))
        b_rows = np.where(np.isfinite(b_rows), b_rows, b_fit)
        Z_lf = (b_rows[:, None] + a_fit * dd).astype(np.float32)
        rel.update(mode="sides", a=a_fit, b=b_fit, rows=int(ok_r.sum()))
        z = np.where(hv, z_out, zp)
        for _ in range(_TARGET_ITERS):
            qx, qy = _back_q(z[:, None] * rS - c_S, fr)
            Zs = _remap_points(Z_lf, qx, qy, border=np.nan).astype(np.float64)
            Zs = np.where(np.isfinite(Zs), Zs, z * rz - zc)
            z = (Zs + zc) / rz
    else:
        z = zp.copy()
        if d is not None:
            sil_h = max(2.0, float(len(sil_rows)))
            hp = d - _masked_blur(d, tb, max(1.0, _RELIEF_BLUR_FRAC * sil_h))
            scale = max(float(np.percentile(np.abs(hp[tb]), 99)), 1e-3)
            h_map = np.where(tb, np.clip(hp / scale, -1.0, 1.0), 0.0).astype(np.float32)
            h_map = push_pull_fill(h_map, tb)
            qx, qy = _back_q(z[:, None] * rS - c_S, fr)
            h = _remap_points(h_map, qx, qy, border=0.0).astype(np.float64)
            amp = min(DETAIL_M, 0.25 * T_prior)
            taper = np.clip(np.asarray(front.dt, dtype=np.float64).ravel()[cells] / 3.0, 0.0, 1.0)
            z = z + amp * h * taper
            rel.update(mode="detail", amplitude_m=amp)
        else:
            rel["mode"] = "prior"
    # clamps: behind the front, inside the envelope (hull, else the prior), carving <= 40 %
    env = np.where(hv, z_out if with_sides else np.minimum(z_out, zp), zp)
    lo = np.maximum(df + _TARGET_GAP_M, env - _CARVE_MAX * (env - df))
    hi = np.where(hv, z_out + _HULL_SLACK_M, np.inf if with_sides else np.maximum(zp, df) + DETAIL_M)
    if with_sides:
        hi = np.where(hv, hi, df + max(_HULL_RANGE_M, _HULL_RANGE_FRAC * H))
    hi = np.maximum(hi, lo)
    z = np.minimum(np.maximum(np.where(np.isfinite(z), z, env), lo), hi)
    target = np.full(matte.shape, np.nan, np.float32)
    target.ravel()[cells] = z
    diag["relief"] = rel
    # rim depth: the middle of the hull interval on boundary rays, else the front's edge
    D_ext = np.asarray(front.D_ext, dtype=np.float64)
    D_ext = np.where(np.isfinite(D_ext) & (D_ext > 0), D_ext, np.where(matte, D_f, np.nan))
    g = D_ext.copy()
    if hull is not None:
        hmid = 0.5 * (np.asarray(hull.z_in, np.float64) + np.asarray(hull.z_out, np.float64))
        g = np.where(np.asarray(hull.valid) & _boundary(matte) & (hmid > D_ext), hmid, g)
    # first colours (the back view at the target depth); V9 refines them per Gaussian
    bled_back = _bleed(back.rgb, back.alpha)
    qx, qy = _back_q(z[:, None] * rS - c_S, fr)
    cols = np.zeros((*matte.shape, 3), np.uint8)
    cols.reshape(-1, 3)[cells] = np.clip(np.rint(_remap_points(bled_back, qx, qy, border=_GREY)), 0, 255)
    _check(cancel)

    # ---- V8: the mould
    part = mould_back(
        front,
        canon,
        target,
        colors=cols,
        z_rim=g,
        stride=stride,
        source=source,
        hull=hull,
        progress=rep.sub(0.45, 0.7),
        cancel=cancel,
    )
    mdiag = dict(part.diag or {})
    diag["mould"] = {k: v for k, v in mdiag.items() if k != "warning"}
    if mdiag.get("warning"):
        warnings.append(str(mdiag["warning"]))
    all_views = reg.views
    if not part.ok or not len(part.sc):
        return _failed(source, part.reason or "the back came out empty", all_views, _jsonable(diag), t_start)
    _check(cancel)

    # ---- V9: colours from the views
    rep(0.72, "Matching colours")
    sc = part.sc
    n_gen = len(sc)
    normals = np.asarray(part.normals, dtype=np.float64) if part.normals is not None else np.zeros((n_gen, 3))
    gen = _Pts.of(sc, fr)
    n_S = normals @ fr.R
    rows = None
    if len(front.sc) > _MAX_OCCLUDERS:
        rng = np.random.default_rng(int(seed))
        rows = np.sort(rng.choice(len(front.sc), _MAX_OCCLUDERS, replace=False))
    occ = _Pts.of(front.sc, fr, rows) if len(front.sc) else _Pts(np.zeros((0, 3)), np.zeros((0, 3, 3)))
    col_views = [back, *used_sides]
    bled = [bled_back] + [_bleed(v.rgb, v.alpha) for v in used_sides]
    col, wsum = _colour_from_views(gen, n_S, occ, col_views, bled, fr, cancel)
    got = np.isfinite(col).all(axis=1)
    colors = np.asarray(sc.colors, dtype=np.float32).copy()
    colors[got] = col[got].astype(np.float32)
    sc = replace(sc, colors=colors)
    diag["colours"] = {
        "views": len(col_views),
        "coloured": int(got.sum()),
        "default": int((~got).sum()),
    }
    _check(cancel)

    # ---- V10: side pixels nothing covers, lifted onto the hull entry
    holes_info: dict[str, Any] = {}
    if fill_holes and used_sides:
        rep(0.85, "Filling the sides")
        n_back = int(mdiag.get("n_back", n_gen))
        extra, extra_n = _fill_holes(
            used_sides,
            bled[1:],
            occ + gen if len(occ) else gen,
            n_back,
            front,
            canon,
            matte,
            D_f,
            cviews,
            fr,
            warnings,
            holes_info,
            cancel,
        )
        if len(extra):
            sc = concat_splats([sc, extra])
            normals = np.concatenate([normals, extra_n.astype(np.float64)])
    diag["holes"] = holes_info

    # ---- V11: QC cost (lower is better; for choosing between tries)
    mads = [
        float(v.meta.get("mad_m", 0.0)) for v in used_sides if math.isfinite(float(v.meta.get("mad_m", 0.0)))
    ]
    agree = orbit.get("agree_m")
    tol = orbit.get("agree_tol_m") or max(_AGREE_M, _AGREE_FRAC * H)
    terms = {
        "iou_back": 1.0 - float(binfo.get("iou", 0.0)),
        "height_spread": 0.0,  # nominal views are scaled to the photo by construction
        "z_axis": abs(float(agree)) / tol if agree is not None else 0.0,
        "mad": (max(mads) / 0.03) if mads else 0.0,
        "yaw": 0.0,
        "face": 0.0,
        **reg.qc,  # orthographic sets: height spread, view 0's overlap, dropped views
    }
    diag["qc"] = {"cost": float(sum(terms.values())), "terms": terms}
    diag.update(
        ok=True,
        n_generated=len(sc),
        n_holes=len(sc) - n_gen,
        seconds=round(time.perf_counter() - t_start, 3),
    )
    if warnings:
        diag["warning"] = " ".join(dict.fromkeys(warnings))
    rep(1.0, "Back ready")
    log.info(
        "View-based back (%s): %d Gaussians (+%d hole fills), %d side view(s), relief %s, QC %.3f, %.1f s",
        source,
        n_gen,
        len(sc) - n_gen,
        len(used_sides),
        rel.get("mode"),
        diag["qc"]["cost"],
        diag["seconds"],
    )
    return BackPart(
        sc=sc,
        kind="mould",
        source=source,
        ok=True,
        reason="",
        normals=normals.astype(np.float32),
        back_z=part.back_z,
        hull=hull,
        views=all_views,
        diag=_jsonable(diag),
        spacing_m=part.spacing_m,
    )


# =========================================================================== contact sheet
_OUTLINE_EXPECTED = (60, 230, 60)
_OUTLINE_OWN = (40, 200, 255)
_OUTLINE_REJECTED = (240, 60, 60)


def _tile(
    img: np.ndarray,
    outlines: list[tuple[np.ndarray, tuple[int, int, int]]],
    label: str,
    tile: int,
    labels: bool,
) -> np.ndarray:
    """One letterboxed ``tile x tile`` picture with outlines and a label."""
    h, w = img.shape[:2]
    s = tile / max(h, w)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    canvas = np.full((tile, tile, 3), 40, np.uint8)
    ox, oy = (tile - nw) // 2, (tile - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = small
    th = max(1, round(tile / 200))
    for mask, colour in outlines:
        m = np.asarray(mask)
        if m.shape != (h, w) or not m.any():
            continue
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pts = [
            np.rint((c.reshape(-1, 2).astype(np.float64) + 0.5) * s - 0.5 + (ox, oy)).astype(np.int32)
            for c in contours
        ]
        cv2.polylines(canvas, pts, True, colour, th, cv2.LINE_AA)
    if labels and label:
        fs = 0.45 * tile / 256.0
        org = (max(3, round(6 * tile / 256)), max(10, round(18 * tile / 256)))
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th + 2, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return canvas


def contact_sheet(
    ortho: OrthoInput, views: Sequence[GenView], *, tile: int = 256, labels: bool = True
) -> np.ndarray:
    """A preview of the view-based back's inputs (design 3.13, ``_sharp360_views.png``): the
    levelled photo with its outline, then every view with the outline it was expected to
    have drawn on top (green: the front silhouette, mirrored for the back - only for views
    registered into the levelled frame) and its own matte (blue; red when the view was not
    used), labelled with name and azimuth.  At most 4 tiles per row; uint8 RGB of
    ``rows * tile x cols * tile``."""
    try:
        t = int(tile)
    except (TypeError, ValueError):
        t = 0
    if t < 32:
        raise InputError(f"The contact sheet tile must be at least 32 pixels (got {tile!r}).")
    rgba = np.asarray(ortho.rgba)
    a0 = np.asarray(ortho.alpha0, dtype=bool)
    S = int(ortho.size)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    base = np.clip(np.rint(rgba[..., :3] * alpha + _GREY * (1 - alpha)), 0, 255).astype(np.uint8)
    tiles = [_tile(base, [(a0, _OUTLINE_EXPECTED)], "photo (levelled) 0 deg", t, labels)]
    for v in views:
        rgb = _rgb_u8(v.rgb, f"view '{v.name}'")
        own = np.asarray(v.alpha) >= 0.5
        used = v.meta.get("used", True) is not False
        outlines = []
        if v.meta.get("registered") and rgb.shape[:2] == (S, S):
            if _is_back(v.phi_deg):
                outlines.append((a0[:, ::-1], _OUTLINE_EXPECTED))
            elif _az_diff(v.phi_deg, 0.0) < 1.0:
                outlines.append((a0, _OUTLINE_EXPECTED))
        outlines.append((own, _OUTLINE_OWN if used else _OUTLINE_REJECTED))
        label = f"{v.name} {v.phi_deg:.0f} deg" + ("" if used else " (not used)")
        tiles.append(_tile(rgb, outlines, label, t, labels))
    cols = min(4, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * t, cols * t, 3), 24, np.uint8)
    for i, tl in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * t : (r + 1) * t, c * t : (c + 1) * t] = tl
    return sheet


__all__ = [
    "MAX_PHOTO_LONG",
    "PHI_NAMES",
    "RELIEFS",
    "VIEWS_FORMAT",
    "VIEWS_JSON",
    "VIEW_KINDS",
    "VIEW_MAX_LONG",
    "VIEW_NAMES",
    "GenView",
    "OrthoInput",
    "choose_view_set",
    "contact_sheet",
    "hull_intervals",
    "levelled_ortho_input",
    "load_views",
    "qc_cost",
    "save_views_folder",
    "segment_views",
    "view_based_back",
    "views_from_cameras",
    "views_from_folder",
]
