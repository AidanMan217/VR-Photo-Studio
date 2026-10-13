"""SHARP 360: optional Brush polish of a back made from views (design 3.11, 1.2 / 1.3, 4.5).

A view-based back (MV-Adapter's six views, the user's own back / side photos) is lifted
onto a mould one view at a time, so the generated views never had to agree with each other
in 3D.  Brush (the Capture add-on's Gaussian-splat trainer, run in-process through
:func:`vrphoto.addons.capture.train_splat`) trains the whole stitched cloud against all the
pictures at once, which makes them mutually consistent and blends the seam.  The
photographed front is protected the only way Brush 0.3.0 allows (it cannot freeze
Gaussians, its training background is fixed black, a black mask pixel means "ignore" and a
masked picture does not train alpha): after training, the EXACT kept SHARP rows are put back.

**Dataset** (``<work>/brush_ds``, world frame = SHARP's camera frame C, OpenCV axes, the
COLMAP / Brush convention): ``images/``, ``masks/`` (black = ignore), the COLMAP text model
``sparse/0/{cameras,images,points3D}.txt`` (one PINHOLE camera per picture, 20k initial
means as points) and ``init.ply`` - the stitched cloud (DC colour only), read back to
validate, the only ``.ply`` in the folder (Brush always starts from it).  The export folder
(``<work>/brush_out``) lies outside the dataset.  Pictures (:func:`build_frames`):

* the photo, three times (``p0_``, ``p1_``, ``p2_``: distinct names, so it weighs three
  pictures): RGBA with the subject's matte as alpha (alpha is trained - nothing may appear
  around the outline from the photo's camera), cropped to the subject plus 8 % and at most
  1024 px on its long side, the intrinsics moved / scaled with it, identity pose;
* SHARP **anchors** (needs masks): CPU renders (:mod:`~vrphoto.geometry.splat_render`) of
  the cleaned front at yaw +-12, +-24 deg and pitch +-10 deg about the subject centre
  ``canon.c`` (``best`` adds yaw +-18 deg at pitch -8 deg and pitch +8 deg), RGB on black at
  <= 768 px.  Mask = (the pixel shows the photographed surface - its depth, projected into
  the photo, is the front depth there - facing the photo camera within the seam angle and
  away from the outline, with render alpha >= 0.95) OR (more than 3 % of the subject's size
  outside its rendered silhouette: a black target, so Brush removes floaters there);
* the generated views (``part.views``, registered into the levelled frame of
  ``views_lift``): those at least 67.5 deg from the front (180, +-90, and 135 / 225 for
  photos) as RGBA with their (locked) alpha; the front quarter views (+-45) as RGB on black
  with mask = NOT(the photographed, reliable front as seen in that view, dilated 5 px) - the
  generated colours never compete with the photo there; views within 22.5 deg of the front
  are left out (the photo is the front).  An orthographic view becomes a telephoto pinhole
  (:func:`telephoto_camera`): centre ``a - 40 m * forward`` with ``a`` the point that maps to
  the picture centre at the orbit axis depth, ``f = 40 m / m``, ``cx = cy = N / 2``.  The
  residual perspective moves a point by ``|r| |d| / (40 m + d)`` (``r``: its offset from the
  picture centre, ``d``: its depth from the axis plane): a few millimetres on the body, about
  1 cm at a hand held out toward a side camera.  Brush takes its scene bounds from the
  initial splats, so the far cameras are harmless (the Brush self-test uses the same cameras).

**Command** (:func:`vrphoto.addons.capture.brush_args` via ``train_splat``):
``--total-steps N``, ``max_splats = clamp(1.3 n_init, 300k, 2.5M)``, ``--sh-degree 0
--growth-stop-iter N/2 --match-alpha-weight 0.5 --max-resolution 1024``.  The Brush
self-test result (``Settings.brush_test``) is honoured: SH degree 1 when degree 0 failed
there, no ``--sh-degree`` when Brush did not accept it, no masked pictures (anchors and
quarter views) when masks could not be mixed with transparent pictures.

**Retries and checks.**  A failed run is retried once with ``--sh-degree 1`` when it ran
with 0; once with ``--max-resolution 768`` and the cap x 0.6 when it ran out of memory; an
option Brush rejects ("unexpected argument") is left out next time; without a passed
self-test (``polish=on``) the masked pictures are dropped as a last try.  A run that hit the
time limit, a missing Brush or a cancel ends the polish.  The export is rejected when it
has fewer than 1000 Gaussians, non-finite values, or a robust box (0.5-99.5th percentiles
per axis) more than 1.5x the initial cloud's - the result is then the unpolished cloud and
a plain reason (the Brush log is kept next to the outputs as ``<base>_sharp360_polish.log``).

**After training.**  Brush's copies of the front rows (``init.ply`` holds them too) are
dropped first: every trained Gaussian whose nearest initial row is an exact front row rather
than an unpolished back row (:func:`front_copies`, a voxel-hash nearest search within 2 cm).  The rest goes through the merge
(:func:`~vrphoto.geometry.splat_merge.merge_parts`) against the front as the generated part
(its kind is the back part's, so the rim wall survives): Gaussians in front of the
photographed surface, around the outline from the photo's camera or on the photo-facing
surface are dropped.  Then the EXACT kept front rows of the
stitched cloud (``merged[is_front]``, bit-identical) are put back in front of the trained
rows, and colour feathering and the floater filter run again.  A polish that keeps fewer
than 10 % as many back Gaussians as the unpolished back had (a cloud Brush moved away from
the photo) is rejected too.

:func:`polish_allowed` decides whether the polish runs at all (design 1.2 / 1.3): ``off``
never; ``auto`` only with ``best`` quality, a back made from views and a passed Brush
self-test; ``on`` whenever the back has views and Brush is installed (with a warning when
the self-test has not passed).
"""

from __future__ import annotations

import math
import re
import shutil
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import AddonError, AddonNotInstalled, Cancelled, InputError, VRPhotoError
from ..core.log import get_logger
from ..core.types import CancelToken, FloatMap, ProgressReporter, SplatCloud
from ..geometry.back_mould import BackPart
from ..geometry.splat import concat_splats, read_splats, subset_splats, write_splats
from ..geometry.splat_fusion import Canon, FrontModel, depth_tol
from ..geometry.splat_merge import feather_colors, merge_parts, remove_floaters
from . import run as R

log = get_logger(__name__)

#: Polish settings: auto | off | on.
POLISH_MODES: tuple[str, ...] = R.POLISH_MODES
#: Brush training steps when the job does not say (``ObjectJob.polish_steps``).
DEFAULT_STEPS = 5000
#: Orthographic views become pinholes this far away (as Brush's self-test cameras).
TELEPHOTO_DIST_M = 40.0
#: The photo is listed this many times (distinct file names) so it outweighs single views.
PHOTO_COPIES = 3
#: Long side of the photo copies (px); the crop is the subject plus this margin.
PHOTO_MAX_PX = 1024
_CROP_MARGIN = 0.08
#: Long side of the SHARP anchor renders (px; CPU time grows with the pixel count).
ANCHOR_MAX_PX = 768
#: ``(yaw, pitch)`` in degrees of the anchor renders about ``canon.c`` (+yaw: the camera moves
#: to its right as in the azimuth contract, +pitch: it moves up and looks down).
ANCHORS: tuple[tuple[float, float], ...] = (
    (12.0, 0.0),
    (-12.0, 0.0),
    (24.0, 0.0),
    (-24.0, 0.0),
    (0.0, 10.0),
    (0.0, -10.0),
)
#: Added with ``best`` quality.
ANCHORS_BEST: tuple[tuple[float, float], ...] = ((18.0, -8.0), (-18.0, -8.0), (0.0, 8.0))
_ANCHOR_ALPHA = 0.95  # anchor pixels trained as SHARP's front: at least this opaque ...
_ANCHOR_RIM_CELLS = 2.0  # ... and at least this far (mould cells) inside the photo's outline
_ANCHOR_OUT_FRAC = 0.03  # black target beyond this share of the subject's size outside it
_ANCHOR_MIN_FRAC = 0.01  # an anchor with fewer trained front pixels than this is left out
_FRONT_VIEW_DEG = 22.5  # views this close to the photo's side are not used (the photo is the front)
_QUARTER_DEG = 67.5  # views closer to the front than this are masked RGB, the others RGBA
_QUARTER_DILATE_PX = 5
_ZBUF_RADIUS_PX = 1.5  # point z-buffer of the stitched cloud in a quarter view
#: Initial means written to points3D.txt.
INIT_POINTS = 20_000
_CAP_PER_INIT = 1.3
CAP_RANGE = (300_000, 2_500_000)
MAX_RESOLUTION = 1024
OOM_RESOLUTION = 768
_OOM_CAP = 0.6
_MATCH_ALPHA_WEIGHT = 0.5
#: Output checks.
MIN_SPLATS = 1000
MAX_BOX_GROWTH = 1.5
_BOX_PCT = 0.5
_BOX_FLOOR_M = 0.01
MIN_KEPT_FRAC = 0.1
#: Trained Gaussians are labelled by their nearest initial row, searched in the 27 voxels
#: around them with these voxel edges in turn (:func:`front_copies`).
COPY_SEARCH_M = (0.0025, 0.008, 0.02)
_NN_PAIRS = 4_000_000  # candidate pairs per block of the nearest-row search (memory bound)
#: Time limit of one Brush run: base + per step (the GTX 1080 is expected at 10-20 steps/s).
TIMEOUT_BASE_S = 900.0
TIMEOUT_PER_STEP_S = 0.6
_MAX_RUNS = 6
_EXPORT_NAME = "polished.ply"
_LOG_NAME = "brush_polish.log"
_LABEL = "Polishing with Brush"
#: Brush options that may be left out when a Brush build does not know them.
_OPTIONAL_FLAGS = ("--sh-degree", "--growth-stop-iter", "--match-alpha-weight", "--max-resolution")
_FLAG_RE = re.compile(r"(?:unexpected argument|unrecognized (?:option|argument)|Found argument) '(--[\w-]+)'")
_OOM_RE = re.compile(r"out of memory|outofmemory|memory allocation|allocation failed", re.I)
_CORE_PLY = (
    "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
    "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip
_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8", "uchar": "u1", "uint8": "u1",
    "char": "i1", "int8": "i1", "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}  # fmt: skip
_TEST_HINT = (
    "'Test Brush on this PC' on the Capture add-on in the Add-ons window, or vrphoto-cli addons brush-test"
)


# --------------------------------------------------------------------------- when to polish
def _usable_views(part: BackPart | None) -> list[Any]:
    """The registered views of ``part`` that the back was built from."""
    if part is None or not part.ok or not part.views:
        return []
    return [
        v
        for v in part.views
        if getattr(v, "meta", None) and v.meta.get("registered") and v.meta.get("used", True)
    ]


def _brush_installed() -> bool:
    from ..addons import capture

    try:
        capture.find_brush()
    except AddonError:
        return False
    return True


def polish_allowed(
    opts: Any,
    part: BackPart | None,
    settings: Any,
    *,
    quality: str,
    brush_installed: bool | None = None,
) -> tuple[bool, str]:
    """Should the Brush polish run for this back?  Returns ``(allowed, note)`` (design 1.2/1.3).

    ``opts`` carries ``polish`` (``Opts`` or the job; a plain string is accepted too),
    ``settings`` the Brush self-test result ``brush_test`` (``Settings``, or the dict itself),
    ``quality`` the back quality.  Rules, in this order:

    * ``off`` - never;
    * ``auto`` - only with ``best`` quality;
    * the back must be made from views (registered ``part.views``: MV-Adapter, your back
      photo; the built-in back and TRELLIS have none to train against);
    * Brush (the Capture add-on) must be installed (``brush_installed``; None looks it up);
    * ``auto`` needs a passed self-test; ``on`` runs anyway and says so.

    When ``allowed`` is False, ``note`` is a fragment saying why ("the Capture add-on (Brush)
    is not installed"), for "Brush polish was skipped (...)" - the caller warns only when
    the user asked for the polish (``on``).  When it is True, ``note`` is empty or a complete
    warning sentence for the user (``on`` without a passed self-test)."""
    mode = str(opts if isinstance(opts, str) else getattr(opts, "polish", "auto") or "auto").strip().lower()
    if mode not in POLISH_MODES:
        return False, f"unknown polish setting {mode!r} (use {', '.join(POLISH_MODES)})"
    if mode == "off":
        return False, "Brush polish is off"
    if mode == "auto" and str(quality).strip().lower() != "best":
        return False, "automatic Brush polish is only used with 'Best' back quality"
    if not _usable_views(part):
        return False, "only a back made from views (MV-Adapter or your back photo) can be polished"
    installed = _brush_installed() if brush_installed is None else bool(brush_installed)
    if not installed:
        return False, "the Capture add-on (Brush) is not installed"
    bt = settings if isinstance(settings, dict) else getattr(settings, "brush_test", None)
    bt = bt if isinstance(bt, dict) else {}
    if bt.get("ok"):
        return True, ""
    if mode == "auto":
        if not bt:
            return False, f"Brush has not been tested on this PC yet ({_TEST_HINT})"
        return False, "Brush failed its self-test on this PC"
    what = "has not been tested on this PC" if not bt else "failed its self-test on this PC"
    return True, (
        f"Brush polish was run although Brush {what}; if it fails, the unpolished result is kept "
        f"(test it with {_TEST_HINT})."
    )


# --------------------------------------------------------------------------- small helpers
def _rot(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rotation by ``deg`` about the unit ``axis`` (right-handed, Rodrigues)."""
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    t = math.radians(float(deg))
    k = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) * math.cos(t) + math.sin(t) * k + (1.0 - math.cos(t)) * np.outer(a, a)


def _rgb_u8(img: Any, what: str) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim != 3 or a.shape[2] < 3:
        raise InputError(f"The {what} must be an H x W x 3 picture (got shape {a.shape}).")
    a = a[..., :3]
    if a.dtype != np.uint8:
        f = np.nan_to_num(a.astype(np.float64))
        scale = 255.0 if a.dtype.kind == "f" and float(f.max(initial=0.0)) <= 1.0 else 1.0
        a = np.clip(np.rint(f * scale), 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def _alpha_u8(a: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.nan_to_num(np.asarray(a, dtype=np.float64)) * 255.0), 0, 255).astype(np.uint8)


def _disc(r: int) -> np.ndarray:
    import cv2

    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _dilate(mask: np.ndarray, r: int) -> np.ndarray:
    import cv2

    if r <= 0:
        return mask.astype(bool)
    return cv2.dilate(mask.astype(np.uint8), _disc(int(r))) > 0


def _check(cancel: CancelToken | None) -> None:
    if cancel is not None:
        cancel.check()


def _cat(first: SplatCloud, second: SplatCloud) -> SplatCloud:
    """``first`` then ``second``; when only ``first`` has higher SH bands, ``second`` gets
    zero bands, so ``first`` keeps its own (as the merge does)."""
    if (
        first.sh_rest is not None
        and len(second)
        and (second.sh_rest is None or second.sh_rest.shape[1] != first.sh_rest.shape[1])
    ):
        second = replace(second, sh_rest=np.zeros((len(second), first.sh_rest.shape[1], 3), np.float32))
    return concat_splats([first, second])


def _finite_rows(sc: SplatCloud) -> np.ndarray:
    return (
        np.isfinite(np.asarray(sc.means)).all(axis=1)
        & np.isfinite(np.asarray(sc.scales)).all(axis=1)
        & np.isfinite(np.asarray(sc.opacity))
        & np.isfinite(np.asarray(sc.colors)).all(axis=1)
        & np.isfinite(np.asarray(sc.quats)).all(axis=1)
    )


_OFFS27 = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], np.int64)


class _VoxelRefs:
    """Reference points sorted by int64 voxel key (edge ``voxel``) for :meth:`nearest`."""

    def __init__(self, R: np.ndarray, in_a: np.ndarray, voxel: float) -> None:
        self.voxel = float(voxel)
        self.lo = R.min(axis=0) - 2.0 * self.voxel
        kr = np.floor((R - self.lo) / self.voxel).astype(np.int64)
        self.dims = kr.max(axis=0) + 3  # reference voxels are 2 .. dims - 3 on each axis
        keys = self._key(kr)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.R = (R[order] - self.lo).astype(np.float32)  # small offsets: float32 is exact enough
        self.in_a = in_a[order]
        self.per_voxel = len(R) / max(1, int(np.count_nonzero(np.diff(self.keys))) + 1)

    def _key(self, k: np.ndarray) -> np.ndarray:
        return (k[:, 0] * int(self.dims[1]) + k[:, 1]) * int(self.dims[2]) + k[:, 2]

    def occupied_near(self, Q: np.ndarray) -> np.ndarray:
        """Per point of ``Q``: does any reference lie in the 27 voxels around it (so possibly
        within ``voxel``; one outside them is farther)?  Occupancy only, no distances."""
        out = np.zeros(len(Q), bool)
        g = np.floor((Q - self.lo) / self.voxel)
        rows = np.flatnonzero(np.all((g >= 0) & (g < self.dims), axis=1))
        if not len(rows):
            return out
        uniq = np.unique(self.keys)
        qk, inv = np.unique(self._key(g[rows].astype(np.int64)), return_inverse=True)
        # back from keys to voxel coordinates (exact: the keys are mixed-radix numbers)
        d1, d2 = int(self.dims[1]), int(self.dims[2])
        qv = np.stack([qk // (d1 * d2), (qk // d2) % d1, qk % d2], axis=1)
        hit = np.zeros(len(qk), bool)
        for o in _OFFS27:
            kk = self._key(qv + o)
            pos = np.minimum(np.searchsorted(uniq, kk), len(uniq) - 1)
            hit |= uniq[pos] == kk
        out[rows] = hit[inv.reshape(-1)]
        return out

    def nearest(self, Q: np.ndarray, cancel: CancelToken | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Squared distances from each point of ``Q`` to its nearest reference of set A and of
        set B among those in the 27 voxels around it (``inf`` when there is none): the true
        nearest whenever it is at most ``voxel`` away (anything outside those voxels is
        farther).  Each query voxel and offset finds its run of references with two
        ``searchsorted``; the pairs are expanded in blocks of about :data:`_NN_PAIRS` and
        reduced per query with ``minimum.reduceat``."""
        dA = np.full(len(Q), np.inf, np.float32)
        dB = np.full(len(Q), np.inf, np.float32)
        g = np.floor((Q - self.lo) / self.voxel)
        rows = np.flatnonzero(np.all((g >= 0) & (g < self.dims), axis=1))  # others see no reference
        kq = g[rows].astype(np.int64)
        Qs = (Q[rows] - self.lo).astype(np.float32)
        block = max(1024, int(_NN_PAIRS / (27.0 * max(self.per_voxel, 1.0))))
        for b0 in range(0, len(rows), block):
            _check(cancel)
            kb, Qb = kq[b0 : b0 + block], Qs[b0 : b0 + block]
            bA = np.full(len(kb), np.inf, np.float32)
            bB = np.full(len(kb), np.inf, np.float32)
            for o in _OFFS27:
                kk = self._key(kb + o)
                s0 = np.searchsorted(self.keys, kk, "left")
                cnt = np.searchsorted(self.keys, kk, "right") - s0
                tot = int(cnt.sum())
                if not tot:
                    continue
                first = np.cumsum(cnt) - cnt  # each query's first pair
                ri = np.arange(tot) + np.repeat(s0 - first, cnt)
                diff = np.repeat(Qb, cnt, axis=0) - self.R[ri]
                d2 = np.einsum("ij,ij->i", diff, diff)
                has = cnt > 0
                starts = first[has]
                a = self.in_a[ri]
                bA[has] = np.minimum(bA[has], np.minimum.reduceat(np.where(a, d2, np.inf), starts))
                bB[has] = np.minimum(bB[has], np.minimum.reduceat(np.where(a, np.inf, d2), starts))
            dA[rows[b0 : b0 + block]], dB[rows[b0 : b0 + block]] = bA, bB
        return dA, dB


def front_copies(
    trained: np.ndarray,
    front_rows: np.ndarray,
    back_rows: np.ndarray,
    *,
    search_m: Sequence[float] = COPY_SEARCH_M,
    cancel: CancelToken | None = None,
) -> np.ndarray:
    """Which trained Gaussians are Brush's copies of the photographed front: those whose
    nearest initial row is an exact front row (``front_rows``: the kept SHARP rows that are
    put back afterwards) rather than an unpolished back row (``back_rows``).

    ``init.ply`` is the whole stitched cloud, so Brush trains (and densifies) the front rows
    too.  The re-merge drops most of their copies (they lie on the photographed surface), but
    not in the protected outline band, where a mould back's own surface Gaussians are kept on
    purpose (the rim wall): there the copies survived next to the exact rows put back, and
    the outline was drawn twice - more opaque hair and soft edges with a halo.

    The nearest initial row is searched in the 27 voxels around each trained Gaussian, with
    the voxel edges of :data:`COPY_SEARCH_M` in turn (2.5 mm, 8 mm, 2 cm): a Gaussian whose
    nearest row is within the edge is labelled exactly, the others go on to the next edge.
    A copy's nearest row is a front row at most 2 cm away; a Gaussian with no front row that
    near is left to the merge."""
    P = np.asarray(trained, dtype=np.float64)
    out = np.zeros(len(P), bool)
    F = np.asarray(front_rows, dtype=np.float64)
    G = np.asarray(back_rows, dtype=np.float64)
    F = F[np.isfinite(F).all(axis=1)]
    G = G[np.isfinite(G).all(axis=1)]
    if not len(F) or not len(P):
        return out
    R = np.concatenate([F, G])
    in_f = np.zeros(len(R), bool)
    in_f[: len(F)] = True
    todo = np.flatnonzero(np.isfinite(P).all(axis=1))
    # only Gaussians with a front row within the largest edge can be copies
    todo = todo[_VoxelRefs(F, np.ones(len(F), bool), float(max(search_m))).occupied_near(P[todo])]
    _check(cancel)
    edges = sorted(float(v) for v in search_m)
    for k, voxel in enumerate(edges):
        if not len(todo):
            break
        dF, dG = _VoxelRefs(R, in_f, voxel).nearest(P[todo], cancel)
        d = np.minimum(dF, dG)
        last = k == len(edges) - 1
        done = np.isfinite(d) if last else d <= np.float32(voxel * voxel)
        # a copy: the nearest initial row is a front row, at most the largest edge away
        out[todo[done]] = (dF[done] < dG[done]) & (dF[done] <= np.float32(voxel * voxel))
        todo = todo[~done]
    return out


def _robust_box(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    P = np.asarray(P, dtype=np.float64)
    P = P[np.isfinite(P).all(axis=1)]
    if not len(P):
        return np.zeros(3), np.zeros(3)
    return np.percentile(P, _BOX_PCT, axis=0), np.percentile(P, 100.0 - _BOX_PCT, axis=0)


def _brush_test(run: Any, given: dict[str, Any] | None) -> dict[str, Any]:
    """The stored Brush self-test result (``Settings.brush_test``; ``{}`` = never run)."""
    if given is not None:
        return dict(given)
    for owner in (getattr(run, "settings", None), getattr(getattr(run, "ctx", None), "settings", None)):
        bt = getattr(owner, "brush_test", None)
        if isinstance(bt, dict):
            return dict(bt)
    return {}


# --------------------------------------------------------------------------- frames
@dataclass
class Frame:
    """One training picture: its pinhole camera (``K``; world = C to camera ``x = R P + t``,
    OpenCV axes, pixel centres at ``i + 0.5``), the picture (uint8 ``H x W x 3`` - trained
    against Brush's black background, or ``H x W x 4`` - its alpha is trained too) and an
    optional mask (uint8 ``H x W``: 255 = train, 0 = ignore)."""

    name: str
    kind: str  # "photo" | "anchor" | "view"
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image: np.ndarray
    mask: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)``."""
        return int(self.image.shape[1]), int(self.image.shape[0])

    def camera(self) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
        """The camera as :func:`vrphoto.addons.capture.write_colmap_text` takes it."""
        return self.name, self.K, self.R, self.t, self.size


def telephoto_camera(
    canon: Canon,
    phi_deg: float,
    m: float,
    size: tuple[int, int],
    offset_m: float,
    y0: float,
    *,
    distance: float = TELEPHOTO_DIST_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(K, R, t)`` (world = C) of a distant pinhole that sees like the registered
    orthographic view at azimuth ``phi_deg`` of ``views_lift``'s levelled frame: a point
    with S coordinates ``P`` lands at ``u = (P . w - offset) / m + W/2``,
    ``v = (P_y - y0) / m + H/2`` (``w = (cos phi, 0, sin phi)``), up to the perspective of a
    camera ``distance`` metres away - the camera sits at ``offset w + y0 y^ - distance f``
    (``f = (-sin phi, 0, cos phi)``, the view direction) with ``f_px = distance / m`` and the
    principal point in the middle of the ``size = (W, H)`` picture."""
    width, height = int(size[0]), int(size[1])
    if not (math.isfinite(float(m)) and float(m) > 0):
        raise InputError(f"A view's scale must be positive (got {m!r} metres per pixel).")
    p = math.radians(float(phi_deg))
    c, s = math.cos(p), math.sin(p)
    w_S = np.array([c, 0.0, s])
    d_S = np.array([0.0, 1.0, 0.0])
    f_S = np.array([-s, 0.0, c])
    Rc = np.asarray(canon.R, dtype=np.float64)
    R = np.stack([Rc @ w_S, Rc @ d_S, Rc @ f_S])
    a_S = float(offset_m) * w_S + float(y0) * d_S  # maps to the picture centre, at the axis depth
    centre = Rc @ (a_S - float(distance) * f_S) + np.asarray(canon.c, dtype=np.float64)
    F = float(distance) / float(m)
    K = np.array([[F, 0.0, width / 2.0], [0.0, F, height / 2.0], [0.0, 0.0, 1.0]])
    return K, R, -R @ centre


def levelled_center(front: FrontModel, canon: Canon) -> tuple[float, float]:
    """``(x0, y0)`` of ``views_lift``'s levelled frame, estimated from the front: the middle
    of the 0.05-99.95th percentile box (frame S) of the matte's front points, as
    ``levelled_ortho_input`` measures it on the photo's pixels (here: on the mould cells,
    within a fraction of a cell).  Used for views that do not carry
    ``meta["frame_center_S"]`` when the caller gives no ``frame_center``."""
    matte = np.asarray(front.matte, dtype=bool)
    D = np.asarray(front.D_f, dtype=np.float64)
    ok = matte & np.isfinite(D) & (D > 0)
    if not ok.any():
        raise InputError("The subject's front depth is missing, so the views cannot be placed.")
    u, v = front.grid.centers()
    P = front.cam.rays(u[ok], v[ok]) * D[ok][:, None]
    S = canon.to_S(P)[:, :2]
    lo = np.percentile(S, 0.05, axis=0)
    hi = np.percentile(S, 99.95, axis=0)
    cx, cy = (float(t) for t in 0.5 * (lo + hi))
    return cx, cy


def _pair(x: Any) -> tuple[float, float] | None:
    if x is None:
        return None
    c = getattr(x, "center_S", x)  # an OrthoInput is accepted as well
    try:
        a = np.asarray(c, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        return None
    if a.size < 2 or not np.isfinite(a[:2]).all():
        return None
    return float(a[0]), float(a[1])


def _photo_source(
    front: FrontModel, photo_rgb: np.ndarray | None, alpha: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(rgb uint8, alpha float32, K)`` of the photo at its own resolution (or of the front's
    mould-grid copy when no photo is given), ``K`` mapping C to its pixels."""
    import cv2

    cam = front.cam
    f, W, H = float(cam.f), float(cam.width), float(cam.height)
    if photo_rgb is not None:
        img = _rgb_u8(photo_rgb, "photo")
        h, w = img.shape[:2]
        if alpha is None:
            a = cv2.resize(np.asarray(front.matte, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            a = np.asarray(alpha, dtype=np.float32)
            if a.ndim == 3:
                a = a[..., 0]
            if a.shape != (h, w):
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        sx, sy = W / w, H / h
    else:
        if front.rgb is None:
            raise InputError("The photo is missing, so Brush has nothing to train against.")
        img = _rgb_u8(front.rgb, "photo")
        a = np.asarray(front.matte, dtype=np.float32)
        sx = sy = float(front.grid.cell_px)
        if a.shape != img.shape[:2]:
            raise InputError("The front's photo and matte do not match.")
    K = np.array([[f / sx, 0.0, W / 2.0 / sx], [0.0, f / sy, H / 2.0 / sy], [0.0, 0.0, 1.0]])
    return img, np.clip(np.nan_to_num(a), 0.0, 1.0).astype(np.float32), K


def _crop_resize(
    img: np.ndarray, a: np.ndarray, K: np.ndarray, max_px: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The subject's box plus :data:`_CROP_MARGIN`, scaled to at most ``max_px`` on its long
    side (never enlarged); ``K`` follows (continuous coordinates: ``u' = k (u - x0)``)."""
    import cv2

    h, w = a.shape
    subj = a > 0.02
    if not subj.any():
        raise InputError("The subject's matte is empty, so Brush has nothing to train against.")
    rows = np.flatnonzero(subj.any(axis=1))
    cols = np.flatnonzero(subj.any(axis=0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    pad = int(math.ceil(_CROP_MARGIN * max(y1 - y0, x1 - x0)))
    y0, y1, x0, x1 = max(0, y0 - pad), min(h, y1 + pad), max(0, x0 - pad), min(w, x1 + pad)
    img, a = img[y0:y1, x0:x1], a[y0:y1, x0:x1]
    ch, cw = a.shape
    k = min(1.0, max_px / max(ch, cw))
    nw, nh = max(1, round(cw * k)), max(1, round(ch * k))
    if (nw, nh) != (cw, ch):
        img = cv2.resize(np.ascontiguousarray(img), (nw, nh), interpolation=cv2.INTER_AREA)
        a = cv2.resize(np.ascontiguousarray(a), (nw, nh), interpolation=cv2.INTER_AREA)
    kx, ky = nw / cw, nh / ch
    K2 = np.array(
        [[K[0, 0] * kx, 0.0, (K[0, 2] - x0) * kx], [0.0, K[1, 1] * ky, (K[1, 2] - y0) * ky], [0.0, 0.0, 1.0]]
    )
    return np.ascontiguousarray(img), np.ascontiguousarray(a, dtype=np.float32), K2


def _photo_frames(img: np.ndarray, a: np.ndarray, K: np.ndarray) -> list[Frame]:
    rgba = np.dstack([np.where(a[..., None] > 0, img, 0).astype(np.uint8), _alpha_u8(a)])
    return [
        Frame(f"p{i}_photo.png", "photo", K.copy(), np.eye(3), np.zeros(3), rgba, None, {"copy": i})
        for i in range(PHOTO_COPIES)
    ]


def _front_lookup(front: FrontModel, P: np.ndarray, seam_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """For points ``P`` (frame C): ``(on_surface, reliable)`` - the point is the photographed
    surface (inside the matte, within ``tau`` of the front depth of its mould cell), and that
    surface faces the photo camera within ``seam_deg`` at least
    :data:`_ANCHOR_RIM_CELLS` inside the outline."""
    grid = front.grid
    u, v, z = front.cam.project(P)
    cell = grid.cell_of(u, v)
    n = len(cell)
    ok = cell >= 0
    on = np.zeros(n, bool)
    rel = np.zeros(n, bool)
    if not ok.any():
        return on, rel
    c = cell[ok]
    matte = np.asarray(front.matte, dtype=bool).ravel()[c]
    D = np.asarray(front.D_f, dtype=np.float64).ravel()[c]
    cos_t = np.nan_to_num(np.asarray(front.cos_theta, dtype=np.float64).ravel()[c], nan=-1.0)
    dt = np.nan_to_num(np.asarray(front.dt, dtype=np.float64).ravel()[c], nan=0.0)
    with np.errstate(invalid="ignore"):
        surf = matte & np.isfinite(D) & (np.abs(z[ok] - D) <= depth_tol(D))
    on[ok] = surf
    rel[ok] = surf & (cos_t > math.cos(math.radians(seam_deg))) & (dt >= _ANCHOR_RIM_CELLS)
    return on, rel


def _anchor_frames(
    front: FrontModel,
    canon: Canon,
    K_photo: np.ndarray,
    size: tuple[int, int],
    seam_deg: float,
    poses: Sequence[tuple[float, float]],
    progress: ProgressReporter,
    cancel: CancelToken | None,
) -> tuple[list[Frame], list[dict[str, Any]]]:
    """Design 3.11 "anchors": renders of the cleaned front from cameras turned about
    ``canon.c`` (see the module docstring for the mask)."""
    from ..geometry.splat_render import SplatCamera, prepare_splats, render_splats

    w0, h0 = size
    k = min(1.0, ANCHOR_MAX_PX / max(w0, h0))
    w, h = max(1, round(w0 * k)), max(1, round(h0 * k))
    kx, ky = w / w0, h / h0
    K = np.array(
        [
            [K_photo[0, 0] * kx, 0.0, K_photo[0, 2] * kx],
            [0.0, K_photo[1, 1] * ky, K_photo[1, 2] * ky],
            [0, 0, 1.0],
        ]
    )
    c = np.asarray(canon.c, dtype=np.float64)
    x_hat, y_hat = np.asarray(canon.R, np.float64)[:, 0], np.asarray(canon.R, np.float64)[:, 1]
    prep = prepare_splats(front.sc, cancel=cancel)
    jj, ii = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    frames: list[Frame] = []
    info: list[dict[str, Any]] = []
    for n, (yaw, pitch) in enumerate(poses):
        _check(cancel)
        # +yaw: the camera walks to its right (toward +x^) about the vertical through c;
        # +pitch: it rises (toward -y^) and looks down
        Q = _rot(y_hat, -yaw) @ _rot(x_hat, -pitch)
        R = Q.T
        centre = c - Q @ c
        cam = SplatCamera(
            R=R, center=centre, width=w, height=h, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        sub = progress.sub(n / len(poses), (n + 1) / len(poses))
        r = render_splats(prep, cam, background=(0, 0, 0), progress=sub, cancel=cancel)
        a = np.asarray(r.alpha, dtype=np.float64)
        z = np.asarray(r.depth_median, dtype=np.float64)
        hit = np.isfinite(z) & (z > 0)
        rel = np.zeros((h, w), bool)
        if hit.any():
            Pc = np.stack(
                [(jj[hit] - K[0, 2]) / K[0, 0] * z[hit], (ii[hit] - K[1, 2]) / K[1, 1] * z[hit], z[hit]], -1
            )
            _, rel_px = _front_lookup(front, Pc @ R + centre, seam_deg)
            rel[hit] = rel_px
        keep = rel & (a >= _ANCHOR_ALPHA)
        sil = a >= 0.5
        if sil.any():
            rows, cols = np.flatnonzero(sil.any(axis=1)), np.flatnonzero(sil.any(axis=0))
            size_px = max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1)
        else:
            size_px = max(w, h)
        out = ~_dilate(sil, max(3, int(math.ceil(_ANCHOR_OUT_FRAC * size_px))))
        rec = {"yaw": yaw, "pitch": pitch, "front_px": int(keep.sum()), "black_px": int(out.sum())}
        info.append(rec)
        if keep.sum() < _ANCHOR_MIN_FRAC * w * h:
            rec["used"] = False
            continue
        rgb = np.asarray(r.rgb, dtype=np.float64)
        straight = np.clip(np.rint(rgb / np.maximum(a, 1e-6)[..., None]), 0, 255)  # un-premultiply
        img = np.where((out | (a < 0.05))[..., None], 0.0, straight).astype(np.uint8)  # black target outside
        mask = np.where(keep | out, 255, 0).astype(np.uint8)
        rec["used"] = True
        frames.append(Frame(f"anchor_{n:02d}.png", "anchor", K.copy(), R, -R @ centre, img, mask, dict(rec)))
    return frames, info


def _view_frames(
    views: Sequence[Any],
    canon: Canon,
    merged: SplatCloud,
    reliable_rows: np.ndarray,
    masks_ok: bool,
    center: tuple[float, float] | None,
    center_fn: Any,
    cancel: CancelToken | None,
) -> tuple[list[Frame], list[dict[str, Any]]]:
    """The registered generated views as pinhole frames (see the module docstring)."""
    from ..geometry.register import point_zbuffer

    frames: list[Frame] = []
    info: list[dict[str, Any]] = []
    P_S = None
    for i, v in enumerate(views):
        _check(cancel)
        phi = float(v.phi_deg) % 360.0
        d = abs((phi + 180.0) % 360.0 - 180.0)  # degrees from the photo's side
        rec: dict[str, Any] = {"name": str(v.name), "phi": phi, "used": False}
        info.append(rec)
        if d < _FRONT_VIEW_DEG:
            rec["reason"] = "a front view (the photo is the front)"
            continue
        quarter = d < _QUARTER_DEG
        if quarter and not masks_ok:
            rec["reason"] = "needs a mask, and Brush cannot mix masks with transparent pictures here"
            continue
        m, off = v.m, v.meta.get("offset_m")
        c = _pair(v.meta.get("frame_center_S")) or center or center_fn()
        if m is None or off is None or c is None:
            rec["reason"] = "its camera is unknown"
            continue
        rgb = _rgb_u8(v.rgb, f"view {v.name}")
        a = np.clip(np.nan_to_num(np.asarray(v.alpha, dtype=np.float32)), 0.0, 1.0)
        if a.shape != rgb.shape[:2] or not (a > 0.5).any():
            rec["reason"] = "its matte is empty"
            continue
        h, w = a.shape
        K, Rk, t = telephoto_camera(canon, phi, float(m), (w, h), float(off), c[1])
        name = f"view_{i:02d}_az{int(round(phi)) % 360:03d}.png"
        if not quarter:
            img = np.dstack([np.where(a[..., None] > 0, rgb, 0).astype(np.uint8), _alpha_u8(a)])
            frames.append(Frame(name, "view", K, Rk, t, img, None, {"phi": phi, "rgba": True}))
            rec.update(used=True, kind="rgba")
            continue
        # front quarter view: never compete with the photographed front - mask out what the
        # stitched cloud shows of the reliable front there (visible, point z-buffer), +5 px
        if P_S is None:
            P_S = canon.to_S(np.asarray(merged.means, dtype=np.float64))
        p = math.radians(phi)
        wv = np.array([math.cos(p), 0.0, math.sin(p)])
        fv = np.array([-math.sin(p), 0.0, math.cos(p)])
        u = (P_S @ wv - float(off)) / float(m) + w / 2.0
        vv = (P_S[:, 1] - c[1]) / float(m) + h / 2.0
        _, idx = point_zbuffer(u, vv, P_S @ fv, (h, w), radius=_ZBUF_RADIUS_PX)
        front_px = np.zeros((h, w), bool)
        won = idx >= 0
        front_px[won] = reliable_rows[idx[won]]
        mask = np.where(_dilate(front_px, _QUARTER_DILATE_PX), 0, 255).astype(np.uint8)
        img = np.clip(np.rint(rgb.astype(np.float64) * a[..., None]), 0, 255).astype(np.uint8)  # on black
        frames.append(Frame(name, "view", K, Rk, t, img, mask, {"phi": phi, "rgba": False}))
        rec.update(used=True, kind="masked", front_px=int(front_px.sum()))
    return frames, info


def build_frames(
    merged: SplatCloud,
    is_front: np.ndarray,
    front: FrontModel,
    canon: Canon,
    part: BackPart,
    *,
    quality: str = "best",
    seam_deg: float = 60.0,
    masks: bool = True,
    photo_rgb: np.ndarray | None = None,
    alpha: np.ndarray | None = None,
    frame_center: Any = None,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
) -> tuple[list[Frame], dict[str, Any]]:
    """The training pictures of design 3.11 (module docstring): the photo three times, the
    SHARP anchors (``masks`` only; ``best`` adds three) and the generated views.  The photo
    is ``photo_rgb`` / ``alpha`` at any resolution with SHARP's field of view (default: the
    front's mould-grid copy and matte).  ``frame_center`` (``(x0, y0)`` or an ``OrthoInput``)
    is the levelled frame's centre for views without ``meta["frame_center_S"]`` (default:
    :func:`levelled_center`).  Returns ``(frames, info)``; raises :class:`InputError` when
    there is no photo or no usable view."""
    rep = progress or ProgressReporter.null()
    isf = np.asarray(is_front, dtype=bool)
    img, a, K = _photo_source(front, photo_rgb, alpha)
    img, a, K = _crop_resize(img, a, K, PHOTO_MAX_PX)
    frames = _photo_frames(img, a, K)
    info: dict[str, Any] = {"photo": {"copies": PHOTO_COPIES, "size": [int(a.shape[1]), int(a.shape[0])]}}
    _check(cancel)
    if masks:
        poses = list(ANCHORS) + (list(ANCHORS_BEST) if str(quality).lower() == "best" else [])
        rep(0.0, "Rendering SHARP views for Brush")
        anchors, info["anchors"] = _anchor_frames(
            front, canon, K, (a.shape[1], a.shape[0]), seam_deg, poses, rep.sub(0.0, 0.85), cancel
        )
        frames += anchors
    else:
        info["anchors"] = "left out: Brush cannot mix masked and transparent pictures on this PC"
    rep(0.85, "Preparing the AI views for Brush")
    _, reliable = _front_lookup(front, np.asarray(merged.means, dtype=np.float64), seam_deg)
    reliable &= isf
    est: list[tuple[float, float]] = []

    def center_fn() -> tuple[float, float]:
        """The levelled frame's centre for views that do not carry it: exactly as
        ``levelled_ortho_input`` finds it when the photo and matte are given (its centre does
        not depend on the picture size, so a 64 px run is enough), else estimated."""
        if not est:
            c, how = None, "estimated from the front"
            if photo_rgb is not None and alpha is not None:
                try:
                    from ..geometry.views_lift import levelled_ortho_input

                    oi = levelled_ortho_input(photo_rgb, alpha, front, canon, size=64, ref=64, cancel=cancel)
                    c, how = _pair(oi.center_S), "from the levelled photo"
                except Cancelled:
                    raise
                except Exception as e:  # the estimate below is close (about a pixel)
                    log.warning("Brush polish: the levelled photo could not be made (%s)", e)
            est.append(c or levelled_center(front, canon))
            info["frame_center"] = {"center_S": list(est[0]), "source": how}
            log.info("Brush polish: levelled frame centre (%.4f, %.4f), %s", *est[0], how)
        return est[0]

    vframes, info["views"] = _view_frames(
        _usable_views(part), canon, merged, reliable, masks, _pair(frame_center), center_fn, cancel
    )
    if not vframes:
        raise InputError("The back has no views Brush could train against.")
    frames += vframes
    info["masked"] = sum(f.mask is not None for f in frames)
    rep(1.0, "Training pictures ready")
    return frames, info


# --------------------------------------------------------------------------- dataset
def write_dataset(
    folder: str | Path,
    frames: Sequence[Frame],
    init: SplatCloud,
    *,
    points: int = INIT_POINTS,
    seed: int = 0,
) -> dict[str, Any]:
    """Write the Brush dataset (module docstring) into ``folder`` (emptied first):
    ``images/``, ``masks/`` (only when a frame has a mask), ``sparse/0/*.txt`` and
    ``init.ply`` (``init`` without higher SH bands, read back to validate).  Returns a
    summary; raises :class:`InputError` for duplicate names or a broken ``init.ply``."""
    from ..addons.capture import write_colmap_text
    from ..core.imaging import save_image

    ds = Path(folder)
    names = [f.name for f in frames]
    if len(set(names)) != len(names):
        raise InputError("Two training pictures have the same name; every picture needs its own file.")
    if not len(init):
        raise InputError("There are no Gaussians to start the polish from.")
    if ds.exists():
        shutil.rmtree(ds)
    (ds / "images").mkdir(parents=True)
    masked = [f for f in frames if f.mask is not None]
    if masked:
        (ds / "masks").mkdir()
    for f in frames:
        save_image(ds / "images" / f.name, f.image)
        if f.mask is not None:
            if f.mask.shape != f.image.shape[:2]:
                raise InputError(f"The mask of {f.name} does not match its picture.")
            save_image(ds / "masks" / f.name, f.mask)
    n = len(init)
    rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF)
    idx = np.sort(rng.choice(n, size=min(int(points), n), replace=False))
    write_colmap_text(
        ds, [f.camera() for f in frames], np.asarray(init.means)[idx], np.asarray(init.colors)[idx]
    )
    init_dc = replace(init, sh_rest=None)
    ply = ds / "init.ply"
    write_splats(init_dc, ply)
    back = read_splats(ply)
    if len(back) != n or not np.allclose(np.asarray(back.means), np.asarray(init_dc.means), atol=1e-6):
        raise InputError("The initial splats for Brush could not be written correctly.")
    return {
        "folder": str(ds),
        "images": len(frames),
        "masked": len(masked),
        "init": n,
        "points": len(idx),
    }


# --------------------------------------------------------------------------- output checks
def _ply_nonfinite(path: Path) -> tuple[int, int] | None:
    """``(vertices, rows with a non-finite core value)`` of a binary PLY (positions, colour,
    opacity, scales, rotation), read in chunks; None when the file is not a plain binary PLY
    (``read_splats`` then decides)."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
            m = re.search(rb"end_header[ \t]*\r?\n", head)
            if not head.startswith(b"ply") or m is None:
                return None
            fmt = ""
            elements: list[tuple[str, int, list[tuple[str, str]]]] = []
            for line in head[: m.start()].decode("ascii", "replace").splitlines()[1:]:
                p = line.split()
                if not p:
                    continue
                if p[0] == "format":
                    fmt = p[1]
                elif p[0] == "element":
                    elements.append((p[1], int(p[2]), []))
                elif p[0] == "property" and elements:
                    if p[1] == "list" or p[1] not in _PLY_TYPES:
                        return None
                    elements[-1][2].append((p[2], _PLY_TYPES[p[1]]))
            if fmt not in ("binary_little_endian", "binary_big_endian"):
                return None
            end = "<" if fmt == "binary_little_endian" else ">"
            offset = m.end()
            for name, count, props in elements:
                dt = np.dtype([(pn, end + t) for pn, t in props])
                if name != "vertex":
                    offset += dt.itemsize * count
                    continue
                cols = [pn for pn, t in props if pn in _CORE_PLY and t.startswith("f")]
                fh.seek(offset)
                bad = 0
                chunk = 1 << 18
                for a in range(0, count, chunk):
                    k = min(chunk, count - a)
                    buf = fh.read(dt.itemsize * k)
                    if len(buf) < dt.itemsize * k:
                        return None
                    rec = np.frombuffer(buf, dtype=dt, count=k)
                    ok = np.ones(k, bool)
                    for cn in cols:
                        ok &= np.isfinite(rec[cn])
                    bad += int(k - ok.sum())
                return count, bad
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return None


def check_polished(ply: str | Path, init: SplatCloud) -> tuple[SplatCloud | None, dict[str, Any]]:
    """Read and check Brush's export against the initial cloud (design 3.11): rejected (None
    and ``info["reason"]``) when it cannot be read, has non-finite values, fewer than
    :data:`MIN_SPLATS` Gaussians or a robust box (0.5-99.5th percentiles per axis) more than
    :data:`MAX_BOX_GROWTH` times the initial one along some axis."""
    path = Path(ply)
    info: dict[str, Any] = {}
    raw = _ply_nonfinite(path)
    try:
        sc = read_splats(path)
    except VRPhotoError as e:
        info["reason"] = f"Brush's result could not be read ({e})"
        return None, info
    n_file, bad = raw if raw is not None else (len(sc), 0)
    if raw is None:
        bad = int((~_finite_rows(sc)).sum())
    info.update(n=int(n_file), non_finite=int(bad))
    if bad:
        info["reason"] = f"Brush's result has invalid numbers (NaN) in {bad} of its {n_file} Gaussians"
        return None, info
    if len(sc) < MIN_SPLATS:
        info["reason"] = f"Brush returned only {len(sc)} Gaussians"
        return None, info
    lo_i, hi_i = _robust_box(np.asarray(init.means))
    lo_t, hi_t = _robust_box(np.asarray(sc.means))
    growth = float(np.max((hi_t - lo_t) / np.maximum(hi_i - lo_i, _BOX_FLOOR_M)))
    info["box_growth"] = round(growth, 4)
    if not growth <= MAX_BOX_GROWTH:
        info["reason"] = (
            f"Brush's result is {growth:.1f} times the size of the subject along one axis (it diverged)"
        )
        return None, info
    return sc, info


# --------------------------------------------------------------------------- training
@dataclass
class _Run:
    """One Brush configuration of the retry ladder."""

    sh: int | None
    resolution: int | None
    cap: int
    masks: bool
    dropped: list[str] = field(default_factory=list)

    def args(self, steps: int) -> list[str]:
        opts: dict[str, str] = {}
        if self.sh is not None:
            opts["--sh-degree"] = str(self.sh)
        opts["--growth-stop-iter"] = str(max(1, steps // 2))
        opts["--match-alpha-weight"] = str(_MATCH_ALPHA_WEIGHT)
        if self.resolution is not None:
            opts["--max-resolution"] = str(self.resolution)
        out: list[str] = []
        for k, v in opts.items():
            if k not in self.dropped:
                out += [k, v]
        return out


def _log_since(path: Path, pos: int) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(pos)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def _log_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _keep_log(run: Any, item: Any, log_path: Path) -> Path | None:
    """Copy the Brush log next to the outputs (the work folder is deleted after the job)."""
    out_path = getattr(run, "out_path", None)
    if out_path is None or not log_path.is_file():
        return None
    try:
        dst = Path(out_path(item, "_sharp360_polish", "log"))
        shutil.copyfile(log_path, dst)
        return dst
    except (OSError, VRPhotoError) as e:
        log.info("Could not keep the Brush polish log %s: %s", log_path, e)
        return None


def _unpolished(
    merged: SplatCloud, is_front: np.ndarray, diag: dict[str, Any], reason: str
) -> tuple[SplatCloud, np.ndarray, dict[str, Any]]:
    diag.update(ok=False, reason=reason)
    note = f" Brush's log was saved as {Path(diag['log']).name}." if diag.get("log") else ""
    diag["warning"] = f"Brush polish did not work ({reason}); the unpolished result was kept.{note}"
    log.warning("Brush polish skipped: %s", reason)
    return merged, np.asarray(is_front, dtype=bool), diag


def polish_with_brush(
    run: Any,
    item: Any,
    merged: SplatCloud,
    is_front: np.ndarray,
    front: FrontModel,
    canon: Canon,
    part: BackPart,
    *,
    steps: int,
    seed: int,
    quality: str,
    progress: ProgressReporter | None = None,
    cancel: CancelToken | None = None,
    seam_deg: float = 60.0,
    photo_rgb: np.ndarray | None = None,
    alpha: FloatMap | None = None,
    frame_center: Any = None,
    brush_test: dict[str, Any] | None = None,
) -> tuple[SplatCloud, np.ndarray, dict[str, Any]]:
    """Polish the stitched cloud ``merged`` (frame C; ``is_front`` marks its exact SHARP
    rows) with Brush and return ``(cloud, is_front, diag)`` (design 3.11, module docstring).

    ``part`` is the view-based back the cloud was stitched with (its registered ``views``
    are trained against; its ``hull`` / ``spacing_m`` feed the floater filter).  ``steps``
    are Brush's training steps, ``seed`` its seed and the seam dither's, ``quality`` the back
    quality (``best`` adds anchors).  Additions: ``seam_deg`` ('Keep the photo up to'),
    ``photo_rgb`` / ``alpha`` (the photo and its matte at any resolution with SHARP's field
    of view - pass them, else the front's mould-grid copy of the photo is used),
    ``frame_center`` (the levelled frame's ``(x0, y0)`` or the ``OrthoInput`` of the views
    when they do not carry ``meta["frame_center_S"]``, as the ``images`` source's views; else
    estimated) and ``brush_test`` (the self-test result; default ``run.settings.brush_test``).
    The work goes to ``run.work_dir(item)``; Brush runs in-process after
    ``run._free_app_gpu()``.

    The front guarantee holds: the returned front rows are exactly the kept rows of
    ``merged[is_front]`` (bit-identical, protected ones never removed).  Only a cancel
    raises (:class:`Cancelled`); any other failure returns ``merged`` / ``is_front``
    unchanged with ``diag["ok"] = False``, a plain ``diag["reason"]`` (a fragment) and
    ``diag["warning"]`` (a sentence for the user).  ``diag`` is JSON-safe: frames, dataset,
    the Brush runs (``attempts``), the output check, merge and floater counts, seconds."""
    t_start = time.perf_counter()
    rep = progress or ProgressReporter.null()
    cancel = cancel if cancel is not None else getattr(run, "cancel", None)
    isf = np.asarray(is_front, dtype=bool)
    diag: dict[str, Any] = {"ok": False, "steps": None, "attempts": []}
    if isf.shape != (len(merged),):
        return _unpolished(merged, isf, diag, "the stitched cloud and its front marks do not match")
    try:
        steps = int(steps)
        if steps < 1:
            raise ValueError
    except (TypeError, ValueError):
        return _unpolished(merged, isf, diag, f"invalid number of training steps ({steps!r})")
    diag["steps"] = steps
    if not _usable_views(part):
        return _unpolished(merged, isf, diag, "the back has no views Brush could train against")
    bt = _brush_test(run, brush_test)
    tested = bool(bt.get("ok"))
    sh_option = bool(bt.get("sh_option", True)) if tested else True
    masks_ok = bool(bt.get("masks")) if tested else True
    sh: int | None = None if not sh_option else (1 if bt.get("sh0") is False else 0)
    diag["brush_test"] = {"tested": tested, "sh_option": sh_option, "masks": masks_ok, "sh0": bt.get("sh0")}
    try:
        work = Path(run.work_dir(item))
        ds, out = work / "brush_ds", work / "brush_out"
        log_path = work / _LOG_NAME
        # ---- 1. the training pictures and the dataset
        rep(0.0, "Preparing the pictures for Brush")
        ok_rows = _finite_rows(merged)
        init = subset_splats(merged, ok_rows) if not ok_rows.all() else merged
        frames, diag["frames"] = build_frames(
            merged, isf, front, canon, part, quality=quality, seam_deg=seam_deg, masks=masks_ok,
            photo_rgb=photo_rgb, alpha=alpha, frame_center=frame_center, progress=rep.sub(0.0, 0.13),
            cancel=cancel,
        )  # fmt: skip
        _check(cancel)
        rep(0.13, "Writing the training data for Brush")
        diag["dataset"] = write_dataset(ds, frames, init, seed=seed)
        _check(cancel)
    except Cancelled:
        raise
    except (VRPhotoError, OSError, MemoryError) as e:
        return _unpolished(merged, isf, diag, f"the training data could not be prepared: {e}")
    except Exception as e:  # a bug here must not cost the result
        log.exception("Brush polish: preparing the dataset failed")
        return _unpolished(merged, isf, diag, f"the training data could not be prepared ({type(e).__name__})")

    # ---- 2. Brush, with the retry ladder
    n_init = len(init)
    cap = int(min(max(round(_CAP_PER_INIT * n_init), CAP_RANGE[0]), CAP_RANGE[1]))
    cfg = _Run(sh=sh, resolution=MAX_RESOLUTION, cap=cap, masks=bool(diag["dataset"]["masked"]))
    diag.update(init=n_init, max_splats=cap)
    timeout = TIMEOUT_BASE_S + TIMEOUT_PER_STEP_S * steps
    ply: Path | None = None
    reason = ""
    reduced = sh_retried = masks_dropped = False
    from ..addons import capture
    from ..addons.runner import LogSink

    sink = LogSink(log_path)
    try:
        sink.header(f"SHARP 360 Brush polish: {steps} steps, {len(frames)} pictures, {n_init} initial splats")
        R._free_app_gpu()  # Brush does not go through _run_addon
        for i in range(_MAX_RUNS):
            _check(cancel)
            label = _LABEL if i == 0 else f"{_LABEL} again"
            extra = cfg.args(steps)
            rec: dict[str, Any] = {
                "try": i + 1,
                "sh_degree": cfg.sh if "--sh-degree" not in cfg.dropped else None,
                "max_resolution": cfg.resolution if "--max-resolution" not in cfg.dropped else None,
                "max_splats": cfg.cap,
                "masks": cfg.masks,
                "args": list(extra),
                "ok": False,
            }
            diag["attempts"].append(rec)
            pos = _log_size(log_path)
            t0 = time.monotonic()
            try:
                ply = capture.train_splat(
                    ds, out, steps=steps, max_splats=cfg.cap, extra_args=extra, export_name=_EXPORT_NAME,
                    seed=seed, progress=rep.sub(0.15, 0.9), cancel=cancel, timeout=timeout, log_sink=sink,
                    cwd=out, label=label,
                )  # fmt: skip
                rec.update(ok=True, seconds=round(time.monotonic() - t0, 2))
                break
            except Cancelled:
                raise
            except AddonNotInstalled:
                rec["error"] = "Brush is not installed"
                reason = "the Capture add-on (Brush) is not installed"
                break
            except (AddonError, InputError, OSError) as e:
                secs = time.monotonic() - t0
                rec.update(seconds=round(secs, 2), error=str(e))
                text = _log_since(log_path, pos)
                reason = str(e) or type(e).__name__
                if secs >= timeout - 1:
                    break  # a hang is no option problem: another run would only wait again
                m = _FLAG_RE.search(text)
                if m and m.group(1) in _OPTIONAL_FLAGS and m.group(1) not in cfg.dropped:
                    cfg.dropped.append(m.group(1))
                    sink.write(f"Brush does not know {m.group(1)}; trying again without it")
                    continue
                oom = bool(_OOM_RE.search(text)) or "memory" in str(getattr(e, "hint", "") or "").lower()
                if oom and not reduced:
                    reduced = True
                    cfg.resolution = OOM_RESOLUTION
                    cfg.cap = max(1, int(cfg.cap * _OOM_CAP))
                    sink.write(f"out of memory: trying again at {OOM_RESOLUTION} px with {cfg.cap} splats")
                    continue
                if cfg.sh == 0 and "--sh-degree" not in cfg.dropped and not sh_retried:
                    sh_retried = True
                    cfg.sh = 1
                    sink.write("trying again with --sh-degree 1")
                    continue
                if cfg.masks and not tested and not masks_dropped:
                    masks_dropped = True
                    plain = [f for f in frames if f.mask is None]
                    try:
                        diag["dataset"] = write_dataset(ds, plain, init, seed=seed)
                    except (VRPhotoError, OSError) as e2:
                        rec["error"] += f"; the pictures without masks could not be written ({e2})"
                        break
                    frames, cfg.masks = plain, False
                    diag["frames"]["masks_dropped"] = True
                    sink.write("trying again without the masked pictures")
                    continue
                break
    finally:
        sink.close()
    if ply is None:
        diag["log"] = str(_keep_log(run, item, log_path) or "") or None
        return _unpolished(merged, isf, diag, reason or "Brush did not run")
    _check(cancel)

    # ---- 3. check the export, put the photographed front back, clean up
    try:
        rep(0.9, "Checking Brush's result")
        trained, diag["check"] = check_polished(ply, init)
        if trained is None:
            diag["log"] = str(_keep_log(run, item, log_path) or "") or None
            return _unpolished(merged, isf, diag, diag["check"]["reason"])
        trained = replace(trained, sh_rest=None)  # the viewer's .splat has no SH bands either
        n_back = int((~isf).sum())
        # Brush's copies of the front rows go: the exact rows are put back below
        copies = front_copies(
            trained.means, np.asarray(merged.means)[isf], np.asarray(merged.means)[~isf], cancel=cancel
        )
        diag["front_copies"] = int(copies.sum())
        if copies.any():
            trained = subset_splats(trained, ~copies)
        tpart = replace(part, sc=trained, normals=None, back_z=None, ok=True, reason="")
        rep(0.91, "Joining the polished back and the photo front")
        joined, j_front, diag["merge"] = merge_parts(
            front, tpart, seam_deg=seam_deg, seed=seed, budget=R.MAX_OBJECT_SPLATS, cancel=cancel,
            progress=rep.sub(0.91, 0.95),
        )  # fmt: skip
        G = subset_splats(joined, ~np.asarray(j_front, dtype=bool))
        if len(G) == 0 or len(G) < MIN_KEPT_FRAC * n_back:
            return _unpolished(
                merged, isf, diag,
                f"the polished back did not line up with the photo (only {len(G)} of its Gaussians fit "
                f"behind the front, the unpolished back had {n_back})",
            )  # fmt: skip
        F = subset_splats(merged, isf)  # the EXACT kept SHARP rows
        sc = _cat(F, G) if len(F) else G
        sc_front = np.zeros(len(sc), bool)
        sc_front[: len(F)] = True
        _check(cancel)
        rep(0.95, "Matching colours")
        sc = feather_colors(sc, sc_front, cancel=cancel)
        sc, sc_front, diag["floaters"] = remove_floaters(
            sc, sc_front, front, canon, hull=part.hull, seam_deg=seam_deg, spacing_m=part.spacing_m,
            cancel=cancel, progress=rep.sub(0.96, 1.0),
        )  # fmt: skip
    except Cancelled:
        raise
    except Exception as e:
        log.exception("Brush polish: joining the polished back failed")
        why = str(e) if isinstance(e, VRPhotoError) else type(e).__name__
        return _unpolished(merged, isf, diag, f"the polished back could not be joined to the front ({why})")
    diag.update(
        ok=True,
        reason="",
        front=int(sc_front.sum()),
        back=int((~sc_front).sum()),
        back_before=n_back,
        seconds=round(time.perf_counter() - t_start, 2),
    )
    log.info(
        "Brush polish: %d steps in %d run(s), %d pictures; back %d -> %d Gaussians, front %d kept exactly",
        steps, len(diag["attempts"]), len(frames), n_back, diag["back"], diag["front"],
    )  # fmt: skip
    rep(1.0, "Brush polish done")
    return sc, sc_front, diag


__all__ = [
    "ANCHORS",
    "ANCHORS_BEST",
    "DEFAULT_STEPS",
    "POLISH_MODES",
    "TELEPHOTO_DIST_M",
    "Frame",
    "build_frames",
    "check_polished",
    "front_copies",
    "levelled_center",
    "polish_allowed",
    "polish_with_brush",
    "telephoto_camera",
    "write_dataset",
]
